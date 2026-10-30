"""Reuse exact candidate-extraction ECG PNGs across environments without relaxing provenance."""
from pathlib import Path
import hashlib
import os
import tempfile
import time
from unittest.mock import patch
import numpy as np
from common import read, sha, signature, latest, immutable, atomic, lock, parent_setup, unique


def archive_folder(cfg):
    return Path(cfg['run_dir']) / 'audits' / 'query_image_archive' / 'shaoxing_v1'


def _inputs(cfg):
    if cfg['cohort'] != 'shaoxing':
        raise ValueError('This repair is only for Shaoxing; use the ordinary path for internal cohorts')
    parent_setup(cfg)
    run = Path(cfg['run_dir']); folder = run / 'queries' / 'shaoxing'
    rows = read(folder / 'manifest.json')
    unique(rows, 'id')
    candidates = latest(folder / 'candidates.jsonl')
    stage = read(run / 'stages' / 'query_extraction_shaoxing.json')['inputs']
    if stage['query_sha256'] != sha(folder / 'manifest.json'):
        raise ValueError('Extraction manifest changed; image archival cannot fix this')
    import sensor_candidate_generation as sensor
    if stage['sensor_sha256'] != sha(sensor.__file__):
        raise ValueError('Original extraction/rendering helper changed; restore the exact original helper')
    from candidate_preparation import extract
    from inputs import observations
    for row in rows:
        c = candidates.get((row['id'], 'sensor_only'))
        if not c or c.get('status') != 'ok':
            raise ValueError('Successful candidate required: ' + row['id'])
        if c['protocol_signature'] != signature(stage):
            raise ValueError('Candidate protocol changed: ' + row['id'])
        if c['response_sha256'] != signature(c['text']) or c['findings'] != observations(extract(c['text'], 'ecg')):
            raise ValueError('Candidate text/parse changed: ' + row['id'])
        if c['input_scope'] != 'sensor_only' or c['model'] != 'PULSE-ECG/PULSE-7B':
            raise ValueError('Unexpected extraction scope/model: ' + row['id'])
    cohort_rows = unique(read(Path(cfg['shaoxing_cohort']) / 'manifest.json'), 'id')
    for row in rows:
        source = cohort_rows[row['id']]
        for key in ('patient_id', 'admission_id', 'study_id', 'record', 'header_sha256', 'waveform_sha256', 'waveform_file'):
            if row[key] != source[key]:
                raise ValueError('Frozen query/cohort source mismatch: ' + row['id'] + ' ' + key)
    return rows, candidates, folder


def _tensor(row):
    from shaoxing_external import waveform
    x = waveform(row)  # Verifies WFDB header and waveform file hashes before reading.
    x = (x / np.max(np.abs(x))).astype(np.float32).reshape(1, 5000, 12)
    digest = signature(dict(shape=list(x.shape), dtype=str(x.dtype),
                            bytes=hashlib.sha256(x.tobytes()).hexdigest()))
    return x, digest


def diagnose(cfg, probes=3):
    """Read-only: never calls an image provider and never overwrites existing PNGs."""
    rows, candidates, _ = _inputs(cfg)
    results = []
    for row in rows[:probes]:
        c = candidates[row['id'], 'sensor_only']; _, tensor = _tensor(row)
        p = Path(cfg['shaoxing_cohort']) / 'images' / (row['id'] + '.png')
        results.append(dict(query_id=row['id'], candidate_protocol_matches=True,
            source_tensor_matches=tensor == c['source_tensor_sha256'],
            current_image_exists=p.is_file(),
            current_png_matches=p.is_file() and sha(p) == c['sensor_sha256']))
    return results


def prepare_archive(cfg, reconstruct=False, max_run_hours=1.0):
    """Archive matching images; optionally reconstruct into temporary files in the PULSE environment.

    No checkpoint/GPU/inference is needed. Exact PNG and tensor hashes must match
    original candidates. The shared cohort images and all historical journals stay untouched.
    """
    rows, candidates, folder = _inputs(cfg)
    dest = archive_folder(cfg); dest.mkdir(parents=True, exist_ok=True)
    from extensions import context_fit
    identity = dict(version=1, query_manifest_sha256=sha(folder / 'manifest.json'),
        candidates_sha256=sha(folder / 'candidates.jsonl'),
        extraction_stage_sha256=sha(Path(cfg['run_dir']) / 'stages/query_extraction_shaoxing.json'),
        context_fit_helper_sha256=sha(context_fit.__file__), helper_sha256=sha(__file__),
        policy='Exact original candidate PNG bytes and normalized waveform digest; no hash replacements')
    started = time.monotonic(); records = []; complete = True
    import sensor_candidate_generation as sensor
    from PIL import Image
    with lock(dest / 'writer.lock'):
        immutable(dest / 'protocol.json', identity)
        for row in rows:
            if (dest / 'STOP').exists() or time.monotonic() - started >= max_run_hours * 3600:
                complete = False; break
            rid = row['id']; c = candidates[rid, 'sensor_only']; x, tensor = _tensor(row)
            if tensor != c['source_tensor_sha256']:
                raise ValueError('Normalized waveform differs: ' + rid + '; this is not a PNG-only issue')
            image = dest / 'images' / (rid + '.png'); image.parent.mkdir(exist_ok=True)
            if image.exists():
                if sha(image) != c['sensor_sha256']:
                    raise ValueError('Archived image changed: ' + rid)
            else:
                source = Path(cfg['shaoxing_cohort']) / 'images' / (rid + '.png')
                fd, tmpname = tempfile.mkstemp(suffix='.png', dir=str(image.parent)); os.close(fd)
                tmp = Path(tmpname)
                try:
                    if source.is_file() and sha(source) == c['sensor_sha256']:
                        tmp.write_bytes(source.read_bytes())
                    elif reconstruct:
                        sensor.materialize(x, 'ecg', tmp)
                    else:
                        raise ValueError('Original PNG missing/different: ' + rid +
                            '. Run archive preparation with reconstruct=True in the exact PULSE rendering environment; no model loading needed.')
                    if sha(tmp) != c['sensor_sha256']:
                        raise ValueError('Reconstruction does not reproduce original PNG: ' + rid +
                            '. Restore the original PULSE environment; do not replace candidate hashes.')
                    os.replace(str(tmp), str(image))
                finally:
                    if tmp.exists(): tmp.unlink()
            with Image.open(image) as im: size = list(im.size)
            records.append(dict(query_id=rid, source_tensor_sha256=tensor,
                                sensor_sha256=sha(image), image_size=size))
        result = dict(complete=complete and len(records)==len(rows), total=len(rows),
                      verified=len(records), archive=str(dest))
        atomic(dest / 'status.json', result)
        if result['complete']:
            immutable(dest / 'manifest.json', records)
    return result


def verified_provider(cfg):
    rows, candidates, folder = _inputs(cfg)
    dest = archive_folder(cfg); protocol = read(dest / 'protocol.json')
    expected = dict(query_manifest_sha256=sha(folder / 'manifest.json'),
        candidates_sha256=sha(folder / 'candidates.jsonl'), helper_sha256=sha(__file__),
        extraction_stage_sha256=sha(Path(cfg['run_dir']) / 'stages/query_extraction_shaoxing.json'))
    if any(protocol.get(k) != v for k,v in expected.items()):
        raise ValueError('Archive provenance changed')
    archived = unique(read(dest / 'manifest.json'), 'query_id')
    queries = unique(rows, 'id')
    if set(archived) != set(queries): raise ValueError('Archive incomplete')
    cache = {}
    def provide(row):
        rid = row['id']
        for k,v in queries[rid].items():
            if row.get(k) != v: raise ValueError('Requested query differs from frozen manifest: ' + rid)
        if rid not in cache:
            _, tensor = _tensor(queries[rid])
            if tensor != candidates[rid,'sensor_only']['source_tensor_sha256']:
                raise ValueError('Source tensor differs: ' + rid)
            cache[rid] = tensor
        image = dest / 'images' / (rid + '.png'); c = candidates[rid,'sensor_only']
        if archived[rid]['source_tensor_sha256'] != cache[rid] or archived[rid]['sensor_sha256'] != c['sensor_sha256'] or sha(image) != c['sensor_sha256']:
            raise ValueError('Archived PNG/tensor differs: ' + rid)
        return str(image), cache[rid]
    return provide


def generate_archived_context_fit(cfg, runtime, **settings):
    """Keep context-size, candidate, leakage and generation-resume checks active."""
    import queries
    from extensions.context_fit import generate_context_fit
    from generation import folder_for
    dest = archive_folder(cfg); provider = verified_provider(cfg)
    folder = folder_for(cfg, runtime['experiment_model_key'], runtime['enable_thinking'], settings.get('max_new_tokens',384))
    immutable(folder / 'archived_input_protocol.json', dict(helper_sha256=sha(__file__),
        archive_protocol_sha256=sha(dest/'protocol.json'), archive_manifest_sha256=sha(dest/'manifest.json'),
        policy='All generators reuse exact PULSE candidate PNGs; existing checks remain active'))
    def factory(config):
        if config != cfg: raise ValueError('Image provider requested a different configuration')
        return provider
    with patch.object(queries, 'image_provider', side_effect=factory):
        return generate_context_fit(cfg, runtime, **settings)
