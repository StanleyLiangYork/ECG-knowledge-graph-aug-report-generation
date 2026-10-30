"""Reproduce historical internal ECG PNGs in isolated directories, without model inference."""
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
from common import (assert_config, atomic, immutable, read, sha, signature, uid,
                    budget, lock, parent_setup)


def pixel_identity(path):
    from PIL import Image
    with Image.open(path) as im:
        rgba = np.asarray(im.convert('RGBA'))
        # Include alpha: visually different transparency cannot pass as identical.
        return dict(size=list(im.size), rgba_sha256=signature(dict(
            shape=list(rgba.shape), bytes=hashlib.sha256(rgba.tobytes()).hexdigest())))


def reproduce(cfg, folders, destination, *, max_new=0, max_run_hours=2.):
    """Run separately in the historical PULSE and modern rendering environments.

    A row is accepted only if the reconstructed file hash matches EVERY recorded
    source file hash in the selected models, including failures. Different pixels
    or an unreproduced history never receive an equivalence certificate.
    """
    assert_config(cfg)
    parent_setup(cfg)
    import sensor_candidate_generation as sensor
    from evaluation import load_results
    if cfg['cohort'] not in ('internal_val', 'internal_test'):
        raise ValueError('This reconstruction adapter currently supports internal Symile ECGs only')
    if not folders or type(max_new) is not int or max_new < 0:
        raise ValueError('Select rendering-group generators and a nonnegative render limit')
    run, destination = Path(cfg['run_dir']), Path(destination)
    manifest_path = run / 'packets' / cfg['cohort'] / 'manifest.json'
    manifest = read(manifest_path)
    manifest['_file_sha256'] = sha(manifest_path)
    queries = {r['id']: r for r in read(run / 'queries' / cfg['cohort'] / 'manifest.json')}
    libraries = {name: version(name) for name in ('numpy', 'Pillow', 'matplotlib')}
    datasets, sources = {}, {}
    for model, folder in folders.items():
        folder = Path(folder)
        protocol = read(folder / 'protocol.json')
        expected = {name: protocol['libraries'][name] for name in libraries}
        if expected != libraries:
            raise ValueError(f'{model}: use its original rendering environment. Expected {expected}, current {libraries}')
        datasets[model] = load_results(folder, manifest)
        sources[model] = dict(folder=str(folder.resolve()), protocol=sha(folder / 'protocol.json'),
                              reports=sha(folder / 'reports.jsonl'))
    expected_sources = {}
    for q in manifest['query_ids']:
        records = [r for rows in datasets.values() for (rid, arm), r in rows.items() if rid == q]
        if not records:
            raise ValueError('No recorded source for query: ' + q)
        tensors = {r.get('source_tensor_sha256') for r in records}
        images = {r.get('sensor_sha256') for r in records}
        if len(tensors) != 1 or len(images) != 1 or None in tensors or None in images:
            raise ValueError('Selected models are not one consistent rendering group for ' + q)
        expected_sources[q] = dict(tensor=next(iter(tensors)), png=next(iter(images)))
    split = 'val' if cfg['cohort'] == 'internal_val' else 'test'
    folder = Path(cfg['symile_root']) / 'data_npy' / split
    waveform_path, admission_path = folder / f'ecg_{split}.npy', folder / f'hadm_id_{split}.npy'
    waveforms = np.load(waveform_path, mmap_mode='r', allow_pickle=False)
    admissions = np.load(admission_path, mmap_mode='r', allow_pickle=False).reshape(-1)
    protocol = dict(cohort=cfg['cohort'], libraries=libraries, sources=sources,
        helper_sha256=sha(__file__), renderer_sha256=sha(sensor.__file__),
        query_manifest_sha256=sha(run / 'queries' / cfg['cohort'] / 'manifest.json'),
        packet_manifest_sha256=sha(manifest_path), expected_sources=expected_sources,
        rule='Exact reconstruction of historical PNG file hash AND tensor hash; then compare decoded RGBA pixels')
    destination.mkdir(parents=True, exist_ok=True)
    immutable(destination / 'protocol.json', protocol)
    processed, checks = 0, []
    with lock(destination / 'writer.lock'), budget(max_run_hours, destination / 'STOP') as stop:
        for q in manifest['query_ids']:
            path = destination / 'checks' / (q + '.json')
            image = destination / 'images' / (q + '.png')
            if path.exists():
                check = read(path)
                if sha(image) != check['png_sha256'] or pixel_identity(image) != check['pixels']:
                    raise ValueError('Cached reconstruction changed: ' + q)
                checks.append(check)
                continue
            if stop() or (max_new and processed >= max_new):
                break
            row = queries[q]
            index = int(row['source_index'])
            if row['split'] != split or uid(admissions[index]) != uid(row['admission_id']):
                raise ValueError('Waveform/admission alignment failed: ' + q)
            raw = np.asarray(waveforms[index])
            tensor = signature(dict(shape=list(raw.shape), dtype=str(raw.dtype),
                bytes=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()))
            if tensor != expected_sources[q]['tensor']:
                raise ValueError('Historical waveform tensor not reproduced: ' + q)
            # Isolated image destination: never call queries.image_provider here.
            sensor.materialize(raw, 'ecg', image)
            png = sha(image)
            if png != expected_sources[q]['png']:
                atomic(destination / 'first_reconstruction_failure.json', dict(query_id=q,
                    expected=expected_sources[q], observed_tensor=tensor, observed_png=png,
                    libraries=libraries, instruction='Historical rendering not reproduced. Do not waive source checks.'))
                raise ValueError('Historical PNG bytes not reproduced for ' + q + '; inspect first_reconstruction_failure.json')
            check = dict(query_id=q, tensor_sha256=tensor, png_sha256=png,
                pixels=pixel_identity(image), protocol_sha256=signature(protocol))
            atomic(path, check)
            checks.append(check)
            processed += 1
            print(q, 'historical PNG reproduced', flush=True)
    status = dict(complete=len(checks) == len(manifest['query_ids']), total=len(manifest['query_ids']),
        reproduced=len(checks), processed_this_call=processed,
        protocol_sha256=sha(destination / 'protocol.json'))
    atomic(destination / 'status.json', status)
    return status


def compare(left, right, destination):
    """Exact pixel comparison. A different pixel never passes through a tolerance."""
    left, right, destination = Path(left), Path(right), Path(destination)
    protocols = [read(p / 'protocol.json') for p in (left, right)]
    if any(p['helper_sha256'] != sha(__file__) for p in protocols):
        raise ValueError('Reconstruction helper changed')
    if any(not read(p / 'status.json')['complete'] for p in (left, right)):
        raise ValueError('Complete both historical rendering reconstructions first')
    for protocol in protocols:
        for source in protocol['sources'].values():
            folder = Path(source['folder'])
            if sha(folder / 'protocol.json') != source['protocol'] or sha(folder / 'reports.jsonl') != source['reports']:
                raise ValueError('Source generation journal/protocol changed after reconstruction')
    if (protocols[0]['packet_manifest_sha256'] != protocols[1]['packet_manifest_sha256']
            or set(protocols[0]['expected_sources']) != set(protocols[1]['expected_sources'])):
        raise ValueError('Reconstructions used different query packets')
    rows = []
    for q in sorted(protocols[0]['expected_sources']):
        checks = [read(p / 'checks' / (q + '.json')) for p in (left, right)]
        for folder, protocol, check in zip((left, right), protocols, checks):
            image = folder / 'images' / (q + '.png')
            expected = protocol['expected_sources'][q]
            if (check['protocol_sha256'] != signature(protocol)
                    or check['tensor_sha256'] != expected['tensor']
                    or check['png_sha256'] != expected['png']
                    or sha(image) != expected['png'] or pixel_identity(image) != check['pixels']):
                raise ValueError('Historical reconstruction evidence changed: ' + q)
        if checks[0]['tensor_sha256'] != checks[1]['tensor_sha256']:
            raise ValueError('Different source waveform tensors: ' + q)
        rows.append(dict(query_id=q, identical_pixels=checks[0]['pixels'] == checks[1]['pixels'],
            left_pixels=checks[0]['pixels'], right_pixels=checks[1]['pixels']))
    result = dict(total=len(rows), identical_pixels=sum(r['identical_pixels'] for r in rows),
        passed=bool(rows) and all(r['identical_pixels'] for r in rows), queries=rows,
        reconstruction_protocols=[dict(path=str(p / 'protocol.json'), sha256=sha(p / 'protocol.json')) for p in (left, right)],
        interpretation='Only all-query exact pixel equality establishes encoding-only PNG differences. '
            'This audit does not change the frozen evaluator or authorize edited report hashes.')
    atomic(destination, result)
    return result


def evaluate_pixel_equivalent(cfg, folders, targets_path, destination, left, right, **kwargs):
    """Allow encoding-only differences AFTER all historical inputs prove exact pixel equality.

    Canonicalize only validated IN-MEMORY source image identities. On-disk native
    journals, protocol hashes, prompts and outputs remain untouched. Different
    pixels cannot use this adapter; use separate within-generator evaluations.
    """
    import copy
    from unittest.mock import patch
    import evaluation
    destination = Path(destination)
    audit = compare(left, right, destination / 'pixel_equivalence.json')
    if not audit['passed']:
        raise ValueError('Historical images contain different pixels. Do not combine as identical-image inputs; evaluate within each generator.')
    protocols = [read(Path(p) / 'protocol.json') for p in (left, right)]
    audited = {}
    for p in protocols:
        for model, source in p['sources'].items():
            if model in audited:
                raise ValueError('A generator appears in both rendering groups')
            audited[model] = (source, p['expected_sources'])
    if set(audited) != set(folders):
        raise ValueError('Exact selected generator set must be covered by the reconstruction audits')
    by_folder = {}
    for model, folder in folders.items():
        if str(Path(folder).resolve()) != audited[model][0]['folder']:
            raise ValueError('Selected folder differs from reconstructed source')
        by_folder[str(Path(folder).resolve())] = audited[model][1]
    identity = dict(helper_sha256=sha(__file__), comparison_sha256=sha(destination / 'pixel_equivalence.json'),
        source_protocols=audit['reconstruction_protocols'],
        rule='Only exact historical decoded-RGBA identity substitutes for PNG-byte identity; native journals unchanged')
    immutable(destination / 'pixel_equivalence_protocol.json', identity)
    native = evaluation.load_results
    pixels = {r['query_id']: r['left_pixels']['rgba_sha256'] for r in audit['queries']}
    def load(folder, manifest):
        rows = native(folder, manifest)  # Validate original protocol, prompts and reports first.
        expected = by_folder[str(Path(folder).resolve())]
        result = copy.deepcopy(rows)
        for (q, arm), row in result.items():
            if (row['source_tensor_sha256'] != expected[q]['tensor']
                    or row['sensor_sha256'] != expected[q]['png']):
                raise ValueError('Report source is not covered by exact reconstruction')
            row['sensor_sha256'] = pixels[q]
        return result
    with patch.object(evaluation, 'load_results', side_effect=load):
        result = evaluation.evaluate(cfg, folders, targets_path, destination, **kwargs)
    result['image_source_assessment'] = dict(mode='exact_historical_pixel_equivalence',
        audit=str(destination / 'pixel_equivalence.json'), original_journals_modified=False)
    if result['complete']:
        atomic(destination / 'results.json', result)
    return result
