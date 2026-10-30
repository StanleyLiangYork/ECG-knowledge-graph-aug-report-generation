"""Retry only truncated CXR reports without changing the original run (Python 3.9)."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import sensor_candidate_generation as generation
from candidate_preparation import read, read_lines, sha, text_sha, write

TRUNCATION = 'Output token budget reached; incomplete output excluded'


def recover_cxr_truncations(tasks, arrays, runtime, directory, max_new_tokens=1024,
                            repetition_control=False):
    """Return a combined export directory. Do not use that directory for generation."""
    if runtime['modality'] != 'cxr':
        raise ValueError('Use the loaded MedGemma CXR runtime')
    base = Path(directory).resolve()
    protocol = read(base / 'cxr_protocol.json')
    if max_new_tokens <= protocol['max_new_tokens']:
        raise ValueError('Recovery must increase the output token budget')
    # Verify the same code, model, prompt and software; only the budget may change.
    expected = dict(
        model=generation.MODELS['cxr'], revision=runtime['revision'],
        prompt=generation.PROMPTS['cxr'], max_new_tokens=protocol['max_new_tokens'],
        rendering='symile_npy_v1_imagenet_inverse_or_12lead_normalized_500Hz',
        code_sha256=sha(generation.__file__),
        candidate_helpers_sha256=sha(Path(generation.__file__).with_name('candidate_preparation.py')),
        **{key: generation.version(package) for key, package in
           [('torch', 'torch'), ('transformers', 'transformers'),
            ('numpy', 'numpy'), ('pillow', 'Pillow')]})
    if protocol != expected:
        raise ValueError('Original model/code/prompt/environment changed; restore the original runtime and helpers')
    signature = text_sha(json.dumps(protocol, sort_keys=True))
    paths = {m: base / f'{m}_sensor_only.jsonl' for m in ('ecg', 'cxr')}
    hashes = {m: sha(p) for m, p in paths.items()}
    latest = {r['id']: r for r in read_lines(paths['cxr'])}
    if len({t['id'] for t in tasks}) != len(tasks) or set(latest) != {t['id'] for t in tasks}:
        raise ValueError('Finish all original CXR attempts before recovery; cohort must match exactly')
    retry_tasks = []
    for task in tasks:
        previous = latest[task['id']]
        array = arrays[task['split']][task['source_index']]
        source_hash = text_sha(text_sha(json.dumps({'shape': list(array.shape), 'dtype': str(array.dtype)}))
                               + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
        if (previous.get('protocol_signature') != signature
                or previous.get('source_tensor_sha256') != source_hash
                or any(previous.get(k) != task[k] for k in
                       ('patient_id', 'admission_id', 'split', 'modality', 'study_id'))):
            raise ValueError('Original sensor, identity or protocol mismatch')
        if previous['status'] != 'ok':
            if previous.get('error') != TRUNCATION:
                raise ValueError('Non-truncation failure found; inspect it before recovery')
            retry_tasks.append(task)
    if not retry_tasks:
        return {'directory': str(base), 'total': len(tasks), 'ok': len(tasks), 'failed': 0}

    suffix = '_rep110_ng12' if repetition_control else ''
    retry_dir = base / f'cxr_truncation_retry_{max_new_tokens}{suffix}'
    decoding_protocol = None
    original_config = None
    if repetition_control:
        original_config = runtime['model'].generation_config
        retry_config = deepcopy(original_config)
        retry_config.repetition_penalty = 1.1
        retry_config.no_repeat_ngram_size = 12
        decoding_protocol = json.loads(json.dumps(dict(
            generation_config=retry_config.to_dict(),
            explicit_generation_kwargs=dict(do_sample=False, max_new_tokens=max_new_tokens,
                                            pad_token_id=runtime['tokenizer'].pad_token_id),
            recovery_code_sha256=sha(__file__), original_hashes=hashes,
            retry_ids=sorted(t['id'] for t in retry_tasks)), default=str))
        retry_dir.mkdir(parents=True, exist_ok=True)
        decoding_path = retry_dir / 'decoding_protocol.json'
        if decoding_path.exists():
            if read(decoding_path) != decoding_protocol:
                raise ValueError('Retry decoding settings, cohort or code changed; refusing to mix runs')
        elif (retry_dir / 'cxr_sensor_only.jsonl').exists():
            raise ValueError('Retry reports lack their decoding protocol')
        write(decoding_path, decoding_protocol)
        runtime['model'].generation_config = retry_config
    try:
        generation.run_generation(retry_tasks, arrays, runtime, retry_dir,
                                  limit=0, retry_errors=True, max_new_tokens=max_new_tokens)
    finally:
        if original_config is not None:
            runtime['model'].generation_config = original_config
    retry_protocol = read(retry_dir / 'cxr_protocol.json')
    if retry_protocol != dict(protocol, max_new_tokens=max_new_tokens):
        raise ValueError('Unexpected retry protocol change')
    retries = {r['id']: r for r in read_lines(retry_dir / 'cxr_sensor_only.jsonl')}
    retry_signature = text_sha(json.dumps(retry_protocol, sort_keys=True))
    if set(retries) != {t['id'] for t in retry_tasks}:
        raise ValueError('Retry cohort mismatch')
    for rid, row in retries.items():
        original = latest[rid]
        if (row.get('protocol_signature') != retry_signature
                or any(row.get(k) != original.get(k) for k in
                       ('patient_id', 'admission_id', 'split', 'modality', 'study_id', 'source_tensor_sha256'))):
            raise ValueError('Retry identity/provenance mismatch')
        if decoding_protocol is not None:
            row['decoding_protocol_sha256'] = text_sha(json.dumps(decoding_protocol, sort_keys=True))
            row['decoding_overrides'] = dict(repetition_penalty=1.1, no_repeat_ngram_size=12)
        latest[rid] = row
    if any(sha(paths[m]) != hashes[m] for m in paths):
        raise ValueError('Original reports changed during recovery; stop other generation jobs')

    # A new export view preserves originals and both protocols. Successful originals
    # are copied unchanged. The usual 00g checks still reject any remaining failures.
    combined = base.with_name(base.name + f'_cxr_recovered_{max_new_tokens}{suffix}')
    combined.mkdir(exist_ok=True)
    audit_path = combined / 'recovery_audit.json'
    if any(combined.iterdir()) and not audit_path.is_file():
        raise ValueError('Combined directory already contains unrelated or incomplete files')
    if audit_path.exists():
        prior = read(audit_path)
        if prior['original_hashes'] != hashes or prior['original_directory'] != str(base):
            raise ValueError('Combined directory belongs to different original reports')
    audit = dict(original_directory=str(base), original_hashes=hashes,
                 original_protocol=protocol, retry_protocol=retry_protocol,
                 retry_directory=str(retry_dir), retry_ids=sorted(retries),
                 recovery_code_sha256=sha(__file__),
                 decoding_protocol=decoding_protocol,
                 policy='Retry every original truncated CXR output with the recorded budget and decoding settings; retain all other outputs unchanged.')
    write(audit_path, audit)
    for modality in ('ecg', 'cxr'):
        destination = combined / f'{modality}_sensor_only.jsonl'
        temp = destination.with_suffix('.jsonl.tmp')
        if modality == 'ecg':
            temp.write_bytes(paths['ecg'].read_bytes())
        else:
            temp.write_text(''.join(json.dumps(latest[t['id']], allow_nan=False) + '\n' for t in tasks))
        temp.replace(destination)
    summary = dict(total=len(tasks), ok=sum(r['status'] == 'ok' for r in latest.values()),
                   failed=sum(r['status'] != 'ok' for r in latest.values()), not_attempted=0)
    write(combined / 'cxr_coverage.json', summary)
    return dict(directory=str(combined), **summary)
