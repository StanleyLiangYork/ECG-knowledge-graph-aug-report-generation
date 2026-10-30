"""Separately signed retries for failed reference images; original journal is read-only."""
import copy
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from common import (append, assert_config, atomic, budget, freeze_stage, immutable,
                    latest, lock, parent_setup, read, sha, signature)


def need(condition, message):
    if not condition: raise ValueError(message)


def layout(cfg, recovery_version):
    need(recovery_version and Path(recovery_version).name == recovery_version, 'Simple recovery version name required')
    run = Path(cfg['run_dir']); folder = run / 'extraction' / recovery_version
    return run, folder, run / 'extraction/cxr_candidates.jsonl'


def context(cfg, recovery_version):
    assert_config(cfg); parent_setup(cfg)
    run, folder, journal = layout(cfg, recovery_version)
    original = read(run / 'stages/cxr_extraction.json')
    need(original['parent_protocol'] == signature(read(run / 'protocol.json')), 'Original extraction stage belongs to another experiment')
    inventory = read(run / 'banks/cxr_inventory.json'); saved = latest(journal)
    need(sha(run / 'banks/cxr_inventory.json') == original['inputs']['inventory_sha256'], 'Inventory changed')
    expected = {(r['id'], 'sensor_only') for r in inventory}
    need(set(saved) == expected, 'Finish attempting every BR01B image before recovery')
    for r in inventory:
        old = saved[r['id'], 'sensor_only']
        need(old['protocol_signature'] == signature(original['inputs']) and old['sensor_sha256'] == r['image_sha256'], 'Original attempt provenance mismatch')
    failed = [r for r in inventory if saved[r['id'], 'sensor_only']['status'] != 'ok']
    return run, folder, journal, original, inventory, saved, failed


def recover(cfg, runtime, recovery_version='recovery_t2048_r115_ng8_v1', max_new=2,
            retry_errors=False, max_run_hours=2., max_new_tokens=2048):
    import sensor_candidate_generation as sensor
    from candidate_preparation import extract
    from inputs import observations
    run, folder, journal, original, inventory, saved, failed = context(cfg, recovery_version)
    need(runtime.get('modality') == 'cxr', 'Use the working MedGemma environment/runtime')
    base = original['inputs']
    need(runtime['revision'] == base['revision'] and sha(sensor.__file__) == base['sensor_helper_sha256'], 'Restore original checkpoint and sensor helper')
    libraries = {x: version(x) for x in ('torch', 'transformers', 'Pillow')}
    need(libraries == base['libraries'], 'Use the same library versions as BR01B')
    need(type(max_new_tokens) is int and max_new_tokens > base['max_new_tokens'] and max_new >= 0, 'Use a larger token budget and nonnegative record limit')
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(original_stage_sha256=signature(original), original_journal_sha256=sha(journal),
        inventory_sha256=sha(run / 'banks/cxr_inventory.json'), failed_ids=[r['id'] for r in failed],
        model=base['model'], revision=base['revision'], input_scope='sensor_only', prompt=base['prompt'],
        max_new_tokens=max_new_tokens, decoding=dict(do_sample=False, num_beams=1, repetition_penalty=1.15, no_repeat_ngram_size=8),
        libraries=libraries, sensor_helper_sha256=sha(sensor.__file__), recovery_helper_sha256=sha(__file__))
    freeze_stage(run, 'cxr_' + recovery_version, settings)
    immutable(folder / 'protocol.json', settings)
    sign = signature(settings); log = folder / 'cxr_candidates.jsonl'; processed = 0
    old_config = runtime['model'].generation_config; decoding = copy.deepcopy(old_config)
    for k, v in settings['decoding'].items(): setattr(decoding, k, v)
    decoding.min_new_tokens = None; decoding.min_length = 0; decoding.forced_eos_token_id = None
    runtime['model'].generation_config = decoding
    reason = 'completed_or_limit'
    try:
        with lock(folder / 'writer.lock'), budget(max_run_hours, folder / 'STOP') as stop:
            retries = latest(log, repair=True)
            need(set(retries) <= {(r['id'], 'sensor_only') for r in failed}, 'Unexpected recovery journal IDs')
            for r in failed:
                if stop() or (max_new and processed >= max_new): break
                previous = retries.get((r['id'], 'sensor_only'))
                if previous:
                    need(previous['protocol_signature'] == sign and previous['sensor_sha256'] == r['image_sha256'], 'Recovery resume mismatch')
                    if previous['status'] == 'ok' or not retry_errors: continue
                need(sha(r['image_path']) == r['image_sha256'], 'Reference image changed')
                row = dict(query_id=r['id'], arm='sensor_only', model=base['model'], revision=base['revision'],
                    protocol_signature=sign, sensor_sha256=r['image_sha256'], input_scope='sensor_only',
                    original_error=saved[r['id'], 'sensor_only'].get('error'), recovery_version=recovery_version)
                try:
                    text = sensor.infer(runtime, r['image_path'], max_new_tokens)
                    ids = runtime['tokenizer'].encode(text, add_special_tokens=False)
                    grams = Counter(tuple(ids[i:i+8]) for i in range(max(0, len(ids)-7)))
                    need(max(grams.values(), default=0) <= 8, 'Repetition QC failed')
                    need(bool(text.strip()), 'Empty output')
                    row.update(status='ok', text=text, response_sha256=signature(text), findings=observations(extract(text, 'cxr')))
                except Exception as exc: row.update(status='error', error=str(exc), error_type=type(exc).__name__)
                append(log, row); retries[r['id'], 'sensor_only'] = row; processed += 1
                print(r['id'], row['status'], row.get('error', ''), flush=True)
                if row['status'] == 'error' and any(s in row['error'] for s in ('device-side assert', 'illegal memory access')):
                    reason = 'cuda_context_error_restart_kernel'; break
            reason = stop() or reason
    finally: runtime['model'].generation_config = old_config
    counts = dict(total=len(inventory), original_ok=len(inventory)-len(failed), recovery_required=len(failed),
        recovered=sum(r['status'] == 'ok' for r in retries.values()), processed_this_call=processed, stop_reason=reason,
        failed=[dict(id=r['query_id'], error=r.get('error')) for r in retries.values() if r['status'] != 'ok'],
        not_attempted=len(failed)-len(retries))
    counts['combined_ok'] = counts['original_ok'] + counts['recovered']
    atomic(folder / 'coverage.json', counts)
    return counts


def finalize_recovered_banks(cfg, recovery_version='recovery_t2048_r115_ng8_v1'):
    """Validate base successes and separately signed retries before writing banks."""
    import numpy as np
    from inputs import assert_inputs, observations
    from candidate_preparation import extract
    from retrieval import normalize
    assert_inputs(cfg)
    run, folder, journal, original, inventory, base_rows, failed = context(cfg, recovery_version)
    settings = read(folder / 'protocol.json'); stage = read(run / 'stages' / ('cxr_' + recovery_version + '.json'))
    need(stage['inputs'] == settings and stage['parent_protocol'] == signature(read(run / 'protocol.json')), 'Recovery stage/protocol mismatch')
    need(settings['original_stage_sha256'] == signature(original) and settings['original_journal_sha256'] == sha(journal)
         and settings['recovery_helper_sha256'] == sha(__file__) and settings['failed_ids'] == [r['id'] for r in failed], 'Recovery source/helper changed')
    need(settings['inventory_sha256'] == sha(run / 'banks/cxr_inventory.json'), 'Recovery inventory changed')
    audit = read(run / 'audits/rad_equivalence.json')
    need(audit['passed'] and audit['symile_bank_sha256'] == sha(run / 'banks/symile.json')
         and audit['inventory_sha256'] == sha(run / 'banks/cxr_inventory.json') and audit['cxr_features_sha256'] == sha(cfg['cxr_features']), 'RAD-DINO verification missing/changed')
    retries = latest(folder / 'cxr_candidates.jsonl'); failed_ids = {r['id'] for r in failed}
    need(set(retries) <= {(rid, 'sensor_only') for rid in failed_ids}, 'Unexpected recovery journal IDs')
    rows = []
    for r in inventory:
        recovered = r['id'] in failed_ids
        v = retries.get((r['id'], 'sensor_only')) if recovered else base_rows[r['id'], 'sensor_only']
        protocol = settings if recovered else original['inputs']
        need(v is not None and v['status'] == 'ok', 'Recovery incomplete for ' + r['id'])
        need(v['protocol_signature'] == signature(protocol) and v['sensor_sha256'] == r['image_sha256']
             and v['response_sha256'] == signature(v['text']) and v['model'] == protocol['model'] and v['revision'] == protocol['revision']
             and v['input_scope'] == 'sensor_only', 'Generation provenance mismatch for ' + r['id'])
        need(v['findings'] == observations(extract(v['text'], 'cxr')), 'Finding parser changed for ' + r['id'])
        extraction = dict(model=v['model'], revision=v['revision'], input_scope='sensor_only')
        if recovered: extraction.update(recovery_version=recovery_version, recovery_protocol_sha256=signature(settings))
        rows.append(dict(r, findings=v['findings'], extraction=extraction))
    recovery = dict(version=recovery_version, protocol_sha256=sha(folder / 'protocol.json'),
        journal_sha256=sha(folder / 'cxr_candidates.jsonl'), helper_sha256=sha(__file__), recovered_ids=sorted(failed_ids))
    freeze_stage(run, 'banks_ready', dict(candidates_sha256=sha(journal), rad_audit_sha256=sha(run / 'audits/rad_equivalence.json'),
        symile_sha256=sha(run / 'banks/symile.json'), mimic_sha256=signature(rows), recovery=recovery))
    immutable(run / 'banks/mimic.json', rows)
    vectors = np.load(cfg['cxr_features'], mmap_mode='r', allow_pickle=False)
    symile = read(run / 'banks/symile.json')
    for name, matrix in [('symile_ecg', [r['ecg_embedding'] for r in symile]),
                         ('symile_cxr', [r['cxr_embedding'] for r in symile]),
                         ('mimic_cxr', [vectors[r['source_index']] for r in rows])]:
        path = run / 'banks' / (name + '.npy'); value = normalize(matrix)
        if path.exists(): need(np.array_equal(np.load(path, allow_pickle=False), value), 'Changed search matrix')
        else:
            tmp = path.with_suffix('.tmp.npy'); np.save(tmp, value, allow_pickle=False); tmp.replace(path)
    freeze_stage(run, 'indexes', dict(matrices={p.name: sha(p) for p in sorted((run / 'banks').glob('*.npy'))}))
    return dict(ready=True, symile_admissions=len(symile), mimic_studies=len(rows), recovered_images=len(failed_ids),
                recovery_protocol=str(folder / 'protocol.json'), next='BR02; BR01 final cell is replaced by this recovered-bank finalizer')
