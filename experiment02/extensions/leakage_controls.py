"""Additive, target-free packet audits and a fail-closed report-generation gate.

Nothing here changes the Python sources signed by BR00. This checks recorded
identities/provenance and prompt contents, not model pretraining or causality.
"""
import csv
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from common import (ARMS, append, assert_config, atomic, budget, csv_save, immutable,
                    latest, lines, lock, parent_setup, patient, read, sha, signature,
                    uid, unique)

VERSION = 'leakage_controls_v1'
LIMITS = [
    'Cross-dataset waveform and near-image duplication are not established by this audit.',
    'Foundation-model pretraining overlap is unknown.',
    'Frozen encoder training history, validation-only selection and clinical label independence require provenance review.',
    'Graph co-occurrence and reverse retrieval do not establish causality.',
    'File signatures are consistency checks, not protection against an adversary rewriting every artifact.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage(run, name):
    value = read(run / 'stages' / (name + '.json'))
    require(value['stage'] == name and value['parent_protocol'] == signature(read(run / 'protocol.json')),
            'Stage belongs to a different experiment: ' + name)
    return value


def graph_from_training(records):
    """Reconstruct distinct-patient/admission counts without the old quadratic scan."""
    groups = defaultdict(list)
    for r in records:
        if r['split'] == 'train':
            groups[uid(r['admission_id'])].append(r)
    vocabulary = sorted({f"{r['modality']}:{c['concept']}:{c['polarity']}"
                         for rows in groups.values() for r in rows for c in r['candidates']})
    lookup = {v: i for i, v in enumerate(vocabulary)}
    counts = {}; training_patients = set()
    for aid, rows in groups.items():
        patients = {patient(r['patient_id']) for r in rows}
        require(len(patients) == 1, 'Training admission has multiple patients: ' + aid)
        pid = next(iter(patients)); training_patients.add(pid)
        nodes = sorted({lookup[f"{r['modality']}:{c['concept']}:{c['polarity']}"]
                        for r in rows for c in r['candidates']})
        for pair in combinations(nodes, 2):
            support = counts.setdefault(pair, [set(), set()])
            support[0].add(pid); support[1].add(aid)
    return dict(vocabulary=vocabulary, edges=[dict(source=a, target=b, patient_count=len(p), admission_count=len(d))
        for (a, b), (p, d) in sorted(counts.items())], predicate='same_admission_cooccurrence',
        training_patients=sorted(training_patients))


def _expected_case(query, arm, retrieval, symile, mimic, graph, control, cfg):
    from retrieval import graph_support
    refs = []; relationships = []
    if arm != 'sensor_only':
        for a in retrieval['anchors']:
            original = symile[a['id']]
            refs.append(dict(record_id=a['id'], source_type='symile_reference_admission',
                findings=[dict(f, modality=m) for m in ('ecg', 'cxr') for f in original[m + '_findings']],
                labs=original['labs']))
        for rid in retrieval['selected'][arm]:
            original = mimic[rid]
            refs.append(dict(record_id=rid, source_type='other_patient_mimic_cxr',
                             findings=[dict(f, modality='cxr') for f in original['findings']], labs=[]))
            if arm in ('bidirectional_graph', 'bidirectional_shuffled_graph'):
                support = graph_support(query.get('findings', []), original['findings'],
                    graph if arm == 'bidirectional_graph' else control, cfg['min_graph_patients'])
                relationships.extend(dict(e, reference_id=rid) for e in support)
    return dict(query_id=query['id'], patient_id=query['patient_id'], admission_id=query['admission_id'],
                study_id=query['study_id'], modality='ecg', arm=arm, references=refs,
                relationships=relationships, conflicts=retrieval['conflicts'] if arm != 'sensor_only' else [],
                fallbacks=retrieval['fallbacks'])


def _expected_conflicts(query, anchors, candidates):
    conflicts = []
    for r in anchors:
        for modality in ('ecg', 'cxr'):
            states = defaultdict(set)
            for f in r[modality + '_findings']: states[f['concept']].add(f['polarity'])
            conflicts.extend(dict(source=r['id'], modality=modality, concept=c, polarities=sorted(v))
                             for c, v in states.items() if {'present', 'absent'} <= v)
    for r in candidates:
        states = defaultdict(set)
        for f in r['findings']: states[f['concept']].add(f['polarity'])
        conflicts.extend(dict(source=r['id'], modality='cxr', concept=c, polarities=sorted(v))
                         for c, v in states.items() if {'present', 'absent'} <= v)
    qstates = defaultdict(set)
    for f in query.get('findings', []): qstates[f['concept']].add(f['polarity'])
    for r in anchors:
        states = defaultdict(set)
        for f in r['ecg_findings']: states[f['concept']].add(f['polarity'])
        for concept, values in states.items():
            q = qstates[concept]
            if ('present' in q and 'absent' in values) or ('absent' in q and 'present' in values):
                conflicts.append(dict(source=r['id'], modality='ecg', concept=concept,
                    query_polarities=sorted(q), reference_polarities=sorted(values),
                    interpretation='query candidate and other-patient reference disagree; do not transfer the reference assertion'))
    return conflicts


def _forbidden_fields(value):
    forbidden = {'targets', 'ground_truth', 'query_report', 'report_text', 'machine_diagnosis',
                 'labels', 'reference_report', 'reference_text', 'target_findings'}
    if isinstance(value, dict):
        if 'target' in value:
            require(set(value) <= {'source', 'target', 'patient_count', 'admission_count', 'source_token', 'target_token'}
                    and type(value['target']) is int, 'Unexpected target field in evidence')
        require(not (set(value) & forbidden), 'Target/report field entered retrieval evidence: ' + str(set(value) & forbidden))
        for child in value.values(): _forbidden_fields(child)
    elif isinstance(value, list):
        for child in value: _forbidden_fields(child)


def _context(cfg):
    from inputs import assert_inputs, observations
    from queries import load_queries
    from retrieval import shuffled_graph
    assert_config(cfg); assert_inputs(cfg); parent_setup(cfg)
    run = Path(cfg['run_dir']); cohort = cfg['cohort']; out = Path(cfg['frozen_out'])
    context_meta = read(run / 'protocol.json').get('context_fit')
    if context_meta:
        require(sha(Path(__file__).with_name('context_fit.py')) == context_meta['helper_sha256'], 'Context-fit helper changed')
        require(all(cfg[k] == v for k, v in context_meta['caps'].items()), 'Common context caps changed')
        source = Path(context_meta['source_run'])
        require(sha(source / 'protocol.json') == context_meta['parent_protocol_sha256']
                and sha(source / 'packets' / cohort / 'manifest.json') == context_meta['source_packet_manifest_sha256'], 'Context-fit source provenance changed')
    records = read(out / 'records.json'); frozen = unique(records, 'id')
    split_patients = defaultdict(set)
    for r in records:
        require(r['split'] in ('train', 'val', 'test'), 'Unknown patient split')
        split_patients[r['split']].add(patient(r['patient_id']))
    for a, b in combinations(split_patients, 2):
        require(not split_patients[a] & split_patients[b], 'Patient overlap between ' + a + ' and ' + b)
    graph = read(out / 'graph.json')
    require(graph == graph_from_training(records), 'Frozen graph differs from training-only candidate graph/counts')
    symile_rows = read(run / 'banks/symile.json'); mimic_rows = read(run / 'banks/mimic.json')
    symile = unique(symile_rows, 'id'); mimic = unique(mimic_rows, 'id')
    require(symile and mimic, 'Empty reference bank')
    ready = stage(run, 'banks_ready')['inputs']
    require(ready['symile_sha256'] == sha(run / 'banks/symile.json') and ready['mimic_sha256'] == signature(mimic_rows),
            'Finalized reference bank changed')
    # Children share their parent's extraction and lab provenance without rewriting it.
    bank_run = (run / 'banks').resolve().parent
    bundle_stage = read(bank_run / 'stages/symile_bundles.json')['inputs']
    require(bundle_stage['bank'] == signature(symile_rows), 'Training admission bundle signature mismatch')
    for name, digest in bundle_stage['arrays'].items():
        require(sha(Path(cfg['symile_root']) / 'data_npy/train' / name) == digest, 'Aligned lab/admission array changed: ' + name)
    require(sha(bank_run / 'extraction/cxr_candidates.jsonl') == ready['candidates_sha256'], 'CXR extraction journal changed after bank finalization')
    rad_path = bank_run / 'audits/rad_equivalence.json'
    require(sha(rad_path) == ready['rad_audit_sha256'] and read(rad_path)['passed'], 'RAD-DINO equivalence audit changed or failed')
    for name, digest in stage(run, 'indexes')['inputs']['matrices'].items():
        require(sha(run / 'banks' / name) == digest, 'Search matrix changed: ' + name)
    if 'recovery' in ready:
        recovery = ready['recovery']; recovery_folder = bank_run / 'extraction' / recovery['version']
        require(sha(recovery_folder / 'protocol.json') == recovery['protocol_sha256']
                and sha(recovery_folder / 'cxr_candidates.jsonl') == recovery['journal_sha256']
                and sha(Path(__file__).with_name('cxr_recovery.py')) == recovery['helper_sha256'], 'Reference recovery provenance changed')
        recovered_protocol = read(recovery_folder / 'protocol.json')
        recovered_stage = read(bank_run / 'stages' / ('cxr_' + recovery['version'] + '.json'))
        require(recovered_stage['inputs'] == recovered_protocol, 'Reference recovery stage/protocol mismatch')
        recovered_ids = {r['id'] for r in mimic_rows if r['extraction'].get('recovery_version') == recovery['version']}
        require(recovered_ids == set(recovery['recovered_ids']), 'Recovered reference identity mismatch')
        for r in mimic_rows:
            if r['id'] in recovered_ids:
                require(r['extraction']['recovery_protocol_sha256'] == signature(recovered_protocol), 'Recovered reference protocol mismatch')
    sources = unique(read(cfg['sensor_sources']), 'id')
    for a in symile_rows:
        require(patient(a['patient_id']) in split_patients['train'], 'Non-training Symile reference patient')
        for modality in ('ecg', 'cxr'):
            r = frozen[a[modality + '_id']]; s = sources[r['id']]
            require(r['split'] == 'train' and r['modality'] == modality and patient(r['patient_id']) == patient(a['patient_id'])
                    and uid(r['admission_id']) == a['id'] == uid(a['admission_id']), 'Symile patient/admission/modality mismatch')
            require(a[modality + '_embedding'] == r['embedding'] and a[modality + '_space'] == r['embedding_space']
                    and a[modality + '_findings'] == observations(r['candidates']), 'Symile features/findings changed')
            require(int(s['source_index']) == a['source_index'] and uid(s['study_id']) == a[modality + '_study_id'], 'Symile sensor alignment mismatch')
    forbidden = set().union(*split_patients.values())
    for split in ('train', 'val', 'test'):
        with (Path(cfg['symile_root']) / (split + '.csv')).open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            require('subject_id' in (reader.fieldnames or []), 'Missing Symile patient crosswalk')
            forbidden.update(patient(r['subject_id']) for r in reader)
    inventory = unique(read(run / 'banks/cxr_inventory.json'), 'id')
    fields = cfg['cxr_fields']; source_rows = lines(cfg['cxr_manifest'])
    partition = unique(lines(cfg['cxr_train_partition']), fields['id'])
    seen_images = set(); seen_studies = set()
    require(set(inventory) == set(mimic), 'CXR inventory and finalized bank disagree')
    for r in mimic_rows:
        require(patient(r['patient_id']).isdigit() and patient(r['patient_id']) not in forbidden, 'MIMIC-CXR overlaps a Symile patient')
        require(r['id'] in partition, 'CXR reference outside explicit training partition')
        require({k: r[k] for k in inventory[r['id']]} == inventory[r['id']], 'CXR inventory identity changed')
        source = source_rows[r['source_index']]
        require(uid(source[fields['id']]) == r['id'] and patient(source[fields['patient']]) == patient(r['patient_id'])
                and uid(source[fields['study']]) == uid(r['study_id']) and source[fields['image']] == r['image_path'], 'CXR source row mismatch')
        require(r['image_sha256'] not in seen_images and (r['patient_id'], r['study_id']) not in seen_studies, 'Duplicate reference image/study')
        seen_images.add(r['image_sha256']); seen_studies.add((r['patient_id'], r['study_id']))
        require(r['extraction']['input_scope'] == 'sensor_only' and r['extraction']['model'] == 'google/medgemma-1.5-4b-it', 'CXR extraction was not approved image-only extraction')
    queries = load_queries(cfg); query_map = unique(queries, 'id')
    query_manifest = read(run / 'queries' / cohort / 'manifest.json')
    require(stage(run, 'queries_' + cohort)['inputs']['rows'] == signature(query_manifest), 'Query manifest changed')
    if cohort.startswith('internal_'):
        split = 'val' if cohort == 'internal_val' else 'test'
        require(set(query_map) == {r['id'] for r in records if r['modality'] == 'ecg' and r['split'] == split}, 'Query cohort missing/extra held-out ECGs')
        for q in queries:
            r = frozen[q['id']]; s = sources[q['id']]
            expected = {k: r[k] for k in ('id', 'patient_id', 'admission_id', 'split', 'modality', 'embedding', 'embedding_space')}
            expected.update(study_id=uid(s['study_id']), source_index=int(s['source_index']), findings=observations(r['candidates']),
                            candidate_scope='existing sensor-only PULSE candidates')
            require(q == expected, 'Query features/candidates are not from frozen sensor-only record')
    else:
        for r in latest(run / 'queries' / cohort / 'candidates.jsonl').values():
            require(r['input_scope'] == 'sensor_only' and r['model'] == 'PULSE-ECG/PULSE-7B', 'External query candidates used unapproved source')
    packet_manifest = read(run / 'packets' / cohort / 'manifest.json')
    arms = list(ARMS if cfg['include_shuffled_graph'] else ARMS[:-1])
    require(packet_manifest['query_ids'] == [q['id'] for q in queries] and packet_manifest['arms'] == arms
            and set(packet_manifest['packets']) == set(query_map), 'Packet manifest cohort/arm mismatch')
    packet_stage = stage(run, 'packets_' + cohort); retrieval_stage = stage(run, 'retrieval_' + cohort)
    require(packet_manifest['protocol_signature'] == signature(packet_stage), 'Packet stage signature mismatch')
    if 'k_sensitivity' not in read(run / 'protocol.json'):
        identity = retrieval_stage['inputs']
        require(identity['queries'] == signature(queries) and identity['symile'] == sha(run / 'banks/symile.json')
                and identity['mimic'] == sha(run / 'banks/mimic.json') and identity['graph'] == signature(graph), 'Retrieval source identity mismatch')
    else:
        meta = read(run / 'protocol.json')['k_sensitivity']
        require(meta['helper_sha256'] == sha(Path(__file__).with_name('k_sweep.py')), 'k-sweep helper changed')
        require(retrieval_stage['inputs']['source_retrieval'] == {k: v for k, v in meta.items() if k != 'selected_k'}
                and meta['selected_k'] == cfg['final_cxr'], 'k-sweep source provenance mismatch')
    return dict(run=run, queries=queries, symile=symile, mimic=mimic, graph=graph,
                control=shuffled_graph(graph, cfg['seed']), packet_manifest=packet_manifest,
                packet_stage=packet_stage, retrieval_stage=retrieval_stage, image_hashes={})


def _check_query(ctx, cfg, query):
    from retrieval import graph_support
    from extensions.k_sweep import FIELDS, select
    run = ctx['run']; qid = query['id']; cohort = cfg['cohort']
    retrieval_path = run / 'retrieval' / cohort / (qid + '.json')
    packet_path = run / 'packets' / cohort / (qid + '.json')
    r = read(retrieval_path); packet = read(packet_path)
    require(r['query_id'] == qid and r['protocol_signature'] == signature(ctx['retrieval_stage']), 'Retrieval query/stage mismatch')
    require(ctx['packet_stage']['inputs']['retrieval'][qid] == sha(retrieval_path), 'Retrieval changed after packet creation')
    require(ctx['packet_manifest']['packets'][qid] == sha(packet_path) and packet['query_id'] == qid
            and packet['protocol_signature'] == signature(ctx['packet_stage']), 'Packet hash/query/stage mismatch')
    _forbidden_fields(r); _forbidden_fields(packet)
    require(r['query_findings'] == query.get('findings', []), 'Query finding evidence changed')
    internal = query['split'] != 'external'
    forbidden = {patient(query['patient_id'])} if internal else set()
    require(len(r['anchors']) <= cfg['k_anchors'], 'Too many forward anchors')
    anchor_patients = set(); anchor_ids = set()
    for a in r['anchors']:
        require(a['id'] in ctx['symile'], 'Unknown forward reference admission')
        original = ctx['symile'][a['id']]
        require({k: a.get(k) for k in original} == original and set(a) == set(original) | {'query_ecg_cosine'}, 'Forward reference content changed')
        pid = patient(original['patient_id'])
        require(pid not in forbidden | anchor_patients and a['id'] not in anchor_ids, 'Forward query/patient overlap or duplicate anchor')
        anchor_patients.add(pid); anchor_ids.add(a['id'])
    candidates = unique(r['candidates'], 'id')
    mimic_order = {rid: i for i, rid in enumerate(ctx['mimic'])}
    reverse_log = []
    for c in candidates.values():
        require(c['id'] in ctx['mimic'], 'Unknown/unapproved MIMIC-CXR candidate')
        original = ctx['mimic'][c['id']]
        require({k: c.get(k) for k in original} == original and c['matrix_index'] == mimic_order[c['id']], 'CXR candidate differs from finalized bank')
        require(c['forward_paths'] and all(p['symile_admission'] in anchor_ids for p in c['forward_paths']), 'CXR candidate has unknown forward path')
        for name, g in (('graph', ctx['graph']), ('shuffled', ctx['control'])):
            require(c[name + '_support'] == graph_support(query.get('findings', []), original['findings'], g, cfg['min_graph_patients']), 'Graph support is not from frozen training graph')
        seen = set()
        require(len(c['reverse_admissions']) <= cfg['reverse_m'], 'Too many reverse admissions')
        for back in c['reverse_admissions']:
            require(back['admission_id'] in ctx['symile'], 'Reverse admission is not training data')
            pid = patient(ctx['symile'][back['admission_id']]['patient_id'])
            require(patient(back['patient_id']) == pid and pid not in forbidden | anchor_patients | seen, 'Reverse patient identity/anchor/query exclusion failed')
            seen.add(pid); reverse_log.append(dict(candidate_id=c['id'], admission_id=back['admission_id'], patient_id=pid))
        random_patients = set()
        require(len(c['random_reverse_admissions']) == len(c['reverse_admissions']), 'Random reverse control size mismatch')
        for aid in c['random_reverse_admissions']:
            require(aid in ctx['symile'], 'Unknown random reverse admission')
            pid = patient(ctx['symile'][aid]['patient_id'])
            require(pid not in forbidden | anchor_patients | random_patients, 'Random reverse control leaks patient')
            random_patients.add(pid)
        if c['id'] not in ctx['image_hashes']:
            ctx['image_hashes'][c['id']] = sha(original['image_path'])
        require(ctx['image_hashes'][c['id']] == original['image_sha256'], 'Candidate CXR image bytes changed')
    require(set(r['selected']) == set(ARMS) and r['selected']['sensor_only'] == [], 'Unexpected selected arm or sensor-only reference')
    for arm, field in FIELDS.items():
        require(r['selected'][arm] == select(r['candidates'], field, cfg['final_cxr']), 'Selected IDs do not follow frozen ranking/patient policy')
    require(r['graph_control_sha256'] == signature(ctx['control']), 'Shuffled graph provenance changed')
    require(r['conflicts'] == _expected_conflicts(query, r['anchors'], r['candidates']), 'Conflict statements are not derived from approved sensor findings')
    expected_short = len({c['patient_id'] for c in r['candidates']}) if 'k_sensitivity' in read(run / 'protocol.json') else len(r['candidates'])
    require(r['fallbacks'] == dict(anchors_short=len(r['anchors']) < cfg['k_anchors'],
        cxr_pool_short=expected_short < cfg['final_cxr'], no_query_concepts=not query.get('findings'),
        no_graph_support=not any(c['graph_support'] for c in r['candidates'])), 'Fallback flags changed')
    local = r['local_subgraph']
    node_sources = defaultdict(list)
    for f in query.get('findings', []):
        node_sources['ecg:' + f['concept'] + ':' + f['polarity']].append(dict(source=qid, role='query_model_candidate'))
    for a in r['anchors']:
        for modality in ('ecg', 'cxr'):
            for f in a[modality + '_findings']:
                node_sources[modality + ':' + f['concept'] + ':' + f['polarity']].append(dict(source=a['id'], role='other_patient_symile_reference'))
    for c in r['candidates']:
        for f in c['findings']:
            node_sources['cxr:' + f['concept'] + ':' + f['polarity']].append(dict(source=c['id'], role='other_patient_mimic_reference'))
    vocab = ctx['graph']['vocabulary']
    expected_local = dict(nodes=[dict(token=k, sources=v, in_training_vocabulary=k in vocab) for k, v in sorted(node_sources.items())],
        edges=[dict(e, source_token=vocab[e['source']], target_token=vocab[e['target']]) for e in ctx['graph']['edges']
               if vocab[e['source']] in node_sources and vocab[e['target']] in node_sources],
        predicate='training_admission_cooccurrence', laboratory_nodes=False)
    require(local == expected_local, 'Local graph contains nodes/edges outside approved evidence')
    expected = [_expected_case(query, arm, r, ctx['symile'], ctx['mimic'], ctx['graph'], ctx['control'], cfg)
                for arm in ctx['packet_manifest']['arms']]
    require(packet['cases'] == expected and set(packet) == {'query_id', 'protocol_signature', 'cases'}, 'Packet contains changed/extra evidence, targets or relationships')
    return dict(query_id=qid, patient_id=query['patient_id'], patient_namespace='MIMIC' if internal else 'Shaoxing',
        passed=True, packet_sha256=sha(packet_path), retrieval_sha256=sha(retrieval_path), graph_sha256=sha(Path(cfg['frozen_out']) / 'graph.json'),
        anchors=[dict(admission_id=a['id'], patient_id=a['patient_id']) for a in r['anchors']],
        candidates=[dict(id=c['id'], patient_id=c['patient_id'], study_id=c['study_id']) for c in candidates.values()],
        selected=r['selected'], reverse_admissions=reverse_log,
        case_sha256={c['arm']: signature(c) for c in expected})


def audit_packets(cfg, max_run_hours=30.):
    """Read experiment inputs; write only separate audit/checkpoint files. Rerunnable."""
    run = Path(cfg['run_dir']); dest = run / 'audits/leakage_controls' / cfg['cohort']
    dest.mkdir(parents=True, exist_ok=True)
    with lock(dest / 'writer.lock'), budget(max_run_hours, dest / 'STOP') as stop:
        # Revoke a previous green summary before checking potentially changed inputs.
        atomic(dest / 'summary.json', dict(complete=False, passed=False, state='checking', version=VERSION))
        try:
            if stop(): raise ValueError('Audit stopped before input verification: ' + stop())
            ctx = _context(cfg); rows = []; details = {}; reason = None
            for query in ctx['queries']:
                if stop(): reason = stop(); break
                try: detail = _check_query(ctx, cfg, query)
                except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
                    detail = dict(query_id=query['id'], patient_id=query['patient_id'], passed=False,
                                  error=type(exc).__name__ + ': ' + str(exc))
                atomic(dest / 'queries' / (query['id'] + '.json'), detail)
                details[query['id']] = detail
                rows.append(dict(query_id=query['id'], patient_id=query['patient_id'], passed=detail['passed'],
                    anchors=len(detail.get('anchors', [])), candidates=len(detail.get('candidates', [])), error=detail.get('error', '')))
            complete = len(rows) == len(ctx['queries'])
            result = dict(version=VERSION, complete=complete, passed=complete and all(r['passed'] for r in rows),
                expected_queries=len(ctx['queries']), checked_queries=len(rows), violating_queries=sum(not r['passed'] for r in rows),
                stop_reason=reason or 'completed', cohort=cfg['cohort'], helper_sha256=sha(__file__),
                dependencies_sha256={'k_sweep.py': sha(Path(__file__).with_name('k_sweep.py'))},
                protocol_sha256=sha(run / 'protocol.json'), packet_manifest_sha256=sha(run / 'packets' / cfg['cohort'] / 'manifest.json'),
                query_audits_sha256=signature(details), limitations=LIMITS, targets_parsed=False, target_labels_used=False,
                scope='Identity/provenance, training-only graph reconstruction and exact approved prompt-evidence checks; no new retrieval or model inference')
            csv_save(dest / 'queries.csv', rows)
        except Exception as exc:
            result = dict(version=VERSION, complete=False, passed=False, state='input_verification_failed',
                          error=type(exc).__name__ + ': ' + str(exc), limitations=LIMITS)
            atomic(dest / 'summary.json', result)
            raise
        atomic(dest / 'summary.json', result)
    return result


def _watch_paths(cfg):
    run = Path(cfg['run_dir']); bank_run = (run / 'banks').resolve().parent
    paths = [Path(cfg[k]) for k in ('prepared_manifest', 'sensor_sources', 'lab_table', 'cxr_manifest',
             'cxr_features', 'cxr_status', 'cxr_train_partition')]
    paths += [Path(cfg['frozen_out']) / n for n in ('freeze.json', 'records.json', 'graph.json')]
    paths += [run / 'protocol.json', run / 'packets' / cfg['cohort'] / 'manifest.json',
              run / 'queries' / cfg['cohort'] / 'manifest.json', Path(__file__)]
    paths += list((run / 'stages').glob('*.json')) + list((run / 'banks').glob('*'))
    paths += [bank_run / 'stages/symile_bundles.json', bank_run / 'extraction/cxr_candidates.jsonl', bank_run / 'audits/rad_equivalence.json']
    paths += [Path(cfg['symile_root']) / (s + '.csv') for s in ('train', 'val', 'test')]
    bundle = read(bank_run / 'stages/symile_bundles.json')['inputs']
    paths += [Path(cfg['symile_root']) / 'data_npy/train' / n for n in bundle['arrays']]
    ready = read(run / 'stages/banks_ready.json')['inputs']
    if 'recovery' in ready:
        recovery_folder = bank_run / 'extraction' / ready['recovery']['version']
        paths += [recovery_folder / 'protocol.json', recovery_folder / 'cxr_candidates.jsonl', Path(__file__).with_name('cxr_recovery.py')]
    context_meta = read(run / 'protocol.json').get('context_fit')
    if context_meta:
        source = Path(context_meta['source_run'])
        paths += [Path(__file__).with_name('context_fit.py'), source / 'protocol.json', source / 'packets' / cfg['cohort'] / 'manifest.json']
    if cfg['cohort'] == 'shaoxing': paths += [run / 'queries/shaoxing/candidates.jsonl']
    paths += list(Path(__file__).resolve().parents[1].glob('*.py'))
    paths += [Path(__file__).with_name('k_sweep.py')]
    return sorted({p for p in paths if p.is_file()})


def _stats(paths):
    return {str(p): (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in paths}


def guarded_generate(cfg, runtime, max_new=5, retry_errors=False, max_run_hours=30., max_new_tokens=384,
                     audit_max_run_hours=1., _runner=None):
    """Full audit on entry, then check each prompt before native inference/resume.

    max_run_hours covers the audit plus generation, excluding model loading.
    No target data or model answers enter the gate.
    """
    import generation
    from packets import format_prompt
    started = time.monotonic()
    result = audit_packets(cfg, max_run_hours=min(audit_max_run_hours, max_run_hours))
    require(result['passed'], 'Leakage audit failed/incomplete; inspect audits/leakage_controls/<cohort>/summary.json')
    run = Path(cfg['run_dir']); cohort = cfg['cohort']; dest = run / 'audits/leakage_controls' / cohort
    details = {qid: read(dest / 'queries' / (qid + '.json')) for qid in read(run / 'packets' / cohort / 'manifest.json')['query_ids']}
    require(signature(details) == result['query_audits_sha256'], 'Per-query audits changed')
    paths = _watch_paths(cfg); stats = _stats(paths)
    folder = generation.folder_for(cfg, runtime['experiment_model_key'], runtime['enable_thinking'], max_new_tokens)
    with lock(folder / 'leakage_guard.lock'):
        baseline = latest(folder / 'reports.jsonl', repair=True)
        identity = dict(version=VERSION, audit_sha256=signature(result), helper_sha256=sha(__file__),
            query_audits_sha256=result['query_audits_sha256'], targets_parsed=False, target_labels_used=False)
        immutable(folder / 'leakage_guard_protocol.json', identity)
        # Retrospective validation of older successes is distinguished from gated new inference.
        initial = folder / 'leakage_guard_initial_journal.json'
        if not initial.exists():
            immutable(initial, dict(preexisting_records=len(baseline), preexisting_keys=[list(k) for k in sorted(baseline)],
                                   reports_sha256=sha(folder / 'reports.jsonl') if (folder / 'reports.jsonl').exists() else None))
        def checked_prompt(case, tokenizer, settings):
            try:
                require(settings == cfg, 'Generator changed audited configuration')
                require(_stats(paths) == stats, 'Audited input changed during generation; stop and rerun audit')
                qid = case['query_id']; detail = details[qid]
                packet = run / 'packets' / cohort / (qid + '.json')
                retrieval = run / 'retrieval' / cohort / (qid + '.json')
                require(sha(packet) == detail['packet_sha256'] and sha(retrieval) == detail['retrieval_sha256'], 'Packet/retrieval changed after audit')
                require(signature(case) == detail['case_sha256'][case['arm']], 'Prompt case differs from approved evidence')
                prompt, exposure = format_prompt(case, tokenizer, settings)
                append(folder / 'leakage_guard_checks.jsonl', dict(query_id=qid, arm=case['arm'], passed=True,
                    audit_sha256=signature(result), case_sha256=signature(case), packet_sha256=sha(packet),
                    prompt_sha256=signature(prompt), check_type='existing_record_validation' if (qid, case['arm']) in baseline else 'before_new_inference'))
                return prompt, exposure
            except Exception as exc:
                append(folder / 'leakage_guard_checks.jsonl', dict(query_id=case.get('query_id'), arm=case.get('arm'),
                    passed=False, error=str(exc), audit_sha256=signature(result)))
                raise ValueError('Leakage generation gate stopped: ' + str(exc)) from exc
        runner = _runner or generation.generate
        remaining = max_run_hours - (time.monotonic() - started) / 3600
        require(remaining > 0, 'Time budget consumed by leakage audit; no generation started')
        with patch.object(generation, 'format_prompt', side_effect=checked_prompt):
            return runner(cfg, runtime, max_new=max_new, retry_errors=retry_errors,
                          max_run_hours=remaining, max_new_tokens=max_new_tokens)


def guarded_generate_sweep(configs, runtime, **kwargs):
    """Use with BR06-created child configurations; preserve its outer STOP/budget."""
    import generation
    from extensions.k_sweep import generate_sweep
    original = generation.generate
    def gated(cfg, loaded, **settings):
        return guarded_generate(cfg, loaded, _runner=original, **settings)
    with patch.object(generation, 'generate', side_effect=gated):
        return generate_sweep(configs, runtime, **kwargs)
