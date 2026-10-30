"""Report-study preparation and reference text linkage; Python 3.9, no model imports."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from candidate_preparation import read, read_lines, sha, uid

ARMS = ('sensor_only', 'ordinary_rag', 'graph_rag')
VERSION = 'matched_reports_v1'


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def resolve(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def validate_frozen(root, out):
    freeze = read(out / 'freeze.json')
    for kind, base in [('artifacts', out), ('source', root)]:
        for name, digest in freeze[kind].items():
            if sha(base / name) != digest:
                raise ValueError('Frozen file changed: ' + str(base / name))
    return freeze


def prepare(root, config, study):
    root, study = Path(root), Path(study)
    out = resolve(root, config['output_dir']); validate_frozen(root, out)
    records = read(out / 'records.json'); rows = {r['id']: r for r in records}
    packets = read(out / 'generation_packets.json')
    rankings = read(out / 'rankings.json'); seed = config['seeds'][0]
    rankmap = {(r['query_id'], r['arm']): r for r in rankings if r['seed'] == seed}
    sensors_path = resolve(root, config['export_sources']['manifest']).with_name('sensor_sources.json')
    sensors = {r['id']: r for r in read(sensors_path)}
    planned = []; seen = set()
    for p in packets:
        q = rows[p['query_id']]; arm = p['arm']; key = (q['id'], arm)
        if key in seen or arm not in ARMS or q['split'] != 'test' or p['patient_id'] != q['patient_id']:
            raise ValueError('Invalid or duplicate packet')
        seen.add(key)
        ids = [] if arm == 'sensor_only' else rankmap[q['id'], 'similarity' if arm == 'ordinary_rag' else 'learned_graph']['ranking'][:config['top_k']]
        if ids != [r['record_id'] for r in p['references']]: raise ValueError('Packet/ranking mismatch')
        refs = []
        for rid in ids:
            r = rows[rid]
            if r['split'] != 'train' or r['patient_id'] == q['patient_id'] or r['modality'] != q['modality']:
                raise ValueError('Unsafe reference identity or split')
            # Only generated candidate observations; no targets or reference report text.
            refs.append(dict(record_id=rid, findings=[dict(concept=c['concept'], polarity=c['polarity']) for c in r['candidates']]))
        sensor = sensors[q['id']]
        if sensor['admission_id'] != q['admission_id'] or sensor['split'] != q['split']:
            raise ValueError('Sensor linkage mismatch')
        planned.append(dict(query_id=q['id'], patient_id=q['patient_id'], admission_id=q['admission_id'],
            modality=q['modality'], split=q['split'], study_id=uid(sensor['study_id']), source_index=sensor['source_index'],
            arm=arm, references=refs))
    expected = {(r['id'], arm) for r in records if r['split'] == 'test' for arm in ARMS}
    if seen != expected: raise ValueError('Incomplete three-arm packet coverage')
    planned.sort(key=lambda r: (r['modality'], r['query_id'], ARMS.index(r['arm'])))
    manifest_path = resolve(root, config['export_sources']['manifest'])
    protocol = dict(version=VERSION, exploratory_after_retrieval_test=True, seed=seed, top_k=config['top_k'],
        max_new_tokens={'ecg': 384, 'cxr': 1024}, repetition_penalty=1.1, no_repeat_ngram_size=12,
        reference_token_cap=80, do_sample=False, arms=list(ARMS), cases_sha256=signature(planned),
        frozen_sha256=sha(out / 'freeze.json'),
        inputs={str(p.resolve()): sha(p) for p in [out/'rankings.json', out/'generation_packets.json', manifest_path, sensors_path]},
        preparation_code_sha256=sha(__file__),
        note='Fixed matched decoding within modality. Query candidates/targets excluded. Reference observations are from other patients. Not a clinical validation.')
    protocol_path = study / 'study_protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('Report study protocol changed; choose a new study directory')
    atomic_json(protocol_path, protocol); atomic_json(study/'generation_cases.json', planned)
    # No target labels/text are included in the generation case file.
    return dict(packets=len(planned), per_modality={m:sum(c['modality']==m for c in planned) for m in ('ecg','cxr')}, study=str(study))


def reference_texts(root, config, cases, extra_jsonl=None, cxr_report_root=None):
    """Load actual text only; never render label dictionaries as reference reports."""
    root=Path(root)
    ann = {r['id']:r for r in read(resolve(root, config['export_sources']['annotations']))}
    wanted={c['query_id']:c for c in cases}; result={}; groups={}
    for rid,c in wanted.items():
        a=ann[rid]; source=a.get('target_source','')
        if c['modality']=='ecg' and '::' in source and a.get('reference_text_sha256'):
            filename, field=source.rsplit('::',1)
            groups.setdefault((filename,field),[]).append((rid,c,a))
    for (filename,field), entries in groups.items():
        path=Path(filename)
        if not path.exists(): continue
        digest=sha(path)
        if any(a.get('target_revision')!=digest for _,_,a in entries):
            raise ValueError('Independent ECG reference source changed: '+filename)
        keys={(uid(c['patient_id']),uid(c['study_id'])):(rid,a) for rid,c,a in entries}
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as f:
            iterator=csv.DictReader(f) if field=='native_report_fields' else (json.loads(line) for line in f if line.strip())
            for row in iterator:
                if row.get('model_id') or row.get('generated_report'): continue
                key=(uid(row.get('subject_id')),uid(row.get('ecg_study_id',row.get('study_id'))))
                if key not in keys: continue
                rid,a=keys[key]
                text=('\n'.join(str(row.get('report_'+str(i)) or '').strip() for i in range(18)
                       if str(row.get('report_'+str(i)) or '').strip().lower() not in ('','nan'))
                      if field=='native_report_fields' else str(row.get(field) or '').strip())
                if hashlib.sha256(text.encode()).hexdigest()==a['reference_text_sha256']:
                    result[rid]=dict(reference_text=text, source=str(path), source_sha256=digest, source_kind='ecg_machine_diagnosis')
    if cxr_report_root:
        base=Path(cxr_report_root)
        for rid,c in wanted.items():
            if c['modality']!='cxr':continue
            patient=uid(c['patient_id']); study_id=uid(c['study_id'])
            relative=Path('p'+patient[:2])/('p'+patient)/('s'+study_id+'.txt')
            candidates=[p for p in [base/relative,base/'files'/relative] if p.is_file()]
            if candidates:
                texts={p.read_text().strip() for p in candidates}
                if len(texts)!=1:raise ValueError('Conflicting CXR reports')
                text=texts.pop()
                if text:result[rid]=dict(reference_text=text,source=str(candidates[0]),source_sha256=sha(candidates[0]),source_kind='cxr_radiology_report')
    if extra_jsonl:
        path=Path(extra_jsonl); source_hash=sha(path)
        seen=set()
        for r in read_lines(path):
            rid=r['id']
            if rid in seen:raise ValueError('Duplicate reference text ID')
            seen.add(rid)
            if rid not in wanted:continue
            c=wanted[rid]
            if any(uid(r.get(k))!=uid(c[k]) for k in ('patient_id','study_id','modality')):
                raise ValueError('Reference text identity mismatch')
            expected='ecg_machine_diagnosis' if c['modality']=='ecg' else 'cxr_radiology_report'
            if r.get('source_kind')!=expected or not r.get('source') or r.get('model_id') or r.get('generated_report'):
                raise ValueError('Only independently sourced reference texts allowed')
            text=r.get('reference_text','').strip()
            if not text:continue
            if rid in result and result[rid]['reference_text']!=text:raise ValueError('Conflicting reference text')
            result[rid]=dict(reference_text=text,source=r['source'],source_sha256=source_hash,source_kind=expected)
    return result
