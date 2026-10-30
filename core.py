"""Symile-only pilot. Sensor candidates are inputs; reference annotations are targets."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def load(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fixture():
    rng = np.random.default_rng(43)
    rows = []; targets = []
    for i in range(36):
        split = 'train' if i < 24 else 'val' if i < 30 else 'test'
        for modality in ['ecg', 'cxr']:
            group = i % 4
            vector = np.eye(4)[group] + rng.normal(0, .4, 4)
            candidate = f'{modality}_finding_{group}'
            rid = f'a{i}_{modality}'
            rows.append(dict(id=rid, patient_id=f'p{i}', admission_id=f'a{i}', split=split,
                modality=modality, embedding=vector.tolist(), embedding_space=f'demo_{modality}_v1',
                candidates=[dict(concept=candidate, polarity='present', model='synthetic',
                    revision='fixture-v1', sensor_sha256='0'*64, text=candidate, quote=candidate)],
                synthetic=True))
            targets.append(dict(id=rid, findings={f'{modality}_finding_{j}': int(j==group) for j in range(4)}, target_source='synthetic_fixture',
                                target_revision='v1'))
    return rows, targets

def audit(rows, targets, synthetic):
    if not rows: raise ValueError('Empty cohort')
    ids = set(); patients = {}; admissions = {}; spaces = {}; counts = {}
    for r in rows:
        if r['id'] in ids: raise ValueError('Duplicate record ID')
        ids.add(r['id'])
        if r['split'] not in ('train','val','test'): raise ValueError('Invalid split')
        if r['modality'] not in ('ecg','cxr'): raise ValueError('Invalid modality')
        if r['synthetic'] is not synthetic: raise ValueError('Mode mismatch; no fixture fallback')
        p = r['patient_id']; a = r['admission_id']
        if p in patients and patients[p] != r['split']: raise ValueError('Patient leakage')
        patients[p] = r['split']
        identity = (p, r['split'])
        if a in admissions and admissions[a] != identity: raise ValueError('Admission ownership mismatch')
        admissions[a] = identity
        key = (a, r['modality'])
        if key in counts: raise ValueError('Export one preselected study per admission/modality')
        counts[key] = 1
        v = np.asarray(r['embedding'], dtype=float)
        if v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) == 0: raise ValueError('Invalid embedding')
        space = (r['embedding_space'], len(v))
        if r['modality'] in spaces and spaces[r['modality']] != space: raise ValueError('Embedding space mismatch')
        spaces[r['modality']] = space
        for c in r['candidates']:
            for field in ['concept','polarity','model','revision','sensor_sha256','text','quote']:
                if not c.get(field): raise ValueError(f'Missing candidate {field}')
            if c['polarity'] not in ('present','absent','uncertain'): raise ValueError('Invalid polarity')
            if c['quote'] not in c['text']: raise ValueError('Candidate quote absent from source')
            if len(c['sensor_sha256']) != 64: raise ValueError('Invalid sensor hash')
            if not synthetic and c['model'] != {'cxr':'google/medgemma-1.5-4b-it','ecg':'PULSE-ECG/PULSE-7B'}[r['modality']]:
                raise ValueError('Unexpected specialist; update protocol explicitly')
    if set(patients.values()) != {'train','val','test'}: raise ValueError('All three splits required')
    for a in admissions:
        if any((a,m) not in counts for m in ('ecg','cxr')): raise ValueError('Missing aligned modality')
    if len(targets) != len(ids) or {t['id'] for t in targets} != ids: raise ValueError('Target coverage mismatch')
    for t in targets:
        if not t.get('target_source') or not t.get('target_revision'): raise ValueError('Target provenance required')
        if any(v not in (0,1,None) for v in t['findings'].values()): raise ValueError('Invalid target state')
    return {'patients':len(patients), 'admissions':len(admissions), 'records':len(rows),
            'synthetic':synthetic, 'spaces':spaces}

def tokens(r):
    return {f"{r['modality']}:{c['concept']}:{c['polarity']}" for c in r['candidates']}

def build_graph(rows):
    train = [r for r in rows if r['split']=='train']
    vocabulary = sorted(set().union(*(tokens(r) for r in train)))
    lookup = {s:i for i,s in enumerate(vocabulary)}
    edge_support = {}
    for admission in sorted({r['admission_id'] for r in train}):
        records = [r for r in train if r['admission_id']==admission]
        nodes = sorted(set().union(*(tokens(r) for r in records)))
        for i,a in enumerate(nodes):
            for b in nodes[i+1:]:
                pair = (lookup[a],lookup[b])
                support = edge_support.setdefault(pair, {'patients':set(), 'admissions':set()})
                support['patients'].add(records[0]['patient_id']); support['admissions'].add(admission)
    return {'vocabulary':vocabulary, 'edges':[dict(source=a,target=b,
        patient_count=len(s['patients']), admission_count=len(s['admissions']))
        for (a,b),s in sorted(edge_support.items())], 'predicate':'same_admission_cooccurrence',
        'training_patients':sorted({r['patient_id'] for r in train})}

def pool(rows, query, limit=50):
    refs = [r for r in rows if r['split']=='train' and r['modality']==query['modality']
            and r['patient_id']!=query['patient_id']]
    q = np.asarray(query['embedding']); q = q / np.linalg.norm(q)
    scored = [(float(np.dot(q,np.asarray(r['embedding'])/np.linalg.norm(r['embedding']))),r['id']) for r in refs]
    return sorted(scored, key=lambda x:(-x[0],x[1]))[:limit]

def relevance(a,b):
    # Unknown/unmentioned findings are not treated as negatives.
    positives = [c for c,v in a['findings'].items() if v==1]
    known = [c for c in positives if b['findings'].get(c) is not None]
    if not known: return None
    return sum(b['findings'][c]==1 for c in known) / len(known)

def ndcg(values,k=3):
    observed = values[:k]
    dcg = sum((2**v-1)/np.log2(i+2) for i,v in enumerate(observed))
    best = sorted(values,reverse=True)[:k]
    ideal = sum((2**v-1)/np.log2(i+2) for i,v in enumerate(best))
    return None if ideal==0 else float(dcg/ideal)

def paired_ci(differences, seed=43, repeats=2000):
    # Caller supplies one paired difference per patient, not per image.
    x = np.asarray(differences, dtype=float)
    if not len(x): return None
    rng=np.random.default_rng(seed)
    boot=np.asarray([rng.choice(x,len(x),replace=True).mean() for _ in range(repeats)])
    return {'mean':float(x.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'patients':len(x)}


def score_report_findings(reference, predicted):
    """Known concept subset only; predicted is an independent annotation mapping."""
    known={k:v for k,v in reference.items() if v in (0,1)}
    tp=sum(v==1 and predicted.get(k)==1 for k,v in known.items())
    fp=sum(v==0 and predicted.get(k)==1 for k,v in known.items())
    fn=sum(v==1 and predicted.get(k)!=1 for k,v in known.items())
    return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        known_concepts=len(known),unassessed_predicted_concepts=sorted(k for k,v in predicted.items() if v==1 and k not in known))
