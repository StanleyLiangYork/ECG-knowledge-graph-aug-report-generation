"""Evaluation-only Shaoxing preparation and frozen Symile retrieval.

Labels live in targets.json and are never read by encoding, candidates or retrieval.
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from report_study import read, sha, signature, atomic_json, validate_frozen, ARMS
from ecg_external_generation import immutable, budget

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
# Conservative shared ontology. Ambiguous ST-T codes, normal sinus rhythm and
# incomplete mappings are deliberately not made into normal/ischemia targets.
CODE_MAP={
 '164889003':'atrial_fibrillation','164890007':'atrial_flutter',
 '426177001':'bradycardia','427084000':'tachycardia','426761007':'tachycardia',
 '713422000':'tachycardia','233896004':'tachycardia','233897008':'tachycardia',
 '270492004':'conduction_abnormality','195042002':'conduction_abnormality',
 '54016002':'conduction_abnormality','28189009':'conduction_abnormality',
 '27885002':'conduction_abnormality','233917008':'conduction_abnormality',
 '698252002':'conduction_abnormality','164909002':'conduction_abnormality',
 '59118001':'conduction_abnormality','164865005':'infarction',
 '111975006':'qt_abnormality'}



def validate_header(path):
    """Validate supported WFDB structure without guessing missing header fields."""
    h=Path(path)
    lines=h.read_text(encoding='utf-8-sig').splitlines()
    records=[line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')]
    if not records: raise ValueError('Empty WFDB header')
    first=records[0].split()
    if len(first)<4: raise ValueError('Incomplete WFDB recording line')
    try:
        channels=int(first[1]); fs=float(first[2]); samples=int(first[3])
    except ValueError:
        raise ValueError('Malformed recording line (possible merged/truncated signal line): '+records[0]) from None
    if first[0]!=h.stem: raise ValueError('Recording name does not match header filename')
    if (channels,fs,samples)!=(12,500.,5000):
        raise ValueError('Unsupported sampling/length; no silent resampling')
    if len(records)!=13: raise ValueError('Expected exactly 12 signal specification lines')
    signals=[line.split() for line in records[1:]]
    if any(len(fields)<9 for fields in signals): raise ValueError('Incomplete signal specification')
    leads=[fields[-1] for fields in signals]
    if len(set(leads))!=12 or set(leads)!=set(LEADS): raise ValueError('Missing/duplicate lead')
    mats={h.parent/fields[0] for fields in signals}
    if len(mats)!=1 or not next(iter(mats)).is_file(): raise ValueError('Missing/ambiguous waveform')
    dx=[line.partition(':')[2].strip() for line in lines if line.startswith('#Dx:')]
    if len(dx)!=1 or not dx[0]: raise ValueError('Missing diagnosis codes')
    return lines, next(iter(mats)), dx[0]


def prepare(dataset, destination, limit=0, seed=43, reviewed_mapping=None,
            negative_concepts=(), mapping_review_note='draft; clinical review pending', invalid_header_policy='error',
            duplicate_policy='error', reference_hashes_path=None):
    """A seeded sample, independent of labels; limit=0 uses all records.

    Missing codes remain unknown by default. Negative concepts can ONLY be used
    after confirming exhaustive annotation for those categories, recorded in note.
    """
    dataset=Path(dataset); dest=Path(destination); dest.mkdir(parents=True,exist_ok=True)
    if invalid_header_policy not in ('error','exclude'):
        raise ValueError('invalid_header_policy must be error or exclude')
    if duplicate_policy not in ('error','exclude_all'):
        raise ValueError('duplicate_policy must be error or exclude_all')
    if (dest/'cohort_protocol.json').exists():
        raise FileExistsError('Cohort already frozen. Use a new external directory for a changed leakage policy.')
    mapping=CODE_MAP if reviewed_mapping is None else reviewed_mapping
    from candidate_preparation import ECG_RULES
    if set(mapping.values())-set(ECG_RULES): raise ValueError('Unrecognized mapped concept')
    if set(negative_concepts)-set(mapping.values()): raise ValueError('Negative policy outside mapping')
    if negative_concepts and mapping_review_note.startswith('draft'):
        raise ValueError('Document annotation-completeness review before treating missing codes as negatives')
    headers=sorted((dataset/'WFDBRecords').rglob('*.hea'))
    if not headers: raise FileNotFoundError('No WFDB headers under '+str(dataset))
    if limit<0: raise ValueError('limit must be nonnegative')
    names=dataset/'ConditionNames_SNOMED-CT.csv'
    with names.open(encoding='utf-8-sig',newline='') as f: label_names=list(csv.DictReader(f))
    known_codes={r['Snomed_CT'].strip() for r in label_names}
    if set(mapping)-known_codes: raise ValueError('Mapping contains codes absent from label dictionary')
    # Validate the entire inventory before leakage control and sampling.
    valid=[]; excluded=[]
    for h in headers:
        try:
            lines, mat, diagnosis=validate_header(h)
            valid.append((h,mat,diagnosis))
        except (ValueError, UnicodeError) as exc:
            excluded.append(dict(header=str(h),record_id=h.stem,header_sha256=sha(h),reason=str(exc)))
    audit=dict(inventory_headers=len(headers),valid_headers=len(valid),excluded_headers=len(excluded),
               policy=invalid_header_policy,excluded=excluded,replacement_sampling=False)
    atomic_json(dest/'header_validation_audit.json',audit)
    if excluded and invalid_header_policy=='error':
        raise ValueError(f'{len(excluded)} invalid selected headers; see header_validation_audit.json. '
                         'Review exclusions and set invalid_header_policy="exclude" to omit them without repairs.')
    if not valid: raise ValueError('No valid records remain; see header_validation_audit.json')
    # File-byte identity is conservative: exclude every member, never choose by labels.
    file_hashes={str(mat):sha(mat) for _,mat,_ in valid}
    groups={}
    for h,mat,_ in valid: groups.setdefault(file_hashes[str(mat)],[]).append(str(h))
    duplicates=[dict(waveform_sha256=digest,headers=paths) for digest,paths in sorted(groups.items()) if len(paths)>1]
    duplicate_headers={h for group in duplicates for h in group['headers']}
    reference_hashes=set()
    if reference_hashes_path is not None:
        payload=read(reference_hashes_path)
        if payload.get('algorithm')!='sha256_file_bytes' or not payload.get('source_description'):
            raise ValueError('Reference inventory must identify algorithm=sha256_file_bytes and source_description')
        reference_hashes=set(payload['hashes'])
        if not reference_hashes or any(not isinstance(x,str) or len(x)!=64 or any(c not in '0123456789abcdef' for c in x) for x in reference_hashes):
            raise ValueError('Reference inventory must contain nonempty valid SHA256 hashes')
    overlap=[str(h) for h,mat,_ in valid if file_hashes[str(mat)] in reference_hashes]
    blocked=duplicate_headers | set(overlap)
    clean=[entry for entry in valid if str(entry[0]) not in blocked]
    leakage=dict(method='Exact waveform-file byte SHA256; does not detect re-encoded or near-duplicate signals',
        duplicate_policy=duplicate_policy,valid_inventory_records=len(valid),duplicate_groups=duplicates,
        duplicate_group_count=len(duplicates),duplicate_records_excluded=len(duplicate_headers),
        reference_overlap_records=overlap,total_excluded_records=len(blocked),eligible_records=len(clean),
        cross_dataset_screening=('exact_file_hashes_only' if reference_hashes_path is not None else 'NOT_PERFORMED'),
        reference_inventory_sha256=sha(reference_hashes_path) if reference_hashes_path is not None else None,
        reference_source_description=payload['source_description'] if reference_hashes_path is not None else None,
        patient_identity_linkage='NOT_ESTABLISHED: distinct files do not prove distinct patients',
        near_duplicate_screening='NOT_PERFORMED',sampling_order='validate entire inventory; exclude; sample',
        labels_used_for_exclusion=False)
    atomic_json(dest/'leakage_audit.json',leakage)
    atomic_json(dest/'duplicate_waveforms.json',duplicates)
    if duplicates and duplicate_policy=='error':
        raise ValueError('Duplicate groups audited in leakage_audit.json. Set duplicate_policy="exclude_all" to exclude every member.')
    if not clean: raise ValueError('No eligible records remain after leakage control')
    if limit>len(clean): raise ValueError('Requested sample exceeds eligible records; inspect leakage_audit.json')
    if limit:
        order=np.random.default_rng(seed).permutation(len(clean))[:limit]
        clean=sorted((clean[i] for i in order),key=lambda entry:str(entry[0]))
    valid=clean
    manifest=[]; targets=[]; unmapped={}; seen=set()
    for h,mat,diagnosis in valid:
        rid='shaoxing_'+h.stem+'_ecg'
        if rid in seen: raise ValueError('Duplicate WFDB record stem')
        seen.add(rid)
        codes=sorted(set(c.strip() for c in diagnosis.split(',')))
        findings={c:(0 if c in negative_concepts else None) for c in sorted(set(mapping.values()))}
        for code in codes:
            if code in mapping: findings[mapping[code]]=1
            else: unmapped[code]=unmapped.get(code,0)+1
        # No diagnosis/age/sex text appears in this generation manifest.
        manifest.append(dict(id=rid,patient_id='shaoxing_'+h.stem,admission_id=rid,
            study_id=h.stem,modality='ecg',split='external',record=str(h.with_suffix('')),
            header_sha256=sha(h),waveform_sha256=file_hashes[str(mat)],waveform_file=str(mat)))
        targets.append(dict(id=rid,findings=findings,snomed_codes=codes,
            target_source=str(h),target_revision=sha(h)))
    protocol=dict(dataset=str(dataset),seed=seed,limit=limit,mapping=mapping,
        duplicate_policy=duplicate_policy,leakage_audit_sha256=signature(leakage),
        invalid_header_policy=invalid_header_policy,header_validation_audit_sha256=signature(audit),
        negative_concepts=list(negative_concepts),mapping_review_note=mapping_review_note,
        label_dictionary_sha256=sha(names),manifest_sha256=signature(manifest),targets_sha256=signature(targets),
        patient_unit='record ID proxy; patient independence not established by duplicate-file exclusion',
        rendering='wfdb physical signal mV; lead order canonical; global max-absolute normalization; existing 12-lead renderer',
        code_sha256=sha(__file__))
    immutable(dest/'cohort_protocol.json',protocol)
    immutable(dest/'manifest.json',manifest); immutable(dest/'targets.json',targets)
    atomic_json(dest/'label_audit.json',dict(records=len(manifest),unmapped_code_counts=unmapped,
        known_positive_decisions=sum(v==1 for t in targets for v in t['findings'].values()),
        negative_policy='unknown unless explicitly reviewed',reference_reports_available=False))
    return dict(records=len(manifest),destination=str(dest),unmapped_codes=len(unmapped),
                excluded_headers=len(excluded),duplicate_groups=len(duplicates),
                leakage_excluded_records=len(blocked),cross_dataset_screening=leakage['cross_dataset_screening'],
                header_audit=str(dest/'header_validation_audit.json'))


def waveform(row):
    import wfdb
    header=Path(row['record']+'.hea')
    if sha(header)!=row['header_sha256'] or sha(row['waveform_file'])!=row['waveform_sha256']:
        raise ValueError('WFDB source changed')
    rec=wfdb.rdrecord(row['record'],physical=True)
    if rec.fs!=500 or rec.sig_len!=5000: raise ValueError('Waveform sampling changed')
    indexes=[rec.sig_name.index(k) for k in LEADS]
    x=np.asarray(rec.p_signal[:,indexes],dtype=np.float32)
    units=[rec.units[i].lower() for i in indexes]
    for j,unit in enumerate(units):
        if unit=='uv': x[:,j]/=1000
        elif unit=='v': x[:,j]*=1000
        elif unit!='mv': raise ValueError('Unknown voltage unit '+unit)
    if x.shape!=(5000,12) or not np.isfinite(x).all() or not np.any(x): raise ValueError('Invalid ECG')
    return x


def image_provider(cohort):
    import hashlib
    import sensor_candidate_generation as sensor
    cohort=Path(cohort); rows={r['id']:r for r in read(cohort/'manifest.json')}; cache={}
    def provide(case):
        rid=case['query_id']; row=rows[rid]
        if rid not in cache:
            x=waveform(row); x=(x/np.max(np.abs(x))).astype(np.float32).reshape(1,5000,12)
            digest=signature(dict(shape=list(x.shape),dtype=str(x.dtype),bytes=hashlib.sha256(x.tobytes()).hexdigest()))
            path=cohort/'images'/(rid+'.png'); sensor.materialize(x,'ecg',path)
            cache[rid]=(path,digest)
        return cache[rid]
    return provide


def base_cases(cohort):
    return [dict(query_id=r['id'],**{k:r[k] for k in ('patient_id','admission_id','study_id','modality','split')},
                 arm='sensor_only',references=[]) for r in read(Path(cohort)/'manifest.json')]


def candidate_study(cohort, base_study):
    cohort=Path(cohort); dest=cohort/'candidate_generation'; cases=base_cases(cohort)
    plan=read(Path(base_study)/'study_protocol.json')
    plan=dict(plan,cases_sha256=signature(cases),cohort_protocol_sha256=sha(cohort/'cohort_protocol.json'),
        role='external_sensor_only_candidates',arms=['sensor_only'])
    immutable(dest/'study_protocol.json',plan); immutable(dest/'generation_cases.json',cases)
    return dest


def encode(cohort, frozen_out, adapter_path, provenance, max_run_hours=30.):
    """Adapter exposes encode_ecg(signal_mv_500hz: [5000,12]) -> [D].
    Adapter owns EXACT original encoder preprocessing. Never substitute a new model.
    """
    cohort=Path(cohort); frozen_out=Path(frozen_out); adapter_path=Path(adapter_path)
    required=('checkpoint_path','checkpoint_sha256','preprocessing_description','bank_equivalence_confirmed')
    if any(not provenance.get(k) for k in required) or provenance['bank_equivalence_confirmed'] is not True:
        raise ValueError('Provide original encoder checkpoint, preprocessing and bank-equivalence confirmation')
    if sha(provenance['checkpoint_path'])!=provenance['checkpoint_sha256']: raise ValueError('Encoder checkpoint hash mismatch')
    refs=[r for r in read(frozen_out/'records.json') if r['modality']=='ecg' and r['split']=='train']
    if not refs: raise ValueError('No frozen ECG references')
    space=refs[0]['embedding_space']; dim=len(refs[0]['embedding'])
    rows=read(cohort/'manifest.json'); folder=cohort/'features'; folder.mkdir(exist_ok=True)
    protocol=dict(provenance=provenance,adapter_sha256=sha(adapter_path),manifest_sha256=signature(rows),
        embedding_space=space,dimension=dim,bank_records_sha256=sha(frozen_out/'records.json'))
    spec=importlib.util.spec_from_file_location('original_ecg_encoder_adapter',adapter_path)
    adapter=importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    with (folder/'writer.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(folder/'protocol.json',protocol)
        with budget(max_run_hours,folder/'STOP') as stop:
            for row in rows:
                if stop(): break
                path=folder/(row['id']+'.json')
                if path.exists():
                    saved=read(path)
                    if saved['protocol_sha256']!=signature(protocol): raise ValueError('Feature provenance mismatch')
                    continue
                v=np.asarray(adapter.encode_ecg(waveform(row)),dtype=np.float32)
                if v.shape!=(dim,) or not np.isfinite(v).all() or np.linalg.norm(v)==0:
                    raise ValueError('Encoder must return a finite nonzero vector in original bank space')
                atomic_json(path,dict(id=row['id'],embedding=v.tolist(),embedding_space=space,protocol_sha256=signature(protocol)))
    count=sum((folder/(r['id']+'.json')).exists() for r in rows)
    return dict(total=len(rows),ok=count,not_attempted=len(rows)-count)


def retrieve(root, config, cohort, max_run_hours=30.):
    """Label-free external top-50 search and frozen reranking, cached per query."""
    import torch
    from learning import Ranker, features, edges
    from candidate_preparation import extract
    from report_generation import read_latest
    root=Path(root); out=Path(config['output_dir']); cohort=Path(cohort)
    freeze=validate_frozen(root,out)
    if freeze.get('synthetic'): raise ValueError('External evaluation requires real trained artifacts')
    graph=read(out/'graph.json'); seed=freeze['seeds'][0]; kind=freeze['selected']
    model=Ranker(len(graph['vocabulary']),kind)
    model.load_state_dict(torch.load(out/f'{kind}_{seed}.pt',map_location='cpu',weights_only=True)); model.eval()
    bank=[r for r in read(out/'records.json') if r['split']=='train' and r['modality']=='ecg']
    if not bank: raise ValueError('Empty ECG bank')
    bank.sort(key=lambda r:r['id']); ids=[r['id'] for r in bank]
    matrix=np.asarray([r['embedding'] for r in bank],dtype=np.float64)
    matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
    journal=cohort/'candidate_generation/ecg_reports.jsonl'; latest=read_latest(journal)
    cp=read(cohort/'candidate_generation/ecg_generation_protocol.json')
    if cp.get('family')!='pulse': raise ValueError('Keep PULSE candidate extraction fixed across generators')
    if cp['study'].get('cohort_protocol_sha256')!=sha(cohort/'cohort_protocol.json'):
        raise ValueError('Candidate source cohort changed')
    rows=read(cohort/'manifest.json'); fp=read(cohort/'features/protocol.json')
    if fp['bank_records_sha256']!=sha(out/'records.json') or fp['manifest_sha256']!=signature(rows):
        raise ValueError('Feature bank/cohort mismatch')
    candidate_hashes={}; feature_hashes={}
    for row in rows:
        r=latest.get((row['id'],'sensor_only'))
        if not r or r['status']!='ok': raise ValueError('Complete all PULSE candidates first: '+row['id'])
        if r['protocol_signature']!=signature(cp) or r['response_sha256']!=signature(r['text']): raise ValueError('Candidate provenance mismatch')
        if any(str(r.get(k))!=str(row[k]) for k in ('patient_id','admission_id','study_id','modality')):
            raise ValueError('Candidate identity mismatch')
        candidate_hashes[row['id']]=signature(r)
        p=cohort/'features'/(row['id']+'.json'); feature_hashes[row['id']]=sha(p)
    folder=cohort/'retrieval'; folder.mkdir(exist_ok=True)
    identity=dict(freeze_sha256=sha(out/'freeze.json'),manifest_sha256=signature(rows),
        candidates=candidate_hashes,features=feature_hashes,feature_protocol=signature(fp),
        top_k=config['top_k'],pool_size=50,seed=seed,code_sha256=sha(__file__))
    with (folder/'writer.lock').open('a') as lock:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(folder/'protocol.json',identity)
        e=edges(graph)
        with torch.no_grad():
            h=torch.relu(model.layer(model.embedding.weight,e)); z=features(bank,graph)@h
        with budget(max_run_hours,folder/'STOP') as stop:
            for row in rows:
                if stop(): break
                path=folder/(row['id']+'.json')
                if path.exists(): continue
                f=read(cohort/'features'/(row['id']+'.json'))
                if f['id']!=row['id'] or f['embedding_space']!=bank[0]['embedding_space'] or f['protocol_sha256']!=signature(fp):
                    raise ValueError('Feature identity/space mismatch')
                qv=np.asarray(f['embedding'],dtype=np.float64)
                if qv.shape!=(matrix.shape[1],) or not np.isfinite(qv).all() or np.linalg.norm(qv)==0: raise ValueError('Invalid vector')
                similarities=matrix@(qv/np.linalg.norm(qv))
                indexes=sorted(range(len(bank)),key=lambda i:(-float(similarities[i]),ids[i]))[:50]
                query=dict(row,candidates=extract(latest[row['id'],'sensor_only']['text'],'ecg'))
                qx=features([query],graph); bx=features([bank[i] for i in indexes],graph)
                with torch.no_grad():
                    a=(qx@h).expand(len(indexes),-1); b=z[indexes]
                    overlap=(qx*bx).sum(1,keepdim=True)
                    sim=torch.tensor(similarities[indexes],dtype=torch.float32)[:,None]
                    scores=model.head(torch.cat([a*b,torch.abs(a-b),sim,overlap],1)).flatten().tolist()
                reordered=sorted(range(len(indexes)),key=lambda j:(-scores[j],ids[indexes[j]]))
                atomic_json(path,dict(query_id=row['id'],similarity=[ids[i] for i in indexes],
                    learned_graph=[ids[indexes[j]] for j in reordered],
                    query_concepts=sorted({m['concept'] for m in query['candidates']})))
    if any(not (folder/(r['id']+'.json')).exists() for r in rows): return {'complete':False,'restart':'Rerun retrieval'}
    lookup={r['id']:r for r in bank}; cases=[]
    for c in base_cases(cohort):
        ranks=read(folder/(c['query_id']+'.json'))
        for arm in ARMS:
            chosen=[] if arm=='sensor_only' else ranks['similarity' if arm=='ordinary_rag' else 'learned_graph'][:config['top_k']]
            references=[dict(record_id=rid,findings=[dict(concept=x['concept'],polarity=x['polarity']) for x in lookup[rid]['candidates']]) for rid in chosen]
            cases.append(dict(c,arm=arm,references=references))
    base=read(out/'report_study_v1/study_protocol.json')
    plan=dict(base,cases_sha256=signature(cases),cohort_protocol_sha256=sha(cohort/'cohort_protocol.json'),
        external_retrieval_sha256=signature(identity),external=True)
    immutable(cohort/'study_protocol.json',plan); immutable(cohort/'generation_cases.json',cases)
    return dict(complete=True,queries=len(rows),packets=len(cases))


def generator_study(cohort, family):
    if family not in ('pulse','qwen','llama'): raise ValueError('Unknown family')
    cohort=Path(cohort); dest=cohort/'generators'/family
    for name in ('study_protocol.json','generation_cases.json'): immutable(dest/name,read(cohort/name))
    return dest
