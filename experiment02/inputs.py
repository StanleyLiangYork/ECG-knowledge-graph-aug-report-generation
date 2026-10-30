"""Prepare label-free banks and query cohorts; all joins use explicit identities."""
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from common import *


def validate_frozen_inputs(cfg):
    parent_setup(cfg)
    from report_study import validate_frozen
    freeze=validate_frozen(Path(cfg['project_root']),Path(cfg['frozen_out']))
    if freeze.get('synthetic'):raise ValueError('Real frozen training artifacts required')
    return freeze


def preflight(cfg):
    run=Path(cfg['run_dir']);run.mkdir(parents=True,exist_ok=True)
    required=['frozen_out','symile_root','artifact_root','prepared_manifest','sensor_sources','lab_table',
              'cxr_manifest','cxr_features','cxr_status','cxr_train_partition']
    missing=[dict(key=k,path=cfg.get(k)) for k in required if not cfg.get(k) or not Path(cfg[k]).exists()]
    atomic(run/'audits/input_readiness.json',dict(ready=not missing,missing=missing))
    if missing:raise FileNotFoundError('Configure missing inputs in experiment02/config.json; see audits/input_readiness.json: '+str(missing))
    validate_frozen_inputs(cfg)
    if cfg['cohort'] not in ('internal_val','internal_test','shaoxing'):raise ValueError('Unsupported cohort')
    for k in ('k_anchors','ecg_pool_size','cxr_per_anchor','reverse_m','final_cxr','rrf_constant','context_tokens'):
        if type(cfg[k]) is not int or cfg[k]<1:raise ValueError('Positive integer required: '+k)
    inputs={k:sha(cfg[k]) for k in ('prepared_manifest','sensor_sources','lab_table','cxr_manifest','cxr_features','cxr_status','cxr_train_partition')}
    out=Path(cfg['frozen_out'])
    inputs.update({name:sha(out/name) for name in ('freeze.json','records.json','graph.json')})
    immutable(run/'protocol.json',dict(config={k:v for k,v in cfg.items() if k!='cohort'},inputs=inputs,code=code_hashes(),claims='association and retrieval consistency; no causal claim'))
    return dict(ready=True,output=str(run),next='BR01 / BR01A; no query targets read')


def assert_inputs(cfg):
    run=Path(cfg['run_dir']);p=read(run/'protocol.json')
    if p['config']!={k:v for k,v in cfg.items() if k!='cohort'}:raise ValueError('Config differs from frozen experiment; new run directory required')
    for k in ('prepared_manifest','sensor_sources','lab_table','cxr_manifest','cxr_features','cxr_status','cxr_train_partition'):
        if sha(cfg[k])!=p['inputs'][k]:raise ValueError('Input changed: '+k)
    for name in ('freeze.json','records.json','graph.json'):
        if sha(Path(cfg['frozen_out'])/name)!=p['inputs'][name]:raise ValueError('Frozen artifact changed: '+name)
    validate_frozen_inputs(cfg)


def observations(values):
    allowed={'present','absent','uncertain'};seen=set();result=[]
    for x in values:
        if x.get('polarity') not in allowed or not isinstance(x.get('concept'),str):raise ValueError('Invalid candidate observation')
        key=x['concept'],x['polarity']
        if key not in seen:result.append(dict(concept=key[0],polarity=key[1]));seen.add(key)
    return result


def eligible_cxr(cfg):
    """Read only explicit identity/sensor fields; omit report_text and labels."""
    all_records=read(Path(cfg['frozen_out'])/'records.json')
    forbidden={patient(r['patient_id']) for r in all_records}
    # Include patients outside any prepared subset: use identity columns from every raw Symile split.
    import csv
    for split in ('train','val','test'):
        with (Path(cfg['symile_root'])/(split+'.csv')).open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if 'subject_id' not in (reader.fieldnames or []):raise ValueError('Symile subject_id crosswalk missing')
            forbidden.update(patient(r['subject_id']) for r in reader)
    heldout={patient(r['patient_id']) for r in all_records if r['split']!='train'}
    if not heldout:raise ValueError('No held-out patients identified')
    if any(not p.isdigit() for p in forbidden):raise ValueError('Symile patient IDs must crosswalk to numeric MIMIC subject IDs')
    fields=cfg['cxr_fields'];source=lines(cfg['cxr_manifest']);partition=lines(cfg['cxr_train_partition'])
    byid=unique(source,fields['id']);selected=unique(partition,fields['id'])
    status=np.load(cfg['cxr_status'],mmap_mode='r',allow_pickle=False)
    vectors=np.load(cfg['cxr_features'],mmap_mode='r',allow_pickle=False)
    if vectors.ndim!=2 or len(source)!=len(vectors) or len(source)!=len(status):raise ValueError('CXR manifest/features/status row mismatch')
    positions={uid(r[fields['id']]):i for i,r in enumerate(source)}
    counts=Counter();rows=[]
    for rid,row in selected.items():
        if rid not in byid:raise ValueError('Partition ID absent from source: '+rid)
        i=positions[rid];original=byid[rid]
        if 'source_index' in row and int(row['source_index'])!=i:raise ValueError('CXR source_index mismatch')
        pid=patient(original.get(fields['patient']));study=uid(original.get(fields['study']))
        if not pid.isdigit():raise ValueError('MIMIC-CXR subject_id must be numeric for crosswalk')
        if fields['patient'] in row and patient(row[fields['patient']])!=pid:raise ValueError('CXR patient crosswalk mismatch')
        if fields['study'] in row and uid(row[fields['study']])!=study:raise ValueError('CXR study crosswalk mismatch')
        if pid in forbidden:counts['symile_patient']+=1;continue
        if int(status[i])!=1 or not np.isfinite(vectors[i]).all() or np.linalg.norm(vectors[i])==0:
            counts['unavailable_feature']+=1;continue
        image=Path(original[fields['image']])
        if not image.is_file():raise FileNotFoundError('Missing CXR image: '+str(image))
        rows.append(dict(id=rid,patient_id=pid,study_id=study,source_index=i,image_path=str(image),image_sha256=sha(image)))
    groups=defaultdict(list)
    for row in rows:groups[row['image_sha256']].append(row['id'])
    duplicate={rid for ids in groups.values() if len(ids)>1 for rid in ids}
    counts['duplicate_image_records']=len(duplicate)
    rows=[r for r in rows if r['id'] not in duplicate]
    # One canonical image per study; choice is fixed by stable ID, never by labels.
    rows.sort(key=lambda r:r['id']);seen=set();kept=[]
    for r in rows:
        key=r['patient_id'],r['study_id']
        if key in seen:counts['additional_study_view']+=1;continue
        seen.add(key);kept.append(r)
    audit=dict(partition_records=len(partition),eligible=len(kept),excluded=dict(counts),
               all_symile_patients_excluded=True,missing_identity_policy='error',
               image_duplicate_policy='exclude every exact file duplicate group',
               cross_dataset_waveform_screening=cfg['cross_dataset_waveform_screening'],near_duplicate_images='NOT_PERFORMED')
    if not kept:raise ValueError('Empty eligible MIMIC-CXR bank; inspect patient partitions')
    run=Path(cfg['run_dir']);atomic(run/'audits/leakage.json',audit)
    immutable(run/'banks/cxr_inventory.json',kept)
    return kept


def symile_bundles(cfg):
    records=read(Path(cfg['frozen_out'])/'records.json');sources=unique(read(cfg['sensor_sources']),'id')
    manifest=unique(read(cfg['prepared_manifest']),'id');byad=defaultdict(dict)
    train_patients={patient(r['patient_id']) for r in records if r['split']=='train'}
    if train_patients & {patient(r['patient_id']) for r in records if r['split']!='train'}:raise ValueError('Symile patient split leakage')
    for r in records:
        if r['split']=='train':
            aid=uid(r['admission_id'])
            if r['modality'] in byad[aid]:raise ValueError('More than one modality record per admission')
            byad[aid][r['modality']]=r
    folder=Path(cfg['symile_root'])/'data_npy/train'
    hadm=np.load(folder/'hadm_id_train.npy',mmap_mode='r',allow_pickle=False).reshape(-1)
    labs=np.load(folder/'labs_percentiles_train.npy',mmap_mode='r',allow_pickle=False)
    mask=np.load(folder/'labs_missingness_train.npy',mmap_mode='r',allow_pickle=False)
    if labs.shape!=(len(hadm),50) or mask.shape!=labs.shape or not np.isin(mask,[0,1]).all():raise ValueError('Invalid lab array/mask')
    table=read(cfg['lab_table']);cols=table['columns']
    if len(cols)!=50 or [x['percentile_index'] for x in cols]!=list(range(50)):raise ValueError('Invalid lab column mapping')
    means=read(Path(cfg['symile_root'])/'labs_means.json')
    if [x['percentile_key'] for x in cols]!=sorted(means):raise ValueError('Lab mapping differs from original order')
    import csv
    with (Path(cfg['symile_root'])/'train.csv').open(newline='',encoding='utf-8-sig') as f:
        raw_csv=list(csv.DictReader(f))
    if len(raw_csv)!=len(hadm):raise ValueError('Lab CSV and arrays differ in length')
    required={'hadm_id','subject_id'}|{col['itemid'] for col in cols}|{col['percentile_key'] for col in cols}
    if not raw_csv or required-set(raw_csv[0]):raise ValueError('Official lab CSV columns missing; cannot verify masks/order')
    rows=[]
    for aid,pair in sorted(byad.items()):
        if set(pair)!={'ecg','cxr'}:raise ValueError('Unpaired training admission '+aid)
        a,b=pair['ecg'],pair['cxr'];sa,sb=sources[a['id']],sources[b['id']]
        i=int(sa['source_index'])
        if (i!=int(sb['source_index']) or not 0<=i<len(hadm) or uid(hadm[i])!=aid
            or patient(a['patient_id'])!=patient(b['patient_id'])):raise ValueError('ECG/CXR admission row mismatch')
        for r,s in ((a,sa),(b,sb)):
            m=manifest[r['id']]
            if any(uid(m[k])!=uid(r[k]) for k in ('patient_id','admission_id','split','modality')) or uid(s['admission_id'])!=aid or s['split']!='train':
                raise ValueError('Prepared/frozen/sensor identity mismatch')
        csv_row=raw_csv[i]
        if uid(csv_row['hadm_id'])!=aid or patient(csv_row['subject_id'])!=patient(a['patient_id']):raise ValueError('Lab CSV patient/admission mismatch')
        observed=[]
        for j,col in enumerate(cols):
            raw_value=csv_row[col['itemid']].strip()
            csv_observed=bool(raw_value and raw_value.lower() not in ('nan','na','none'))
            if bool(mask[i,j])!=csv_observed:raise ValueError('Lab observed mask differs from official CSV')
            value=float(labs[i,j]) if mask[i,j] else None
            if value is not None and not np.isclose(value,float(csv_row[col['percentile_key']]),rtol=1e-5,atol=1e-6):raise ValueError('Lab percentile differs from official CSV')
            if value is not None and (not np.isfinite(value) or not 0<=value<=1):raise ValueError('Invalid observed lab percentile')
            observed.append(dict(itemid=col['itemid'],name=col['display_name'],observed=bool(mask[i,j]),
                                 training_ecdf_percentile=value,clinical_abnormality='not determined'))
        rows.append(dict(id=aid,patient_id=patient(a['patient_id']),admission_id=aid,ecg_id=a['id'],cxr_id=b['id'],
                         ecg_embedding=a['embedding'],cxr_embedding=b['embedding'],ecg_space=a['embedding_space'],cxr_space=b['embedding_space'],
                         ecg_findings=observations(a['candidates']),cxr_findings=observations(b['candidates']),labs=observed,
                         source_index=i,cxr_study_id=uid(sb['study_id']),ecg_study_id=uid(sa['study_id'])))
    if not rows:raise ValueError('No paired training bank')
    # Sign array contents, not merely source paths.
    hashes={name:sha(folder/name) for name in ('hadm_id_train.npy','labs_percentiles_train.npy','labs_missingness_train.npy')}
    freeze_stage(cfg['run_dir'],'symile_bundles',dict(arrays=hashes,bank=signature(rows)))
    immutable(Path(cfg['run_dir'])/'banks/symile.json',rows)
    return rows
