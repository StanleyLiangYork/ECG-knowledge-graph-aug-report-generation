"""Import existing frozen sensor features with record identity and duplicate checks."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def identifier(value):
    value=str(value).strip()
    return value[:-2] if value.endswith('.0') and value[:-2].isdigit() else value


def read_jsonl(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def import_features(artifact_root,symile_root,manifest_path,sensor_path):
    artifact_root=Path(artifact_root);symile_root=Path(symile_root)
    manifest=json.loads(Path(manifest_path).read_text())
    sensors=json.loads(Path(sensor_path).read_text())
    expected={r['id']:r for r in manifest};sources={r['id']:r for r in sensors}
    if not expected or len(expected)!=len(manifest) or len(sources)!=len(sensors) or set(sources)!=set(expected):
        raise ValueError('Prepared manifest/sensor IDs must be unique and match; rerun 00b')
    files={};splits={};arrays={};dimensions={};patient_split={};admission_split={}
    for split in ('train','val','test'):
        paths={'manifest':artifact_root/'manifests'/f'{split}.jsonl',
               'csv':symile_root/f'{split}.csv',
               'hadm':symile_root/'data_npy'/split/f'hadm_id_{split}.npy',
               'ecg':artifact_root/'features'/split/'ecg_hubert.npy',
               'cxr':artifact_root/'features'/split/'cxr_raddino.npy'}
        for label,path in paths.items():
            if not path.is_file():raise FileNotFoundError(f'{split} {label}: {path}')
            files[str(path)]={'sha256':file_hash(path),'bytes':path.stat().st_size}
        linked=read_jsonl(paths['manifest'])
        with paths['csv'].open(newline='',encoding='utf-8-sig') as f:csv_rows=list(csv.DictReader(f))
        hadm=np.load(paths['hadm'],mmap_mode='r',allow_pickle=False).reshape(-1)
        if not linked or len(linked)!=len(csv_rows) or len(linked)!=len(hadm):
            raise ValueError(f'{split}: CSV/manifest/admission-array row counts differ')
        groups={}
        for i,(r,c,h) in enumerate(zip(linked,csv_rows,hadm)):
            if identifier(r.get('hadm_id'))!=identifier(h) or identifier(c.get('hadm_id'))!=identifier(h):
                raise ValueError(f'{split}: manifest/CSV/admission-array identity mismatch at row {i}')
            if identifier(r.get('subject_id'))!=identifier(c.get('subject_id')):
                raise ValueError(f'{split}: manifest/CSV patient mismatch at row {i}')
            if 'npy_index' in r and identifier(r['npy_index'])!=str(i):
                raise ValueError(f'{split}: npy_index mismatch at row {i}')
            admission=identifier(h);patient=identifier(r['subject_id'])
            if patient in patient_split and patient_split[patient]!=split:raise ValueError('Patient leakage across splits')
            if admission in admission_split and admission_split[admission]!=split:raise ValueError('Admission leakage across splits')
            patient_split[patient]=split;admission_split[admission]=split
            for modality in ('ecg','cxr'):
                rid=f'{admission}_{modality}'
                if rid not in expected:raise ValueError(f'{split}: source admission absent from prepared manifest')
                target=expected[rid];sensor=sources[rid]
                if target['split']!=split or target['modality']!=modality or target['patient_id']!=patient or target['admission_id']!=admission:
                    raise ValueError(f'{split}: prepared record identity mismatch at row {i}')
                if sensor['split']!=split or sensor['admission_id']!=admission:
                    raise ValueError('Prepared sensor identity mismatch')
                study=r.get(f'{modality}_study_id');path=r.get(f'{modality}_path')
                if study is None and path is None:
                    raise ValueError(f'{split}: manifest needs {modality}_study_id or {modality}_path for study verification')
                if study is not None and identifier(study)!=identifier(sensor['study_id']):raise ValueError('Study ID mismatch')
                if path is not None and str(path)!=sensor['source_path']:raise ValueError('Sensor path mismatch')
            groups.setdefault(admission,[]).append(i)
        for modality in ('ecg','cxr'):
            x=np.load(paths[modality],mmap_mode='r',allow_pickle=False)
            if x.ndim!=2 or len(x)!=len(linked) or x.shape[1]<1 or not np.issubdtype(x.dtype,np.number):
                raise ValueError(f'{split}/{modality}: expected a numeric (rows,features) array')
            if modality in dimensions and dimensions[modality]!=x.shape[1]:raise ValueError('Feature dimensions differ across splits')
            dimensions[modality]=x.shape[1]
            for begin in range(0,len(x),1024):
                block=np.asarray(x[begin:begin+1024],dtype=np.float64)
                if not np.isfinite(block).all() or np.any(np.linalg.norm(block,axis=1)==0):
                    raise ValueError(f'{split}/{modality}: invalid or zero feature vector')
            for admission,indexes in groups.items():
                sensor=sources[f'{admission}_{modality}']
                if int(sensor['source_index'])!=indexes[0]:raise ValueError('00b source_index does not match first occurrence')
                if len(indexes)>1 and not np.allclose(x[indexes],x[indexes[0]],atol=1e-5,rtol=1e-5):
                    raise ValueError(f'{split}/{modality}: repeated admission has inconsistent features')
            arrays[(split,modality)]=x
        splits[split]={'source_rows':len(linked),'unique_admissions':len(groups)}
    spaces={m:f'imported_{m}_'+hashlib.sha256(json.dumps({p:v for p,v in files.items() if p.endswith("ecg_hubert.npy" if m=='ecg' else 'cxr_raddino.npy')},sort_keys=True).encode()).hexdigest()[:16] for m in ('ecg','cxr')}
    # Exact manifest coverage, including no extra prepared records.
    if len(manifest)!=2*len(admission_split):raise ValueError('Prepared manifest contains unmatched records')
    rows=[]
    for r in manifest:
        sensor=sources[r['id']]
        rows.append({'id':r['id'],'embedding':arrays[(r['split'],r['modality'])][int(sensor['source_index'])].tolist(),
                     'embedding_space':spaces[r['modality']]})
    provenance={str(Path(p)): {'sha256':file_hash(p)} for p in (manifest_path,sensor_path)}
    config=artifact_root/'config.json'
    if config.is_file():provenance[str(config)]={'sha256':file_hash(config)}
    return rows,{'status':'imported_existing_features','records':len(rows),'splits':splits,
        'dimensions':dimensions,'embedding_spaces':spaces,'source_files':files,'prepared_inputs':provenance,
        'limitation':'File hashes identify imported arrays, not encoder checkpoint/preprocessing equivalence. Confirm all splits used the same frozen encoder and preprocessing from extraction logs.'}


def save_new_pair(embeddings,audit,path,audit_path):
    path=Path(path);audit_path=Path(audit_path)
    values=[(path,embeddings),(audit_path,audit)]
    for destination,value in values:
        if destination.exists() and json.loads(destination.read_text())!=value:
            raise FileExistsError(f'{destination} differs; choose a fresh output path instead of overwriting')
    created=[]
    try:
        for destination,value in values:
            if destination.exists():continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('x') as f:
                created.append(destination);json.dump(value,f,allow_nan=False)
    except Exception:
        for destination in created:destination.unlink(missing_ok=True)
        raise
