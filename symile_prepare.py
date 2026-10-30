"""Read documented Symile-MIMIC v1 CSVs; no model inference or generated labels."""
import csv
import hashlib
import json
import re
from pathlib import Path

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
          'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
          'Pneumothorax', 'Support Devices']

def locate(root, name):
    direct = root / name
    if direct.is_file(): return direct
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f'Expected one {name}; found {len(matches)}. Set CSV_PATHS explicitly.')
    return matches[0]

def label_value(value):
    value = value.strip()
    if value.lower() in ('', 'nan', 'none', '-1', '-1.0'): return None
    if value in ('0', '0.0'): return 0
    if value in ('1', '1.0'): return 1
    raise ValueError('Unexpected CheXpert label value; review the source encoding')

def clean(value):
    value = str(value or '').strip()
    return '' if value.lower() in ('nan', 'none', 'null') else value


def numeric_id(value):
    value = clean(value)
    if re.fullmatch(r'\d+\.0', value): value = value[:-2]
    return value


def sensor_identity(row, modality):
    path = clean(row.get(f'{modality}_path'))
    aliases = (('ecg_study_id', 'ecg_adm', 'ecg_file_name') if modality == 'ecg'
               else ('cxr_study_id', 'study_id'))
    values = []
    sources = []
    for name in aliases:
        value = clean(row.get(name))
        if value:
            if name == 'ecg_file_name':
                value = Path(value).name
                if value.endswith(('.hea', '.dat')): value = value.rsplit('.', 1)[0]
            values.append(numeric_id(value)); sources.append(name)
    # MIMIC paths identify studies as an s<digits> directory.
    match = re.search(r'(?:^|/)s(\d+)(?:/|$)', path)
    if match:
        values.append(match.group(1)); sources.append(f'{modality}_path')
    if modality == 'ecg' and not match:
        name = Path(path).name
        if name.endswith(('.hea', '.dat')): name = name.rsplit('.', 1)[0]
        if name.isdigit(): values.append(name); sources.append('ecg_path')
    if not values:
        raise ValueError(f'Cannot identify {modality} study. Expected aliases {aliases} '
                         'or a standard MIMIC path containing s<study_id>.')
    if len(set(values)) != 1:
        raise ValueError(f'Conflicting {modality} study identities from {sources}')
    return values[0], sources


def cxr_image_identity(row):
    values = [clean(row.get(k)) for k in ('cxr_dicom_id', 'cxr_24_72_hr') if clean(row.get(k))]
    path = clean(row.get('cxr_path'))
    if Path(path).suffix.lower() in ('.jpg', '.jpeg', '.png', '.dcm'):
        values.append(Path(path).stem)
    if not values: raise ValueError('Cannot identify CXR image from aliases or image path')
    if len(set(values)) != 1: raise ValueError('Conflicting CXR image identities')
    return values[0]


def prepare(csv_paths, tensor_root=None):
    manifest = []; cxr_targets = []; sensors = []; patients = {}; admissions = set(); stats = {}
    required = ['subject_id', 'hadm_id', 'ecg_path', 'cxr_path']
    fingerprints = {}
    for split in ('train', 'val', 'test'):
        path = Path(csv_paths[split])
        fingerprints[split] = {'path': str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        unique = {}; raw_count = 0; ordered_hadm = []
        identity_sources = {'ecg':set(), 'cxr':set()}
        with path.open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = set(required) - set(header)
            if missing: raise ValueError(f'{split}: missing required columns {sorted(missing)}; available: {header}')
            available_labels = [k for k in LABELS if k in header]
            absent_labels = [k for k in LABELS if k not in header]
            if not available_labels:
                raise ValueError(f'{split}: no recognized CXR label columns; inspect the CSV header before mapping labels')
            for row in reader:
                source_index = raw_count
                raw_count += 1
                if any(not clean(row.get(k)) for k in required):
                    raise ValueError(f'{split}: empty identity or sensor-path field at CSV row {raw_count+1}')
                admission = numeric_id(row['hadm_id'])
                row = dict(row, hadm_id=admission, subject_id=numeric_id(row['subject_id']))
                ordered_hadm.append(admission)
                for modality in ('ecg', 'cxr'):
                    study, sources = sensor_identity(row, modality)
                    row[f'{modality}_study_id'] = study
                    identity_sources[modality].update(sources)
                row['cxr_dicom_id'] = cxr_image_identity(row)
                findings = {label:label_value(row.get(label) or '') for label in LABELS}
                identity = tuple(clean(row.get(k)) for k in required + ['ecg_study_id', 'cxr_study_id',
                    'cxr_dicom_id', 'ecg_time', 'cxr_StudyDateTime', 'admittime']) + tuple(findings.values())
                if admission in unique:
                    if unique[admission][0] != identity:
                        raise ValueError(f'{split}: inconsistent repeated admission; refusing arbitrary selection')
                    continue
                unique[admission] = (identity, row, source_index, findings)
        if not unique: raise ValueError(f'{split}: empty cohort')
        tensor_check = 'not checked: no tensor_root supplied'
        if tensor_root is not None:
            import numpy as np
            ids_path = Path(tensor_root) / split / f'hadm_id_{split}.npy'
            if not ids_path.is_file(): raise FileNotFoundError(f'Missing admission identity array: {ids_path}')
            ids = np.load(ids_path, mmap_mode='r', allow_pickle=False).reshape(-1)
            if [numeric_id(str(v)) for v in ids] != ordered_hadm:
                raise ValueError(f'{split}: CSV/admission array identity mismatch')
            tensor_check = 'CSV row order matches hadm_id array'
            fingerprints[split]['hadm_array_sha256'] = hashlib.sha256(ids_path.read_bytes()).hexdigest()
        for admission, (_, row, source_index, findings) in unique.items():
            patient = row['subject_id']
            if admission in admissions: raise ValueError('Admission appears in multiple splits')
            if patient in patients and patients[patient] != split: raise ValueError('Patient leakage across official splits')
            admissions.add(admission); patients[patient] = split
            for modality in ('ecg','cxr'):
                rid = f'{admission}_{modality}'
                manifest.append(dict(id=rid,patient_id=patient,admission_id=admission,split=split,modality=modality))
                sensors.append(dict(id=rid,split=split,admission_id=admission,
                    source_index=source_index, source_csv=str(path),
                    source_path=row[f'{modality}_path'],study_id=row[f'{modality}_study_id'],
                    dicom_id=row['cxr_dicom_id'] if modality=='cxr' else None,
                    acquisition_time=row.get('ecg_time' if modality=='ecg' else 'cxr_StudyDateTime'),
                    admission_time=row.get('admittime')))
            cxr_targets.append(dict(id=f'{admission}_cxr', findings=findings,
                target_source='Symile-MIMIC v1.0.0 supplied CheXpert labels',
                target_revision='1.0.0; absent columns/-1/missing=unknown; explicit 0/1 retained'))
        stats[split] = dict(csv_rows=raw_count,unique_admissions=len(unique),
            repeated_rows_removed=raw_count-len(unique), available_cxr_label_columns=available_labels,
            unavailable_cxr_label_columns=absent_labels,
            identity_sources={k:sorted(v) for k,v in identity_sources.items()},
            tensor_identity_check=tensor_check)
    return manifest,cxr_targets,sensors,dict(splits=stats,patients=len(patients),
        admissions=len(admissions),source_files=fingerprints,
        status='Manifest and available CXR targets only; embeddings, specialist candidates and ECG targets still required')


def save_unchanged_or_new(path, value):
    path = Path(path); payload = json.dumps(value,indent=2,allow_nan=False)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'{path}: different existing data; select a fresh prepared directory')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: f.write(payload)

def merge_annotations(manifest,cxr_targets,ecg_targets):
    expected = {r['id'] for r in manifest if r['modality']=='ecg'}
    if not isinstance(ecg_targets,list) or any(not isinstance(t,dict) for t in ecg_targets):
        raise ValueError('ECG annotations must be a JSON array of objects')
    ids = [t.get('id') for t in ecg_targets]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError('ECG annotation IDs must match the manifest exactly, with no duplicates')
    for target in ecg_targets:
        if not target.get('target_source') or not target.get('target_revision'):
            raise ValueError('Independent ECG target provenance required')
        if not isinstance(target.get('findings'),dict) or not target['findings']:
            raise ValueError('ECG findings must contain independent annotations, not placeholder objects')
        if any(v not in (0,1,None) for v in target['findings'].values()):
            raise ValueError('ECG labels must be 1/0/null')
    return cxr_targets + ecg_targets
