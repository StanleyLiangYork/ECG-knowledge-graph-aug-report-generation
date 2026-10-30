"""Python 3.9-compatible preparation helpers. No model imports at module load."""
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

PARSER_VERSION = 'graph_reasoning_pilot_rules_v1'
MODELS = {'ecg':'PULSE-ECG/PULSE-7B', 'cxr':'google/medgemma-1.5-4b-it'}
ECG_RULES = {'normal_ecg': ['\\bnormal(?:\\s+12[- ]lead)?\\s+(?:ecg|electrocardiogram)\\b',
                '\\b(?:ecg|electrocardiogram)\\s+(?:is\\s+)?within normal '
                'limits\\b'],
 'atrial_fibrillation': ['\\batrial fibrillation\\b',
                         '\\ba[- ]?fib(?:rillation)?\\b'],
 'atrial_flutter': ['\\batrial flutter\\b'],
 'tachycardia': ['\\btachycard(?:ia|ic)\\b', '\\b(?:svt|v[- ]?tach)\\b'],
 'bradycardia': ['\\bbradycard(?:ia|ic)\\b'],
 'conduction_abnormality': ['\\bbundle branch block\\b',
                            '\\b(?:rbbb|lbbb|ivcd)\\b',
                            '\\b(?:av|atrioventricular|heart) block\\b',
                            '\\bintraventricular conduction '
                            '(?:delay|defect)\\b'],
 'ventricular_hypertrophy': ['\\b(?:(?:left|right)\\s+|bi)ventricular '
                             'hypertrophy\\b',
                             '\\b(?:lvh|rvh)\\b'],
 'ischemia': ['\\bischemi\\w*\\b',
              '\\bst[- ]t abnormalit\\w*\\b',
              '\\bst(?:[- ]segment)?\\s+(?:depression|elevation)\\b'],
 'infarction': ['\\bmyocardial infarction\\b',
                '\\binfarct(?:ion|ed)?\\b',
                '\\b(?:stemi|nstemi)\\b'],
 'paced_rhythm': ['\\b(?:atrial|ventricular|dual[- ]chamber)?\\s*paced '
                  'rhythm\\b',
                  '\\bpacing spikes?\\b',
                  '\\bpacemaker rhythm\\b'],
 'low_voltage': ['\\blow[- ]voltage\\b'],
 'qt_abnormality': ['\\b(?:prolonged|long|short)\\s+qt(?:c)?(?: interval)?\\b',
                    '\\bqt(?:c)? prolongation\\b']}
CXR_RULES = {'Atelectasis': ['\\batelecta\\w*\\b'],
 'Cardiomegaly': ['\\bcardiomegal\\w*\\b', '\\benlarged cardiac silhouette\\b'],
 'Consolidation': ['\\bconsolidat\\w*\\b'],
 'Edema': ['\\b(?:pulmonary )?edema\\b', '\\bvascular congestion\\b'],
 'Enlarged Cardiomediastinum': ['\\bwidened mediastinum\\b',
                                '\\benlarged cardiomediastin\\w*\\b'],
 'Fracture': ['\\bfracture\\b'],
 'Lung Lesion': ['\\b(?:lung|pulmonary) (?:nodule|mass|lesion)\\b'],
 'Lung Opacity': ['\\b(?:lung|pulmonary|airspace) opacit\\w*\\b',
                  '\\binfiltrate\\b'],
 'No Finding': ['\\bno acute cardiopulmonary '
                '(?:abnormality|process|finding)\\b',
                '\\bnormal chest\\b'],
 'Pleural Effusion': ['\\bpleural effusion\\b'],
 'Pleural Other': ['\\bpleural thickening\\b', '\\bpleural plaque\\b'],
 'Pneumonia': ['\\bpneumonia\\b'],
 'Pneumothorax': ['\\bpneumothorax\\b'],
 'Support Devices': ['\\b(?:endotracheal|enteric|feeding|ng) tube\\b',
                     '\\b(?:central venous|picc) (?:line|catheter)\\b',
                     '\\bpacemaker\\b',
                     '\\bdefibrillator\\b']}

def read(path): return json.loads(Path(path).read_text())
def read_lines(path):
    with Path(path).open() as f: return [json.loads(line) for line in f if line.strip()]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def text_sha(text):return hashlib.sha256(text.encode()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def uid(x):
    x=str(x).strip();return x[:-2] if re.fullmatch(r'\d+\.0',x) else x

def extract(text,modality):
    """Conservative pilot span parser; requires clinical audit before manuscript use."""
    rules=ECG_RULES if modality=='ecg' else CXR_RULES
    mentions=[]
    for concept,patterns in rules.items():
        for pattern in patterns:
            for m in re.finditer(pattern,text,re.I):
                # Preserve the original local clause as evidence; no rewritten quotation.
                start=max(text.rfind('.',0,m.start()),text.rfind(';',0,m.start()),text.rfind('\n',0,m.start()))+1
                ends=[v for v in (text.find('.',m.end()),text.find(';',m.end()),text.find('\n',m.end())) if v>=0]
                end=min(ends)+1 if ends else len(text)
                before=re.split(r'\b(?:but|however|although|yet)\b',text[start:m.start()],flags=re.I)[-1]
                after=text[m.end():end]
                uncertain=re.search(r'\b(?:possible|possibly|probable|likely|may|might|cannot exclude|cannot rule out|suspicious for|history of|prior|previous|old)\b',before,re.I) or re.match(r'\W*(?:is |are )?(?:not excluded|cannot be excluded)',after,re.I)
                negative=re.search(r'\b(?:no|without|negative for|absence of|free of|not)\b',re.sub(r'\bnot only\b','',before,flags=re.I),re.I) or re.match(r'\W*(?:is |are |was |were )?(?:not seen|absent|not present)\b',after,re.I)
                state='uncertain' if uncertain else 'absent' if negative else 'present'
                mentions.append({'concept':concept,'polarity':state,'quote':text[start:end].strip()})
    # Multiple states remain visible in candidate evidence; targets mask disagreement.
    return list({(m['concept'],m['polarity'],m['quote']):m for m in mentions}.values())

def finding_targets(text):
    mentions=extract(text,'ecg');result={k:None for k in ECG_RULES}
    for concept in result:
        states={m['polarity'] for m in mentions if m['concept']==concept}
        if states=={'present'}:result[concept]=1
        elif states=={'absent'}:result[concept]=0
    return result

def build_annotations(manifest_path,sensor_path,cxr_path,artifact_root,destination,ecg_measurements_csv=None):
    manifest=read(manifest_path);sensors={r['id']:r for r in read(sensor_path)}
    cxr=read(cxr_path);expected_cxr={r['id'] for r in manifest if r['modality']=='cxr'}
    if len(cxr)!=len(expected_cxr) or {r['id'] for r in cxr}!=expected_cxr:raise ValueError('CXR annotation coverage mismatch; rerun 00b')
    artifact=Path(artifact_root)
    sources=[(artifact/'experiment4'/'partition'/'symile'/f'{s}.jsonl','ecg_machine_diagnosis','ecg_study_id') for s in ('train','val','test')]
    sources.append((artifact/'external'/'ecg'/'manifest.jsonl','machine_diagnosis','study_id'))
    lookup={};source_hashes={};available=[]
    for path,field,study_field in sources:
        if not path.is_file():continue
        available.append(str(path));source_hashes[str(path)]=sha(path)
        for r in read_lines(path):
            text=str(r.get(field) or '').strip()
            study=r.get(study_field);patient=r.get('subject_id')
            if not text or study is None or patient is None:continue
            if r.get('model_id') or r.get('generated_report'):continue
            key=(uid(patient),uid(study))
            # Prefer the enriched exact-study partition. Conflicts are retained in the audit.
            lookup.setdefault(key,[]).append((text,str(path),field))
    if ecg_measurements_csv:
        path=Path(ecg_measurements_csv)
        source_hashes[str(path)]=sha(path);available.append(str(path))
        opener=gzip.open if path.suffix=='.gz' else open
        wanted={(r['patient_id'],uid(sensors[r['id']]['study_id'])) for r in manifest if r['modality']=='ecg'}
        with opener(path,'rt',newline='') as f:
            reader=csv.DictReader(f)
            if not {'subject_id','study_id'}.issubset(reader.fieldnames or []):raise ValueError('ECG metadata needs subject_id and study_id')
            report_fields=[f'report_{i}' for i in range(18) if f'report_{i}' in reader.fieldnames]
            if not report_fields:raise ValueError('Expected native MIMIC ECG report_0 ... report_17 fields')
            for row in reader:
                key=(uid(row['subject_id']),uid(row['study_id']))
                if key not in wanted:continue
                text='\n'.join(row[k].strip() for k in report_fields if row.get(k) and row[k].strip().lower()!='nan')
                if text:lookup.setdefault(key,[]).append((text,str(path),'native_report_fields'))
    annotations=list(cxr);coverage=[]
    for r in manifest:
        if r['modality']!='ecg':continue
        sensor=sensors[r['id']];key=(r['patient_id'],uid(sensor['study_id']))
        matches=lookup.get(key,[]);distinct={re.sub(r'\s+',' ',m[0]).strip() for m in matches}
        if len(distinct)==1:
            text,path,field=matches[0];findings=finding_targets(text)
            status='matched' if any(v is not None for v in findings.values()) else 'matched_no_known_concepts'
            source=f'{path}::{field}';source_revision=source_hashes[path];text_hash=text_sha(text)
        else:
            findings={k:None for k in ECG_RULES};status='conflicting_reference_text' if matches else 'missing_reference_text'
            source='unavailable_independent_ecg_reference';source_revision=PARSER_VERSION;text_hash=None
        annotations.append(dict(id=r['id'],findings=findings,target_source=source,
            target_revision=source_revision,parser_revision=PARSER_VERSION,reference_status=status,reference_text_sha256=text_hash))
        coverage.append(dict(id=r['id'],split=r['split'],status=status,known_concepts=sum(v is not None for v in findings.values())))
    audit={'source_files':source_hashes,'available_source_files':available,'ecg_coverage':coverage,
           'policy':'Missing, uncertain, unmentioned, historical and conflicting ECG findings are unknown, not negative. Machine interpretations are noisy reference targets, not adjudicated ground truth.',
           'parser_revision':PARSER_VERSION,'parser_code_sha256':sha(__file__)}
    destination=Path(destination)
    write(destination.with_name('annotation_coverage.json'),audit)
    if not any(c['known_concepts'] for c in coverage):
        raise ValueError('No usable independently sourced ECG concepts. See annotation_coverage.json; no annotations.json was created. Supply linked ECG machine interpretations or expert labels.')
    # Partial reference coverage is explicit; downstream metrics must report masks and denominators.
    write(destination,annotations)
    return {'records':len(annotations),'ecg_cases':len(coverage),'ecg_cases_with_known_concepts':sum(c['known_concepts']>0 for c in coverage),'unknown_or_unmatched_ecg_cases':sum(c['known_concepts']==0 for c in coverage)}

def assemble_candidates(manifest_path,sensor_path,generation_dir,destination):
    manifest=read(manifest_path);sensors={r['id']:r for r in read(sensor_path)}
    latest={};input_hashes={}
    for modality in ('ecg','cxr'):
        path=Path(generation_dir)/f'{modality}_sensor_only.jsonl'
        if not path.is_file():continue
        input_hashes[str(path)]=sha(path)
        for row in read_lines(path):latest[row['id']]=row
    expected={r['id'] for r in manifest}
    if set(latest)-expected:raise ValueError('Unexpected generation IDs: use a fresh Symile-only run directory')
    candidates=[];coverage=[]
    for r in manifest:
        g=latest.get(r['id']);reason=None
        if not g or g.get('status')!='ok':reason='missing_or_failed_generation'
        elif g.get('input_scope')!='sensor_only' or g.get('model')!=MODELS[r['modality']]:reason='wrong_model_or_input_scope'
        elif any(g.get(k)!=r[k] for k in ('patient_id','admission_id','split','modality')):reason='identity_mismatch'
        elif uid(g.get('study_id'))!=uid(sensors[r['id']]['study_id']):reason='study_mismatch'
        elif not g.get('text') or not g.get('revision') or not g.get('sensor_sha256'):reason='incomplete_provenance'
        elif text_sha(g['text'])!=g.get('response_sha256'):reason='response_hash_mismatch'
        if reason:coverage.append(dict(id=r['id'],status=reason));continue
        mentions=extract(g['text'],r['modality'])
        cs=[dict(m,model=g['model'],revision=g['revision'],sensor_sha256=g['sensor_sha256'],
            text=g['text'],parser_revision=PARSER_VERSION,prompt_sha256=g['prompt_sha256'],processor_revision=g['processor_revision']) for m in mentions]
        candidates.append(dict(id=r['id'],candidates=cs))
        coverage.append(dict(id=r['id'],status='ok' if cs else 'ok_no_recognized_concepts',concept_count=len(cs)))
    audit={'records_expected':len(manifest),'records_available':len(candidates),'coverage':coverage,
           'source_hashes':input_hashes,'parser_revision':PARSER_VERSION,'parser_code_sha256':sha(__file__),
           'warning':'Rule extraction is a pilot proxy requiring clinical review. Empty candidates are retained, never silently excluded.'}
    destination=Path(destination);write(destination.with_name('candidate_coverage.json'),audit)
    if len(candidates)!=len(manifest):
        raise ValueError(f'Only {len(candidates)}/{len(manifest)} records have valid sensor-only generations. Finish/retry generation; see candidate_coverage.json. No new candidates.json written.')
    write(destination,candidates)
    return {'records':len(candidates),'records_with_no_recognized_concepts':sum(not r['candidates'] for r in candidates)}
