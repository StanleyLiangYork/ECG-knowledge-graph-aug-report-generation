"""Matched, blinded multi-generator ECG review packets, separate from scoring."""
import json
import random
import shutil
import tempfile
from pathlib import Path
from report_study import read, sha, signature, atomic_json, ARMS
from report_generation import read_latest

RUBRIC = '''Review each ECG image and its accompanying generated report independently.
Assess the final report, not its writing style. Do not infer missing clinical history.
clinically_significant_error_count: number of incorrect assertions that could change clinical interpretation.
clinically_significant_omission_count: number of important ECG findings omitted.
overall_quality_1_to_5: 1 unusable, 2 major corrections, 3 moderate corrections,
4 minor corrections, 5 acceptable without corrections.
uninterpretable_image: true/false; if true leave clinical ratings null and explain.
comments: free text describing errors/omissions, uncertainty, and rationale.
reviewer_id: use your assigned reviewer code. Leave unassessed fields null.
Have ECG-qualified reviewers score independently before adjudication. Reviewers
must not receive the private linkage, reference labels, or generation coverage.
The same ECG may appear with multiple reports; presentation order is randomized.
Model identity can sometimes be inferred from style, so blinding is imperfect.
'''


def prepare_review(folders, targets_path, destination, sample_queries=30, seed=43):
    if type(sample_queries) is not int or sample_queries < 0:
        raise ValueError('sample_queries must be nonnegative; 0 means all matched queries')
    if len(folders)<2: raise ValueError('Select at least two generators')
    destination=Path(destination)
    # Never overwrite completed reviewer forms or existing blinded mappings.
    if destination.exists():
        raise FileExistsError('Review bundle already exists; retain it or choose a new versioned destination')
    maps={}; logs={}; eligible=[]; failures=[]; coverage={}; protocols={}; inputs={}
    for family, folder in folders.items():
        folder=Path(folder); plan=read(folder/'study_protocol.json'); cases=read(folder/'generation_cases.json')
        if signature(cases)!=plan['cases_sha256']: raise ValueError('Case hash mismatch: '+family)
        ecg=[c for c in cases if c['modality']=='ecg']
        cmap={(c['query_id'],c['arm']):c for c in ecg}
        queries={c['query_id'] for c in ecg}
        expected={(q,a) for q in queries for a in ARMS}
        if not queries or len(cmap)!=len(ecg) or set(cmap)!=expected:
            raise ValueError('Incomplete/duplicate case plan: '+family)
        protocol=read(folder/'ecg_generation_protocol.json')
        latest=read_latest(folder/'ecg_reports.jsonl')
        if set(latest)-expected: raise ValueError('Unexpected reports: '+family)
        good=set()
        for key,r in latest.items():
            if r['protocol_signature']!=signature(protocol) or r['prompt_sha256']!=signature(r['prompt']):
                raise ValueError('Protocol/prompt mismatch: '+family)
            if any(str(r.get(k))!=str(cmap[key][k]) for k in ('patient_id','admission_id','study_id','modality')):
                raise ValueError('Identity mismatch: '+family)
            if r['status']=='ok':
                if not r.get('text','').strip() or r['response_sha256']!=signature(r['text']):
                    raise ValueError('Response hash mismatch: '+family)
                if [x['record_id'] for x in r['evidence_budget_audit']]!=[x['record_id'] for x in cmap[key]['references']]:
                    raise ValueError('Evidence mismatch: '+family)
                good.add(key)
            else:
                failures.append(dict(model_family=family,query_id=key[0],arm=key[1],error=r.get('error'),status=r['status']))
        for key in expected-set(latest):
            failures.append(dict(model_family=family,query_id=key[0],arm=key[1],status='not_attempted',error=None))
        complete={q for q in queries if all((q,a) in good for a in ARMS)}
        coverage[family]=dict(expected=len(expected),ok=len(good),failed=len(latest)-len(good),
                              not_attempted=len(expected)-len(latest),complete_queries=len(complete))
        maps[family]=cmap;logs[family]=latest;eligible.append(complete);protocols[family]=protocol
        inputs[family]=dict(folder=str(folder),reports_sha256=sha(folder/'ecg_reports.jsonl'),protocol_sha256=signature(protocol))
    first=next(iter(maps.values()))
    if any(m!=first for m in maps.values()): raise ValueError('Generators have different cases/evidence')
    shared=set.intersection(*eligible)
    if not shared: raise ValueError('No jointly complete queries')
    if sample_queries>len(shared): raise ValueError('Requested sample exceeds matched queries')
    pool=sorted(shared); rng=random.Random(seed); rng.shuffle(pool)
    selected=pool[:sample_queries] if sample_queries else pool
    targets={r['id']:r for r in read(targets_path)}
    if set(selected)-set(targets): raise ValueError('Missing private reference targets')
    items=[(q,f,a) for q in selected for f in sorted(folders) for a in ARMS];rng.shuffle(items)
    # Verify tensors and image contents before publishing any review files.
    for q in selected:
        reports=[logs[f][q,a] for f in folders for a in ARMS]
        if len({r['source_tensor_sha256'] for r in reports})!=1: raise ValueError('Sensor tensor mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.review_build_',dir=destination.parent))
    try:
        public=staging/'PUBLIC'; private=staging/'PRIVATE'
        (public/'review_images').mkdir(parents=True);private.mkdir(mode=0o700)
        forms=[]; linkage=[]
        for i,(q,f,a) in enumerate(items):
            rid=f'review_{i+1:05d}'; r=logs[f][q,a]; c=maps[f][q,a]
            source=Path(folders[f])/'images'/(q+'.png')
            if not source.is_file() or sha(source)!=r['sensor_sha256']:
                raise ValueError('Missing/changed generated-report image: '+str(source))
            shutil.copyfile(source,public/'review_images'/(rid+'.png'))
            forms.append(dict(review_id=rid,image='review_images/'+rid+'.png',generated_report=r['text'],
                reviewer_id=None,uninterpretable_image=None,clinically_significant_error_count=None,
                clinically_significant_omission_count=None,overall_quality_1_to_5=None,comments=None))
            linkage.append(dict(review_id=rid,query_id=q,patient_id=c['patient_id'],admission_id=c['admission_id'],
                model_family=f,arm=a,reference_findings=targets[q]['findings'],source_report_sha256=r['response_sha256']))
        atomic_json(public/'review_forms.json',forms)
        (public/'REVIEW_INSTRUCTIONS.txt').write_text(RUBRIC)
        atomic_json(private/'linkage.json',linkage)
        atomic_json(private/'generation_failures.json',failures)
        manifest=dict(seed=seed,sampling='Uniform query sample from all-family, all-arm complete intersection',
            matched_queries=len(shared),selected_queries=selected,review_items=len(forms),coverage=coverage,
            inputs=inputs,targets_sha256=sha(targets_path),helper_sha256=sha(__file__),
            excluded_queries=sorted({q for q,a in first}-shared),
            limitations='Complete-case sampling excludes failed generations; report completion rates separately. '
                         'No clinical scores created. Reference labels are private and imperfect. '
                         'Reviewers may recognize images/styles. Query sampling is not patient stratification.')
        atomic_json(private/'review_manifest.json',manifest)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    return dict(matched_queries=len(shared),sampled_queries=len(selected),review_items=len(forms),
                public_folder=str(destination/'PUBLIC'),private_folder=str(destination/'PRIVATE'),coverage=coverage)
