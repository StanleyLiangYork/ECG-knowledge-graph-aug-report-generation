"""Evaluate Qwen and specialist reports on identical query sets."""
from pathlib import Path
from qwen_report_extension import qwen_run_name
import shutil
import numpy as np
from report_study import read, sha, atomic_json, ARMS
from report_evaluation import evaluate, bootstrap_difference, mean, ratio, csv_export

METRICS=('f1','hamming_accuracy','reference_discordant_positive_rate','omission_rate','rougeL_f1')


def compare_models(root,config,base_study,extra_texts=None,cxr_report_root=None,allow_partial=False,enable_thinking=True):
    base=Path(base_study);qwen=base/qwen_run_name(enable_thinking)
    if read(base/'generation_cases.json')!=read(qwen/'generation_cases.json') or read(base/'study_protocol.json')!=read(qwen/'study_protocol.json'):
        raise ValueError('Model studies must use the identical case/evidence plan')
    for folder in (base,qwen):
        evaluate(root,config,folder,extra_texts=extra_texts,cxr_report_root=cxr_report_root,allow_partial=allow_partial)
    from report_generation import read_latest
    specialist_logs={};qwen_logs={}
    for modality in ('ecg','cxr'):
        specialist_logs.update(read_latest(base/f'{modality}_reports.jsonl'))
        qwen_logs.update(read_latest(qwen/f'{modality}_reports.jsonl'))
    files={'specialist':base/'report_metrics.json','qwen35_4b':qwen/'report_metrics.json'}
    results={family:read(path) for family,path in files.items()}
    available={family:{r['query_id'] for r in result['per_report']} for family,result in results.items()}
    shared=available['specialist'] & available['qwen35_4b']
    evidence_differences=0;sensor_file_differences=0
    for key,left in specialist_logs.items():
        if key[0] not in shared:continue
        right=qwen_logs[key]
        if left['source_tensor_sha256']!=right['source_tensor_sha256']:
            raise ValueError('Model families did not use identical source sensor tensors')
        if left['sensor_sha256']!=right['sensor_sha256']:sensor_file_differences+=1
        a=[(r['record_id'],r['included_observations']) for r in left['evidence_budget_audit']]
        b=[(r['record_id'],r['included_observations']) for r in right['evidence_budget_audit']]
        if a!=b:evidence_differences+=1
    rows=[]
    for family,result in results.items():
        for r in result['per_report']:
            if r['query_id'] in shared:rows.append(dict(r,model_family=family))
    summary=[];intervals={}
    for modality in ('ecg','cxr'):
        for family in files:
            for arm in ARMS:
                group=[r for r in rows if r['modality']==modality and r['model_family']==family and r['arm']==arm]
                counts={k:sum(r[k] for r in group) for k in ('tp','fp','fn','tn')}
                tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
                entry=dict(modality=modality,model_family=family,arm=arm,matched_queries=len(group),
                           micro_f1=ratio(2*tp,2*tp+fp+fn),micro_hamming_accuracy=ratio(tp+tn,tp+fp+fn+tn))
                for metric in METRICS:
                    entry['mean_'+metric]=mean([r[metric] for r in group])
                    entry['n_'+metric]=sum(r[metric] is not None for r in group)
                summary.append(entry)
        group=[r for r in rows if r['modality']==modality]
        for arm in ARMS:
            paired=[dict(r,arm=r['model_family']) for r in group if r['arm']==arm]
            for metric in METRICS:
                intervals[f'{modality}_{arm}_qwen_minus_specialist_{metric}']=bootstrap_difference(paired,metric,'qwen35_4b','specialist')
        # Difference of within-model graph gains: (Qwen graph - ordinary RAG)
        # minus (specialist graph - ordinary RAG), at the paired-patient level.
        lookup={(r['model_family'],r['query_id'],r['arm']):r for r in group}
        gain_rows=[]
        for r in group:
            if r['arm']!='graph_rag':continue
            other=lookup[r['model_family'],r['query_id'],'ordinary_rag']
            gain=dict(query_id=r['query_id'],patient_id=r['patient_id'],arm=r['model_family'])
            for metric in METRICS:
                gain[metric]=r[metric]-other[metric] if r[metric] is not None and other[metric] is not None else None
            gain_rows.append(gain)
        for metric in METRICS:
            intervals[f'{modality}_graph_gain_qwen_minus_specialist_{metric}']=bootstrap_difference(gain_rows,metric,'qwen35_4b','specialist')
    output=dict(exploratory=True,matched_queries=len(shared),
        reports_with_different_included_evidence_counts=evidence_differences,
        reports_with_different_rendered_file_hashes=sensor_file_differences,
        available_queries={k:len(v) for k,v in available.items()},
        excluded_from_joint_comparison={k:len(v-shared) for k,v in available.items()},
        input_metrics_sha256={family:sha(path) for family,path in files.items()},comparison_code_sha256=sha(__file__),
        summary=summary,patient_paired_differences=intervals,
        note='Same queries and reference IDs; tokenizer and image-processing differences remain. Shared candidate evidence comes from PULSE/MedGemma. Retrieval MRR is shared, not a generator metric.')
    atomic_json(base/'generator_comparison.json',output);csv_export(base/'generator_comparison.csv',summary)
    return output


def combined_review(base_study,enable_thinking=True):
    from report_generation import read_latest
    base=Path(base_study);cases=read(base/'generation_cases.json');rows=[]
    for family,folder in [('specialist',base),('qwen35_4b',base/qwen_run_name(enable_thinking))]:
        latest={}
        for m in ('ecg','cxr'):latest.update(read_latest(folder/f'{m}_reports.jsonl'))
        for c in cases:
            r=latest.get((c['query_id'],c['arm']))
            if r and r['status']=='ok':rows.append((family,folder,c,r))
    np.random.default_rng(43).shuffle(rows)
    image_dir=base/'combined_review_images';image_dir.mkdir(exist_ok=True)
    public=[];key=[]
    for i,(family,folder,c,r) in enumerate(rows):
        rid=f'combined_{i:05d}'
        shutil.copyfile(folder/'images'/f"{c['query_id']}.png",image_dir/f'{rid}.png')
        public.append(dict(review_id=rid,modality=c['modality'],image='combined_review_images/'+rid+'.png',
            generated_report=r['text'],clinically_significant_errors=None,omissions=None,reviewer_id=None))
        key.append(dict(review_id=rid,model_family=family,query_id=c['query_id'],arm=c['arm'],patient_id=c['patient_id']))
    atomic_json(base/'combined_review_blinded.json',public);atomic_json(base/'combined_review_key_PRIVATE.json',key)
    return dict(review_items=len(public),instruction='Share only combined_review_blinded.json and combined_review_images; keep PRIVATE key separate.')
