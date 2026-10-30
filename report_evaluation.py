"""Masked finding metrics, true-text ROUGE-L and original-rank retrieval audit."""
import csv
import json
from collections import Counter
from pathlib import Path
from importlib.metadata import version
import numpy as np
from candidate_preparation import extract, ECG_RULES, CXR_RULES
from core import relevance
from report_study import read, sha, signature, atomic_json, resolve, reference_texts, validate_frozen, ARMS
from report_generation import read_latest, make_prompt


def ratio(a,b):return float(a/b) if b else None

def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def score_findings(text, modality, target):
    states={k:set() for k in (ECG_RULES if modality=='ecg' else CXR_RULES)}
    for mention in extract(text,modality):states[mention['concept']].add(mention['polarity'])
    # Binary assertion task: any positive mention counts as an assertion, even if
    # contradicted elsewhere. Unmentioned/uncertain-only concepts are not asserted.
    positive={k for k,v in states.items() if 'present' in v}
    known={k:v for k,v in target.items() if v in (0,1)}
    per_label={k:dict(tp=int(v==1 and k in positive),fp=int(v==0 and k in positive),
                      fn=int(v==1 and k not in positive),tn=int(v==0 and k not in positive)) for k,v in known.items()}
    counts={k:sum(c[k] for c in per_label.values()) for k in ('tp','fp','fn','tn')}
    tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
    return dict(**counts,known_labels=len(known),reference_positive_labels=tp+fn,
        f1=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
        hamming_accuracy=ratio(tp+tn,len(known)),
        reference_discordant_positive_rate=ratio(fp,tp+fp),omission_rate=ratio(fn,tp+fn),
        unknown_reference_positive_assertions=len(positive-set(known)),positive_assertions=len(positive),
        contradictory_concepts=sum('present' in s and 'absent' in s for s in states.values()),
        predicted_positive_concepts=sorted(positive),per_label=per_label)


def bootstrap_difference(rows, metric, left, right, repeats=2000):
    mapping={(r['query_id'],r['arm']):r for r in rows};patients={}
    for r in rows:
        if r['arm']!=left:continue
        other=mapping.get((r['query_id'],right))
        if other and r.get(metric) is not None and other.get(metric) is not None:
            patients.setdefault(r['patient_id'],[]).append(r[metric]-other[metric])
    values=np.asarray([np.mean(v) for v in patients.values()],dtype=float)
    if not len(values):return dict(mean=None,ci95=None,patients=0,paired_queries=0)
    rng=np.random.default_rng(43)
    samples=[float(rng.choice(values,len(values),replace=True).mean()) for _ in range(repeats)]
    return dict(mean=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),patients=len(values),
                paired_queries=sum(len(v) for v in patients.values()))


def retrieval_audit(out, threshold=.5):
    truth={r['id']:r for r in read(out/'targets.json')};per_query=[]
    for r in read(out/'rankings.json'):
        gains=[relevance(truth[r['query_id']],truth[rid]) for rid in r['ranking']]
        known=[g for g in gains if g is not None]
        first=next((i+1 for i,g in enumerate(gains) if g is not None and g>=threshold),None)
        possible=next((i+1 for i,g in enumerate(gains) if g is None or g>=threshold),None)
        per_query.append(dict(query_id=r['query_id'],patient_id=r['patient_id'],modality=r['modality'],arm=r['arm'],seed=r['seed'],
            observed_rr_at50=1/first if first else 0.,optimistic_rr_at50=1/possible if possible else 0.,
            known_judgments=len(known),top3_judged_fraction=ratio(sum(g is not None for g in gains[:3]),len(gains[:3])),
            all_known_relevance_equal=len(set(known))==1 if known else None,
            no_query_positive_labels=not any(v==1 for v in truth[r['query_id']]['findings'].values())))
    summary=[];averaged=[]
    for modality in ('ecg','cxr'):
        for arm in sorted({r['arm'] for r in per_query}):
            group=[r for r in per_query if r['modality']==modality and r['arm']==arm]
            eligible=[r for r in group if not r['no_query_positive_labels']]
            summary.append(dict(modality=modality,arm=arm,
                observed_mrr_at50=mean([r['observed_rr_at50'] for r in eligible]),
                optimistic_mrr_at50=mean([r['optimistic_rr_at50'] for r in eligible]),
                eligible_queries=len({r['query_id'] for r in eligible}),
                queries_with_no_judgments=len({r['query_id'] for r in eligible if not r['known_judgments']}),
                mean_top3_judged_fraction=mean([r['top3_judged_fraction'] for r in group]),
                equal_relevance_queries=len({r['query_id'] for r in group if r['all_known_relevance_equal']})))
            for qid in sorted({r['query_id'] for r in eligible}):
                q=[r for r in eligible if r['query_id']==qid]
                averaged.append(dict(query_id=qid,patient_id=q[0]['patient_id'],modality=modality,arm=arm,
                    observed_rr_at50=mean([r['observed_rr_at50'] for r in q])))
    cis={}
    for m in ('ecg','cxr'):
        for other in ('similarity','entity_mlp','shuffled_graph'):
            cis[m+'_graph_minus_'+other]=bootstrap_difference([r for r in averaged if r['modality']==m],'observed_rr_at50','learned_graph',other)
    return dict(threshold=threshold,definition='Observed MRR@50 at original ranks. Unknown items retain rank; no observed hit gives 0. Queries without positive target concepts excluded. Optimistic bound treats unknown relevance as a hit. Proxy, not clinical usefulness.',
                summary=summary,patient_confidence_intervals=cis,per_query=per_query)


def csv_export(path, rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(root, config, study, extra_texts=None, cxr_report_root=None, allow_partial=False):
    root,study=Path(root),Path(study);out=resolve(root,config['output_dir']);validate_frozen(root,out)
    cases=read(study/'generation_cases.json');plan=read(study/'study_protocol.json')
    if signature(cases)!=plan['cases_sha256'] or sha(out/'freeze.json')!=plan['frozen_sha256']:
        raise ValueError('Study inputs/freeze changed')
    retrieval=retrieval_audit(out);atomic_json(study/'retrieval_extension.json',retrieval)
    latest={};protocols={}
    for m in ('ecg','cxr'):
        path=study/f'{m}_reports.jsonl';latest.update(read_latest(path))
        if path.exists():
            protocols[m]=read(study/f'{m}_generation_protocol.json')
            if protocols[m]['study']!=plan:raise ValueError('Generation study protocol mismatch')
    expected={(c['query_id'],c['arm']) for c in cases}
    if set(latest)-expected:raise ValueError('Unexpected generated reports')
    complete={qid for qid,_ in expected if all(latest.get((qid,arm),{}).get('status')=='ok' for arm in ARMS)}
    coverage=dict(expected_reports=len(cases),successful_reports=sum(r['status']=='ok' for r in latest.values()),
        failed_reports=sum(r['status']!='ok' for r in latest.values()),not_attempted=len(cases)-len(latest),
        matched_three_arm_queries=len(complete),excluded_queries=len({qid for qid,_ in expected}-complete))
    atomic_json(study/'report_evaluation_coverage.json',coverage)
    if coverage['successful_reports']!=len(cases) and not allow_partial:
        raise ValueError('Reports incomplete; coverage saved. Finish generation or explicitly set ALLOW_PARTIAL=True for a labeled matched-complete-case analysis.')
    texts=reference_texts(root,config,cases,extra_texts,cxr_report_root)
    from rouge_score import rouge_scorer
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False)
    truth={r['id']:r for r in read(out/'targets.json')};scores=[]
    for c in cases:
        if c['query_id'] not in complete:continue
        r=latest[c['query_id'],c['arm']];protocol=protocols[c['modality']]
        if r['protocol_signature']!=signature(protocol) or signature(r['text'])!=r['response_sha256']:
            raise ValueError('Report provenance/hash mismatch')
        if any(str(r[k])!=str(c[k]) for k in ('patient_id','admission_id','modality','study_id')):
            raise ValueError('Report identity mismatch')
        if signature(r['prompt'])!=r['prompt_sha256']:raise ValueError('Prompt hash mismatch')
        if [x['record_id'] for x in r['evidence_budget_audit']]!=[x['record_id'] for x in c['references']]:
            raise ValueError('Evidence identity mismatch')
        score=dict(query_id=c['query_id'],patient_id=c['patient_id'],modality=c['modality'],arm=c['arm'],
                   **score_findings(r['text'],c['modality'],truth[c['query_id']]['findings']))
        reference=texts.get(c['query_id'])
        score['rougeL_f1']=scorer.score(reference['reference_text'],r['text'])['rougeL'].fmeasure if reference else None
        scores.append(score)
    summaries=[];intervals={}
    measures=['f1','precision','recall','hamming_accuracy','reference_discordant_positive_rate','omission_rate','rougeL_f1']
    labelrows=[]
    for modality in ('ecg','cxr'):
        for arm in ARMS:
            group=[r for r in scores if r['modality']==modality and r['arm']==arm]
            counts={k:sum(r[k] for r in group) for k in ('tp','fp','fn','tn')};tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
            labels={}
            for r in group:
                for label,cs in r['per_label'].items():
                    total=labels.setdefault(label,Counter());total.update(cs)
            label_f1=[]
            for label,cs in labels.items():
                f1=ratio(2*cs['tp'],2*cs['tp']+cs['fp']+cs['fn']);label_f1.append(f1)
                labelrows.append(dict(modality=modality,arm=arm,concept=label,**dict(cs),f1=f1))
            entry=dict(modality=modality,arm=arm,reports=len(group),known_label_decisions=tp+fp+fn+tn,
                micro_f1=ratio(2*tp,2*tp+fp+fn),macro_label_f1=mean(label_f1),
                micro_hamming_accuracy=ratio(tp+tn,tp+fp+fn+tn),
                unknown_reference_positive_assertions=sum(r['unknown_reference_positive_assertions'] for r in group),
                contradictory_concepts=sum(r['contradictory_concepts'] for r in group))
            for metric in measures:
                entry['mean_'+metric]=mean([r[metric] for r in group]);entry['n_'+metric]=sum(r[metric] is not None for r in group)
            summaries.append(entry)
        group=[r for r in scores if r['modality']==modality]
        for other in ('sensor_only','ordinary_rag'):
            for metric in measures:
                intervals[f'{modality}_graph_minus_{other}_{metric}']=bootstrap_difference(group,metric,'graph_rag',other)
    output=dict(exploratory=True,coverage=coverage,summary=summaries,patient_paired_differences=intervals,
        per_report=scores,reference_text_coverage={m:sum(c['modality']==m and c['query_id'] in texts for c in cases if c['arm']=='sensor_only') for m in ('ecg','cxr')},
        metric_sources=dict(evaluation_sha256=sha(__file__),parser_sha256=sha(Path(__file__).with_name('candidate_preparation.py')),rouge_score=version('rouge-score')),
        limitations='Machine diagnoses/radiology labels are noisy references; regex extraction is unvalidated. Unknown reference labels excluded. Hamming measures positive assertion presence, not explicit negative wording. Null metrics have zero denominator. No causal or clinical accuracy claim.')
    atomic_json(study/'report_metrics.json',output);atomic_json(study/'reference_text_audit.json',texts)
    csv_export(study/'report_summary.csv',summaries);csv_export(study/'report_label_metrics.csv',labelrows)
    return dict(coverage=coverage,summary=summaries,reference_text_coverage=output['reference_text_coverage'])


def review_packets(root, config, study):
    study=Path(study);out=resolve(root,config['output_dir']);cases=read(study/'generation_cases.json')
    reports={}
    for m in ('ecg','cxr'):reports.update(read_latest(study/f'{m}_reports.jsonl'))
    targets={r['id']:r['findings'] for r in read(out/'targets.json')}
    rows=[c for c in cases if reports.get((c['query_id'],c['arm']),{}).get('status')=='ok']
    rng=np.random.default_rng(43);rng.shuffle(rows);key=[];public=[]
    images=study/'review_images';images.mkdir(exist_ok=True)
    import shutil
    for i,c in enumerate(rows):
        rid=f'review_{i:05d}';image=study/'images'/f"{c['query_id']}.png"
        if not image.exists():raise FileNotFoundError('Missing rendered review image: '+str(image))
        shutil.copyfile(image,images/f'{rid}.png')
        key.append(dict(review_id=rid,query_id=c['query_id'],arm=c['arm'],patient_id=c['patient_id']))
        public.append(dict(review_id=rid,modality=c['modality'],image='review_images/'+rid+'.png',
            generated_report=reports[c['query_id'],c['arm']]['text'],reference_findings=targets[c['query_id']],
            clinically_significant_errors=None,omissions=None,reviewer_id=None))
    atomic_json(study/'report_review_key_PRIVATE.json',key);atomic_json(study/'report_review_blinded.json',public)
    return dict(review_items=len(public),instruction='Give reviewers only report_review_blinded.json and review_images; retain the PRIVATE key.')
