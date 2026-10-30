"""Independent-target five-arm metrics, matched analyses and patient bootstrap."""
from pathlib import Path
from collections import defaultdict,Counter
import math
import numpy as np
from common import *

CONTRASTS=(('bidirectional_graph','bidirectional_pool'),('bidirectional_pool','forward_pool'),
           ('forward_pool','sensor_only'),('bidirectional_graph','bidirectional_shuffled_graph'))
METRICS=('f1','recall','precision','hamming_accuracy','omission_rate','reference_discordant_positive_rate','rougeL_f1')


def avg(values):
    x=[v for v in values if v is not None];return float(np.mean(x)) if x else None


def paired(rows,metric,left,right,repeats=2000,seed=43):
    mapping={(r['query_id'],r['arm']):r for r in rows};groups=defaultdict(list)
    for r in rows:
        other=mapping.get((r['query_id'],right))
        if r['arm']==left and other and r.get(metric) is not None and other.get(metric) is not None:
            groups[r['patient_id']].append(r[metric]-other[metric])
    values=np.array([np.mean(groups[p]) for p in sorted(groups)])
    if not len(values):return dict(mean=None,ci95=None,patients=0,paired_queries=0,p_value=None)
    rng=np.random.default_rng(seed);boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(repeats)])
    null=np.array([(values*rng.choice([-1,1],len(values))).mean() for _ in range(repeats)])
    p=float((1+np.sum(np.abs(null)>=abs(values.mean())))/(repeats+1))
    return dict(mean=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),patients=len(values),
        paired_queries=sum(map(len,groups.values())),p_value=p,p_method='paired patient sign-flip Monte Carlo; assumes sign symmetry under null')


def holm(values):
    valid=sorted((v,k) for k,v in values.items() if v is not None);result={k:None for k in values};last=0.
    for i,(v,k) in enumerate(valid):
        last=max(last,min(1.,v*(len(valid)-i)));result[k]=last
    return result


def micro_ci(rows,left,right,repeats=2000,seed=43):
    mapping={(r['query_id'],r['arm']):r for r in rows};groups=defaultdict(lambda:np.zeros((2,3),dtype=float))
    for r in rows:
        o=mapping.get((r['query_id'],right))
        if r['arm']==left and o:
            groups[r['patient_id']]+=np.array([[r[k] for k in ('tp','fp','fn')],[o[k] for k in ('tp','fp','fn')]])
    counts=np.array([groups[p] for p in sorted(groups)])
    def difference(x):
        n=2*x[:,0];d=n+x[:,1]+x[:,2]
        return float(n[0]/d[0]-n[1]/d[1]) if (d>0).all() else None
    if not len(counts):return dict(mean=None,ci95=None,patients=0)
    observed=difference(counts.sum(0));rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        v=difference(counts[rng.integers(0,len(counts),len(counts))].sum(0))
        if v is not None:values.append(v)
    return dict(mean=observed,ci95=np.quantile(values,[.025,.975]).tolist() if values else None,patients=len(counts))


def load_results(folder,packet_manifest):
    folder=Path(folder);p=read(folder/'protocol.json');saved=latest(folder/'reports.jsonl')
    if p['packets_manifest_sha256']!=signature_file(packet_manifest):raise ValueError('Generator used different packets')
    expected={(q,a) for q in packet_manifest['query_ids'] for a in packet_manifest['arms']}
    if set(saved)-expected:raise ValueError('Unexpected report keys')
    for key,r in saved.items():
        if r['protocol_signature']!=signature(p) or r['prompt_sha256']!=signature(r['prompt']):raise ValueError('Report protocol/prompt hash mismatch')
        if r['packet_sha256']!=packet_manifest['packets'][r['query_id']]:raise ValueError('Report packet mismatch')
        if r['status']=='ok' and r['response_sha256']!=signature(r['text']):raise ValueError('Report text changed')
    return saved


def signature_file(manifest):
    # Atomic JSON formatting is shared, so hash exact original manifest bytes, passed via hidden field by caller.
    return manifest['_file_sha256']


def evaluate(cfg,folders,targets_path,destination,reference_texts=None,allow_partial=False,
             mapping_review_note='',max_run_hours=30.,repeats=2000):
    assert_config(cfg)
    parent_setup(cfg)
    from report_evaluation import score_findings
    import report_evaluation,candidate_preparation
    run=Path(cfg['run_dir']);packet_root=run/'packets'/cfg['cohort'];manifest=read(packet_root/'manifest.json');manifest['_file_sha256']=sha(packet_root/'manifest.json')
    if cfg['cohort']=='shaoxing' and (not mapping_review_note.strip() or mapping_review_note.lower().startswith('draft')):
        raise ValueError('Document completed external concept/annotation mapping review before scoring')
    if cfg['cohort']=='shaoxing':
        cohort_protocol=read(Path(cfg['shaoxing_cohort'])/'cohort_protocol.json')
        if signature(read(targets_path))!=cohort_protocol['targets_sha256']:
            raise ValueError('External targets differ from frozen cohort; version the reviewed mapping/cohort explicitly')
    targets=unique(read(targets_path),'id');reference_texts=reference_texts or {};destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if set(manifest['query_ids'])-set(targets):raise ValueError('Missing independent query targets')
    datasets={};coverage={};identities={}
    for model,folder in folders.items():
        data=load_results(folder,manifest);datasets[model]=data;byarm={}
        for a in manifest['arms']:
            group=[data.get((q,a)) for q in manifest['query_ids']]
            byarm[a]=dict(expected=len(group),ok=sum(r is not None and r['status']=='ok' for r in group),
                         failed=sum(r is not None and r['status']!='ok' for r in group),not_attempted=sum(r is None for r in group),
                         failure_reasons=dict(Counter(r.get('error','Unknown') for r in group if r is not None and r['status']!='ok')))
        coverage[model]=byarm;identities[model]=dict(protocol=sha(Path(folder)/'protocol.json'),reports=sha(Path(folder)/'reports.jsonl'))
    shared={q for q in manifest['query_ids'] if all(data.get((q,a),{}).get('status')=='ok' for data in datasets.values() for a in manifest['arms'])}
    coverage_report=dict(models=coverage,matched_all_models_all_arms=len(shared),total_queries=len(manifest['query_ids']),
                         excluded_queries=sorted(set(manifest['query_ids'])-shared),allow_partial=allow_partial)
    atomic(destination/'coverage.json',coverage_report)
    if not allow_partial and any(x['failed'] or x['not_attempted'] for model in coverage.values() for x in model.values()):
        raise ValueError('Reports incomplete; inspect coverage.json; explicitly allow matched partial analysis if intended')
    if not shared:raise ValueError('No jointly complete queries')
    # Same sensor for every successful model/arm of a query.
    for q in manifest['query_ids']:
        sources={(r['source_tensor_sha256'],r['sensor_sha256']) for data in datasets.values() for (rid,a),r in data.items() if rid==q and r['status']=='ok'}
        if len(sources)>1:raise ValueError('Generators/arms used different source ECGs')
    identity=dict(generators=identities,targets=sha(targets_path),reference_texts=signature(reference_texts),
                  scorer=sha(report_evaluation.__file__),parser=sha(candidate_preparation.__file__),repeats=repeats,
                  mapping_review_note=mapping_review_note,code=code_hashes(),allow_partial=allow_partial)
    immutable(destination/'protocol.json',identity);fingerprint=signature(identity);cache=destination/'checkpoints';cache.mkdir(exist_ok=True)
    rows=[];scores={}
    with lock(destination/'writer.lock'),budget(max_run_hours,destination/'STOP') as stop:
        def guard():
            if stop():
                atomic(destination/'status.json',dict(complete=False,stop_reason=stop(),resume='Rerun same evaluation'))
                return False
            return True
        for model,data in datasets.items():
            for (q,arm),r in sorted(data.items()):
                if not guard():return read(destination/'status.json')
                if r['status']!='ok':continue
                key=signature([fingerprint,model,q,arm]);path=cache/(key+'.json')
                if path.exists():score=read(path)
                else:
                    s=score_findings(r['text'],'ecg',targets[q]['findings'])
                    # Primary report F1 requires evaluable positive labels; zero-positive cases remain available for other metrics.
                    if not s['reference_positive_labels']:s['f1']=None
                    rouge=None;reference=reference_texts.get(q)
                    if reference:
                        if isinstance(reference,dict):reference=reference['reference_text']
                        from rouge_score import rouge_scorer
                        rouge=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True).score(reference,r['text'])['rougeL'].fmeasure
                    score=dict(s,query_id=q,patient_id=r['patient_id'],model_family=model,arm=arm,rougeL_f1=rouge,
                               context_tokens=r['evidence_audit']['context_tokens'],output_tokens=r.get('output_tokens'))
                    atomic(path,score)
                rows.append(score);scores[model,q,arm]=score
        summary=[]
        for model in datasets:
            for arm in manifest['arms']:
                group=[scores[model,q,arm] for q in sorted(shared)];counts={k:sum(r[k] for r in group) for k in ('tp','fp','fn','tn')}
                tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
                entry=dict(model_family=model,arm=arm,reports=len(group),known_label_decisions=tp+fp+fn+tn,
                    micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                    micro_hamming_accuracy=(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None,
                    contradictory_concepts=sum(r['contradictory_concepts'] for r in group),
                    unknown_reference_positive_assertions=sum(r['unknown_reference_positive_assertions'] for r in group))
                for m in METRICS:
                    entry['mean_'+m]=avg([r[m] for r in group]);entry['n_'+m]=sum(r[m] is not None for r in group)
                # Patient-averaged F1, the prespecified primary estimand.
                patients=defaultdict(list)
                for r in group:
                    if r['f1'] is not None:patients[r['patient_id']].append(r['f1'])
                entry['patient_mean_f1']=avg([avg(v) for v in patients.values()]);entry['f1_patients']=len(patients)
                summary.append(entry)
        comparisons=[]
        for model,data in datasets.items():
            for left,right in CONTRASTS:
                if left not in manifest['arms'] or right not in manifest['arms']:continue
                for subset in ('all_arm_complete','pairwise_complete'):
                    selected=shared if subset=='all_arm_complete' else {q for q in manifest['query_ids'] if all(data.get((q,a),{}).get('status')=='ok' for a in (left,right))}
                    group=[scores[model,q,a] for q in sorted(selected) for a in (left,right)]
                    for m in METRICS:
                        if not guard():return read(destination/'status.json')
                        name=signature([fingerprint,model,left,right,subset,m]);path=cache/('ci_'+name+'.json')
                        if not path.exists():atomic(path,paired(group,m,left,right,repeats,cfg['seed']))
                        comparisons.append(dict(model_family=model,contrast=left+'_minus_'+right,subset=subset,metric=m,**read(path)))
                    path=cache/('micro_'+signature([fingerprint,model,left,right,subset])+'.json')
                    if not path.exists():atomic(path,micro_ci(group,left,right,repeats,cfg['seed']))
                    comparisons.append(dict(model_family=model,contrast=left+'_minus_'+right,subset=subset,metric='micro_f1',**read(path)))
            # Completion-sensitive F1: only eligible positive-label queries, including failed/not-attempted reports as zero.
            for arm in manifest['arms']:
                group=[]
                for q in manifest['query_ids']:
                    if not any(v==1 for v in targets[q]['findings'].values()):continue
                    packet=read(packet_root/(q+'.json'));case=next(c for c in packet['cases'] if c['arm']==arm)
                    s=scores.get((model,q,arm))
                    group.append(dict(query_id=q,patient_id=case['patient_id'],arm=arm,f1=s['f1'] if s else 0.))
                patients=defaultdict(list)
                for r in group:patients[r['patient_id']].append(r['f1'])
                summary.append(dict(model_family=model,arm=arm,analysis='completion_sensitive_f1',
                    eligible_queries=len(group),patient_mean_f1=avg([avg(v) for v in patients.values()]),f1_patients=len(patients)))
        primary=[r for r in comparisons if r['contrast']=='bidirectional_graph_minus_bidirectional_pool' and r['metric']=='f1' and r['subset']=='all_arm_complete']
        adjusted=holm({r['model_family']:r['p_value'] for r in primary})
        for r in primary:r['p_holm']=adjusted[r['model_family']]
        # Model interactions on the common all-model/all-arm cohort.
        models=list(datasets)
        for i,leftmodel in enumerate(models):
            for rightmodel in models[i+1:]:
                group=[]
                for model in (leftmodel,rightmodel):
                    for q in sorted(shared):
                        a=scores[model,q,'bidirectional_graph'];b=scores[model,q,'bidirectional_pool']
                        if a['f1'] is not None and b['f1'] is not None:
                            group.append(dict(query_id=q,patient_id=a['patient_id'],arm=model,f1=a['f1']-b['f1']))
                comparisons.append(dict(model_family='interaction',contrast=leftmodel+'_minus_'+rightmodel,subset='all_arm_complete',metric='graph_gain_f1',
                                        **paired(group,'f1',leftmodel,rightmodel,repeats,cfg['seed'])))
        scalar=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows]
        csv_save(destination/'per_report.csv',scalar);csv_save(destination/'summary.csv',summary)
        csv_save(destination/'paired_comparisons.csv',[dict({k:v for k,v in r.items() if k!='ci95'},ci95_lower=(r.get('ci95') or [None,None])[0],ci95_upper=(r.get('ci95') or [None,None])[1]) for r in comparisons])
        result=dict(complete=True,matched_queries=len(shared),summary=summary,comparisons=comparisons,
            limitations='Sparse labels and unvalidated regex extraction; unknowns masked; sign-flip p-values require symmetry; clinical review pending.',
            reference_text_queries=len(set(reference_texts)&shared),coverage=coverage_report)
        atomic(destination/'results.json',result);atomic(destination/'status.json',dict(complete=True));return result
