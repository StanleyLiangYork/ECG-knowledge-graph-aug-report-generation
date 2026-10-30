"""Matched ECG evaluation across model families; cached scores and patient CIs."""
import csv
import fcntl
from pathlib import Path
from itertools import combinations
from report_study import read, sha, signature, atomic_json, ARMS
from report_generation import read_latest, make_prompt
from report_evaluation import score_findings, bootstrap_difference, mean, ratio
from ecg_external_generation import immutable, budget

METRICS=('f1','precision','recall','hamming_accuracy','reference_discordant_positive_rate','omission_rate','rougeL_f1')


def csv_save(path, rows):
    path=Path(path); temp=path.with_suffix('.tmp')
    if not rows: return
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def evaluate(folders, targets_path, destination, reference_texts=None,
             allow_partial=False, max_run_hours=30.):
    """Requires ECG coverage for EVERY selected family; never reports 0-ECG success.

    Match all arms and all selected families, then compute known-label metrics.
    This evaluator does not load a model, encoder, raw waveform or retrain anything.
    """
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    atomic_json(destination/'evaluation_status.json',dict(complete=False,state='validating_inputs'))
    reference_texts=reference_texts or {}
    if len(folders)<2: raise ValueError('Select at least two generators')
    targets=read(targets_path); truth={t['id']:t for t in targets}
    if len(truth)!=len(targets): raise ValueError('Duplicate target IDs')
    cohort_protocol=Path(targets_path).parent/'cohort_protocol.json'
    if cohort_protocol.exists() and read(cohort_protocol)['targets_sha256']!=signature(targets):
        raise ValueError('External labels changed after cohort freeze')
    datasets={}; protocols={}; identity={}; case_maps={}; complete_sets={}; coverage={}
    for family,folder in folders.items():
        folder=Path(folder); plan=read(folder/'study_protocol.json'); all_cases=read(folder/'generation_cases.json')
        if signature(all_cases)!=plan['cases_sha256']: raise ValueError('Changed cases')
        cases=[c for c in all_cases if c['modality']=='ecg']; cmap={(c['query_id'],c['arm']):c for c in cases}
        if not cases or len(cmap)!=len(cases): raise ValueError('Missing/duplicate ECG cases for '+family)
        queries={c['query_id'] for c in cases}; expected={(q,a) for q in queries for a in ARMS}
        if set(cmap)!=expected: raise ValueError('Expected exactly three arms')
        latest=read_latest(folder/'ecg_reports.jsonl'); pp=folder/'ecg_generation_protocol.json'
        protocol=read(pp) if pp.exists() else None
        if set(latest)-expected: raise ValueError('Unexpected ECG reports')
        if latest and protocol is None: raise ValueError('Reports without protocol')
        valid=set()
        for key,r in latest.items():
            c=cmap[key]
            if any(str(r.get(k))!=str(c[k]) for k in ('patient_id','admission_id','study_id','modality')):
                raise ValueError('Report identity mismatch')
            if r['protocol_signature']!=signature(protocol) or r['prompt_sha256']!=signature(r['prompt']):
                raise ValueError('Report protocol/prompt mismatch')
            if r['status']=='ok':
                if r['response_sha256']!=signature(r['text']): raise ValueError('Response hash mismatch')
                if [x['record_id'] for x in r['evidence_budget_audit']]!=[x['record_id'] for x in c['references']]:
                    raise ValueError('Evidence mismatch')
                valid.add(key)
        complete={q for q in queries if all((q,a) in valid for a in ARMS)}
        coverage[family]=dict(expected=len(expected),ok=len(valid),failed=len(latest)-len(valid),
            not_attempted=len(expected)-len(latest),complete_queries=len(complete))
        complete_sets[family]=complete; datasets[family]=latest; case_maps[family]=cmap; protocols[family]=protocol
        identity[family]=dict(cases_sha256=signature(cases),reports_sha256=sha(folder/'ecg_reports.jsonl') if latest else None,
                              generation_protocol_sha256=signature(protocol))
    first=next(iter(case_maps.values()))
    if any(c!=first for c in case_maps.values()): raise ValueError('Families must use identical ECG cases/evidence')
    shared=set.intersection(*complete_sets.values())
    atomic_json(destination/'coverage.json',dict(families=coverage,matched_queries=len(shared),
        excluded_by_family={f:len(s-shared) for f,s in complete_sets.items()},allow_partial=allow_partial))
    if not shared: raise ValueError('No jointly complete ECG queries. Finish ECG generation in selected folders.')
    if not allow_partial and any(c['failed'] or c['not_attempted'] for c in coverage.values()):
        raise ValueError('Incomplete generation; inspect coverage or explicitly allow matched partial analysis')
    if shared-set(truth): raise ValueError('Missing targets')
    evidence_differences=0
    for q in shared:
        for arm in ARMS:
            reports=[d[q,arm] for d in datasets.values()]
            if len({r['source_tensor_sha256'] for r in reports})!=1: raise ValueError('Different source tensors')
            evidence_differences+=int(len({signature(r['evidence_budget_audit']) for r in reports})>1)
    fingerprint=signature(dict(inputs=identity,targets=sha(targets_path),texts=reference_texts,
        shared=sorted(shared),code=sha(__file__),scorer=sha(Path(__file__).with_name('report_evaluation.py')),
        parser=sha(Path(__file__).with_name('candidate_preparation.py'))))
    cache=destination/'checkpoints'/fingerprint; cache.mkdir(parents=True,exist_ok=True)
    def stopped(reason):
        status=dict(complete=False,state='interrupted',stop_reason=reason,checkpoint=str(cache),fingerprint=fingerprint)
        atomic_json(destination/'evaluation_status.json',status)
        return status
    rows=[]; cis={}; summary=[]
    with (destination/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with budget(max_run_hours,destination/'STOP_evaluation') as stop:
            for family,data in datasets.items():
                for q in sorted(shared):
                    for arm in ARMS:
                        if stop(): return stopped(stop())
                        key=signature([family,q,arm]); path=cache/(key+'.json')
                        if path.exists(): row=read(path)
                        else:
                            r=data[q,arm]; c=first[q,arm]
                            score=score_findings(r['text'],'ecg',truth[q]['findings'])
                            ref=reference_texts.get(q); rouge=None
                            if ref:
                                from rouge_score.rouge_scorer import RougeScorer
                                rouge=RougeScorer(['rougeL'],use_stemmer=False).score(ref['reference_text'],r['text'])['rougeL'].fmeasure
                            row=dict(query_id=q,patient_id=c['patient_id'],modality='ecg',arm=arm,model_family=family,
                                rougeL_f1=rouge,**score)
                            atomic_json(path,row)
                        rows.append(row)
            for family in folders:
                for arm in ARMS:
                    group=[r for r in rows if r['model_family']==family and r['arm']==arm]
                    counts={k:sum(r[k] for r in group) for k in ('tp','fp','fn','tn')}; tp,fp,fn,tn=(counts[k] for k in ('tp','fp','fn','tn'))
                    labels={}
                    for r in group:
                        for label,cs in r['per_label'].items():
                            total=labels.setdefault(label,dict(tp=0,fp=0,fn=0,tn=0))
                            for k,v in cs.items(): total[k]+=v
                    macro=mean([ratio(2*v['tp'],2*v['tp']+v['fp']+v['fn']) for v in labels.values()])
                    entry=dict(model_family=family,arm=arm,reports=len(group),known_label_decisions=tp+fp+fn+tn,
                        micro_f1=ratio(2*tp,2*tp+fp+fn),macro_label_f1=macro,micro_hamming_accuracy=ratio(tp+tn,tp+fp+fn+tn),
                        unknown_reference_positive_assertions=sum(r['unknown_reference_positive_assertions'] for r in group),
                        contradictory_concepts=sum(r['contradictory_concepts'] for r in group))
                    for m in METRICS: entry['mean_'+m]=mean([r[m] for r in group]); entry['n_'+m]=sum(r[m] is not None for r in group)
                    summary.append(entry)
            jobs=[]
            for family in folders:
                group=[r for r in rows if r['model_family']==family]
                for other in ('sensor_only','ordinary_rag'):
                    for m in METRICS: jobs.append((family+'_graph_minus_'+other+'_'+m,group,m,'graph_rag',other))
            for left,right in combinations(folders,2):
                group=[r for r in rows if r['model_family'] in (left,right)]
                for arm in ARMS:
                    paired=[dict(r,arm=r['model_family']) for r in group if r['arm']==arm]
                    for m in METRICS: jobs.append((arm+'_'+left+'_minus_'+right+'_'+m,paired,m,left,right))
                lookup={(r['model_family'],r['query_id'],r['arm']):r for r in group}; gains=[]
                for r in group:
                    if r['arm']!='graph_rag': continue
                    other=lookup[r['model_family'],r['query_id'],'ordinary_rag']
                    gain=dict(query_id=r['query_id'],patient_id=r['patient_id'],arm=r['model_family'])
                    for m in METRICS: gain[m]=r[m]-other[m] if r[m] is not None and other[m] is not None else None
                    gains.append(gain)
                for m in METRICS: jobs.append(('graph_gain_'+left+'_minus_'+right+'_'+m,gains,m,left,right))
            for name,group,m,left,right in jobs:
                if stop(): return stopped(stop())
                path=cache/('ci_'+signature(name)+'.json')
                if not path.exists(): atomic_json(path,bootstrap_difference(group,m,left,right))
                cis[name]=read(path)
    result=dict(complete=True,exploratory=True,fingerprint=fingerprint,matched_queries=len(shared),coverage=coverage,
        summary=summary,patient_paired_differences=cis,reports_with_different_evidence_audits=evidence_differences,
        reference_text_queries=len(shared & set(reference_texts)),
        limitations='Unknown labels masked; regex extraction unvalidated; no multiplicity adjustment; code-label agreement is not clinical report validation.')
    atomic_json(destination/'comparison.json',result); csv_save(destination/'summary.csv',summary)
    csv_save(destination/'paired_comparisons.csv',[dict(comparison=k,mean=v['mean'],ci95_lower=(v['ci95'] or [None,None])[0],
        ci95_upper=(v['ci95'] or [None,None])[1],patients=v['patients'],paired_queries=v['paired_queries']) for k,v in cis.items()])
    csv_save(destination/'per_report.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rows])
    atomic_json(destination/'evaluation_status.json',dict(complete=True,state='complete',fingerprint=fingerprint))
    return result
