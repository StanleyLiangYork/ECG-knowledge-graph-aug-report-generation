"""Retrieval diagnostics; relevance metrics require independent complete judgments."""
from pathlib import Path
import math
from common import *


def summarize(cfg,relevance_path=None):
    run=Path(cfg['run_dir']);root=run/'retrieval'/cfg['cohort'];queries=read(run/'queries'/cfg['cohort']/'manifest.json')
    judgments={}
    if relevance_path:
        for r in read(relevance_path):
            key=(r['query_id'],r['candidate_id'])
            if key in judgments:raise ValueError('Duplicate relevance judgment')
            gain=float(r['relevance'])
            if not math.isfinite(gain) or not 0<=gain<=1:raise ValueError('Relevance must be in [0,1]')
            judgments[key]=gain
    rows=[]
    for q in queries:
        r=read(root/(q['id']+'.json'));pool=r['candidates'];supported=[c for c in pool if c['reverse_ecg'] is not None]
        row=dict(query_id=q['id'],patient_id=q['patient_id'],anchors=len(r['anchors']),cxr_candidates=len(pool),
            reverse_supported_candidates=len(supported),unrestricted_anchor_returns=sum(c['unrestricted_returns_anchor'] for c in pool),
            graph_supported_candidates=sum(bool(c['graph_support']) for c in pool),conflicts=len(r['conflicts']),
            mean_reverse_ecg=sum(c['reverse_ecg'] for c in supported)/len(supported) if supported else None,
            mean_random_reverse_ecg=sum(c['random_reverse_ecg'] for c in pool if c['random_reverse_ecg'] is not None)/sum(c['random_reverse_ecg'] is not None for c in pool) if any(c['random_reverse_ecg'] is not None for c in pool) else None)
        rows.append(row)
        if judgments:
            for arm in ARMS[1:]:
                ids=r['selected'][arm];full=all((q['id'],c['id']) in judgments for c in pool)
                metric=dict(query_id=q['id'],patient_id=q['patient_id'],arm=arm,selected=len(ids),complete_candidate_pool_judgments=full,mrr_at2=None,ndcg_at2=None)
                if full and pool:
                    gains=[judgments[q['id'],rid] for rid in ids]
                    bypatient={}
                    for c in pool:bypatient[c['patient_id']]=max(bypatient.get(c['patient_id'],0.),judgments[q['id'],c['id']])
                    ideal=sorted(bypatient.values(),reverse=True)[:2]
                    first=next((i+1 for i,g in enumerate(gains) if g>=.5),None)
                    metric['mrr_at2']=1/first if first else 0.
                    dcg=lambda x:sum((2**g-1)/math.log2(i+2) for i,g in enumerate(x))
                    metric['ndcg_at2']=dcg(gains)/dcg(ideal) if dcg(ideal)>0 else None
                rows.append(dict(metric,analysis='independent_relevance'))
    folder=run/'evaluation'/cfg['cohort'];folder.mkdir(parents=True,exist_ok=True);csv_save(folder/'retrieval_diagnostics.csv',rows)
    result=dict(queries=len(queries),relevance_metrics='computed only on complete independently judged pools' if judgments else 'NOT_COMPUTED: independent relevance judgments absent',
                interpretation='Reverse/random similarities are consistency diagnostics, not clinical accuracy or causal evidence')
    atomic(folder/'retrieval_diagnostics.json',result);return result
