"""Five-arm source-attributed evidence; target labels never enter this module."""
from pathlib import Path
from collections import defaultdict
from common import *

BASE_PROMPT=('Describe the query 12-lead ECG image in a concise clinical-style English ECG report. '
 'State supported findings, pertinent negatives and uncertainty. The ECG is amplitude-normalized; '
 'do not infer absolute voltage or amplitude-based diagnoses. Do not invent measurements, patient history, '
 'differential diagnoses or management instructions. All optional ECG, CXR and laboratory references belong '
 'to OTHER PATIENTS. Their observations and graph associations are analogies, never findings established '
 'for the query patient. A graph edge is training-admission co-occurrence, not causality.\n')


def prepare_packets(cfg):
    assert_config(cfg)
    from queries import load_queries
    run=Path(cfg['run_dir']);queries=load_queries(cfg);folder=run/'packets'/cfg['cohort'];folder.mkdir(parents=True,exist_ok=True)
    source=run/'retrieval'/cfg['cohort'];identity={q['id']:sha(source/(q['id']+'.json')) for q in queries}
    sign=freeze_stage(run,'packets_'+cfg['cohort'],dict(retrieval=identity))
    expected=signature(read(run/'stages'/('retrieval_'+cfg['cohort']+'.json')))
    arms=list(ARMS if cfg['include_shuffled_graph'] else ARMS[:-1]);counts=defaultdict(int)
    for q in queries:
        r=read(source/(q['id']+'.json'))
        if r['protocol_signature']!=expected:raise ValueError('Retrieval protocol mismatch')
        lookup={x['id']:x for x in r['candidates']};cases=[]
        for arm in arms:
            refs=[];relationships=[]
            if arm!='sensor_only':
                for a in r['anchors']:
                    refs.append(dict(record_id=a['id'],source_type='symile_reference_admission',
                        findings=[dict(x,modality=m) for m in ('ecg','cxr') for x in a[m+'_findings']],labs=a['labs']))
                for rid in r['selected'][arm]:
                    c=lookup[rid];refs.append(dict(record_id=rid,source_type='other_patient_mimic_cxr',
                                                  findings=[dict(x,modality='cxr') for x in c['findings']],labs=[]))
                    if arm in ('bidirectional_graph','bidirectional_shuffled_graph'):
                        name='graph' if arm=='bidirectional_graph' else 'shuffled'
                        relationships.extend(dict(e,reference_id=rid) for e in c[name+'_support'])
            case=dict(query_id=q['id'],patient_id=q['patient_id'],admission_id=q['admission_id'],study_id=q['study_id'],
                      modality='ecg',arm=arm,references=refs,relationships=relationships,
                      conflicts=r['conflicts'] if arm!='sensor_only' else [],fallbacks=r['fallbacks'])
            cases.append(case);counts[arm]+=1
        immutable(folder/(q['id']+'.json'),dict(query_id=q['id'],protocol_signature=sign,cases=cases))
    immutable(folder/'manifest.json',dict(query_ids=[q['id'] for q in queries],arms=arms,protocol_signature=sign,
        packets={q['id']:sha(folder/(q['id']+'.json')) for q in queries}))
    return dict(queries=len(queries),arms=dict(counts),folder=str(folder))


def format_prompt(case,tokenizer,cfg):
    """Reserve bounded source/lab/relation slots; retain included/omitted counts."""
    total_cap=cfg['context_tokens'];used=0;lines=[];audit=[]
    def add(text,slot,source,limit):
        nonlocal used
        n=len(tokenizer.encode(text,add_special_tokens=False))
        accepted=n<=limit and used+n<=total_cap
        audit.append(dict(slot=slot,source=source,tokens=n,included=accepted))
        if accepted:lines.append(text);used+=n
        return accepted
    for i,ref in enumerate(case['references'],1):
        identity=f'Reference {i} ({ref["source_type"]}; source={ref["record_id"]})'
        observations=[]
        for f in ref['findings']:
            text=f'{f["modality"]}:{f["concept"]}={f["polarity"]}'
            proposal=identity+': '+'; '.join(observations+[text])
            if len(tokenizer.encode(proposal,add_special_tokens=False))<=cfg['finding_tokens_per_source']:observations.append(text)
            else:audit.append(dict(slot='omitted_finding',source=ref['record_id'],included=False,tokens=len(tokenizer.encode(text,add_special_tokens=False))))
        add(identity+': '+('; '.join(observations) or 'No extracted findings.'),'findings',ref['record_id'],cfg['finding_tokens_per_source'])
        # Missing/unshown labs are never described as normal or observed.
        labs=[x for x in ref['labs'] if x['observed']]
        labs.sort(key=lambda x:(-abs(x['training_ecdf_percentile']-.5),x['itemid']))
        selected=[]
        for x in labs:
            item=f'{x["name"]} [item {x["itemid"]}]: training-distribution percentile {x["training_ecdf_percentile"]:.3f}'
            text=f'Reference {i} labs (not clinical ranges): '+'; '.join(selected+[item])
            if len(tokenizer.encode(text,add_special_tokens=False))<=cfg['lab_tokens_per_source']:selected.append(item)
            else:audit.append(dict(slot='omitted_lab',source=ref['record_id'],included=False,tokens=len(tokenizer.encode(item,add_special_tokens=False))))
        if selected:add(f'Reference {i} labs (not clinical ranges): '+'; '.join(selected),'labs',ref['record_id'],cfg['lab_tokens_per_source'])
    relation=[]
    for e in case['relationships']:
        text=f'query ECG candidate {e["ecg_concept"]} co-occurred with reference CXR {e["cxr_concept"]} in {e["patient_count"]} training patients / {e["admission_count"]} admissions; reference={e["reference_id"]}'
        if len(tokenizer.encode('Graph associations: '+'; '.join(relation+[text]),add_special_tokens=False))<=cfg['relation_tokens']:relation.append(text)
        else:audit.append(dict(slot='omitted_relation',source=e['reference_id'],included=False,tokens=len(tokenizer.encode(text,add_special_tokens=False))))
    if case['arm'] in ('bidirectional_graph','bidirectional_shuffled_graph'):
        add('Graph associations: '+('; '.join(relation) or 'No supported query-to-reference associations.'),'relations','graph',cfg['relation_tokens'])
    # Conflict information is source-filtered so an unselected candidate cannot leak into the prompt.
    selected_ids={r['record_id'] for r in case['references']}
    conflicts=[f'{c["source"]}: conflicting {c["modality"]} {c["concept"]} assertions' for c in case['conflicts'] if c['source'] in selected_ids]
    if conflicts:add('Reference uncertainty: '+'; '.join(conflicts),'conflicts','references',cfg['relation_tokens'])
    context='\n'.join(lines) if lines else 'Other-patient references: none.'
    # Newlines can change tokenization; enforce the cap on the actual composed context as well.
    while lines and len(tokenizer.encode(context,add_special_tokens=False))>total_cap:
        removed=lines.pop();audit.append(dict(slot='composition_trim',source='context',included=False,tokens=len(tokenizer.encode(removed,add_special_tokens=False))))
        context='\n'.join(lines) if lines else 'Other-patient references: none.'
    return BASE_PROMPT+context+'\nFinal ECG report:\n',dict(context_tokens=len(tokenizer.encode(context,add_special_tokens=False)),cap=total_cap,slots=audit)
