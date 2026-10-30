"""Selection-only k sensitivity; no changes to frozen root Python helpers."""
import copy
import json
import shutil
from pathlib import Path
from common import read,sha,signature,immutable,atomic,freeze_stage,assert_config,csv_save

FIELDS={'forward_pool':'forward_cxr','bidirectional_pool':'bidirectional_score',
        'bidirectional_graph':'graph_score','bidirectional_shuffled_graph':'shuffled_score'}


def select(pool,field,k):
    chosen=[];seen=set()
    for c in sorted(pool,key=lambda c:(-c[field],c['id'])):
        if c['patient_id'] in seen:continue
        chosen.append(c['id']);seen.add(c['patient_id'])
        if len(chosen)==k:break
    return chosen


def prepare_sweep(cfg,values=(1,2,3,4,5),sweep_version='v1'):
    """Call after BR02 completes; reuse exactly the same anchors, pools and scores."""
    from queries import load_queries
    from packets import prepare_packets
    assert_config(cfg)
    if not values or len(set(values))!=len(values) or any(type(k) is not int or not 1<=k<=5 for k in values):
        raise ValueError('Unique integer k values from 1 through 5 required')
    if not sweep_version or Path(sweep_version).name!=sweep_version:raise ValueError('Use a simple sweep version name')
    # One conservative per-source finding cap for the ENTIRE sweep; reserve room for graph context.
    available=cfg['context_tokens']-cfg['relation_tokens']-cfg['k_anchors']*cfg['lab_tokens_per_source']-64
    shared_finding_cap=min(cfg['finding_tokens_per_source'],available//(cfg['k_anchors']+max(values)))
    if shared_finding_cap<32:raise ValueError('Context budget too small for the requested sweep with graph/lab reserves')
    source=Path(cfg['run_dir']).resolve();cohort=cfg['cohort'];queries=load_queries(cfg)
    source_stage=read(source/'stages'/('retrieval_'+cohort+'.json'))
    retrieval_sign=signature(source_stage);source_paths={q['id']:source/'retrieval'/cohort/(q['id']+'.json') for q in queries}
    for q,path in source_paths.items():
        if not path.is_file() or read(path).get('protocol_signature')!=retrieval_sign:
            raise ValueError('Complete valid BR02 retrieval first: '+q)
    source_protocol=read(source/'protocol.json');base=source/'k_sensitivity'/sweep_version/cohort
    parent=dict(parent_protocol_sha256=sha(source/'protocol.json'),retrieval_stage_sha256=signature(source_stage),
                source_retrieval_sha256={q:sha(p) for q,p in source_paths.items()},
                query_manifest_sha256=sha(source/'queries'/cohort/'manifest.json'),
                helper_sha256=sha(__file__),k_values=list(values),shared_finding_tokens_per_source=shared_finding_cap,
                policy='Selection-only k sweep; same queries, anchors, initial pools, scores and total context-token budget')
    immutable(base/'sweep_protocol.json',parent)
    results={}
    for k in values:
        child=base/f'k_{k}';child_cfg=copy.deepcopy(cfg);child_cfg.update(run_dir=str(child),final_cxr=k,finding_tokens_per_source=shared_finding_cap)
        protocol=copy.deepcopy(source_protocol);protocol['config']={name:value for name,value in child_cfg.items() if name!='cohort'}
        protocol['k_sensitivity']=dict(parent,selected_k=k)
        immutable(child/'protocol.json',protocol)
        # Read-only shared bank data: do not run BR00/BR01 bank writes in children.
        bank_link=child/'banks'
        if not bank_link.exists():bank_link.symlink_to(source/'banks',target_is_directory=True)
        elif bank_link.resolve()!=(source/'banks').resolve():raise ValueError('Unexpected child banks directory')
        queries_folder=child/'queries'/cohort;queries_folder.mkdir(parents=True,exist_ok=True)
        immutable(queries_folder/'manifest.json',read(source/'queries'/cohort/'manifest.json'))
        # Preserve candidate identity/provenance for external queries without rerunning PULSE.
        if cohort=='shaoxing':
            original=source/'queries'/cohort/'candidates.jsonl';target=queries_folder/'candidates.jsonl'
            if target.exists():
                if sha(target)!=sha(original):raise ValueError('Query candidates changed during sweep')
            else:shutil.copyfile(original,target)
        for name in ('queries_'+cohort,'query_extraction_'+cohort,'indexes','banks_ready'):
            stage=source/'stages'/(name+'.json')
            if stage.is_file():freeze_stage(child,name,read(stage)['inputs'])
        child_identity=dict(source_retrieval=parent,selected_k=k,selection_fields=FIELDS)
        child_sign=freeze_stage(child,'retrieval_'+cohort,child_identity)
        folder=child/'retrieval'/cohort;folder.mkdir(parents=True,exist_ok=True)
        short=[]
        for q,path in source_paths.items():
            r=read(path)
            # Detect changes even while a large sweep is being prepared.
            if sha(path)!=parent['source_retrieval_sha256'][q]:raise ValueError('Source retrieval changed')
            r['selected']={'sensor_only':[],**{arm:select(r['candidates'],field,k) for arm,field in FIELDS.items()}}
            r['fallbacks']=dict(r['fallbacks'],cxr_pool_short=len({c['patient_id'] for c in r['candidates']})<k)
            if r['fallbacks']['cxr_pool_short']:short.append(q)
            r['protocol_signature']=child_sign;r['requested_final_cxr']=k
            immutable(folder/(q+'.json'),r)
        atomic(folder/'coverage.json',dict(total=len(queries),ok=len(queries),complete=True))
        packet_result=prepare_packets(child_cfg)
        atomic(child/'audits/k_selection.json',dict(requested_k=k,queries=len(queries),queries_with_fewer_distinct_patients=short,
              context_tokens=cfg['context_tokens'],anchors_unchanged=True,searches_recomputed=False))
        # Write absolute config paths so the snapshot works irrespective of notebook cwd.
        atomic(child/'config.json',child_cfg)
        results[k]=dict(config=child_cfg,config_path=str(child/'config.json'),packets=packet_result)
    return results


def generate_sweep(configs,runtime,max_new_per_k=5,retry_errors=False,max_run_hours=30.,max_new_tokens=384):
    """One model load; one total time budget across all k values."""
    from common import budget
    from generation import generate
    import time
    if not configs:raise ValueError('No sweep configurations')
    started=time.monotonic();results={}
    base=Path(next(iter(configs.values()))['run_dir']).parent
    with budget(max_run_hours,base/'STOP_generation_sweep') as stop:
        for k,cfg in sorted(configs.items()):
            if stop():break
            remaining=max_run_hours-(time.monotonic()-started)/3600
            if remaining<=0:break
            # Outer signals/STOP also stop the current k at its next record boundary.
            import generation
            original_budget=generation.budget
            from contextlib import contextmanager
            @contextmanager
            def combined_budget(hours,path):
                with original_budget(hours,path) as inner:
                    yield lambda:stop() or inner()
            from unittest.mock import patch
            with patch.object(generation,'budget',combined_budget):
                results[k]=generate(cfg,runtime,max_new=max_new_per_k,retry_errors=retry_errors,
                                    max_run_hours=remaining,max_new_tokens=max_new_tokens)
            if results[k].get('stop_reason')!='completed_or_limit':break
    return results


def summarize_sweep(configs,folders_by_k,destinations,output):
    """Aggregate completed per-k evaluations; no automatic best-k selection."""
    from evaluation import holm
    from common import latest
    if set(configs)!=set(folders_by_k) or set(configs)!=set(destinations):
        raise ValueError('Finish evaluation for every k before aggregation')
    model_sets=[set(folders_by_k[k]) for k in configs]
    if any(v!=model_sets[0] for v in model_sets):raise ValueError('Use the same model set for every k')
    summary=[];primary=[];exposure=[];per_k_rows={};complete_sets=[]
    for k,cfg in sorted(configs.items()):
        root=Path(destinations[k]);result=read(root/'results.json')
        if not result['complete']:raise ValueError('Finish evaluation for k='+str(k))
        for row in result['summary']:summary.append(dict(row,final_cxr=k,matched_queries=result['matched_queries']))
        for r in result['comparisons']:
            if r['contrast']=='bidirectional_graph_minus_bidirectional_pool' and r['metric']=='f1' and r['subset']=='all_arm_complete':
                primary.append(dict(r,final_cxr=k))
        import csv
        with (root/'per_report.csv').open(newline='') as f:
            parsed=[]
            for row in csv.DictReader(f):
                for name in ('f1','tp','fp','fn','tn'):
                    row[name]=float(row[name]) if row.get(name) else None
                parsed.append(row)
            per_k_rows[k]=parsed
        for model,folder in folders_by_k[k].items():
            reports=latest(Path(folder)/'reports.jsonl');groups={}
            packet_manifest=read(Path(cfg['run_dir'])/'packets'/cfg['cohort']/'manifest.json')
            complete_sets.append({q for q in packet_manifest['query_ids'] if all(reports.get((q,a),{}).get('status')=='ok' for a in packet_manifest['arms'])})
            for (q,arm),r in reports.items():
                if r['status']!='ok':continue
                packet=read(Path(cfg['run_dir'])/'packets'/cfg['cohort']/(q+'.json'))
                case=next(c for c in packet['cases'] if c['arm']==arm)
                selected=[x for x in case['references'] if x['source_type']=='other_patient_mimic_cxr']
                actual=sum(('(other_patient_mimic_cxr; source='+x['record_id']+')') in r['prompt'] for x in selected)
                group=groups.setdefault(arm,dict(context=[],selected=[],prompted=[]))
                group['context'].append(r['evidence_audit']['context_tokens']);group['selected'].append(len(selected));group['prompted'].append(actual)
            for arm,g in groups.items():
                mean=lambda xs:sum(xs)/len(xs)
                exposure.append(dict(final_cxr=k,model=model,arm=arm,successful_reports=len(g['context']),
                    mean_context_tokens=mean(g['context']),mean_selected_cxr=mean(g['selected']),mean_prompted_cxr=mean(g['prompted'])))
    adjusted=holm({f"{r['model_family']}_k{r['final_cxr']}":r['p_value'] for r in primary})
    for r in primary:r['p_holm_across_models_and_k']=adjusted[f"{r['model_family']}_k{r['final_cxr']}"]
    common=set.intersection(*complete_sets) if complete_sets else set()
    common_summary=[];common_effects=[]
    from evaluation import paired
    from collections import defaultdict
    for k,data in sorted(per_k_rows.items()):
        for model in folders_by_k[k]:
            selected=[r for r in data if r['model_family']==model and r['query_id'] in common]
            for arm in sorted({r['arm'] for r in selected}):
                group=[r for r in selected if r['arm']==arm];patients=defaultdict(list)
                for r in group:
                    if r['f1'] is not None:patients[r['patient_id']].append(r['f1'])
                tp=sum(r['tp'] for r in group);fp=sum(r['fp'] for r in group);fn=sum(r['fn'] for r in group)
                means=[sum(v)/len(v) for v in patients.values()]
                common_summary.append(dict(final_cxr=k,model=model,arm=arm,queries=len(group),patients=len(patients),
                    patient_mean_f1=sum(means)/len(means) if means else None,
                    micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
            effect=paired(selected,'f1','bidirectional_graph','bidirectional_pool',2000,configs[k]['seed'])
            common_effects.append(dict(final_cxr=k,model=model,**effect))
    common_adjusted=holm({f"{r['model']}_k{r['final_cxr']}":r['p_value'] for r in common_effects})
    for r in common_effects:r['p_holm_across_models_and_k']=common_adjusted[f"{r['model']}_k{r['final_cxr']}"]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    csv_save(output/'summary_by_k.csv',summary);csv_save(output/'context_exposure_by_k.csv',exposure)
    csv_save(output/'primary_effects_by_k.csv',[dict({name:value for name,value in r.items() if name!='ci95'},ci95_lower=(r.get('ci95') or [None,None])[0],ci95_upper=(r.get('ci95') or [None,None])[1]) for r in primary])
    csv_save(output/'common_query_summary_by_k.csv',common_summary)
    csv_save(output/'common_query_primary_effects.csv',[dict({name:value for name,value in r.items() if name!='ci95'},ci95_lower=(r.get('ci95') or [None,None])[0],ci95_upper=(r.get('ci95') or [None,None])[1]) for r in common_effects])
    atomic(output/'k_sensitivity.json',dict(summary=summary,primary_effects=primary,context_exposure=exposure,
        common_queries=sorted(common),common_query_summary=common_summary,common_query_effects=common_effects,
        interpretation='Per-k analyses may have different complete-case cohorts; no automatic k selection. Pick k using validation only.'))
    return dict(output=str(output),k_values=sorted(configs),automatic_k_selection=False)
