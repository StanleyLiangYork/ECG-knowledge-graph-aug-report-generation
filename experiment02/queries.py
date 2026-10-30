"""Freeze query selection before scoring; targets are intentionally never read."""
from pathlib import Path
import hashlib
import numpy as np
from common import *
from inputs import observations


def prepare_queries(cfg):
    assert_config(cfg)
    parent_setup(cfg);run=Path(cfg['run_dir']);cohort=cfg['cohort'];sources=unique(read(cfg['sensor_sources']),'id')
    out=Path(cfg['frozen_out']);rows=[];identity={}
    if cohort.startswith('internal_'):
        split='val' if cohort=='internal_val' else 'test'
        for r in sorted(read(out/'records.json'),key=lambda r:r['id']):
            if r['modality']!='ecg' or r['split']!=split:continue
            s=sources[r['id']]
            if uid(s['admission_id'])!=uid(r['admission_id']) or s['split']!=split:raise ValueError('Internal query source mismatch')
            rows.append({k:r[k] for k in ('id','patient_id','admission_id','split','modality','embedding','embedding_space')})
            rows[-1].update(study_id=uid(s['study_id']),source_index=int(s['source_index']),
                           findings=observations(r['candidates']),candidate_scope='existing sensor-only PULSE candidates')
        identity=dict(parent_records_sha256=sha(out/'records.json'),sensor_sha256=sha(cfg['sensor_sources']))
    else:
        root=Path(cfg['shaoxing_cohort']);manifest=read(root/'manifest.json');cp=read(root/'cohort_protocol.json');fp=read(root/'features/protocol.json')
        if cp['manifest_sha256']!=signature(manifest) or fp['manifest_sha256']!=signature(manifest) or fp['bank_records_sha256']!=sha(out/'records.json'):
            raise ValueError('External feature/cohort/bank mismatch')
        n=cfg['external_sample_size']
        if type(n) is not int or not 0<n<=len(manifest):raise ValueError('Invalid fixed external sample size')
        order=np.random.default_rng(cfg['seed']).permutation(len(manifest))[:n]
        for i in sorted(order,key=lambda i:manifest[i]['id']):
            r=manifest[i];feature=read(root/'features'/(r['id']+'.json'))
            if feature['id']!=r['id'] or feature['protocol_sha256']!=signature(fp):raise ValueError('External feature identity mismatch')
            rows.append({k:r[k] for k in ('id','patient_id','admission_id','study_id','modality','split','record','header_sha256','waveform_sha256','waveform_file')})
            rows[-1].update(embedding=feature['embedding'],embedding_space=feature['embedding_space'])
        identity=dict(cohort_protocol_sha256=sha(root/'cohort_protocol.json'),feature_protocol_sha256=signature(fp),
                      sampled_features={r['id']:sha(root/'features'/(r['id']+'.json')) for r in rows},selection='seeded label-independent record sample')
    if not rows:raise ValueError('No queries selected')
    folder=run/'queries'/cohort;freeze_stage(run,'queries_'+cohort,dict(identity=identity,rows=signature(rows)))
    immutable(folder/'manifest.json',rows)
    return dict(queries=len(rows),cohort=cohort,query_candidates_needed=cohort=='shaoxing')


def image_provider(cfg):
    """Always derive/check query images from source sensors; never accept stale cached images."""
    parent_setup(cfg);cohort=cfg['cohort'];run=Path(cfg['run_dir']);cache={};arrays={};admissions={}
    if cohort=='shaoxing':
        from shaoxing_external import image_provider as external_provider
        provide=external_provider(cfg['shaoxing_cohort'])
    else:
        import sensor_candidate_generation as sensor
        split='val' if cohort=='internal_val' else 'test';root=Path(cfg['symile_root'])/'data_npy'/split
        arrays[split]=np.load(root/f'ecg_{split}.npy',mmap_mode='r',allow_pickle=False)
        admissions[split]=np.load(root/f'hadm_id_{split}.npy',mmap_mode='r',allow_pickle=False).reshape(-1)
    def provider(row):
        rid=row['id']
        if rid not in cache:
            if cohort=='shaoxing':path,digest=provide(dict(query_id=rid))
            else:
                i=row['source_index'];split=row['split']
                if uid(admissions[split][i])!=uid(row['admission_id']):raise ValueError('Query sensor admission mismatch')
                arr=np.asarray(arrays[split][i])
                digest=signature(dict(shape=list(arr.shape),dtype=str(arr.dtype),bytes=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()))
                path=run/'queries'/cohort/'images'/(rid+'.png');sensor.materialize(arr,'ecg',path)
            cache[rid]=(str(path),digest)
        return cache[rid]
    return provider


def extract_queries(cfg,runtime,max_new=3,retry_errors=False,max_run_hours=30.,max_new_tokens=384):
    assert_config(cfg)
    parent_setup(cfg)
    import sensor_candidate_generation as sensor
    from candidate_preparation import extract
    if cfg['cohort']!='shaoxing':return {'needed':False,'reason':'Internal bank has existing sensor-only candidates'}
    if runtime.get('modality')!='ecg':raise ValueError('PULSE ECG runtime required')
    folder=Path(cfg['run_dir'])/'queries'/cfg['cohort'];rows=read(folder/'manifest.json');provider=image_provider(cfg)
    identity=dict(model=MODEL_IDS['pulse'],revision=runtime['revision'],prompt=sensor.PROMPTS['ecg'],max_new_tokens=max_new_tokens,
                  query_sha256=sha(folder/'manifest.json'),sensor_sha256=sha(sensor.__file__))
    freeze_stage(cfg['run_dir'],'query_extraction_'+cfg['cohort'],identity);sign=signature(identity)
    journal=folder/'candidates.jsonl';processed=0
    with lock(folder/'candidates.lock'),budget(max_run_hours,folder/'STOP_candidates') as stop:
        saved=latest(journal,repair=True)
        for row in rows:
            if stop() or (max_new and processed>=max_new):break
            image,tensor=provider(row);old=saved.get((row['id'],'sensor_only'))
            if old:
                if old['protocol_signature']!=sign or old['source_tensor_sha256']!=tensor:raise ValueError('Query extraction resume mismatch')
                if old['status']=='ok' or not retry_errors:continue
            value=dict(query_id=row['id'],arm='sensor_only',protocol_signature=sign,source_tensor_sha256=tensor,
                       sensor_sha256=sha(image),input_scope='sensor_only',model=MODEL_IDS['pulse'])
            try:
                text=sensor.infer(runtime,image,max_new_tokens)
                value.update(status='ok',text=text,response_sha256=signature(text),findings=observations(extract(text,'ecg')))
            except Exception as exc:value.update(status='error',error=str(exc))
            append(journal,value);saved[row['id'],'sensor_only']=value;processed+=1;print(row['id'],value['status'],flush=True)
            if value['status']=='error' and ('device-side assert' in value['error'] or 'illegal memory access' in value['error']):break
    return dict(total=len(rows),attempted=len(saved),ok=sum(r['status']=='ok' for r in saved.values()),processed_this_call=processed)


def load_queries(cfg):
    folder=Path(cfg['run_dir'])/'queries'/cfg['cohort'];rows=read(folder/'manifest.json')
    if cfg['cohort']!='shaoxing':return rows
    saved=latest(folder/'candidates.jsonl');identity=read(Path(cfg['run_dir'])/'stages'/('query_extraction_'+cfg['cohort']+'.json'))['inputs']
    from candidate_preparation import extract
    provider=image_provider(cfg)
    result=[]
    for row in rows:
        c=saved.get((row['id'],'sensor_only'))
        if not c:raise ValueError('Attempt query candidate extraction for every sampled ECG before retrieval: '+row['id'])
        image,tensor=provider(row)
        if c['protocol_signature']!=signature(identity) or c['source_tensor_sha256']!=tensor or c['sensor_sha256']!=sha(image):raise ValueError('Query candidate provenance mismatch')
        if c['status']=='ok' and (c['response_sha256']!=signature(c['text']) or c['findings']!=observations(extract(c['text'],'ecg'))):raise ValueError('Query candidate text/parse changed')
        result.append(dict(row,findings=c.get('findings',[]),candidate_status=c['status'],candidate_error=c.get('error')))
    return result
