"""Four generator adapters and durable five-arm generation, independent of parent ARMS."""
import copy
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
from collections import Counter
import numpy as np
from common import *
from packets import format_prompt


def model_snapshot(model_key,path=None,download=False,token_file='/path/to/private/hf_token',cache_dir='/path/to/data/hf_cache',pin_dir=None):
    from ecg_external_generation import local_snapshot
    if path is not None or not download:return local_snapshot(MODEL_IDS[model_key],path)
    if pin_dir is None:raise ValueError('Set pin_dir for reproducible downloads')
    from huggingface_hub import HfApi,snapshot_download
    pin=Path(pin_dir)/(model_key+'.json');pin.parent.mkdir(parents=True,exist_ok=True)
    with Path(token_file).open() as f:token=f.readline().strip()
    if not token:raise ValueError('Empty first token-file line')
    try:
        old=read(pin) if pin.exists() else None
        if old and old['model_id']!=MODEL_IDS[model_key]:raise ValueError('Pinned download repository differs')
        revision=old['revision'] if old else HfApi(token=token).model_info(MODEL_IDS[model_key]).sha
        immutable(pin,dict(model_id=MODEL_IDS[model_key],revision=revision))
        return Path(snapshot_download(MODEL_IDS[model_key],revision=revision,cache_dir=cache_dir,token=token,
            allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','*.bin','*.py'],ignore_patterns=['original/*'],max_workers=4))
    finally:del token


def load_generator(cfg,key,path,enable_thinking=False,load_in_4bit=False,pulse_repo=None):
    parent_setup(cfg)
    if key not in MODEL_IDS:raise ValueError('Unknown generator')
    if enable_thinking and key!='qwen38':raise ValueError('Thinking switch applies only to Qwen')
    if key=='pulse':
        if load_in_4bit:raise ValueError('PULSE precision is controlled by its working loader; explicit 4bit switch unsupported')
        from sensor_candidate_generation import load_runtime
        runtime=load_runtime('ecg',path,pulse_repo=pulse_repo);runtime.update(family='pulse',model_id=MODEL_IDS[key])
    elif key=='llama_ecg':
        from ecg_external_generation import load_llama
        runtime=load_llama(path,load_in_4bit)
    elif key=='llama_base':
        from general_ecg_generators import load
        runtime=load('llama32_base',path,load_in_4bit)
    else:
        from qwen38_generation_v2 import load
        runtime=load(path,enable_thinking,load_in_4bit)
    runtime.update(experiment_model_key=key,enable_thinking=enable_thinking,load_in_4bit=load_in_4bit)
    return runtime


def folder_for(cfg,key,thinking,tokens):
    return Path(cfg['run_dir'])/'generations'/cfg['cohort']/key/f'{"thinking" if thinking else "nonthinking"}_t{tokens}'


def generate(cfg,runtime,max_new=5,retry_errors=False,max_run_hours=30.,max_new_tokens=384):
    assert_config(cfg)
    parent_setup(cfg)
    from ecg_external_generation import configure_decoding,infer as parent_infer
    from queries import load_queries,image_provider
    import report_generation,sensor_candidate_generation,ecg_external_generation
    if type(max_new_tokens) is not int or max_new_tokens<1 or max_new<0:raise ValueError('Invalid report/token limit')
    run=Path(cfg['run_dir']);cohort=cfg['cohort'];packet_root=run/'packets'/cohort;manifest=read(packet_root/'manifest.json')
    key=runtime['experiment_model_key'];thinking=runtime['enable_thinking'];folder=folder_for(cfg,key,thinking,max_new_tokens);folder.mkdir(parents=True,exist_ok=True)
    libraries={}
    for name in ('torch','transformers','numpy','Pillow','matplotlib'):
        try:libraries[name]=version(name)
        except PackageNotFoundError:libraries[name]='unavailable'
    original=runtime['model'].generation_config
    decoding=configure_decoding(original,dict(repetition_penalty=1.1,no_repeat_ngram_size=12),runtime['family'])
    import candidate_preparation,report_study,shaoxing_external,qwen_report_extension
    helpers={Path(m.__file__).name:sha(m.__file__) for m in
             (sensor_candidate_generation,ecg_external_generation,candidate_preparation,report_study,shaoxing_external,qwen_report_extension)}
    if key in ('llama_base','qwen38'):
        import general_ecg_generators
        helpers['general_ecg_generators.py']=sha(general_ecg_generators.__file__)
    if key=='qwen38':
        import qwen38_generation_v2
        helpers['qwen38_generation_v2.py']=sha(qwen38_generation_v2.__file__)
        runtime['diagnostic_dir']=folder/'diagnostics'
    protocol=dict(experiment=signature(read(run/'protocol.json')),model_key=key,model=runtime['model_id'],revision=runtime['revision'],
        enable_thinking=thinking,load_in_4bit=runtime.get('load_in_4bit',False),max_new_tokens=max_new_tokens,
        packets_manifest_sha256=sha(packet_root/'manifest.json'),libraries=libraries,helpers=helpers,
        decoding=decoding.to_dict(),code=code_hashes())
    freeze_stage(run,'generation_'+cohort+'_'+key+'_'+folder.name,protocol)
    immutable(folder/'protocol.json',protocol);sign=signature(protocol);queries=load_queries(cfg);provider=image_provider(cfg)
    log=folder/'reports.jsonl';processed=0;stop_reason='completed_or_limit';expected={(q,a) for q in manifest['query_ids'] for a in manifest['arms']}
    with lock(folder/'writer.lock'):
        saved=latest(log,repair=True)
        if set(saved)-expected:raise ValueError('Unexpected journal keys')
        runtime['model'].generation_config=decoding
        try:
            with budget(max_run_hours,folder/'STOP') as stop:
                for query in queries:
                    if stop() or (max_new and processed>=max_new):break
                    packet_path=packet_root/(query['id']+'.json')
                    if sha(packet_path)!=manifest['packets'][query['id']]:raise ValueError('Packet changed')
                    image,tensor=provider(query);image_hash=sha(image)
                    for case in read(packet_path)['cases']:
                        if stop() or (max_new and processed>=max_new):break
                        prompt,audit=format_prompt(case,runtime['tokenizer'],cfg);old=saved.get((query['id'],case['arm']))
                        if old:
                            if any(old.get(k)!=v for k,v in dict(protocol_signature=sign,prompt_sha256=signature(prompt),source_tensor_sha256=tensor,sensor_sha256=image_hash).items()):raise ValueError('Generation resume provenance mismatch')
                            if old['status']=='ok' or not retry_errors:continue
                        row={k:case[k] for k in ('query_id','patient_id','admission_id','study_id','arm')}
                        row.update(model_key=key,protocol_signature=sign,source_tensor_sha256=tensor,sensor_sha256=image_hash,
                                   prompt_sha256=signature(prompt),prompt=prompt,evidence_audit=audit,packet_sha256=sha(packet_path))
                        try:
                            if key=='qwen38':text=qwen38_generation_v2.infer(runtime,image,prompt,max_new_tokens)
                            else:text=parent_infer(runtime,image,prompt,max_new_tokens)
                            ids=runtime['tokenizer'].encode(text,add_special_tokens=False)
                            grams=Counter(tuple(ids[i:i+8]) for i in range(max(0,len(ids)-7)))
                            if max(grams.values(),default=0)>8:raise ValueError('Repetition QC failed')
                            if not text or sensor_candidate_generation.re_search_instruction(text):raise ValueError('Empty or instruction-echo report')
                            row.update(status='ok',text=text,response_sha256=signature(text),output_tokens=len(ids))
                        except Exception as exc:row.update(status='error',error=str(exc),error_type=type(exc).__name__)
                        append(log,row);saved[query['id'],case['arm']]=row;processed+=1
                        print(query['id'],case['arm'],row['status'],flush=True)
                        if row['status']=='error' and ('device-side assert' in row['error'] or 'illegal memory access' in row['error']):
                            stop_reason='cuda_context_error_restart_kernel';break
                    if stop_reason.startswith('cuda'):break
                stop_reason=stop() or stop_reason
        except KeyboardInterrupt:stop_reason='keyboard_interrupt'
        finally:runtime['model'].generation_config=original
    counts=dict(total=len(expected),ok=sum(r['status']=='ok' for r in saved.values()),failed=sum(r['status']!='ok' for r in saved.values()),
                not_attempted=len(expected)-len(saved),processed_this_call=processed,stop_reason=stop_reason,folder=str(folder))
    atomic(folder/'coverage.json',counts);return counts
