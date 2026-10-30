"""Additive Qwen3.5-4B adapter. Existing specialist report helpers stay unchanged."""
import copy
import fcntl
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from importlib.metadata import version
import numpy as np
import sensor_candidate_generation as sensor
import report_generation as common
from report_generation import make_prompt, read_latest, BASE_PROMPT
from report_study import read, sha, signature, atomic_json

MODEL_ID='Qwen/Qwen3.5-4B'
def qwen_run_name(enable_thinking=True):
    if type(enable_thinking) is not bool:
        raise TypeError('enable_thinking must be True or False, not a string')
    return 'qwen35_4b_thinking_v2' if enable_thinking else 'qwen35_4b_nonthinking_v2'



def prepare_qwen(base_study, enable_thinking=True):
    base=Path(base_study);dest=base/qwen_run_name(enable_thinking)
    protocol=read(base/'study_protocol.json');cases=read(base/'generation_cases.json')
    if signature(cases)!=protocol['cases_sha256']:raise ValueError('Base generation cases changed')
    for name in ('study_protocol.json','generation_cases.json'):
        value=read(base/name)
        if (dest/name).exists() and read(dest/name)!=value:raise ValueError('Qwen cohort/settings changed; use a new study')
        atomic_json(dest/name,value)
    return dest


def load_qwen(model_path,modality,enable_thinking=True):
    qwen_run_name(enable_thinking)  # Validate the mode before loading GPU weights.
    import torch
    import transformers
    if modality not in ('ecg','cxr'):raise ValueError('Choose ecg or cxr')
    if not torch.cuda.is_available():raise RuntimeError('Use a GPU allocation and your working Experiment 4 Qwen environment')
    path=Path(model_path).expanduser().resolve()
    if not (path/'config.json').is_file():raise FileNotFoundError('Set MODEL_PATH to the existing Qwen3.5-4B snapshot')
    cfg=read(path/'config.json');text_cfg=cfg.get('text_config',{})
    if cfg.get('model_type')!='qwen3_5' or text_cfg.get('hidden_size')!=2560 or text_cfg.get('num_hidden_layers')!=32:
        raise ValueError('Checkpoint configuration does not match the selected Qwen3.5-4B architecture')
    loader=getattr(transformers,'AutoModelForMultimodalLM',None)
    if loader is None:loader=getattr(transformers,'AutoModelForImageTextToText',None)
    if loader is None:raise ImportError('Use the working Experiment 4 Qwen3.5 Transformers environment; do not modify PULSE')
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor=transformers.AutoProcessor.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    model=loader.from_pretrained(str(path),dtype=dtype,device_map='auto',local_files_only=True,trust_remote_code=False).eval()
    if any(p.is_meta for p in model.parameters()):raise RuntimeError('Qwen has unloaded meta parameters')
    if any(b.is_meta for b in model.buffers()):raise RuntimeError('Qwen has unloaded meta buffers')
    revision=path.name
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        revision=signature({str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()})
    context=getattr(getattr(model.config,'text_config',model.config),'max_position_embeddings',None)
    if not isinstance(context,int):raise ValueError('Cannot determine Qwen context limit')
    return dict(model=model,processor=processor,tokenizer=processor.tokenizer,revision=revision,
                dtype=dtype,context=context,modality=modality,model_id=MODEL_ID,
                enable_thinking=enable_thinking)


def infer_qwen(runtime,image_path,prompt,max_new_tokens):
    import torch
    from PIL import Image
    model=runtime['model'];processor=runtime['processor'];tokenizer=runtime['tokenizer']
    with Image.open(image_path) as im:image=im.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',enable_thinking=runtime.get('enable_thinking',True))
    length=inputs['input_ids'].shape[-1]
    if length+max_new_tokens>runtime['context']:raise ValueError('Qwen context budget exceeded; no silent input truncation')
    device=model.get_input_embeddings().weight.device
    inputs={k:v.to(device=device,dtype=runtime['dtype']) if v.is_floating_point() else v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    generated=output[0,length:]
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else list(eos or [])
    if len(generated)>=max_new_tokens and (not len(generated) or int(generated[-1]) not in eos):
        raise ValueError('Output token budget reached; incomplete output excluded')
    final=generated
    if runtime.get('enable_thinking',True):
        # The opening <think> may already be in the prompt. Require a completed
        # reasoning channel, then decode only final-answer tokens for evaluation.
        closing=tokenizer.encode('</think>',add_special_tokens=False)
        ids=generated.tolist()
        if not closing:raise ValueError('Tokenizer cannot encode the thinking terminator')
        ends=[i+len(closing) for i in range(len(ids)-len(closing)+1)
              if ids[i:i+len(closing)]==closing]
        if len(ends)!=1:raise ValueError('Missing or repeated thinking terminator; no final report accepted')
        final=generated[ends[0]:]
    raw=tokenizer.decode(final,skip_special_tokens=False)
    if any(marker in raw.lower() for marker in ('<think','</think','<analysis','</analysis')):
        raise ValueError('Unexpected reasoning markers in final report')
    text=tokenizer.decode(final,skip_special_tokens=True).strip()
    if not text or sensor.re_search_instruction(text):raise ValueError('Empty or instruction-echo response')
    return text


def generate_qwen(study, tasks, arrays, runtime, limit=9, retry_errors=False):
    if runtime.get('model_id') != MODEL_ID: raise ValueError('Expected Qwen3.5-4B runtime')
    study=Path(study);modality=runtime['modality'];casefile=study/'generation_cases.json'
    if study.name!=qwen_run_name(runtime.get('enable_thinking',True)):
        raise ValueError('Study folder and runtime thinking mode differ; rerun setup and model loading')
    plan=read(study/'study_protocol.json');all_cases=read(casefile)
    if signature(all_cases)!=plan['cases_sha256']:raise ValueError('Generation cases changed')
    cases=[c for c in all_cases if c['modality']==modality]
    lookup={t['id']:t for t in tasks}
    original_config=runtime['model'].generation_config
    decoding=copy.deepcopy(original_config)
    decoding.repetition_penalty=plan['repetition_penalty'];decoding.no_repeat_ngram_size=plan['no_repeat_ngram_size']
    decoding.do_sample=False;decoding.num_beams=1;decoding.num_return_sequences=1
    decoding.min_length=0;decoding.min_new_tokens=0;decoding.forced_eos_token_id=None
    decoding.return_dict_in_generate=False
    protocol=dict(study=plan,model=MODEL_ID, enable_thinking=runtime.get('enable_thinking',True),revision=runtime['revision'],
        decoding=json.loads(json.dumps(decoding.to_dict(),default=str)),
        max_new_tokens=plan['max_new_tokens'][modality],reference_token_cap=plan['reference_token_cap'],
        base_prompt=BASE_PROMPT,report_text_scope='final_answer_only',
        token_budget_scope=('reasoning_plus_final_answer' if runtime.get('enable_thinking',True) else 'final_answer'),
        source={str(Path(f).name):sha(f) for f in [__file__,common.__file__,sensor.__file__,Path(sensor.__file__).with_name('candidate_preparation.py'),Path(__file__).with_name('report_study.py')]},
        libraries={name:version(name) for name in ('torch','transformers','numpy','Pillow')})
    if modality=='ecg': protocol['libraries']['matplotlib']=version('matplotlib')
    sign=signature(protocol);path=study/f'{modality}_reports.jsonl'
    with (study/f'{modality}_reports.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        other=study/f"{'cxr' if modality=='ecg' else 'ecg'}_generation_protocol.json"
        if other.exists() and read(other)['revision']!=runtime['revision']:
            raise ValueError('Use the same Qwen checkpoint revision for ECG and CXR')
        pp=study/f'{modality}_generation_protocol.json'
        if pp.exists() and read(pp)!=protocol:raise ValueError('Generation protocol changed: use a new study directory')
        if path.exists() and not pp.exists():raise ValueError('Existing reports lack a protocol')
        atomic_json(pp,protocol);latest=read_latest(path,repair_tail=True)
        if set(latest)-{(c['query_id'],c['arm']) for c in cases}:raise ValueError('Unexpected report IDs')
        attempted=0
        runtime['model'].generation_config=decoding
        try:
            for c in cases:
                key=(c['query_id'],c['arm']);t=lookup[c['query_id']]
                if any(str(c[k])!=str(t[k]) for k in ('patient_id','admission_id','modality','split','study_id','source_index')):
                    raise ValueError('Generation case/sensor identity mismatch')
                arr=arrays[t['split']][t['source_index']]
                tensor_hash=signature(dict(shape=list(arr.shape),dtype=str(arr.dtype),bytes=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()))
                prompt,evidence=make_prompt(c,runtime['tokenizer'],plan['reference_token_cap'])
                previous=latest.get(key)
                if previous:
                    if previous['protocol_signature']!=sign or previous['source_tensor_sha256']!=tensor_hash or previous['prompt_sha256']!=signature(prompt):
                        raise ValueError('Resume inputs or protocol changed')
                    if previous['status']=='ok' or not retry_errors:continue
                if limit and attempted>=limit:break
                result={k:c[k] for k in ('query_id','patient_id','admission_id','modality','study_id','arm')}
                result.update(protocol_signature=sign,source_tensor_sha256=tensor_hash,prompt_sha256=signature(prompt),
                    prompt=prompt,evidence_budget_audit=evidence,model=protocol['model'],revision=runtime['revision'])
                try:
                    image=study/'images'/f"{c['query_id']}.png"
                    result['sensor_sha256']=sensor.materialize(arr,modality,image)
                    text=infer_qwen(runtime,image,prompt,plan['max_new_tokens'][modality])
                    tokens=runtime['tokenizer'].encode(text,add_special_tokens=False)
                    repeated=Counter(tuple(tokens[i:i+8]) for i in range(max(0,len(tokens)-7)))
                    if max(repeated.values(),default=0)>8:raise ValueError('Repetition QC failed')
                    result.update(status='ok',text=text,response_sha256=signature(text),output_tokens=len(tokens))
                except Exception as exc:result.update(status='error',error=str(exc))
                with path.open('a') as f:
                    f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                latest[key]=result;attempted+=1
                print(f"{modality}: {attempted} new reports; {c['arm']}: {result['status']}",flush=True)
        finally:runtime['model'].generation_config=original_config
        summary=dict(total=len(cases),ok=sum(r['status']=='ok' for r in latest.values()),
            failed=sum(r['status']!='ok' for r in latest.values()),not_attempted=len(cases)-len(latest),processed_this_call=attempted)
        atomic_json(study/f'{modality}_generation_coverage.json',summary)
        return summary
