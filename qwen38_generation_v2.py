"""Versioned Qwen3.8 mode/budget experiments; preserve signed v1/Llama helpers."""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from report_study import read, sha, signature, atomic_json
import ecg_external_generation as engine
from general_ecg_generators import load as load_original

MODEL_ID = 'Qwen/Qwen3.8-27B'


def run_name(enable_thinking, max_new_tokens):
    if type(enable_thinking) is not bool or type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('Boolean thinking switch and positive integer token budget required')
    return f"qwen38_27b_{'thinking' if enable_thinking else 'nonthinking'}_t{max_new_tokens}_v2"


def prepare(base, enable_thinking=False, max_new_tokens=1024):
    base=Path(base); study=base/run_name(enable_thinking,max_new_tokens)
    plan=read(base/'study_protocol.json'); all_cases=read(base/'generation_cases.json')
    if signature(all_cases)!=plan['cases_sha256']: raise ValueError('Base case hash mismatch')
    cases=[c for c in all_cases if c['modality']=='ecg']
    if not cases: raise ValueError('No ECG cases')
    import general_ecg_generators
    settings=dict(model_id=MODEL_ID,enable_thinking=enable_thinking,max_new_tokens=max_new_tokens,
                  helper_sha256=sha(__file__),loader_sha256=sha(general_ecg_generators.__file__))
    plan=dict(plan,cases_sha256=signature(cases),parent_cases_sha256=signature(all_cases),
              max_new_tokens=dict(plan['max_new_tokens'],ecg=max_new_tokens),qwen38_settings=settings)
    engine.immutable(study/'study_protocol.json',plan)
    engine.immutable(study/'generation_cases.json',cases)
    return study


def load(path, enable_thinking=False, load_in_4bit=False):
    if type(enable_thinking) is not bool: raise ValueError('Thinking must be boolean')
    runtime=load_original('qwen38_27b',path,load_in_4bit=load_in_4bit)
    runtime['enable_thinking']=enable_thinking
    return runtime


def final_text(tokenizer, ids, thinking):
    if thinking:
        closing=tokenizer.encode('</think>',add_special_tokens=False)
        if not closing: raise ValueError('Missing thinking delimiter encoding')
        ends=[i+len(closing) for i in range(len(ids)-len(closing)+1) if ids[i:i+len(closing)]==closing]
        if len(ends)!=1: raise ValueError('Missing/repeated thinking terminator; no final report accepted')
        ids=ids[ends[0]:]
    raw=tokenizer.decode(ids,skip_special_tokens=False)
    if any(m in raw.lower() for m in ('<think','</think','<analysis','</analysis')):
        raise ValueError('Unexpected reasoning markers in final report')
    return tokenizer.decode(ids,skip_special_tokens=True).strip()


def infer(runtime,image_path,prompt,max_new_tokens):
    import torch
    from PIL import Image
    model=runtime['model']; processor=runtime['processor']; tok=runtime['tokenizer']
    with Image.open(image_path) as im: image=im.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',enable_thinking=runtime['enable_thinking'])
    length=inputs['input_ids'].shape[-1]
    prompt_tail=tok.decode(inputs['input_ids'][0,-100:].tolist(),skip_special_tokens=False)
    if length+max_new_tokens>runtime['context']: raise ValueError('Context budget exceeded')
    device=model.get_input_embeddings().weight.device
    inputs={k:v.to(device=device,dtype=runtime['dtype']) if v.is_floating_point() else v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    ids=output[0,length:].tolist()
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else list(eos or [])
    grams=Counter(tuple(ids[i:i+8]) for i in range(max(0,len(ids)-7)))
    diagnostic=dict(enable_thinking=runtime['enable_thinking'],generated_tokens=len(ids),
        max_new_tokens=max_new_tokens,generation_eos_ids=eos,eos_positions=[i for i,x in enumerate(ids) if x in eos],
        most_repeated_8_token_sequence_count=max(grams.values(),default=0),
        formatted_prompt_tail=prompt_tail,raw_output=tok.decode(ids,skip_special_tokens=False),error=None)
    try:
        if not ids or (len(ids)>=max_new_tokens and ids[-1] not in eos):
            raise ValueError('Output token budget reached; incomplete output excluded')
        text=final_text(tok,ids,runtime['enable_thinking'])
        if not text: raise ValueError('Empty final report')
        return text
    except Exception as exc:
        diagnostic['error']=str(exc)
        raise
    finally:
        key=signature({'prompt':prompt,'image_sha256':sha(image_path)})
        atomic_json(Path(runtime['diagnostic_dir'])/(key+'.json'),diagnostic)


def generate(study,runtime,provider,**kwargs):
    import general_ecg_generators
    study=Path(study); settings=read(study/'study_protocol.json')['qwen38_settings']
    if settings['helper_sha256']!=sha(__file__) or settings['loader_sha256']!=sha(general_ecg_generators.__file__):
        raise ValueError('Helper changed; new versioned study required')
    if runtime['model_id']!=MODEL_ID or runtime['family']!='qwen' or runtime['enable_thinking']!=settings['enable_thinking']:
        raise ValueError('Runtime mode/model differs from study; rerun setup and loading')
    runtime['diagnostic_dir']=study/'diagnostics'
    with patch.object(engine,'infer',side_effect=infer):
        return engine.generate(study,runtime,provider,**kwargs)
