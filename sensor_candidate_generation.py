"""Separate-environment sensor-only generation. Compatible with Python 3.9."""
import json
import sys
from pathlib import Path
from importlib.metadata import version
import numpy as np
from candidate_preparation import read,read_lines,sha,text_sha,write,MODELS,uid

PROMPTS={
 'ecg':'Interpret only this 12-lead ECG image. Write a concise clinical-style ECG report in English. State visible findings, explicit negative findings when assessable, and uncertainty. Do not invent measurements. The waveform is amplitude-normalized: do not infer absolute voltage or amplitude-based diagnoses. Do not output JSON or instructions.',
 'cxr':'Interpret only this chest radiograph. Write a concise clinical-style report in English. State visible findings, explicit negative findings when assessable, and uncertainty. Do not infer patient history. Do not output JSON or instructions.'}
LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def sensor_plan(manifest_path,sensor_path,symile_root,modality):
    manifest=[r for r in read(manifest_path) if r['modality']==modality]
    sources={r['id']:r for r in read(sensor_path)};arrays={};ids={}
    for split in ('train','val','test'):
        folder=Path(symile_root)/'data_npy'/split
        path=folder/f'{modality}_{split}.npy'
        arrays[split]=np.load(path,mmap_mode='r',allow_pickle=False)
        ids[split]=np.load(folder/f'hadm_id_{split}.npy',mmap_mode='r',allow_pickle=False).reshape(-1)
        if len(arrays[split])!=len(ids[split]):raise ValueError(f'{split}: tensor/admission-array length mismatch')
        expected_tail=(1,5000,12) if modality=='ecg' else (3,320,320)
        if arrays[split].shape[1:]!=expected_tail:raise ValueError(f'{split}: unexpected {modality} tensor shape {arrays[split].shape}')
    tasks=[]
    for r in manifest:
        s=sources[r['id']];i=int(s['source_index']);split=r['split']
        if s['split']!=split or s['admission_id']!=r['admission_id'] or not 0<=i<len(ids[split]) or uid(ids[split][i])!=r['admission_id']:
            raise ValueError('Sensor/admission identity mismatch; rerun 00b')
        tasks.append(dict(r,study_id=s['study_id'],source_index=i))
    return tasks,arrays

def materialize(array,modality,path):
    from PIL import Image
    x=np.asarray(array,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Non-finite sensor tensor')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if modality=='cxr':
        pixels=x.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])
        if pixels.min() < -.02 or pixels.max() > 1.02:raise ValueError('CXR tensor does not match expected ImageNet normalization')
        Image.fromarray(np.round(np.clip(pixels,0,1)*255).astype(np.uint8)).save(path)
    else:
        if x.min() < -1.001 or x.max() > 1.001 or not np.any(x):raise ValueError('Unexpected normalized ECG signal')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        signal=x.reshape(5000,12);t=np.arange(5000)/500.
        fig,axes=plt.subplots(12,1,figsize=(12,10),sharex=True)
        for j,ax in enumerate(axes):
            ax.plot(t,signal[:,j],color='black',linewidth=.6);ax.set_ylim(-1.05,1.05)
            ax.set_ylabel(LEADS[j],rotation=0,ha='right');ax.set_yticks([]);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Seconds (500 Hz); normalized amplitude, not mV')
        fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    return sha(path)

def load_runtime(modality,model_path,pulse_repo=None):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Run model inference on a CUDA GPU allocation, not a CPU/login node')
    path=Path(model_path).expanduser()
    if not path.is_dir():raise FileNotFoundError('Set MODEL_PATH to your existing local model snapshot directory')
    # Require an immutable snapshot or explicit local checkpoint fingerprint.
    revision=path.resolve().name
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        # Hash all model files once, including processor/tokenizer config; may take time.
        print('Fingerprinting local model directory for provenance...',flush=True)
        files={str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
        revision=text_sha(json.dumps(files,sort_keys=True))
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if modality=='ecg':
        if pulse_repo:
            llava_root=Path(pulse_repo)/'LLaVA'
            if not (llava_root/'llava').is_dir():raise FileNotFoundError('PULSE_REPO must contain LLaVA/llava')
            sys.path.insert(0,str(llava_root))
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        disable_torch_init()
        tok,model,proc,limit=load_pretrained_model(str(path),None,'pulse-7b',device_map='auto')
        limits=[getattr(model.config,k,None) for k in ('max_position_embeddings','tokenizer_model_max_length')]
        limits=[int(v) for v in limits if isinstance(v,int) and 0<v<10**7]
        context=min(limits) if limits else int(limit)
    else:
        from transformers import AutoProcessor,AutoModelForImageTextToText
        proc=AutoProcessor.from_pretrained(str(path),local_files_only=True)
        model=AutoModelForImageTextToText.from_pretrained(str(path),torch_dtype=dtype,device_map='auto',local_files_only=True)
        tok=proc.tokenizer;context=getattr(getattr(model.config,'text_config',model.config),'max_position_embeddings',8192)
    model.eval()
    return dict(model=model,processor=proc,tokenizer=tok,revision=revision,dtype=dtype,context=context,modality=modality)

def infer(runtime,image_path,max_new_tokens=384):
    import torch
    from PIL import Image
    m=runtime['model'];tok=runtime['tokenizer'];modality=runtime['modality'];prompt=PROMPTS[modality]
    device=m.get_input_embeddings().weight.device
    with Image.open(image_path) as im:image=im.convert('RGB')
    if modality=='ecg':
        from llava.constants import DEFAULT_IMAGE_TOKEN,DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN,IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token,process_images
        image_token=DEFAULT_IM_START_TOKEN+DEFAULT_IMAGE_TOKEN+DEFAULT_IM_END_TOKEN if getattr(m.config,'mm_use_im_start_end',False) else DEFAULT_IMAGE_TOKEN
        conv=conv_templates['llava_v1'].copy();conv.append_message(conv.roles[0],image_token+'\n'+prompt);conv.append_message(conv.roles[1],None)
        ids=tokenizer_image_token(conv.get_prompt(),tok,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0).to(device)
        images=process_images([image],runtime['processor'],m.config)
        if getattr(m.config,'mm_patch_merge_type','flat')!='flat':raise ValueError('Unsupported PULSE patch merge; use verified upstream adapter')
        tensor=images[0] if isinstance(images,list) or images.ndim==5 else images
        expanded=ids.shape[-1]-1+int(tensor.shape[0])*int(m.get_vision_tower().num_patches)
        if expanded+max_new_tokens>runtime['context']:raise ValueError('PULSE context budget exceeded')
        images=[v.to(device=device,dtype=torch.float16) for v in images] if isinstance(images,list) else images.to(device=device,dtype=torch.float16)
        with torch.inference_mode():output=m.generate(ids,images=images,image_sizes=[image.size],do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        # PULSE/LLaVA generate may return only generated tokens. Strip an actual matching prefix only.
        generated=output[0]
        if len(generated)>=ids.shape[-1] and torch.equal(generated[:ids.shape[-1]],ids[0]):generated=generated[ids.shape[-1]:]
    else:
        proc=runtime['processor'];messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        chat=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=proc(text=chat,images=[image],return_tensors='pt',add_special_tokens=False)
        if inputs['input_ids'].shape[-1]+max_new_tokens>runtime['context']:raise ValueError('MedGemma context budget exceeded')
        inputs={k:v.to(device=device,dtype=runtime['dtype']) if v.is_floating_point() else v.to(device) for k,v in inputs.items()}
        with torch.inference_mode():output=m.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,pad_token_id=tok.pad_token_id)
        generated=output[0,inputs['input_ids'].shape[-1]:]
    eos=getattr(m.generation_config,'eos_token_id',tok.eos_token_id);eos=[eos] if isinstance(eos,int) else eos or []
    if len(generated)>=max_new_tokens and (not len(generated) or int(generated[-1]) not in eos):raise ValueError('Output token budget reached; incomplete output excluded')
    text=tok.decode(generated,skip_special_tokens=True).strip()
    if '<think' in text.lower() or '<analysis' in text.lower():raise ValueError('Unexpected reasoning-tag output; excluded from clinical candidate text')
    if not text or re_search_instruction(text):raise ValueError('Empty or instruction-echo response')
    return text

def re_search_instruction(text):
    import re
    return bool(re.match(r'^(?:please|write only|interpret only|return one JSON)\b',text,re.I))

def run_generation(tasks,arrays,runtime,directory,limit=10,retry_errors=False,max_new_tokens=384):
    import fcntl
    modality=runtime['modality'];directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # One process per modality/file; prevents concurrent append corruption.
    with (directory/f'{modality}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=directory/f'{modality}_sensor_only.jsonl'
        old=read_lines(path) if path.is_file() else [];latest={r['id']:r for r in old}
        if set(latest)-{t['id'] for t in tasks}:raise ValueError('Output directory contains a different cohort')
        protocol=dict(model=MODELS[modality],revision=runtime['revision'],prompt=PROMPTS[modality],
          max_new_tokens=max_new_tokens,rendering='symile_npy_v1_imagenet_inverse_or_12lead_normalized_500Hz',
          code_sha256=sha(__file__),candidate_helpers_sha256=sha(Path(__file__).with_name('candidate_preparation.py')),
          torch=version('torch'),transformers=version('transformers'),numpy=version('numpy'),pillow=version('Pillow'))
        if modality=='ecg':
            import llava
            protocol['llava_source_root']=str(Path(llava.__file__).resolve().parent)
            protocol['llava_source_sha256']=text_sha(json.dumps({str(p.relative_to(Path(llava.__file__).parent)):sha(p) for p in sorted(Path(llava.__file__).parent.rglob('*.py'))},sort_keys=True))
            protocol['matplotlib']=version('matplotlib')
        signature=text_sha(json.dumps(protocol,sort_keys=True));pp=directory/f'{modality}_protocol.json'
        if pp.exists() and read(pp)!=protocol:raise ValueError('Generation protocol changed. Use a fresh run directory.')
        write(pp,protocol)
        n=0
        for task in tasks:
            rid=task['id'];previous=latest.get(rid);array=arrays[task['split']][task['source_index']]
            source_hash=text_sha(text_sha(json.dumps({'shape':list(array.shape),'dtype':str(array.dtype)}))+__import__('hashlib').sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
            if previous:
                if previous.get('protocol_signature')!=signature or previous.get('source_tensor_sha256')!=source_hash or any(previous.get(k)!=task[k] for k in ('patient_id','admission_id','split','modality','study_id')):
                    raise ValueError('Resume identity, protocol or sensor changed. Use a fresh run directory.')
                if previous['status']=='ok' or not retry_errors:continue
            if limit and n>=limit:break
            result={k:task[k] for k in ('id','patient_id','admission_id','split','modality','study_id')}
            result.update(model=MODELS[modality],revision=runtime['revision'],processor_revision=runtime['revision'],
                input_scope='sensor_only',prompt_sha256=text_sha(PROMPTS[modality]),protocol_signature=signature,source_tensor_sha256=source_hash)
            image_path=directory/'images'/f'{rid}.png'
            try:
                result['sensor_sha256']=materialize(array,modality,image_path)
                text=infer(runtime,image_path,max_new_tokens)
                result.update(status='ok',text=text,response_sha256=text_sha(text))
            except Exception as exc:result.update(status='error',error=str(exc))
            with path.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
            latest[rid]=result;n+=1
            if n%10==0:print('Processed this invocation:',n,flush=True)
        summary={'total':len(tasks),'ok':sum(r['status']=='ok' for r in latest.values()),'failed':sum(r['status']!='ok' for r in latest.values()),'not_attempted':len(tasks)-len(latest)}
        write(directory/f'{modality}_coverage.json',summary)
        return summary

def find_snapshot(model_id,configured=None,cache_roots=None):
    """Resolve cached snapshots; duplicate copies of one commit are not ambiguous."""
    import os
    if configured:
        path=Path(configured).expanduser().resolve()
        if not (path/'config.json').is_file():
            raise FileNotFoundError(f'MODEL_PATH is not a local model snapshot: {path}')
        return str(path)
    if cache_roots is None:
        roots=[]
        if os.environ.get('HF_HUB_CACHE'):roots.append(Path(os.environ['HF_HUB_CACHE']))
        if os.environ.get('HF_HOME'):roots += [Path(os.environ['HF_HOME']),Path(os.environ['HF_HOME'])/'hub']
        roots += [Path('/path/to/data/hf_cache'),Path.home()/'.cache/huggingface/hub']
    else:roots=[Path(p) for p in cache_roots]
    candidates=[]
    for root in roots:
        for base in (root,root/'hub'):
            folder=base/('models--'+model_id.replace('/','--'))
            ref=folder/'refs'/'main'
            if ref.is_file():
                options=[folder/'snapshots'/ref.read_text().strip()]
            else:
                options=sorted((folder/'snapshots').iterdir()) if (folder/'snapshots').is_dir() else []
            for path in options:
                if (path/'config.json').is_file():
                    resolved=str(path.resolve())
                    if resolved not in candidates:candidates.append(resolved)
    if len(candidates)==1:return candidates[0]
    revisions={Path(p).name for p in candidates}
    if len(revisions)==1:
        revision=next(iter(revisions))
        if len(revision)==40 and all(c in '0123456789abcdef' for c in revision.lower()):
            if len({sha(Path(p)/'config.json') for p in candidates})!=1:
                raise ValueError('Cached copies of the same revision have conflicting config.json files; set MODEL_PATH after inspection')
            print(f'Found {len(candidates)} cached copies of revision {revision}; using cache priority: {candidates[0]}')
            return candidates[0]
    raise FileNotFoundError('Set MODEL_PATH to one existing local snapshot. No unique revision was found. Candidates: '+repr(candidates))
