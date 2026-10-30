"""Additive ECG generators and durable journals. Does not modify frozen helpers."""
import copy
import fcntl
import json
import os
import signal
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from importlib.metadata import version
from report_study import read, sha, signature, atomic_json
from report_generation import make_prompt
from report_generation_resume import _read_resume_log

LLAMA_ID = 'convaiinnovations/ECG-Instruct-Llama-3.2-11B-Vision'
FAMILIES = ('pulse', 'qwen', 'llama')


def check_internal_runtime(runtime, base_study):
    """External generator must match the corresponding existing internal run."""
    base=Path(base_study)
    folders={'pulse':base,'qwen':base/'qwen35_4b_nonthinking_v2',
             'llama':base/'ecg_instruct_llama32_11b_v2'}
    p=read(folders[runtime['family']]/'ecg_generation_protocol.json')
    if p['model']!=runtime['model_id'] or p['revision']!=runtime['revision']:
        raise ValueError('External checkpoint differs from the internal generator')
    if runtime['family']=='llama' and p.get('load_in_4bit',False)!=runtime.get('load_in_4bit',False):
        raise ValueError('Llama precision differs from the internal run')
    if runtime['family']=='qwen' and p.get('enable_thinking')!=runtime.get('enable_thinking'):
        raise ValueError('Qwen thinking mode differs from the internal run')


def immutable(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Inputs/settings changed; choose a new output directory: ' + str(path))
    atomic_json(path, value)


@contextmanager
def budget(hours, stop_file):
    if not isinstance(hours, (int, float)) or not 0 < hours < float('inf'):
        raise ValueError('Positive finite hours required, below allocation time remaining')
    started = time.monotonic(); requested = []; handlers = {}
    def receive(signum, frame): requested.append(signal.Signals(signum).name)
    def reason():
        if requested: return requested[0]
        if Path(stop_file).exists(): return 'stop_file'
        if time.monotonic() - started >= hours * 3600: return 'time_budget'
        return None
    try:
        for name in ('SIGTERM', 'SIGUSR1'):
            sig = getattr(signal, name, None)
            if sig is not None:
                handlers[sig] = signal.getsignal(sig); signal.signal(sig, receive)
        yield reason
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


def local_snapshot(model_id, configured=None):
    if configured:
        p = Path(configured).expanduser().resolve()
        if not (p/'config.json').is_file(): raise FileNotFoundError(str(p))
        return p
    roots = [Path.home()/'.cache/huggingface/hub', Path('/path/to/data/hf_cache')]
    for key in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        if os.environ.get(key): roots.insert(0, Path(os.environ[key]))
    if os.environ.get('HF_HOME'): roots.insert(0, Path(os.environ['HF_HOME'])/'hub')
    found = sorted({p.resolve() for root in roots for p in (root/('models--'+model_id.replace('/', '--'))/'snapshots').glob('*') if (p/'config.json').is_file()})
    # Duplicate caches with the same commit/config are equivalent for discovery.
    identities = {(p.name, sha(p/'config.json')) for p in found}
    if len(identities) == 1 and found: return found[0]
    raise FileNotFoundError('Set MODEL_PATH to an existing immutable snapshot. Found: '+str(found))


def load_llama(path, load_in_4bit=False):
    import torch
    from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
    path = Path(path).resolve()
    if not torch.cuda.is_available(): raise RuntimeError('Use a CUDA GPU allocation')
    if read(path/'config.json').get('model_type') != 'mllama':
        raise ValueError('Expected the merged Mllama ECG-Instruct checkpoint')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    kw = dict(local_files_only=True, device_map='auto', torch_dtype=dtype)
    if load_in_4bit:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = MllamaForConditionalGeneration.from_pretrained(str(path), **kw).eval()
    if any(p.is_meta for p in model.parameters()): raise RuntimeError('Unloaded model parameters')
    revision = path.name
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        revision = signature({str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file()})
    return dict(model=model, processor=processor, tokenizer=processor.tokenizer,
        revision=revision, dtype=dtype, context=model.config.text_config.max_position_embeddings,
        modality='ecg', model_id=LLAMA_ID, family='llama', load_in_4bit=load_in_4bit)


def infer_llama(runtime, image_path, prompt, max_new_tokens):
    import torch
    from PIL import Image
    processor = runtime['processor']; model = runtime['model']; tok = runtime['tokenizer']
    with Image.open(image_path) as im: image = im.convert('RGB')
    messages = [{'role':'user','content':[{'type':'image'}, {'type':'text','text':prompt}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=image, return_tensors='pt', add_special_tokens=False)
    # Check on CPU before dispatch: Mllama's image placeholder is valid for its
    # input embeddings but intentionally absent from its output vocabulary.
    input_vocab=model.get_input_embeddings().weight.shape[0]
    output_vocab=model.get_output_embeddings().weight.shape[0]
    image_token=getattr(model.config,'image_token_index',None)
    validate_mllama_tokens(inputs['input_ids'],input_vocab,output_vocab,image_token)
    length = inputs['input_ids'].shape[-1]
    if length + max_new_tokens > runtime['context']: raise ValueError('Context budget exceeded')
    device = model.get_input_embeddings().weight.device
    inputs = {k:v.to(device=device, dtype=runtime['dtype']) if v.is_floating_point() else v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
            use_cache=True, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    generated = output[0, length:]
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if not len(generated) or (len(generated) >= max_new_tokens and int(generated[-1]) not in eos):
        raise ValueError('Output token budget reached; incomplete output excluded')
    return tok.decode(generated, skip_special_tokens=True).strip()


def validate_mllama_tokens(ids, input_vocab, output_vocab, image_token):
    if ids.numel()==0 or int(ids.min())<0 or int(ids.max())>=input_vocab:
        raise ValueError('Processor token IDs exceed model input embeddings; check processor/checkpoint pairing')
    invalid=ids[ids>=output_vocab]
    if invalid.numel() and (image_token is None or bool((invalid!=image_token).any())):
        raise ValueError('Unexpected non-image token outside model output vocabulary')


def configure_decoding(original, plan, family):
    decoding=copy.deepcopy(original)
    for key,value in dict(do_sample=False,num_beams=1,num_return_sequences=1,
        repetition_penalty=plan['repetition_penalty'],no_repeat_ngram_size=plan['no_repeat_ngram_size'],
        min_length=0,min_new_tokens=None,forced_eos_token_id=None,return_dict_in_generate=False).items():
        setattr(decoding,key,value)
    if family=='llama':
        # Avoid processors that index output scores using the input image token.
        # Do not resize the trained output head or alter checkpoint weights.
        decoding.repetition_penalty=1.0
        decoding.encoder_repetition_penalty=1.0
        decoding.no_repeat_ngram_size=0
    return decoding


def infer(runtime, image, prompt, limit):
    if runtime['family'] == 'llama': return infer_llama(runtime, image, prompt, limit)
    if runtime['family'] == 'qwen':
        from qwen_report_extension import infer_qwen
        return infer_qwen(runtime, image, prompt, limit)
    if runtime['family'] == 'pulse':
        import sensor_candidate_generation as sensor
        from unittest.mock import patch
        with patch.dict(sensor.PROMPTS, {'ecg':prompt}):
            return sensor.infer(runtime, image, limit)
    raise ValueError('Unknown family')


def prepare_internal(base, destination):
    base = Path(base); destination = Path(destination)
    plan = read(base/'study_protocol.json'); all_cases = read(base/'generation_cases.json')
    if signature(all_cases) != plan['cases_sha256']: raise ValueError('Base case hash mismatch')
    cases = [c for c in all_cases if c['modality'] == 'ecg']
    if not cases: raise ValueError('No ECG cases')
    plan = dict(plan, cases_sha256=signature(cases), parent_cases_sha256=signature(all_cases))
    immutable(destination/'study_protocol.json', plan)
    immutable(destination/'generation_cases.json', cases)
    return destination


def generate(study, runtime, image_provider, limit=9, retry_errors=False, max_run_hours=30.):
    """Provider(case) returns (image_path, source_tensor_sha256); labels never needed."""
    study = Path(study); plan = read(study/'study_protocol.json'); cases = read(study/'generation_cases.json')
    if signature(cases) != plan['cases_sha256']: raise ValueError('Case plan changed')
    if any(c['modality'] != 'ecg' for c in cases): raise ValueError('ECG-only extension')
    if limit < 0: raise ValueError('limit must be nonnegative; 0 means all remaining')
    original = runtime['model'].generation_config
    decoding = configure_decoding(original,plan,runtime['family'])
    protocol = dict(study=plan, model=runtime['model_id'], family=runtime['family'], revision=runtime['revision'],
        load_in_4bit=runtime.get('load_in_4bit',False), enable_thinking=runtime.get('enable_thinking',False),
        decoding=json.loads(json.dumps(decoding.to_dict(), default=str)), source_sha256=sha(__file__),
        libraries={name:version(name) for name in ('torch','transformers','numpy','Pillow')})
    import report_generation, sensor_candidate_generation
    protocol['helpers'] = {Path(m.__file__).name:sha(m.__file__) for m in (report_generation,sensor_candidate_generation)}
    if runtime['family']=='qwen':
        import qwen_report_extension
        protocol['helpers']['qwen_report_extension.py']=sha(qwen_report_extension.__file__)
    sign = signature(protocol); path = study/'ecg_reports.jsonl'; attempted=0; reason='completed_or_limit'
    with (study/'ecg_reports.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable(study/'ecg_generation_protocol.json', protocol)
        latest = _read_resume_log(path, repair_tail=True)
        expected = {(c['query_id'],c['arm']) for c in cases}
        if len(expected)!=len(cases) or set(latest)-expected: raise ValueError('Duplicate/unexpected case IDs')
        runtime['model'].generation_config=decoding
        try:
            with budget(max_run_hours, study/'STOP_ecg') as stop:
                for c in cases:
                    if stop(): reason=stop(); break
                    if limit and attempted>=limit: break
                    key=(c['query_id'],c['arm']); prompt,evidence=make_prompt(c,runtime['tokenizer'],plan['reference_token_cap'])
                    previous=latest.get(key)
                    image,tensor_hash=image_provider(c)
                    image_hash=sha(image)
                    if previous:
                        if any(previous.get(k)!=v for k,v in dict(protocol_signature=sign,prompt_sha256=signature(prompt),source_tensor_sha256=tensor_hash,sensor_sha256=image_hash).items()):
                            raise ValueError('Resume provenance mismatch')
                        if previous['status']=='ok' or not retry_errors: continue
                    row={k:c[k] for k in ('query_id','patient_id','admission_id','modality','study_id','arm')}
                    row.update(protocol_signature=sign,source_tensor_sha256=tensor_hash,sensor_sha256=image_hash,
                        prompt=prompt,prompt_sha256=signature(prompt),evidence_budget_audit=evidence,
                        model=runtime['model_id'],revision=runtime['revision'])
                    try:
                        text=infer(runtime,image,prompt,plan['max_new_tokens']['ecg'])
                        ids=runtime['tokenizer'].encode(text,add_special_tokens=False)
                        repeats=Counter(tuple(ids[i:i+8]) for i in range(max(0,len(ids)-7)))
                        if not text or sensor_candidate_generation.re_search_instruction(text): raise ValueError('Empty/instruction echo')
                        if max(repeats.values(),default=0)>8: raise ValueError('Repetition QC failed')
                        row.update(status='ok',text=text,response_sha256=signature(text),output_tokens=len(ids))
                    except Exception as exc:
                        import traceback
                        row.update(status='error',error=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc())
                    with path.open('a') as f:
                        f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
                    latest[key]=row; attempted+=1
                    print(c['query_id'],c['arm'],row['status'],flush=True)
                    if row['status']=='error':
                        print(row['error'],flush=True)
                        if any(s in row['error'].lower() for s in ('device-side assert','illegal memory access')):
                            reason='cuda_context_error_restart_kernel'
                            print('CUDA context is unsafe. Restart the kernel before further inference.',flush=True)
                            break
        except KeyboardInterrupt: reason='keyboard_interrupt'
        finally: runtime['model'].generation_config=original
        result=dict(total=len(cases),ok=sum(r['status']=='ok' for r in latest.values()),
            failed=sum(r['status']!='ok' for r in latest.values()),not_attempted=len(cases)-len(latest),
            processed_this_call=attempted,stop_reason=reason)
        atomic_json(study/'ecg_generation_coverage.json',result)
        return result


def internal_provider(study, tasks, arrays):
    import hashlib
    import numpy as np
    import sensor_candidate_generation as sensor
    lookup={t['id']:t for t in tasks}; cache={}
    def provide(c):
        rid=c['query_id']; t=lookup[rid]
        for key in ('patient_id','admission_id','study_id','split','source_index'):
            if str(t[key])!=str(c[key]): raise ValueError('Sensor identity mismatch')
        if rid not in cache:
            arr=arrays[t['split']][t['source_index']]
            digest=signature(dict(shape=list(arr.shape),dtype=str(arr.dtype),bytes=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()))
            image=Path(study)/'images'/(rid+'.png'); sensor.materialize(arr,'ecg',image)
            cache[rid]=(image,digest)
        return cache[rid]
    return provide
