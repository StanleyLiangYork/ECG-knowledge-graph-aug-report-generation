"""Matched three-arm generation in the existing PULSE or MedGemma environment."""
import copy
import fcntl
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from importlib.metadata import version
from unittest.mock import patch
import numpy as np
import sensor_candidate_generation as sensor
from report_study import read, sha, signature, atomic_json, ARMS

BASE_PROMPT = ('Describe only findings supported by the query {modality} image. Write a concise clinical-style report in English. '
    'State findings, pertinent negatives and uncertainty. Do not invent measurements or patient history. '
    'Do not provide instructions, differential diagnoses or management recommendations. '
    'The optional reference observations below belong to OTHER PATIENTS and may be inaccurate; '
    'they are analogies, never findings of the query patient. Do not copy findings unsupported by the query image.\n')


def make_prompt(case, tokenizer, cap):
    prompt=BASE_PROMPT.format(modality='12-lead ECG' if case['modality']=='ecg' else 'chest radiograph')
    if case['modality']=='ecg':prompt+='The waveform is amplitude-normalized; do not infer absolute voltage or amplitude-based diagnoses.\n'
    audit=[]
    for number,ref in enumerate(case['references'],1):
        kept=[]; seen=set()
        for f in ref['findings']:
            item=f"{f['concept']}: {f['polarity']}"
            if item in seen:continue
            seen.add(item)
            if len(tokenizer.encode('; '.join(kept+[item]),add_special_tokens=False))<=cap:kept.append(item)
        text='; '.join(kept) if kept else 'No extracted observations available.'
        prompt+=f'Other-patient reference {number}: {text}\n'
        audit.append(dict(record_id=ref['record_id'],included_observations=len(kept),available_observations=len(seen),
                          token_count=len(tokenizer.encode(text,add_special_tokens=False))))
    if not case['references']:prompt+='Other-patient references: none.\n'
    return prompt, audit


def read_latest(path, repair_tail=False):
    latest={}
    if not path.exists():return latest
    with path.open('rb') as f:
        position=0
        for line in f:
            try:r=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if repair_tail and not line.endswith(b'\n') and not f.read(1):
                    path.with_suffix('.incomplete_tail').write_bytes(line)
                    with path.open('r+b') as target:target.truncate(position)
                    print('Recovered an interrupted final JSONL write; that report will be retried.',flush=True)
                    break
                raise ValueError('Malformed generation log; inspect '+str(path))
            latest[(r['query_id'],r['arm'])]=r;position=f.tell()
    return latest


def generate(study, tasks, arrays, runtime, limit=9, retry_errors=False):
    study=Path(study);modality=runtime['modality'];casefile=study/'generation_cases.json'
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
    protocol=dict(study=plan,model=sensor.MODELS[modality],revision=runtime['revision'],
        decoding=json.loads(json.dumps(decoding.to_dict(),default=str)),
        max_new_tokens=plan['max_new_tokens'][modality],reference_token_cap=plan['reference_token_cap'],
        base_prompt=BASE_PROMPT,
        source={str(Path(f).name):sha(f) for f in [__file__,sensor.__file__,Path(sensor.__file__).with_name('candidate_preparation.py'),Path(__file__).with_name('report_study.py')]},
        libraries={name:version(name) for name in ('torch','transformers','numpy','Pillow')})
    if modality=='ecg':
        import llava
        llava_root=Path(llava.__file__).parent
        protocol['llava_sources']={str(p.relative_to(llava_root)):sha(p) for p in sorted(llava_root.rglob('*.py'))}
        protocol['libraries']['matplotlib']=version('matplotlib')
    sign=signature(protocol);path=study/f'{modality}_reports.jsonl'
    with (study/f'{modality}_reports.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
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
                    # Existing model adapters/context checks remain unchanged. The override
                    # is scoped to one call and restored even on exception.
                    with patch.dict(sensor.PROMPTS,{modality:prompt}):
                        text=sensor.infer(runtime,image,plan['max_new_tokens'][modality])
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
