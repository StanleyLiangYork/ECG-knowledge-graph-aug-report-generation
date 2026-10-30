"""RAD-DINO equivalence probes and sensor-only MedGemma CXR extraction."""
import copy
from pathlib import Path
from importlib.metadata import version
import numpy as np
from common import *


def rad_bundle(cfg,snapshot=None):
    parent_setup(cfg)
    import torch
    from transformers import AutoImageProcessor,AutoModel
    from ecg_external_generation import local_snapshot
    bridge=read(Path(cfg['artifact_root'])/'config.json')
    if bridge['cxr_model_id']!='microsoft/rad-dino':raise ValueError('Expected original microsoft/rad-dino encoder')
    if snapshot is None:
        roots=[Path(bridge['hf_cache']),Path(bridge['hf_cache'])/'hub']
        found=sorted({p.resolve() for root in roots for p in (root/'models--microsoft--rad-dino'/'snapshots').glob('*') if (p/'config.json').is_file()})
        if len({(p.name,sha(p/'config.json')) for p in found})==1:snapshot=found[0]
        elif found:raise ValueError('Multiple RAD-DINO revisions; set original SNAPSHOT_PATH explicitly: '+str(found))
    snapshot=local_snapshot(bridge['cxr_model_id'],snapshot)
    if not torch.cuda.is_available():raise RuntimeError('Use original RAD-DINO CUDA environment for float16 equivalence')
    processor=AutoImageProcessor.from_pretrained(str(snapshot),local_files_only=True)
    model=AutoModel.from_pretrained(str(snapshot),torch_dtype=torch.float16,local_files_only=True).to('cuda').eval().requires_grad_(False)
    return dict(model=model,processor=processor,torch=torch,snapshot=Path(snapshot))


def rad_encode(bundle,image):
    torch=bundle['torch']
    with torch.inference_mode():
        x=bundle['processor'](images=[image],return_tensors='pt')['pixel_values'].to('cuda',dtype=torch.float16)
        out=bundle['model'](pixel_values=x)
        v=out.pooler_output if getattr(out,'pooler_output',None) is not None else out.last_hidden_state[:,0]
    return v.float().cpu().numpy()[0]


def verify_rad(cfg,snapshot=None,probes=5):
    from PIL import Image
    from inputs import assert_inputs
    assert_inputs(cfg);run=Path(cfg['run_dir']);bank=read(run/'banks/symile.json');inventory=read(run/'banks/cxr_inventory.json')
    bundle=rad_bundle(cfg,snapshot);snap=bundle['snapshot']
    raw=np.load(Path(cfg['symile_root'])/'data_npy/train/cxr_train.npy',mmap_mode='r',allow_pickle=False)
    external=np.load(cfg['cxr_features'],mmap_mode='r',allow_pickle=False)
    rng=np.random.default_rng(cfg['seed']);checks=[]
    for kind,rows in [('symile',bank),('mimic',inventory)]:
        for i in rng.choice(len(rows),min(probes,len(rows)),replace=False):
            row=rows[i]
            if kind=='symile':
                x=np.asarray(raw[row['source_index']],dtype=np.float32)
                x=x*np.array([.229,.224,.225],dtype=np.float32)[:,None,None]+np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
                image=Image.fromarray((np.clip(x,0,1).transpose(1,2,0)*255).round().astype('uint8'))
                expected=np.asarray(row['cxr_embedding'],dtype=np.float32)
            else:
                with Image.open(row['image_path']) as im:image=im.convert('RGB')
                expected=np.asarray(external[row['source_index']],dtype=np.float32)
            actual=rad_encode(bundle,image)
            good=actual.shape==expected.shape and np.isfinite(actual).all() and np.linalg.norm(expected)>0
            rel=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)) if good else None
            cosine=float(np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))) if good else None
            checks.append(dict(source=kind,id=row['id'],relative_l2=rel,cosine=cosine,passed=bool(good and rel<=.005 and cosine>=.99999)))
    result=dict(passed=bool(checks) and all(x['passed'] for x in checks),checks=checks,
        snapshot=str(snap),checkpoint_files={str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file()},
        symile_bank_sha256=sha(run/'banks/symile.json'),inventory_sha256=sha(run/'banks/cxr_inventory.json'),
        cxr_features_sha256=sha(cfg['cxr_features']),libraries={name:version(name) for name in ('torch','transformers','Pillow','numpy')})
    atomic(run/'audits/rad_equivalence.json',result)
    if not result['passed']:raise ValueError('RAD-DINO spaces not reproduced; inspect rad_equivalence.json, do not bypass')
    return result


def extract_cxr(cfg,runtime,max_new=3,retry_errors=False,max_run_hours=30.,max_new_tokens=1024):
    assert_config(cfg)
    parent_setup(cfg)
    import sensor_candidate_generation as sensor
    from candidate_preparation import extract
    from inputs import observations
    if runtime.get('modality')!='cxr':raise ValueError('Use MedGemma CXR runtime')
    run=Path(cfg['run_dir']);inventory=read(run/'banks/cxr_inventory.json');folder=run/'extraction';folder.mkdir(exist_ok=True)
    settings=dict(model='google/medgemma-1.5-4b-it',revision=runtime['revision'],input_scope='sensor_only',
        prompt=sensor.PROMPTS['cxr'],max_new_tokens=max_new_tokens,inventory_sha256=sha(run/'banks/cxr_inventory.json'),
        sensor_helper_sha256=sha(sensor.__file__),libraries={x:version(x) for x in ('torch','transformers','Pillow')})
    freeze_stage(run,'cxr_extraction',settings);sign=signature(settings);journal=folder/'cxr_candidates.jsonl';processed=0
    original=runtime['model'].generation_config;decoding=copy.deepcopy(original)
    decoding.do_sample=False;decoding.num_beams=1;decoding.repetition_penalty=1.1;decoding.no_repeat_ngram_size=12
    decoding.min_new_tokens=None;decoding.min_length=0;decoding.forced_eos_token_id=None
    runtime['model'].generation_config=decoding
    try:
        with lock(folder/'writer.lock'),budget(max_run_hours,folder/'STOP') as stop:
            saved=latest(journal,repair=True)
            for row in inventory:
                if stop() or (max_new and processed>=max_new):break
                previous=saved.get((row['id'],'sensor_only'))
                if previous:
                    if previous['protocol_signature']!=sign or previous['sensor_sha256']!=row['image_sha256']:raise ValueError('Candidate resume provenance mismatch')
                    if previous['status']=='ok' or not retry_errors:continue
                if sha(row['image_path'])!=row['image_sha256']:raise ValueError('CXR image changed')
                r=dict(query_id=row['id'],arm='sensor_only',model=settings['model'],revision=runtime['revision'],
                       protocol_signature=sign,sensor_sha256=row['image_sha256'],input_scope='sensor_only')
                try:
                    text=sensor.infer(runtime,row['image_path'],max_new_tokens)
                    ids=runtime['tokenizer'].encode(text,add_special_tokens=False)
                    from collections import Counter
                    grams=Counter(tuple(ids[i:i+8]) for i in range(max(0,len(ids)-7)))
                    if max(grams.values(),default=0)>8:raise ValueError('Repetition QC failed')
                    r.update(status='ok',text=text,response_sha256=signature(text),findings=observations(extract(text,'cxr')))
                except Exception as exc:r.update(status='error',error=str(exc),error_type=type(exc).__name__)
                append(journal,r);saved[row['id'],'sensor_only']=r;processed+=1;print(row['id'],r['status'],flush=True)
                if r['status']=='error' and ('device-side assert' in r['error'] or 'illegal memory access' in r['error']):break
    finally:runtime['model'].generation_config=original
    counts=dict(total=len(inventory),ok=sum(saved.get((r['id'],'sensor_only'),{}).get('status')=='ok' for r in inventory),processed_this_call=processed)
    atomic(folder/'coverage.json',counts);return counts


def finalize_banks(cfg):
    from inputs import assert_inputs
    assert_inputs(cfg);run=Path(cfg['run_dir']);audit=read(run/'audits/rad_equivalence.json')
    if not audit['passed'] or audit['symile_bank_sha256']!=sha(run/'banks/symile.json') or audit['inventory_sha256']!=sha(run/'banks/cxr_inventory.json') or audit['cxr_features_sha256']!=sha(cfg['cxr_features']):
        raise ValueError('RAD-DINO equivalence missing/changed')
    inventory=read(run/'banks/cxr_inventory.json');vectors=np.load(cfg['cxr_features'],mmap_mode='r',allow_pickle=False)
    journal=run/'extraction/cxr_candidates.jsonl';saved=latest(journal);settings=read(run/'stages/cxr_extraction.json')['inputs']
    from candidate_preparation import extract
    from inputs import observations
    rows=[]
    for r in inventory:
        v=saved.get((r['id'],'sensor_only'))
        if not v or v['status']!='ok':raise ValueError('Complete MedGemma sensor-only extraction: '+r['id'])
        if v['protocol_signature']!=signature(settings) or v['sensor_sha256']!=r['image_sha256'] or v['response_sha256']!=signature(v['text']):raise ValueError('Candidate provenance mismatch')
        if v['findings']!=observations(extract(v['text'],'cxr')):raise ValueError('Candidate parse mismatch')
        rows.append(dict(r,findings=v['findings'],extraction=dict(model=v['model'],revision=v['revision'],input_scope='sensor_only')))
    freeze_stage(run,'banks_ready',dict(candidates_sha256=sha(journal),rad_audit_sha256=sha(run/'audits/rad_equivalence.json'),
        symile_sha256=sha(run/'banks/symile.json'),mimic_sha256=signature(rows)))
    immutable(run/'banks/mimic.json',rows)
    # Separate spaces; normalized immutable arrays are written once after verification.
    from retrieval import normalize
    bank=read(run/'banks/symile.json')
    for name,arr in [('symile_ecg',np.array([r['ecg_embedding'] for r in bank])),('symile_cxr',np.array([r['cxr_embedding'] for r in bank])),
                     ('mimic_cxr',np.array([vectors[r['source_index']] for r in rows]))]:
        path=run/'banks'/(name+'.npy');value=normalize(arr)
        if path.exists():
            if not np.array_equal(np.load(path,allow_pickle=False),value):raise ValueError('Changed index matrix')
        else:
            tmp=path.with_suffix('.tmp.npy');np.save(tmp,value,allow_pickle=False);tmp.replace(path)
    freeze_stage(run,'indexes',dict(matrices={p.name:sha(p) for p in sorted((run/'banks').glob('*.npy'))}))
    return dict(symile_admissions=len(bank),mimic_studies=len(rows),ready=True)
