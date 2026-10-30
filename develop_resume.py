"""Notebook 03 orchestration with epoch checkpoints and cached judged pairs."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import numpy as np
import torch
import torch_geometric
from core import load, digest, build_graph
from learning import fit, pair_data, atomic_torch_save


def atomic_json(path, value):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def develop(root, out, config):
    root=Path(root); out=Path(out)
    with (out/'development.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _develop(root,out,config)


def _develop(root,out,config):
    inputs=['records.json','targets.json','pools.json','bank_audit.json',
            'ecg.faiss','cxr.faiss','ecg_ids.json','cxr_ids.json']
    missing=[name for name in inputs if not (out/name).is_file()]
    if missing: raise FileNotFoundError('Finish notebook 02 first: '+repr(missing))
    identity=dict(inputs={f:digest(out/f) for f in inputs},
        source={f:digest(root/f) for f in ['core.py','learning.py','develop_resume.py','config.json']},
        config=config,torch=str(torch.__version__),pyg=str(torch_geometric.__version__),numpy=np.__version__)
    folder=out/'training_checkpoints'; folder.mkdir(exist_ok=True)
    manifest=folder/'manifest.json'
    if manifest.exists() and load(manifest)!=identity:
        raise ValueError('Training inputs, configuration, code or libraries changed. Use a separate output directory; do not mix checkpoints.')
    atomic_json(manifest,identity)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    # Freeze is written only when every required fit is complete.
    (out/'freeze.json').unlink(missing_ok=True)
    rows=load(out/'records.json'); targets=load(out/'targets.json'); pools=load(out/'pools.json')
    bank_audit=load(out/'bank_audit.json')
    if not all(bank_audit.get(k) for k in ['training_only','own_patient_excluded','faiss_cosine_agreement']):
        raise ValueError('Notebook 02 audit failed')
    if bank_audit.get('records_sha256',identity['inputs']['records.json'])!=identity['inputs']['records.json']:
        raise ValueError('Banks belong to changed records')
    if set(pools)!={r['id'] for r in rows}: raise ValueError('Saved pool coverage mismatch')
    graph=build_graph(rows); atomic_json(out/'graph.json',graph)
    print(f"Training graph: {len(graph['vocabulary'])} concepts, {len(graph['edges'])} edges",flush=True)
    prepared=[]
    for split in ('train','val'):
        path=folder/f'{split}_pairs.pt'
        if path.exists():
            cache=torch.load(path,map_location='cpu',weights_only=True)
            if cache['fingerprint']!=fingerprint: raise ValueError('Pair cache mismatch')
            data=cache['data']; print(f'Reusing {split} pairs: {len(data[2])}',flush=True)
        else:
            print(f'Preparing {split} pairs from notebook 02 pools...',flush=True)
            data=pair_data(rows,targets,split,pools=pools)
            atomic_torch_save(dict(fingerprint=fingerprint,data=data),path)
            print(f'Saved {split} pairs: {len(data[2])}',flush=True)
        prepared.append(data)
    results=[]
    batch_size=config.get('training_pair_batch_size',8192)
    def train(kind,seed,shuffle=False):
        label='shuffled' if shuffle else kind
        path=out/f'{label}_{seed}.pt'
        model,loss=fit(rows,targets,graph,kind,seed=seed,shuffle=shuffle,
            epochs=config['epochs'],prepared=prepared,checkpoint_path=folder/f'{label}_{seed}.pt',
            fingerprint=fingerprint,pair_batch_size=batch_size)
        atomic_torch_save(model.state_dict(),path)
        result=dict(kind=label,seed=seed,val_loss=loss,path=path.name)
        if shuffle: result['architecture']=kind
        results.append(result)
    for seed in config['seeds']:
        for kind in ('mlp','sage','transformer'): train(kind,seed)
    means={kind:float(np.mean([r['val_loss'] for r in results if r['kind']==kind])) for kind in ('sage','transformer')}
    selected=min(means,key=means.get)
    for seed in config['seeds']: train(selected,seed,True)
    atomic_json(out/'development.json',dict(selected=selected,validation=results))
    # Detect file edits during the run before publishing a frozen artifact set.
    if any(digest(out/f)!=value for f,value in identity['inputs'].items()) or any(digest(root/f)!=value for f,value in identity['source'].items()):
        raise ValueError('Inputs/code changed during training; freeze not written')
    files=['records.json','targets.json','graph.json','development.json','pools.json',
           'ecg.faiss','cxr.faiss','ecg_ids.json','cxr_ids.json']+[r['path'] for r in results]
    atomic_json(out/'freeze.json',dict(synthetic=config['mode']=='synthetic',selected=selected,seeds=config['seeds'],
        artifacts={f:digest(out/f) for f in files},source={f:digest(root/f) for f in ['core.py','learning.py','config.json']}))
    print('READY: frozen graph architecture:',selected,'; validation losses:',means,flush=True)
    return selected
