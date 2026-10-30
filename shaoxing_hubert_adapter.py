"""Reproduce notebook2/03 ECG-HuBERT preprocessing; no external fitting."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from report_study import read, sha, signature, atomic_json
from ecg_external_generation import immutable

DESCRIPTION = '500 Hz, 5000 samples, canonical I II III aVR aVL aVF V1-V6; per-lead nanmean/nanstd over time; std<1e-6 ->1; nan_to_num; float32; transpose leads then flatten to 60000; CUDA float16 HuBERT; mean last_hidden_state over sequence; float32 output; no projection or L2 normalization.'


def hubert_input(batch):
    signal=np.asarray(batch,dtype='float32')
    if signal.ndim==4:signal=signal[:,0]
    if signal.ndim!=3 or signal.shape[1:]!=(5000,12):raise ValueError('Expected [B,5000,12]')
    mean=np.nanmean(signal,axis=1,keepdims=True); std=np.nanstd(signal,axis=1,keepdims=True)
    std[std<1e-6]=1.0
    signal=np.nan_to_num((signal-mean)/std).astype('float32')
    return signal.transpose(0,2,1).reshape(len(signal),-1)


def setup(artifact_root, destination, snapshot=None):
    artifact_root=Path(artifact_root); config=read(artifact_root/'config.json')
    model_id=config['hubert_model_id']
    if model_id!='Edoardo-BS/hubert-ecg-base':raise ValueError('Unexpected original encoder ID: '+model_id)
    if snapshot is None:
        roots=[Path(config['hf_cache']),Path(config['hf_cache'])/'hub']
        found=sorted({p.resolve() for root in roots for p in (root/('models--'+model_id.replace('/','--'))/'snapshots').glob('*') if (p/'config.json').is_file()})
        if len({(p.name,sha(p/'config.json')) for p in found})!=1:
            raise FileNotFoundError('Set SNAPSHOT_PATH to the original HuBERT snapshot; found: '+str(found))
        snapshot=found[0]
    snapshot=Path(snapshot).resolve()
    if not (snapshot/'config.json').is_file():raise FileNotFoundError(str(snapshot))
    weights=list(snapshot.glob('*.safetensors'))+list(snapshot.glob('*.bin'))
    if not weights:raise FileNotFoundError('No encoder weights in '+str(snapshot))
    # Descriptor fingerprints all local weights, configs and custom model code.
    files={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    descriptor=dict(model_id=model_id,snapshot=str(snapshot),files=files,preprocessing=DESCRIPTION)
    path=Path(destination)/'hubert_checkpoint_manifest.json'
    immutable(path,descriptor)
    os.environ['SHAOXING_HUBERT_DESCRIPTOR']=str(path.resolve())
    return path


def runtime():
    import torch
    from transformers import AutoModel
    descriptor=read(os.environ['SHAOXING_HUBERT_DESCRIPTOR']); key=signature(descriptor)
    cached=sys.modules.get('_shaoxing_verified_hubert_runtime')
    if cached is not None and cached.key==key:return cached
    if not torch.cuda.is_available():raise RuntimeError('Use the original HuBERT CUDA environment (float16)')
    snapshot=Path(descriptor['snapshot'])
    for name,digest in descriptor['files'].items():
        if sha(snapshot/name)!=digest:raise ValueError('Checkpoint contents changed: '+name)
    model=AutoModel.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=True,
                                   torch_dtype=torch.float16).to('cuda').eval().requires_grad_(False)
    cached=SimpleNamespace(key=key,model=model,torch=torch)
    sys.modules['_shaoxing_verified_hubert_runtime']=cached
    return cached


def encode_ecg(signal_mv_500hz):
    rt=runtime();torch=rt.torch
    with torch.inference_mode():
        values=torch.from_numpy(hubert_input(np.asarray(signal_mv_500hz)[None])).to('cuda',dtype=torch.float16)
        output=rt.model(input_values=values)
        hidden=output.last_hidden_state if hasattr(output,'last_hidden_state') else output[0]
        pooled=hidden.mean(1) if hidden.ndim==3 else hidden
        return pooled.float().cpu().numpy()[0]


def verify_bank(out, sensor_path, destination, probes=10, *, symile_root):
    # source_path is a WFDB provenance identifier, not the saved Symile tensor.
    if type(probes) is not int or probes < 1: raise ValueError('At least one verification probe required')
    folder=Path(symile_root)/'data_npy'/'train'
    tensor_path=folder/'ecg_train.npy'; admission_path=folder/'hadm_id_train.npy'
    for path in (tensor_path,admission_path):
        if not path.is_file(): raise FileNotFoundError('Missing original Symile array: '+str(path))
    arrays=np.load(tensor_path,mmap_mode='r',allow_pickle=False)
    admissions=np.load(admission_path,mmap_mode='r',allow_pickle=False).reshape(-1)
    if len(arrays)!=len(admissions) or arrays.shape[1:] not in ((1,5000,12),(5000,12)):
        raise ValueError('Original Symile ECG shape/admission length mismatch')
    def uid(value):
        value=str(value)
        return value[:-2] if value.endswith('.0') and value[:-2].isdigit() else value

    fingerprint=dict(bank_sha256=sha(Path(out)/'records.json'),sensor_sources_sha256=sha(sensor_path),
                     checkpoint_descriptor_sha256=sha(os.environ['SHAOXING_HUBERT_DESCRIPTOR']),
                     adapter_sha256=sha(__file__),probes=probes,
                     tensor_path=str(tensor_path.resolve()),tensor_sha256=sha(tensor_path),
                     admission_array_sha256=sha(admission_path))
    audit_path=Path(destination)/'hubert_bank_equivalence.json'
    if audit_path.is_file():
        saved=read(audit_path)
        if saved.get('fingerprint')==fingerprint and saved.get('passed') is True:return saved
    refs=sorted([r for r in read(Path(out)/'records.json') if r['modality']=='ecg' and r['split']=='train'],key=lambda r:r['id'])
    sources={r['id']:r for r in read(sensor_path)}
    if not refs:raise ValueError('No training ECG references')
    indices=np.random.default_rng(43).choice(len(refs),min(probes,len(refs)),replace=False)
    checks=[]
    for i in indices:
        row=refs[i];source=sources[row['id']];index=int(source['source_index'])
        if (source['split']!='train' or not 0<=index<len(admissions)
                or uid(source['admission_id'])!=uid(row['admission_id'])
                or uid(admissions[index])!=uid(row['admission_id'])):
            raise ValueError('Symile tensor/admission identity mismatch: '+row['id'])
        raw=np.asarray(arrays[index])
        if raw.shape==(1,5000,12):raw=raw[0]
        actual=encode_ecg(raw);expected=np.asarray(row['embedding'],dtype=np.float32)
        same=actual.shape==expected.shape and np.isfinite(actual).all()
        rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12)) if same else None
        cosine=float(np.dot(actual,expected)/max(np.linalg.norm(actual)*np.linalg.norm(expected),1e-12)) if same else None
        passed=bool(same and rel<=0.005 and cosine>=0.99999)
        checks.append(dict(id=row['id'],relative_l2=rel,cosine=cosine,passed=passed))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,fingerprint=fingerprint,
                thresholds=dict(max_relative_l2=0.005,min_cosine=0.99999),
                bank_sha256=sha(Path(out)/'records.json'),sensor_sources_sha256=sha(sensor_path),
                checkpoint_descriptor_sha256=sha(os.environ['SHAOXING_HUBERT_DESCRIPTOR']))
    atomic_json(Path(destination)/'hubert_bank_equivalence.json',result)
    if not result['passed']:raise ValueError('Original bank not reproduced; inspect hubert_bank_equivalence.json. Do not bypass this check.')
    return result
