"""Experiment 02 IO, immutable stage protocols and allocation-safe checkpoints."""
import csv
import fcntl
import hashlib
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

ARMS=('sensor_only','forward_pool','bidirectional_pool','bidirectional_graph','bidirectional_shuffled_graph')
MODEL_IDS={'pulse':'PULSE-ECG/PULSE-7B','llama_ecg':'convaiinnovations/ECG-Instruct-Llama-3.2-11B-Vision',
           'llama_base':'meta-llama/Llama-3.2-11B-Vision-Instruct','qwen38':'Qwen/Qwen3.8-27B'}


def read(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def immutable(path,value):
    p=Path(path)
    if p.exists():
        if read(p)!=value:raise ValueError('Protocol changed; choose a new run directory: '+str(p))
    else:atomic(p,value)


def lines(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def uid(value):
    if value is None:raise ValueError('Missing identity')
    s=str(value).strip()
    if not s or s.lower() in ('nan','none'):raise ValueError('Missing identity')
    return s[:-2] if s.endswith('.0') and s[:-2].isdigit() else s


def patient(value):
    s=uid(value)
    # Only normalize explicit MIMIC p<digits>; never guess numeric IDs from arbitrary strings.
    return s[1:] if s.startswith('p') and s[1:].isdigit() else s


def unique(rows,key):
    d={uid(r[key]):r for r in rows}
    if len(d)!=len(rows):raise ValueError('Duplicate '+key)
    return d


def load_config(path):
    cfg=read(path);cfg['project_root']=str(Path(cfg['project_root']).expanduser().resolve())
    for k in ('run_dir','frozen_out','symile_root','artifact_root','prepared_manifest','sensor_sources','lab_table','cxr_manifest','cxr_features','cxr_status','cxr_train_partition','shaoxing_cohort'):
        p=Path(cfg[k]).expanduser()
        cfg[k]=str((p if p.is_absolute() else Path(path).resolve().parent/p).resolve())
    return cfg


def assert_config(cfg):
    frozen=read(Path(cfg['run_dir'])/'protocol.json')
    if frozen['config']!={k:v for k,v in cfg.items() if k!='cohort'}:
        raise ValueError('Configuration changed; use a new run_dir and rerun BR00')
    if frozen['code']!=code_hashes():raise ValueError('Code changed; use a new run_dir')


def parent_setup(cfg):
    import sys
    root=cfg['project_root']
    if root not in sys.path:sys.path.insert(0,root)
    return Path(root)


def code_hashes():
    folder=Path(__file__).parent
    return {p.name:sha(p) for p in sorted(folder.glob('*.py')) if not p.name.startswith('test_')}


def freeze_stage(folder,stage,payload):
    folder=Path(folder);freeze=read(folder/'protocol.json')
    if freeze['code']!=code_hashes():raise ValueError('Experiment code changed after BR00; use a new output directory')
    value=dict(parent_protocol=signature(freeze),stage=stage,inputs=payload)
    immutable(folder/'stages'/(stage+'.json'),value)
    return signature(value)


@contextmanager
def lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


@contextmanager
def budget(hours,stop_file):
    if not isinstance(hours,(float,int)) or not 0<hours<float('inf'):raise ValueError('Positive finite run hours required')
    start=time.monotonic();requested=[];saved={}
    def stop():
        if requested:return requested[0]
        if Path(stop_file).exists():return 'stop_file'
        if time.monotonic()-start>=hours*3600:return 'time_budget'
        return None
    try:
        for name in ('SIGTERM','SIGUSR1'):
            sig=getattr(signal,name,None)
            if sig is not None:
                saved[sig]=signal.getsignal(sig)
                signal.signal(sig,lambda n,f:requested.append(signal.Signals(n).name))
        yield stop
    finally:
        for sig,handler in saved.items():signal.signal(sig,handler)


def latest(path,repair=False):
    p=Path(path);result={}
    if not p.exists():return result
    data=p.read_bytes();parts=data.splitlines(keepends=True);offset=0
    for i,line in enumerate(parts):
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if repair and i==len(parts)-1 and not line.endswith(b'\n'):
                backup=p.with_name(p.name+'.interrupted.'+str(time.time_ns()));backup.write_bytes(line)
                with p.open('r+b') as f:f.truncate(offset);f.flush();os.fsync(f.fileno())
                break
            raise ValueError('Malformed journal: '+str(p))
        result[(r['query_id'],r['arm'])]=r;offset+=len(line)
    return result


def append(path,row):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    # A valid JSON object without the final newline is also a recoverable interrupted write.
    if p.exists() and p.stat().st_size:
        with p.open('rb') as f:f.seek(-1,2);tail=f.read(1)
        if tail!=b'\n':
            with p.open('ab') as f:f.write(b'\n');f.flush();os.fsync(f.fileno())
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def csv_save(path,rows):
    if not rows:return
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    tmp.replace(p)
