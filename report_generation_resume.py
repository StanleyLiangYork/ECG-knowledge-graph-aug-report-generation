"""Operational stop/resume safeguards without changing the signed generator."""
import fcntl
import math
import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import patch
import report_generation as generation
from report_study import read, atomic_json, sha


class _StopRequested(BaseException):
    pass


def _snapshot(study, modality):
    """Read/repair the journal only while holding the existing writer lock."""
    study=Path(study)
    with (study/f'{modality}_reports.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest=_read_resume_log(study/f'{modality}_reports.jsonl',repair_tail=True)
        cases=[c for c in read(study/'generation_cases.json') if c['modality']==modality]
        expected={(c['query_id'],c['arm']) for c in cases}
        if set(latest)-expected:raise ValueError('Unexpected report IDs in generation journal')
        summary=dict(total=len(cases),ok=sum(r['status']=='ok' for r in latest.values()),
                     failed=sum(r['status']!='ok' for r in latest.values()),
                     not_attempted=len(cases)-len(latest))
        atomic_json(study/f'{modality}_generation_coverage.json',summary)
        return summary


# Keep the original reader so the temporary patch does not recurse.
_original_reader=generation.read_latest


def _read_resume_log(path, repair_tail=False):
    records=_original_reader(path,repair_tail=repair_tail)
    if repair_tail and path.exists() and path.stat().st_size:
        # A killed writer can leave valid JSON without its terminating newline.
        # Preserve that completed record and separate it from the next append.
        with path.open('r+b') as f:
            f.seek(-1,os.SEEK_END)
            if f.read(1)!=b'\n':
                f.seek(0,os.SEEK_END);f.write(b'\n')
            f.flush();os.fsync(f.fileno())
    return records


def generate_resumable(study,tasks,arrays,runtime,limit=0,retry_errors=False,
                       max_run_hours=30.0,stop_file=None):
    """Stop between reports, retaining the original protocol and per-report fsync.

    max_run_hours counts this invocation, not the scheduler allocation age.
    SIGTERM/SIGUSR1 request a stop after the current report. SIGKILL cannot
    be handled. A manual KeyboardInterrupt leaves the in-flight report pending.
    """
    if not isinstance(max_run_hours,(int,float)) or not math.isfinite(max_run_hours) or max_run_hours<=0:
        raise ValueError('max_run_hours must be a positive finite number below remaining allocation time')
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Run from the notebook main thread for signal handling')
    study=Path(study);modality=runtime['modality']
    stop_file=Path(stop_file) if stop_file is not None else study/f'STOP_{modality}'
    started=time.monotonic();deadline=started+max_run_hours*3600
    requested={'reason':None};handlers={}
    original_infer=generation.sensor.infer
    original_materialize=generation.sensor.materialize
    before=_snapshot(study,modality)
    def receive(signum,frame):
        requested['reason']=signal.Signals(signum).name
    def guard():
        if requested['reason']:raise _StopRequested(requested['reason'])
        if stop_file.exists():raise _StopRequested('stop_file')
        if time.monotonic()>=deadline:raise _StopRequested('time_budget')
    def materialize(*args,**kwargs):
        guard();return original_materialize(*args,**kwargs)
    def infer(*args,**kwargs):
        guard();return original_infer(*args,**kwargs)
    reason='completed_or_report_limit'
    try:
        for name in ('SIGTERM','SIGUSR1'):
            sig=getattr(signal,name,None)
            if sig is not None:
                handlers[sig]=signal.getsignal(sig);signal.signal(sig,receive)
        with patch.object(generation.sensor,'materialize',side_effect=materialize), \
             patch.object(generation.sensor,'infer',side_effect=infer), \
             patch.object(generation,'read_latest',side_effect=_read_resume_log):
            try:
                guard()
                summary=generation.generate(study,tasks,arrays,runtime,limit=limit,retry_errors=retry_errors)
            except _StopRequested as exc:
                reason=str(exc);summary=None
            except KeyboardInterrupt:
                reason='keyboard_interrupt';summary=None
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    # Existing generate() has released its writer lock and restored model config.
    coverage=_snapshot(study,modality)
    operational=dict(coverage,processed_this_call=(summary or {}).get('processed_this_call'),
        stop_reason=requested['reason'] or reason,elapsed_hours=(time.monotonic()-started)/3600,
        max_run_hours=max_run_hours,stop_file=str(stop_file),coverage_at_start=before,
        resume_helper_sha256=sha(Path(__file__)),
        restart_instruction='Reload the same environment/checkpoint and rerun notebook 09; saved successful reports are skipped.')
    atomic_json(study/f'{modality}_resume_status.json',operational)
    print('Saved reports:',coverage,'Stop reason:',operational['stop_reason'],flush=True)
    return operational
