"""Read-only, resumable validity audit; never alters the frozen cohort."""
from pathlib import Path
from collections import Counter
import numpy as np
from report_study import read, sha, signature, atomic_json
from ecg_external_generation import budget
import shaoxing_external as external


def diagnose(row):
    result=dict(id=row['id'],record=row['record'],status='ok',error=None)
    try:
        external.waveform(row)
        return result
    except (ValueError, OSError, IndexError, TypeError, OverflowError) as exc:
        result.update(status='invalid',error=str(exc),error_type=type(exc).__name__)
    # Read diagnostic data only; do not impute or resample invalid signals.
    try:
        import wfdb
        rec=wfdb.rdrecord(row['record'],physical=True)
        x=np.asarray(rec.p_signal)
        result.update(shape=list(x.shape),sampling_hz=float(rec.fs),sig_len=int(rec.sig_len),
                      leads=list(rec.sig_name),units=list(rec.units),
                      nan_count=int(np.isnan(x).sum()),infinity_count=int(np.isinf(x).sum()),
                      all_zero=bool(x.size and np.isfinite(x).all() and not np.any(x)))
        if x.ndim==2:
            result['nonfinite_by_lead']={str(name):int((~np.isfinite(x[:,j])).sum()) for j,name in enumerate(rec.sig_name)}
            result['all_zero_leads']=[str(name) for j,name in enumerate(rec.sig_name) if np.isfinite(x[:,j]).all() and not np.any(x[:,j])]
        result['adc_gain']=[None if v is None else float(v) for v in rec.adc_gain]
        result['baseline']=[None if v is None else int(v) for v in rec.baseline]
    except Exception as exc:
        result['diagnostic_read_error']=str(exc)
    return result


def audit(cohort,max_run_hours=1.):
    cohort=Path(cohort);rows=read(cohort/'manifest.json')
    cp=read(cohort/'cohort_protocol.json')
    if cp['manifest_sha256']!=signature(rows):raise ValueError('Frozen cohort manifest changed')
    code=signature(dict(audit=sha(__file__),reader=sha(external.__file__)))
    folder=cohort/'waveform_audit';folder.mkdir(exist_ok=True)
    entries=[];reason=None
    with budget(max_run_hours,folder/'STOP') as stop:
        for row in rows:
            reason=stop()
            if reason:break
            try:
                actual=dict(header=sha(row['record']+'.hea'),waveform=sha(row['waveform_file']))
            except OSError as exc:
                entries.append(dict(id=row['id'],record=row['record'],status='invalid',error=str(exc)))
                continue
            key=signature(dict(row=row,actual=actual,code=code))
            path=folder/(key+'.json')
            if path.exists():entry=read(path)
            else:
                entry=diagnose(row);atomic_json(path,entry)
            entries.append(entry)
    failures=[r for r in entries if r['status']!='ok']
    result=dict(complete=len(entries)==len(rows),total=len(rows),checked=len(entries),
                valid=len(entries)-len(failures),invalid=len(failures),not_checked=len(rows)-len(entries),
                stop_reason=reason,failures=failures,reason_counts=dict(Counter(r['error'] for r in failures)),
                manifest_sha256=signature(rows),code_sha256=code,cohort_modified=False)
    atomic_json(cohort/'waveform_validation_audit.json',result)
    return result
