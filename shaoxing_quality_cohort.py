"""Derive a new cohort by audited signal-quality exclusion, never imputation."""
from pathlib import Path
import shutil
import tempfile
from report_study import read, sha, signature, atomic_json
from shaoxing_waveform_audit import audit


def create_clean_cohort(source,destination,max_run_hours=1.):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or destination.exists():
        raise FileExistsError('Choose a new, empty cohort destination; original cohort is preserved')
    result=audit(source,max_run_hours=max_run_hours)
    if not result['complete']:raise RuntimeError('Audit incomplete; rerun to resume before creating a cohort')
    rows=read(source/'manifest.json');targets=read(source/'targets.json');parent=read(source/'cohort_protocol.json')
    if signature(rows)!=parent['manifest_sha256'] or signature(targets)!=parent['targets_sha256']:
        raise ValueError('Original cohort hashes changed')
    ids={r['id'] for r in rows};truth={r['id']:r for r in targets}
    if len(ids)!=len(rows) or len(truth)!=len(targets) or ids!=set(truth):raise ValueError('Cohort identity mismatch')
    failures=result['failures'];excluded={r['id'] for r in failures}
    if len(excluded)!=len(failures) or excluded-ids:raise ValueError('Audit identity mismatch')
    # Only known signal-validity failures may be excluded. Changed/missing files must be investigated.
    if any(r.get('error')!='Invalid ECG' for r in failures):
        raise ValueError('Audit includes source/decoder errors beyond invalid signals; investigate before exclusion')
    kept=[r for r in rows if r['id'] not in excluded]
    if not kept:raise ValueError('No valid ECG records remain')
    filtered=[truth[r['id']] for r in kept]
    policy=dict(parent_cohort=str(source),parent_protocol_sha256=sha(source/'cohort_protocol.json'),
                audit_sha256=sha(source/'waveform_validation_audit.json'),source_records=len(rows),
                retained_records=len(kept),excluded_records=len(excluded),exclusions=failures,
                rule='Exclude all records failing finite/nonzero/shape ECG validation; no interpolation or replacement sampling',
                labels_used=False,helper_sha256=sha(__file__))
    protocol=dict(parent,manifest_sha256=signature(kept),targets_sha256=signature(filtered),
                  waveform_quality_control=policy)
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.shaoxing_qc_',dir=destination.parent))
    try:
        atomic_json(stage/'manifest.json',kept);atomic_json(stage/'targets.json',filtered)
        atomic_json(stage/'cohort_protocol.json',protocol);atomic_json(stage/'quality_exclusions.json',policy)
        # Preserve upstream exclusion audits alongside the new QC stage.
        for name in ('leakage_audit.json','header_validation_audit.json','duplicate_waveforms.json'):
            if (source/name).is_file():shutil.copyfile(source/name,stage/name)
        atomic_json(stage/'label_audit.json',dict(records=len(kept),
            known_positive_decisions=sum(v==1 for t in filtered for v in t['findings'].values()),
            negative_policy='inherited unchanged from parent cohort',reference_reports_available=False))
        # Matching rows + unchanged source hashes mean successful raw-waveform audit cache entries remain usable.
        if (source/'waveform_audit').is_dir():
            (stage/'waveform_audit').mkdir()
            for p in (source/'waveform_audit').glob('*.json'):
                entry=read(p)
                if entry.get('status')=='ok' and entry.get('id') in ids-excluded:
                    shutil.copyfile(p,stage/'waveform_audit'/p.name)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
    return dict(source_records=len(rows),excluded=len(excluded),retained=len(kept),destination=str(destination),
                features_copied=False,instruction='Use this EXTERNAL directory for notebooks 13–17; encode the new cohort.')
