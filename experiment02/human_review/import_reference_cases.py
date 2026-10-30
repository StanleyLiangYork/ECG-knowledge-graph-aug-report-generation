"""Import offline clinician ECG interpretations without replacing a protected template."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from common import atomic, immutable, lock, read, sha
from clinical_review.workflow import _check, save_reference, progress

FIELDS = ('case_id', 'reviewer_id', 'interpretable', 'independent_ecg_interpretation', 'limitations')


def validate_batch(rows, templates, reviewer_id, public_dir):
    """Validate the WHOLE batch before writing any native reference draft."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError('Completed references must be a nonempty JSON list of case objects')
    expected = {r['case_id']: r for r in templates}
    ids = [r.get('case_id') for r in rows]
    if any(not isinstance(k, str) for k in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate case identifiers')
    if set(ids) != set(expected):
        raise ValueError('Completed references must match every case in THIS frozen package')
    records = []
    for row in rows:
        cid = row['case_id']
        if row.get('reviewer_id') != reviewer_id:
            raise ValueError('Reviewer identity differs from the frozen package: ' + cid)
        if type(row.get('interpretable')) is not bool:
            raise ValueError('Interpretability must be true or false: ' + cid)
        if not isinstance(row.get('independent_ecg_interpretation'), str) or not row['independent_ecg_interpretation'].strip():
            raise ValueError('Independent ECG interpretation is missing: ' + cid)
        if not isinstance(row.get('limitations'), str):
            raise ValueError('Limitations must be a string: ' + cid)
        if row.get('image') != expected[cid]['image']:
            raise ValueError('Case/image mapping differs from the package: ' + cid)
        image = Path(public_dir) / expected[cid]['image']
        if (not row.get('source_image_sha256') or not image.is_file()
                or sha(image) != row['source_image_sha256']):
            raise ValueError('Reviewed local image differs from the frozen package: ' + cid)
        records.append({key: row[key] for key in FIELDS})
    return records


def import_references(review_dir, completed_path):
    review_dir, completed_path = Path(review_dir), Path(completed_path)
    if not (review_dir/'PRIVATE/protocol.json').is_file():
        raise FileNotFoundError('Existing clinical review package not found: ' + str(review_dir))
    if not completed_path.is_file():
        raise FileNotFoundError('Completed reference file not found: ' + str(completed_path))
    # This validates native template/image/source signatures. Never repair them
    # by accepting a different baseline or rewriting assets.json.
    with lock(review_dir / 'reference_import.lock'):
        protocol = _check(review_dir)
        if (review_dir / 'PRIVATE/references_submitted.json').exists() or (review_dir / 'PUBLIC/reports.json').exists():
            raise ValueError('References already submitted/reports released. Preserve them; use adjudication for later assessment changes.')
        records = validate_batch(read(completed_path), read(review_dir/'PUBLIC/reference_cases.json'),
                                 protocol['reviewer_id'], review_dir/'PUBLIC')
        source_sha = sha(completed_path)
        snapshot = review_dir / 'PRIVATE/imported_reference_cases.json'
        if snapshot.exists() and sha(snapshot) != source_sha:
            raise ValueError('A different completed-reference file was previously imported; preserve it and review the discrepancy explicitly')
        # Do not silently overwrite clinician entries made in the notebook.
        existing = set()
        for row in records:
            path = review_dir / 'PUBLIC/reference_drafts' / (row['case_id']+'.json')
            if path.exists():
                saved = read(path)
                if any(saved.get(k) != row[k] for k in FIELDS):
                    raise ValueError('Existing clinician reference differs: ' + row['case_id'] + '. No drafts overwritten.')
                existing.add(row['case_id'])
        if not snapshot.exists():
            shutil.copyfile(completed_path, snapshot)
        for row in records:
            if row['case_id'] in existing:
                continue
            save_reference(review_dir, row['case_id'], reviewer_id=row['reviewer_id'],
                interpretable=row['interpretable'], independent_ecg_interpretation=row['independent_ecg_interpretation'],
                limitations=row['limitations'])
        log_path = review_dir / 'PRIVATE/reference_import.json'
        imported_at = read(log_path)['imported_at_utc'] if log_path.exists() else datetime.now(timezone.utc).isoformat()
        immutable(log_path, dict(imported_at_utc=imported_at, source_sha256=source_sha,
            cases=[r['case_id'] for r in records], reviewer_id=protocol['reviewer_id'],
            package_protocol_sha256=sha(review_dir/'PRIVATE/protocol.json'),
            importer_sha256=sha(__file__), image_identity_verified=True,
            protected_template_modified=False))
        result = progress(review_dir)
        print(result)
        print('Independent references imported. Continue to BR08 to freeze references and release reports.')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent/'PUBLIC/reference_cases_completed.json')
    parser.add_argument('--review-dir', type=Path, default=Path(
        '/path/to/repository/outputs/real_run_001/experiment02_v1/'
        'context_fit/v1/internal_val/clinical_review/internal_val/calibration_v1'))
    args = parser.parse_args()
    print('Completed reference file:', args.input)
    print('Existing clinical package:', args.review_dir)
    import_references(args.review_dir, args.input)


if __name__ == '__main__':
    main()
