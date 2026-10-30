"""Exercise BR08A notebook cells with isolated review-package fixtures."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from common import read, sha, signature, immutable, lock, atomic
from clinical_review import workflow as w

ROOT = Path(__file__).parent


class ImportNotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        source = ROOT / 'human_review/PUBLIC'
        shutil.copytree(source / 'images', self.folder / 'PUBLIC/images')
        for name in ('reports.json', 'report_ratings_completed.json', 'reference_cases.json'):
            shutil.copyfile(source / name, self.folder / 'PUBLIC' / name)
        reports = read(self.folder / 'PUBLIC/reports.json')
        atomic(self.folder / 'PRIVATE/protocol.json', {'reviewer_id': 'ECG_reviewer_01', 'stage': 'calibration'})
        atomic(self.folder / 'PRIVATE/references_submitted.json', [
            dict(case_id=cid, interpretable=True) for cid in sorted({r['case_id'] for r in reports})])
        self.protocol = read(self.folder / 'PRIVATE/protocol.json')
        self.patcher = patch.object(w, '_check', return_value=self.protocol)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.ns = dict(Path=Path, json=json, hashlib=hashlib, os=os, datetime=datetime,
                       timezone=timezone, read=read, sha=sha, signature=signature,
                       immutable=immutable, lock=lock, _check=w._check,
                       RATING_FIELDS=w.RATING_FIELDS, validate_rating=w.validate_rating,
                       needs_adjudication=w.needs_adjudication, save_rating=w.save_rating,
                       progress=w.progress, REVIEW_DIR=self.folder,
                       INPUT_PATH=self.folder / 'PUBLIC/report_ratings_completed.json')
        cells = read(ROOT / 'BR08A_import_completed_report_ratings.ipynb')['cells']
        validation = ''.join(cells[3]['source']).split('\nprotocol, records, existing, payload, source_sha, already_frozen =')[0]
        importer = ''.join(cells[5]['source']).split('\nprint(import_completed_ratings())')[0]
        exec(compile(validation, 'BR08A validation', 'exec'), self.ns)
        exec(compile(importer, 'BR08A import', 'exec'), self.ns)

    def edit(self, fn):
        p = self.ns['INPUT_PATH']; d = read(p); fn(d); atomic(p, d)

    def test_complete_import_resume_and_freeze(self):
        result = self.ns['import_completed_ratings']()
        self.assertEqual(result['new_ratings_imported'], 40)
        self.assertEqual(self.ns['import_completed_ratings']()['new_ratings_imported'], 0)
        result = w.prepare_adjudication(self.folder)
        expected = sum(w.needs_adjudication(r) for r in read(self.ns['INPUT_PATH'])['ratings'])
        self.assertEqual(result['adjudications_required'], expected)
        self.assertTrue(self.ns['import_completed_ratings']()['initial_ratings_already_frozen'])
        self.assertEqual(sha(self.ns['INPUT_PATH']), sha(self.folder / 'PRIVATE/imported_report_ratings.json'))

    def test_invalid_batch_has_no_writes(self):
        self.edit(lambda d: d['ratings'][-1].update(overall_quality_1_to_5=None))
        with self.assertRaises(ValueError): self.ns['import_completed_ratings']()
        self.assertFalse((self.folder / 'PRIVATE/imported_report_ratings.json').exists())
        self.assertFalse((self.folder / 'PUBLIC/rating_drafts').exists())

    def test_conflicting_draft_is_preserved(self):
        row = read(self.ns['INPUT_PATH'])['ratings'][0]
        rating = {k:row[k] for k in w.RATING_FIELDS}
        rating['comments'] += ' Existing different judgment.'
        w.save_rating(self.folder, row['review_id'], **rating)
        p = self.folder / 'PUBLIC/rating_drafts' / (row['review_id'] + '.json')
        before = sha(p)
        with self.assertRaises(ValueError): self.ns['import_completed_ratings']()
        self.assertEqual(sha(p), before)
        self.assertFalse((self.folder / 'PRIVATE/imported_report_ratings.json').exists())

    def test_report_identity_mismatch(self):
        p = self.folder / 'PUBLIC/reports.json'; reports = read(p)
        reports[0]['generated_report'] += ' Alteration.'; atomic(p, reports)
        with self.assertRaises(ValueError): self.ns['validate_completed_ratings']()

    def test_image_identity_mismatch(self):
        p = next((self.folder / 'PUBLIC/images').glob('*.png'))
        p.write_bytes(p.read_bytes() + b'changed')
        with self.assertRaises(ValueError): self.ns['validate_completed_ratings']()

    def test_resume_after_interruption(self):
        original = w.save_rating
        calls = []
        def interrupted(*args, **kwargs):
            if len(calls) == 3: raise KeyboardInterrupt()
            calls.append(args[1]); return original(*args, **kwargs)
        self.ns['save_rating'] = interrupted
        with self.assertRaises(KeyboardInterrupt): self.ns['import_completed_ratings']()
        self.ns['save_rating'] = original
        result = self.ns['import_completed_ratings']()
        self.assertEqual(result['new_ratings_imported'], 37)
        self.assertEqual(result['identical_existing_drafts'], 3)


if __name__ == '__main__':
    unittest.main()
