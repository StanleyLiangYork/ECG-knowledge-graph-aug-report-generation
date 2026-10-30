"""Test actual BR08B notebook functions using isolated review-package fixtures."""
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from common import read, sha, signature, immutable, lock, atomic
from clinical_review import workflow as w

ROOT = Path(__file__).parent


@unittest.skipUnless((ROOT / 'human_review/PUBLIC/images').is_dir(),
                     'Private historical review fixtures are not distributed')
class AdjudicationImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        source = ROOT / 'human_review/PUBLIC'
        shutil.copytree(source / 'images', self.folder / 'PUBLIC/images')
        for name in ('reports.json', 'reference_cases.json', 'adjudication_queue.json',
                     'adjudications_completed.json'):
            shutil.copyfile(source / name, self.folder / 'PUBLIC' / name)
        self.queue = read(self.folder / 'PUBLIC/adjudication_queue.json')
        initial = {r['review_id']:r for r in read(source / 'report_ratings_completed.json')['ratings']}
        initial.update({r['review_id']:r['initial_rating'] for r in self.queue})
        refs = {r['case_id']:r['independent_interpretation'] for r in self.queue}
        atomic(self.folder / 'PRIVATE/initial_submitted.json', list(initial.values()))
        atomic(self.folder / 'PRIVATE/references_submitted.json', list(refs.values()))
        self.protocol = dict(reviewer_id='ECG_reviewer_01', stage='calibration')
        atomic(self.folder / 'PRIVATE/protocol.json', self.protocol)
        self.freeze()
        patcher = patch.object(w, '_check', return_value=self.protocol)
        patcher.start(); self.addCleanup(patcher.stop)
        self.ns = dict(Path=Path, json=json, hashlib=hashlib, os=os, Counter=Counter,
            datetime=datetime, timezone=timezone, read=read, sha=sha, signature=signature,
            immutable=immutable, lock=lock, _check=w._check, RATING_FIELDS=w.RATING_FIELDS,
            validate_rating=w.validate_rating, save_adjudication=w.save_adjudication,
            progress=w.progress, REVIEW_DIR=self.folder,
            INPUT_PATH=self.folder / 'PUBLIC/adjudications_completed.json')
        cells = read(ROOT / 'BR08B_import_completed_adjudications.ipynb')['cells']
        validation = ''.join(cells[3]['source']).split('\nprotocol, records, existing, payload, source_sha =')[0]
        importer = ''.join(cells[5]['source']).split('\nprint(import_completed_adjudications())')[0]
        exec(compile(validation, 'BR08B validation', 'exec'), self.ns)
        exec(compile(importer, 'BR08B import', 'exec'), self.ns)

    def freeze(self):
        atomic(self.folder / 'PRIVATE/adjudication_protocol.json', dict(
            initial_sha256=sha(self.folder / 'PRIVATE/initial_submitted.json'),
            references_sha256=sha(self.folder / 'PRIVATE/references_submitted.json'),
            queue_sha256=sha(self.folder / 'PUBLIC/adjudication_queue.json'),
            reviewer_id='ECG_reviewer_01'))

    def edit(self, fn):
        p = self.ns['INPUT_PATH']; data = read(p); fn(data); atomic(p, data)

    def test_import_and_idempotence(self):
        self.assertEqual(self.ns['import_completed_adjudications']()['new_adjudications_imported'], 39)
        self.assertEqual(self.ns['import_completed_adjudications']()['new_adjudications_imported'], 0)
        for item in self.queue:
            saved = read(self.folder / 'PUBLIC/adjudications' / (item['review_id']+'.json'))
            original = {k:item['initial_rating'][k] for k in w.RATING_FIELDS}
            self.assertEqual(saved['final_rating'], original)
        self.assertEqual(sha(self.ns['INPUT_PATH']), sha(self.folder / 'PRIVATE/imported_adjudications.json'))

    def test_invalid_last_item_causes_no_writes(self):
        self.edit(lambda d:d['adjudications'][-1].update(reason=''))
        with self.assertRaises(ValueError): self.ns['import_completed_adjudications']()
        self.assertFalse((self.folder / 'PUBLIC/adjudications').exists())
        self.assertFalse((self.folder / 'PRIVATE/imported_adjudications.json').exists())

    def test_revision_and_unresolved(self):
        def change(data):
            row = data['adjudications'][0]
            rating = {k:self.queue[0]['initial_rating'][k] for k in w.RATING_FIELDS}
            rating['comments'] += ' Second-pass clarification.'
            row.update(decision='revise_initial', final_rating=rating)
            data['adjudications'][1].update(decision='unresolved', final_rating=None)
        self.edit(change)
        result = self.ns['import_completed_adjudications']()
        self.assertEqual(result['decisions']['revise_initial'], 1)
        self.assertEqual(result['decisions']['unresolved'], 1)
        rid = self.queue[1]['review_id']
        self.assertIsNone(read(self.folder/'PUBLIC/adjudications'/(rid+'.json'))['final_rating'])

    def test_conflicting_existing_is_not_overwritten(self):
        r = read(self.ns['INPUT_PATH'])['adjudications'][0]
        w.save_adjudication(self.folder, r['review_id'], reviewer_id=r['reviewer_id'],
            decision=r['decision'], reason='Different original rationale.',
            evidence_reviewed=r['evidence_reviewed'], final_rating=r['final_rating'])
        path = self.folder/'PUBLIC/adjudications'/(r['review_id']+'.json'); before=sha(path)
        with self.assertRaises(ValueError): self.ns['import_completed_adjudications']()
        self.assertEqual(sha(path), before)
        self.assertFalse((self.folder/'PRIVATE/imported_adjudications.json').exists())

    def test_queue_tamper_detected(self):
        path = self.folder/'PUBLIC/adjudication_queue.json'
        rows = read(path); rows[0]['generated_report'] += ' Alteration.'; atomic(path, rows)
        self.freeze()  # Even if native protocol was rewritten, local binding still rejects.
        with self.assertRaises(ValueError): self.ns['validate_completed_adjudications']()

    def test_initial_tamper_detected(self):
        path = self.folder/'PRIVATE/initial_submitted.json'
        rows=read(path); rows[0]['comments'] += ' Alteration.'; atomic(path, rows)
        with self.assertRaises(ValueError): self.ns['validate_completed_adjudications']()

    def test_wrong_upload_initial_signature(self):
        self.edit(lambda d:d['adjudications'][0].update(initial_rating_sha256='wrong'))
        with self.assertRaises(ValueError): self.ns['import_completed_adjudications']()

    def test_interruption_resumes(self):
        calls=[]
        def interrupted(*args, **kwargs):
            if len(calls)==3: raise KeyboardInterrupt()
            calls.append(args[1]); return w.save_adjudication(*args, **kwargs)
        self.ns['save_adjudication']=interrupted
        with self.assertRaises(KeyboardInterrupt): self.ns['import_completed_adjudications']()
        self.ns['save_adjudication']=w.save_adjudication
        result=self.ns['import_completed_adjudications']()
        self.assertEqual(result['new_adjudications_imported'],36)
        self.assertEqual(result['identical_existing_adjudications'],3)


if __name__ == '__main__': unittest.main()
