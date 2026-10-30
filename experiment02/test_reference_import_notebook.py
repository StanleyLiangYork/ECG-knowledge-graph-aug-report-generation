"""Exercise BR07A's actual notebook cells without scripts or helper changes."""
import copy
import datetime
import hashlib
import json
import os
import unittest
from pathlib import Path

from common import read, sha, atomic, immutable, lock, code_hashes
from clinical_review.workflow import _check, save_reference, progress, release_reports
import test_clinical_review_extension as fixture_tests


class ReferenceImportNotebookTests(unittest.TestCase):
    def setUp(self):
        fixture_tests.ClinicalReviewTests.setUp(self)
        fixture_tests.ClinicalReviewTests.prepare(self)
        self.template_hash=sha(self.review/'PUBLIC/reference_cases.json')
        self.input=self.review/'PUBLIC/reference_cases_completed.json'
        self.rows=copy.deepcopy(read(self.review/'PUBLIC/reference_cases.json'))
        for row in self.rows:
            row.update(interpretable=True, independent_ecg_interpretation='Clinician fixture assessment',
                limitations='Normalized amplitude', source_image_sha256=sha(self.review/'PUBLIC'/row['image']))
        atomic(self.input,self.rows)
        self.nb=read(Path(__file__).parent/'BR07A_import_completed_reference_reviews.ipynb')
        self.scope=dict(Path=Path,json=json,hashlib=hashlib,os=os,
            datetime=datetime.datetime,timezone=datetime.timezone,
            read=read,sha=sha,immutable=immutable,lock=lock,_check=_check,
            save_reference=save_reference,progress=progress,
            REVIEW_DIR=self.review,INPUT_PATH=self.input)

    def run_cell(self,index):
        exec(''.join(self.nb['cells'][index]['source']),self.scope)

    def test_import_actual_cells_resume_and_native_release(self):
        frozen=code_hashes()
        for i in (3,5,6):self.run_cell(i)
        self.assertEqual(progress(self.review)['references_saved'],2)
        self.assertEqual(progress(self.review)['reports_released'],0)
        self.assertEqual(sha(self.review/'PUBLIC/reference_cases.json'),self.template_hash)
        self.assertEqual(sha(self.review/'PRIVATE/imported_reference_cases.json'),sha(self.input))
        drafts={p.name:sha(p) for p in (self.review/'PUBLIC/reference_drafts').glob('*.json')}
        for i in (3,5,6):self.run_cell(i)
        self.assertEqual(drafts,{p.name:sha(p) for p in (self.review/'PUBLIC/reference_drafts').glob('*.json')})
        self.assertEqual(frozen,code_hashes())
        self.assertEqual(release_reports(self.review)['reports'],8)
        with self.assertRaisesRegex(ValueError,'already submitted'):
            self.run_cell(3)

    def test_import_revalidates_after_preview_and_writes_nothing_for_invalid_batch(self):
        self.run_cell(3)
        self.rows[-1]['interpretable']=None
        atomic(self.input,self.rows)
        with self.assertRaisesRegex(ValueError,'interpretable must'):
            self.run_cell(5)
        self.assertFalse((self.review/'PUBLIC/reference_drafts').exists())
        self.assertFalse((self.review/'PRIVATE/imported_reference_cases.json').exists())

    def test_wrong_image_and_existing_conflict_cannot_overwrite(self):
        self.rows[-1]['source_image_sha256']='wrong'
        atomic(self.input,self.rows)
        with self.assertRaisesRegex(ValueError,'Locally reviewed image'):
            self.run_cell(3)
        self.rows[-1]['source_image_sha256']=sha(self.review/'PUBLIC'/self.rows[-1]['image'])
        atomic(self.input,self.rows)
        row=self.rows[0]
        save_reference(self.review,row['case_id'],reviewer_id='clinician',interpretable=True,
            independent_ecg_interpretation='Previous clinician assessment',limitations='')
        with self.assertRaisesRegex(ValueError,'Existing clinician assessment differs'):
            self.run_cell(3)

    def test_compilation_and_no_script_execution(self):
        for cell in self.nb['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source'])
                compile(source,'BR07A','exec')
                self.assertNotIn('%run',source)
                self.assertNotIn('subprocess',source)
                self.assertNotIn('import_reference_cases',source)


if __name__=='__main__':
    unittest.main()
