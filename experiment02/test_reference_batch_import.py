"""Offline reference imports preserve native templates and refuse case/image mismatches."""
import copy
import unittest

from common import atomic, read, sha, code_hashes
from human_review.import_reference_cases import import_references
from clinical_review.workflow import release_reports
import test_clinical_review_extension as fixture_tests


class ReferenceImportTests(unittest.TestCase):
    def setUp(self):
        fixture_tests.ClinicalReviewTests.setUp(self)
        fixture_tests.ClinicalReviewTests.prepare(self)
        self.template = self.review/'PUBLIC/reference_cases.json'
        self.original_hash = sha(self.template)
        self.rows = copy.deepcopy(read(self.template))
        for r in self.rows:
            r.update(interpretable=True, independent_ecg_interpretation='Human fixture interpretation',
                limitations='Normalized image', source_image_sha256=sha(self.review/'PUBLIC'/r['image']))
        self.input = self.root/'reference_cases_completed.json'
        atomic(self.input,self.rows)

    def test_import_idempotence_template_preservation_and_native_release(self):
        frozen=code_hashes()
        result=import_references(self.review,self.input)
        self.assertEqual(result['references_saved'],2)
        self.assertEqual(result['reports_released'],0)
        saved={p.name:sha(p) for p in (self.review/'PUBLIC/reference_drafts').glob('*.json')}
        self.assertEqual(import_references(self.review,self.input),result)
        self.assertEqual(saved,{p.name:sha(p) for p in (self.review/'PUBLIC/reference_drafts').glob('*.json')})
        self.assertEqual(sha(self.template),self.original_hash)
        self.assertEqual(code_hashes(),frozen)
        self.assertEqual(release_reports(self.review)['reports'],8)
        with self.assertRaisesRegex(ValueError,'already submitted'):
            import_references(self.review,self.input)

    def test_entire_batch_validation_precedes_writes(self):
        self.rows[-1]['interpretable']=None
        atomic(self.input,self.rows)
        with self.assertRaisesRegex(ValueError,'Interpretability'):
            import_references(self.review,self.input)
        self.assertFalse((self.review/'PUBLIC/reference_drafts').exists())
        self.assertEqual(sha(self.template),self.original_hash)

    def test_wrong_local_image_and_existing_assessment_are_not_overwritten(self):
        self.rows[-1]['source_image_sha256']='wrong'
        atomic(self.input,self.rows)
        with self.assertRaisesRegex(ValueError,'local image differs'):
            import_references(self.review,self.input)
        self.assertFalse((self.review/'PUBLIC/reference_drafts').exists())
        self.rows[-1]['source_image_sha256']=sha(self.review/'PUBLIC'/self.rows[-1]['image'])
        atomic(self.input,self.rows)
        row=self.rows[0]
        atomic(self.review/'PUBLIC/reference_drafts'/(row['case_id']+'.json'),
            {k:('Different human interpretation' if k=='independent_ecg_interpretation' else row[k])
             for k in ('case_id','reviewer_id','interpretable','independent_ecg_interpretation','limitations')})
        with self.assertRaisesRegex(ValueError,'Existing clinician reference differs'):
            import_references(self.review,self.input)

    def test_missing_package_does_not_create_a_new_review_directory(self):
        missing=self.root/'nonexistent_review'
        with self.assertRaises(FileNotFoundError):
            import_references(missing,self.input)
        self.assertFalse(missing.exists())


if __name__=='__main__':
    unittest.main()
