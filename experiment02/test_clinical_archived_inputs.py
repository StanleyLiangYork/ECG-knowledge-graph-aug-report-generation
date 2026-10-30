"""Clinical review must copy exact historical inputs without rendering or waiving hashes."""
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from common import atomic, read, sha, signature, code_hashes
from clinical_review.archived_inputs import prepare_from_reconstruction
from clinical_review.workflow import save_reference, release_reports
from extensions import rendering_audit
import test_clinical_review_extension as fixture_tests


class ArchivedClinicalInputTests(unittest.TestCase):
    def setUp(self):
        fixture_tests.ClinicalReviewTests.setUp(self)
        # The base fixture has dummy image bytes; replace them with a real PNG
        # and update only the synthetic report journals before freezing them.
        Image.new('RGB', (12, 8), (40, 50, 60)).save(self.image)
        for folder in self.folders.values():
            path = folder / 'reports.jsonl'
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            for r in rows:
                r['sensor_sha256'] = sha(self.image)
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.archive = self.root / 'historical_modern'
        run = Path(self.cfg['run_dir'])
        query_manifest = run / 'queries/internal_val/manifest.json'
        packet = run / 'packets/internal_val/manifest.json'
        atomic(query_manifest, self.queries)
        sources = {m: dict(folder=str(f.resolve()), protocol=sha(f/'protocol.json'),
            reports=sha(f/'reports.jsonl')) for m, f in self.folders.items()}
        protocol = dict(cohort='internal_val', helper_sha256=sha(rendering_audit.__file__),
            packet_manifest_sha256=sha(packet), query_manifest_sha256=sha(query_manifest), sources=sources,
            expected_sources={q['id']:dict(tensor='tensor', png=sha(self.image)) for q in self.queries})
        atomic(self.archive / 'protocol.json', protocol)
        atomic(self.archive / 'status.json', dict(complete=True, total=3, reproduced=3,
            protocol_sha256=sha(self.archive / 'protocol.json')))
        (self.archive / 'images').mkdir()
        for q in self.queries:
            image = self.archive / 'images' / (q['id']+'.png')
            shutil.copyfile(self.image, image)
            atomic(self.archive / 'checks' / (q['id']+'.json'), dict(query_id=q['id'],
                protocol_sha256=signature(protocol), tensor_sha256='tensor', png_sha256=sha(image),
                pixels=rendering_audit.pixel_identity(image)))

    def prepare(self, **changes):
        settings = dict(n_queries=2, reviewer_id='clinician', reviewer_expertise='ECG interpretation fixture',
                        sample_size_justification='Calibration only')
        settings.update(changes)
        return prepare_from_reconstruction(self.cfg, self.folders, self.review, self.archive, **settings)

    def test_exact_archived_images_and_native_workflow_without_renderer(self):
        frozen = code_hashes()
        with patch('queries.image_provider', side_effect=AssertionError('Must never rerender')) as renderer:
            result = self.prepare()
            renderer.assert_not_called()
        self.assertEqual(result['reports'], 8)
        self.assertFalse(result['rendered_new_images'])
        protocol = read(self.review / 'PRIVATE/protocol.json')
        self.assertEqual(protocol['image_source']['mode'], 'verified_historical_png')
        self.assertEqual(code_hashes(), frozen)
        for case in read(self.review / 'PUBLIC/reference_cases.json'):
            self.assertEqual(sha(self.review/'PUBLIC'/case['image']), sha(self.image))
            save_reference(self.review, case['case_id'], reviewer_id='clinician', interpretable=True,
                independent_ecg_interpretation='Human fixture interpretation')
        self.assertEqual(release_reports(self.review)['reports'], 8)
        self.assertFalse(list(self.review.parent.glob('.historical_review_*')))

    def test_reconstruction_image_tampering_stops_before_package_creation(self):
        for image in (self.archive/'images').glob('*.png'):
            Image.new('RGB',(12,8),(200,50,60)).save(image)
        with self.assertRaisesRegex(ValueError, 'reconstruction proof failed'):
            self.prepare()
        self.assertFalse(self.review.exists())
        self.assertFalse(list(self.review.parent.glob('.historical_review_*')))

    def test_other_rendering_group_is_rejected(self):
        protocol = read(self.archive/'protocol.json')
        protocol['sources'].pop('pulse')
        atomic(self.archive/'protocol.json', protocol)
        status = read(self.archive/'status.json')
        status['protocol_sha256'] = sha(self.archive/'protocol.json')
        atomic(self.archive/'status.json', status)
        with self.assertRaisesRegex(ValueError, 'absent from this rendering group'):
            self.prepare()

    def test_incomplete_audit_and_changed_generation_are_rejected(self):
        status = read(self.archive/'status.json')
        atomic(self.archive/'status.json', dict(status, complete=False))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.prepare()
        atomic(self.archive/'status.json', status)
        path = self.folders['pulse']/'reports.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'generation folder/provenance differs'):
            self.prepare()

    def test_original_ratings_and_archive_provenance_remain_protected(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        source = next((self.archive/'checks').glob('*.json'))
        # All selected check files are pinned; change both unique-patient options
        # so at least one selected source must be affected.
        for source in (self.archive/'checks').glob('*.json'):
            source.write_text(source.read_text()+'\n')
        case = read(self.review/'PUBLIC/reference_cases.json')[0]
        with self.assertRaisesRegex(ValueError, 'Frozen review source changed'):
            save_reference(self.review, case['case_id'], reviewer_id='clinician', interpretable=True,
                independent_ecg_interpretation='Assessment after changed audit')

    def test_updated_notebook_compiles(self):
        notebook = read(Path(__file__).parent/'BR07_prepare_single_reviewer_clinical_study.ipynb')
        self.assertIn('prepare_from_reconstruction', ''.join(notebook['cells'][2]['source']))
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']), 'BR07', 'exec')


if __name__ == '__main__':
    unittest.main()
