"""CPU tests of human-entry gates, adjudication and paired endpoint calculations."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import atomic, sha, signature, read, code_hashes
from clinical_review.workflow import (prepare, save_reference, release_reports,
    save_rating, prepare_adjudication, save_adjudication, analyze, progress)


def rating(error=0, agreement=None, uninterpretable=False):
    return dict(reviewer_id='clinician', agreement=agreement or ('disagree' if error else 'agree'),
        clinically_significant_error_count=error, clinically_significant_omission_count=0,
        minor_error_count=0, overall_quality_1_to_5=2 if error else 5,
        uninterpretable_image=uninterpretable,
        discrepancy_details='Incorrect rhythm assertion' if error else ('Uncertain image' if uninterpretable or agreement == 'uncertain' else ''),
        comments='')


class ClinicalReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = dict(run_dir=str(self.root / 'run'), cohort='internal_val')
        run = Path(self.cfg['run_dir'])
        packet = run / 'packets/internal_val/manifest.json'
        atomic(run / 'protocol.json', {'frozen': True})
        self.queries = [dict(id='q1', patient_id='p1'), dict(id='q2', patient_id='p1'), dict(id='q3', patient_id='p2')]
        self.arms = ('bidirectional_pool', 'bidirectional_graph')
        atomic(packet, dict(query_ids=[r['id'] for r in self.queries], arms=list(self.arms),
            packets={q['id']: 'packet-' + q['id'] for q in self.queries}))
        self.image = self.root / 'ecg.png'
        self.image.write_bytes(b'fixture image bytes; no real data')
        self.folders = {}
        for model in ('pulse', 'qwen38'):
            folder = self.root / model
            self.folders[model] = folder
            protocol = dict(packets_manifest_sha256=sha(packet))
            atomic(folder / 'protocol.json', protocol)
            rows = []
            for q in self.queries:
                for arm in self.arms:
                    text = model + '-' + arm + '-' + q['id']
                    rows.append(dict(query_id=q['id'], arm=arm, status='ok', text=text,
                        prompt='ECG image', prompt_sha256=signature('ECG image'),
                        response_sha256=signature(text), protocol_signature=signature(protocol),
                        packet_sha256='packet-' + q['id'], sensor_sha256=sha(self.image),
                        source_tensor_sha256='tensor', patient_id=q['patient_id']))
            (folder / 'reports.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.review = self.root / 'clinical'
        self.patchers = [patch('common.assert_config'),
            patch('queries.load_queries', return_value=self.queries),
            patch('queries.image_provider', return_value=lambda q: (str(self.image), 'tensor'))]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def prepare(self):
        return prepare(self.cfg, self.folders, self.review, n_queries=2, reviewer_id='clinician',
            reviewer_expertise='ECG interpretation fixture', sample_size_justification='Calibration only')

    def references(self):
        for r in read(self.review / 'PUBLIC/reference_cases.json'):
            save_reference(self.review, r['case_id'], reviewer_id='clinician',
                interpretable=True, independent_ecg_interpretation='Independent rhythm assessment')
        release_reports(self.review)

    def fill_initial(self, with_errors=True):
        links = {r['review_id']: r for r in read(self.review / 'PRIVATE/linkage.json')}
        for item in read(self.review / 'PUBLIC/reports.json'):
            error = int(with_errors and links[item['review_id']]['arm'] == 'bidirectional_pool')
            save_rating(self.review, item['review_id'], **rating(error))

    def test_blinding_reference_gate_and_resume(self):
        before = code_hashes()
        summary = self.prepare()
        self.assertEqual(before, code_hashes())
        self.assertEqual(summary['cases'], 2)
        self.assertEqual(summary['reports'], 8)
        self.assertFalse((self.review / 'PUBLIC/reports.json').exists())
        cases = read(self.review / 'PUBLIC/reference_cases.json')
        self.assertTrue(all(not {'model', 'arm', 'query_id', 'generated_report'} & set(r) for r in cases))
        with self.assertRaisesRegex(ValueError, 'Missing independent'):
            release_reports(self.review)
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.references()
        release_reports(self.review)  # restart is idempotent
        with self.assertRaisesRegex(ValueError, 'already frozen'):
            save_reference(self.review, cases[0]['case_id'], reviewer_id='clinician',
                interpretable=True, independent_ecg_interpretation='Post-report rewrite')
        self.fill_initial()
        self.assertEqual(progress(self.review)['initial_ratings_saved'], 8)
        self.assertEqual(len({r['patient_id'] for r in read(self.review / 'PRIVATE/linkage.json')}), 2)
        self.assertTrue(all(not {'model', 'arm', 'query_id', 'patient_id'} & set(r)
            for r in read(self.review / 'PUBLIC/reports.json')))

    def test_adjudication_required_and_confirmed_disagreement_remains_error(self):
        self.prepare()
        self.references()
        with self.assertRaisesRegex(ValueError, 'Unassessed'):
            prepare_adjudication(self.review)
        self.fill_initial()
        info = prepare_adjudication(self.review)
        self.assertEqual(info['adjudications_required'], 4)
        prepare_adjudication(self.review)
        with self.assertRaisesRegex(ValueError, 'Initial ratings frozen'):
            save_rating(self.review, read(self.review / 'PUBLIC/reports.json')[0]['review_id'], **rating())
        with self.assertRaisesRegex(ValueError, 'Adjudication not completed'):
            analyze(self.review, self.root / 'analysis', repeats=100)
        initial_sha = sha(self.review / 'PRIVATE/initial_submitted.json')
        for item in read(self.review / 'PUBLIC/adjudication_queue.json'):
            save_adjudication(self.review, item['review_id'], reviewer_id='clinician',
                decision='retain_initial', reason='Rhythm error confirmed on ECG', evidence_reviewed='Query ECG rhythm strip')
        self.assertEqual(initial_sha, sha(self.review / 'PRIVATE/initial_submitted.json'))
        result = analyze(self.review, self.root / 'analysis', repeats=100)
        differences = [r for r in result['comparisons'] if r['metric'] == 'major_error']
        self.assertTrue(all(r['estimate'] == -1.0 and r['paired_patients'] == 2 for r in differences))
        self.assertEqual(result['adjudication_decisions']['retain_initial'], 4)
        again = analyze(self.review, self.root / 'analysis', repeats=100)
        self.assertEqual(result, again)

    def test_unresolved_and_revision_keep_initial_and_denominators(self):
        self.prepare()
        self.references()
        self.fill_initial()
        prepare_adjudication(self.review)
        queue = read(self.review / 'PUBLIC/adjudication_queue.json')
        for i, item in enumerate(queue):
            decision = 'unresolved' if i == 0 else 'revise_initial'
            save_adjudication(self.review, item['review_id'], reviewer_id='clinician',
                decision=decision, reason='Rechecked original interpretation',
                evidence_reviewed='Query ECG rhythm and lead quality',
                final_rating=None if i == 0 else rating())
        with self.assertRaisesRegex(ValueError, 'Protocol changed'):
            save_adjudication(self.review, queue[0]['review_id'], reviewer_id='clinician',
                decision='retain_initial', reason='Changed after submission', evidence_reviewed='ECG')
        result = analyze(self.review, self.root / 'analysis', repeats=100)
        self.assertEqual(sum(r['unresolved'] for r in result['summary'] if r['phase'] == 'adjudicated'), 1)
        self.assertTrue(any(r['major_error_upper_bound'] > r['major_error_lower_bound'] for r in result['summary']))
        self.assertTrue(all(r['major_error'] == 1 for r in result['summary'] if r['phase'] == 'initial' and r['arm'] == 'bidirectional_pool'))

    def test_schema_source_tampering_and_incomplete_generation(self):
        path = self.folders['pulse'] / 'reports.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['status'] = 'error'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'Finish selected reports'):
            self.prepare()
        rows[0]['status'] = 'ok'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.prepare()
        self.references()
        rid = read(self.review / 'PUBLIC/reports.json')[0]['review_id']
        bad = rating()
        bad['clinically_significant_error_count'] = None
        with self.assertRaisesRegex(ValueError, 'nonnegative integers'):
            save_rating(self.review, rid, **bad)
        self.image.write_bytes(b'changed after review freeze')
        # Image is copied into package; source report mutation is also detected independently.
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'Frozen review source changed'):
            save_rating(self.review, rid, **rating())

    def test_retained_uncertainty_never_becomes_a_correct_report(self):
        self.prepare()
        self.references()
        for item in read(self.review / 'PUBLIC/reports.json'):
            save_rating(self.review, item['review_id'], **rating(agreement='uncertain'))
        info = prepare_adjudication(self.review)
        self.assertEqual(info['adjudications_required'], 8)
        for item in read(self.review / 'PUBLIC/adjudication_queue.json'):
            save_adjudication(self.review, item['review_id'], reviewer_id='clinician',
                decision='retain_initial', reason='Still clinically uncertain', evidence_reviewed='Query ECG')
        result = analyze(self.review, self.root / 'analysis', repeats=100)
        self.assertTrue(all(r['evaluable'] == 0 and r['major_error'] is None for r in result['summary']))
        self.assertTrue(all(r['major_error_upper_bound'] == 1.0 for r in result['summary']))
        self.assertTrue(all(r['paired_patients'] == 0 for r in result['comparisons']))

    def test_no_discrepancies_needs_no_artificial_adjudication(self):
        self.prepare()
        self.references()
        self.fill_initial(with_errors=False)
        self.assertEqual(prepare_adjudication(self.review)['adjudications_required'], 0)
        result = analyze(self.review, self.root / 'analysis', repeats=100)
        self.assertEqual(result['adjudication_decisions'], {'not_required': 8})
        self.assertTrue(all(r['estimate'] == 0.0 for r in result['comparisons'] if r['metric'] == 'major_error'))

    def test_notebooks(self):
        import nbformat
        folder = Path(__file__).parent
        for pattern in ('BR07_*.ipynb', 'BR08_*.ipynb', 'BR09_*.ipynb'):
            for p in folder.glob(pattern):
                notebook = nbformat.read(p, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == 'code':
                        compile(cell.source, str(p), 'exec')


if __name__ == '__main__':
    unittest.main()
