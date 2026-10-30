"""Pixel provenance acceptance requires exact historical reconstruction, not version guessing."""
import hashlib
import json
import sys
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from common import atomic, read, sha, signature, code_hashes
from extensions.rendering_audit import reproduce, compare, evaluate_pixel_equivalent


class RenderingAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = dict(project_root=str(ROOT.parent), cohort='internal_val',
            run_dir=str(self.root / 'run'), symile_root=str(self.root / 'symile'))
        self.run = Path(self.cfg['run_dir'])
        folder = Path(self.cfg['symile_root']) / 'data_npy/val'
        folder.mkdir(parents=True)
        self.raw = np.ones((2, 1, 5000, 12), dtype='float32') * .1
        np.save(folder / 'ecg_val.npy', self.raw)
        np.save(folder / 'hadm_id_val.npy', [101, 102])
        queries = [dict(id='q'+str(i), source_index=i, split='val', admission_id=str(101+i)) for i in range(2)]
        self.manifest_path = self.run / 'packets/internal_val/manifest.json'
        atomic(self.manifest_path, dict(query_ids=['q0', 'q1'], arms=['sensor_only'], packets={'q0': 'p0', 'q1': 'p1'}))
        atomic(self.run / 'queries/internal_val/manifest.json', queries)
        self.folders = {}
        self.mode = 'old'
        self.changed_pixels = False
        for mode in ('old', 'new'):
            image = self.root / (mode + '.png')
            self.write_image(image, mode)
            folder = self.root / mode
            self.folders[mode] = folder
            protocol = dict(packets_manifest_sha256=sha(self.manifest_path),
                libraries={name: ('fixture-renderer' if name == 'matplotlib' else version(name))
                           for name in ('numpy', 'Pillow', 'matplotlib')})
            atomic(folder / 'protocol.json', protocol)
            rows = []
            for i, raw in enumerate(self.raw):
                tensor = signature(dict(shape=list(raw.shape), dtype=str(raw.dtype),
                    bytes=hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()))
                rows.append(dict(query_id='q'+str(i), arm='sensor_only', status='ok',
                    protocol_signature=signature(protocol), prompt='prompt', prompt_sha256=signature('prompt'),
                    text='report', response_sha256=signature('report'), packet_sha256='p'+str(i),
                    source_tensor_sha256=tensor, sensor_sha256=sha(image)))
            (folder / 'reports.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.patchers = [patch('extensions.rendering_audit.assert_config'),
                        patch('extensions.rendering_audit.version', side_effect=lambda name:
                              'fixture-renderer' if name == 'matplotlib' else version(name)),
                        patch('sensor_candidate_generation.materialize', side_effect=self.render)]
        for p in self.patchers:
            p.start()
            self.addCleanup(p.stop)

    def write_image(self, path, mode):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('Software', mode)
        image = Image.new('RGBA', (8, 6), (20, 30, 40, 255))
        if self.changed_pixels and mode == 'new':
            image.putpixel((0, 0), (200, 30, 40, 255))
        image.save(path, pnginfo=metadata)

    def render(self, array, modality, path):
        self.write_image(path, self.mode)
        return sha(path)

    def render_groups(self):
        for mode in ('old', 'new'):
            self.mode = mode
            result = reproduce(self.cfg, {mode: self.folders[mode]}, self.root / ('audit_'+mode))
            self.assertTrue(result['complete'])

    def test_exact_metadata_only_reconstruction_resume_and_comparison(self):
        frozen = code_hashes()
        self.mode = 'old'
        result = reproduce(self.cfg, {'old': self.folders['old']}, self.root / 'audit_old', max_new=1)
        self.assertFalse(result['complete'])
        result = reproduce(self.cfg, {'old': self.folders['old']}, self.root / 'audit_old')
        self.assertEqual(result['processed_this_call'], 1)
        self.mode = 'new'
        reproduce(self.cfg, {'new': self.folders['new']}, self.root / 'audit_new')
        result = compare(self.root / 'audit_old', self.root / 'audit_new', self.root / 'comparison.json')
        self.assertTrue(result['passed'])
        self.assertEqual(result['identical_pixels'], 2)
        self.assertEqual(code_hashes(), frozen)

    def test_wrong_rendering_or_environment_stops(self):
        self.mode = 'new'
        with self.assertRaisesRegex(ValueError, 'Historical PNG bytes not reproduced'):
            reproduce(self.cfg, {'old': self.folders['old']}, self.root / 'wrong')
        self.assertTrue((self.root / 'wrong/first_reconstruction_failure.json').exists())
        protocol = read(self.folders['new'] / 'protocol.json')
        protocol['libraries']['matplotlib'] = 'different'
        atomic(self.folders['new'] / 'protocol.json', protocol)
        with self.assertRaisesRegex(ValueError, 'original rendering environment'):
            reproduce(self.cfg, {'new': self.folders['new']}, self.root / 'wrong_environment')

    def test_pixels_that_differ_cannot_use_combined_adapter(self):
        # Create a genuine historical rendering difference, with valid source signatures.
        self.changed_pixels = True
        self.write_image(self.root / 'new.png', 'new')
        path = self.folders['new'] / 'reports.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        for r in rows:
            r['sensor_sha256'] = sha(self.root / 'new.png')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.render_groups()
        result = compare(self.root / 'audit_old', self.root / 'audit_new', self.root / 'comparison.json')
        self.assertFalse(result['passed'])
        with patch('evaluation.evaluate') as native:
            with self.assertRaisesRegex(ValueError, 'different pixels'):
                evaluate_pixel_equivalent(self.cfg, self.folders, self.root / 'targets', self.root / 'eval',
                    self.root / 'audit_old', self.root / 'audit_new')
            native.assert_not_called()

    def test_combined_adapter_validates_originals_and_never_rewrites_journals(self):
        self.render_groups()
        hashes = {m:sha(f/'reports.jsonl') for m,f in self.folders.items()}
        def native(cfg, folders, targets, destination, **kwargs):
            from evaluation import load_results
            manifest = read(self.manifest_path)
            manifest['_file_sha256'] = sha(self.manifest_path)
            images = set()
            for folder in folders.values():
                images.update(r['sensor_sha256'] for r in load_results(folder, manifest).values())
            self.assertEqual(len(images), 1)
            return dict(complete=True)
        with patch('evaluation.evaluate', side_effect=native):
            result = evaluate_pixel_equivalent(self.cfg, self.folders, self.root/'targets', self.root/'eval',
                self.root/'audit_old', self.root/'audit_new')
        self.assertFalse(result['image_source_assessment']['original_journals_modified'])
        self.assertEqual(hashes, {m:sha(f/'reports.jsonl') for m,f in self.folders.items()})
        self.assertTrue((self.root/'eval/pixel_equivalence_protocol.json').exists())

    def test_changed_source_after_audit_is_rejected(self):
        self.render_groups()
        path = self.folders['old'] / 'reports.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'changed after reconstruction'):
            compare(self.root/'audit_old', self.root/'audit_new', self.root/'comparison.json')

    def test_notebook_compiles(self):
        n = read(ROOT/'BR05A_rendering_provenance_and_within_generator_evaluation.ipynb')
        for cell in n['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'BR05A', 'exec')


if __name__ == '__main__':
    unittest.main()
