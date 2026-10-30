"""Meaningful target/identity tampering and inference-boundary tests; no GPU needed."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nbformat

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP.parent)); sys.path.insert(0, str(EXP))
from common import atomic, read, sha, signature, code_hashes
from test_experiment02 import fixture, build_ready, GenerationConfig, Tok
from extensions.leakage_controls import audit_packets, guarded_generate, graph_from_training


def ready(tmp):
    cfg = fixture(Path(tmp)); cfg['external_sample_size'] = 2; out = Path(cfg['frozen_out'])
    atomic(out / 'graph.json', graph_from_training(read(out / 'records.json')))
    build_ready(cfg)
    return cfg


def paths(cfg):
    run = Path(cfg['run_dir']); qid = read(run / 'queries/internal_val/manifest.json')[0]['id']
    return run, qid, run / 'retrieval/internal_val' / (qid + '.json'), run / 'packets/internal_val' / (qid + '.json')


def resign(cfg, retrieval_path=None, packet_path=None):
    """Even edited manifests cannot authorize changed clinical evidence."""
    run = Path(cfg['run_dir'])
    if retrieval_path:
        stage = run / 'stages/packets_internal_val.json'; v = read(stage)
        qid = read(retrieval_path)['query_id']; v['inputs']['retrieval'][qid] = sha(retrieval_path); atomic(stage, v)
        folder = run / 'packets/internal_val'
        for p in folder.glob('*.json'):
            if p.name == 'manifest.json': continue
            packet = read(p); packet['protocol_signature'] = signature(v); atomic(p, packet)
        m = read(folder / 'manifest.json'); m['protocol_signature'] = signature(v); atomic(folder / 'manifest.json', m)
    mpath = run / 'packets/internal_val/manifest.json'; m = read(mpath)
    for q in m['query_ids']: m['packets'][q] = sha(mpath.parent / (q + '.json'))
    atomic(mpath, m)


class LeakageTests(unittest.TestCase):
    def audit(self, cfg):
        with patch('inputs.validate_frozen_inputs', return_value={}):
            return audit_packets(cfg, max_run_hours=1.)

    def test_clean_audit_target_independence_and_unchanged_frozen_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ready(tmp); before = code_hashes(); run, qid, r, p = paths(cfg)
            hashes = {str(f): sha(f) for f in (run / 'protocol.json', r, p)}
            a = self.audit(cfg); self.assertTrue(a['passed']); self.assertFalse(a['target_labels_used'])
            Path(cfg['frozen_out'], 'targets.json').write_text('not even JSON; must not be read by audit')
            self.assertEqual(a, self.audit(cfg))
            self.assertEqual(code_hashes(), before)
            self.assertEqual(hashes, {str(f): sha(f) for f in (run / 'protocol.json', r, p)})

    def test_graph_cannot_use_heldout_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ready(tmp); graphpath = Path(cfg['frozen_out']) / 'graph.json'
            graph = read(graphpath); graph['training_patients'].append('20'); atomic(graphpath, graph)
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                self.audit(cfg)
            self.assertFalse(read(Path(cfg['run_dir']) / 'audits/leakage_controls/internal_val/summary.json')['passed'])

    def test_reverse_anchor_patient_and_unknown_id_fail_even_after_resigning(self):
        for kind in ('reverse', 'unknown', 'forward'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                cfg = ready(tmp); run, qid, rpath, packet = paths(cfg); r = read(rpath)
                if kind == 'reverse':
                    a = r['anchors'][0]; r['candidates'][0]['reverse_admissions'][0].update(admission_id=a['id'], patient_id=a['patient_id'])
                elif kind == 'unknown': r['candidates'][0]['id'] = 'UNAPPROVED_REFERENCE'
                else: r['anchors'][0]['patient_id'] = read(run / 'queries/internal_val/manifest.json')[0]['patient_id']
                atomic(rpath, r); resign(cfg, retrieval_path=rpath)
                result = self.audit(cfg); self.assertFalse(result['passed']); self.assertEqual(result['violating_queries'], 1)

    def test_prompt_labels_changed_labs_and_conflict_injection_fail(self):
        for kind in ('target', 'labs', 'conflict', 'relationship'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                cfg = ready(tmp); run, qid, rpath, p = paths(cfg); packet = read(p)
                if kind == 'target': packet['cases'][0]['ground_truth'] = 'atrial fibrillation'
                elif kind == 'labs': packet['cases'][1]['references'][0]['labs'][0]['name'] = 'query ground truth diagnosis'
                elif kind == 'conflict':
                    r = read(rpath); r['conflicts'].append(dict(source=r['anchors'][0]['id'], modality='ecg', concept='heldout_answer'))
                    atomic(rpath, r); resign(cfg, retrieval_path=rpath)
                    packet = read(p); packet['cases'][1]['conflicts'] = r['conflicts']
                else: packet['cases'][3]['relationships'].append(dict(ecg_concept='heldout_answer', reference_id='c2'))
                atomic(p, packet); resign(cfg, packet_path=p)
                self.assertFalse(self.audit(cfg)['passed'])

    def test_guarded_generation_resume_and_live_tampering_stops_before_inference(self):
        from generation import folder_for
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ready(tmp); run, qid, rpath, p = paths(cfg)
            image = Path(tmp) / 'query.png'; image.write_bytes(b'image')
            runtime = dict(model=SimpleNamespace(generation_config=GenerationConfig()), tokenizer=Tok(), family='llama',
                model_id='meta-llama/Llama-3.2-11B-Vision-Instruct', experiment_model_key='llama_base', revision='fixed', enable_thinking=False)
            with patch('inputs.validate_frozen_inputs', return_value={}), patch('queries.image_provider', return_value=lambda q: (str(image), 'tensor')), \
                 patch('ecg_external_generation.infer', return_value='Atrial fibrillation is present.') as infer:
                a = guarded_generate(cfg, runtime, max_new=5, max_run_hours=1.)
                self.assertEqual(a['ok'], 5)
                b = guarded_generate(cfg, runtime, max_new=0, max_run_hours=1.)
                self.assertEqual(b['ok'], 10); self.assertEqual(infer.call_count, 10)
                c = guarded_generate(cfg, runtime, max_new=0, max_run_hours=1.)
                self.assertEqual(c['processed_this_call'], 0)
            checks = folder_for(cfg, 'llama_base', False, 384) / 'leakage_guard_checks.jsonl'
            self.assertTrue(checks.is_file())
            # Native generation must pass the exact approved case immediately before inference.
            def malicious_runner(config, loaded, **kw):
                import generation
                changed = copy.deepcopy(read(p)['cases'][0]); changed['ground_truth'] = 'answer'
                generation.format_prompt(changed, Tok(), config)
                raise AssertionError('Gate should already have stopped')
            with patch('inputs.validate_frozen_inputs', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Leakage generation gate stopped'):
                    guarded_generate(cfg, runtime, max_new=1, _runner=malicious_runner)
            def changed_input_runner(config, loaded, **kw):
                import generation
                case = read(p)['cases'][0]
                graphpath = Path(config['frozen_out']) / 'graph.json'
                graphpath.write_text(graphpath.read_text() + ' ')
                generation.format_prompt(case, Tok(), config)
            with patch('inputs.validate_frozen_inputs', return_value={}):
                with self.assertRaisesRegex(ValueError, 'Audited input changed'):
                    guarded_generate(cfg, runtime, max_new=1, _runner=changed_input_runner)

    def test_k_child_audit(self):
        from extensions.k_sweep import prepare_sweep
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ready(tmp); children = prepare_sweep(cfg, [1, 2, 3])
            for k, r in children.items():
                result = self.audit(r['config']); self.assertTrue(result['passed'], result)

    def test_external_failed_candidate_retained_without_query_targets(self):
        from queries import prepare_queries, extract_queries
        from retrieval import run_retrieval
        from packets import prepare_packets
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ready(tmp); cfg = dict(cfg, cohort='shaoxing')
            root = Path(cfg['shaoxing_cohort'])
            manifest = [dict(id='external' + str(i), patient_id='sx' + str(i), admission_id='sx' + str(i),
                study_id='s' + str(i), modality='ecg', split='external', record='wfdb' + str(i),
                header_sha256='h', waveform_sha256='w', waveform_file='file') for i in range(3)]
            atomic(root / 'manifest.json', manifest)
            atomic(root / 'cohort_protocol.json', dict(manifest_sha256=signature(manifest)))
            fp = dict(manifest_sha256=signature(manifest), bank_records_sha256=sha(Path(cfg['frozen_out']) / 'records.json'))
            atomic(root / 'features/protocol.json', fp)
            for r in manifest:
                atomic(root / 'features' / (r['id'] + '.json'), dict(id=r['id'], embedding=[1, .3],
                    embedding_space='ecg_space', protocol_sha256=signature(fp)))
            prepare_queries(cfg)
            image = Path(tmp) / 'query'; image.write_bytes(b'image')
            with patch('queries.image_provider', return_value=lambda q: (str(image), 'tensor')), \
                 patch('sensor_candidate_generation.infer', side_effect=['Atrial fibrillation.', ValueError('truncated')]), \
                 patch('inputs.validate_frozen_inputs', return_value={}):
                extract_queries(cfg, dict(modality='ecg', revision='fixed'), max_new=0)
                run_retrieval(cfg); prepare_packets(cfg)
                result = audit_packets(cfg)
            self.assertTrue(result['passed'], result)
            self.assertEqual(result['checked_queries'], 2)  # Failed extraction is not a dropped query.

    def test_stop_file_and_notebooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ready(tmp); dest = Path(cfg['run_dir']) / 'audits/leakage_controls/internal_val'
            dest.mkdir(parents=True); (dest / 'STOP').touch()
            with self.assertRaisesRegex(ValueError, 'stopped before input verification'): self.audit(cfg)
        for name in ('BR03A_audit_retrieval_and_prompt_leakage.ipynb', 'BR04A_generate_reports_with_leakage_gate.ipynb', 'BR06A_guarded_k_sensitivity.ipynb'):
            p = EXP / name; n = nbformat.read(p, as_version=4); nbformat.validate(n)
            for c in n.cells:
                if c.cell_type == 'code': compile(c.source, str(p), 'exec')


if __name__ == '__main__': unittest.main()
