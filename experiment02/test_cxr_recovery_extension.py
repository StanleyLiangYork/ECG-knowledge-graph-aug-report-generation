import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nbformat
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP.parent))
from common import read,atomic,sha,signature
from test_experiment02 import fixture,GenerationConfig,Tok
from inputs import preflight,symile_bundles,eligible_cxr
from encoders import extract_cxr
from extensions.leakage_controls import graph_from_training,audit_packets
from extensions.cxr_recovery import recover,finalize_recovered_banks


def failed_fixture(tmp):
    cfg=fixture(Path(tmp)); out=Path(cfg['frozen_out'])
    atomic(out/'graph.json',graph_from_training(read(out/'records.json')))
    preflight(cfg);symile_bundles(cfg);eligible_cxr(cfg)
    run=Path(cfg['run_dir'])
    atomic(run/'audits/rad_equivalence.json',dict(passed=True,symile_bank_sha256=sha(run/'banks/symile.json'),
        inventory_sha256=sha(run/'banks/cxr_inventory.json'),cxr_features_sha256=sha(cfg['cxr_features'])))
    runtime=dict(modality='cxr',revision='fixed',model=SimpleNamespace(generation_config=GenerationConfig()),tokenizer=Tok())
    with patch('encoders.version',return_value='test'),patch('sensor_candidate_generation.infer',side_effect=[ValueError('token limit'),ValueError('Repetition QC failed'),'Edema.','Edema.']):
        extract_cxr(cfg,runtime,max_new=0)
    return cfg,runtime


class RecoveryTests(unittest.TestCase):
    def test_signed_recovery_resume_finalization_and_leakage_gate(self):
        from queries import prepare_queries
        from retrieval import run_retrieval
        from packets import prepare_packets
        with tempfile.TemporaryDirectory() as tmp,patch('inputs.validate_frozen_inputs',return_value={}):
            cfg,runtime=failed_fixture(tmp);run=Path(cfg['run_dir']);original=sha(run/'extraction/cxr_candidates.jsonl')
            with patch('extensions.cxr_recovery.version',return_value='test'),patch('sensor_candidate_generation.infer',return_value='Pulmonary edema is present.') as infer:
                a=recover(cfg,runtime,max_new=1);self.assertEqual(a['combined_ok'],3)
                b=recover(cfg,runtime,max_new=0);self.assertEqual(b['combined_ok'],4)
                self.assertEqual(recover(cfg,runtime,max_new=0)['processed_this_call'],0)
                self.assertEqual(infer.call_count,2)
            result=finalize_recovered_banks(cfg);self.assertTrue(result['ready'])
            self.assertEqual(original,sha(run/'extraction/cxr_candidates.jsonl'))
            self.assertEqual(sum('recovery_version' in r['extraction'] for r in read(run/'banks/mimic.json')),2)
            prepare_queries(cfg);run_retrieval(cfg);prepare_packets(cfg)
            self.assertTrue(audit_packets(cfg)['passed'])
            # Changed retry provenance invalidates the downstream gate, not just the finalizer.
            log=run/'extraction/recovery_t2048_r115_ng8_v1/cxr_candidates.jsonl'
            log.write_text(log.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'recovery provenance changed'):audit_packets(cfg)

    def test_failed_retries_cannot_finalize(self):
        with tempfile.TemporaryDirectory() as tmp,patch('inputs.validate_frozen_inputs',return_value={}):
            cfg,runtime=failed_fixture(tmp)
            with patch('extensions.cxr_recovery.version',return_value='test'),patch('sensor_candidate_generation.infer',side_effect=[ValueError('Output token budget reached'),'edema '*100]):
                result=recover(cfg,runtime,max_new=0)
            self.assertEqual(result['recovered'],0);self.assertEqual(len(result['failed']),2)
            with self.assertRaisesRegex(ValueError,'Recovery incomplete'):finalize_recovered_banks(cfg)
            self.assertFalse((Path(cfg['run_dir'])/'banks/mimic.json').exists())

    def test_notebook(self):
        p=EXP/'BR01C_recover_failed_cxr_candidates.ipynb';nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        for c in nb.cells:
            if c.cell_type=='code':compile(c.source,str(p),'exec')

if __name__=='__main__':unittest.main()
