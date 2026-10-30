import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nbformat
EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP.parent))
from common import read,sha,code_hashes
from test_leakage_controls_extension import ready
from test_experiment02 import GenerationConfig,Tok
from extensions.context_fit import prepare_context_fit,generate_context_fit
from extensions.leakage_controls import audit_packets


def runtime():
    return dict(model=SimpleNamespace(generation_config=GenerationConfig()),tokenizer=Tok(),family='pulse',
        modality='ecg',revision='fixed',context=1400,model_id='PULSE-ECG/PULSE-7B',experiment_model_key='pulse',enable_thinking=False)


def meter(loaded,image):
    return (lambda prompt:150+len(prompt)//5),dict(image_size=[4,4],visual_tokens=150,context=1400,revision='fixed',model='PULSE-ECG/PULSE-7B')


class ContextTests(unittest.TestCase):
    def test_calibration_preserves_source_and_all_case_evidence(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp,patch('inputs.validate_frozen_inputs',return_value={}):
            cfg=ready(tmp);source=Path(cfg['run_dir']);before=sha(source/'protocol.json');code=code_hashes()
            image=Path(tmp)/'query.png';Image.new('RGB',(4,4)).save(image)
            with patch('queries.image_provider',return_value=lambda q:(str(image),'tensor')),patch('extensions.context_fit.pulse_meter',side_effect=meter):
                fit=prepare_context_fit(cfg,runtime());again=prepare_context_fit(cfg,runtime())
            self.assertEqual(fit,again);self.assertEqual(before,sha(source/'protocol.json'));self.assertEqual(code,code_hashes())
            child=Path(fit['config']['run_dir']);self.assertTrue((child/'banks').is_symlink())
            self.assertTrue(audit_packets(fit['config'])['passed'])
            for qid in read(source/'packets/internal_val/manifest.json')['query_ids']:
                self.assertEqual(read(source/'packets/internal_val'/(qid+'.json'))['cases'],read(child/'packets/internal_val'/(qid+'.json'))['cases'])
            self.assertLessEqual(fit['maximum_input_tokens']+384+32,fit['model_context'])
            # Use native journal and leakage gate with the calibrated child.
            with patch('queries.image_provider',return_value=lambda q:(str(image),'tensor')),patch('ecg_external_generation.infer',return_value='Atrial fibrillation is present.') as infer:
                a=generate_context_fit(fit['config'],runtime(),max_new=5,max_run_hours=1.)
                b=generate_context_fit(fit['config'],runtime(),max_new=0,max_run_hours=1.)
                self.assertEqual(a['ok'],5);self.assertEqual(b['ok'],10);self.assertEqual(infer.call_count,10)
            Image.new('RGB',(8,4)).save(image)
            with patch('queries.image_provider',return_value=lambda q:(str(image),'tensor')),patch('ecg_external_generation.infer') as infer:
                with self.assertRaisesRegex(ValueError,'image size differs'):generate_context_fit(fit['config'],runtime(),max_new=1,max_run_hours=1.)
                infer.assert_not_called()

    def test_cannot_bypass_unfit_budget_or_recalibrate_test(self):
        with tempfile.TemporaryDirectory() as tmp,patch('inputs.validate_frozen_inputs',return_value={}):
            cfg=ready(tmp)
            def impossible(r,i):return lambda p:99999,dict(image_size=[4,4],visual_tokens=99999,context=1400,revision='fixed')
            with patch('queries.image_provider',return_value=lambda q:('image','tensor')),patch('extensions.context_fit.pulse_meter',side_effect=impossible):
                with self.assertRaisesRegex(ValueError,'No tested common caps fit'):prepare_context_fit(cfg,runtime())
            self.assertFalse((Path(cfg['run_dir'])/'context_fit').exists())

    def test_later_cohort_reuses_frozen_caps(self):
        from queries import prepare_queries
        from retrieval import run_retrieval
        from packets import prepare_packets
        with tempfile.TemporaryDirectory() as tmp,patch('inputs.validate_frozen_inputs',return_value={}):
            cfg=ready(tmp)
            with patch('queries.image_provider',return_value=lambda q:('image','tensor')),patch('extensions.context_fit.pulse_meter',side_effect=meter):
                fit=prepare_context_fit(cfg,runtime())
                later=dict(cfg,cohort='internal_test');prepare_queries(later);run_retrieval(later);prepare_packets(later)
                with self.assertRaisesRegex(ValueError,'Choose caps on internal_val'):prepare_context_fit(later,runtime())
                test_fit=prepare_context_fit(later,runtime(),frozen_context_config=fit['config_path'])
                self.assertEqual(fit['caps'],test_fit['caps'])
                self.assertTrue(audit_packets(test_fit['config'])['passed'])

    def test_notebook(self):
        p=EXP/'BR04B_pulse_context_fit_and_generation.ipynb';n=nbformat.read(p,as_version=4);nbformat.validate(n)
        for c in n.cells:
            if c.cell_type=='code':compile(c.source,str(p),'exec')

if __name__=='__main__':unittest.main()
