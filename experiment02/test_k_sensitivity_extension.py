import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nbformat

EXP=Path(__file__).resolve().parent
sys.path.insert(0,str(EXP));sys.path.insert(0,str(EXP.parent))
from test_experiment02 import fixture,build_ready
from common import read,sha,signature,assert_config,code_hashes
from extensions.k_sweep import prepare_sweep,select,summarize_sweep

class SweepTests(unittest.TestCase):
    def test_shared_pools_and_distinct_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));build_ready(cfg);root=Path(cfg['run_dir'])
            original_protocol=sha(root/'protocol.json');code_before=code_hashes()
            result=prepare_sweep(cfg,[1,2,3,4,5]);again=prepare_sweep(cfg,[1,2,3,4,5])
            self.assertEqual(sha(root/'protocol.json'),original_protocol)
            self.assertEqual(code_before,code_hashes());self.assertEqual(result,again)
            queries=read(root/'queries/internal_val/manifest.json')
            for k,r in result.items():
                child=r['config'];assert_config(child)
                self.assertEqual(child['context_tokens'],cfg['context_tokens'])
                self.assertEqual(child['k_anchors'],cfg['k_anchors'])
                run=Path(child['run_dir']);self.assertTrue((run/'banks').is_symlink())
                for q in queries:
                    source=read(root/'retrieval/internal_val'/(q['id']+'.json'))
                    copied=read(run/'retrieval/internal_val'/(q['id']+'.json'))
                    self.assertEqual(source['candidates'],copied['candidates'])
                    self.assertEqual(source['anchors'],copied['anchors'])
                    lookup={c['id']:c for c in copied['candidates']}
                    for arm,ids in copied['selected'].items():
                        if arm=='sensor_only':self.assertEqual(ids,[]);continue
                        self.assertEqual(len(ids),min(k,len({c['patient_id'] for c in source['candidates']})))
                        self.assertEqual(len(ids),len({lookup[rid]['patient_id'] for rid in ids}))
                    packets=read(run/'packets/internal_val'/(q['id']+'.json'))
                    for case in packets['cases']:
                        selected=[r for r in case['references'] if r['source_type']=='other_patient_mimic_cxr']
                        self.assertEqual(len(selected),len(copied['selected'][case['arm']]))
            # The parent's protected code and original retrieval selection remain unchanged.
            self.assertEqual(code_hashes(),code_before)
            self.assertEqual(read(root/'protocol.json')['config']['final_cxr'],2)
    def test_patient_dedup_and_ties(self):
        pool=[dict(id='a',patient_id='1',score=.9),dict(id='b',patient_id='1',score=.8),dict(id='c',patient_id='2',score=.8)]
        self.assertEqual(select(pool,'score',2),['a','c'])
    def test_sweep_stops_after_interruption(self):
        from extensions.k_sweep import generate_sweep
        with tempfile.TemporaryDirectory() as tmp:
            configs={k:dict(run_dir=str(Path(tmp)/f'k_{k}')) for k in (1,2)}
            with patch('generation.generate',return_value={'stop_reason':'time_budget'}) as call:
                result=generate_sweep(configs,{},max_run_hours=1.)
            self.assertEqual(list(result),[1]);self.assertEqual(call.call_count,1)
    def test_common_query_aggregation(self):
        from common import atomic,append,csv_save
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);configs={};folders={};destinations={}
            arms=['sensor_only','forward_pool','bidirectional_pool','bidirectional_graph','bidirectional_shuffled_graph']
            for k in (1,2):
                root=base/f'k_{k}';configs[k]=dict(run_dir=str(root),cohort='internal_val',seed=43)
                folder=root/'generator';folder.mkdir(parents=True);folders[k]={'pulse':folder}
                dest=root/'eval';destinations[k]=dest
                atomic(root/'packets/internal_val/manifest.json',dict(query_ids=['q1','q2'],arms=arms))
                scored=[]
                for q in ('q1','q2'):
                    atomic(root/'packets/internal_val'/(q+'.json'),dict(cases=[dict(arm=a,references=[dict(source_type='other_patient_mimic_cxr',record_id='c1')]) for a in arms]))
                    for a in arms:
                        failed=k==2 and q=='q2' and a=='sensor_only'
                        append(folder/'reports.jsonl',dict(query_id=q,arm=a,status='error' if failed else 'ok',
                            prompt='Reference 1 (other_patient_mimic_cxr; source=c1)',evidence_audit={'context_tokens':25}))
                        if not failed:scored.append(dict(query_id=q,patient_id=q,model_family='pulse',arm=a,f1=.5,tp=1,fp=0,fn=2,tn=0))
                csv_save(dest/'per_report.csv',scored)
                atomic(dest/'results.json',dict(complete=True,matched_queries=2 if k==1 else 1,
                    summary=[dict(model_family='pulse',arm='bidirectional_graph',micro_f1=.5)],
                    comparisons=[dict(model_family='pulse',contrast='bidirectional_graph_minus_bidirectional_pool',metric='f1',subset='all_arm_complete',mean=0.,ci95=[0.,0.],p_value=1.)]))
            output=base/'summary';result=summarize_sweep(configs,folders,destinations,output)
            report=read(output/'k_sensitivity.json')
            self.assertEqual(report['common_queries'],['q1'])
            self.assertTrue(all(r['queries']==1 for r in report['common_query_summary']))
            self.assertTrue(all(r['mean_prompted_cxr']==1 for r in report['context_exposure']))
            self.assertFalse(result['automatic_k_selection'])

    def test_notebook(self):
        p=EXP/'BR06_cxr_reference_k_sensitivity.ipynb';n=nbformat.read(p,as_version=4);nbformat.validate(n)
        for c in n.cells:
            if c.cell_type=='code':compile(c.source,str(p),'exec')

if __name__=='__main__':unittest.main()
