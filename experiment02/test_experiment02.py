"""CPU integration and leakage-control tests; fixtures never enter real run outputs."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

EXP=Path(__file__).resolve().parent;PARENT=EXP.parent
sys.path.insert(0,str(PARENT));sys.path.insert(0,str(EXP))
from common import *
from inputs import preflight,symile_bundles,eligible_cxr
from encoders import extract_cxr,finalize_banks
from queries import prepare_queries
from retrieval import Index,normalize,run_retrieval,infer_query,shuffled_graph
from packets import prepare_packets,format_prompt
from generation import generate,folder_for
from evaluation import evaluate,holm,paired
from review import prepare_review

class GenerationConfig(SimpleNamespace):
    def to_dict(self):return vars(self).copy()
class Tok:
    def encode(self,text,**kw):return text.split()


def fixture(root):
    from PIL import Image
    cfg=read(EXP/'config.example.json')
    for k in ('frozen_out','symile_root','artifact_root','run_dir','shaoxing_cohort'):cfg[k]=str(root/k)
    cfg.update(project_root=str(PARENT),lab_table=str(EXP/'lab_code_table.json'),cohort='internal_val',
               k_anchors=2,reverse_m=2,cxr_per_anchor=3,search_backend='exact')
    out=Path(cfg['frozen_out']);out.mkdir();symile=Path(cfg['symile_root']);symile.mkdir();Path(cfg['artifact_root']).mkdir()
    table=read(cfg['lab_table']);cols=table['columns'];means={c['percentile_key']:.5 for c in cols};atomic(symile/'labs_means.json',means)
    records=[];sources=[];manifest=[];rng=np.random.default_rng(2)
    for split,n,offset in [('train',6,10),('val',2,20),('test',1,30)]:
        folder=symile/'data_npy'/split;folder.mkdir(parents=True)
        hadm=np.array([100+offset+i for i in range(n)]);np.save(folder/f'hadm_id_{split}.npy',hadm)
        np.save(folder/f'ecg_{split}.npy',rng.normal(size=(n,1,5000,12)).astype('float32')*.1)
        np.save(folder/f'labs_percentiles_{split}.npy',np.full((n,50),.5,dtype='float32'))
        np.save(folder/f'labs_missingness_{split}.npy',np.ones((n,50),dtype='int8'))
        csv_rows=[]
        for i in range(n):
            pid=str(offset+i);aid=str(hadm[i]);vec=[1.,.15*(i+1)]
            raw=dict(subject_id=pid,hadm_id=aid)
            for c in cols:raw[c['itemid']]='1';raw[c['percentile_key']]='0.5'
            csv_rows.append(raw)
            for modality in ('ecg','cxr'):
                rid=aid+'_'+modality
                findings=[dict(concept='atrial_fibrillation' if modality=='ecg' else 'Edema',polarity='present')]
                row=dict(id=rid,patient_id=pid,admission_id=aid,modality=modality,split=split,
                         embedding=vec,embedding_space=modality+'_space',candidates=findings)
                records.append(row);manifest.append({k:v for k,v in row.items() if k not in ('embedding','embedding_space','candidates')})
                sources.append(dict(id=rid,patient_id=pid,admission_id=aid,split=split,source_index=i,study_id=aid+'s',source_path='files/provenance-only'))
        with (symile/(split+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    atomic(out/'records.json',records);atomic(out/'freeze.json',dict(synthetic=False));atomic(out/'targets.json',[dict(id=r['id'],findings={'atrial_fibrillation':1,'bradycardia':0}) for r in records if r['modality']=='ecg'])
    graph=dict(vocabulary=['ecg:atrial_fibrillation:present','cxr:Edema:present','ecg:bradycardia:present','cxr:Cardiomegaly:present'],
               edges=[dict(source=0,target=1,patient_count=3,admission_count=4),dict(source=2,target=3,patient_count=2,admission_count=2)])
    atomic(out/'graph.json',graph)
    cfg['prepared_manifest']=str(root/'manifest.json');cfg['sensor_sources']=str(root/'sensors.json')
    atomic(cfg['prepared_manifest'],manifest);atomic(cfg['sensor_sources'],sources)
    m=[]
    for i,pid in enumerate(['10','20','1001','1002','1003','1004']):
        image=root/(f'image{i}.png');Image.new('RGB',(4,4),(i*30,10,5)).save(image)
        m.append(dict(sample_id=f'c{i}',subject_id=pid,study_id=str(900+i),source_index=i,image_path=str(image),
                      report_text='GROUND_TRUTH_MUST_NOT_ENTER_CONTEXT',labels=['forbidden']))
    cfg['cxr_manifest']=str(root/'cxr.jsonl');cfg['cxr_train_partition']=str(root/'partition.jsonl')
    for key in ('cxr_manifest','cxr_train_partition'):
        Path(cfg[key]).write_text(''.join(json.dumps(r)+'\n' for r in m))
    cfg['cxr_features']=str(root/'cxr.npy');cfg['cxr_status']=str(root/'status.npy')
    np.save(cfg['cxr_features'],np.array([[1,.1+i*.1] for i in range(len(m))],dtype='float32'));np.save(cfg['cxr_status'],np.ones(len(m),dtype='int8'))
    return cfg


def build_ready(cfg):
    with patch('inputs.validate_frozen_inputs',return_value={}):
        preflight(cfg);symile_bundles(cfg);inventory=eligible_cxr(cfg)
        run=Path(cfg['run_dir']);atomic(run/'audits/rad_equivalence.json',dict(passed=True,
            symile_bank_sha256=sha(run/'banks/symile.json'),inventory_sha256=sha(run/'banks/cxr_inventory.json'),cxr_features_sha256=sha(cfg['cxr_features'])))
        runtime=dict(modality='cxr',revision='test',model=SimpleNamespace(generation_config=GenerationConfig()),tokenizer=Tok())
        with patch('encoders.version',return_value='test'),patch('sensor_candidate_generation.infer',return_value='Pulmonary edema is present.'):
            self_result=extract_cxr(cfg,runtime,max_new=0)
            assert self_result['ok']==len(inventory)
        finalize_banks(cfg);prepare_queries(cfg);run_retrieval(cfg,max_run_hours=1.)
        prepare_packets(cfg)
    return inventory

class ExperimentTests(unittest.TestCase):
    def test_search_spaces_exclusions_ties(self):
        rows=[dict(id='b',patient_id='1'),dict(id='a',patient_id='1'),dict(id='c',patient_id='2')]
        index=Index(rows,[[1,0],[1,0],[.9,.1]],'ecg',backend='exact')
        self.assertEqual([rows[i]['id'] for i,_ in index.search([1,0],'ecg',2,distinct_patients=True)],['a','c'])
        self.assertEqual([i for i,_ in index.search([1,0],'ecg',3,exclude={'1'})],[2])
        with self.assertRaises(ValueError):index.search([1,0],'cxr',2)
        with self.assertRaises(ValueError):normalize([[0,0]])
    def test_faiss_matches_exact(self):
        try:import faiss
        except ImportError:self.skipTest('faiss optional')
        rows=[dict(id=str(i),patient_id=str(i//2)) for i in range(100)]
        x=np.random.default_rng(3).normal(size=(100,8));x[:4]=x[0]
        a=Index(rows,x,'cxr','exact');b=Index(rows,x,'cxr','faiss')
        for excluded in (set(),{'0','1','2'}):
            left=a.search(x[0],'cxr',5,excluded,True);right=b.search(x[0],'cxr',5,excluded,True)
            self.assertEqual([i for i,_ in left],[i for i,_ in right])
    def test_end_to_end_and_failure_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));inventory=build_ready(cfg);self.assertEqual(len(inventory),4)
            run=Path(cfg['run_dir']);queries=read(run/'queries/internal_val/manifest.json')
            for q in queries:
                result=read(run/'retrieval/internal_val'/(q['id']+'.json'))
                anchor_patients={r['patient_id'] for r in result['anchors']}
                for c in result['candidates']:
                    self.assertTrue(all(r['patient_id'] not in anchor_patients for r in c['reverse_admissions']))
                packets=read(run/'packets/internal_val'/(q['id']+'.json'))['cases']
                self.assertEqual(len(packets),5)
                for c in packets:
                    prompt,audit=format_prompt(c,Tok(),cfg)
                    self.assertNotIn('GROUND_TRUTH_MUST_NOT_ENTER_CONTEXT',prompt)
                    self.assertLessEqual(audit['context_tokens'],cfg['context_tokens'])
            # Changing target labels cannot change existing label-free retrieval/packet hashes.
            before=sha(run/'packets/internal_val'/(queries[0]['id']+'.json'))
            atomic(Path(cfg['frozen_out'])/'targets.json',[dict(id=q['id'],findings={'atrial_fibrillation':1,'bradycardia':0}) for q in queries])
            self.assertEqual(before,sha(run/'packets/internal_val'/(queries[0]['id']+'.json')))
            image=Path(tmp)/'query.png';image.write_bytes(b'image')
            runtime=dict(model=SimpleNamespace(generation_config=GenerationConfig()),tokenizer=Tok(),family='llama',
                model_id=MODEL_IDS['llama_base'],experiment_model_key='llama_base',revision='fixed',enable_thinking=False)
            def provider(q):return str(image),'source-tensor'
            with patch('queries.image_provider',return_value=provider),patch('ecg_external_generation.infer',return_value='Atrial fibrillation is present.') as infer:
                a=generate(cfg,runtime,max_new=5,max_run_hours=1.)
                b=generate(cfg,runtime,max_new=0,max_run_hours=1.)
                self.assertEqual(a['ok'],5);self.assertEqual(b['ok'],10);self.assertEqual(infer.call_count,10)
                self.assertEqual(generate(cfg,runtime,max_new=0,max_run_hours=1.)['processed_this_call'],0)
            folder=folder_for(cfg,'llama_base',False,384)
            result=evaluate(cfg,{'llama_base':folder},Path(cfg['frozen_out'])/'targets.json',run/'eval',repeats=50,max_run_hours=1.)
            self.assertTrue(result['complete']);self.assertEqual(result['matched_queries'],2)
            with patch('queries.image_provider',return_value=provider),patch('queries.load_queries',return_value=queries):
                review=prepare_review(cfg,{'llama_base':folder},run/'review',n_queries=1)
            self.assertEqual(review['review_items'],5)
            for r in read(run/'review/PUBLIC/forms.json'):
                self.assertNotIn('model',r);self.assertNotIn('arm',r);self.assertNotIn('query_id',r)
            # A failed sensor-only report should not shrink graph-vs-bidirectional pairwise analysis.
            log=latest(folder/'reports.jsonl');r=copy.deepcopy(log[queries[0]['id'],'sensor_only']);r.update(status='error',error='truncated');append(folder/'reports.jsonl',r)
            with self.assertRaises(ValueError):evaluate(cfg,{'llama_base':folder},Path(cfg['frozen_out'])/'targets.json',run/'eval_partial',repeats=50,max_run_hours=1.)
            partial=evaluate(cfg,{'llama_base':folder},Path(cfg['frozen_out'])/'targets.json',run/'eval_partial',allow_partial=True,repeats=50,max_run_hours=1.)
            self.assertEqual(partial['matched_queries'],1)
            pair=[r for r in partial['comparisons'] if r['contrast']=='bidirectional_graph_minus_bidirectional_pool' and r['subset']=='pairwise_complete' and r['metric']=='f1'][0]
            self.assertEqual(pair['paired_queries'],2)
    def test_input_guards_and_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp))
            with patch('inputs.validate_frozen_inputs',return_value={}):preflight(cfg)
            sources=read(cfg['sensor_sources']);sources[0]['source_index']=99;atomic(cfg['sensor_sources'],sources)
            with self.assertRaises(ValueError):symile_bundles(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));rows=lines(cfg['cxr_manifest'])
            Path(rows[3]['image_path']).write_bytes(Path(rows[2]['image_path']).read_bytes())
            eligible=eligible_cxr(cfg)
            self.assertEqual(len(eligible),2)
            self.assertFalse({'c2','c3'} & {r['id'] for r in eligible})
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));rows=lines(cfg['cxr_manifest']);rows[2]['subject_id']=None
            Path(cfg['cxr_manifest']).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):eligible_cxr(cfg)
    def test_cohort_routing_and_config_changes(self):
        from common import assert_config
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp))
            with patch('inputs.validate_frozen_inputs',return_value={}):preflight(cfg)
            routed=dict(cfg,cohort='internal_test');assert_config(routed)
            with self.assertRaises(ValueError):assert_config(dict(cfg,k_anchors=3))
    def test_external_sampling_and_failed_candidate_fallback(self):
        from queries import extract_queries,load_queries
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));cfg.update(cohort='shaoxing',external_sample_size=2)
            with patch('inputs.validate_frozen_inputs',return_value={}):preflight(cfg)
            external=Path(cfg['shaoxing_cohort']);manifest=[]
            for i in range(3):
                manifest.append(dict(id='external'+str(i),patient_id='sx'+str(i),admission_id='sx'+str(i),study_id='s'+str(i),
                    modality='ecg',split='external',record='wfdb'+str(i),header_sha256='h',waveform_sha256='w',waveform_file='file'))
            atomic(external/'manifest.json',manifest);atomic(external/'cohort_protocol.json',dict(manifest_sha256=signature(manifest)))
            fp=dict(manifest_sha256=signature(manifest),bank_records_sha256=sha(Path(cfg['frozen_out'])/'records.json'))
            atomic(external/'features/protocol.json',fp)
            for r in manifest:atomic(external/'features'/(r['id']+'.json'),dict(id=r['id'],embedding=[1,.3],embedding_space='ecg_space',protocol_sha256=signature(fp)))
            prepared=prepare_queries(cfg);self.assertEqual(prepared['queries'],2)
            image=Path(tmp)/'query';image.write_bytes(b'image')
            provider=lambda row:(str(image),'tensor')
            runtime=dict(modality='ecg',revision='frozen')
            with patch('queries.image_provider',return_value=provider),patch('sensor_candidate_generation.infer',side_effect=['Atrial fibrillation.',ValueError('truncated')]):
                result=extract_queries(cfg,runtime,max_new=0,max_run_hours=1.)
            self.assertEqual(result['attempted'],2);self.assertEqual(result['ok'],1)
            with patch('queries.image_provider',return_value=provider):
                rows=load_queries(cfg)
            self.assertEqual(len(rows),2)
            failed=[r for r in rows if r['candidate_status']=='error'];self.assertEqual(len(failed),1);self.assertEqual(failed[0]['findings'],[])

    def test_graph_absence_does_not_invent_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));build_ready(cfg);run=Path(cfg['run_dir'])
            symile=read(run/'banks/symile.json');mimic=read(run/'banks/mimic.json')
            for i,r in enumerate(mimic):r['matrix_index']=i
            q=read(run/'queries/internal_val/manifest.json')[0];q['findings']=[]
            indices=[Index(symile,[r['ecg_embedding'] for r in symile],'ecg_space','exact'),
                     Index(symile,[r['cxr_embedding'] for r in symile],'cxr_space','exact'),
                     Index(mimic,np.load(run/'banks/mimic_cxr.npy'),'cxr_space','exact')]
            result=infer_query(q,symile,mimic,*indices,read(Path(cfg['frozen_out'])/'graph.json'),cfg)
            self.assertTrue(result['fallbacks']['no_graph_support'])
            self.assertEqual(result['selected']['bidirectional_pool'],result['selected']['bidirectional_graph'])
            self.assertTrue(all(not c['graph_support'] for c in result['candidates']))

    def test_graph_empty_support_and_shuffle(self):
        graph=dict(vocabulary=['ecg:a:present','ecg:b:present','cxr:x:present','cxr:y:present'],edges=[dict(source=0,target=2,patient_count=4,admission_count=5)])
        control=shuffled_graph(graph,43)
        self.assertEqual(control,shuffled_graph(graph,43))
        for old,new in zip(graph['edges'],control['edges']):
            self.assertEqual(old['patient_count'],new['patient_count'])
            self.assertTrue(control['vocabulary'][new['source']].startswith('ecg:'))
    def test_journal_repair_and_zero_patients(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl';append(p,dict(query_id='q',arm='a',status='ok'))
            with p.open('ab') as f:f.write(b'{"broken":')
            saved=latest(p,repair=True);self.assertEqual(len(saved),1)
            append(p,dict(query_id='r',arm='a',status='ok'));self.assertEqual(len(latest(p)),2)
        self.assertEqual(paired([],'f1','a','b')['patients'],0)
        self.assertEqual(holm({'a':.01,'b':.04}),{'a':.02,'b':.04})
    def test_notebooks(self):
        import nbformat
        for p in EXP.glob('BR*.ipynb'):
            n=nbformat.read(p,as_version=4);nbformat.validate(n)
            for c in n.cells:
                if c.cell_type=='code':compile(c.source,str(p),'exec')

if __name__=='__main__':unittest.main()
