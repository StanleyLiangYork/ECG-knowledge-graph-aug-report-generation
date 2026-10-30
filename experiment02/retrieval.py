"""Separate-space forward/reverse search, patient exclusions and graph controls."""
from pathlib import Path
from collections import defaultdict,OrderedDict
import math
import numpy as np
from common import *


def normalize(values):
    a=np.asarray(values,dtype=np.float32)
    norms=np.linalg.norm(a,axis=-1,keepdims=True)
    if a.ndim not in (1,2) or not np.isfinite(a).all() or np.any(norms==0):raise ValueError('Invalid embedding matrix')
    return a/norms


class Index:
    def __init__(self,rows,matrix,space,backend="auto"):
        self.rows=rows;self.matrix=np.ascontiguousarray(normalize(matrix));self.space=space
        self.cache=OrderedDict();self.faiss=None
        if backend not in ('auto','exact','faiss'):raise ValueError('Unknown search backend')
        if backend!='exact':
            try:
                import faiss
                self.faiss=faiss.IndexFlatIP(self.matrix.shape[1]);self.faiss.add(self.matrix)
            except ImportError:
                if backend=='faiss':raise
        self.backend='faiss' if self.faiss is not None else 'exact'
        if len(rows)!=len(self.matrix):raise ValueError('Index row mismatch')
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Index duplicate ID')
    def search(self,vector,space,k,exclude=(),distinct_patients=False):
        if space!=self.space:raise ValueError('Cross-space comparison forbidden')
        v=normalize(vector)
        if v.shape!=(self.matrix.shape[1],):raise ValueError('Embedding dimension mismatch')
        forbidden=set(exclude);key=(v.tobytes(),k,tuple(sorted(forbidden)),distinct_patients)
        if key in self.cache:
            self.cache.move_to_end(key);return list(self.cache[key])
        if self.faiss is None:
            scores=self.matrix@v
            order=sorted(range(len(self.rows)),key=lambda j:(-float(scores[j]),self.rows[j]['id']))
        else:
            # Expand until enough eligible distinct patients are found. Expand through cutoff ties.
            count=min(len(self.rows),max(32,k*2));order=[]
            while True:
                distances,indexes=self.faiss.search(v[None],count)
                pairs=sorted(zip(indexes[0].tolist(),distances[0].tolist()),key=lambda p:(-p[1],self.rows[p[0]]['id']))
                chosen=[];seen=set()
                for j,score in pairs:
                    pid=patient(self.rows[j]['patient_id'])
                    if pid in forbidden or (distinct_patients and pid in seen):continue
                    chosen.append((j,score));seen.add(pid)
                if count==len(self.rows) or (len(chosen)>=k and chosen[k-1][1]>pairs[-1][1]):break
                count=min(len(self.rows),count*2)
            order=[j for j,_ in pairs];scores=dict(pairs)
        seen=set();result=[]
        for i in order:
            r=self.rows[i];pid=patient(r['patient_id'])
            if pid in forbidden or (distinct_patients and pid in seen):continue
            result.append((i,float(scores[i])));seen.add(pid)
            if len(result)>=k:break
        self.cache[key]=result
        if len(self.cache)>512:self.cache.popitem(last=False)
        return list(result)


def shuffled_graph(graph,seed):
    vocab=graph['vocabulary'];groups=defaultdict(list);rng=np.random.default_rng(seed);perm=list(range(len(vocab)))
    for i,token in enumerate(vocab):
        modality,_,polarity=token.split(':',2);groups[modality,polarity].append(i)
    for ids in groups.values():
        values=rng.permutation(ids)
        for i,j in zip(ids,values):perm[i]=int(j)
    return dict(graph,edges=[dict(e,source=perm[e['source']],target=perm[e['target']]) for e in graph['edges']],
                shuffle_seed=seed,permutation=perm)


def graph_support(query_findings,cxr_findings,graph,min_patients):
    vocab=graph['vocabulary'];lookup={token:i for i,token in enumerate(vocab)};edge_map={frozenset((e['source'],e['target'])):e for e in graph['edges']}
    result=[]
    for a in sorted({x['concept'] for x in query_findings if x['polarity']=='present'}):
        for b in sorted({x['concept'] for x in cxr_findings if x['polarity']=='present'}):
            x=lookup.get('ecg:'+a+':present');y=lookup.get('cxr:'+b+':present')
            edge=edge_map.get(frozenset((x,y))) if x is not None and y is not None else None
            if edge and edge['patient_count']>=min_patients:
                result.append(dict(ecg_concept=a,cxr_concept=b,predicate='training_admission_cooccurrence',
                                   patient_count=edge['patient_count'],admission_count=edge['admission_count']))
    return result


def ranks(values,key):
    return {r['id']:i+1 for i,r in enumerate(sorted(values,key=lambda x:(-key(x),x['id'])))}


def infer_query(query,symile,mimic,ecg_index,symile_cxr_index,mimic_index,graph,cfg):
    spaces={r['ecg_space'] for r in symile};cxrspaces={r['cxr_space'] for r in symile}
    if len(spaces)!=1 or len(cxrspaces)!=1:raise ValueError('Inconsistent embedding spaces')
    exclude={patient(query['patient_id'])} if query['split']!='external' else set()
    qspace=query['embedding_space']
    if qspace!=next(iter(spaces)):raise ValueError('Query does not match verified HuBERT bank space')
    initial=ecg_index.search(query['embedding'],qspace,cfg['ecg_pool_size'],exclude,distinct_patients=False)
    anchors=[];seen=set()
    for i,sim in initial:
        r=symile[i]
        if r['patient_id'] in seen:continue
        anchors.append(dict(r,query_ecg_cosine=sim));seen.add(r['patient_id'])
        if len(anchors)==cfg['k_anchors']:break
    candidates={};anchor_ids={r['id'] for r in anchors};anchor_patients={r['patient_id'] for r in anchors}
    for anchor in anchors:
        for i,sim in mimic_index.search(anchor['cxr_embedding'],next(iter(cxrspaces)),cfg['cxr_per_anchor']):
            r=mimic[i];entry=candidates.setdefault(r['id'],dict(r,forward_paths=[],forward_cxr=-float('inf')))
            entry['forward_paths'].append(dict(symile_admission=anchor['id'],cosine=sim))
            entry['forward_cxr']=max(entry['forward_cxr'],sim)
    q=normalize(query['embedding']);ecgs=ecg_index.matrix
    for c in candidates.values():
        i=c['matrix_index'];v=mimic_index.matrix[i]
        reverse=symile_cxr_index.search(v,next(iter(cxrspaces)),cfg['reverse_m'],anchor_patients|exclude,True)
        unrestricted=symile_cxr_index.search(v,next(iter(cxrspaces)),cfg['reverse_m'],exclude,True)
        c['unrestricted_returns_anchor']=any(symile[j]['id'] in anchor_ids for j,_ in unrestricted)
        c['reverse_admissions']=[dict(admission_id=symile[j]['id'],patient_id=symile[j]['patient_id'],
            cxr_cosine=s,query_ecg_cosine=float(ecgs[j]@q)) for j,s in reverse]
        c['reverse_ecg']=float(np.mean([r['query_ecg_cosine'] for r in c['reverse_admissions']])) if reverse else None
        # Chance control: deterministic random ECG link per reverse admission, independent of labels.
        seed=int(signature([cfg['seed'],query['id'],c['id']])[:16],16)
        rng=np.random.default_rng(seed);eligible=[j for j,r in enumerate(symile) if r['patient_id'] not in anchor_patients|exclude]
        random_ids=[];random_patients=set()
        for j in rng.permutation(eligible):
            pid=symile[j]['patient_id']
            if pid in random_patients:continue
            random_ids.append(int(j));random_patients.add(pid)
            if len(random_ids)==len(reverse):break
        if not reverse:random_ids=[]
        c['random_reverse_ecg']=float(np.mean([ecgs[j]@q for j in random_ids])) if random_ids else None
        c['random_reverse_admissions']=[symile[j]['id'] for j in random_ids]
    pool=list(candidates.values());forward=ranks(pool,lambda c:c['forward_cxr']);reverse=ranks([c for c in pool if c['reverse_ecg'] is not None],lambda c:c['reverse_ecg'])
    for c in pool:c['bidirectional_score']=1/(cfg['rrf_constant']+forward[c['id']])+(1/(cfg['rrf_constant']+reverse[c['id']]) if c['id'] in reverse else 0)
    control=shuffled_graph(graph,cfg['seed']);graph_ranks={}
    for name,g in [('graph',graph),('shuffled',control)]:
        supported=[]
        for c in pool:
            support=graph_support(query.get('findings',[]),c['findings'],g,cfg['min_graph_patients']);c[name+'_support']=support
            c[name+'_count']=len(support);c[name+'_weight']=sum(math.log1p(e['patient_count']) for e in support)
            if support:supported.append(c)
        graph_ranks[name]={c['id']:j+1 for j,c in enumerate(sorted(supported,key=lambda c:(-c[name+'_count'],-c[name+'_weight'],c['id'])))}
        for c in pool:c[name+'_score']=c['bidirectional_score']+(1/(cfg['rrf_constant']+graph_ranks[name][c['id']]) if c['id'] in graph_ranks[name] else 0)
    # Preserve all candidate study records; final selection has distinct patients.
    def choose(field):
        selected=[];patients=set()
        for c in sorted(pool,key=lambda c:(-c[field],c['id'])):
            if c['patient_id'] in patients:continue
            selected.append(c['id']);patients.add(c['patient_id'])
            if len(selected)==cfg['final_cxr']:break
        return selected
    selected=dict(sensor_only=[],forward_pool=choose('forward_cxr'),bidirectional_pool=choose('bidirectional_score'),
                  bidirectional_graph=choose('graph_score'),bidirectional_shuffled_graph=choose('shuffled_score'))
    conflicts=[]
    for r in anchors:
        for modality in ('ecg','cxr'):
            states=defaultdict(set)
            for f in r[modality+'_findings']:states[f['concept']].add(f['polarity'])
            conflicts.extend(dict(source=r['id'],modality=modality,concept=c,polarities=sorted(v)) for c,v in states.items() if {'present','absent'}<=v)
    for c in pool:
        states=defaultdict(set)
        for f in c['findings']:states[f['concept']].add(f['polarity'])
        conflicts.extend(dict(source=c['id'],modality='cxr',concept=k,polarities=sorted(v)) for k,v in states.items() if {'present','absent'}<=v)
    query_states=defaultdict(set)
    for f in query.get('findings',[]):query_states[f['concept']].add(f['polarity'])
    for r in anchors:
        reference_states=defaultdict(set)
        for f in r['ecg_findings']:reference_states[f['concept']].add(f['polarity'])
        for concept,states in reference_states.items():
            qstates=query_states[concept]
            if ('present' in qstates and 'absent' in states) or ('absent' in qstates and 'present' in states):
                conflicts.append(dict(source=r['id'],modality='ecg',concept=concept,query_polarities=sorted(qstates),reference_polarities=sorted(states),
                                      interpretation='query candidate and other-patient reference disagree; do not transfer the reference assertion'))
    node_sources=defaultdict(list)
    for f in query.get('findings',[]):node_sources['ecg:'+f['concept']+':'+f['polarity']].append(dict(source=query['id'],role='query_model_candidate'))
    for r in anchors:
        for modality in ('ecg','cxr'):
            for f in r[modality+'_findings']:node_sources[modality+':'+f['concept']+':'+f['polarity']].append(dict(source=r['id'],role='other_patient_symile_reference'))
    for c in pool:
        for f in c['findings']:node_sources['cxr:'+f['concept']+':'+f['polarity']].append(dict(source=c['id'],role='other_patient_mimic_reference'))
    local_edges=[dict(e,source_token=graph['vocabulary'][e['source']],target_token=graph['vocabulary'][e['target']]) for e in graph['edges']
                 if graph['vocabulary'][e['source']] in node_sources and graph['vocabulary'][e['target']] in node_sources]
    local_subgraph=dict(nodes=[dict(token=k,sources=v,in_training_vocabulary=k in graph['vocabulary']) for k,v in sorted(node_sources.items())],
                        edges=local_edges,predicate='training_admission_cooccurrence',laboratory_nodes=False)
    return dict(query_id=query['id'],query_findings=query.get('findings',[]),candidate_status=query.get('candidate_status','ok'),
                anchors=anchors,candidates=pool,selected=selected,conflicts=conflicts,local_subgraph=local_subgraph,
                fallbacks=dict(anchors_short=len(anchors)<cfg['k_anchors'],cxr_pool_short=len(pool)<cfg['final_cxr'],
                               no_query_concepts=not query.get('findings'),no_graph_support=not graph_ranks['graph']),
                graph_control_sha256=signature(control),graph_control_changed_edges=sum((a['source'],a['target'])!=(b['source'],b['target']) for a,b in zip(graph['edges'],control['edges'])))


def run_retrieval(cfg,max_run_hours=30.):
    from inputs import assert_inputs
    from queries import load_queries
    assert_inputs(cfg);run=Path(cfg['run_dir']);symile=read(run/'banks/symile.json');mimic=read(run/'banks/mimic.json');queries=load_queries(cfg)
    for i,r in enumerate(mimic):r['matrix_index']=i
    ecgspace=symile[0]['ecg_space'];cxrspace=symile[0]['cxr_space']
    matrices={k:np.load(run/'banks'/(k+'.npy'),allow_pickle=False) for k in ('symile_ecg','symile_cxr','mimic_cxr')}
    expected=read(run/'stages/indexes.json')['inputs']['matrices']
    for name,digest in expected.items():
        if sha(run/'banks'/name)!=digest:raise ValueError('Index contents changed')
    idx=[Index(symile,matrices['symile_ecg'],ecgspace,cfg['search_backend']),Index(symile,matrices['symile_cxr'],cxrspace,cfg['search_backend']),Index(mimic,matrices['mimic_cxr'],cxrspace,cfg['search_backend'])]
    graph=read(Path(cfg['frozen_out'])/'graph.json')
    identity=dict(queries=signature(queries),symile=sha(run/'banks/symile.json'),mimic=sha(run/'banks/mimic.json'),graph=signature(graph),indexes=expected,search_backends=[i.backend for i in idx])
    sign=freeze_stage(run,'retrieval_'+cfg['cohort'],identity);folder=run/'retrieval'/cfg['cohort'];folder.mkdir(parents=True,exist_ok=True)
    with lock(folder/'writer.lock'),budget(max_run_hours,folder/'STOP') as stop:
        for q in queries:
            if stop():break
            path=folder/(q['id']+'.json')
            if path.exists():
                saved=read(path)
                if saved['protocol_signature']!=sign or saved['query_id']!=q['id']:raise ValueError('Retrieval resume mismatch')
                continue
            result=infer_query(q,symile,mimic,*idx,graph,cfg)
            atomic(path,dict(result,protocol_signature=sign))
    complete=sum((folder/(q['id']+'.json')).exists() for q in queries)
    status=dict(total=len(queries),ok=complete,complete=complete==len(queries));atomic(folder/'coverage.json',status);return status
