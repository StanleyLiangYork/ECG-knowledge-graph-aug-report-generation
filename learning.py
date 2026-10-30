"""Train graph-aware reference rerankers; never import FAISS in this process."""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SAGEConv, TransformerConv
from core import tokens, pool, relevance

def features(rows, graph):
    vocab={s:i for i,s in enumerate(graph['vocabulary'])}
    x=torch.zeros((len(rows),len(vocab)))
    for i,r in enumerate(rows):
        for s in tokens(r):
            if s in vocab: x[i,vocab[s]]=1
    return x / x.sum(1,keepdim=True).clamp(min=1)

def edges(graph, shuffle=False, seed=43):
    pairs=[(e['source'],e['target']) for e in graph['edges']]
    if shuffle:
        # Fixed node permutation changes semantic endpoints, retaining graph topology.
        perm=np.random.default_rng(seed).permutation(len(graph['vocabulary']))
        pairs=[(int(perm[a]),int(perm[b])) for a,b in pairs]
    pairs=pairs+[(b,a) for a,b in pairs]
    return torch.tensor(pairs,dtype=torch.long).T.contiguous() if pairs else torch.empty((2,0),dtype=torch.long)

class Ranker(nn.Module):
    def __init__(self,n,kind):
        super().__init__(); self.kind=kind
        self.embedding=nn.Embedding(n,16)
        self.layer = (SAGEConv(16,16) if kind=='sage' else
                      TransformerConv(16,16,heads=1,dropout=0) if kind=='transformer' else nn.Linear(16,16))
        self.head=nn.Sequential(nn.Linear(34,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x,e,pairs,similarity):
        h=self.embedding.weight
        h=torch.relu(self.layer(h,e) if self.kind!='mlp' else self.layer(h))
        z=x@h; a=z[pairs[:,0]]; b=z[pairs[:,1]]
        overlap=(x[pairs[:,0]]*x[pairs[:,1]]).sum(1,keepdim=True)
        return self.head(torch.cat([a*b,torch.abs(a-b),similarity[:,None],overlap],1)).flatten()

def pair_data(rows,targets,split,pools=None):
    pos={r['id']:i for i,r in enumerate(rows)}; truth={t['id']:t for t in targets}
    pairs=[]; sims=[]; ys=[]
    vectors = {r['id']: np.asarray(r['embedding'], dtype=float) for r in rows} if pools is not None else {}
    vectors = {rid: v / np.linalg.norm(v) for rid, v in vectors.items()}
    for q in rows:
        if q['split']!=split: continue
        candidates = pool(rows,q) if pools is None else [(float(np.dot(vectors[q['id']], vectors[rid])), rid) for rid in pools[q['id']]]
        for sim,rid in candidates:
            reference = rows[pos[rid]]
            if reference['split'] != 'train' or reference['modality'] != q['modality'] or reference['patient_id'] == q['patient_id']:
                raise ValueError('Invalid saved training reference pool')
            y=relevance(truth[q['id']],truth[rid])
            if y is None: continue
            pairs.append((pos[q['id']],pos[rid])); sims.append(sim); ys.append(y)
    if not pairs: raise ValueError(f'No judged {split} pairs: supply adequate independent annotations')
    return torch.tensor(pairs),torch.tensor(sims,dtype=torch.float32),torch.tensor(ys,dtype=torch.float32)

def atomic_torch_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def fit(rows,targets,graph,kind,seed=43,shuffle=False,epochs=40,
        prepared=None, checkpoint_path=None, fingerprint=None, pair_batch_size=8192):
    if epochs < 1 or pair_batch_size < 1: raise ValueError('Positive epochs and batch size required')
    torch.set_num_threads(1); torch.manual_seed(seed)
    model=Ranker(len(graph['vocabulary']),kind)
    x=features(rows,graph); e=edges(graph,shuffle,seed)
    tr,va = prepared if prepared is not None else (pair_data(rows,targets,'train'),pair_data(rows,targets,'val'))
    opt=torch.optim.Adam(model.parameters(),lr=.01); best=float('inf'); checkpoint=None
    start=0
    identity=dict(fingerprint=fingerprint,kind=kind,seed=seed,shuffle=shuffle,
                  epochs=epochs,pair_batch_size=pair_batch_size)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        if state['identity'] != identity: raise ValueError('Training checkpoint configuration mismatch')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        best=state['best_loss']; checkpoint=state['best_model']; start=state['epoch']
        torch.set_rng_state(state['torch_rng'])
        print(f'{kind} seed={seed} shuffled={shuffle}: resume after epoch {start}/{epochs}',flush=True)
    for epoch in range(start,epochs):
        model.train(); opt.zero_grad()
        # Accumulate the full-pair mean BCE gradient; one Adam update per epoch.
        # Chunking bounds activation memory without changing sampling or weighting.
        for offset in range(0,len(tr[2]),pair_batch_size):
            sl=slice(offset,offset+pair_batch_size)
            loss=nn.functional.binary_cross_entropy_with_logits(model(x,e,tr[0][sl],tr[1][sl]),tr[2][sl],reduction='sum')/len(tr[2])
            loss.backward()
        opt.step(); model.eval()
        total=0.
        with torch.no_grad():
            for offset in range(0,len(va[2]),pair_batch_size):
                sl=slice(offset,offset+pair_batch_size)
                total+=nn.functional.binary_cross_entropy_with_logits(model(x,e,va[0][sl],va[1][sl]),va[2][sl],reduction='sum').item()
        score=total/len(va[2])
        if not np.isfinite(score): raise ValueError('Non-finite validation loss; previous checkpoint retained')
        if score < best: best=score; checkpoint=copy.deepcopy(model.state_dict())
        if checkpoint_path is not None:
            atomic_torch_save(dict(identity=identity,epoch=epoch+1,model=model.state_dict(),
                optimizer=opt.state_dict(),best_loss=best,best_model=checkpoint,
                torch_rng=torch.get_rng_state()),checkpoint_path)
        print(f'{kind} seed={seed} shuffled={shuffle}: epoch {epoch+1}/{epochs}, val_BCE={score:.6f}, best={best:.6f}',flush=True)
    model.load_state_dict(checkpoint)
    return model, best

def rank(rows,graph,query,model=None,rule=False,shuffle=False,seed=43):
    candidates=pool(rows,query); pos={r['id']:i for i,r in enumerate(rows)}
    if not candidates: raise ValueError('Empty reference pool')
    if model is None and not rule: return [rid for _,rid in candidates]
    x=features(rows,graph)
    if rule:
        adj=torch.eye(len(graph['vocabulary']))
        for edge in graph['edges']:
            a,b=edge['source'],edge['target']; adj[a,b]=adj[b,a]=edge['patient_count']
        adj=adj/adj.sum(1,keepdim=True).clamp(min=1)
        z=x@adj; q=z[pos[query['id']]]
        scores=[sim+.25*float(torch.cosine_similarity(q[None],z[pos[rid]][None])) for sim,rid in candidates]
    else:
        pairs=torch.tensor([(pos[query['id']],pos[rid]) for _,rid in candidates])
        with torch.no_grad(): scores=model(x,edges(graph,shuffle,seed),pairs,torch.tensor([s for s,_ in candidates])).tolist()
    return [rid for _,rid in sorted(zip(scores,[rid for _,rid in candidates]),key=lambda p:(-p[0],p[1]))]
