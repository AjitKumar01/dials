"""Checks that later changes keep TRIAD as it is.

PYTHONPATH=. python -m pytest tests -q

leave-one-out   training-mode content, label, graph and gate scores of a bundle equal evaluation-mode scores of a model
                built without that bundle (synthetic catalogue, no data needed). The label term reads only item labels,
                which do not depend on bundles, so it must be unchanged.
metrics         top_items against a brute-force sort, with ties and seeds.
golden          the Electronic seed-0 model reproduces tests/golden (needs the data and runs/).
resume          a run interrupted after epoch 2 and resumed equals an uninterrupted run (needs the Electronic data).
"""
import triad  # pins BLAS threads before numpy loads
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from triad import data
from triad import train as T
from triad.evaluate import scores,top_items
from triad.model import TRIAD

ROOT=Path(__file__).resolve().parents[1];GOLDEN=ROOT/'tests'/'golden'/'electronic_seed0.npz'
torch.use_deterministic_algorithms(True)


def synthetic(seed=0,n=60,bundles=82):
    rng=np.random.default_rng(seed)
    train=[rng.choice(n,int(rng.integers(2,7)),replace=False).tolist() for _ in range(bundles)]
    train.append(list(train[0]))  # a duplicate bundle
    train.append([train[1][0],train[1][0],train[1][1]])  # a repeated item
    features=rng.standard_normal((n,16)).astype(np.float32)
    levels=[rng.integers(0,7,n),rng.integers(0,11,n)]
    return train,features,levels


def model_from(train,features,levels,state=None):
    model=TRIAD(features,levels,train,hidden=8)
    if state is None:
        torch.manual_seed(1)
        with torch.no_grad():
            for p in model.parameters():p.add_(.3*torch.randn_like(p))  # move every parameter off its initialization
    else:model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def test_leave_one_out():
    train,features,levels=synthetic();full=model_from(train,features,levels)
    worst=dict(content=0.,labels=0.,graph=0.,gate=0.)
    for i,bundle in enumerate(train):
        b=sorted(set(bundle))
        if len(b)<2:continue
        seeds=[torch.as_tensor(b[:max(1,len(b)//2)])];whole=[torch.as_tensor(b)]
        without=model_from(train[:i]+train[i+1:],features,levels,full.state_dict())
        pairs=dict(content=(full.content_scores(seeds,whole,None),without.content_scores(seeds,None,without.cache())),
                   labels=(torch.stack([full.label_scores(l,seeds) for l in range(full.levels)]),
                           torch.stack([without.label_scores(l,seeds) for l in range(without.levels)])),
                   graph=(full.graph_scores(seeds,whole),without.graph_scores(seeds,None)),
                   gate=(full.gate_weights(seeds,whole),without.gate_weights(seeds,None)))
        for k,(a,c) in pairs.items():worst[k]=max(worst[k],float((a-c).abs().max()))
    assert all(v<1e-4 for v in worst.values()),worst


def brute_top(values,seeds,k):
    order=sorted((i for i in range(len(values)) if i not in set(seeds)),key=lambda i:(-values[i],i))
    return order[:k]


def test_top_items_matches_brute_force():
    rng=np.random.default_rng(0)
    for _ in range(200):
        n=int(rng.integers(5,60));values=rng.integers(0,6,n).astype(float)  # many exact ties
        seeds=rng.choice(n,int(rng.integers(0,4)),replace=False).tolist();k=int(rng.integers(1,25))
        assert top_items(values,seeds,k).tolist()==brute_top(values,seeds,k)


def electronic():
    try:return data.load('electronic',ROOT/'cache')
    except FileNotFoundError:pytest.skip('Electronic data not found')


@torch.inference_mode()
def test_golden_scores():
    if not GOLDEN.exists() or not (ROOT/'runs'/'electronic'/'seed0'/'model.pt').exists():pytest.skip('golden file or run missing')
    ds=electronic();golden=np.load(GOLDEN);torch.set_num_threads(4)
    model=T.load(ds,ROOT/'runs'/'electronic'/'seed0',ROOT/'cache')
    rows=ds.test_pairs[:len(golden['top20'])];values=scores(model,[s for s,_ in rows])
    np.testing.assert_allclose(values[:,golden['items']],golden['scores'],rtol=0,atol=1e-4)
    same=np.mean([(top_items(v,s,20)==t).all() for v,(s,_),t in zip(values,rows,golden['top20'])])
    assert same>=.95,f'{same:.0%} of top-20 lists unchanged'  # near-ties may swap under floating-point reordering


def test_resume(tmp_path):
    ds=electronic();torch.set_num_threads(1);config=dict(T.CONFIG,epochs=4,patience=10)
    T.fit(ds,0,tmp_path/'straight',ROOT/'cache',config)
    log=T.log
    def interrupt(message):
        log(message)
        if ' epoch=2 ' in message:raise KeyboardInterrupt
    T.log=interrupt
    try:
        with pytest.raises(KeyboardInterrupt):T.fit(ds,0,tmp_path/'resumed',ROOT/'cache',config)
    finally:T.log=log
    assert (tmp_path/'resumed'/'last.pt').exists()
    T.fit(ds,0,tmp_path/'resumed',ROOT/'cache',config)
    a,b=(json.loads((tmp_path/r/'history.json').read_text()) for r in ('straight','resumed'))
    assert a==b
    sa,sb=(torch.load(tmp_path/r/'model.pt',weights_only=True) for r in ('straight','resumed'))
    assert all(torch.equal(sa[k],sb[k]) for k in sa)


@torch.no_grad()
def test_start_is_simple_scorer():
    """Untrained, s = maxabs(10 xbar'x_c) + maxabs(log(1 + sum_s N_sc / sqrt(deg_s deg_c))): no label term, no prior."""
    train,features,levels=synthetic();model=TRIAD(features,levels,train,hidden=8).eval()
    seeds=[torch.as_tensor(sorted(set(b[:2]))) for b in train[:20]]
    x=torch.nn.functional.normalize(torch.as_tensor(features),dim=1)
    content=torch.stack([10*x[s].mean(0)@x.T for s in seeds])
    deg=model.history_mass.clamp_min(1);N=torch.zeros(model.n,model.n)
    for b in train:
        b=sorted(set(b))
        for i in b:
            for j in b:
                if i!=j:N[i,j]+=1
    graph=torch.stack([(N[s]/(deg[s][:,None]*deg[None,:]).sqrt()).sum(0).log1p() for s in seeds])
    scale=lambda v:v/v.abs().amax(1,keepdim=True).clamp_min(1e-6)
    assert torch.allclose(model(seeds,cache=model.cache()),scale(content)+scale(graph),atol=1e-5)
