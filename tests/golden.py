"""Golden scores: the frozen Electronic seed-0 model on the first 64 test queries, at 512 fixed items, plus top-20 lists.

Written once from the code as frozen (commit 47e0a33); tests/test_dials.py checks every later version against it.
PYTHONPATH=. python tests/golden.py
"""
import dials  # pins BLAS threads before numpy loads
from pathlib import Path
import numpy as np
import torch
from dials import data
from dials.evaluate import top_items
from dials.train import load

PATH=Path(__file__).parent/'golden'/'electronic_seed0.npz';QUERIES=64


@torch.inference_mode()
def compute(model,ds):
    """Scores at the fixed items and top-20 lists for the first QUERIES test queries, through evaluate's code path."""
    from dials.evaluate import scores
    rows=ds.test_pairs[:QUERIES];values=scores(model,[s for s,_ in rows])
    items=np.random.default_rng(0).choice(ds.n_items,512,replace=False)
    return dict(items=items,scores=values[:,items],top20=np.stack([top_items(v,s,20) for v,(s,_) in zip(values,rows)]))


def frozen():
    ds=data.load('electronic');return load(ds,Path('runs')/'electronic'/'seed0','cache'),ds


if __name__=='__main__':
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    PATH.parent.mkdir(exist_ok=True);np.savez(PATH,**compute(*frozen()));print('wrote',PATH)
