"""Content term alone: rank the clean test queries by TRIAD's content term (label and graph terms set to zero), for the
untrained model (cosine) and the trained runs in runs/<dataset> (mean of seeds 0-2). No training. Run from the
repository root:

    PYTHONPATH=. python paper_v2/figdata/content_only.py

Output (paper_v2 §5, "No single relation is enough"), NDCG@20 untrained / trained:
electronic .0866 / .0911, food .2579 / .2688, clothing .2134 / .2368, pog_dense .0001 / .0099, pog .0001 / .0062,
spotify_sparse .0274 / .0347.
"""
import triad  # pins BLAS threads before numpy loads
from pathlib import Path
import numpy as np
import torch
from triad import data, train as T
from triad.evaluate import evaluate

torch.set_num_threads(3)


def only_content(model):
    """With the label and graph terms at zero, the gate weight of content is positive, so the ranking is by content."""
    model.label_scores = lambda i, seeds: torch.zeros(len(seeds), model.n)
    model.graph_scores = lambda seeds, bundles, cache=None: torch.zeros(len(seeds), model.n)
    return model


def ndcg(model, ds):
    return np.array([q['ndcg20'] for q in evaluate(model, ds.test_pairs)['queries']])[np.array(ds.clean_test)]


for name in ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse'):
    ds = data.load(name, Path('cache'))
    torch.manual_seed(0);untrained = ndcg(only_content(T.build(ds, None, T.config_for(name))), ds).mean()
    trained = np.mean([ndcg(only_content(T.load(ds, f'runs/{name}/seed{s}', None)), ds) for s in (0, 1, 2)], 0).mean()
    print(f'{name:15s} cosine (untrained content) {untrained:.4f}  trained content only {trained:.4f}', flush=True)
