"""Facts about the data on the clean test queries (the queries the results are reported on), with the definitions of
data_facts in analysis/readings.py. No model is trained or used: only training counts, labels and item features.

  queries_with_unseen_seed  share of queries with at least one seed that no training bundle holds
  queries_all_unseen        share of queries in which every seed is in no training bundle
  labels[l]                 targets_sharing: mean over queries of the share of hidden items whose level-l label is a
                            seed's label; catalogue_sharing: the same share among the non-seed catalogue items
  cosine_recall20           Recall@20 when the candidates are ranked by cosine between the mean seed feature and the
                            candidate's feature (rows of X scaled to unit length)

Writes results/data_facts_test.json.   PYTHONPATH=. python -m analysis.data_facts_test
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from triad import data
from triad.evaluate import query_metrics
from triad.train import write_json

CACHE = Path('cache');OUT = Path('results/data_facts_test.json')
NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify')


def facts(ds):
    pairs = [ds.test_pairs[i] for i in ds.clean_test];n = ds.n_items
    f = np.bincount(np.concatenate([np.unique(b) for b in ds.train]), minlength=n)
    x = ds.features/np.maximum(np.linalg.norm(ds.features, axis=1, keepdims=True), 1e-12)
    r = dict(queries=len(pairs))
    unseen = [np.mean(f[sorted(set(s))] == 0) for s, _ in pairs]
    r['queries_with_unseen_seed'] = float(np.mean([u > 0 for u in unseen]))
    r['queries_all_unseen'] = float(np.mean([u == 1 for u in unseen]))
    r['labels'] = []
    for labels in ds.levels:
        labels = np.asarray(labels);tgt, cat = [], []
        for s, g in pairs:
            s = sorted(set(s));seen = set(labels[s]);rest = np.setdiff1d(np.arange(n), s)
            tgt.append(np.mean([labels[t] in seen for t in set(g)]));cat.append(np.isin(labels[rest], list(seen)).mean())
        r['labels'].append(dict(targets_sharing=float(np.mean(tgt)), catalogue_sharing=float(np.mean(cat))))
    hit = []
    for a in range(0, len(pairs), 256):
        P = pairs[a:a + 256];q = np.stack([x[sorted(set(s))].mean(0) for s, _ in P]);v = q@x.T
        hit += [query_metrics(v[k], s, g)['recall20'] for k, (s, g) in enumerate(P)]
    r['cosine_recall20'] = float(np.mean(hit))
    return r


def main():
    out = {}
    for name in NAMES:
        out[name] = facts(data.load(name, CACHE))
        print(name, json.dumps(out[name]), flush=True)
    write_json(OUT, dict(note=__doc__, datasets=out))


if __name__ == '__main__':
    main()
