"""CLHE on the warm and cold clean test queries of Electronic, Food and Clothing (definitions of analysis/cold.py).

CLHE = our reruns of the authors' code (github.com/Xiaohao-Liu/CLHE, commit 74eb236) on this repository's splits and
BLIP title features, three runs (seeds 0, 1, 2023); the configuration preset was chosen on validation. Their per-query
test rows (Recall@20, NDCG@20) are in results/clhe/<dataset>/seed<s>.json. The rows are in the order of CLHE's
bi_test_input.txt, which is sorted by bundle id, as are the test pairs of triad.data; on these datasets every test
query is clean. Checked: the overall means equal Table 2 of the paper and the warm/cold counts equal results/cold.json.
Writes results/clhe_cold.json.
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. python -m analysis.clhe_cold
"""
import triad  # pins BLAS threads before numpy loads
import json
from pathlib import Path
import numpy as np
from triad import data
from triad.train import write_json

CACHE = Path('cache');R = Path('results')


def main():
    check = json.loads((R/'cold.json').read_text())['rows'];out = {}
    for name in ('electronic', 'food', 'clothing'):
        ds = data.load(name, CACHE);assert len(ds.test_pairs) == len(ds.clean_test)
        runs = [json.loads(p.read_text()) for p in sorted((R/'clhe'/name).glob('seed*.json'))]
        per = np.mean([r['queries'] for r in runs], 0);assert len(per) == len(ds.test_pairs)
        f = np.bincount(np.concatenate([np.unique(b) for b in ds.train]), minlength=ds.n_items)
        warm = np.array([all(f[x] > 0 for x in set(s) | set(g)) for s, g in ds.test_pairs])
        assert int(warm.sum()) == check[name]['groups']['warm']['n']
        out[name] = dict(runs=len(runs), preset=runs[0]['preset'], all=float(per[:, 1].mean()),
                         warm=dict(n=int(warm.sum()), ndcg20=float(per[warm, 1].mean())),
                         cold=dict(n=int((~warm).sum()), ndcg20=float(per[~warm, 1].mean())))
        print(name, out[name], flush=True)
    write_json(R/'clhe_cold.json', dict(note=__doc__, datasets=out))


if __name__ == '__main__':
    main()
