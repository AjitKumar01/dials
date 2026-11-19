"""TRIAD without the unseen-item bonus: at test time rho(log(1 + f)) of every item with f = 0 is replaced by rho(log 2),
its value for an item seen once, so items held only by validation or test bundles get no advantage from being in the
catalogue (ARCHITECTURE.md section 14). Seeds 0, 1, 2; writes paper/figdata/unseen_bonus.json with clean per-query
[R@20, N@20] (mean of the seeds) and the share of test targets with f = 0.

PYTHONPATH=. python paper/figdata/unseen_bonus.py [datasets ...]
"""
import triad  # pins BLAS threads before numpy loads
import json, math, sys
from pathlib import Path
import numpy as np, torch
from triad import data, train as T
from triad.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[2];OUT = Path(__file__).parent/'unseen_bonus.json'


@torch.no_grad()
def main(names):
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        ds = data.load(name, ROOT/'cache');keep = sorted(ds.clean_test);runs = []
        for s in (0, 1, 2):
            m = T.load(ds, ROOT/'runs'/name/f'seed{s}', ROOT/'cache');full = m.cache
            def without_bonus(m=m, full=full):
                c = full();once = m.prior(torch.tensor([[math.log(2.)]])).squeeze()
                c['prior'] = torch.where(m.frequency == 0, once, c['prior']);return c
            m.cache = without_bonus;r = evaluate(m, ds.test_pairs)
            runs.append([[r['queries'][i]['recall20'], r['queries'][i]['ndcg20']] for i in keep])
        q = np.mean(runs, 0);f = m.frequency.numpy()
        report[name] = dict(mean=q.mean(0).tolist(), queries=q.tolist(),
                            unseen_target_share=float(np.mean([f[g] == 0 for _, G in ds.test_pairs for g in set(G)])))
        print(name, np.round(q.mean(0), 4), flush=True);OUT.write_text(json.dumps(report))


if __name__ == '__main__':
    main(sys.argv[1:] or ['electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify'])
