"""Cold-start and seed-removal checks, in the spirit of CLHE's Sec. 3.4 (WSDM 2024), on the clean test queries.
No training and no retuning: TRIAD = runs/<dataset> (seeds 0-2, per-query mean); best simple = the setting chosen on
validation in runs/baselines/<dataset>.json (python -m triad baselines). Both are first checked to reproduce the
clean-test NDCG@20 of results/table.json (python -m analysis.tables results).
  cold item   f = 0: in no training bundle (CLHE's definition)
  hybrid      all clean test queries
  warm        clean test queries in which every seed and every hidden item has f > 0
  cold        the other clean test queries (at least one cold seed or cold hidden item)
  drop r      every seed removed with probability r (r = .25, .5), at least one kept (a random one if all were
              drawn); hidden items unchanged; the same reduced queries for both methods (rng seed 20261123)
Paired test per row (bootstrap + sign flips), Holm over all 28 rows (7 datasets x warm, cold, drop .25, drop .5).
Writes results/cold.json.
VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. python -m analysis.cold [dataset ...]
"""
import triad  # pins BLAS threads before numpy loads
import json, sys
from pathlib import Path
import numpy as np
import torch
from triad import data, train as T
from triad.baselines import Baselines, base_rate
from triad.evaluate import evaluate
from triad.statistics import paired_comparison, holm_adjust
from triad.train import write_json

CACHE = Path('cache');RUNS = Path('runs');R = Path('results')
NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify');RATES = (.25, .5)


def best_simple(name):
    """(method, params, lambda) of the best simple baseline chosen on validation."""
    r = json.loads((RUNS/'baselines'/f'{name}.json').read_text());method = r['best_simple']['method']
    base = method.replace('+prior', '');m = r['methods'][base]
    return (base, m['prior']['params'], m['prior']['lam']) if method.endswith('+prior') else (base, m['params'], 0.)


def drop(pairs, r, rng):
    out = []
    for s, g in pairs:
        s = sorted(set(s));keep = [x for x in s if rng.random() >= r]
        out.append((keep or [s[rng.integers(len(s))]], g))
    return out


def main(names):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    table = json.loads((R/'table.json').read_text())['rows']
    rows, tests = {}, []
    for name in names:
        ds = data.load(name, CACHE);clean = list(ds.clean_test);pairs = [ds.test_pairs[i] for i in clean]
        bl = Baselines(ds);method, params, lam = best_simple(name)
        if lam:bl.prior = base_rate(bl.f, ds.val_pairs);bl.lam = lam
        f = bl.f
        # full queries: reproduction check
        simple = {'full': np.array([q['ndcg20'] for q in bl.evaluate(method, params, pairs)])}
        tri = {'full': np.mean([[q['ndcg20'] for q in json.loads((RUNS/name/f'seed{s}'/'test.json').read_text())['queries']]
                                for s in (0, 1, 2)], 0)[clean]}
        print(f"{name}: simple {method} lam {lam} reproduced {simple['full'].mean():.6f} vs table {table[name]['best_simple']:.6f}; "
              f"TRIAD {tri['full'].mean():.6f} vs {table[name]['triad']:.6f}", flush=True)
        assert abs(simple['full'].mean()-table[name]['best_simple']) < 1e-5, 'baseline not reproduced'
        assert abs(tri['full'].mean()-table[name]['triad']) < 1e-5, 'TRIAD not reproduced'
        # seed removal
        models = [T.load(ds, RUNS/name/f'seed{s}', CACHE) for s in (0, 1, 2)];changed = {}
        for r in RATES:
            red = drop(pairs, r, np.random.default_rng(20261123))
            changed[r] = float(np.mean([len(set(a)) < len(set(b)) for (a, _), (b, _) in zip(red, pairs)]))
            simple[r] = np.array([q['ndcg20'] for q in bl.evaluate(method, params, red)])
            with torch.no_grad():
                tri[r] = np.mean([[q['ndcg20'] for q in evaluate(m, red)['queries']] for m in models], 0)
            print(f'  drop {r}: TRIAD {tri[r].mean():.4f} simple {simple[r].mean():.4f} ({changed[r]:.0%} of queries changed)', flush=True)
        del models
        # warm / cold groups (full queries)
        warm = np.array([all(f[x] > 0 for x in set(s)) and all(f[x] > 0 for x in set(g)) for s, g in pairs])
        cold_seed = np.array([any(f[x] == 0 for x in set(s)) for s, _ in pairs])
        cold_target = np.array([any(f[x] == 0 for x in set(g)) for _, g in pairs])
        groups = dict(hybrid=('full', np.ones(len(pairs), bool)), warm=('full', warm), cold=('full', ~warm),
                      **{f'drop{int(r*100)}': (r, np.ones(len(pairs), bool)) for r in RATES})
        rows[name] = dict(method=method, lam=lam, queries=len(pairs), warm_share=float(warm.mean()),
                          cold_seed_share=float(cold_seed.mean()), cold_target_share=float(cold_target.mean()),
                          changed=changed, groups={})
        for g, (cond, mask) in groups.items():
            a, b = tri[cond][mask], simple[cond][mask]
            rows[name]['groups'][g] = dict(n=int(mask.sum()), triad=float(a.mean()), simple=float(b.mean()))
            if g != 'hybrid':
                t = paired_comparison(a, b, [str(i) for i in range(len(a))], bootstrap=2000, permutations=10000, seed=20261121)
                tests.append(dict(t, key=f'{name}:{g}'))
    holm_adjust(tests, len(tests))
    write_json(R/'cold.json', dict(note=__doc__, rows=rows, tests=tests))
    tb = {t['key']: t for t in tests}
    for name, r in rows.items():
        print(f"\n{name} ({r['method']}; {r['queries']} queries; warm {r['warm_share']:.0%}, cold seed {r['cold_seed_share']:.0%}, cold target {r['cold_target_share']:.0%})")
        for g, v in r['groups'].items():
            t = tb.get(f'{name}:{g}')
            extra = f"  diff {t['delta']:+.4f} [{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.4f}" if t else ''
            print(f"  {g:7s} n={v['n']:5d}  TRIAD {v['triad']:.4f}  simple {v['simple']:.4f}{extra}")


if __name__ == '__main__':
    main(sys.argv[1:] or NAMES)
