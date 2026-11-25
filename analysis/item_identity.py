"""Item-identity ablation: does TRIAD get better if every item also gets its own learned parameters?

TRIAD has no item IDs: every item is described by its features, labels, counts and partner history. This adds the
per-item parameters that ID-based bundle models (CLHE, RaMen) rely on, to TRIAD, and trains
with the same loop and configuration (triad.train.fit, config_for):

    id_vec       a learned vector e_c per item (n x d) inside the content features: z_c = normalize(x_c + MLP(x_c) + e_c)
    id_vec_bias  id_vec plus a learned score b_c per item, added to the final score

Both start at zero, so the start is TRIAD's (the base model draws the same random numbers: the new parameters are
created after it, as zeros). e_c trains with the content maps (shared_lr), b_c with the small parameters (fast_lr).
The baseline is runs/<dataset> (TRIAD, same seeds and configuration).

Test NDCG@20 on clean test queries (mean of seeds 0-2 per query), paired tests (bootstrap + sign flips, one cluster
per query), variant minus TRIAD, Holm over the 3 datasets within each variant. Writes results/item_identity.json;
runs in runs/item_identity/<variant>/<dataset>/seed<s> (resumed if interrupted).

PYTHONPATH=. python -m analysis.item_identity
"""
import triad  # pins BLAS threads before numpy loads
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from triad import data
from triad import train as T
from triad.evaluate import evaluate
from triad.model import TRIAD
from triad.statistics import paired_comparison, holm_adjust

CACHE = Path('cache');BASE = Path('runs');OUT = BASE/'item_identity';RESULT = Path('results/item_identity.json')
NAMES = ('electronic', 'food', 'clothing');SEEDS = (0, 1, 2);VARIANTS = ('id_vec', 'id_vec_bias')


class TRIAD_ID(TRIAD):
    def __init__(self, features, levels, train, hidden=32, bias=False):
        super().__init__(features, levels, train, hidden)
        self.item_vector = nn.Parameter(torch.zeros(self.n, self.features.shape[1]))   # e_c
        self.item_bias = nn.Parameter(torch.zeros(self.n)) if bias else None            # b_c

    def fast_parameters(self):
        return super().fast_parameters() + ([self.item_bias] if self.item_bias is not None else [])

    def content(self):
        return nn.functional.normalize(self.features + self.encoder(self.features) + self.item_vector, dim=1)

    def forward(self, seeds, bundles=None, cache=None):
        out = super().forward(seeds, bundles, cache)
        return out + self.item_bias if self.item_bias is not None else out


def builder(variant):
    return lambda ds, cache, config=T.CONFIG: TRIAD_ID(ds.features, ds.levels, ds.train, config['hidden'], bias=variant == 'id_vec_bias')


def per_query(root):
    return np.mean([[q['ndcg20'] for q in json.loads((root/f'seed{s}'/'test.json').read_text())['queries']] for s in SEEDS], 0)


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    rows, tests = {}, {v: [] for v in VARIANTS}
    for name in NAMES:
        ds = data.load(name, CACHE);clean = np.array(ds.clean_test);base = per_query(BASE/name)[clean]
        rows[name] = dict(triad=dict(test=float(base.mean()),
                                     validation=float(np.mean([json.loads((BASE/name/f'seed{s}'/'selected.json').read_text())['validation']['ndcg20'] for s in SEEDS])),
                                     epochs=[json.loads((BASE/name/f'seed{s}'/'selected.json').read_text())['best_epoch'] for s in SEEDS]))
        for v in VARIANTS:
            T.build = builder(v)
            for seed in SEEDS:
                run = OUT/v/name/f'seed{seed}'
                T.fit(ds, seed, run, CACHE, config=T.config_for(name))
                if not (run/'test.json').exists():T.write_json(run/'test.json', evaluate(T.load(ds, run, CACHE), ds.test_pairs))
            sel = [json.loads((OUT/v/name/f'seed{s}'/'selected.json').read_text()) for s in SEEDS]
            x = per_query(OUT/v/name)[clean]
            rows[name][v] = dict(test=float(x.mean()), validation=float(np.mean([r['validation']['ndcg20'] for r in sel])),
                                 epochs=[r['best_epoch'] for r in sel], parameters=sel[0]['parameters'])
            tests[v].append(dict(paired_comparison(x, base, [str(i) for i in range(len(base))], bootstrap=2000, permutations=10000, seed=20261121), dataset=name))
    for v in tests:holm_adjust(tests[v], len(tests[v]))
    T.write_json(RESULT, dict(note=__doc__, rows=rows, tests=tests))
    by = {v: {t['dataset']: t for t in ts} for v, ts in tests.items()}
    for name, r in rows.items():
        print(f"{name:11s} TRIAD test {r['triad']['test']:.4f} val {r['triad']['validation']:.4f} epochs {r['triad']['epochs']}")
        for v in VARIANTS:
            t = by[v][name]
            print(f"  {v:12s} test {r[v]['test']:.4f} val {r[v]['validation']:.4f} epochs {r[v]['epochs']} params {r[v]['parameters']:,} "
                  f"delta {t['delta']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] p_holm {t['p_holm']:.3f}")


if __name__ == '__main__':
    main()
