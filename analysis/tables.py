"""The results tables and figures of TRIAD_DESIGN.md. Run from the repository root.

python -m analysis.tables results         full training data, seven datasets: test NDCG@20 on clean test queries (mean
                                          of seeds 0-2 per query) of the best simple baseline (runs/baselines, python -m
                                          triad baselines), the original TRIAD and TRIAD (runs/<dataset>), with paired
                                          tests (bootstrap + sign flips, one cluster per query); Holm over the 10 tests
                                          of the first five datasets, and over the 2 tests of each Spotify set.
                                          Writes results/table.json.
python -m analysis.tables training_size   Electronic, Food, Clothing with nested random subsets of 50, 100, 300, 600 and
                                          all training bundles (one fixed permutation per dataset, seed 20261113): the
                                          best simple baseline, TRIAD untrained, the original TRIAD and TRIAD. Trains
                                          what is missing (TRIAD: runs/training_size/<dataset>_<n>; baselines:
                                          runs/training_size/baselines). Paired tests against the best simple baseline,
                                          Holm over the 15 conditions per model. Writes results/training_size.json and
                                          figures/design/training_size.png.
python -m analysis.tables knockouts       the knock-out heatmap figures/design/knockouts.png from results/readings.json.

The original TRIAD is the earlier version of the model (rank-64 content maps, raw-count start, no per-query scaling).
Its runs are made with the code of branch archive/original-triad (python -m triad train / test there) and read from
--original (default runs_original_triad: <dataset>/seed<s>/test.json, training_size/<dataset>_<n>/seed<s>/test.json).
"""
import triad  # pins BLAS threads before numpy loads
import dataclasses, json, sys
from pathlib import Path
import numpy as np
import torch
from triad import data
from triad import train as T
from triad.baselines import run as baselines
from triad.evaluate import evaluate
from triad.statistics import paired_comparison, holm_adjust
from triad.train import write_json

CACHE = Path('cache');RUNS = Path('runs');R = Path('results');FIG = Path('figures/design')
NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify');SMALL = ('electronic', 'food', 'clothing')
SIZES = (50, 100, 300, 600);SEED = 20261113;SEEDS = (0, 1, 2)
LABEL = dict(electronic='Electronic', food='Food', clothing='Clothing', pog_dense='POG_DENSE', pog='POG',
             spotify_sparse='Spotify_sparse', spotify='Spotify')


def per_query(root):
    """NDCG@20 per test query, mean over seeds 0-2 (from <root>/seed<s>/test.json)."""
    return np.mean([[q['ndcg20'] for q in json.loads((root/f'seed{s}'/'test.json').read_text())['queries']] for s in SEEDS], 0)


def best_simple(report):
    """Per-query clean-test NDCG@20 and name of the best simple baseline of a triad.baselines report."""
    method = report['best_simple']['method'];m = report['methods'][method.replace('+prior', '')]
    return np.array((m['prior'] if method.endswith('+prior') else m)['queries'])[:, 1], method


def compare(first, second):
    return paired_comparison(first, second, [str(i) for i in range(len(first))], bootstrap=2000, permutations=10000, seed=20261121)


def results(original):
    rows, tests = {}, []
    for name in NAMES:
        clean = data.load(name, CACHE).clean_test
        s, method = best_simple(json.loads((RUNS/'baselines'/f'{name}.json').read_text()))
        f, o = per_query(RUNS/name)[clean], per_query(original/name)[clean]
        rows[name] = dict(queries=len(clean), best_simple=float(s.mean()), best_simple_method=method, original=float(o.mean()), triad=float(f.mean()))
        group = name if name.startswith('spotify') else 'main'
        for versus, base in (('best_simple', s), ('original', o)):tests.append(dict(compare(f, base), dataset=name, versus=versus, group=group))
    for g in {t['group'] for t in tests}:
        sub = [t for t in tests if t['group'] == g];holm_adjust(sub, len(sub))
    write_json(R/'table.json', dict(note=__doc__, rows=rows, tests=tests))
    by = {(t['dataset'], t['versus']): t for t in tests}
    print(f"{'dataset':15s} {'queries':>7s} {'best simple':>12s} {'original':>9s} {'TRIAD':>7s} {'-simple':>9s} {'-original':>10s}  best simple")
    for n, r in rows.items():
        cell = lambda v: f"{r['triad'] - r[v]:+.4f}{'*' if by[(n, v)]['p_holm'] < .05 else ' '}"
        print(f"{n:15s} {r['queries']:7d} {r['best_simple']:12.4f} {r['original']:9.4f} {r['triad']:7.4f} {cell('best_simple'):>9s} {cell('original'):>10s}  {r['best_simple_method']}")
    print('* significant after Holm')


def subset(ds, k):
    """The first k bundles of one fixed permutation of the training bundles, in their original order."""
    order = np.random.default_rng(SEED).permutation(len(ds.train))
    return dataclasses.replace(ds, train=[ds.train[i] for i in sorted(order[:k])])


def training_size(original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
    rows, tests = {}, dict(triad=[], original=[])
    for name in SMALL:
        full = data.load(name, CACHE)
        for cond in [f'n{k}' for k in SIZES] + ['full']:
            ds = full if cond == 'full' else subset(full, int(cond[1:]));key = f'{name}:{cond}'
            # TRIAD: trained with the repository's code (missing runs are trained here).
            root = RUNS/name if cond == 'full' else RUNS/'training_size'/f'{name}_{cond}'
            for seed in SEEDS:
                run = root/f'seed{seed}'
                T.fit(ds, seed, run, CACHE, config=T.config_for(name))
                if not (run/'test.json').exists():write_json(run/'test.json', evaluate(T.load(ds, run, CACHE), ds.test_pairs))
            # Baselines on the same training bundles.
            path = RUNS/'baselines'/f'{name}.json' if cond == 'full' else RUNS/'training_size'/'baselines'/f'{name}_{cond}.json'
            if not path.exists():write_json(path, baselines(ds, log=lambda *_: None))
            s, method = best_simple(json.loads(path.read_text()))
            # Untrained: the start does not depend on the seed (every random layer feeds a zero layer).
            torch.manual_seed(0);start = np.array([q['ndcg20'] for q in evaluate(T.build(ds, CACHE, T.config_for(name)), ds.test_pairs)['queries']])
            f = per_query(root);o = per_query(original/name if cond == 'full' else original/'training_size'/f'{name}_{cond}')
            rows[key] = dict(bundles=len(ds.train), best_simple=float(s.mean()), best_simple_method=method, untrained=float(start.mean()),
                             original=float(o.mean()), triad=float(f.mean()),
                             epochs=[json.loads((root/f'seed{s}'/'selected.json').read_text())['best_epoch'] for s in SEEDS])
            for k, v in (('triad', f), ('original', o)):tests[k].append(dict(compare(v, s), condition=key))
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[key].items()}, flush=True)
    for k in tests:holm_adjust(tests[k], len(tests[k]))
    write_json(R/'training_size.json', dict(note=__doc__, rows=rows, tests=tests))
    for k in tests:
        print(k, 'significantly behind / ahead of the best simple baseline:', sum(t['p_holm'] < .05 and t['delta'] < 0 for t in tests[k]), '/',
              sum(t['p_holm'] < .05 and t['delta'] > 0 for t in tests[k]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, name in zip(axes, SMALL):
        rr = [rows[f'{name}:{c}'] for c in [f'n{k}' for k in SIZES] + ['full']];x = [r['bundles'] for r in rr]
        for key, colour, style, label in (('best_simple', '#888888', '-', 'best simple baseline'), ('original', '#2A7AB0', '-', 'original TRIAD'),
                                          ('untrained', '#D9541E', '--', 'TRIAD untrained'), ('triad', '#D9541E', '-', 'TRIAD trained')):
            ax.plot(x, [r[key] for r in rr], style, color=colour, lw=2, marker='o', ms=4, label=label)
        ax.set_xscale('log');ax.set_xticks(x, [str(v) for v in x], fontsize=8);ax.minorticks_off();ax.set_title(LABEL[name], fontsize=10)
        ax.set_xlabel('training bundles', fontsize=9);ax.tick_params(labelsize=8);ax.spines[['top', 'right']].set_visible(False);ax.grid(axis='y', color='#eee')
    axes[0].set_ylabel('test NDCG@20', fontsize=9);axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout();FIG.mkdir(parents=True, exist_ok=True);fig.savefig(FIG/'training_size.png', dpi=180);print('wrote', FIG/'training_size.png')


ROWS = [('content_maps_at_start', 'learned content matrices'), ('symmetric_content', 'antisymmetric part of M_zz'),
        ('no_partner_history', 'partner histories'), ('no_feature_refinement', 'refined features'),
        ('no_label_rule', 'label rule'), ('exponents_at_start', 'learned count exponents'),
        ('shared_exponent', 'separate seed / candidate exponents'), ('no_prior', 'popularity prior'), ('fixed_gate', 'per-query gate')]


def knockouts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = json.loads((R/'readings.json').read_text())['datasets'];names = [n for n in NAMES if d.get(n)]
    loss = np.array([[np.mean([100*(s['knockouts']['trained'] - s['knockouts'][k])/s['knockouts']['trained'] for s in d[n]])
                      if d[n][0]['knockouts'][k] is not None else np.nan for n in names] for k, _ in ROWS])
    fig, ax = plt.subplots(figsize=(1.25*len(names) + 2.6, 4.2))
    ax.imshow(np.clip(loss, 0, 60), cmap='Blues', vmin=0, vmax=60, aspect='auto')
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            ax.text(j, i, f'{round(loss[i, j]):d}', ha='center', va='center', fontsize=8.5, color='white' if loss[i, j] > 35 else '#222')
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], fontsize=8.5)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab in ROWS], fontsize=8.5);ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
    ax.set_title('validation NDCG@20 lost when the pattern is removed (% of trained)', fontsize=9.5)
    fig.tight_layout();FIG.mkdir(parents=True, exist_ok=True);fig.savefig(FIG/'knockouts.png', dpi=180);print('wrote', FIG/'knockouts.png')


if __name__ == '__main__':
    what = sys.argv[1];original = Path(sys.argv[sys.argv.index('--original') + 1]) if '--original' in sys.argv else Path('runs_original_triad')
    dict(results=lambda: results(original), training_size=lambda: training_size(original), knockouts=knockouts)[what]()
