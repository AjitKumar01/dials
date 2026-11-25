"""Figures of paper_v2 from stored results (no training). Run from the repository root:

    python paper_v2/figdata/make_figs.py

figs/training_size.pdf  test NDCG@20 against training bundles on Electronic, Food and Clothing: the best simple
                        baseline, TRIAD untrained and TRIAD trained (results/training_size.json, written by
                        python -m analysis.tables training_size). Filled markers: TRIAD significantly ahead of / behind
                        the best simple baseline (Holm over the 15 conditions).
figs/data_properties.pdf  the two properties of the data, on the clean test queries (results/data_facts_test.json,
                        written by python -m analysis.data_facts_test): (a) share of queries with a seed that no training
                        bundle holds, at least one and all; (b) share of hidden items that share a label with a seed,
                        against the share of catalogue items that would; (c) Recall@20 of plain feature cosine.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.loads(Path('results/training_size.json').read_text())
FIG = Path('paper_v2/figs')
NAMES = dict(electronic='Electronic', food='Food', clothing='Clothing')
CONDS = ('n50', 'n100', 'n300', 'n600', 'full')
tests = {t['condition']: t for t in R['tests']['triad']}

fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
for ax, (name, label) in zip(axes, NAMES.items()):
    rows = [R['rows'][f'{name}:{c}'] for c in CONDS];x = [r['bundles'] for r in rows]
    ax.plot(x, [r['best_simple'] for r in rows], '-', color='#888888', lw=1.8, marker='o', ms=3.5, label='best simple baseline')
    ax.plot(x, [r['untrained'] for r in rows], '--', color='#D9541E', lw=1.5, label='TRIAD untrained')
    ax.plot(x, [r['triad'] for r in rows], '-', color='#D9541E', lw=1.8, marker='o', ms=3.5, label='TRIAD trained')
    for c, r in zip(CONDS, rows):
        t = tests[f'{name}:{c}']
        if t['p_holm'] < .05:
            ax.plot(r['bundles'], r['triad'], marker='^' if t['delta'] > 0 else 'v', ms=8, color='#D9541E', mec='black', mew=.6)
    ax.set_xscale('log');ax.set_xticks(x, [str(v) for v in x], fontsize=7.5);ax.minorticks_off();ax.set_title(label, fontsize=9.5)
    ax.set_xlabel('training bundles', fontsize=8.5);ax.tick_params(labelsize=7.5)
    ax.spines[['top', 'right']].set_visible(False);ax.grid(axis='y', color='#eeeeee')
axes[0].set_ylabel('test NDCG@20', fontsize=8.5);axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout();fig.savefig(FIG/'training_size.pdf');fig.savefig(FIG/'training_size.png', dpi=160)
print('wrote', FIG/'training_size.pdf')


# ---- data properties ---------------------------------------------------------------------------------------------
D = json.loads(Path('results/data_facts_test.json').read_text())['datasets']
ORDER = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify')
SHORT = dict(electronic='Electronic', food='Food', clothing='Clothing', pog_dense='POG_DENSE', pog='POG',
             spotify_sparse='Spotify_sp.', spotify='Spotify')
BLUE, ORANGE = '#2a78d6', '#eb6834'
facts = {n: D[n] for n in ORDER}
x = range(len(ORDER));w = .38
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))


def bars(ax, left, right, labels, title, ylabel, top):
    fmt = lambda v: '0' if v == 0 else ('<0.1' if v < .1 else (f'{v:.1f}' if v < 1 else f'{v:.0f}'))
    shift = w/2 if right is not None else 0
    ax.bar([i - shift for i in x], left, w if right is not None else .55, color=BLUE, label=labels[0], edgecolor='white', linewidth=.6)
    for i, v in enumerate(left):
        ax.text(i - shift, v + top*.015, fmt(v), ha='center', va='bottom', fontsize=6)
    if right is not None:
        ax.bar([i + w/2 for i in x], right, w, color=ORANGE, label=labels[1], edgecolor='white', linewidth=.6)
        for i, v in enumerate(right):
            ax.text(i + w/2, v + top*.015, fmt(v), ha='center', va='bottom', fontsize=6)
    ax.set_xticks(list(x), [SHORT[n] for n in ORDER], rotation=40, ha='right', fontsize=7.5)
    ax.set_ylim(0, top);ax.set_title(title, fontsize=9);ax.set_ylabel(ylabel, fontsize=8);ax.tick_params(labelsize=7.5)
    ax.spines[['top', 'right']].set_visible(False);ax.grid(axis='y', color='#eeeeee');ax.set_axisbelow(True)
    for k in (2.5, 4.5):ax.axvline(k, color='#dddddd', lw=.8)


bars(axes[0], [100*facts[n]['queries_with_unseen_seed'] for n in ORDER], [100*facts[n]['queries_all_unseen'] for n in ORDER],
     ('at least one seed', 'all seeds'), '(a) queries with never-bundled seeds', '% of test queries', 100)
axes[0].legend(fontsize=7, frameon=False, loc='upper left')
bars(axes[1], [100*facts[n]['labels'][0]['targets_sharing'] for n in ORDER],
     [100*facts[n]['labels'][0]['catalogue_sharing'] for n in ORDER],
     ('hidden items', 'catalogue items'), '(b) share a seed\'s category / artist', '% of items', 45)
axes[1].legend(fontsize=7, frameon=False, loc='upper left')
bars(axes[2], [100*facts[n]['cosine_recall20'] for n in ORDER], None, ('',), '(c) plain feature cosine',
     'test Recall@20 (%)', 45)
fig.tight_layout();fig.savefig(FIG/'data_properties.pdf');fig.savefig(FIG/'data_properties.png', dpi=160)
print('wrote', FIG/'data_properties.pdf')
