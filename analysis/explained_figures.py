"""Figures for TRIAD_EXPLAINED.md, drawn from saved results only (nothing is retrained).

  results/readings.json, results/readings_extra.json, results/taste.json   (python -m analysis.readings, extra, taste)
  results/table.json, results/cold.json                                      (python -m analysis.tables results, analysis.cold)
  results/item_identity.json, runs/item_identity/                            (python -m analysis.item_identity)
  runs/pog_dense/seed0/model.pt                                              (the worked example, one real test query)

The frequency groups of the prior figure are recomputed from the data with the rule of analysis/readings.py and checked
against the saved base rates. Writes figures/explained/*.png.

PYTHONPATH=. python -m analysis.explained_figures
"""
import triad  # pins BLAS threads before numpy loads
import json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from triad import data, train as T

CACHE = Path('cache');RUNS = Path('runs');R = Path('results');OUT = Path('figures/explained')
NAMES = ('electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify')
LABEL = dict(electronic='Electronic', food='Food', clothing='Clothing', pog_dense='POG_DENSE', pog='POG',
             spotify_sparse='Spotify_sparse', spotify='Spotify')
DOMAIN = dict(electronic='e-commerce', food='e-commerce', clothing='e-commerce', pog_dense='outfits', pog='outfits',
              spotify_sparse='playlists', spotify='playlists')
EDGES = [0, 1, 2, 3, 5, 10, 20, 50, 1e9];GROUP = ['0', '1', '2', '3-4', '5-9', '10-19', '20-49', '50+']

# Palette (dataviz reference palette, light mode; validated with scripts/validate_palette.js).
SURFACE = '#fcfcfb';INK = '#0b0b0b';INK2 = '#52514e';GRID = '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'               # categorical slots 1-3 (all pairs pass)
TERM = dict(content='#6250d6', label='#1baf7a', graph='#008300')    # the architecture figure's term hues (all pairs pass)

plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'font.size': 9, 'axes.titlesize': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6,
                     'axes.axisbelow': True, 'legend.frameon': False})


def readings():return json.loads((R/'readings.json').read_text())['datasets']


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True);fig.savefig(OUT/name, dpi=170, bbox_inches='tight');plt.close(fig)
    print('wrote', OUT/name)


def domain_axis(ax, names, y=-0.22):
    """Dataset names as ticks, with the domain written under each group."""
    ax.set_xticks(range(len(names)), [LABEL[n] for n in names], rotation=0, fontsize=8)
    starts = {}
    for i, n in enumerate(names):starts.setdefault(DOMAIN[n], []).append(i)
    for d, idx in starts.items():
        ax.text(np.mean(idx), y, d, transform=ax.get_xaxis_transform(), ha='center', va='top', color=INK2, fontsize=8.5, style='italic')
    for k in list(starts.values())[:-1]:ax.axvline(k[-1] + .5, color=GRID, lw=1.2, zorder=0)


# ---------------------------------------------------------------------------------------------------------------------
# 1. The worked example: one real POG_DENSE test query, every part of the score.
# ---------------------------------------------------------------------------------------------------------------------
def worked_example():
    name = 'pog_dense';run = RUNS/name/'seed0';torch.set_num_threads(2)
    ds = data.load(name, CACHE);m = T.load(ds, run, CACHE)
    per = [q['ndcg20'] for q in json.loads((run/'test.json').read_text())['queries']]
    clean = list(ds.clean_test);mean = np.mean([per[i] for i in clean])
    qi = min(clean, key=lambda i: abs(per[i] - mean))                      # the most typical query, by a fixed rule
    seeds, truth = ds.test_pairs[qi];seeds = sorted(set(seeds));truth = sorted(set(truth))
    with torch.no_grad():
        cache = m.cache();S = [torch.tensor(seeds)]
        parts = [m.content_scores(S, None, cache), m.label_scores(0, S), m.graph_scores(S, None, cache)]
        peak = [p.abs().amax(1).item() for p in parts];w = m.gate_weights(S, None)[0].tolist()
        weighted = [w[k]*parts[k][0]/peak[k] for k in range(3)];total = m(S, cache=cache)[0]
    assert torch.allclose(sum(weighted), total, atol=1e-4)
    score = total.clone();score[seeds] = -float('inf');order = torch.argsort(score, descending=True).tolist()
    lab = m.label0.numpy();same = [c for c in order if lab[c] in set(lab[seeds])]
    rows = [(f'rank {k + 1}', c) for k, c in enumerate(order[:10])]
    rows += [(f'rank {order.index(c) + 1}: hidden item', c) for c in truth]
    rows += [(f'rank {order.index(same[0]) + 1:,}: best item in a seed\'s category', same[0])]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for y, (text, c) in enumerate(rows):
        v = [float(x[c]) for x in weighted];left_pos = 0.;left_neg = 0.
        for k, key in enumerate(('content', 'label', 'graph')):
            if v[k] >= 0:ax.barh(y, v[k], left=left_pos, color=TERM[key], height=.62, edgecolor=SURFACE, linewidth=1);left_pos += v[k]
            else:ax.barh(y, v[k], left=left_neg, color=TERM[key], height=.62, edgecolor=SURFACE, linewidth=1);left_neg += v[k]
        ax.plot(sum(v), y, 'o', color=INK, ms=4, zorder=5)
        ax.text(max(left_pos, 0) + .6, y, f'{sum(v):.1f}', va='center', fontsize=8)
    ax.set_yticks(range(len(rows)), [t for t, _ in rows]);ax.invert_yaxis()
    for t in ax.get_yticklabels():
        if 'hidden' in t.get_text():t.set_fontweight('bold')
    ax.axvline(0, color=INK2, lw=.8);ax.set_xlim(-14, 56);ax.grid(axis='y', visible=False)
    ax.set_xlabel('weighted, per-query-scaled score  (= gate weight × part ÷ its largest value in this query)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=TERM[k]) for k in ('content', 'label', 'graph')]
    ax.legend(handles, [f'content  (gate weight {w[0]:.1f})', f'label  (gate weight {w[1]:.1f})', f'graph  (gate weight {w[2]:.1f})'],
              loc='upper center', bbox_to_anchor=(.45, -.13), ncol=3, fontsize=8)
    ax.set_title(f'POG_DENSE test query #{qi} (2 seeds, 1 hidden item): how each candidate\'s score is made', loc='left')
    save(fig, 'worked_example.png')
    return dict(query=qi, seeds=seeds, truth=truth, weights=w, peak=peak,
                rank=order.index(truth[0]) + 1, same_category=len(same), best_same=order.index(same[0]) + 1)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Popularity prior rho against the validation base rate b, per frequency group.
# ---------------------------------------------------------------------------------------------------------------------
def prior_groups(name, bins):
    """Labels of the frequency groups kept by analysis/readings.py (>= 20 items and >= 20 targets); checks b."""
    ds = data.load(name, CACHE);n = ds.n_items
    f = np.bincount(np.concatenate([np.unique(b) for b in ds.train]), minlength=n)
    tg = np.array([f[g] for _, t in ds.val_pairs for g in t]);kept = []
    for j, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        items = (f >= lo) & (f < hi);k = ((tg >= lo) & (tg < hi)).sum()
        if items.sum() >= 20 and k >= 20:kept.append((GROUP[j], math.log((k/len(tg))/(items.sum()/n))))
    assert len(kept) == len(bins) and all(abs(a[1] - b[0]) < 1e-9 for a, b in zip(kept, bins)), name
    return [g for g, _ in kept]


def prior():
    d = readings();fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.2), sharey=True);axes = axes.ravel()
    z = lambda v: (np.asarray(v) - np.mean(v))/np.std(v)
    for ax, name in zip(axes, NAMES):
        bins = [s['prior_vs_base_rate']['bins'] for s in d[name]];groups = prior_groups(name, bins[0]);x = range(len(groups))
        ax.plot(x, z([b for b, _ in bins[0]]), color=INK, lw=2, ls='--', marker='o', ms=4, label='validation base rate $b$')
        for k, s in enumerate(bins):
            ax.plot(x, z([r for _, r in s]), color=BLUE, lw=2 if k == 0 else 1, alpha=1 if k == 0 else .55, marker='o' if k == 0 else None,
                    ms=3.5, label='learned prior $\\rho$ (seed 0; thin: seeds 1, 2)' if k == 0 else None)
        corr = [s['prior_vs_base_rate']['corr'] for s in d[name]]
        rng = None if corr[0] is None else (f'{min(corr):.2f}' if f'{min(corr):.2f}' == f'{max(corr):.2f}' else f'{min(corr):.2f} to {max(corr):.2f}')
        title = LABEL[name] + (f'   r = {rng}' if corr[0] is not None else '   (2 groups only)')
        ax.set_title(title, fontsize=9, loc='left');ax.set_xticks(x, groups, fontsize=7, rotation=45)
        if name == 'pog_dense':ax.annotate('dip at f = 1', xy=(1, z([b for b, _ in bins[0]])[1]), xytext=(2.3, -1.9), fontsize=8,
                                          arrowprops=dict(arrowstyle='->', color=INK2), color=INK2)
    axes[-1].axis('off');h, l = axes[0].get_legend_handles_labels();axes[-1].legend(h, l, loc='center', fontsize=8.5)
    fig.supxlabel('training bundles the item appears in (f), grouped', fontsize=9)
    fig.supylabel('value, standardised per dataset (only the shape matters)', fontsize=9)
    fig.suptitle('The prior learns the base rate of being the missing item from training data alone', x=.01, ha='left', fontsize=10.5)
    fig.tight_layout();save(fig, 'prior.png')


# ---------------------------------------------------------------------------------------------------------------------
# 3. Label rule: sign and shape of delta, and what the data says.
# ---------------------------------------------------------------------------------------------------------------------
def label_rule():
    d = readings();fig, axes = plt.subplots(1, 7, figsize=(12, 2.9), sharey=True)
    for ax, name in zip(axes, NAMES):
        rule = [s['label_rule'][0] for s in d[name]];sizes = rule[0]['size_q10_50_90']
        vals = np.array([np.array(r['m1'])/np.abs(r['m1']).max() for r in rule])
        for k in range(3):ax.plot(range(3), vals[k], color=BLUE, lw=1, alpha=.5, marker='o', ms=3.5)
        agree = bool((np.sign(vals) == np.sign(vals[0])).all())
        if agree:ax.plot(range(3), vals.mean(0), color=BLUE, lw=2.2)
        else:ax.text(1, 0, 'seeds disagree\nfor big labels', ha='center', va='center', fontsize=7.5, color=INK2, bbox=dict(fc=SURFACE, ec='none', pad=1))
        ax.axhline(0, color=INK2, lw=.9);ax.set_ylim(-1.15, 1.15)
        ax.set_xticks(range(3), [f'{s:,}' for s in sizes], fontsize=7.5);ax.set_title(LABEL[name], fontsize=9)
        if agree:ax.text(1, 1.05 if vals.mean() > 0 else -1.05, 'bonus' if vals.mean() > 0 else 'penalty', ha='center',
                         va='top' if vals.mean() > 0 else 'bottom', fontsize=8.5, color=INK2, style='italic')
    axes[0].set_ylabel('δ ÷ its largest |value|\n(one shared seed)')
    fig.supxlabel('items carrying the label: small, medium, large label (10th, 50th, 90th percentile)', fontsize=9, y=-.02)
    fig.suptitle('Label rule δ for one shared seed: sign and shape   (thin lines: seeds 0, 1, 2; thick: their mean, '
                 'drawn where all seeds agree in sign; Spotify sets: artist)', x=.01, ha='left', fontsize=10)
    fig.tight_layout();save(fig, 'label_rule.png')

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    for i, name in enumerate(NAMES):
        f0 = d[name][0]['data_facts']['labels'][0];lift = f0['targets_sharing']/f0['catalogue_sharing']
        if lift > 0:
            ax.bar(i, lift, color=BLUE, width=.6);ax.text(i, lift*1.15, (f'{lift:.0f}×' if lift >= 10 else f'{lift:.3f}×'), ha='center', fontsize=8)
        else:ax.text(i, 1.3e-3, 'never\n(0 of the\ntargets)', ha='center', fontsize=7.5, color=INK2)
    ax.set_yscale('log');ax.set_ylim(1e-3, 1e3);ax.axhline(1, color=INK2, lw=.9)
    ax.text(3, 1.2, 'as often as a random item', fontsize=7.5, color=INK2, ha='center', va='bottom')
    domain_axis(ax, NAMES);ax.grid(axis='x', visible=False)
    ax.set_ylabel('how much more often a hidden item\nshares a label with a seed than a\nrandom catalogue item does (log)')
    ax.set_title('What the validation data says (category; artist on Spotify)', loc='left')
    save(fig, 'label_data.png')


# ---------------------------------------------------------------------------------------------------------------------
# 4. Count exponents.
# ---------------------------------------------------------------------------------------------------------------------
def exponents():
    d = readings();fig, ax = plt.subplots(figsize=(7.8, 3.4))
    for key, color, off, text in (('alpha', BLUE, -.12, 'α (seed side)'), ('beta', ORANGE, .12, 'β (candidate side)')):
        v = np.array([[s['exponents'][key] for s in d[n]] for n in NAMES])
        for i in range(len(NAMES)):ax.plot([i + off]*3, v[i], 'o', color=color, ms=3, alpha=.45)
        ax.plot(np.arange(len(NAMES)) + off, v.mean(1), 'D', color=color, ms=6.5, label=text + ': mean of 3 seeds (dots: seeds)', ls='none')
    ax.axhline(.5, color=INK2, lw=1, ls='--');ax.axhline(0, color=INK2, lw=1)
    ax.text(6.55, .51, 'start: cosine (½)', fontsize=8, color=INK2, ha='right', va='bottom')
    ax.text(6.55, .01, 'raw counts (0)', fontsize=8, color=INK2, ha='right', va='bottom')
    ax.set_ylim(-.12, .62);domain_axis(ax, NAMES);ax.grid(axis='x', visible=False)
    ax.set_ylabel('learned exponent');ax.legend(loc='upper center', bbox_to_anchor=(.5, -.3), ncol=2, fontsize=8)
    ax.set_title(r'Popularity discount of the counts: $N_{sc}\,\deg_s^{-\alpha}\,\deg_c^{-\beta}$', loc='left')
    save(fig, 'exponents.png')


# ---------------------------------------------------------------------------------------------------------------------
# 5. Content matrix: how far it moved and whether it still ranks look-alikes the simple way.
# ---------------------------------------------------------------------------------------------------------------------
def content():
    d = readings();fig, ax = plt.subplots(figsize=(6.6, 4.4))
    off = dict(electronic=(9, 0), food=(9, 6), clothing=(9, -12), pog_dense=(0, 11), pog=(9, -4), spotify_sparse=(9, 4), spotify=(9, -4))
    for name in NAMES:
        x = [s['Mzz_change'] for s in d[name]];y = [s['geometry']['corr_near_neighbours'] for s in d[name]]
        ax.plot(x, y, 'o', color=BLUE, alpha=.4, ms=3.5);ax.plot(np.mean(x), np.mean(y), 'o', color=BLUE, ms=8)
        ax.annotate(f'{LABEL[name]} ({DOMAIN[name]})', (np.mean(x), np.mean(y)), xytext=off[name], textcoords='offset points', fontsize=8,
                    ha='center' if off[name][0] == 0 else 'left', va='bottom' if off[name][0] == 0 else 'center')
    ax.axhline(0, color=INK2, lw=.9);ax.set_xlim(0, 1.45);ax.set_ylim(-.35, 1.1)
    ax.set_xlabel('how far the content matrix moved from its start:  ‖M − 10I‖ / ‖10I‖')
    ax.set_ylabel('agreement with plain similarity among\nlook-alike pairs (correlation)')
    ax.text(.55, 1.0, 'top left: still "look-alikes go together"', fontsize=8, color=INK2)
    ax.text(1.42, -.31, 'bottom right: dropped it', fontsize=8, color=INK2, ha='right')
    ax.set_title('Content matrix M_zz  (big dots: mean of 3 seeds; small: each seed)', loc='left')
    save(fig, 'content.png')


# ---------------------------------------------------------------------------------------------------------------------
# 6. Partner histories, refined features, gate: one value per dataset.
# ---------------------------------------------------------------------------------------------------------------------
def bars(values, ylabel, title, name, fmt, ylim=None, ref=None):
    fig, ax = plt.subplots(figsize=(7.8, 3.1))
    for i, n in enumerate(NAMES):
        v = values[n];ax.bar(i, np.mean(v), color=BLUE, width=.6)
        ax.plot([i]*len(v), v, 'o', color=INK, ms=2.8, alpha=.6)
        top = max(np.mean(v), max(v));ax.text(i, top + (ylim[1] - ylim[0])*.03 if ylim else top*1.04, fmt(np.mean(v)), ha='center', fontsize=8)
    if ref is not None:ax.axhline(ref, color=INK2, lw=.9)
    if ylim:ax.set_ylim(*ylim)
    domain_axis(ax, NAMES);ax.grid(axis='x', visible=False);ax.set_ylabel(ylabel);ax.set_title(title, loc='left')
    save(fig, name)


def partner_history():
    d = readings()
    bars({n: [sum(v for k, v in s['content_blocks']['share'].items() if k != 'zz') for s in d[n]] for n in NAMES},
         'share of the content score\'s spread\nover the shortlist', 'Partner histories: how much of the content score comes from "what the seeds were bundled with"\n(bars: mean of 3 seeds; dots: seeds)',
         'partner_history.png', lambda v: f'{v:.2f}', ylim=(0, .85))


def gate():
    d = readings()
    bars({n: [100*s['gate']['content_rise'] for s in d[n]] for n in NAMES}, 'change in the content weight (%)',
         'Gate: content weight when every seed is new, against no new seeds\n(bars: mean of 3 seeds; dots: seeds)',
         'gate.png', lambda v: f'{v:+.0f}%', ylim=(-10, 100), ref=0)


def taste():
    t = json.loads((R/'taste.json').read_text())['datasets'];names = [n for n in NAMES if n in t]
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    for i, n in enumerate(names):
        a = [s['auc_catalogue'] for s in t[n]['seeds']];b = [s['shortlist_auc'] for s in t[n]['seeds']]
        ax.plot([i, i], [np.mean(a), np.mean(b)], color=GRID, lw=3, zorder=1)
        ax.plot(i, np.mean(a), 'o', color=ORANGE, ms=7, label='whole catalogue' if i == 0 else None, zorder=3)
        ax.plot(i, np.mean(b), 'o', color=BLUE, ms=7, label='inside the shortlist (top 100)' if i == 0 else None, zorder=3)
        ax.text(i + .1, np.mean(b), f'{np.mean(b):.2f}', va='center', fontsize=8);ax.text(i + .1, np.mean(a), f'{np.mean(a):.2f}', va='center', fontsize=8)
    ax.axhline(.5, color=INK2, lw=.9);ax.text(len(names) - .5, .505, 'coin flip', fontsize=8, color=INK2, ha='right', va='bottom')
    ax.set_ylim(.4, .72);domain_axis(ax, names);ax.grid(axis='x', visible=False)
    ax.set_ylabel('AUC: chance a hidden item gets\na higher taste score than a non-target')
    ax.legend(loc='upper left', fontsize=8);ax.set_title('The fixed taste K_M helps only inside the shortlist (mean of 3 seeds)', loc='left')
    save(fig, 'taste.png')


# ---------------------------------------------------------------------------------------------------------------------
# 7. Results: lead over the best simple baseline, overall, warm and cold; seeds removed.
# ---------------------------------------------------------------------------------------------------------------------
def results():
    main = {t['dataset']: t for t in json.loads((R/'table.json').read_text())['tests']}
    cs = {t['key']: t for t in json.loads((R/'cold.json').read_text())['tests']}
    for fname, series, title in (
            ('results_cold.png', (('all clean test queries', BLUE, lambda n: main[n]), ('warm queries', ORANGE, lambda n: cs[f'{n}:warm']),
                                  ('cold queries', AQUA, lambda n: cs[f'{n}:cold'])),
             'TRIAD − best simple baseline, test NDCG@20, with 95% interval: all, warm and cold queries'),
            ('results_seeds.png', (('all seeds', BLUE, lambda n: main[n]), ('25% of seeds removed', ORANGE, lambda n: cs[f'{n}:drop25']),
                                   ('50% removed', AQUA, lambda n: cs[f'{n}:drop50'])),
             'TRIAD − best simple baseline, test NDCG@20, with 95% interval: seeds removed at random')):
        fig, ax = plt.subplots(figsize=(8.4, 3.4))
        for k, (text, color, get) in enumerate(series):
            for i, n in enumerate(NAMES):
                t = get(n);x = i + (k - 1)*.22
                ax.plot([x, x], t['ci95'], color=color, lw=2);ax.plot(x, t['delta'], 'o', color=color, ms=5.5, label=text if i == 0 else None)
        ax.axhline(0, color=INK2, lw=1);domain_axis(ax, NAMES);ax.grid(axis='x', visible=False)
        ax.set_ylabel('difference in NDCG@20\n(above 0: TRIAD ahead)');ax.legend(loc='upper left', fontsize=8, ncol=3)
        ax.set_ylim(-.02, .08);ax.set_title(title, loc='left');save(fig, fname)


# ---------------------------------------------------------------------------------------------------------------------
# 8. Item identities added to TRIAD (analysis/item_identity.py): results and training curves.
# ---------------------------------------------------------------------------------------------------------------------
ID = RUNS/'item_identity';ECOM = ('electronic', 'food', 'clothing')
VARIANT = (('TRIAD', 'TRIAD', BLUE), ('id_vec', 'A: + a vector per item', ORANGE), ('id_vec_bias', 'B: + a vector and a score per item', AQUA))


def history(variant, name, seed):
    root = RUNS/name if variant == 'TRIAD' else ID/variant/name
    h = json.loads((root/f'seed{seed}'/'history.json').read_text())
    return h if isinstance(h, list) else h['history']


def identity():
    r = json.loads((R/'item_identity.json').read_text());tests = r['tests']
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k, (v, text, color) in enumerate(VARIANT[1:]):
        for i, t in enumerate(tests[v]):
            x = i + (k - .5)*.25;ax.plot([x, x], t['ci95'], color=color, lw=2)
            ax.plot(x, t['delta'], 'o', color=color, ms=6, label=text if i == 0 else None)
            ax.text(x + .05, t['delta'], f"{t['delta']:+.3f}" + ('' if t['p_holm'] < .05 else ' (n.s.)'), fontsize=7.5, va='center')
    ax.axhline(0, color=INK2, lw=1);ax.set_xticks(range(3), [LABEL[n] for n in ECOM]);ax.grid(axis='x', visible=False)
    ax.set_xlim(-.5, 2.8);ax.set_ylabel('test NDCG@20 minus TRIAD\n(below 0: worse than TRIAD)')
    ax.legend(loc='lower left', fontsize=8);ax.set_title('Adding item identities to TRIAD: change in test accuracy, 95% interval', loc='left')
    save(fig, 'identity_results.png')

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharex=True)
    for j, name in enumerate(ECOM):
        for v, text, color in VARIANT:
            for s in (0, 1, 2):
                h = history(v, name, s);e = [x['epoch'] for x in h if x['loss'] is not None]
                axes[0, j].plot(e, [x['loss'] for x in h if x['loss'] is not None], color=color, lw=1.3, alpha=.8, label=text if s == 0 else None)
                axes[1, j].plot([x['epoch'] for x in h], [x['ndcg20'] for x in h], color=color, lw=1.3, alpha=.8)
                best = json.loads(((RUNS/name if v == 'TRIAD' else ID/v/name)/f'seed{s}'/'selected.json').read_text())['best_epoch']
                axes[1, j].plot(best, [x['ndcg20'] for x in h if x['epoch'] == best][0], 'o', color=color, ms=5)
        axes[0, j].set_title(LABEL[name], fontsize=9.5);axes[1, j].set_xlabel('epoch')
    axes[0, 0].set_ylabel('training loss\n(lower: fits training bundles)');axes[1, 0].set_ylabel('validation NDCG@20\n(dot: epoch kept)')
    axes[0, 0].legend(loc='lower left', fontsize=7.5)
    fig.suptitle('Training curves, seeds 0, 1 and 2: with item parameters the training loss collapses while validation falls',
                 x=.01, ha='left', fontsize=10)
    fig.tight_layout();save(fig, 'identity_curves.png')


if __name__ == '__main__':
    print(json.dumps(worked_example()))
    prior();label_rule();exponents();content();partner_history();gate();taste();results();identity()
