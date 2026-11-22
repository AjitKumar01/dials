"""The fixed taste learned by the antisymmetric part K = (M_zz - M_zz')/2 of TRIAD's content matrix (TRIAD_DESIGN.md
§3.5). Its query-independent piece scores candidate c by t_c = mu' K z_c, mu the catalogue mean of the refined
features z. Per dataset, on validation queries only (never test), seeds 0-2:

  seed_agreement   correlation of t over items between each pair of seeds
  popularity       correlation of t with log(1 + f_c), per seed
  auc_catalogue    AUC of t for validation targets against all other items (.5 = no better than chance), per seed
  shortlist        inside each query's top 100 (ranked by the model with t removed): AUC of t for the targets, and the
                   spread of t against the spread of the rest of the score (standard deviations), per seed
  categories       seed 0: per category with at least 300 shortlist appearances, the mean of t and the share of its
                   shortlist appearances that are targets; and their correlation
  extremes         the 25 items with the highest and the lowest mean t over seeds (item IDs and categories only; item
                   titles are data and stay out of the repository: --titles prints them from the dataset files)

python -m analysis.taste pog_dense [pog ...] [--titles]    writes results/taste.json
"""
import triad  # pins BLAS threads before numpy loads
import json, sys
import numpy as np, torch
from triad import data
from triad.model import TRIAD
from triad.train import write_json
from analysis.readings import CACHE, R, VQ, model_path, blocks, content_from, scaled

OUT = R/'taste.json'


def auc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half)."""
    v = np.concatenate([pos, neg]);r = v.argsort().argsort() + 1.
    return float((r[:len(pos)].sum() - len(pos)*(len(pos) + 1)/2)/(len(pos)*len(neg)))


@torch.no_grad()
def parts(m):
    """(content without the taste, taste t over items) for a trained model."""
    c = m.cache();M = blocks(m, c);z = c['content'];K = (M['zz'] - M['zz'].T)/2;mu = z.mean(0)
    rest = {k: v for k, v in M.items() if k != 'zz'};rest['zz'] = (M['zz'] + M['zz'].T)/2
    return c, content_from(rest, c), ((mu@K)@z.T)


@torch.no_grad()
def shortlist(ds, m, pairs, labels=None):
    """Shortlist AUC and spread ratio of the taste; per-category shortlist appearances and hits."""
    c, base, t = parts(m);aucs, ratios = [], [];L = int(labels.max()) + 1 if labels is not None else 0
    seen, hit = np.zeros(L), np.zeros(L)
    for a in range(0, len(pairs), 128):
        P = pairs[a:a + 128];S = [torch.as_tensor(sorted(set(s))) for s, _ in P]
        den = m.content_scores(S, None, c).abs().amax(1, keepdim=True).clamp_min(1e-6)
        rs = [m.label_scores(i, S) for i in range(m.levels)] + [m.graph_scores(S, None, c)];w = m.gate_weights(S, None)
        tot = w[:, 0, None]*base(S)/den + sum(w[:, j + 1, None]*scaled(p) for j, p in enumerate(rs))
        for i, s in enumerate(S):tot[i, s] = -1e9
        top = tot.topk(100, 1).indices;tt = w[:, 0, None]*t[None, :]/den
        for i, (s, g) in enumerate(P):
            ti = top[i].numpy();pos = np.isin(ti, list(g));v = tt[i, ti].numpy()
            if pos.any() and (~pos).any():aucs.append(auc(v[pos], v[~pos]))
            ratios.append(float(v.std()/tot[i, ti].numpy().std()))
            if labels is not None:np.add.at(seen, labels[ti], 1);np.add.at(hit, labels[ti[pos]], 1)
    return float(np.mean(aucs)), float(np.mean(ratios)), seen, hit


def main(names, titles=False):
    report = json.loads(OUT.read_text())['datasets'] if OUT.exists() else {}
    for name in names:
        ds = data.load(name, CACHE);pairs = ds.val_pairs[:VQ];lab = np.asarray(ds.levels[0]);T, rows = [], []
        target = np.zeros(ds.n_items, bool)
        for _, g in ds.val_pairs:target[list(g)] = True
        for seed in (0, 1, 2):
            m = TRIAD(ds.features, ds.levels, ds.train);m.load_state_dict(torch.load(model_path(name, seed), map_location='cpu'));m.eval()
            _, _, t = parts(m);t = t.numpy();T.append(t);f = m.frequency.numpy()
            a, ratio, seen, hit = shortlist(ds, m, pairs, lab if seed == 0 else None)
            row = dict(seed=seed, popularity=float(np.corrcoef(t, np.log1p(f))[0, 1]), auc_catalogue=auc(t[target], t[~target]),
                       shortlist_auc=a, shortlist_spread_ratio=ratio)
            if seed == 0:
                keep = np.flatnonzero(seen >= 300);mean_t = np.array([t[lab == l].mean() for l in keep]);rate = hit[keep]/seen[keep]
                row['categories'] = dict(correlation=float(np.corrcoef(mean_t, rate)[0, 1]),
                                         rows=[dict(category=int(l), taste=float(x), hit_rate=float(y), appearances=int(seen[l])) for l, x, y in zip(keep, mean_t, rate)])
            rows.append(row);print(name, seed, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
        T = np.array(T);order = T.mean(0).argsort()
        report[name] = dict(seed_agreement=np.corrcoef(T)[np.triu_indices(3, 1)].tolist(), seeds=rows,
                            highest=[dict(item=int(i), category=int(lab[i])) for i in order[::-1][:25]],
                            lowest=[dict(item=int(i), category=int(lab[i])) for i in order[:25]])
        print(name, 'seed agreement', np.round(report[name]['seed_agreement'], 2).tolist(), flush=True)
        if titles:
            info = json.loads((data.DATA_ROOT/'bundlemllm'/'datasets'/name/'item_info.json').read_text())
            for side in ('highest', 'lowest'):print(side, [info[str(r['item'])].get('title', '')[:30] for r in report[name][side]])
        write_json(OUT, dict(note=__doc__, datasets=report))


if __name__ == '__main__':
    main([a for a in sys.argv[1:] if not a.startswith('--')], titles='--titles' in sys.argv)
