"""What each trained parameter encodes (ARCHITECTURE.md §14, "Reading the parameters"), measured on the first 512
validation queries of seeds 0, 1 and 2 (never on test). Writes paper/figdata/parameters.json, one list entry per seed.

  alpha, beta, theta          graph term; *_factor_deg90_vs_1 = deg^-beta (candidate) and deg^-alpha (seed) at the
                              90th-percentile degree, the size of the discount
  rho_bins, rho_vs_base_corr  w_G * mean rho(log(1+f)) per frequency bin against the base rate
                              log(share of validation targets in the bin / share of catalogue items in the bin), the
                              best popularity-only score under a full-catalogue softmax (bins with >= 20 items and
                              targets); only the shape matters, since a softmax ignores a constant shift
  gate_mean, gate_sweep       gate weights (content, labels..., graph); the sweep varies one input around the median
                              query: share of unseen seeds 0/.5/1, |S| = 2/5/20, mean seed log(1+f) = 0..3
  spread_top100               per term, the std of its weighted score over each query's top 100 candidates
  content_block_spread_top100 the same for the four blocks of the content score
                              zz: zbar' A_z'B_z z_c   zh: zbar' A_z'B_h h_c   hz: hbar' A_h'B_z z_c   hh: hbar' A_h'B_h h_c
  Mzz_rel_change              ||A_z'B_z - M0|| / ||M0||, M0 = its initial value 10 [I_64 0; 0 0]
  Mzz_mass_outside_first64    share of ||A_z'B_z||^2 outside the initial 64 x 64 block
  encoder_cos_z_x             mean cosine between refined features Z and input features X
  delta                       delta_l at m = 1, median log|S|, label sizes at the 10/50/90th item percentiles

PYTHONPATH=. python paper/figdata/parameters.py [datasets ...]
"""
import triad  # pins BLAS threads before numpy loads
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from triad import data
from triad import train as T

ROOT = Path(__file__).resolve().parents[2];OUT = Path(__file__).parent/'parameters.json'
Q = 512
EDGES = [0, 1, 2, 3, 5, 10, 20, 50, 1e9]


def seed_report(ds, m):
    c = m.cache();f = m.frequency;f_np = f.numpy();n = m.n
    S = [torch.as_tensor(sorted(set(s))) for s, _ in ds.val_pairs[:Q]]
    w = m.gate_weights(S, None)
    r = {}
    # graph term
    a, b = m.exponents.tolist();r['alpha'], r['beta'], r['theta'] = a, b, m.theta.item()
    deg = m.history_mass.numpy();d90 = float(np.percentile(deg[deg > 0], 90))
    r['deg90'] = d90;r['cand_factor_deg90_vs_1'] = d90**-b;r['seed_factor_deg90_vs_1'] = d90**-a
    # rho against the base rate
    tg = np.array([f_np[g] for _, t in ds.val_pairs for g in t]);rho = c['prior'].numpy();wg = w[:, -1].mean().item()
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        items = (f_np >= lo) & (f_np < hi);k = ((tg >= lo) & (tg < hi)).sum()
        if items.sum() >= 20 and k >= 20:
            bins.append(dict(f=f'{int(lo)}-{int(hi)-1}' if hi < 1e8 else f'{int(lo)}+', items=int(items.sum()), targets=int(k),
                             base=float(np.log((k/len(tg))/(items.sum()/n))), prior=float(wg*rho[items].mean())))
    r['rho_bins'] = bins
    if len(bins) >= 3:
        r['rho_vs_base_corr'] = float(np.corrcoef([x['base'] for x in bins], [x['prior'] for x in bins])[0, 1])
    # gate
    st = torch.tensor([[math.log(len(s)), f[s].log1p().mean().item(), (f[s] == 0).float().mean().item()] for s in S])
    med = st.median(0).values;r['gate_mean'] = w.mean(0).tolist();sweep = {}
    for col, lab, vals in ((2, 'unseen', [0, .5, 1]), (0, 'logS', [math.log(2), math.log(5), math.log(20)]), (1, 'logf', [0, 1, 2, 3])):
        rows = med.repeat(len(vals), 1);rows[:, col] = torch.tensor(vals, dtype=torch.float)
        if lab == 'unseen':rows[:, 1] = rows[:, 1]*(1 - rows[:, 2])  # unseen seeds add log(1 + 0) = 0
        sweep[lab] = dict(at=vals, w=torch.nn.functional.softplus(m.gate(rows)).tolist())
    r['gate_sweep'] = sweep
    # how much each term reorders the shortlist
    parts = [m.content_scores(S, None, c)] + [m.label_scores(i, S) for i in range(m.levels)] + [m.graph_scores(S, None, c)]
    tot = sum(w[:, j, None]*p for j, p in enumerate(parts))
    for i, s in enumerate(S):tot[i, s] = -1e9
    top = tot.topk(100, 1).indices
    r['spread_top100'] = [float((w[:, j, None]*p).gather(1, top).std(1).mean()) for j, p in enumerate(parts)]
    # the four blocks of the content score
    z, h = c['content'], c['history']
    zb = torch.stack([z[s].mean(0) for s in S]);hb = torch.stack([h[s].mean(0) for s in S])
    A_z, B_z, A_h, B_h = m.query_content, m.key_content, m.query_history, m.key_history
    blocks = dict(zz=(zb@A_z.T)@(z@B_z.T).T, zh=(zb@A_z.T)@(h@B_h.T).T, hz=(hb@A_h.T)@(z@B_z.T).T, hh=(hb@A_h.T)@(h@B_h.T).T)
    r['content_block_spread_top100'] = {k: float((w[:, 0, None]*v).gather(1, top).std(1).mean()) for k, v in blocks.items()}
    M = A_z.T@B_z;M0 = 10*torch.eye(M.shape[0]);M0[64:, 64:] = 0
    r['Mzz_rel_change'] = float((M - M0).norm()/M0.norm())
    r['Mzz_mass_outside_first64'] = float(M[64:, :].norm()**2 + M[:64, 64:].norm()**2)/float(M.norm()**2)
    r['encoder_cos_z_x'] = float((z*m.features).sum(1).mean())
    # delta_l
    r['delta'] = []
    for i in range(m.levels):
        sizes = getattr(m, f'label_size{i}');lab = getattr(m, f'label{i}')
        qs = torch.quantile(sizes[lab], torch.tensor([.1, .5, .9]))
        x3 = torch.stack([torch.full((3,), math.log(2)), torch.full((3,), med[0].item()), qs], 1)
        r['delta'].append(dict(size_q10_50_90=[round(math.expm1(v)) for v in qs.tolist()],
                               delta=m.same_label[i](x3).squeeze(1).tolist(), w=w[:, 1 + i].mean().item()))
    return r


@torch.no_grad()
def main(names):
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        ds = data.load(name, ROOT/'cache');report[name] = []
        for seed in (0, 1, 2):
            run = ROOT/'runs'/name/f'seed{seed}'
            if (run/'model.pt').exists():report[name].append(seed_report(ds, T.load(ds, run, ROOT/'cache')))
        print(name, 'done', flush=True)
        OUT.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['electronic', 'food', 'clothing', 'dunnhumby', 'pog_dense', 'pog', 'spotify_sparse', 'spotify'])
