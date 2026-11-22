"""What TRIAD learns: its trained parameters read on the first 512 validation queries of seeds 0-2 (never test), full
training data. Models: runs/<dataset>/seed<s>/model.pt (python -m triad train). Writes results/readings.json.

Per seed:
  start_distance      relative change of each part from its initial value
  exponents           alpha (seed), beta (candidate), theta; discount factor deg^-alpha, deg^-beta at the 90th-percentile
                      degree (1 = no discount, < 1 discount, > 1 boost)
  gate                mean weights (content, labels..., graph) over the queries; content weight as the share of unseen
                      seeds goes 0 -> 1 (other inputs at the median query), relative to 0
  influence_top100    per term, std over each query's top 100 candidates of its weighted, per-query scaled score: how
                      much the term reorders the shortlist (shares of the total)
  content_blocks      the same for the four blocks of the content score (zz, zh, hz, hh), scaled as the content term
  Mzz_change          ||A_z'B_z - 10 I|| / ||10 I||
  Mzz_asymmetry       ||M - M'|| / ||M|| for M = A_z'B_z (0 = symmetric, the start)
  history_vs_feature  ||A_z'B_h|| / ||A_z'B_z||: weight on partner histories relative to features
  encoder_cos         mean cosine between refined features Z and inputs X
  label_rule          delta_l at m = 1 and m = 2, median log|S|, label sizes at the 10/50/90th percentiles of items
  prior_vs_base_rate  correlation over frequency bins of rho with the base rate of validation targets
                      log(share of targets in bin / share of items in bin) (bins with >= 20 items and targets)
  knockouts           validation NDCG@20 (first VQ validation queries) of the trained model and of copies with one learned
                      behaviour removed: content made symmetric (M_zz -> (M_zz + M_zz')/2), partner-history blocks
                      removed, content maps back to the start 10 I (history removed too), label rule removed, prior
                      removed, one shared exponent (alpha = beta = their mean), exponents back to the start 1/2, gate
                      fixed at its mean over the queries, feature refinement removed (Z = X)
  geometry            the learned content matrix M = A_z'B_z on the refined features Z (start: M = 10 I, so the content score
                      is 10 x cosine): self score z'Mz (start 10), Pearson correlation between z_s'Mz_c and 10 cos(z_s, z_c) over
                      20,000 random item pairs and over each item's 10 nearest neighbours (start 1), share of the
                      squared spectrum of the symmetric part (M + M')/2 on negative eigenvalues (start 0), and
                      ||M - M'|| / ||M + M'|| (antisymmetric against symmetric part)
  data_facts          facts about the data that the simple scorers get wrong, on the same validation queries:
                      share of targets in the top 20 of start cosine content (and of the trained content term), share
                      of targets sharing a label with a seed against the share of catalogue items that do, share of
                      queries with an unseen seed / only unseen seeds

python -m analysis.readings [names]          (from the repository root)
python -m analysis.readings --geometry [names]
"""
import triad  # pins BLAS threads before numpy loads
import copy, json, math, sys
from pathlib import Path
import numpy as np, torch
from triad import data
from triad.evaluate import query_metrics
from triad.model import TRIAD
from triad.train import write_json

CACHE = Path('cache');RUNS = Path('runs');R = Path('results')
Q = 512;VQ = 1000;EDGES = [0, 1, 2, 3, 5, 10, 20, 50, 1e9]


def model_path(name, seed):return RUNS/name/f'seed{seed}'/'model.pt'


def scaled(v):return v/v.abs().amax(1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def seed_report(ds, m):
    init = TRIAD(ds.features, ds.levels, ds.train);r = {}
    r['start_distance'] = {k: float((v - init.state_dict()[k]).norm()/max(init.state_dict()[k].norm(), 1e-6))
                           for k, v in m.state_dict().items() if v.dtype.is_floating_point and v.numel() > 1 or k in ('theta',)}
    c = m.cache();f = m.frequency;n = m.n
    S = [torch.as_tensor(sorted(set(s))) for s, _ in ds.val_pairs[:Q]];w = m.gate_weights(S, None)
    a, b = m.exponents.tolist();deg = m.history_mass.numpy();d90 = float(np.percentile(deg[deg > 0], 90))
    r['exponents'] = dict(alpha=a, beta=b, theta=m.theta.item(), deg90=d90, seed_factor=d90**-a, cand_factor=d90**-b)
    st = torch.tensor([[math.log(len(s)), f[s].log1p().mean().item(), (f[s] == 0).float().mean().item()] for s in S])
    med = st.median(0).values;rows = med.repeat(3, 1);rows[:, 2] = torch.tensor([0., .5, 1.]);rows[:, 1] = rows[:, 1]*(1 - rows[:, 2])
    sweep = torch.nn.functional.softplus(m.gate(rows))
    r['gate'] = dict(mean=w.mean(0).tolist(), unseen_sweep=sweep.tolist(), content_rise=float(sweep[2, 0]/sweep[0, 0] - 1))
    parts = [m.content_scores(S, None, c)] + [m.label_scores(i, S) for i in range(m.levels)] + [m.graph_scores(S, None, c)]
    sp = [scaled(p) for p in parts];tot = sum(w[:, j, None]*p for j, p in enumerate(sp))
    for i, s in enumerate(S):tot[i, s] = -1e9
    top = tot.topk(100, 1).indices
    inf = [float((w[:, j, None]*p).gather(1, top).std(1).mean()) for j, p in enumerate(sp)]
    r['influence_top100'] = dict(raw=inf, share=[x/sum(inf) for x in inf])
    z, h = c['content'], c['history'];zb = torch.stack([z[s].mean(0) for s in S]);hb = torch.stack([h[s].mean(0) for s in S])
    A_z, B_z, A_h, B_h = m.query_content, m.key_content, m.query_history, m.key_history
    scale = parts[0].abs().amax(1, keepdim=True).clamp_min(1e-6)
    blocks = dict(zz=(zb@A_z.T)@(z@B_z.T).T, zh=(zb@A_z.T)@(h@B_h.T).T, hz=(hb@A_h.T)@(z@B_z.T).T, hh=(hb@A_h.T)@(h@B_h.T).T)
    bl = {k: float((w[:, 0, None]*v/scale).gather(1, top).std(1).mean()) for k, v in blocks.items()}
    r['content_blocks'] = dict(raw=bl, share={k: v/sum(bl.values()) for k, v in bl.items()})
    M = A_z.T@B_z;I = 10*torch.eye(M.shape[0])
    r['Mzz_change'] = float((M - I).norm()/I.norm());r['Mzz_asymmetry'] = float((M - M.T).norm()/M.norm())
    r['history_vs_feature'] = float((A_z.T@B_h).norm()/M.norm());r['encoder_cos'] = float((z*m.features).sum(1).mean())
    r['label_rule'] = []
    for i in range(m.levels):
        sizes = getattr(m, f'label_size{i}');lab = getattr(m, f'label{i}');qs = torch.quantile(sizes[lab], torch.tensor([.1, .5, .9]))
        out = {}
        for k in (1, 2):
            x3 = torch.stack([torch.full((3,), math.log(1 + k)), torch.full((3,), med[0].item()), qs], 1);out[f'm{k}'] = m.same_label[i](x3).squeeze(1).tolist()
        r['label_rule'].append(dict(size_q10_50_90=[round(math.expm1(v)) for v in qs.tolist()], **out, gate=w[:, 1 + i].mean().item()))
    tg = np.array([f.numpy()[g] for _, t in ds.val_pairs for g in t]);rho = c['prior'].numpy();fn = f.numpy();bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        items = (fn >= lo) & (fn < hi);k = ((tg >= lo) & (tg < hi)).sum()
        if items.sum() >= 20 and k >= 20:bins.append((float(np.log((k/len(tg))/(items.sum()/n))), float(rho[items].mean())))
    r['prior_vs_base_rate'] = dict(bins=bins, corr=float(np.corrcoef(*zip(*bins))[0, 1]) if len(bins) >= 3 else None)
    return r


def blocks(m, c):
    A_z, B_z, A_h, B_h = m.query_content, m.key_content, m.query_history, m.key_history
    return dict(zz=A_z.T@B_z, zh=A_z.T@B_h, hz=A_h.T@B_z, hh=A_h.T@B_h)


def content_from(M, c):
    """Content score from the four block matrices: zbar' M_zz z_c + zbar' M_zh h_c + hbar' M_hz z_c + hbar' M_hh h_c."""
    z, h = c['content'], c['history']
    def score(S):
        zb = torch.stack([z[s].mean(0) for s in S]);hb = torch.stack([h[s].mean(0) for s in S]);out = 0
        for k, (a, b) in dict(zz=(zb, z), zh=(zb, h), hz=(hb, z), hh=(hb, h)).items():
            if k in M:out = out + (a@M[k])@b.T
        return out
    return score


def ndcg(m, pairs, content=None, gate=None, batch=128):
    c = m.cache();rows = []
    for a in range(0, len(pairs), batch):
        P = pairs[a:a + batch];S = [torch.as_tensor(sorted(set(s))) for s, _ in P]
        parts = [content(S) if content else m.content_scores(S, None, c)] + [m.label_scores(i, S) for i in range(m.levels)] + [m.graph_scores(S, None, c)]
        w = gate.expand(len(S), -1) if gate is not None else m.gate_weights(S, None)
        v = sum(w[:, j, None]*scaled(p) for j, p in enumerate(parts)).numpy()
        rows += [query_metrics(v[k], s, g)['ndcg20'] for k, (s, g) in enumerate(P)]
    return float(np.mean(rows))


@torch.no_grad()
def knockouts(ds, m):
    pairs = ds.val_pairs[:VQ];c = m.cache();M = blocks(m, c);r = dict(trained=ndcg(m, pairs))
    sym = dict(M, zz=(M['zz'] + M['zz'].T)/2);r['symmetric_content'] = ndcg(m, pairs, content_from(sym, c))
    r['no_partner_history'] = ndcg(m, pairs, content_from(dict(zz=M['zz']), c))
    r['content_maps_at_start'] = ndcg(m, pairs, content_from(dict(zz=10*torch.eye(M['zz'].shape[0])), c))
    def variant(change):
        k = copy.deepcopy(m);change(k);return ndcg(k, pairs)
    def zero_last(nets):
        for n in nets:n[-1].weight.zero_();n[-1].bias.zero_()
    r['no_label_rule'] = variant(lambda k: zero_last(k.same_label)) if m.levels else None
    r['no_prior'] = variant(lambda k: zero_last([k.prior]))
    r['shared_exponent'] = variant(lambda k: k.exponents.fill_(float(m.exponents.mean())))
    r['exponents_at_start'] = variant(lambda k: k.exponents.fill_(.5))
    S = [torch.as_tensor(sorted(set(s))) for s, _ in pairs];r['fixed_gate'] = ndcg(m, pairs, gate=m.gate_weights(S, None).mean(0, keepdim=True))
    r['no_feature_refinement'] = variant(lambda k: zero_last([k.encoder]))
    return r


@torch.no_grad()
def geometry(m):
    z = m.content();M = m.query_content.T@m.key_content;g = torch.Generator().manual_seed(0);n = z.shape[0]
    a, b = torch.randint(n, (20000,), generator=g), torch.randint(n, (20000,), generator=g)
    corr = lambda u, v: float(np.corrcoef(u.numpy(), v.numpy())[0, 1])
    rnd = corr(((z[a]@M)*z[b]).sum(1), 10*(z[a]*z[b]).sum(1))
    q = torch.randint(n, (2000,), generator=g);sim = z[q]@z.T;sim[torch.arange(2000), q] = -2;nb = sim.topk(10, 1).indices
    zs = z[q][:, None, :].expand(-1, 10, -1);zc = z[nb]
    near = corr(((zs@M)*zc).sum(-1).flatten(), 10*(zs*zc).sum(-1).flatten())
    ev = torch.linalg.eigvalsh((M + M.T)/2)
    return dict(self_score=float(((z@M)*z).sum(1).mean()), corr_random_pairs=rnd, corr_near_neighbours=near,
                neg_spectrum_share=float((ev[ev < 0]**2).sum()/(ev**2).sum()), antisym_vs_sym=float((M - M.T).norm()/(M + M.T).norm()))


@torch.no_grad()
def data_facts(ds, m):
    pairs = ds.val_pairs[:VQ];c = m.cache();x = torch.nn.functional.normalize(m.features, dim=1);f = m.frequency;r = {}
    for lab, score in (('start_cosine', lambda S: torch.stack([x[s].mean(0) for s in S])@x.T),
                       ('trained_content', lambda S: m.content_scores(S, None, c))):
        hit = []
        for a in range(0, len(pairs), 128):
            P = pairs[a:a + 128];S = [torch.as_tensor(sorted(set(s))) for s, _ in P];v = score(S).numpy()
            hit += [query_metrics(v[k], s, g)['recall20'] for k, (s, g) in enumerate(P)]
        r[f'{lab}_recall20'] = float(np.mean(hit))
    r['labels'] = []
    for i in range(m.levels):
        labels = getattr(m, f'label{i}').numpy();tgt, cat = [], []
        for s, g in pairs:
            seen = set(labels[list(s)]);rest = np.setdiff1d(np.arange(m.n), list(s))
            tgt.append(np.mean([labels[t] in seen for t in g]));cat.append(np.isin(labels[rest], list(seen)).mean())
        r['labels'].append(dict(targets_sharing=float(np.mean(tgt)), catalogue_sharing=float(np.mean(cat))))
    un = [(f[list(s)] == 0).float().mean().item() for s, _ in pairs]
    r['queries_with_unseen_seed'] = float(np.mean([u > 0 for u in un]));r['queries_all_unseen'] = float(np.mean([u == 1 for u in un]))
    return r


def add_geometry(names):
    doc = json.loads((R/'readings.json').read_text());report = doc['datasets']
    for name in names:
        ds = data.load(name, CACHE)
        for s, rep in enumerate(report[name]):
            m = TRIAD(ds.features, ds.levels, ds.train);m.load_state_dict(torch.load(model_path(name, s), map_location='cpu'));m.eval()
            rep['geometry'] = geometry(m)
        print(name, 'geometry', flush=True);write_json(R/'readings.json', dict(note=__doc__, datasets=report))


def main(names):
    report = json.loads((R/'readings.json').read_text())['datasets'] if (R/'readings.json').exists() else {}
    for name in names:
        ds = data.load(name, CACHE);seeds = []
        for s in (0, 1, 2):
            p = model_path(name, s)
            if p is None:continue
            m = TRIAD(ds.features, ds.levels, ds.train);m.load_state_dict(torch.load(p, map_location='cpu'));m.eval()
            rep = seed_report(ds, m);rep['knockouts'] = knockouts(ds, m);rep['geometry'] = geometry(m)
            if s == 0:rep['data_facts'] = data_facts(ds, m)
            seeds.append(rep)
        report[name] = seeds;print(name, len(seeds), 'seeds', flush=True)
        write_json(R/'readings.json', dict(note=__doc__, datasets=report))


if __name__ == '__main__':
    if sys.argv[1:2] == ['--geometry']:add_geometry(sys.argv[2:]);sys.exit()
    main(sys.argv[1:] or ['electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify'])
