"""Every number quoted in FORWARD_PASS.md, from the frozen Electronic seed-0 model on test query 20 (seeds 1912, 2708).

PYTHONPATH=. python paper/figdata/forward_numbers.py
"""
import triad  # pins BLAS threads before numpy loads
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from triad import data
from triad import train as T

ROOT = Path(__file__).resolve().parents[2];R = lambda x, k=3: round(float(x), k)


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ds = data.load('electronic', ROOT/'cache');m = T.load(ds, ROOT/'runs'/'electronic'/'seed0', ROOT/'cache')
    S = torch.tensor([1912, 2708]);seeds = [S];FOLLOW = [943, 717, 2775, 2982];out = {}
    a, b = [float(v) for v in m.exponents];out['alpha_beta_theta'] = [R(a, 4), R(b, 4), R(m.theta, 4)];out['parameters'] = m.parameter_count()
    out['pointer'] = {int(s): [int(m.pointer[s]), int(m.pointer[s + 1])] for s in S}
    out['partners'] = {int(s): [(int(t), int(c)) for t, c in zip(m.target[m.pointer[s]:m.pointer[s + 1]], m.counts[m.pointer[s]:m.pointer[s + 1]])] for s in S}
    out['stored_pairs'] = len(m.target)
    # content
    X = m.features;Z = m.content();H = m.history();mlp = m.encoder(X)
    out['mlp_norm'] = [R(mlp[s].norm(), 2) for s in S];out['cos_x_z'] = [R(F.cosine_similarity(X[s], Z[s], 0), 2) for s in S]
    out['h_norm'] = [R(H[s].norm(), 2) for s in S]
    zq = Z[S]@m.query_content.T;hq = H[S]@m.query_history.T;phi = (zq + hq).mean(0)
    out['zq_len'] = [R(v.norm(), 2) for v in zq];out['hq_len'] = [R(v.norm(), 2) for v in hq];out['phi_len'] = R(phi.norm(), 2)
    out['query_len'] = [R(v.norm(), 2) for v in zq + hq]
    C = m.content_scores(seeds, None, m.cache())[0];own = Z@m.key_content.T@phi;part = H@m.key_history.T@phi
    out['content_table'] = {c: [R(own[c]), R(part[c]), R(C[c])] for c in FOLLOW}
    out['C_mean_std'] = [R(C.mean(), 2), R(C.std(), 2)];top = int(C.argmax());out['C_max'] = [top, R(C[top], 2)]
    non = torch.ones(m.n, dtype=torch.bool);non[S] = False
    rank_of = lambda v, c: 1 + int((v[non] > v[c]).sum())
    out['content_rank_target'] = rank_of(C, 943)
    # labels
    L = m.label_scores(0, seeds)[0];lab = m.label0
    out['seed_labels'] = [int(lab[s]) for s in S];out['n_labels'] = int(lab.max()) + 1
    shares = (lab[None, :] == lab[S][:, None]).any(0);out['items_sharing'] = int(shares.sum())
    out['label_table'] = {c: [int(lab[c]), int((lab == lab[c]).sum()), R(L[c], 4)] for c in FOLLOW + [1697, 3317]}
    out['label_rank_target'] = rank_of(L, 943);out['label_items_positive'] = int((L[non] > 0).sum())
    delta = {}
    for mm in (1, 2):
        delta[mm] = [R(m.same_label[0](torch.tensor([[math.log1p(mm), math.log(2), math.log1p(size)]])), 2) for size in (1, 5, 20, 50)]
    out['delta_table'] = delta
    # graph
    G = m.graph_scores(seeds, None, m.cache())[0];prior = m.prior(m.frequency.log1p()[:, None]).squeeze(1);deg = m.history_mass;f = m.frequency
    def wsum(c):
        tot = 0.
        for s in S:
            lo, hi = int(m.pointer[s]), int(m.pointer[s + 1]);t = m.target[lo:hi];k = (t == c).nonzero()
            if len(k):tot += float(m.counts[lo:hi][k[0]])*math.exp(-a*math.log(max(float(deg[s]), 1)) - b*math.log(max(float(deg[c]), 1)))
        return tot
    out['graph_table'] = {c: [int(deg[c]), int(f[c]), R(wsum(c)), R(m.theta*math.log1p(wsum(c))), R(prior[c]), R(G[c])] for c in FOLLOW}
    out['graph_943_terms'] = {int(s): R(math.exp(-a*math.log(float(deg[s])) - b*math.log(52.)), 3) for s in S}
    out['rho_table'] = {k: R(m.prior(torch.tensor([[math.log1p(k)]])), 2) for k in (0, 1, 2, 5, 10, 20)}
    out['keep_deg52_deg11'] = [R(math.exp(-b*math.log(52)), 2), R(math.exp(-b*math.log(11)), 2)]
    out['graph_rank_target'] = rank_of(G, 943);g = sorted(((float(G[c]), c) for c in range(m.n) if non[c]), reverse=True)
    out['graph_top2'] = [(c, R(v)) for v, c in g[:2]];out['graph_nonzero'] = int(((G - prior) != 0).sum())
    # gate and total
    w = m.gate_weights(seeds, None)[0];out['weights'] = [R(v) for v in w]
    total = m(seeds)[0];order = [int(i) for i in torch.argsort(-total) if non[int(i)]]
    out['total_table'] = {c: [R(w[0]*C[c]), R(w[1]*L[c]), R(w[2]*G[c]), R(total[c]), 1 + order.index(c)] for c in FOLLOW}
    out['rank5'] = [order[4], R(total[order[4]])];out['top20'] = order[:20]
    # training mode on bundle [2982, 2775, 717, 943, 1912], seeds [2982, 2775]
    bundle = torch.tensor(sorted([2982, 2775, 717, 943, 1912]));ts = [torch.tensor([2775, 2982])]
    Ce = m.content_scores(ts, None, m.cache())[0];Ct = m.content_scores(ts, [bundle])[0]
    out['train_content'] = {int(c): [R(Ce[c]), R(Ct[c])] for c in bundle};out['train_content_maxchange'] = R((Ct - Ce)[bundle].abs().max())
    Ge = m.graph_scores(ts, None, m.cache())[0];Gt = m.graph_scores(ts, [bundle])[0]
    out['train_graph'] = {c: [R(Ge[c]), R(Gt[c])] for c in (717, 943, 1912)}
    Le = m.label_scores(0, ts)[0];out['train_labels_equal'] = True
    out['train_gate'] = [[R(v, 2) for v in m.gate_weights(ts, None)[0]], [R(v, 2) for v in m.gate_weights(ts, [bundle])[0]]]
    (Path(__file__).parent/'forward_numbers.json').write_text(json.dumps(out, indent=1));print(json.dumps(out))


if __name__ == '__main__':
    main()
