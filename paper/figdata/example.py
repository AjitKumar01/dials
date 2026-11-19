"""The running example of ARCHITECTURE.md §5-9, FORWARD_PASS.md and figures/forward.tex, from the frozen model.

Electronic test query 20, frozen seed-0 model (runs/electronic/seed0). Prints and writes paper/figdata/example.json:
the seeds and the target, every intermediate number of the three terms and the gate for the target, and the weighted
parts of the score for seven candidates (the target, the next four by score, and the two best-ranked items never in
a training bundle). Checks that the parts add up to the model's own scores.

PYTHONPATH=. python paper/figdata/example.py
"""
import triad  # pins BLAS threads before numpy loads
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from triad import data
from triad import train as T

ROOT = Path(__file__).resolve().parents[2];QUERY = 20


def titles_and_categories(n):
    root = data.DATA_ROOT/'BundleRec'/'dataset'/'electronic'
    items = sorted({int(i) for _, i in data.read_csv(root/'bundle_item.csv')});index = {it: k for k, it in enumerate(items)}
    titles, cats = ['']*n, ['']*n
    for row in data.read_csv(root/'item_titles.csv'):
        if int(row[0]) in index:titles[index[int(row[0])]] = ' '.join(row[1:])
    for row in data.read_csv(root/'item_categories.csv'):
        if int(row[0]) in index:cats[index[int(row[0])]] = row[1]
    return titles, cats


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    ds = data.load('electronic', ROOT/'cache');model = T.load(ds, ROOT/'runs'/'electronic'/'seed0', ROOT/'cache')
    seeds, truth = ds.test_pairs[QUERY];target = truth[0];titles, cats = titles_and_categories(ds.n_items)
    s = [torch.as_tensor(sorted(set(seeds)), dtype=torch.long)]
    C = model.content_scores(s, None, model.cache())[0];L = model.label_scores(0, s)[0];G = model.graph_scores(s, None, model.cache())[0]
    w = model.gate_weights(s, None)[0];total = model(s)[0]
    assert float((w[0]*C + w[1]*L + w[2]*G - total).abs().max()) < 1e-4
    # graph pieces
    alpha, beta = [float(v) for v in model.exponents];theta = float(model.theta);deg = model.history_mass;f = model.frequency
    prior = model.prior(f.log1p()[:, None]).squeeze(1)
    cooc = G - prior
    # ranks among non-seeds
    order = [int(i) for i in torch.argsort(-total) if int(i) not in set(seeds)];rank = {c: k + 1 for k, c in enumerate(order)}
    others = [c for c in order if c != target][:4];unseen = [c for c in order if f[c] == 0][:2]
    # content pieces
    Z = model.content();H = model.history()
    q = Z[s[0]]@model.query_content.T + H[s[0]]@model.query_history.T;phi = q.mean(0)
    key = Z[target]@model.key_content.T + H[target]@model.key_history.T
    labels = model.label0;m = int((labels[s[0]] == labels[target]).sum())
    def pair(a, b):
        lo, hi = int(model.pointer[a]), int(model.pointer[a + 1]);t = model.target[lo:hi];k = (t == b).nonzero()
        return int(model.counts[lo:hi][k[0]]) if len(k) else 0
    u = [math.log(len(seeds)), float(np.mean([math.log1p(float(f[x])) for x in seeds])), float(np.mean([float(f[x]) == 0 for x in seeds]))]
    cand = lambda c: dict(id=c, title=titles[c][:80], category=cats[c], f=int(f[c]), rank=rank[c], content=float(w[0]*C[c]), labels=float(w[1]*L[c]),
                          cooccurrence=float(w[2]*cooc[c]), prior=float(w[2]*prior[c]), total=float(total[c]), raw=dict(C=float(C[c]), L=float(L[c]), G=float(G[c])))
    out = dict(query=QUERY, seeds=[dict(id=x, title=titles[x][:80], category=cats[x], f=int(f[x]), deg=float(deg[x])) for x in seeds],
               target=dict(id=target, title=titles[target][:80], category=cats[target], f=int(f[target]), deg=float(deg[target]), n_label=int((labels == labels[target]).sum())),
               held_out=truth, gate_inputs=u, weights=[float(x) for x in w], alpha=alpha, beta=beta, theta=theta,
               content=dict(query_lengths=[float(x) for x in q.norm(dim=1)], phi_length=float(phi.norm()), key_length=float(key.norm()), C=float(C[target]),
                            content_only_rank=1 + int((C[[c for c in range(ds.n_items) if c not in set(seeds)]] > C[target]).sum())),
               labels=dict(m=m, L=float(L[target])),
               graph=dict(counts=[pair(x, target) for x in seeds], sum=float(sum(pair(x, target)*float(deg[x])**-alpha for x in seeds)*float(deg[target])**-beta),
                          cooc=float(cooc[target]), prior=float(prior[target]), G=float(G[target]),
                          prior_by_f={str(k): float(model.prior(torch.tensor([[math.log1p(k)]])).item()) for k in (0, 1, 2, 4, 8, 11)}),
               score=float(total[target]), rank=rank[target], candidates=[cand(c) for c in [target] + others + unseen], non_seed_items=len(order))
    (Path(__file__).parent/'example.json').write_text(json.dumps(out, indent=1));print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
