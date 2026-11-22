"""Follow-up readings of the trained TRIAD (same models and validation queries as analysis.readings, never test).
Writes results/readings_extra.json.

1. asymmetry  What does the antisymmetric part K = (M - M')/2 of the content matrix M = A_z'B_z do inside a query?
   For seeds S the asymmetric score of candidate c is  zbar' K z_c = mu' K z_c + (zbar - mu)' K z_c,  with mu the mean
   refined feature over the catalogue:
     - mu' K z_c is the same for every query: a candidate-only preference ("this kind of item tends to be the one that
       is added"), carried by the asymmetric part;
     - (zbar - mu)' K z_c depends on the seeds: a directional relation between these seeds and this candidate.
   Reported: the variance of each over (validation query, candidate in the query's top 100) as a share of their sum, and
   validation NDCG@20 with K removed, with only the candidate-only part kept, and with only the seed-dependent part kept
   (the symmetric part and the partner-history blocks are kept in all three).
   Per label (category): mean of mu' K z_c over the label's items, against log(share of validation targets with the label
   / share of validation seeds with the label), i.e. how much more often the label is the missing item than a given one.

2. interactions  Knock-outs of two parts at once, against each alone (validation NDCG@20, first 1,000 queries):
     label_x_content  label rule removed, M_zz back to 10 I (partner-history blocks kept), both
     prior_x_exponents  prior removed, exponents back to 1/2, both
     gate_x_history  gate fixed at its mean, partner-history blocks removed, both

python -m analysis.extra [names]          (from the repository root)
"""
import triad  # pins BLAS threads before numpy loads
import copy, json, sys
import numpy as np, torch
from triad import data
from triad.model import TRIAD
from triad.train import write_json
from analysis.readings import CACHE, R, VQ, model_path, blocks, content_from, ndcg, scaled

OUT = R/'readings_extra.json'


@torch.no_grad()
def asymmetry(ds, m):
    pairs = ds.val_pairs[:VQ];c = m.cache();z = c['content'];M = blocks(m, c)
    K = (M['zz'] - M['zz'].T)/2;Sym = (M['zz'] + M['zz'].T)/2;mu = z.mean(0)
    bias = (mu@K)@z.T  # mu' K z_c for every item
    rest = {k: v for k, v in M.items() if k != 'zz'}
    base = content_from(dict(rest, zz=Sym), c)
    def with_part(part):
        def score(S):
            zb = torch.stack([z[s].mean(0) for s in S]);out = base(S)
            if part in ('bias', 'both'):out = out + bias[None, :]
            if part in ('seed', 'both'):out = out + ((zb - mu)@K)@z.T
            return out
        return score
    r = dict(trained=ndcg(m, pairs), check_both=ndcg(m, pairs, with_part('both')), no_asymmetry=ndcg(m, pairs, with_part('none')),
             candidate_only=ndcg(m, pairs, with_part('bias')), seed_dependent_only=ndcg(m, pairs, with_part('seed')))
    vb, vs = [], []
    for a in range(0, min(512, len(pairs)), 128):
        S = [torch.as_tensor(sorted(set(s))) for s, _ in pairs[a:a + 128]];tot = m(S)
        for i, s in enumerate(S):tot[i, s] = -1e9
        top = tot.topk(100, 1).indices;zb = torch.stack([z[s].mean(0) for s in S])
        vb.append(bias[top].var(1));vs.append((((zb - mu)@K)@z.T).gather(1, top).var(1))
    vb, vs = torch.cat(vb).mean().item(), torch.cat(vs).mean().item()
    r['variance_share'] = dict(candidate_only=vb/(vb + vs), seed_dependent=vs/(vb + vs))
    lab = m.label0.numpy() if m.levels else None;r['labels'] = None
    if lab is not None:
        seeds = np.concatenate([list(set(s)) for s, _ in ds.val_pairs]);tg = np.concatenate([list(set(g)) for _, g in ds.val_pairs])
        L = lab.max() + 1;cs, ct = np.bincount(lab[seeds], minlength=L), np.bincount(lab[tg], minlength=L)
        keep = np.where((cs >= 30) & (ct >= 30))[0];b = bias.numpy()
        mb = np.array([b[lab == l].mean() for l in keep]);ratio = np.log((ct[keep]/ct.sum())/(cs[keep]/cs.sum()))
        r['labels'] = dict(n=len(keep), corr=float(np.corrcoef(mb, ratio)[0, 1]) if len(keep) >= 3 else None,
                           rows=[dict(label=int(l), bias=float(x), log_target_over_seed=float(y), seeds=int(cs[l]), targets=int(ct[l]))
                                 for l, x, y in zip(keep, mb, ratio)])
    return r


@torch.no_grad()
def interactions(ds, m, which):
    pairs = ds.val_pairs[:VQ];c = m.cache();M = blocks(m, c);r = {}
    def zero_last(nets):
        for n in nets:n[-1].weight.zero_();n[-1].bias.zero_()
    S = [torch.as_tensor(sorted(set(s))) for s, _ in pairs];gmean = m.gate_weights(S, None).mean(0, keepdim=True)
    start = content_from(dict(M, zz=10*torch.eye(M['zz'].shape[0])), c);nohist = content_from(dict(zz=M['zz']), c)
    nolab = copy.deepcopy(m)
    if m.levels:zero_last(nolab.same_label)
    if 'label_x_content' in which and m.levels:
        r['label_x_content'] = dict(label=ndcg(nolab, pairs), content=ndcg(m, pairs, start), both=ndcg(nolab, pairs, start))
    if 'prior_x_exponents' in which:
        a = copy.deepcopy(m);zero_last([a.prior]);b = copy.deepcopy(m);b.exponents.fill_(.5);ab = copy.deepcopy(a);ab.exponents.fill_(.5)
        r['prior_x_exponents'] = dict(prior=ndcg(a, pairs), exponents=ndcg(b, pairs), both=ndcg(ab, pairs))
    if 'gate_x_history' in which:
        r['gate_x_history'] = dict(gate=ndcg(m, pairs, gate=gmean), history=ndcg(m, pairs, nohist), both=ndcg(m, pairs, nohist, gate=gmean))
    r['trained'] = ndcg(m, pairs)
    return r


WHICH = ['label_x_content', 'prior_x_exponents', 'gate_x_history']


def main(names):
    report = json.loads(OUT.read_text())['datasets'] if OUT.exists() else {}
    for name in names:
        ds = data.load(name, CACHE);seeds = []
        for s in (0, 1, 2):
            m = TRIAD(ds.features, ds.levels, ds.train);m.load_state_dict(torch.load(model_path(name, s), map_location='cpu'));m.eval()
            seeds.append(dict(asymmetry=asymmetry(ds, m), interactions=interactions(ds, m, WHICH)))
            print(name, 'seed', s, flush=True)
        report[name] = seeds;write_json(OUT, dict(note=__doc__, datasets=report))


if __name__ == '__main__':
    main(sys.argv[1:] or ['electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify'])
