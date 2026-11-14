"""DIALS: diagonal-plus-low-rank label relations with spectral cross-fitting, for bundle completion.

Read this file next to ARCHITECTURE.md; section numbers (§) below refer to it.

Given seed items S, every catalogue item c is scored as (§5)

    s(c | S) = w_C(S) C(S,c) + sum_l w_l(S) L_l(S,c) + w_G(S) G(S,c)

Content (§6)  C = phi(S)^T psi(c). phi averages over the seeds and psi maps the candidate, each a rank-64 map of the
              item's encoded features z and its partner history h (the co-occurrence-weighted mean feature of its
              partners).
Labels (§7)   for each attribute level l (category; or artist and album), with t(S) the seeds' label counts and g(c)
              the candidate's label:
                  L_l = delta_l(t_g(c)(S), |S|, size of g(c)) [t_g(c)(S) > 0]  +  (t(S)/|S|) P_l M_l Q_l[g(c)]^T
              The diagonal delta_l (a small MLP) scores a shared label; its sign is learned. The low-rank part says
              which labels go with which: P_l, Q_l are fixed factors from a rank-16 SVD of the normalized
              label-by-label co-occurrence matrix Gamma^T N Gamma, and only the 16 x 16 matrix M_l is learned.
Graph (§8)    G = theta log(1 + sum_{s in S} n_sc deg_s^-alpha deg_c^-beta) + rho(f_c): damped co-occurrence counts
              with learned degree exponents, plus a learned curve rho of the candidate's training frequency f_c.
Gate (§9)     w(S) = softplus(MLP(log|S|, mean seed log-frequency, share of seeds never seen in training)).

Leakage-free training (§11). A training bundle is scored with its own contribution removed from every count, degree,
frequency and partner history (leave-one-out). The label SVD cannot be corrected that way, so it is cross-fitted:
training bundles are split into five folds (identical bundles share a fold) and a bundle being trained on uses
factors computed without its fold. Evaluation uses factors from all training bundles.

How the code maps to the symbols of ARCHITECTURE.md §4:

    code                        symbol      meaning
    features                    X           item features, n x d, rows L2-normalized
    source, target, counts      N           sparse co-occurrence counts, stored as sorted (s, c, N_sc) triples
    pointer                                 CSR-style index: the partners of item s are entries pointer[s]:pointer[s+1]
    frequency                   f           number of training bundles containing each item
    history_mass                deg         co-occurrence degree, deg_c = sum_s N_sc
    history_sum                 N X         unnormalized partner history; history() divides by deg to give H
    label{i}                    g_l(c)      label value of every item at level l = i
    label_left{i}, label_right{i}           P_l, Q_l from all training bundles (used at evaluation)
    label_fold_left{i}, ...                 P_l^(k), Q_l^(k) computed without fold k (used in training)
    mixing[i]                   M_l         learned 16 x 16 label mixing matrix
    diagonal[i]                 delta_l     learned MLP for "the candidate shares a seed's label"
    query_content, query_history  A_z, A_h  seed-side content maps
    key_content, key_history      B_z, B_h  candidate-side content maps
    exponents                   alpha, beta degree discounts of the graph term
    theta                       theta       scale of the graph term
    prior                       rho         learned popularity curve of log(1 + f_c)
    gate                                    MLP giving the term weights w(S)

Shapes used in comments: q = number of queries in a batch, n = catalogue size, d = feature dimension (128),
r = content rank (64), K = number of label values at a level, k = label rank (16).
"""
import hashlib
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

# Number of folds for cross-fitting the label factors (§11).
FOLDS=5


# =====================================================================================================================
# Fixed structures built from the training bundles (❄️ in ARCHITECTURE.md: computed once, never trained)
# =====================================================================================================================

def pair_counts(train,n):
    """Build the co-occurrence matrix N, item frequencies f, and an index for fast partner lookup (§4).

    Args:
        train: list of training bundles, each a list of item IDs (repeats inside a bundle are ignored).
        n: catalogue size.

    Returns:
        source, target, counts: N as triples, sorted by source then target. counts[e] = N_{source[e], target[e]}, the
            number of bundles containing both items. Only s != c pairs are stored, in both directions.
        frequency: length n; frequency[c] = f_c, the number of bundles containing c.
        pointer: length n+1; the partners of item s are target[pointer[s]:pointer[s+1]].
    """
    # Each bundle as a sorted array of distinct items.
    rows=[np.unique(np.asarray(b,np.int64)) for b in train]

    # f_c: how many bundles contain item c.
    frequency=np.bincount(np.concatenate(rows),minlength=n)

    # Every ordered pair (s, c), s != c, inside each bundle, encoded as one integer s*n + c.
    keys=[]
    for r in rows:
        if len(r)<2:
            continue
        s=np.repeat(r,len(r))    # s1 s1 s1 s2 s2 s2 ...
        c=np.tile(r,len(r))      # c1 c2 c3 c1 c2 c3 ...
        keep=s!=c                # drop the diagonal (an item with itself)
        keys.append(s[keep]*n+c[keep])

    # Counting identical keys gives N_sc. np.unique also sorts the keys, i.e. by source, then target.
    keys,counts=np.unique(np.concatenate(keys),return_counts=True)
    source=keys//n
    target=keys%n
    del keys

    # pointer[s] is the first entry whose source is s (searchsorted on the sorted sources).
    pointer=np.searchsorted(source,np.arange(n+1))
    return source,target,counts,frequency,pointer


def fold_of(bundle):
    """Deterministic fold (0..FOLDS-1) of a bundle, for cross-fitting (§11).

    The fold is a hash of the bundle's sorted distinct items, so identical bundles always share a fold and a duplicate
    cannot leak its twin's label statistics.
    """
    return int(hashlib.md5(np.asarray(sorted(set(bundle)),dtype='<i8').tobytes()).hexdigest(),16)%FOLDS


def label_factors(counts,labels,rank=16):
    """Spectral label factors P, Q (§7, the low-rank part).

    Computes the label co-occurrence  N_bar = Gamma^T N Gamma  (K x K), normalizes it by the label degrees,
        N_hat = D^-1/2 N_bar D^-1/2,
    and takes a rank-k truncated SVD  N_hat ~ U Sigma V^T. The singular values are split evenly:
        P = U Sigma^1/2,  Q = V Sigma^1/2   (each K x k),  so P Q^T ~ N_hat.

    Args:
        counts: N as a sparse n x n matrix.
        labels: length n; labels[c] = g_l(c), the label value of item c at this level.
        rank: k, the number of spectral directions (capped at K - 1).
    """
    from sklearn.utils.extmath import randomized_svd

    # Gamma: one-hot item-to-label incidence, n x K.
    g=sp.csr_matrix((np.ones(len(labels),np.float32),(np.arange(len(labels)),labels)),
                    shape=(len(labels),int(labels.max())+1))

    # N_bar = Gamma^T N Gamma: how often label g appears in a bundle with label g'.
    c=(g.T@counts@g).tocsr()

    # sqrt of the label degrees (row sums of N_bar), floored at 1 to avoid dividing by zero.
    scale=np.sqrt(np.maximum(np.asarray(c.sum(1)).ravel(),1)).astype(np.float32)

    # Truncated SVD of D^-1/2 N_bar D^-1/2. Fixed random_state: the factors are reproducible.
    u,sigma,vt=randomized_svd(sp.diags(1/scale)@c@sp.diags(1/scale),
                              min(rank,c.shape[0]-1),
                              n_iter=4,
                              random_state=0)
    root=np.sqrt(sigma)
    return (u*root).astype(np.float32),(vt.T*root).astype(np.float32)   # P, Q


def cross_fitted_labels(train,n,levels,rank=16):
    """Label factors for every level: one pair from all training bundles, and one pair per fold without that fold (§11).

    Returns, for each level l, a tuple
        ((P_l, Q_l), [(P_l^(0), Q_l^(0)), ..., (P_l^(FOLDS-1), Q_l^(FOLDS-1))])
    The first pair is used at evaluation; P_l^(k), Q_l^(k) score training bundles that fall in fold k.
    """
    def factors(rows):
        # N from the given bundles, then (P, Q) for every level.
        source,target,counts,_,_=pair_counts(rows,n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(n,n))
        return [label_factors(matrix,np.asarray(l),rank) for l in levels]

    assignment=[fold_of(b) for b in train]

    # From all training bundles (evaluation).
    full=factors(train)

    # One per fold k, built from the bundles outside fold k (training).
    parts=[factors([b for b,f in zip(train,assignment) if f!=k]) for k in range(FOLDS)]

    # Regroup from [fold][level] to [level][fold].
    return [(full[i],[p[i] for p in parts]) for i in range(len(levels))]


def curve(inputs,hidden=16):
    """A small MLP  inputs -> 16 -> 1  used for delta_l (§7) and the popularity prior rho (§8)."""
    return nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,1))


# =====================================================================================================================
# The model
# =====================================================================================================================

class DIALS(nn.Module):
    """The DIALS scorer. model(seeds) returns a q x n matrix of scores, one row per query (§5).

    Args:
        features: n x d item features X.
        levels: list of label arrays, one per level, each of length n (e.g. [category] or [artist, album]).
        train: the training bundles; N, deg, f and H are built from them here (§4).
        spectral: the output of cross_fitted_labels(train, n, levels) (cached by train.spectral).
        rank: r, the rank of the content maps (64).
        hidden: width of the content encoder MLP (32).

    Everything derived from the training split is rebuilt on construction and stored as non-persistent buffers, so a
    checkpoint holds the trained parameters only.
    """

    def __init__(self,features,levels,train,spectral,rank=64,hidden=32):
        super().__init__()

        # ---- fixed data (❄️) ---------------------------------------------------------------------------------------
        # X: features with L2-normalized rows (n x d).
        x=F.normalize(torch.as_tensor(features,dtype=torch.float32),dim=1)
        self.n,d=x.shape
        self.levels=len(levels)

        # N as sorted triples, f, and the partner index (see pair_counts); also N as a sparse matrix.
        source,target,counts,frequency,pointer=pair_counts(train,self.n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(self.n,self.n))

        # Everything derived from the training split is rebuilt on construction, so checkpoints hold parameters only.
        data=dict(
            features=x,                                                          # X
            pointer=torch.as_tensor(pointer,dtype=torch.long),                   # partner index
            target=torch.as_tensor(target,dtype=torch.long),                     # partner of each stored pair
            counts=torch.as_tensor(counts,dtype=torch.float32),                  # N_sc of each stored pair
            frequency=torch.as_tensor(frequency,dtype=torch.float32),            # f
            history_sum=torch.as_tensor(matrix@x.numpy(),dtype=torch.float32),   # N X  (n x d)
            history_mass=torch.as_tensor(np.asarray(matrix.sum(1)).ravel(),dtype=torch.float32),  # deg = N 1
        )
        for k,v in data.items():
            self.register_buffer(k,v,persistent=False)

        # log deg, floored at log 1 = 0: the degree discount of the graph term (§8).
        self.register_buffer('log_degree',self.history_mass.clamp_min(1).log(),persistent=False)

        # ---- label term, one set per level l (§7) --------------------------------------------------------------
        self.mixing=nn.ParameterList()   # M_l, trained
        self.diagonal=nn.ModuleList()    # delta_l, trained
        for i,(labels,((left,right),folds)) in enumerate(zip(levels,spectral)):
            g=torch.as_tensor(labels,dtype=torch.long)
            size=int(g.max())+1    # K_l, the number of label values at this level

            # g_l(c) for every item.
            self.register_buffer(f'label{i}',g,persistent=False)

            # log(1 + n_g): how many items carry each label value (an input of delta_l).
            self.register_buffer(f'label_size{i}',torch.bincount(g,minlength=size).float().log1p(),persistent=False)

            # P_l, Q_l from all training bundles (evaluation).
            self.register_buffer(f'label_left{i}',torch.as_tensor(left),persistent=False)
            self.register_buffer(f'label_right{i}',torch.as_tensor(right),persistent=False)

            # P_l^(k), Q_l^(k) stacked over the folds: FOLDS x K x k (training).
            self.register_buffer(f'label_fold_left{i}',torch.as_tensor(np.stack([f[0] for f in folds])),persistent=False)
            self.register_buffer(f'label_fold_right{i}',torch.as_tensor(np.stack([f[1] for f in folds])),persistent=False)

            # M_l starts at zero, so the low-rank part starts switched off.
            self.mixing.append(nn.Parameter(torch.zeros(left.shape[1],left.shape[1])))  # M_l
            # delta_l takes (log(1+m), log|S|, log(1+n_g)).
            self.diagonal.append(curve(3))

        # ---- content term (§6) -----------------------------------------------------------------------------------
        # Residual encoder Z = normalize(X + MLP(X)); the last layer starts at zero, so Z = X at initialization.
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

        # A_z = B_z = sqrt(10) [I_r 0] and A_h = B_h = 0: the content score starts as 10 x the dot product of the
        # first r feature coordinates (seed mean against candidate).
        start=math.sqrt(10.)*torch.eye(rank,d)
        self.query_content=nn.Parameter(start.clone())         # A_z, r x d
        self.key_content=nn.Parameter(start.clone())           # B_z, r x d
        self.query_history=nn.Parameter(torch.zeros(rank,d))   # A_h, r x d
        self.key_history=nn.Parameter(torch.zeros(rank,d))     # B_h, r x d

        # ---- graph term (§8) -------------------------------------------------------------------------------------
        self.exponents=nn.Parameter(torch.zeros(2))  # alpha, beta; 0 means raw counts
        self.theta=nn.Parameter(torch.tensor(1.))    # theta
        self.prior=curve(1)                          # rho, a curve of log(1 + f_c)

        # ---- gate (§9) -------------------------------------------------------------------------------------------
        # 3 query statistics -> one weight per term: content, one per label level, graph.
        self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,2+self.levels))  # content, labels..., graph
        # Zero weights and bias log(e - 1) make softplus(output) = 1: every term starts with weight 1.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias,math.log(math.e-1))  # softplus = 1

    def fast_parameters(self):
        """Small parameters trained at the higher learning rate (§10): M_l, delta_l, alpha, beta, theta, rho, gate."""
        return [*self.mixing,
                *self.diagonal.parameters(),
                self.exponents,
                self.theta,
                *self.prior.parameters(),
                *self.gate.parameters()]

    # =================================================================================================================
    # Term 1: content (§6)
    # =================================================================================================================
    def content(self):
        """Z = normalize(X + MLP(X)), the refined item features (n x d)."""
        return F.normalize(self.features+self.encoder(self.features),dim=1)

    def history(self):
        """H = D^-1 N X, the partner history of every item (n x d); a row is zero when the item has no partners."""
        return self.history_sum/self.history_mass.clamp_min(1)[:,None]*(self.history_mass>0)[:,None]

    def loo_history(self,items,bundle):
        """Leave-one-out partner history (§11): rows of H for `items`, with the bundle's own co-occurrences removed.

        Each item in `bundle` was counted as a partner of every other member, once per bundle. Removing the bundle
        takes |b| - 1 off the item's degree and the features of the other members off its N X row.
        `items` are members of `bundle`.
        """
        mass=self.history_mass[items]-(len(bundle)-1)
        value=(self.history_sum[items]-(self.features[bundle].sum(0)-self.features[items]))/mass.clamp_min(1)[:,None]
        return value*(mass>0)[:,None]

    def keys(self,z=None):
        """key(c) = B_z z_c + B_h h_c for every item (n x r)."""
        z=self.content() if z is None else z
        return z@self.key_content.T+self.history()@self.key_history.T

    def cache(self):
        """The query-independent parts of evaluation scores, computed once per evaluation instead of once per batch.

        Returns Z, H, the key matrix and the popularity prior rho(log(1 + f_c)) for every item.
        """
        z=self.content()
        h=self.history()
        return dict(content=z,
                    history=h,
                    keys=z@self.key_content.T+h@self.key_history.T,
                    prior=self.prior(self.frequency.log1p()[:,None]).squeeze(1))

    def content_scores(self,seeds,bundles,cache=None):
        """C(S, c) = phi(S)^T key(c) for every query and every item (q x n).

        Args:
            seeds: list of q tensors of seed item IDs.
            bundles: None at evaluation; in training, the q complete bundles, for leave-one-out histories.
            cache: model.cache(), at evaluation only.
        """
        if cache is None:
            z=self.content()
            k=self.keys(z)
            h=self.history()
        else:
            z,k,h=cache['content'],cache['keys'],cache['history']

        phi=[]     # phi(S) per query
        fixes=[]   # training only: corrections to the keys of the bundle's own members
        for i,s in enumerate(seeds):
            hs=h[s]   # partner histories of the seeds
            if bundles is not None:
                # Training: the seeds' histories must not contain this bundle (leave-one-out).
                b=bundles[i]
                hs=self.loo_history(s,b)
                # The bundle's members are also candidates; their keys use leave-one-out histories too.
                # Store the change of their keys: (h_loo - h) B_h^T.
                fixes.append((i,b,(self.loo_history(b,b)-h[b])@self.key_history.T))
            # query(s) = A_z z_s + A_h h_s, averaged over the seeds.
            phi.append((z[s]@self.query_content.T+hs@self.query_history.T).mean(0))

        # One matrix product scores every item: (q x r) @ (r x n).
        phi=torch.stack(phi)
        scores=phi@k.T

        if fixes:  # bundle members as candidates: their keys use leave-one-out histories too
            # Add phi(S)^T (change of key) at the (query, member) entries.
            rows=torch.cat([torch.full((len(b),),i) for i,b,_ in fixes])
            cols=torch.cat([b for _,b,_ in fixes])
            scores=scores.index_put((rows,cols),
                                    torch.cat([delta@phi[i] for i,_,delta in fixes]),
                                    accumulate=True)
        return scores

    # =================================================================================================================
    # Term 2: labels, diagonal plus low rank (§7)
    # =================================================================================================================
    def label_scores(self,i,seeds,bundles):
        """L_l(S, c) at level l = i for every query and every item (q x n).

        L_l = (1/|S|) t(S)^T P_l M_l Q_l[g(c)]^T                     low rank: which labels go with which
            + delta_l(log(1+m), log|S|, log(1+n_g(c))) [m > 0]       diagonal: c shares a seed's label
        with t(S) the seeds' label counts and m = t(S)_g(c).

        In training (bundles given) each query uses the fold-k factors P_l^(k), Q_l^(k) of its bundle's fold (§11).
        """
        labels=getattr(self,f'label{i}')   # g_l(c) for every item
        q=len(seeds)
        size=int(labels.max())+1           # K_l
        out=torch.zeros(q,self.n)

        # t(S): how many seeds carry each label value (q x K).
        tally=torch.zeros(q,size)
        for j,s in enumerate(seeds):
            tally[j]=torch.bincount(labels[s],minlength=size).float()

        # t(S)/|S|: the seeds' label mix.
        mean=tally/tally.sum(1,keepdim=True)
        mix=self.mixing[i]   # M_l

        # ---- low-rank part: (t(S)/|S|) P M Q[g(c)]^T --------------------------------------------------------------
        # (q x K)(K x k)(k x k) gives a q x k query vector; Q[labels] is n x k, one row per item.
        if bundles is None:
            # Evaluation: factors from all training bundles.
            out=out+(mean@getattr(self,f'label_left{i}')@mix)@getattr(self,f'label_right{i}')[labels].T
        else:
            # Training: each bundle is scored with the factors computed without its fold.
            fold=torch.tensor([fold_of(b.tolist()) for b in bundles])
            for k in fold.unique().tolist():
                rows=(fold==k).nonzero(as_tuple=True)[0]
                part=(mean[rows]@getattr(self,f'label_fold_left{i}')[k]@mix)@getattr(self,f'label_fold_right{i}')[k][labels].T
                out=out.index_put((rows,),out[rows]+part)

        # ---- diagonal part: delta_l, only where m > 0 -------------------------------------------------------------
        # m[j, c] = t(S_j)_g(c): how many seeds share item c's label (q x n).
        m=tally[:,labels]
        row,col=m.nonzero(as_tuple=True)
        if len(row):
            # log|S| per query.
            count=torch.tensor([float(len(s)) for s in seeds]).log()
            # The three inputs of delta_l for every (query, candidate) with m > 0.
            x=torch.stack((m[row,col].log1p(),                            # log(1 + m)
                           count[row],                                    # log|S|
                           getattr(self,f'label_size{i}')[labels[col]]),  # log(1 + n_g(c))
                          1)
            out=out.index_put((row,col),self.diagonal[i](x).squeeze(1),accumulate=True)
        return out

    # =================================================================================================================
    # Term 3: co-occurrence graph (§8)
    # =================================================================================================================
    def graph_scores(self,seeds,bundles,cache=None):
        """G(S, c) = theta log(1 + sum_s N_sc deg_s^-alpha deg_c^-beta) + rho(log(1 + f_c)) for every query and item.

        Only the observed partners of the seeds are touched, so the cost is sum_s deg_s, not n. In training, the
        bundle's own pairs are removed from the counts, degrees and frequencies (leave-one-out, §11).
        """
        alpha,beta=self.exponents
        q=len(seeds)

        # Every stored pair (s, c) whose source s is a seed, tagged with its query index.
        query,source,edge=self.expand(torch.cat([torch.full((len(s),),i) for i,s in enumerate(seeds)]),
                                      torch.cat(seeds))
        target=self.target[edge]   # c
        n=self.counts[edge]        # N_sc
        ds=self.log_degree[source] # log deg_s
        dt=self.log_degree[target] # log deg_c

        if bundles is not None:  # leave-one-out: remove the bundle's own pairs from counts and degrees
            # member[j, c] is True when item c is in query j's bundle.
            member=torch.zeros(q,self.n,dtype=torch.bool)
            for i,b in enumerate(bundles):
                member[i,b]=True
            # A pair inside the bundle was counted once by this bundle: N_sc - 1.
            inside=member[query,target].float()
            n=n-inside
            # A member's degree includes |b| - 1 partners from this bundle.
            others=torch.tensor([float(len(b)-1) for b in bundles])[query]
            ds=(self.history_mass[source]-others).clamp_min(1).log()
            dt=(self.history_mass[target]-inside*others).clamp_min(1).log()

        # N_sc deg_s^-alpha deg_c^-beta, summed over the seeds into a q x n matrix.
        w=n*torch.exp(-alpha*ds-beta*dt)
        first=torch.zeros(q*self.n).scatter_add(0,query*self.n+target,w).view(q,self.n)
        out=self.theta*first.log1p()

        # Popularity prior rho(log(1 + f_c)), the same for every query.
        prior=(self.prior(self.frequency.log1p()[:,None]).squeeze(1) if cache is None else cache['prior']).expand(q,-1)
        if bundles is not None:
            # Leave-one-out: for the bundle's own members, use f_c - 1.
            rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)])
            cols=torch.cat(bundles)
            left=(self.frequency[cols]-1).clamp_min(0)
            prior=prior.index_put((rows,cols),
                                  self.prior(left.log1p()[:,None]).squeeze(1)-prior[rows,cols],
                                  accumulate=True)
        return out+prior

    def expand(self,query,rows):
        """Every observed pair (row, partner) for the given rows, tagged with its query.

        For each row s, its partners are the contiguous entries pointer[s]:pointer[s+1] of (target, counts). This
        returns, for every such entry, (query index, s, entry index), without a Python loop.
        """
        start=self.pointer[rows]                     # first entry of each row
        length=self.pointer[rows+1]-start            # number of partners of each row
        offsets=torch.cumsum(length,0)-length        # where each row's block starts in the output
        # Entry index = start of the row + position within the row's block.
        edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
        return torch.repeat_interleave(query,length),torch.repeat_interleave(rows,length),edge

    # =================================================================================================================
    # Gate (§9) and total score (§5)
    # =================================================================================================================
    def gate_weights(self,seeds,bundles):
        """w(S) = softplus(MLP(u(S))), one weight per term (q x (2 + levels)).

        u(S) = (log|S|, mean over seeds of log(1 + f_s), share of seeds with f_s = 0). In training, f_s - 1 is used
        (leave-one-out: the bundle itself is not counted).
        """
        stats=[]
        for s in seeds:
            f=(self.frequency[s]-(1 if bundles is not None else 0)).clamp_min(0)
            stats.append(torch.stack((torch.tensor(float(len(s))).log(),   # log|S|
                                      f.log1p().mean(),                    # mean log(1 + f_s)
                                      (f==0).float().mean())))             # share of unseen seeds
        return F.softplus(self.gate(torch.stack(stats)))

    def forward(self,seeds,bundles=None,cache=None):
        """Scores (queries x catalogue). bundles: the complete training bundles, for leave-one-out training scores.
        cache: model.cache(), for evaluation only.

        s(c | S) = w_C C + sum_l w_l L_l + w_G G   (§5)
        """
        if cache is not None and bundles is not None:
            raise ValueError('The cache is for evaluation only')

        # Distinct, sorted item IDs as tensors.
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:
            bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]

        # The terms, in the gate's order: content, one per label level, graph. Each is q x n.
        parts=([self.content_scores(seeds,bundles,cache)]
               +[self.label_scores(i,seeds,bundles) for i in range(self.levels)]
               +[self.graph_scores(seeds,bundles,cache)])

        # Weighted sum, with one weight per query and term.
        w=self.gate_weights(seeds,bundles)
        return sum(w[:,j,None]*v for j,v in enumerate(parts))

    def parameter_count(self):
        """Number of trained parameters (41,624 with one label level and 41,978 with two, at the default settings)."""
        return sum(p.numel() for p in self.parameters())
