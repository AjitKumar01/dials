"""DIALS: diagonal-plus-low-rank label relations with spectral cross-fitting, for bundle completion.

Given seed items S, every catalogue item c is scored as

    s(c | S) = w_C(S) C(S,c) + sum_l w_l(S) L_l(S,c) + w_G(S) G(S,c)

Content   C = phi(S)^T psi(c). phi averages over the seeds and psi maps the candidate, each a rank-64 map of the item's
          encoded features z and its partner history h (the co-occurrence-weighted mean feature of its partners).
Labels    for each attribute level l (category; or artist and album), with t(S) the seeds' label counts and g(c) the
          candidate's label:
          L_l = delta_l(t_g(c)(S), |S|, size of g(c)) [t_g(c)(S) > 0]  +  (t(S)/|S|) P_l M_l Q_l[g(c)]^T
          The diagonal delta_l (a small MLP) scores a shared label; its sign is learned. The low-rank part says
          which labels go with which: P_l, Q_l are fixed factors from a rank-16 SVD of the normalized label-by-label
          co-occurrence matrix Gamma^T N Gamma, and only the 16 x 16 matrix M_l is learned.
Graph     G = theta log(1 + sum_{s in S} n_sc deg_s^-alpha deg_c^-beta) + rho(f_c): damped co-occurrence counts with
          learned degree exponents, plus a learned curve rho of the candidate's training frequency f_c.
Gate      w(S) = softplus(MLP(log|S|, mean seed log-frequency, share of seeds never seen in training)).

Leakage-free training. A training bundle is scored with its own contribution removed from every count, degree,
frequency and partner history (leave-one-out). The label SVD cannot be corrected that way, so it is cross-fitted:
training bundles are split into five folds (identical bundles share a fold) and a bundle being trained on uses
factors computed without its fold. Evaluation uses factors from all training bundles.
"""
import hashlib
import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

FOLDS=5


def pair_counts(train,n):
    """Directed co-occurrence counts n_sc sorted by source, item frequencies, and the source index pointer."""
    rows=[np.unique(np.asarray(b,np.int64)) for b in train]
    frequency=np.bincount(np.concatenate(rows),minlength=n)
    keys=[]
    for r in rows:
        if len(r)<2:continue
        s=np.repeat(r,len(r));c=np.tile(r,len(r));keep=s!=c
        keys.append(s[keep]*n+c[keep])
    keys,counts=np.unique(np.concatenate(keys),return_counts=True)
    source=keys//n;target=keys%n;del keys
    pointer=np.searchsorted(source,np.arange(n+1))
    return source,target,counts,frequency,pointer


def fold_of(bundle):
    """Deterministic fold of a bundle; identical bundles always share a fold."""
    return int(hashlib.md5(np.asarray(sorted(set(bundle)),dtype='<i8').tobytes()).hexdigest(),16)%FOLDS


def label_factors(counts,labels,rank=16):
    """P, Q from a truncated SVD of D^-1/2 (Gamma^T N Gamma) D^-1/2, the singular values split evenly between them."""
    from sklearn.utils.extmath import randomized_svd
    g=sp.csr_matrix((np.ones(len(labels),np.float32),(np.arange(len(labels)),labels)),shape=(len(labels),int(labels.max())+1))
    c=(g.T@counts@g).tocsr();scale=np.sqrt(np.maximum(np.asarray(c.sum(1)).ravel(),1)).astype(np.float32)
    u,sigma,vt=randomized_svd(sp.diags(1/scale)@c@sp.diags(1/scale),min(rank,c.shape[0]-1),n_iter=4,random_state=0)
    root=np.sqrt(sigma)
    return (u*root).astype(np.float32),(vt.T*root).astype(np.float32)


def cross_fitted_labels(train,n,levels,rank=16):
    """Per level: factors from all training bundles, and one pair per fold computed without that fold."""
    def factors(rows):
        source,target,counts,_,_=pair_counts(rows,n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(n,n))
        return [label_factors(matrix,np.asarray(l),rank) for l in levels]
    assignment=[fold_of(b) for b in train]
    full=factors(train);parts=[factors([b for b,f in zip(train,assignment) if f!=k]) for k in range(FOLDS)]
    return [(full[i],[p[i] for p in parts]) for i in range(len(levels))]


def curve(inputs,hidden=16):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,1))


class DIALS(nn.Module):
    def __init__(self,features,levels,train,spectral,rank=64,hidden=32):
        super().__init__()
        x=F.normalize(torch.as_tensor(features,dtype=torch.float32),dim=1)
        self.n,d=x.shape;self.levels=len(levels)
        source,target,counts,frequency,pointer=pair_counts(train,self.n)
        matrix=sp.csr_matrix((counts.astype(np.float32),(source,target)),shape=(self.n,self.n))
        # Everything derived from the training split is rebuilt on construction, so checkpoints hold parameters only.
        data=dict(features=x,pointer=torch.as_tensor(pointer,dtype=torch.long),target=torch.as_tensor(target,dtype=torch.long),
                  counts=torch.as_tensor(counts,dtype=torch.float32),frequency=torch.as_tensor(frequency,dtype=torch.float32),
                  history_sum=torch.as_tensor(matrix@x.numpy(),dtype=torch.float32),
                  history_mass=torch.as_tensor(np.asarray(matrix.sum(1)).ravel(),dtype=torch.float32))
        for k,v in data.items():self.register_buffer(k,v,persistent=False)
        self.register_buffer('log_degree',self.history_mass.clamp_min(1).log(),persistent=False)
        self.mixing=nn.ParameterList();self.diagonal=nn.ModuleList()
        for i,(labels,((left,right),folds)) in enumerate(zip(levels,spectral)):
            g=torch.as_tensor(labels,dtype=torch.long);size=int(g.max())+1
            self.register_buffer(f'label{i}',g,persistent=False)
            self.register_buffer(f'label_size{i}',torch.bincount(g,minlength=size).float().log1p(),persistent=False)
            self.register_buffer(f'label_left{i}',torch.as_tensor(left),persistent=False)
            self.register_buffer(f'label_right{i}',torch.as_tensor(right),persistent=False)
            self.register_buffer(f'label_fold_left{i}',torch.as_tensor(np.stack([f[0] for f in folds])),persistent=False)
            self.register_buffer(f'label_fold_right{i}',torch.as_tensor(np.stack([f[1] for f in folds])),persistent=False)
            self.mixing.append(nn.Parameter(torch.zeros(left.shape[1],left.shape[1])))  # M_l
            self.diagonal.append(curve(3))
        self.encoder=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)
        start=math.sqrt(10.)*torch.eye(rank,d)
        self.query_content=nn.Parameter(start.clone());self.key_content=nn.Parameter(start.clone())
        self.query_history=nn.Parameter(torch.zeros(rank,d));self.key_history=nn.Parameter(torch.zeros(rank,d))
        self.exponents=nn.Parameter(torch.zeros(2))  # alpha, beta
        self.theta=nn.Parameter(torch.tensor(1.))
        self.prior=curve(1)
        self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,2+self.levels))  # content, labels..., graph
        nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,math.log(math.e-1))  # softplus = 1

    def fast_parameters(self):
        """Small parameters trained at the higher learning rate."""
        return [*self.mixing,*self.diagonal.parameters(),self.exponents,self.theta,*self.prior.parameters(),*self.gate.parameters()]

    # ---- content -------------------------------------------------------------------------------------
    def content(self):
        return F.normalize(self.features+self.encoder(self.features),dim=1)

    def history(self):
        return self.history_sum/self.history_mass.clamp_min(1)[:,None]*(self.history_mass>0)[:,None]

    def loo_history(self,items,bundle):
        """Partner history of items with the bundle's own co-occurrences removed."""
        mass=self.history_mass[items]-(len(bundle)-1)
        value=(self.history_sum[items]-(self.features[bundle].sum(0)-self.features[items]))/mass.clamp_min(1)[:,None]
        return value*(mass>0)[:,None]

    def keys(self,z=None):
        z=self.content() if z is None else z
        return z@self.key_content.T+self.history()@self.key_history.T

    def cache(self):
        """The query-independent parts of evaluation scores, computed once per evaluation instead of once per batch."""
        z=self.content();h=self.history()
        return dict(content=z,history=h,keys=z@self.key_content.T+h@self.key_history.T,prior=self.prior(self.frequency.log1p()[:,None]).squeeze(1))

    def content_scores(self,seeds,bundles,cache=None):
        if cache is None:z=self.content();k=self.keys(z);h=self.history()
        else:z,k,h=cache['content'],cache['keys'],cache['history']
        phi=[];fixes=[]
        for i,s in enumerate(seeds):
            hs=h[s]
            if bundles is not None:
                b=bundles[i];hs=self.loo_history(s,b)
                fixes.append((i,b,(self.loo_history(b,b)-h[b])@self.key_history.T))
            phi.append((z[s]@self.query_content.T+hs@self.query_history.T).mean(0))
        phi=torch.stack(phi);scores=phi@k.T
        if fixes:  # bundle members as candidates: their keys use leave-one-out histories too
            rows=torch.cat([torch.full((len(b),),i) for i,b,_ in fixes]);cols=torch.cat([b for _,b,_ in fixes])
            scores=scores.index_put((rows,cols),torch.cat([delta@phi[i] for i,_,delta in fixes]),accumulate=True)
        return scores

    # ---- labels --------------------------------------------------------------------------------------
    def label_scores(self,i,seeds,bundles):
        labels=getattr(self,f'label{i}');q=len(seeds);size=int(labels.max())+1;out=torch.zeros(q,self.n)
        tally=torch.zeros(q,size)
        for j,s in enumerate(seeds):tally[j]=torch.bincount(labels[s],minlength=size).float()
        mean=tally/tally.sum(1,keepdim=True);mix=self.mixing[i]
        if bundles is None:
            out=out+(mean@getattr(self,f'label_left{i}')@mix)@getattr(self,f'label_right{i}')[labels].T
        else:
            fold=torch.tensor([fold_of(b.tolist()) for b in bundles])
            for k in fold.unique().tolist():
                rows=(fold==k).nonzero(as_tuple=True)[0]
                part=(mean[rows]@getattr(self,f'label_fold_left{i}')[k]@mix)@getattr(self,f'label_fold_right{i}')[k][labels].T
                out=out.index_put((rows,),out[rows]+part)
        m=tally[:,labels];row,col=m.nonzero(as_tuple=True)
        if len(row):
            count=torch.tensor([float(len(s)) for s in seeds]).log()
            x=torch.stack((m[row,col].log1p(),count[row],getattr(self,f'label_size{i}')[labels[col]]),1)
            out=out.index_put((row,col),self.diagonal[i](x).squeeze(1),accumulate=True)
        return out

    # ---- graph ---------------------------------------------------------------------------------------
    def graph_scores(self,seeds,bundles,cache=None):
        alpha,beta=self.exponents;q=len(seeds)
        query,source,edge=self.expand(torch.cat([torch.full((len(s),),i) for i,s in enumerate(seeds)]),torch.cat(seeds))
        target=self.target[edge];n=self.counts[edge]
        ds=self.log_degree[source];dt=self.log_degree[target]
        if bundles is not None:  # leave-one-out: remove the bundle's own pairs from counts and degrees
            member=torch.zeros(q,self.n,dtype=torch.bool)
            for i,b in enumerate(bundles):member[i,b]=True
            inside=member[query,target].float();n=n-inside
            others=torch.tensor([float(len(b)-1) for b in bundles])[query]
            ds=(self.history_mass[source]-others).clamp_min(1).log();dt=(self.history_mass[target]-inside*others).clamp_min(1).log()
        w=n*torch.exp(-alpha*ds-beta*dt)
        first=torch.zeros(q*self.n).scatter_add(0,query*self.n+target,w).view(q,self.n)
        out=self.theta*first.log1p()
        prior=(self.prior(self.frequency.log1p()[:,None]).squeeze(1) if cache is None else cache['prior']).expand(q,-1)
        if bundles is not None:
            rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)]);cols=torch.cat(bundles)
            left=(self.frequency[cols]-1).clamp_min(0)
            prior=prior.index_put((rows,cols),self.prior(left.log1p()[:,None]).squeeze(1)-prior[rows,cols],accumulate=True)
        return out+prior

    def expand(self,query,rows):
        """Every observed pair (row, partner) for the given rows, tagged with its query."""
        start=self.pointer[rows];length=self.pointer[rows+1]-start;offsets=torch.cumsum(length,0)-length
        edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
        return torch.repeat_interleave(query,length),torch.repeat_interleave(rows,length),edge

    # ---- gate and total ------------------------------------------------------------------------------
    def gate_weights(self,seeds,bundles):
        stats=[]
        for s in seeds:
            f=(self.frequency[s]-(1 if bundles is not None else 0)).clamp_min(0)
            stats.append(torch.stack((torch.tensor(float(len(s))).log(),f.log1p().mean(),(f==0).float().mean())))
        return F.softplus(self.gate(torch.stack(stats)))

    def forward(self,seeds,bundles=None,cache=None):
        """Scores (queries x catalogue). bundles: the complete training bundles, for leave-one-out training scores.
        cache: model.cache(), for evaluation only."""
        if cache is not None and bundles is not None:raise ValueError('The cache is for evaluation only')
        seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
        if bundles is not None:bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
        parts=[self.content_scores(seeds,bundles,cache)]+[self.label_scores(i,seeds,bundles) for i in range(self.levels)]+[self.graph_scores(seeds,bundles,cache)]
        w=self.gate_weights(seeds,bundles)
        return sum(w[:,j,None]*v for j,v in enumerate(parts))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
