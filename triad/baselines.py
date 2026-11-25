"""The simple baselines of the paper, tuned on validation NDCG@20; test scored once with the chosen setting.

Every method maps the seeds of a query to a score for every catalogue item, using training bundles only. Write y for
the query's seed indicator (1 x n), X for the training bundle-item incidence matrix, C = X^T X with a zero diagonal
(co-occurrence counts), f_c for the number of training bundles holding c, and F for the item features of TRIAD
(rows normalized). Exact ties are broken by training popularity: every score gets + 1e-6 log(1 + f_c).

  itemknn            y K, K the cosine C_sc / (sqrt(f_s f_c) + shrink), keeping each row's k largest entries
                     (k in 50, 200, 1000; shrink in 0, 10, 100)
  rp3beta            y W, W = f^-a X^T diag(|b|^-a) X f^-b without its diagonal (a in .5, 1; b in 0, .25, .5, .75)
  ease               y B, B the EASE item-item solution with ridge l2 in 50 ... 3000 (only when at most 40,000 items were
                     bundled in training: B is dense)
  hybrid             content + ItemKNN: (1 - w) itemknn + w y F F^T, both divided by their largest absolute value per
                     query; ItemKNN at its own validation choice; w in 0, .1, .2, .3, .5, .7, .9, 1
  label_knn_content  label + ItemKNN + content (labels: category; Spotify: artist and album):
                       label score: samelabel  sum_l a_l log(1 + m_l) + gamma log(1 + f_c), m_l = seeds sharing c's label
                                    transition sum_l a_l log(eps + t_l(S)/|S| T_l[:, g_l(c)]) + gamma log P(c | g(c)), T_l
                                               the row-normalized label-by-label co-occurrence, P(c | g) = (f_c + 1) /
                                               (F_g + |g|) the item's add-one share of its label's training frequency
                       the better of the two (on validation), min-max scaled per query, mixed with max-abs scaled
                       ItemKNN: (1 - w) itemknn + w label (w as for hybrid); then + v times the max-abs scaled
                       content score y F F^T (v in 0, .05, .1, .2, .3, .5)

Popularity prior (optional, as TRIAD learns one). The base rate b(f) of the validation targets per frequency bin (0, 1,
2, 3-4, 5-9, 10-19, 20-49, 50+): b = log((targets in bin + 1) / (targets + 8)) - log(items in bin / n). With the prior,
score' = maxabs(score) + lambda b(f_c), lambda in 0, .01, .03, .1, .3, 1, 3 chosen on validation together with the
blends' mixing weight (w for hybrid; v for label_knn_content, in 0, .05, .1, .2, .3, .5, .7, 1, 1.5, 2, 3); lambda = 0
is no prior.

  link               LINK (Choi et al., SIGIR 2025), the closed-form linear item-item model, with bundles as unordered
                     sessions. LIS: B^S = I - P diag(gamma), P = (X^T X + l2 I)^-1, gamma_j = l2 if 1 - l2 P_jj <= xi
                     else (1 - xi) / P_jj; self-distillation X' = X ((1 - beta) I + beta B^S), the same solve on X'.
                     LINK's past -> future pairs become seed -> hidden pairs: each training bundle without one item
                     (Y, rows normalized) predicts that item (Z). B = (alpha X~'^T X~' + (1 - alpha) Y^T Y + lam I)^-1
                     (alpha X~'^T X~' + (1 - alpha) Y^T Z + lam T), X~' = X' with rows normalized; score y B. The
                     teacher T is LINK's neural model in the paper; here the content teacher softmax(F F^T / tau) per row
                     (tau in .05, .1, .2, .5) or none (T = 0). LIS (l2 in 50 ... 3000, beta in 0, .25, .5, .75, xi in 0,
                     .2) is chosen first on validation, then alpha in 0, .25, .5, .75, 1, lam in .1, 1, 10, 100 and T.
                     Dense n x n: run only when the catalogue has at most 40,000 items (python -m triad link).

The best simple baseline of a dataset is the method (with its prior choice) with the highest validation NDCG@20.

python -m triad baselines [--datasets NAME ...]   writes runs/baselines/<dataset>.json
python -m triad link [--datasets NAME ...]        adds LINK to runs/baselines/<dataset>.json
"""
import numpy as np
import scipy.sparse as sp
from .evaluate import query_metrics

EDGES=[0,1,2,3,5,10,20,50,1e9]
LAMBDAS=(0.,.01,.03,.1,.3,1.,3.)
METHODS=('itemknn','rp3beta','ease','hybrid','label_knn_content')
EASE_LIMIT=40000


def maxabs(s):
    """Divide each row by its largest absolute value."""
    return s/np.maximum(np.abs(s).max(1,keepdims=True),1e-12)


def minmax(s):
    """Scale each row to [0, 1]."""
    lo=s.min(1,keepdims=True)
    return (s-lo)/np.maximum(s.max(1,keepdims=True)-lo,1e-12)


def indicator(seeds,n):
    """y: the seed indicator of every query (q x n, sparse)."""
    rows=[sorted(set(s)) for s in seeds]
    return sp.csr_matrix((np.ones(sum(map(len,rows)),np.float32),
                          (np.repeat(np.arange(len(rows)),[len(r) for r in rows]),np.concatenate(rows))),shape=(len(rows),n))


def base_rate(f,pairs):
    """b(f_c): log of the validation targets' share in each frequency bin over the catalogue's share, add-one."""
    t=np.array([f[g] for _,G in pairs for g in set(G)])
    n=len(f)
    b=np.zeros(n,np.float32)
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        items=(f>=lo)&(f<hi)
        if items.any():
            b[items]=np.log((((t>=lo)&(t<hi)).sum()+1)/(len(t)+len(EDGES)-1))-np.log(items.sum()/n)
    return b


class Baselines:
    """The simple baselines of one dataset, built from its training bundles, features and labels."""

    def __init__(self,ds):
        self.n=ds.n_items
        rows=[np.unique(np.asarray(b,np.int64)) for b in ds.train]
        indptr=np.cumsum([0]+[len(r) for r in rows])
        self.X=sp.csr_matrix((np.ones(indptr[-1],np.float32),np.concatenate(rows),indptr),shape=(len(rows),self.n))
        self.f=np.asarray(self.X.sum(0)).ravel()
        C=(self.X.T@self.X).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        self.C=C.astype(np.float32)
        self.tie=1e-6*np.log1p(self.f)[None,:].astype(np.float32)
        self.features=(ds.features/np.maximum(np.linalg.norm(ds.features,axis=1,keepdims=True),1e-12)).astype(np.float32)
        self.levels=[np.asarray(g) for g in ds.levels]
        self.onehot,self.transition=[],[]
        for g in self.levels:
            G=sp.csr_matrix((np.ones(self.n,np.float32),(np.arange(self.n),g)),shape=(self.n,int(g.max())+1))
            T=(G.T@self.C@G).tocsr().astype(np.float64)
            T=sp.diags(1/np.maximum(np.asarray(T.sum(1)).ravel(),1e-12))@T
            self.onehot.append(G)
            self.transition.append(T.tocsr().astype(np.float32))
        fine=self.levels[-1] if len(self.levels)==1 else max(self.levels,key=lambda g:g.max())
        total,size=np.bincount(fine,weights=self.f),np.bincount(fine)
        self.within=np.log((self.f+1)/(total[fine]+size[fine])).astype(np.float32)
        self.cache={}
        self.prior=None   # b(f), set by run()
        self.lam=0.

    # ---- item-item matrices ----------------------------------------------------------------------------------------
    def matrix(self,name,p):
        key=(name,tuple(sorted(p.items())))
        if key in self.cache:
            return self.cache[key]
        f=np.maximum(self.f,1)
        if name=='itemknn':
            c=self.C.tocoo()
            cos=sp.csr_matrix((c.data/(np.sqrt(f[c.row]*f[c.col])+p['shrink']),(c.row,c.col)),shape=c.shape)
            m=top_k_rows(cos,p['k'])
        elif name=='rp3beta':
            size=np.asarray(self.X.sum(1)).ravel()
            m=(sp.diags(f**-p['alpha'])@(self.X.T@sp.diags(size**-p['alpha'])@self.X)@sp.diags(f**-p['beta'])).tocsr()
            m.setdiag(0)
            m.eliminate_zeros()
        elif name=='ease':
            from scipy.linalg import inv
            active=np.flatnonzero(self.f>0)
            g=(self.X[:,active].T@self.X[:,active]).toarray().astype(np.float32)
            g[np.diag_indices_from(g)]+=p['l2']
            B=inv(g,overwrite_a=True,check_finite=False)
            del g
            B/=-np.diag(B).copy()[None,:]
            B[np.diag_indices_from(B)]=0
            m=(active,B.astype(np.float32,copy=False))
            self.cache={k:v for k,v in self.cache.items() if k[0]!='ease'}   # keep one dense EASE matrix at a time
        elif name=='link_lis':
            m=self.link_lis(p['l2'],p['beta'],p['xi'])[0]
        elif name=='link':
            m=self.link(p)
        if name in ('link_lis','link'):
            self.cache={k:v for k,v in self.cache.items() if k[0] not in ('link_lis','link')}   # one dense matrix at a time
        self.cache[key]=m
        return m

    # ---- LINK --------------------------------------------------------------------------------------------------------
    @staticmethod
    def bounded_ease(G,l2,xi):
        """argmin ||X - XB||^2 + l2 ||B||^2 s.t. diag(B) <= xi, from the Gram matrix G = X^T X (LINK, eq. of LIS)."""
        P=np.linalg.inv(G+l2*np.eye(len(G)))
        d=np.diag(P)
        gamma=np.where(1-l2*d<=xi,l2,(1-xi)/d)
        B=-P*gamma[None,:]
        B[np.diag_indices_from(B)]+=1
        return B

    def link_lis(self,l2,beta,xi):
        """LIS: self-distilled bounded EASE. Returns B^LIS and the distilled bundles X'."""
        key=('lis',l2,beta,xi)
        if key not in self.cache:
            X=self.X.toarray().astype(np.float64)
            G=X.T@X
            M=(1-beta)*np.eye(self.n)+beta*self.bounded_ease(G,l2,xi)
            self.cache={k:v for k,v in self.cache.items() if k[0]!='lis'}
            self.cache[key]=(self.bounded_ease(M.T@G@M,l2,xi),X@M)
        return self.cache[key]

    def seed_hidden(self):
        """Y^T Y and Y^T Z over all (bundle without item j -> j) pairs, Y rows normalized, in closed form from X."""
        if 'seed_hidden' not in self.cache:
            k=np.asarray(self.X.sum(1)).ravel()
            ok=k>=2
            X=self.X[ok];k=k[ok]
            w,v,u=(k-2)/(k-1)**2,1/(k-1)**2,1/(k-1)
            YY=(X.T@sp.diags(w)@X).toarray()+np.diag(np.asarray(X.T@v).ravel())
            YZ=(X.T@sp.diags(u)@X).toarray()-np.diag(np.asarray(X.T@u).ravel())
            self.cache['seed_hidden']=(YY.astype(np.float64),YZ.astype(np.float64))
        return self.cache['seed_hidden']

    def link(self,p):
        """LINK's aggregated closed form with the LIS rows, the seed -> hidden rows and the teacher T."""
        _,Xd=self.link_lis(p['l2'],p['beta'],p['xi'])
        rows=Xd.sum(1,keepdims=True)
        Xn=Xd/np.where(np.abs(rows)>1e-12,rows,1.)
        L=Xn.T@Xn
        YY,YZ=self.seed_hidden()
        T=np.zeros((self.n,self.n))
        if p['tau'] is not None:
            S=(self.features@self.features.T).astype(np.float64)/p['tau']
            S-=S.max(1,keepdims=True)
            T=np.exp(S)
            T/=T.sum(1,keepdims=True)
        a,lam=p['alpha'],p['lam']
        return np.linalg.solve(a*L+(1-a)*YY+lam*np.eye(self.n),a*L+(1-a)*YZ+lam*T).astype(np.float32)

    # ---- scores ----------------------------------------------------------------------------------------------------
    def label_scores(self,name,p,y,size):
        out=np.zeros((y.shape[0],self.n),np.float32)
        for a,G,T,g in zip(p['a'],self.onehot,self.transition,self.levels):
            if a==0:
                continue
            t=y@G   # seeds' label counts
            if name=='samelabel':
                out+=a*np.log1p(np.asarray(t.todense())[:,g])
            else:
                out+=a*np.log(p['eps']+np.asarray((t@T).todense())[:,g]/size)
        prior=np.log1p(self.f).astype(np.float32) if name=='samelabel' else self.within
        return out+p['gamma']*prior[None,:]

    def raw(self,name,p,seeds):
        """Scores without the tie term (q x n)."""
        y=indicator(seeds,self.n)
        if name in ('itemknn','rp3beta'):
            return np.asarray((y@self.matrix(name,p)).todense())
        if name=='ease':
            active,B=self.matrix(name,p)
            out=np.zeros((len(seeds),self.n),np.float32)
            out[:,active]=np.asarray(y[:,active].todense())@B
            return out
        if name in ('link_lis','link'):
            return np.asarray(y@self.matrix(name,p))
        content=(y@self.features)@self.features.T
        if name=='hybrid':
            knn=np.asarray((y@self.matrix('itemknn',p['itemknn'])).todense())
            return (1-p['w'])*maxabs(knn)+p['w']*maxabs(content)
        size=np.array([len(set(s)) for s in seeds],np.float32)[:,None]
        if name in ('samelabel','transition'):
            return self.label_scores(name,p,y,size)
        knn=maxabs(np.asarray((y@self.matrix('itemknn',p['itemknn'])).todense()))
        out=(1-p['w'])*knn+p['w']*minmax(self.label_scores(p['label'],p['label_params'],y,size))
        if name=='label_knn_content':
            out=out+p['v']*maxabs(content)
        return out

    def scores(self,name,p,seeds):
        s=self.raw(name,p,seeds).astype(np.float32)
        if self.lam:
            s=maxabs(s)+self.lam*self.prior[None,:]
        return (s+self.tie).astype(np.float32)

    def evaluate(self,name,p,pairs,batch=128):
        """Recall@20 and NDCG@20 per query."""
        out=[]
        for a in range(0,len(pairs),batch):
            rows=pairs[a:a+batch]
            values=self.scores(name,p,[s for s,_ in rows])
            out+=[{k:v for k,v in query_metrics(values[i],s,g).items() if k!='top20'} for i,(s,g) in enumerate(rows)]
        return out


def top_k_rows(m,k):
    """Keep the k largest entries of every row (ItemKNN neighbourhoods)."""
    m=m.tocsr()
    data,indptr=m.data,m.indptr
    keep=np.zeros(len(data),bool)
    for r in range(m.shape[0]):
        a,b=indptr[r],indptr[r+1]
        if b-a<=k:
            keep[a:b]=True
        else:
            keep[a+np.argpartition(-data[a:b],k)[:k]]=True
    out=sp.csr_matrix((data*keep,m.indices,indptr),shape=m.shape)
    out.eliminate_zeros()
    return out


def mean(rows,indices=None):
    rows=rows if indices is None else [rows[i] for i in indices]
    return dict(recall20=float(np.mean([r['recall20'] for r in rows])),ndcg20=float(np.mean([r['ndcg20'] for r in rows])))


def run(ds,log=print):
    """Tune every method on validation, without and with the prior; score test once per method. Returns the report."""
    model=Baselines(ds)
    report=dict(dataset=ds.name,methods={},validation={})

    def tune(name,grid):
        rows=[]
        for p in grid:
            v=mean(model.evaluate(name,p,ds.val_pairs))['ndcg20']
            rows.append(dict(params=p,ndcg20=v))
        report['validation'][name]=rows
        best=max(rows,key=lambda r:r['ndcg20'])
        log(f'{ds.name} {name}: validation N20 {best["ndcg20"]:.4f} at {best["params"]}')
        return best['params'],best['ndcg20']

    # Without the prior.
    chosen={}
    chosen['itemknn']=tune('itemknn',[dict(k=k,shrink=h) for k in (50,200,1000) for h in (0,10,100)])
    chosen['rp3beta']=tune('rp3beta',[dict(alpha=a,beta=b) for a in (.5,1.) for b in (0.,.25,.5,.75)])
    if (model.f>0).sum()<=EASE_LIMIT:
        chosen['ease']=tune('ease',[dict(l2=l) for l in (50.,200.,500.,1000.,3000.)])
    knn=chosen['itemknn'][0]
    chosen['hybrid']=tune('hybrid',[dict(itemknn=knn,w=w) for w in (0.,.1,.2,.3,.5,.7,.9,1.)])
    weights=[(1.,)] if len(model.levels)==1 else [(1.,0.),(0.,1.),(1.,1.),(1.,.5),(.5,1.)]
    labels={name:tune(name,grid) for name,grid in (
        ('samelabel',[dict(a=a,gamma=c) for a in weights for c in (0.,.1,.3,1.)]),
        ('transition',[dict(a=a,gamma=c,eps=e) for a in weights for c in (0.,.5,1.,2.) for e in (1e-4,1e-2)]))}
    label=max(labels,key=lambda k:labels[k][1])
    lknn=tune('label_knn',[dict(label=label,label_params=labels[label][0],itemknn=knn,w=w) for w in (0.,.1,.2,.3,.5,.7,.9,1.)])[0]
    chosen['label_knn_content']=tune('label_knn_content',[dict(lknn,v=v) for v in (0.,.05,.1,.2,.3,.5)])

    # With the prior: lambda chosen together with the blends' mixing weight; the other settings stay fixed.
    model.prior=base_rate(model.f,ds.val_pairs)
    mixes=dict(hybrid=('w',(0.,.1,.2,.3,.5,.7,.9,1.)),label_knn_content=('v',(0.,.05,.1,.2,.3,.5,.7,1.,1.5,2.,3.)))
    for name,(params,validation) in chosen.items():
        mix=mixes.get(name)
        settings=[params]+([dict(params,**{mix[0]:x}) for x in mix[1] if x!=params[mix[0]]] if mix else [])
        rows=[]
        for lam in LAMBDAS:
            model.lam=lam
            rows+=[(mean(model.evaluate(name,q,ds.val_pairs))['ndcg20'],lam,q) for q in settings]
        best=max(rows,key=lambda r:r[0])
        model.lam=0.
        plain=model.evaluate(name,params,ds.test_pairs)
        model.lam=best[1]
        prior=model.evaluate(name,best[2],ds.test_pairs)
        model.lam=0.
        report['methods'][name]=dict(
            params=params,validation_ndcg20=validation,
            test=mean(plain,ds.clean_test),queries=[[plain[i]['recall20'],plain[i]['ndcg20']] for i in ds.clean_test],
            prior=dict(lam=best[1],params=best[2],validation_ndcg20=best[0],test=mean(prior,ds.clean_test),
                       queries=[[prior[i]['recall20'],prior[i]['ndcg20']] for i in ds.clean_test]))
        log(f'{ds.name} {name}: test N20 {report["methods"][name]["test"]["ndcg20"]:.4f}; with prior (lambda {best[1]}) '
            f'{report["methods"][name]["prior"]["test"]["ndcg20"]:.4f}')

    # The best simple baseline: highest validation NDCG@20 over methods, with or without the prior.
    options=[(m,v['validation_ndcg20'],False) for m,v in report['methods'].items()]
    options+=[(m,v['prior']['validation_ndcg20'],True) for m,v in report['methods'].items()]
    name,_,prior=max(options,key=lambda o:(o[1],not o[2]))
    chosen=report['methods'][name]['prior'] if prior else report['methods'][name]
    report['best_simple']=dict(method=name+('+prior' if prior else ''),test=chosen['test'])
    return report


def run_link(ds,report,log=print):
    """Tune LINK on validation (LIS first, then the aggregation and teacher), without and with the prior as for the other
    methods; score test once. Adds report['methods']['link'] and re-chooses the best simple baseline."""
    model=Baselines(ds)

    def tune(name,grid):
        rows=[dict(params=p,ndcg20=mean(model.evaluate(name,p,ds.val_pairs))['ndcg20']) for p in grid]
        report['validation'][name]=rows
        best=max(rows,key=lambda r:r['ndcg20'])
        log(f'{ds.name} {name}: validation N20 {best["ndcg20"]:.4f} at {best["params"]}')
        return best['params'],best['ndcg20']

    lis,_=tune('link_lis',[dict(l2=l,beta=b,xi=x) for l in (50.,200.,500.,1000.,3000.) for b in (0.,.25,.5,.75)
                            for x in (0.,.2)])
    params,validation=tune('link',[dict(lis,alpha=a,lam=m,tau=t) for a in (0.,.25,.5,.75,1.) for m in (.1,1.,10.,100.)
                                   for t in (None,.05,.1,.2,.5)])
    model.prior=base_rate(model.f,ds.val_pairs)
    rows=[]
    for lam in LAMBDAS:
        model.lam=lam
        rows.append((mean(model.evaluate('link',params,ds.val_pairs))['ndcg20'],lam))
    best=max(rows,key=lambda r:r[0])
    model.lam=0.
    plain=model.evaluate('link',params,ds.test_pairs)
    model.lam=best[1]
    prior=model.evaluate('link',params,ds.test_pairs)
    model.lam=0.
    report['methods']['link']=dict(
        params=params,validation_ndcg20=validation,
        test=mean(plain,ds.clean_test),queries=[[plain[i]['recall20'],plain[i]['ndcg20']] for i in ds.clean_test],
        prior=dict(lam=best[1],params=params,validation_ndcg20=best[0],test=mean(prior,ds.clean_test),
                   queries=[[prior[i]['recall20'],prior[i]['ndcg20']] for i in ds.clean_test]))
    log(f'{ds.name} link: test N20 {report["methods"]["link"]["test"]["ndcg20"]:.4f}; with prior (lambda {best[1]}) '
        f'{report["methods"]["link"]["prior"]["test"]["ndcg20"]:.4f}')
    options=[(m,v['validation_ndcg20'],False) for m,v in report['methods'].items()]
    options+=[(m,v['prior']['validation_ndcg20'],True) for m,v in report['methods'].items()]
    name,_,prior=max(options,key=lambda o:(o[1],not o[2]))
    chosen=report['methods'][name]['prior'] if prior else report['methods'][name]
    report['best_simple']=dict(method=name+('+prior' if prior else ''),test=chosen['test'])
    return report
