"""Datasets: splits, item features and item labels for the seven bundle-completion datasets.

POG, POG_DENSE, Spotify_sparse, Spotify: the supplied train/validation/test files (BundleMLLM release), unmodified.
Features are the released modalities (POG: title, image, description; Spotify: audio), each row-normalized,
concatenated and projected to 128 dimensions by a truncated SVD. Labels: category (POG); artist and album (Spotify).
Test queries whose complete bundle also occurs in training or validation are flagged; results are reported on the rest.

Electronic, Food, Clothing (BundleRec release): duplicate bundles removed, a seeded 70/10/20 split, and half of each
held-out bundle hidden. Features: TF-IDF of item titles, projected to 128 dimensions. Label: category.

Data is not part of the repository. By default it sits next to the repository, in
<root>/bundlemllm/datasets/<name> and <root>/BundleRec/dataset/<name>; set DIALS_DATA_ROOT to point elsewhere.
"""
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_ROOT=Path(os.environ.get('DIALS_DATA_ROOT') or Path(__file__).resolve().parents[2])
SUPPLIED=['pog_dense','pog','spotify_sparse','spotify']
RESPLIT=['electronic','food','clothing']
DATASETS=RESPLIT+SUPPLIED


@dataclass
class Dataset:
    name:str
    n_items:int
    train:list        # training bundles (item ID lists)
    val_pairs:list    # (seeds, held-out items)
    test_pairs:list
    features:np.ndarray
    levels:list       # one label array per attribute level, e.g. [category] or [artist, album]
    clean_test:list   # indices of test queries whose bundle is in neither training nor validation


def load(name,cache=Path('cache')):
    if name in SUPPLIED:return load_supplied(name,Path(cache)/name)
    if name in RESPLIT:return load_resplit(name)
    raise ValueError(f'unknown dataset {name!r}; known: {DATASETS}')


# ---- supplied splits ---------------------------------------------------------------------------------
def read_rows(path,n):
    rows={}
    for line in Path(path).read_text().splitlines():
        bid,*items=[int(v) for v in line.split(',')]
        if bid in rows or not items or len(items)!=len(set(items)) or min(items)<0 or max(items)>=n:
            raise ValueError(f'{path}: duplicate bundle ID, repeated or unknown item, or empty row: {bid}')
        rows[bid]=items
    return rows


def load_supplied(name,cache):
    root=DATA_ROOT/'bundlemllm'/'datasets'/name
    n=json.loads((root/'count.json').read_text())['#I']
    train=read_rows(root/'bi_train.txt',n);pairs={};used=set(train)
    for split in ('valid','test'):
        seeds,truth=read_rows(root/f'bi_{split}_input.txt',n),read_rows(root/f'bi_{split}_gt.txt',n)
        if seeds.keys()!=truth.keys() or used&seeds.keys() or any(set(seeds[b])&set(truth[b]) for b in seeds):
            raise ValueError(f'{name} {split}: inconsistent IDs, IDs shared with another split, or seeds in the ground truth')
        used|=seeds.keys();pairs[split]=[(seeds[b],truth[b]) for b in sorted(seeds)]
    train=list(train.values())
    # Label IDs follow first appearance in item_info.json; the first level is then renumbered in sorted order.
    info=json.loads((root/'item_info.json').read_text())
    if name.startswith('spotify'):attributes=[lambda v:v.get('artist_uri','?'),lambda v:v.get('album_uri','?')]
    else:attributes=[lambda v:v.get('cate',v.get('cate_id'))]
    levels=[]
    for attribute in attributes:
        ids={};labels=np.zeros(n,int)
        for k,v in info.items():labels[int(k)]=ids.setdefault(attribute(v),len(ids))
        levels.append(labels)
    levels[0]=np.unique(levels[0],return_inverse=True)[1]
    seen={tuple(sorted(set(b))) for b in train}|{tuple(sorted(set(s)|set(g))) for s,g in pairs['valid']}
    clean=[i for i,(s,g) in enumerate(pairs['test']) if tuple(sorted(set(s)|set(g))) not in seen]
    return Dataset(name,n,train,pairs['valid'],pairs['test'],projected_features(name,root,cache),levels,clean)


def projected_features(name,root,cache):
    """Released modalities, each row-normalized, concatenated, rank-128 SVD, row-normalized. Cached."""
    path=cache/'features.npy'
    if path.exists():return np.load(path)
    load_pt=lambda f:torch.load(root/f,map_location='cpu',weights_only=False).detach().float().numpy()
    if name.startswith('spotify'):modalities={'audio':load_pt('content_feature.pt')}
    else:modalities={'content':np.load(root/'title_emb.npy').astype(np.float32),'image':load_pt('content_feature.pt'),'desc':load_pt('description_feature.pt')}
    modalities={k:(v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-8)).astype(np.float32) for k,v in modalities.items()}
    x=normalize(np.concatenate([normalize(modalities[k],axis=1) for k in sorted(modalities)],axis=1),axis=1)
    features=normalize(TruncatedSVD(n_components=128,n_iter=5,random_state=0).fit_transform(x).astype(np.float32),axis=1)
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,features)
    return features


# ---- resplit e-commerce data -------------------------------------------------------------------------
def read_csv(path):
    with open(path) as f:
        rows=csv.reader(f);next(rows)
        return list(rows)


def hide(bundles,seed,fraction=.5):
    """Split each bundle into (seeds, held out): shuffle, keep round((1-fraction)|b|) items, at least one."""
    r=random.Random(seed);out=[]
    for b in bundles:
        b=b[:];r.shuffle(b);k=max(1,int(round(len(b)*(1-fraction))))
        out.append((b[:k],b[k:]) if b[k:] else (b[:-1],[b[-1]]))
    return out


def load_resplit(name,seed=0):
    root=DATA_ROOT/'BundleRec'/'dataset'/name
    raw={}
    for b,item in read_csv(root/'bundle_item.csv'):raw.setdefault(int(b),[]).append(int(item))
    items=sorted({i for v in raw.values() for i in v});index={item:k for k,item in enumerate(items)};n=len(items)
    seen=set();bundles=[]
    for v in raw.values():
        b=list(dict.fromkeys(index[i] for i in v));key=tuple(sorted(b))
        if len(b)>=2 and key not in seen:seen.add(key);bundles.append(b)
    titles=['']*n
    for row in read_csv(root/'item_titles.csv'):
        if int(row[0]) in index:titles[index[int(row[0])]]=' '.join(row[1:])
    tfidf=normalize(TfidfVectorizer(max_features=8000,stop_words='english',ngram_range=(1,2),sublinear_tf=True).fit_transform(titles),norm='l2',axis=1)
    features=TruncatedSVD(min(128,tfidf.shape[1]-1),random_state=0).fit_transform(tfidf).astype(np.float32)
    categories={};labels=np.zeros(n,int)
    for row in read_csv(root/'item_categories.csv'):
        if int(row[0]) in index:labels[index[int(row[0])]]=categories.setdefault(row[1],len(categories))
    order=list(range(len(bundles)));random.Random(seed).shuffle(order)
    a=int(.7*len(order));b=a+int(.1*len(order))
    train=[bundles[i] for i in order[:a]]
    val=hide([bundles[i] for i in order[a:b]],seed+1);test=hide([bundles[i] for i in order[b:]],seed+2)
    return Dataset(name,n,train,val,test,features,[np.unique(labels,return_inverse=True)[1]],list(range(len(test))))
