"""Datasets: splits, item features and item labels for the eight bundle-completion datasets.

POG, POG_DENSE, Spotify_sparse, Spotify: the supplied train/validation/test files (BundleMLLM release), unmodified.
Features are exactly the released ones (huggingface.co/datasets/xhLiu/BundleConstruction, built for CLHE, Ma et al.
WSDM 2024): content_feature.pt and description_feature.pt, i.e. BLIP image and text embeddings on POG and CLAP audio
and text embeddings on Spotify. Each is row-normalized; the two are concatenated and projected to 128 dimensions by a
truncated SVD. Labels: category (POG); artist and album (Spotify).
Test queries whose complete bundle also occurs in training or validation are flagged; results are reported on the rest.

Electronic, Food, Clothing (BundleRec release): duplicate bundles removed, a seeded 70/10/20 split, and half of each
held-out bundle hidden. Features: TF-IDF of item titles, projected to 128 dimensions. Label: category.

Dunnhumby (The Complete Journey): grocery baskets of 3-20 distinct products, a seeded sample of 30,000, products seen
fewer than twice dropped, duplicate baskets removed, then the same 70/10/20 split and hiding. Features: TF-IDF of
department, commodity, sub-commodity and brand, projected to 128 dimensions. Label: sub-commodity. Households are
not used.

Data is not part of the repository. By default it sits next to the repository, in <root>/bundlemllm/datasets/<name>,
<root>/BundleRec/dataset/<name> and <root>/dunnhumby; set DIALS_DATA_ROOT to point elsewhere.
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
DATASETS=RESPLIT+SUPPLIED+['dunnhumby']


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
    if name=='dunnhumby':return load_baskets()
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
    """The two released modalities, each row-normalized, concatenated, rank-128 SVD, row-normalized. Cached."""
    path=cache/'features_released.npy'  # features.npy held an earlier choice of modalities
    if path.exists():return np.load(path)
    load_pt=lambda f:torch.load(root/f,map_location='cpu',weights_only=True).detach().float().numpy()
    modalities={'media':load_pt('content_feature.pt'),'text':load_pt('description_feature.pt')}  # BLIP image/text; CLAP audio/text
    if any(len(v)!=json.loads((root/'count.json').read_text())['#I'] for v in modalities.values()):raise ValueError(f'{name}: feature rows differ from the catalogue')
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


# ---- grocery baskets ---------------------------------------------------------------------------------
def load_baskets(seed=0,cap=30000):
    root=DATA_ROOT/'dunnhumby'
    raw={}
    with open(root/'transaction_data.csv') as f:
        rows=csv.reader(f);header=next(rows);b,p=header.index('BASKET_ID'),header.index('PRODUCT_ID')
        for row in rows:raw.setdefault(row[b],set()).add(int(row[p]))
    baskets=[sorted(v) for v in raw.values() if 3<=len(v)<=20]
    random.Random(0).shuffle(baskets);baskets=baskets[:cap]
    counts={}
    for basket in baskets:
        for i in basket:counts[i]=counts.get(i,0)+1
    baskets=[[i for i in basket if counts[i]>=2] for basket in baskets];baskets=[v for v in baskets if len(v)>=3]
    items=sorted({i for v in baskets for i in v});index={item:k for k,item in enumerate(items)};n=len(items)
    seen=set();bundles=[]
    for v in baskets:
        key=tuple(index[i] for i in v)
        if key not in seen:seen.add(key);bundles.append(list(key))
    labels=np.full(n,-1);ids={};text=['']*n
    with open(root/'product.csv') as f:
        rows=csv.reader(f);header=next(rows);product=list(rows)
    col={k:header.index(k) for k in ('PRODUCT_ID','DEPARTMENT','COMMODITY_DESC','SUB_COMMODITY_DESC','BRAND')}
    for row in product:
        if int(row[col['PRODUCT_ID']]) in index:
            j=index[int(row[col['PRODUCT_ID']])];labels[j]=ids.setdefault(row[col['SUB_COMMODITY_DESC']].strip() or '?',len(ids))
            text[j]=' '.join(row[col[k]] for k in ('DEPARTMENT','COMMODITY_DESC','SUB_COMMODITY_DESC','BRAND'))
    for j in np.flatnonzero(labels<0):labels[j]=len(ids);ids[f'_unknown{j}']=labels[j]  # products missing from product.csv
    tfidf=normalize(TfidfVectorizer(max_features=8000,stop_words='english',ngram_range=(1,2),sublinear_tf=True).fit_transform(text),norm='l2',axis=1)
    features=TruncatedSVD(min(128,tfidf.shape[1]-1),random_state=0).fit_transform(tfidf).astype(np.float32)
    order=list(range(len(bundles)));random.Random(seed).shuffle(order)
    a=int(.7*len(order));b=a+int(.1*len(order))
    train=[bundles[i] for i in order[:a]]
    val=hide([bundles[i] for i in order[a:b]],seed+1);test=hide([bundles[i] for i in order[b:]],seed+2)
    return Dataset('dunnhumby',n,train,val,test,features,[np.unique(labels,return_inverse=True)[1]],list(range(len(test))))
