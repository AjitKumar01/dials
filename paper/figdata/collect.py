"""Dataset statistics (Table 1) and the learned same-label curves delta_l (Fig. 4), from the frozen runs.

PYTHONPATH=. python paper/figdata/collect.py
"""
import dials  # pins BLAS threads
import json, numpy as np, torch
from torch import nn
from dials import data
from dials.model import curve
D=['electronic','food','clothing','pog_dense','pog','spotify_sparse','spotify']
out={}
for name in D:
    ds=data.load(name)
    tr=[len(set(b)) for b in ds.train]; seeds=[len(set(s)) for s,_ in ds.test_pairs]; tg=[len(set(g)) for _,g in ds.test_pairs]
    st=dict(items=ds.n_items,train=len(ds.train),val=len(ds.val_pairs),test=len(ds.test_pairs),clean_test=len(ds.clean_test),
            mean_size=float(np.mean(tr)),median_seeds=float(np.median(seeds)),mean_targets=float(np.mean(tg)),
            levels=[int(np.max(l))+1 for l in ds.levels],features=int(ds.features.shape[1]))
    curves=[]
    for lvl,lab in enumerate(ds.levels):
        sizes=np.bincount(lab); sizes=sizes[sizes>0]
        grid=np.unique(np.round(np.geomspace(1,sizes.max(),40)).astype(int))
        per_seed=[]
        for s in (0,1,2):
            sd=torch.load(f'runs/{name}/seed{s}/model.pt',map_location='cpu',weights_only=False)
            mlp=curve(3); mlp.load_state_dict({k.split(f'diagonal.{lvl}.')[1]:v for k,v in sd.items() if k.startswith(f'diagonal.{lvl}.')})
            S=st['median_seeds']
            x=torch.tensor([[np.log1p(1.),np.log(S),np.log1p(g)] for g in grid],dtype=torch.float32)
            with torch.no_grad(): per_seed.append(mlp(x).squeeze(1).tolist())
        curves.append(dict(level=lvl,grid=grid.tolist(),delta=per_seed,label_size_median=float(np.median(sizes)),label_size_q90=float(np.quantile(sizes,.9))))
    st['delta_curves']=curves; out[name]=st
    print(name,{k:v for k,v in st.items() if k!='delta_curves'},flush=True)
json.dump(out,open('paper/figdata/datasets.json','w'),indent=1)
