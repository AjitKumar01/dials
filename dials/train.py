"""Training: random seed/target splits of training bundles, a full-catalogue softmax loss, early stopping on validation.

Each step takes 32 training bundles, hides a random part of each (half, or a uniform cut, with equal chance), and
scores the whole catalogue from the visible part with the bundle's own counts removed (leave-one-out). Each hidden
item competes against every item outside the bundle. Two AdamW groups: the small relation parameters at 1e-2, the
content maps at 1e-3, weight decay 1e-3. The epoch is chosen on validation NDCG@20 (patience 6, at most 100 epochs).

See ARCHITECTURE.md §10 (training) and §11 (leakage). The flow of one run is:

    fit
     ├─ build            DIALS(...) with the cached label factors from spectral
     ├─ for each epoch
     │   ├─ for each batch of 32 training bundles
     │   │   ├─ hide     split each bundle into visible seeds and hidden targets
     │   │   ├─ model    leave-one-out scores for the whole catalogue (q x n)
     │   │   └─ loss     full-catalogue softmax; AdamW step
     │   └─ evaluate     validation NDCG@20; keep the best epoch; stop after 6 epochs without improvement
     └─ write            model.pt (best epoch), selected.json, history.json
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .evaluate import evaluate
from .model import DIALS,cross_fitted_labels

# The one configuration used on every dataset; nothing is tuned per dataset.
#   epochs, patience: at most 100 epochs; stop after 6 without a better validation NDCG@20
#   batch:            training bundles per step
#   shared_lr:        content maps and encoder
#   fast_lr:          the small relation parameters (model.fast_parameters)
#   rank:             r, rank of the content maps;  label_rank: k, rank of the label factors;  hidden: encoder width
CONFIG=dict(epochs=100,batch=32,patience=6,shared_lr=.001,fast_lr=.01,weight_decay=.001,rank=64,label_rank=16,hidden=32)


def log(message):
    """Print a message with a UTC timestamp."""
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),message,flush=True)


def write_json(path,value):
    """Write JSON atomically: to a temporary file first, then rename, so a crash never leaves half a file."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=1))
    temporary.replace(path)


def spectral(ds,cache,config=CONFIG):
    """Cross-fitted label factors; cached, since they depend only on the training bundles and labels.

    Returns model.cross_fitted_labels(...): per level, (P_l, Q_l) from all training bundles and one pair per fold.
    The cache file name contains a hash of the training bundles and labels, so a changed split never reuses it.
    """
    import hashlib

    # Hash of the training bundles and every label level.
    h=hashlib.sha256(json.dumps([sorted(set(b)) for b in ds.train]).encode())
    for l in ds.levels:
        h.update(np.asarray(l,dtype='<i8').tobytes())
    path=Path(cache)/ds.name/f'labels_r{config["label_rank"]}_{h.hexdigest()[:16]}.npz'

    # Cached: rebuild the same nested structure from the saved arrays.
    if path.exists():
        saved=np.load(path)
        return [((saved[f'{i}_left'],saved[f'{i}_right']),
                 list(zip(saved[f'{i}_fold_left'],saved[f'{i}_fold_right'])))
                for i in range(len(ds.levels))]

    # Not cached: compute, then save one array per level and factor.
    out=cross_fitted_labels(ds.train,ds.n_items,ds.levels,config['label_rank'])
    arrays={}
    for i,((left,right),folds) in enumerate(out):
        arrays.update({f'{i}_left':left,
                       f'{i}_right':right,
                       f'{i}_fold_left':np.stack([f[0] for f in folds]),
                       f'{i}_fold_right':np.stack([f[1] for f in folds])})
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,**arrays)
    return out


def build(ds,cache,config=CONFIG):
    """A freshly initialized DIALS for a dataset (fixed structures built, parameters at their initial values)."""
    return DIALS(ds.features,ds.levels,ds.train,spectral(ds,cache,config),rank=config['rank'],hidden=config['hidden'])


def load(ds,run,cache):
    """The selected model of a finished run directory."""
    choice=json.loads((Path(run)/'selected.json').read_text())
    model=build(ds,cache,choice['config'])
    model.load_state_dict(torch.load(Path(run)/'model.pt',map_location='cpu',weights_only=False))
    return model.eval()


def hide(bundles,rng):
    """Split each training bundle into visible seeds and hidden targets (§10, "Examples").

    The bundle's distinct items are shuffled. With probability 1/2 the cut is at the middle; otherwise it is uniform
    in 1..|b|-1. Either way at least one item is on each side. Varying the cut trains for queries of every size.
    """
    seeds=[]
    truth=[]
    for b in bundles:
        row=rng.permutation(sorted(set(b)))
        cut=max(1,len(row)//2) if rng.random()<.5 else int(rng.integers(1,len(row)))
        seeds.append(row[:cut].tolist())
        truth.append(row[cut:].tolist())
    return seeds,truth


def loss(scores,seeds,truth):
    """Mean over hidden items of softplus(logsumexp over items outside the bundle - item score).

    For a hidden item j:  softplus(LSE - s_j) = -log sigma(s_j - LSE),  LSE = log sum_{c not in bundle} exp(s_c).
    Every catalogue item outside the bundle is a negative; seeds and the other hidden items are excluded, since they
    are positives too. The per-query mean over hidden items is averaged over the batch.

    Args:
        scores: q x n model scores.
        seeds, truth: per query, the visible and hidden items.
    """
    # allowed: candidates that count as negatives (outside the bundle). positive: the hidden items.
    allowed=torch.ones_like(scores,dtype=torch.bool)
    positive=torch.zeros_like(scores,dtype=torch.bool)
    for q,(s,g) in enumerate(zip(seeds,truth)):
        allowed[q,list(set(s)|set(g))]=False
        positive[q,list(g)]=True

    # LSE over the negatives, one per query (q x 1).
    negatives=scores.masked_fill(~allowed,-torch.inf).logsumexp(1,keepdim=True)
    return ((F.softplus(negatives-scores)*positive).sum(1)/positive.sum(1)).mean()


def fit(ds,seed,run,cache,config=CONFIG):
    """Train one seed into the run directory; resumes from the last saved epoch. Returns selected.json.

    Files in the run directory:
        model.pt        parameters of the best epoch on validation NDCG@20
        last.pt         full training state after the latest epoch, for resuming; deleted when the run finishes
        history.json    loss and validation metrics per epoch
        selected.json   the chosen epoch, its validation metrics, the config, parameter count and training time
    """
    run=Path(run)
    run.mkdir(parents=True,exist_ok=True)
    # Already finished.
    if (run/'selected.json').exists():
        return json.loads((run/'selected.json').read_text())

    # ---- set up ------------------------------------------------------------------------------------------------
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=build(ds,cache,config)
    # A bundle needs at least two distinct items to give a seed and a target.
    train=[b for b in ds.train if len(set(b))>=2]

    # Two optimizer groups (§10): the small relation parameters learn 10 x faster than the content maps.
    fast=model.fast_parameters()
    shared=[p for p in model.parameters() if all(p is not t for t in fast)]
    optimizer=torch.optim.AdamW([dict(params=fast,lr=config['fast_lr']),
                                 dict(params=shared,lr=config['shared_lr'])],
                                weight_decay=config['weight_decay'])

    # Early-stopping state: best validation NDCG@20 so far, its epoch, and epochs since it improved.
    state=dict(epoch=0,best=-float('inf'),best_epoch=0,stale=0,history=[],elapsed=0.)

    # ---- resume an interrupted run -----------------------------------------------------------------------------
    if (run/'last.pt').exists():
        saved=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if saved['config']!=config or saved['seed']!=seed:
            raise RuntimeError('Resume mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        # Restoring both random generators makes a resumed run identical to an uninterrupted one.
        torch.set_rng_state(saved['torch_rng'])
        rng.bit_generator.state=saved['numpy_rng']
        state=saved['state']

    log(f'{ds.name} seed={seed} parameters={model.parameter_count()} train_bundles={len(train)} start_epoch={state["epoch"]+1}')
    started=time.monotonic()

    # ---- epochs ------------------------------------------------------------------------------------------------
    for epoch in range(state['epoch']+1,config['epochs']+1):
        if state['stale']>=config['patience']:
            break
        model.train()
        order=rng.permutation(len(train))
        losses=[]

        # One step per batch of training bundles.
        for offset in range(0,len(order),config['batch']):
            rows=[train[i] for i in order[offset:offset+config['batch']]]
            seeds,truth=hide(rows,rng)
            optimizer.zero_grad(set_to_none=True)
            # Passing the complete bundles switches the model to leave-one-out scoring (§11).
            value=loss(model(seeds,bundles=[list(set(b)) for b in rows]),seeds,truth)
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(value.detach()),len(rows)))

        # Validation; keep the parameters of the best epoch.
        val=evaluate(model,ds.val_pairs)
        if val['ndcg20']>state['best']:
            state.update(best=val['ndcg20'],best_epoch=epoch,stale=0)
            torch.save(model.state_dict(),run/'model.pt')
        else:
            state['stale']+=1

        # Record the epoch: bundle-weighted mean loss and validation metrics.
        state['history'].append(dict(epoch=epoch,
                                     loss=sum(v*n for v,n in losses)/sum(n for _,n in losses),
                                     recall20=val['recall20'],
                                     ndcg20=val['ndcg20']))
        state.update(epoch=epoch,elapsed=state['elapsed']+time.monotonic()-started)
        started=time.monotonic()

        # Save everything needed to resume, atomically (write last.tmp, then rename).
        torch.save(dict(model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(),
                        numpy_rng=rng.bit_generator.state,
                        config=config,
                        seed=seed,
                        state=state),
                   run/'last.tmp')
        (run/'last.tmp').replace(run/'last.pt')
        log(f'{ds.name} seed={seed} epoch={epoch} loss={state["history"][-1]["loss"]:.5f} VAL R20={val["recall20"]:.6f} N20={val["ndcg20"]:.6f} best_epoch={state["best_epoch"]} patience={state["stale"]}/{config["patience"]}')

    # ---- finish ------------------------------------------------------------------------------------------------
    # Reload the best epoch and check that it reproduces its validation score exactly.
    model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=False))
    val=evaluate(model,ds.val_pairs)
    if abs(val['ndcg20']-state['best'])>1e-9:
        raise RuntimeError('Selected checkpoint does not replay its validation score')

    choice=dict(dataset=ds.name,
                seed=seed,
                config=config,
                best_epoch=state['best_epoch'],
                epochs_run=state['epoch'],
                validation=dict(recall20=val['recall20'],ndcg20=val['ndcg20']),
                parameters=model.parameter_count(),
                seconds=state['elapsed'])
    write_json(run/'history.json',state['history'])
    write_json(run/'selected.json',choice)
    (run/'last.pt').unlink()
    return choice
