"""python -m dials train --dataset pog_dense --seed 0
python -m dials test [--datasets pog_dense pog ...]

train: one seed into runs/<dataset>/seed<s>/ (model.pt, selected.json, history.json); resumes if interrupted.
test: scores the selected model of seeds 0, 1, 2 on the test queries once (runs/<dataset>/seed<s>/test.json) and
writes runs/summary.json: three-seed means on all test queries and on the clean ones (bundle in neither training nor
validation), parameter count and inference time per query.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import data
from .evaluate import evaluate,subset
from .train import fit,load,log,write_json

# Where runs and caches live (relative to the working directory), and the three training seeds.
RUNS=Path('runs')
CACHE=Path('cache')
SEEDS=(0,1,2)


def train(args):
    """Train one dataset and seed (deterministic algorithms; thread count from --threads)."""
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    fit(data.load(args.dataset,CACHE),args.seed,RUNS/args.dataset/f'seed{args.seed}',CACHE)
    log(f'{args.dataset} seed {args.seed} COMPLETE')


def test(args):
    """Score the test queries with every finished seed, once, and update runs/summary.json."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    path=RUNS/'summary.json'
    summary=json.loads(path.read_text()) if path.exists() else {}

    for name in args.datasets:
        runs=[RUNS/name/f'seed{s}' for s in SEEDS]
        if not all((r/'selected.json').exists() for r in runs):
            log(f'{name}: not all seeds trained, skipped')
            continue

        # Test each seed once; test.json is reused afterwards.
        ds=data.load(name,CACHE)
        results=[]
        for run in runs:
            if not (run/'test.json').exists():
                write_json(run/'test.json',evaluate(load(ds,run,CACHE),ds.test_pairs))
            results.append(json.loads((run/'test.json').read_text()))

        # Mean recall20 and ndcg20 over the given queries and over the three seeds.
        mean=lambda indices:dict(zip(('recall20','ndcg20'),
                                     np.mean([[[r['queries'][i][m] for m in ('recall20','ndcg20')] for i in indices]
                                              for r in results],
                                             axis=(0,1)).tolist()))

        summary[name]=dict(
            test_queries=len(ds.test_pairs),
            clean_test_queries=len(ds.clean_test),
            all=mean(range(len(ds.test_pairs))),
            clean=mean(ds.clean_test),
            # Per seed: clean-query metrics, the selected epoch and its validation metrics.
            seeds=[dict(seed=s,
                        clean=subset(r,ds.clean_test),
                        **{k:json.loads((run/'selected.json').read_text())[k] for k in ('best_epoch','validation')})
                   for s,r,run in zip(SEEDS,results,runs)],
            parameters=json.loads((runs[0]/'selected.json').read_text())['parameters'],
            milliseconds_per_query=1000*float(np.mean([r['seconds'] for r in results]))/len(ds.test_pairs))
        write_json(path,summary)
        log(f'{name} clean test R20={summary[name]["clean"]["recall20"]:.4f} N20={summary[name]["clean"]["ndcg20"]:.4f}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(required=True)

    # python -m dials train --dataset NAME --seed S [--threads T]
    t=sub.add_parser('train')
    t.set_defaults(run=train)
    t.add_argument('--dataset',required=True,choices=data.DATASETS)
    t.add_argument('--seed',type=int,required=True)
    t.add_argument('--threads',type=int,default=1)

    # python -m dials test [--datasets NAME ...]
    t=sub.add_parser('test')
    t.set_defaults(run=test)
    t.add_argument('--datasets',nargs='+',default=data.DATASETS,choices=data.DATASETS)

    args=p.parse_args()
    args.run(args)
