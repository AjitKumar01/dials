# TRIAD

A small neural model for bundle completion. Given some items of a bundle (the seeds), it ranks the whole catalogue
for the missing ones. It combines three relations between the seeds and a candidate (content, labels and
co-occurrence), weighted per query by a gate. It has no per-item embeddings, so its size does not grow with the
catalogue.

One configuration, unchanged across eight datasets, with 41,368 trained parameters (41,466 with two label levels).

- [ARCHITECTURE.md](ARCHITECTURE.md) explains the model from first principles.
- [FORWARD_PASS.md](FORWARD_PASS.md) follows one real Electronic bundle through `TRIAD.forward`, with the code and the trained model's actual numbers at every step.
- [SPARSE_GRAPH.md](SPARSE_GRAPH.md) explains how the graph term is computed from sparse storage: the goal of each step, then how the code does it, on a 5-item toy example.
- [PIPELINE.md](PIPELINE.md) follows one dataset through the code, from raw files to test scores, naming every function and variable.
- [TASK_AND_BASELINES.md](TASK_AND_BASELINES.md) defines the task and describes CLHE, RaMen, the simple baselines and TRIAD in one notation.
- [RESULTS.md](RESULTS.md) has every result: against simple and neural baselines, under DDBC's protocol, the
  ablations, and model sizes.

## Model

For seeds S and candidate item c,

    s(c | S) = w_C(S) C(S,c) + Σ_l w_l(S) L_l(S,c) + w_G(S) G(S,c)

- **Content** `C = φ(S)ᵀψ(c)`. φ averages over the seeds and ψ maps the candidate, each a rank-64 map of an item's
  encoded features and its partner history, the co-occurrence-weighted mean feature of its partners.
- **Labels**, one term per attribute level (category; artist and album). With `m` the number of seeds that share the
  candidate's label value g(c), and `|g(c)|` the number of items with that value:

      L_l(S,c) = δ_l(m, |S|, |g(c)|)·[m > 0]

  δ_l is a small MLP, and its sign is learned. The model learns to avoid the seeds' category on fashion and to prefer
  the same artist on music.
- **Graph** `G = θ·log(1 + Σ_{s∈S} n_sc deg_s^-α deg_c^-β) + ρ(f_c)`. These are damped co-occurrence counts with
  learned degree exponents, plus a learned curve ρ of the candidate's training frequency.
- **Gate** `w(S) = softplus(MLP(log|S|, mean seed log-frequency, share of unseen seeds))`.

**Leakage-free training.** A training bundle is scored with its own contribution removed from every count, degree,
frequency and partner history (leave-one-out). The label term reads only item labels, which no bundle changes.

**Training.** Each step takes 32 training bundles and hides a random part of each. A full-catalogue softmax loss then
makes every hidden item compete with all items outside its bundle. AdamW is used, at 1e-2 for the small relation
parameters and 1e-3 for the content maps, with weight decay 1e-3. The epoch is chosen on validation NDCG@20
(patience 6, at most 100 epochs). Configuration: `triad/train.py:CONFIG`.

**Earlier version.** TRIAD was called DIALS while it also had a low-rank label-to-label part with cross-fitted label
factors. Retrained without that part, test NDCG@20 changed by at most .004 on every dataset (ARCHITECTURE.md §15),
so it was removed.

## Results

Test Recall@20 and NDCG@20, full-catalogue ranking, mean of seeds 0, 1 and 2. On POG and Spotify, "clean" excludes
test queries whose complete bundle also occurs in training or validation, which happens in the supplied splits. The
e-commerce splits have no such overlap.

| Dataset | Test queries (clean) | Recall@20 clean | NDCG@20 clean | NDCG@20 all | Inference ms/query |
|---|---:|---:|---:|---:|---:|
| Electronic | 346 | .4669 | .3345 | .3345 | 0.18 |
| Food | 353 | .6178 | .4738 | .4738 | 0.16 |
| Clothing | 371 | .5927 | .4388 | .4388 | 0.17 |
| Dunnhumby | 5,707 | .1481 | .1012 | .1012 | 0.57 |
| POG_DENSE | 5,938 (5,547) | .3742 | .2781 | .3063 | 0.41 |
| POG | 4,000 (3,995) | .0524 | .0306 | .0314 | 0.51 |
| Spotify_sparse | 2,498 (2,497) | .1837 | .2562 | .2565 | 2.5 |
| Spotify | 4,000 (3,995) | .1674 | .2823 | .2830 | 5.4 |

`python -m triad test` writes these numbers to `runs/summary.json`. Inference time is the test pass on 4 CPU threads
divided by the number of queries, after a one-time setup of the pair counts. Item features are the released ones
(BLIP image and text on POG, CLAP audio and text on Spotify, the files CLHE uses); where nothing is released, BLIP text
embeddings of the item titles that we compute (e-commerce) and TF-IDF of the product text (Dunnhumby). RESULTS.md
gives the comparisons with baselines.

## Data

Data is not part of the repository. By default it is read from the repository's parent directory; set
`TRIAD_DATA_ROOT` to read it from elsewhere.

    <root>/bundlemllm/datasets/{pog,pog_dense,spotify,spotify_sparse}/   supplied splits, item_info.json, released features
    <root>/BundleRec/dataset/{electronic,food,clothing}/                 bundle_item.csv, item_titles.csv, item_categories.csv

- **POG and Spotify:** the supplied train, validation and test files are used unmodified. The released item features
  (POG: BLIP image and text; Spotify: CLAP audio and text; only the released files) are projected to 128 dimensions by a
  truncated SVD.
- **Electronic, Food and Clothing:** duplicate bundles are removed, then a seeded 70/10/20 split is made, with half of
  each held-out bundle hidden. No features are released, so each item title is embedded with BLIP's text encoder
  (checkpoint `Salesforce/blip-itm-base-coco`, the 768-d [CLS] vector) and projected to 128 dimensions. The embeddings
  are cached in `cache/blip/`; computing them needs the `transformers` package and downloads the checkpoint once.

Details: `triad/data.py`.

## Usage

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    for d in electronic food clothing pog_dense pog spotify_sparse spotify; do
      for s in 0 1 2; do .venv/bin/python -m triad train --dataset $d --seed $s; done
    done
    .venv/bin/python -m triad test

`train` writes `runs/<dataset>/seed<s>/` (`model.pt`, `selected.json`, `history.json`) and resumes if interrupted.
Seeds can run in parallel. One seed takes seconds on the e-commerce data, about an hour on POG and a few hours on
Spotify on one CPU thread. `test` scores each selected model on the test queries once.

**Reproducibility.** Training is deterministic: fixed seeds, `torch.use_deterministic_algorithms`, and one thread by
default. The truncated SVDs behind the features are bit-reproducible only at a fixed BLAS thread count, so importing
`triad` pins BLAS to one thread. Import it before numpy, or `cache/` may differ in the last digits.

## Layout

    triad/data.py       datasets: splits, features, labels, clean test queries
    triad/model.py      the model and the pair counts
    triad/train.py      configuration, loss, training loop, checkpoint loading
    triad/evaluate.py   full-catalogue Recall@20 / NDCG@20
    triad/__main__.py   train and test commands
