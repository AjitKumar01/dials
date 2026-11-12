# DIALS

**DI**agonal-plus-**L**ow-rank label relations with **S**pectral cross-fitting: a small neural model for bundle
completion. Given some items of a bundle (the seeds), rank the whole catalogue for the missing ones.

One configuration, unchanged across seven datasets, with 41,624 trained parameters (41,978 with two label levels).

- [ARCHITECTURE.md](ARCHITECTURE.md) explains the model from first principles.
- [TASK_AND_BASELINES.md](TASK_AND_BASELINES.md) defines the task and describes CLHE, RaMen, the simple baselines and DIALS in one notation.
- [RESULTS.md](RESULTS.md) has every result: against simple and neural baselines, under DDBC's protocol, the
  second-hop ablation, and model sizes.

## Model

For seeds S and candidate item c,

    s(c | S) = w_C(S) C(S,c) + Σ_l w_l(S) L_l(S,c) + w_G(S) G(S,c)

- **Content** `C = φ(S)ᵀψ(c)`. φ averages over the seeds and ψ maps the candidate, each a rank-64 map of an item's
  encoded features and its partner history, the co-occurrence-weighted mean feature of its partners.
- **Labels**, one term per attribute level (category; artist and album). With `t(S)` the seeds' label counts and
  `Γ` the item-to-label incidence:

      L_l(S,c) = δ_l(t_{g(c)}(S), |S|, |g(c)|)·[t_{g(c)}(S) > 0]  +  (t(S)/|S|) P_l M_l Q_l[g(c)]ᵀ

  The **diagonal** δ_l (a small MLP) scores a shared label, and its sign is learned. The model learns to avoid the
  seeds' category on fashion and to prefer the same artist on music. The **low-rank** part says which labels go
  together. `P_l, Q_l` are fixed factors from a rank-16 SVD of `D^-½ ΓᵀNΓ D^-½`, where N is the item co-occurrence
  matrix, and only the 16×16 matrix `M_l` is learned.
- **Graph** `G = θ·log(1 + Σ_{s∈S} n_sc deg_s^-α deg_c^-β) + ρ(f_c)`. These are damped co-occurrence counts with
  learned degree exponents, plus a learned curve ρ of the candidate's training frequency.
- **Gate** `w(S) = softplus(MLP(log|S|, mean seed log-frequency, share of unseen seeds))`.

**Leakage-free training.** A training bundle is scored with its own contribution removed from every count, degree,
frequency and partner history (leave-one-out). The label SVD cannot be corrected exactly that way, so it is
cross-fitted. Training bundles are split into five folds, with identical bundles in the same fold, and a bundle being
trained on uses factors computed without its fold.

**Training.** Each step takes 32 training bundles and hides a random part of each. A full-catalogue softmax loss then
makes every hidden item compete with all items outside its bundle. AdamW is used, at 1e-2 for the small relation
parameters and 1e-3 for the content maps, with weight decay 1e-3. The epoch is chosen on validation NDCG@20
(patience 6, at most 100 epochs). Configuration: `dials/train.py:CONFIG`.

## Results

Test Recall@20 and NDCG@20, full-catalogue ranking, mean of seeds 0, 1 and 2. On POG and Spotify, "clean" excludes
test queries whose complete bundle also occurs in training or validation, which happens in the supplied splits. The
e-commerce splits have no such overlap.

| Dataset | Test queries (clean) | Recall@20 clean | NDCG@20 clean | NDCG@20 all | Inference ms/query |
|---|---:|---:|---:|---:|---:|
| Electronic | 346 | .4691 | .3346 | .3346 | 0.11 |
| Food | 353 | .5862 | .4388 | .4388 | 0.11 |
| Clothing | 371 | .5338 | .3876 | .3876 | 0.12 |
| Dunnhumby | 5,707 | .1490 | .1023 | .1023 | 1.99 |
| POG_DENSE | 5,938 (5,547) | .3756 | .2802 | .3080 | 0.38 |
| POG | 4,000 (3,995) | .0543 | .0313 | .0320 | 0.49 |
| Spotify_sparse | 2,498 (2,497) | .1837 | .2562 | .2565 | 2.38 |
| Spotify | 4,000 (3,995) | .1674 | .2824 | .2831 | 3.24 |

`python -m dials test` writes these numbers to `runs/summary.json`. Inference time is the test pass on 4 CPU threads
divided by the number of queries, after a one-time setup of the pair counts and label SVDs. Item features are the released ones
(BLIP image and text on POG, CLAP audio and text on Spotify, the files CLHE uses) and TF-IDF of the item text where
nothing is released. The four POG and Spotify rows were retrained with these on 2026-11-11; RESULTS.md gives the
earlier numbers and the comparisons with baselines.

## Data

Data is not part of the repository. By default it is read from the repository's parent directory; set
`DIALS_DATA_ROOT` to read it from elsewhere.

    <root>/bundlemllm/datasets/{pog,pog_dense,spotify,spotify_sparse}/   supplied splits, item_info.json, released features
    <root>/BundleRec/dataset/{electronic,food,clothing}/                 bundle_item.csv, item_titles.csv, item_categories.csv

- **POG and Spotify:** the supplied train, validation and test files are used unmodified. The released item features
  (POG: BLIP image and text; Spotify: CLAP audio and text; only the released files) are projected to 128 dimensions by a
  truncated SVD.
- **Electronic, Food and Clothing:** duplicate bundles are removed, then a seeded 70/10/20 split is made, with half of
  each held-out bundle hidden. Features are TF-IDF of item titles, projected to 128 dimensions.

Details: `dials/data.py`.

## Usage

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    for d in electronic food clothing pog_dense pog spotify_sparse spotify; do
      for s in 0 1 2; do .venv/bin/python -m dials train --dataset $d --seed $s; done
    done
    .venv/bin/python -m dials test

`train` writes `runs/<dataset>/seed<s>/` (`model.pt`, `selected.json`, `history.json`) and resumes if interrupted.
Seeds can run in parallel. One seed takes seconds on the e-commerce data, about an hour on POG and a few hours on
Spotify on one CPU thread. `test` scores each selected model on the test queries once.

**Reproducibility.** Training is deterministic: fixed seeds, `torch.use_deterministic_algorithms`, and one thread by
default. The randomized SVDs behind the features and label factors are bit-reproducible only at a fixed BLAS thread
count, so importing `dials` pins BLAS to one thread. Import it before numpy, or `cache/` may differ in the last digits.

## Layout

    dials/data.py       datasets: splits, features, labels, clean test queries
    dials/model.py      the model, pair counts, cross-fitted label factors
    dials/train.py      configuration, loss, training loop, checkpoint loading
    dials/evaluate.py   full-catalogue Recall@20 / NDCG@20
    dials/__main__.py   train and test commands
