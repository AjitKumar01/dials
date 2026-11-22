# TRIAD

A small neural model for bundle completion. Given some items of a bundle (the seeds), it ranks the whole catalogue
for the missing ones. It combines three relations between the seeds and a candidate (content, labels and
co-occurrence), weighted per query by a gate. It has no per-item embeddings, so its size does not grow with the
catalogue: 74,136 trained parameters (74,234 with two label levels).

**It starts as the simple scorer.** Before training, TRIAD is an equal blend of content cosine and cosine-normalized
co-occurrence. Every learned part (the content matrices, the count exponents, the label rule, the popularity prior and
the gate) starts at that simple choice, and training moves it only where the data supports it.

- [TRIAD_DESIGN.md](TRIAD_DESIGN.md) builds the model step by step from the simple scorer and reads what each part
  learned, with the evidence in the trained parameters.
- [ARCHITECTURE.md](ARCHITECTURE.md) is the technical description: every component, its shapes and starting values,
  training, leakage control, evaluation and cost.
- [TASK_AND_BASELINES.md](TASK_AND_BASELINES.md) defines the task and describes the simple baselines, CLHE, RaMen and
  TRIAD in one notation.
- [RESULTS.md](RESULTS.md) has the results and where every number comes from.

## Model

For seeds S and candidate item c, with each term divided by its largest absolute value over the catalogue (T^):

    s(c | S) = w_C(S) C^(S,c) + Σ_l w_l(S) L_l^(S,c) + w_G(S) G^(S,c)

- **Content** `C = z̄ᵀ M_zz z_c + z̄ᵀ M_zh h_c + h̄ᵀ M_hz z_c + h̄ᵀ M_hh h_c`, with learned 128 × 128 matrices
  `M = Aᵀ B`, refined features `z = normalize(x + MLP(x))` and partner histories `h` (the average feature of an item's
  past bundle partners). It starts as `10 x̄ᵀ x_c`: cosine.
- **Labels**, one term per attribute level (category; artist and album): `L_l = δ_l(log(1+m), log|S|, log(1+n_g))`
  when `m > 0` seeds share the candidate's label, a small MLP whose sign is learned. It starts at 0.
- **Graph** `G = θ log(1 + Σ_s N_sc deg_s^-α deg_c^-β) + ρ(log(1 + f_c))`: damped co-occurrence counts with learned
  exponents (starting at ½, cosine) and a learned popularity prior ρ (starting at 0).
- **Gate** `w(S) = softplus(MLP(log|S|, mean seed log-frequency, share of never-bundled seeds))`, starting at 1.

**Leakage-free training.** A training bundle is scored with its own contribution removed from every count, degree,
frequency and partner history (leave-one-out).

**Training.** Each step takes 32 training bundles and hides a random part of each; a full-catalogue softmax loss makes
every hidden item compete with all items outside its bundle. AdamW at 1e-2 for the small relation parameters and 1e-3
for the content maps and encoder (both halved on the three small e-commerce sets), weight decay 1e-3. The epoch is
chosen on validation NDCG@20 (patience 6, at most 100 epochs), and epoch 0, the untrained simple scorer, can be chosen.
Configuration: `triad/train.py` (`CONFIG`, `config_for`).

## Results

Test Recall@20 and NDCG@20, full-catalogue ranking, mean of seeds 0, 1 and 2. On POG and Spotify, "clean" excludes
test queries whose complete bundle also occurs in training or validation, which happens in the supplied splits. The
e-commerce splits have no such overlap.

| Dataset | Test queries (clean) | Recall@20 clean | NDCG@20 clean | NDCG@20 all | Inference ms/query |
|---|---:|---:|---:|---:|---:|
| Electronic | 346 | .4665 | .3413 | .3413 | 0.09 |
| Food | 353 | .6211 | .4881 | .4881 | 0.09 |
| Clothing | 371 | .5990 | .4509 | .4509 | 0.10 |
| POG_DENSE | 5,938 (5,547) | .3742 | .2798 | .3084 | 0.42 |
| POG | 4,000 (3,995) | .0532 | .0308 | .0315 | 0.56 |
| Spotify_sparse | 2,498 (2,497) | .1845 | .2565 | .2568 | 2.7 |
| Spotify | 4,000 (3,995) | .1682 | .2824 | .2831 | 3.6 |

`python -m triad test` writes these numbers to `runs/summary.json`. TRIAD is significantly ahead of the best
validation-tuned simple baseline on all seven datasets (RESULTS.md). Inference time is the test pass on 4 CPU threads
divided by the number of queries, after a one-time setup of the pair counts.

## Data

Data is not part of the repository. By default it is read from the repository's parent directory; set
`TRIAD_DATA_ROOT` to read it from elsewhere.

    <root>/bundlemllm/datasets/{pog,pog_dense,spotify,spotify_sparse}/   supplied splits, item_info.json, released features
    <root>/BundleRec/dataset/{electronic,food,clothing}/                 bundle_item.csv, item_titles.csv, item_categories.csv

- **POG and Spotify:** the supplied train, validation and test files are used unmodified. The released item features
  (POG: BLIP image and text; Spotify: CLAP audio and text; the files CLHE uses) are projected to 128 dimensions by a
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
    .venv/bin/python -m triad baselines

- `train` writes `runs/<dataset>/seed<s>/` (`model.pt`, `selected.json`, `history.json`) and resumes if interrupted.
  Seeds can run in parallel. One seed takes seconds on the e-commerce data and about an hour on POG on the CPU; use
  `--device cuda` for the Spotify sets.
- `test` scores each selected model on the test queries once.
- `baselines` tunes the simple baselines on validation and scores test once (`runs/baselines/<dataset>.json`).
- The analyses behind TRIAD_DESIGN.md and RESULTS.md: `python -m analysis.readings`, `analysis.extra`,
  `analysis.taste` and `analysis.tables` (outputs in `results/` and `figures/design/`).

**Reproducibility.** CPU training is deterministic: fixed seeds, `torch.use_deterministic_algorithms`, and one thread
by default. The truncated SVDs behind the features are bit-reproducible only at a fixed BLAS thread count, so
importing `triad` pins BLAS to one thread. Import it before numpy, or `cache/` may differ in the last digits.
`PYTHONPATH=. python -m pytest tests -q` checks leave-one-out, the metrics, golden scores, resuming, and that the
untrained model is the simple scorer.

## Layout

    triad/data.py         datasets: splits, features, labels, clean test queries
    triad/model.py        the model and the pair counts
    triad/train.py        configuration, loss, training loop, checkpoint loading
    triad/evaluate.py     full-catalogue Recall@20 / NDCG@20
    triad/baselines.py    the simple baselines and their tuning
    triad/statistics.py   paired tests and Holm correction
    triad/__main__.py     train, test and baselines commands
    analysis/             parameter readings, knock-outs, results tables and figures
    paper/                the paper (its method and results sections still describe the original TRIAD; being updated)
    tests/                checks of the model, training and metrics

Earlier versions of the model, their documents and experiments are on the branch `archive/original-triad`.
