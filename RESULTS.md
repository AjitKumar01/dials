# Results

All numbers are for TRIAD as it is in this repository (ARCHITECTURE.md), seeds 0, 1 and 2.

- **Item features:** the released ones on POG (BLIP image and text) and Spotify (CLAP audio and text), the same files
  CLHE uses. On Electronic, Food and Clothing nothing is released, so the item titles are embedded with BLIP's text
  encoder (computed by us), and every method gets these vectors.
- **Metrics:** Recall@20 (R) and NDCG@20 (N), full-catalogue ranking with the seeds removed.
- **Clean queries:** on POG and Spotify, "clean" excludes test queries whose complete bundle also occurs in training
  or validation. The e-commerce splits have no such overlap, so all their queries are clean.
- **Statistics:** paired tests over queries, a cluster bootstrap for the interval and a sign-flip test for p
  (`triad/statistics.py`), Holm-corrected within each table.

## 1. TRIAD

| Dataset | Items | Test queries (clean) | R clean | N clean | R all | N all | ms / query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 3,499 | 346 | .4665 | .3413 | .4665 | .3413 | 0.09 |
| Food | 3,767 | 353 | .6211 | .4881 | .6211 | .4881 | 0.09 |
| Clothing | 4,487 | 371 | .5990 | .4509 | .5990 | .4509 | 0.10 |
| POG_DENSE | 31,217 | 5,938 (5,547) | .3742 | .2798 | .4065 | .3084 | 0.42 |
| POG | 48,676 | 4,000 (3,995) | .0532 | .0308 | .0539 | .0315 | 0.56 |
| Spotify_sparse | 213,325 | 2,498 (2,497) | .1845 | .2565 | .1848 | .2568 | 2.7 |
| Spotify | 254,155 | 4,000 (3,995) | .1682 | .2824 | .1688 | .2831 | 3.6 |

`python -m triad test` writes these numbers to `runs/summary.json`. Inference time is the test pass on 4 CPU threads
divided by the number of queries, after a one-time setup of the pair counts.

**Where the runs come from.**
- **Electronic, Food, Clothing:** trained with this repository's code (`python -m triad train`), both learning rates
  halved (ARCHITECTURE.md §15).
- **POG_DENSE, POG, Spotify_sparse, Spotify:** trained before this repository took the model over, by earlier scripts
  with the same model and training loop (POG sets on the CPU, Spotify sets on a Kaggle GPU). `python -m triad test`
  on their saved models reproduces their test numbers exactly. Retraining them with `python -m triad train` uses the
  same model and loop but draws different random starting weights, so its numbers differ within seed noise.

## 2. Against the best simple baseline and the original TRIAD

Test NDCG@20 on clean queries, full training data. "Best simple" is the best of ItemKNN, RP3β, EASE, the content +
ItemKNN blend and the label + ItemKNN + content rule, each with or without a popularity prior, chosen on validation
(`python -m triad baselines`; TASK_AND_BASELINES.md). "Original TRIAD" is the earlier version (rank-64 content maps,
raw-count start, no per-query scaling; branch `archive/original-triad`), with the same features. Holm over the 10 tests
of the first five datasets, and over the 2 tests of each Spotify set.

| Dataset | Best simple (method) | Original TRIAD | TRIAD | TRIAD − best simple | TRIAD − original |
|---|---|---|---|---|---|
| Electronic | .312 (content + ItemKNN, prior) | .335 | **.341** | **+.029** | +.007 (n.s.) |
| Food | .460 (content + ItemKNN, prior) | .474 | **.488** | **+.028** | **+.014** |
| Clothing | .427 (content + ItemKNN, prior) | .439 | **.451** | **+.024** | +.012 (n.s.) |
| POG_DENSE | .241 (RP3β, prior) | .278 | **.280** | **+.039** | **+.002** |
| POG | .023 (label + ItemKNN + content, prior) | .031 | .031 | **+.007** | +.000 (n.s.) |
| Spotify_sparse | .209 (label + ItemKNN + content, prior) | .256 | .257 | **+.048** | +.000 (n.s.) |
| Spotify | .247 (label + ItemKNN + content, prior) | .282 | .282 | **+.036** | +.000 (n.s.) |

Bold differences are significant; n.s. = not significant. Code: `python -m analysis.tables results`.

## 3. Training-set size

Electronic, Food and Clothing with nested random subsets of 50, 100, 300 and 600 training bundles and with all of
them (one fixed permutation per dataset), seeds 0 to 2. The baselines are re-tuned on each subset.

![Training-set size](figures/design/training_size.png)

| Condition | Best simple | TRIAD untrained | Original TRIAD | TRIAD | TRIAD's chosen epochs |
|---|---|---|---|---|---|
| Electronic, 50 | .095 | .095 | .089 | .095 | 0, 0, 0 |
| Electronic, 100 | .117 | .116 | .106 | .116 | 0, 0, 0 |
| Electronic, 300 | .168 | .168 | .154 | .168 | 0, 0, 0 |
| Electronic, 600 | .214 | .209 | .210 | .215 | 9, 15, 15 |
| Electronic, all | .312 | .310 | .335 | **.341** | 5, 10, 5 |
| Food, 50 | .252 | .257 | .235 | .257 | 0, 0, 0 |
| Food, 100 | .276 | .281 | .250 | .281 | 0, 0, 0 |
| Food, 300 | .301 | .299 | .300 | .299 | 0, 0, 0 |
| Food, 600 | .355 | .349 | .358 | .371 | 8, 8, 8 |
| Food, all | .460 | .455 | .474 | **.488** | 3, 3, 7 |
| Clothing, 50 | .235 | .222 | .206 | *.207* | 15, 4, 6 |
| Clothing, 100 | .243 | .227 | .215 | *.229* | 30, 0, 5 |
| Clothing, 300 | .241 | .246 | .250 | **.273** | 33, 24, 17 |
| Clothing, 600 | .323 | .291 | .325 | **.340** | 9, 8, 12 |
| Clothing, all | .427 | .359 | .439 | **.451** | 7, 11, 4 |

Bold: TRIAD significantly ahead of the best simple baseline; italic: significantly behind (Holm over the 15 conditions).

- With 50 to 300 bundles on Electronic and Food, validation keeps the untrained model (epoch 0) in every seed, so
  TRIAD stays the simple scorer. With more bundles it moves away and pulls ahead.
- Over the 15 conditions, TRIAD is significantly ahead of the best simple baseline in 5 and behind in 2 (Clothing with
  50 and 100 bundles). The original TRIAD is ahead in none and behind in 4.
- Code: `python -m analysis.tables training_size` (output `results/training_size.json`).

## 4. Against neural baselines rerun on identical splits

Test NDCG@20, on identical splits and features, with each method run from its authors' code over 3 seeds
(TASK_AND_BASELINES.md §3 and §4):

| Dataset | CLHE | RaMen | TRIAD |
|---|---|---|---|
| Electronic | .259 | .256 | **.341** |
| Food | .271 | .265 | **.488** |
| Clothing | .196 | .190 | **.451** |
| POG_DENSE | .248 | | **.280** |
| POG | .0172 | | **.0308** |
| Spotify_sparse | .115 | | **.257** |

The paired tests were run against the original TRIAD (.335, .474, .439, .278, .0306, .256): all 18 had p = .002
(Holm over 20). TRIAD scores higher than the original on every row. RaMen was rerun on the three e-commerce datasets
only. The CLHE and RaMen runs are kept outside this repository.

## 5. Seed variance on the small datasets

With the default learning rates, the three small e-commerce sets (about 1,200 training bundles) were unstable across
seeds: over five seeds, test NDCG@20 on Food ranged from .425 to .488 (standard deviation .027). The model moved so far
in its first one or two epochs that validation could not tell a good early model from a bad one. Halving both
learning rates, chosen on validation over five seeds on the three sets, gave:

| Five seeds, mean ± sd | Food validation | Food test | Electronic validation | Clothing validation |
|---|---|---|---|---|
| default learning rates | .441 ± .009 | .473 ± .027 | .379 ± .008 | .525 ± .004 |
| both halved | .449 ± .005 | .489 ± .004 | .379 ± .002 | .526 ± .002 |

The halved rates are used on these three datasets; the others keep the defaults.
