# Results

All numbers are for TRIAD as it is in this repository, seeds 0, 1 and 2.

- **Item features:** the released ones on POG (BLIP image and text) and Spotify (CLAP audio and text), the same files
  CLHE uses. On Electronic, Food and Clothing nothing is released, so the item titles are embedded with BLIP's text
  encoder (computed by us) and every method gets these vectors. Dunnhumby uses TF-IDF of the product text.
- **Metrics:** Recall@20 (R) and NDCG@20 (N), full-catalogue ranking with the seeds removed.
- **Clean queries:** on POG and Spotify, "clean" excludes test queries whose complete bundle also occurs in training
  or validation. The e-commerce splits have no such overlap, so all their queries are clean.
- **Statistics:** "Δ" is TRIAD minus the other method on clean queries, with a 95% paired cluster-bootstrap interval;
  clusters are identical bundles. "p" is a paired sign-flip test, Holm-corrected within each table.
- **Baselines:** the outputs and comparison scripts for the baselines are kept outside this repository; this
  repository holds only the model.

## 1. TRIAD

| Dataset | Items | Test queries (clean) | R clean | N clean | R all | N all | ms / query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Electronic | 3,499 | 346 | .4669 | .3345 | .4669 | .3345 | 0.18 |
| Food | 3,767 | 353 | .6178 | .4738 | .6178 | .4738 | 0.16 |
| Clothing | 4,487 | 371 | .5927 | .4388 | .5927 | .4388 | 0.17 |
| Dunnhumby | 21,744 | 5,707 | .1481 | .1012 | .1481 | .1012 | 0.57 |
| POG_DENSE | 31,217 | 5,938 (5,547) | .3742 | .2781 | .4060 | .3063 | 0.41 |
| POG | 48,676 | 4,000 (3,995) | .0524 | .0306 | .0532 | .0314 | 0.51 |
| Spotify_sparse | 213,325 | 2,498 (2,497) | .1837 | .2562 | .1840 | .2565 | 2.5 |
| Spotify | 254,155 | 4,000 (3,995) | .1674 | .2823 | .1679 | .2830 | 5.4 |

Inference time is the test pass on 4 CPU threads divided by the number of queries, after a one-time setup of the pair
counts, measured on 2026-11-17 on an otherwise lightly loaded laptop.

**The earlier version.** TRIAD was called DIALS while it also had a low-rank label-to-label part with cross-fitted
label factors. Retrained without it, test NDCG@20 changed by at most .004 on every dataset (ARCHITECTURE.md §15).

## 2. Against the best simple baseline

**Equal information.** On all seven datasets the catalogue is exactly the set of items in some training, validation
or test bundle, so every item with training frequency f = 0 is a validation- or test-only item (41–67% of e-commerce
test targets, 23% POG_DENSE, 59% POG, 29% Spotify_sparse, 14% Spotify). TRIAD's prior ρ learns a bonus for f = 0
from its leave-one-out training loss (ARCHITECTURE.md §14); frequency baselines rank such items last. So the five main
simple baselines (ItemKNN, RP3β, EASE, content+ItemKNN, label+ItemKNN+content) also get a frequency prior:
score' = maxabs(score) + λ·b(f), b = smoothed base rate of validation targets per frequency bin, λ ∈ {0, .01, .03, .1,
.3, 1, 3} chosen on validation with the blends' mixing weight; λ = 0 reproduces the published numbers (asserted).
"TRIAD w/o bonus" sets ρ of f = 0 items to ρ(log 2) at test time.
Code: bundlefuse `experiments/fair_prior/{baselines,compare,no_bonus}.py`; output `results/fair_prior/comparison.json`.

The pool is popularity, co-occurrence, ItemKNN, RP3β, GF-CF, EASE, iALS, SGFCF, a content+ItemKNN blend, four
label-aware methods, and the five main ones with the prior; the best is chosen on validation NDCG@20. Holm over the
14 tests of each family.

| Dataset | Best simple (validation) | Its R / N | TRIAD R / N | Δ N [95% CI] | p (N) | p (R) | w/o bonus R / N | Δ N w/o bonus | p |
|---|---|---|---|---|---:|---:|---|---|---:|
| Electronic | content+ItemKNN + prior | .4346 / .3122 | .4669 / .3345 | +.022 [.004, .042] | .116 | .116 | .4046 / .3063 | −.006 [−.025, .013] | 1.00 |
| Food | content+ItemKNN + prior | .5834 / .4601 | .6178 / .4738 | +.014 [−.003, .031] | .373 | **.026** | .5807 / .4446 | −.015 [−.031, .001] | .198 |
| Clothing | content+ItemKNN + prior | .5823 / .4273 | .5927 / .4388 | +.011 [−.005, .027] | .373 | .383 | .4850 / .3549 | **−.072 [−.094, −.052]** | .001 |
| POG_DENSE | RP3β (λ = 0) | .3447 / .2407 | .3742 / .2781 | **+.037 [.034, .041]** | .001 | .001 | .3743 / .2782 | **+.037** | .001 |
| POG | label+ItemKNN+content | .0363 / .0234 | .0524 / .0306 | **+.007 [.005, .009]** | .001 | .001 | .0484 / .0289 | **+.005** | .001 |
| Spotify_sparse | label+ItemKNN+content + prior | .1513 / .2085 | .1837 / .2562 | **+.048 [.043, .052]** | .001 | .001 | .1832 / .2553 | **+.047** | .001 |
| Spotify | label+ItemKNN+content | .1466 / .2469 | .1674 / .2823 | **+.035 [.031, .040]** | .001 | .001 | .1673 / .2822 | **+.035** | .001 |

- **Outfits and playlists:** TRIAD is ahead on both metrics, with or without its bonus.
- **E-commerce:** level. Without the prior the blend was .3024 / .4450 / .3593 and TRIAD looked ahead by .032 / .029 /
  .079; most of that was the bonus. Without the bonus TRIAD is behind, significantly on Clothing.
- **Without the label-aware methods** the best simple baseline is EASE on POG (.0198) and the blend on both Spotify
  sets (.1526, .2228), without the prior.
- **Dunnhumby** (grocery baskets, not part of the paper): against the content+ItemKNN blend without the prior
  (.1441 / .1044), TRIAD is −.003 [−.006, −.001] on NDCG and +.004 [−.001, .009] on recall; neither separated after Holm.

### 2a. Fresh splits and padded catalogues (Electronic, Food)

Does the bonus depend on the split or on how the catalogue is built? Fresh seeded splits (1, 2; 0 is published,
retrained by the same script and reproducing the numbers above exactly), and the published split with P = 1, 4, 16 × n0
extra items in no bundle (each copies the features and category of a random catalogue item). Learned bonus:
w_G(ρ(0) − ρ(log 2)) averaged over test queries; base rate: log[(share of f=0 targets / share of f=0 items) /
(same for f=1)] on test. Gain: TRIAD − TRIAD w/o bonus; vs simple: TRIAD − the better content blend with the prior
chosen on validation. NDCG@20, three seeds; bold: Holm p < .05 over all 48 tests.
Code: bundlefuse `experiments/catalogue_check/run.py`; output `results/catalogue_check/summary.json`.

| Dataset | Condition | n | n0 | Learned | Base rate | Gain | vs simple |
|---|---|---:|---:|---:|---:|---:|---:|
| Electronic | split 0 | 3,499 | 680 | +1.92 | +1.19 | **+.028** | +.022 |
| | split 1 | 3,499 | 658 | +1.70 | +1.43 | **+.042** | +.020 |
| | split 2 | 3,499 | 693 | +1.79 | +1.41 | **+.025** | +.025 |
| | pad 1 n0 | 4,179 | 1,360 | +1.05 | +0.50 | +.007 | +.014 |
| | pad 4 n0 | 6,219 | 3,400 | +0.35 | −0.42 | −.001 | +.007 |
| | pad 16 n0 | 14,379 | 11,560 | −0.63 | −1.64 | .000 | +.005 |
| Food | split 0 | 3,767 | 788 | +1.64 | +1.31 | **+.029** | +.014 |
| | split 1 | 3,767 | 797 | +1.43 | +1.35 | **+.017** | +.011 |
| | split 2 | 3,767 | 789 | +1.58 | +1.53 | **+.038** | +.019 |
| | pad 1 n0 | 4,555 | 1,576 | +0.87 | +0.62 | **+.013** | +.009 |
| | pad 4 n0 | 6,919 | 3,940 | −0.11 | −0.30 | −.001 | −.009 |
| | pad 16 n0 | 16,375 | 13,396 | −1.05 | −1.52 | −.002 | −.002 |

- The bonus and its gain do not depend on the split. The test base-rate value falls by log(1 + P/n0); the learned
  bonus follows it from training data alone (0.05–1.0 above it) and becomes a penalty at 16 n0; its gain goes to zero
  from 4 n0. TRIAD vs the best blend is never separated (R@20 at 16 n0: −.022 Electronic, −.026 Food, n.s.).
- The padding items are copies; the never-bundled products of the source data have no titles in the release.

## 3. Against neural baselines rerun on identical splits

RaMen (ECAI'25) and CLHE (WSDM'24) were run with their own code on exactly these splits, three seeds each, settings
chosen on validation, with the same item features as TRIAD.

- **On e-commerce,** their user–item graph and collaborative features are built from training bundles only, which is
  the information TRIAD has. RaMen's item–item graph builder is not released, so it was rebuilt from the paper.
- **CLHE on POG and Spotify** used its shipped preset, on split files whose SHA-256 matches. On full Spotify it ran at
  batch size 128 instead of the preset's 256, because 256 does not fit on a 16 GB GPU.

Holm is applied over 20 planned tests, including the two on full Spotify that could not be run.

| Dataset | Method | R / N | TRIAD R / N | Δ N [95% CI] | p (N and R) |
|---|---|---|---|---|---:|
| Electronic | RaMen | .345 / .256 | .4669 / .3345 | +.078 [.053, .104] | .002 |
| Electronic | CLHE | .362 / .259 | | +.076 [.049, .104] | .002 |
| Food | RaMen | .345 / .265 | .6178 / .4738 | +.209 [.172, .244] | .002 |
| Food | CLHE | .419 / .271 | | +.203 [.173, .233] | .002 |
| Clothing | RaMen | .240 / .190 | .5927 / .4388 | +.249 [.214, .282] | .002 |
| Clothing | CLHE | .325 / .196 | | +.242 [.213, .271] | .002 |
| POG_DENSE | CLHE | .3486 / .2476 | .3742 / .2781 | +.031 [.028, .034] | .002 |
| POG | CLHE | .0259 / .0172 | .0524 / .0306 | +.013 [.011, .016] | .002 |
| Spotify_sparse | CLHE | .0971 / .1153 | .1837 / .2562 | +.141 [.131, .151] | .002 |
| Spotify | CLHE | 1 of 3 seeds | .1674 / .2823 | | |

Full Spotify: only one CLHE seed finished within Kaggle's 12-hour limit (test N .1788 on all queries), so it is not
tested.

**Other user–item graphs on e-commerce.** BundleRec releases two user–item files with the bundles:

- `user_item.csv` holds the items of every bundle, test bundles included. A graph built from it links every test
  target to its seeds, against 26–38% for a graph built from training bundles only.
- `user_item_pretrain.csv` is a separate purchase log (1.95 M rows on Electronic, 325 k on Food, 749 k on Clothing)
  that shares only 2–3 (user, item) pairs with the bundles, so it does not contain the test bundles.

| Dataset | RaMen, training-only graph | RaMen, `user_item.csv` | RaMen, purchase log | CLHE, training-only graph | CLHE, purchase log |
|---|---|---|---|---|---|
| Electronic | .345 / .256 | .505 / .371 | .334 / .239 | .362 / .259 | .349 / .247 |
| Food | .345 / .265 | .490 / .344 | .338 / .259 | .419 / .271 | .360 / .270 |
| Clothing | .240 / .190 | .499 / .303 | .229 / .182 | .325 / .196 | .237 / .194 |

The purchase log does not help either model, so real purchase histories do not explain RaMen's published e-commerce
numbers; the leaky file does part of it. These runs are shown for reference and not tested.

## 4. Published numbers

These are copied from the papers, not rerun, and are for all test queries.

| Dataset | Published | R / N | TRIAD R / N (all queries) | Comparable? |
|---|---|---|---|---|
| POG | RaMen | .0375 / .0226 | .0532 / .0314 | same split files |
| Spotify | RaMen | .1091 / .1882 | .1679 / .2830 | same split files |
| Spotify | CLHE | .1082 / .1806 | .1679 / .2830 | same split files |
| Electronic | RaMen | .8371 / .5373 | .4669 / .3345 | **no**: different, unreleased split, and its user–item graph contains the test bundles (section 3) |
| Food | RaMen | .8459 / .5242 | .6178 / .4738 | **no**: same reasons |

**Bundle-MLLM's protocol (HitRate@1).** Bundle-MLLM (KDD'25) asks one question per (test bundle, held-out item):
the held-out item and 9 items drawn uniformly from outside the bundle; a hit if the held-out item scores highest.
This follows its code (commit 7818c05) and its released split files. TRIAD scores every question, 3 seeds × 5 draws;
the published numbers are from 2,048 sampled questions.

| | TRIAD | Bundle-MLLM (published) | CLHE (published in Bundle-MLLM's paper) |
|---|---|---|---|
| POG | **.624** (seed sd .001) | .413 | .348 |
| POG_DENSE | **.771** (seed sd .001) | .668 | .637 |

DIALS, the earlier version, scored .630 and .781 here. The paper states an 8:1:1 split, but the only released files
are 7:1:2 (14,000 / 2,000 / 4,000 on POG); these are the files used here. Bundle-MLLM's checkpoints are released, so
a rerun on exactly these questions is possible; it has not been done.

## 5. DDBC's protocol

DDBC (ICLR'26) uses our POG_DENSE and Spotify training files, and its test queries are windows of our test bundles,
with the same IDs.

- **Protocol:** each query's window is split 1:1 into seeds and targets. The candidate pool is the targets plus
  99·|targets| random items outside the window, and the prediction is the top-|targets| of the pool. That makes
  precision, recall and F1 equal; Jaccard is h/(2|targets| − h).
- **Setup:** 5 pool draws per query, the same draws for every method, single attempt as in DDBC's Table 1. DDBC's
  numbers are published, not rerun.

| Case | Queries | TRIAD F1 / Jaccard | Best simple baseline F1 / Jaccard | Δ F1 [95% CI] | DDBC (published) | Best baseline in DDBC's paper |
|---|---:|---|---|---|---|---|
| POG k=4 | 2,521 | **.483 / .392** | RP3β .452 / .360 | +.031 [.025, .037] | .139 / .098 | TIGER .213 / .157 |
| Spotify k=30 | 2,744 | **.625 / .482** | blend .540 / .402 | +.085 [.081, .090] | .282 / .176 | BundleNAT .153 / .090 |
| Spotify k=60 | 1,582 | **.643 / .495** | blend .560 / .413 | +.083 [.077, .089] | .285 / .178 | TIGER .129 / .076 |
| Spotify k=90 | 969 | **.651 / .501** | blend .561 / .412 | +.090 [.081, .099] | .287 / .177 | TIGER .123 / .070 |

- **Without a candidate pool,** taking the top-|targets| from the whole catalogue, TRIAD F1 is .189, .101, .154 and
  .192. That is a much harder setting and is not comparable with DDBC's pool numbers.
- **Queries whose bundle also occurs in training:** 89 of the POG queries, and 4 / 1 / 0 of the Spotify ones. DDBC
  keeps them. Dropping them gives TRIAD pool F1 .468 on POG; Spotify does not change.
- **CIs** here are a plain per-query bootstrap.
- **DIALS,** the earlier version, scored .488 / .396 on POG and the same as TRIAD on Spotify.

## 6. Training-set size

TRIAD is retrained and ten simple baselines (all but GF-CF, SGFCF and iALS) are retuned on nested random subsets of
the training bundles, with the same validation and test queries; the five main ones also get the frequency prior
(§2). Δ N is TRIAD minus the best simple baseline chosen on validation for that subset; Holm within each family over
both metrics (e-commerce 30 tests, music 28, fashion 20). Code: bundlefuse `experiments/fair_prior/budget_compare.py`;
output `results/fair_prior/budget_comparison.json`.

| Training bundles | Electronic | Food | Clothing |
|---|---|---|---|
| all (1,211–1,294) | +.022 [.004, .042] | +.014 [−.003, .031] | +.011 [−.005, .027] |
| 600 | −.004 [−.023, .014] | +.003 [−.013, .018] | +.002 [−.015, .017] |
| 300 | −.014 [−.030, .001] | −.001 [−.012, .010] | +.009 [−.009, .027] |
| 100 | −.011 [−.026, .002] | **−.027 [−.041, −.014]** | **−.028 [−.044, −.012]** |
| 50 | −.006 [−.013, .001] | **−.017 [−.025, −.010]** | **−.029 [−.043, −.016]** |

| Training bundles | Spotify_sparse | Spotify |
|---|---|---|
| all | **+.048** | **+.035** |
| 3,000 | **+.029** | **+.030** |
| 1,000 | **+.022** | **+.025** |
| 600 | **+.019** | **+.022** |
| 300 | **+.016** | **+.018** |
| 100 | **+.010** | **+.020** |
| 50 | **+.004** | **+.012** |

| Training data | POG_DENSE | POG |
|---|---|---|
| all | **+.037** | **+.007** |
| 10% | **+.025** | +.002 (p .06) |
| 1% | **+.017** | **−.001** |
| users' purchase histories, no outfits | **−.006** | **−.003** |
| same, overlapping histories removed | **−.006** | **−.001** |

Bold: Holm-corrected p < .05. On small e-commerce subsets the chosen baseline is label+ItemKNN+content with the prior.
Before the prior was added, TRIAD looked ahead with all e-commerce bundles and on Clothing with 600 and 300.

**Overfitting, not too few steps.** On the e-commerce subsets of 300 and fewer, repeating the bundles so that each
epoch has as many gradient steps as on the full data changes test NDCG@20 by at most .02, and validation NDCG peaks
after one to five epochs (mostly the first) while the training loss keeps falling (Food, 50 bundles, seed 0: loss
5.86 to 0.44, validation NDCG .213 to .133). Training improves on the untrained model in all nine cases.

| Electronic / Food / Clothing | 300 | 100 | 50 |
|---|---|---|---|
| untrained TRIAD | .123 / .257 / .209 | .100 / .243 / .199 | .088 / .232 / .196 |
| trained TRIAD | .154 / .300 / .250 | .106 / .250 / .215 | .089 / .235 / .206 |
| best simple baseline (with prior) | .168 / .301 / .241 | .117 / .276 / .243 | .095 / .252 / .235 |

## 7. Ablations

TRIAD retrained with one part removed at a time, same configuration and seeds 0, 1, 2; full Spotify is left out (one
training run takes 1--1.5 h of CPU time). Each cell: full model minus ablated model, NDCG@20 [95% CI] / Recall@20,
clean test queries, three-seed means per side. Bold: Holm-corrected p < .05 over the 12 tests of each row (6 datasets x
2 metrics). "training (initial parameters)" is the model at initialization (delta and rho set to zero), not trained.
POG_DENSE and Spotify_sparse ran on Kaggle CPU (experiments/triad_ablations/kaggle.py, same model and training code);
three Spotify_sparse runs that reached the 11.3-hour kernel limit were resumed from their saved epoch.

| Removed part | Electronic | Food | Clothing | POG_DENSE | POG | Spotify_sparse |
|---|---:|---:|---:|---:|---:|---:|
| *full model N / R* | .335 / .467 | .474 / .618 | .439 / .593 | .278 / .374 | .031 / .052 | .256 / .184 |
| leave-one-out statistics | **+.054** [+.033, +.075] / **+.114** | **+.140** [+.110, +.170] / **+.204** | **+.165** [+.138, +.192] / **+.238** | **+.073** [+.068, +.078] / **+.108** | **+.010** [+.008, +.012] / **+.020** | **+.112** [+.104, +.121] / **+.076** |
| gate (all weights 1) | **+.065** [+.049, +.082] / **+.067** | **+.036** [+.021, +.051] / **+.025** | **+.031** [+.016, +.047] / +.016 | **+.002** [+.001, +.004] / +.002 | +.001 [−.000, +.001] / +.002 | **+.008** [+.006, +.009] / **+.007** |
| content term | **+.046** [+.026, +.068] / **+.071** | **+.125** [+.095, +.156] / **+.160** | **+.153** [+.122, +.183] / **+.221** | **+.031** [+.027, +.034] / **+.022** | **+.010** [+.008, +.012] / **+.021** | **+.016** [+.014, +.019] / **+.010** |
| label terms | +.002 [−.001, +.006] / −.003 | +.002 [−.005, +.009] / +.003 | +.009 [+.001, +.018] / +.005 | **+.003** [+.002, +.005] / **+.003** | **+.002** [+.001, +.002] / **+.003** | **+.088** [+.080, +.095] / **+.052** |
| graph term | **+.185** [+.154, +.219] / **+.201** | **+.172** [+.141, +.203] / **+.163** | **+.152** [+.125, +.180] / **+.151** | **+.014** [+.012, +.017] / **+.020** | **+.019** [+.016, +.022] / **+.030** | **+.059** [+.054, +.064] / **+.051** |
| training (initial parameters) | **+.159** [+.130, +.190] / **+.199** | **+.129** [+.105, +.156] / **+.148** | **+.179** [+.149, +.209] / **+.194** | **+.177** [+.169, +.184] / **+.228** | **+.030** [+.026, +.032] / **+.051** | **+.117** [+.109, +.126] / **+.074** |

- Co-occurrence (graph term), leave-one-out training and content help on all six datasets; the graph term most on
  e-commerce (.15--.19).
- Without leave-one-out a training bundle's own pairs stay in its counts and the model learns to read the answer from
  them: training loss 2.6 against 6.5 on POG_DENSE. There, training diverged at epoch 13 on seeds 1 and 2; the best
  validation epoch is used.
- The gate matters most on e-commerce (.03--.07); elsewhere it adds at most .008.
- The label terms matter on playlists (+.088 on Spotify_sparse: a track by a seed's artist); elsewhere they add at most
  .009, significant only on POG and POG_DENSE (.002--.003).

Results: bundlefuse/results/triad_ablations/summary.json (`experiments.triad_ablations.run test`).

## 8. Ablation of the earlier second graph hop

An earlier version had a second graph hop (ARCHITECTURE.md, section 15). Retrained with and without it, three seeds
each, clean test queries, Holm over 14 tests. These runs predate both the feature change on POG and Spotify and the
BLIP features on e-commerce, so their "without" column is not the current model.

| Dataset | With second hop N | Without N | Δ N (with − without) [95% CI] | p (N) |
|---|---:|---:|---|---:|
| Electronic | .3356 | .3346 | +.0010 [.0001, .0023] | .65 |
| Food | .4376 | .4388 | −.0013 [−.0040, .0013] | 1.0 |
| Clothing | .3883 | .3876 | +.0006 [.0001, .0015] | .65 |
| POG_DENSE | .2805 | .2808 | −.0002 [−.0020, .0015] | 1.0 |
| POG | .0353 | .0360 | −.0007 [−.0012, −.0002] | .04 |
| Spotify_sparse | .2548 | .2546 | +.0002 [−.0002, .0007] | 1.0 |
| Spotify | **.2865** | .2816 | **+.0049 [.0039, .0060]** | .001 |

Removing the second hop costs .005 NDCG on full Spotify and nothing elsewhere; on POG it slightly helps. It also
removes six rank-64 SVDs of the item graph, about 2.5 min and 0.8 GB on Spotify.

## 9. Size

Trained parameters:

| Dataset | TRIAD | CLHE | RaMen |
|---|---:|---:|---:|
| Electronic | 41,368 | 3.70 M | 1.31 M |
| Food | 41,368 | 3.73 M | 1.38 M |
| Clothing | 41,368 | 3.83 M | 1.56 M |
| POG_DENSE | 41,368 | 7.25 M | 11.6 M |
| POG | 41,368 | 9.48 M | 16.1 M |
| Spotify_sparse | 41,466 | 28.98 M | 56.5 M |
| Spotify | 41,466 | | 67.0 M |

- **CLHE:** counted from the trained models (on e-commerce, the runs with BLIP's 768-d title vectors).
- **RaMen:** counted from its code at its published settings; it learns 256 parameters per item.
- **TRIAD** has no per-item parameters; two label levels (artist, album) add 98 parameters.
