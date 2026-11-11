# Results

> **Status (2026-11-11):** the POG, POG_DENSE, Spotify_sparse and Spotify numbers here come from an earlier feature
> set: on POG it added our own bge-small-zh embedding of the item titles, and on Spotify it used the CLAP audio
> features without the CLAP text. The code now uses only the released features (BLIP image and text on POG, CLAP audio
> and text on Spotify), the same files CLHE uses. These four datasets are being retrained, and their rows will be
> replaced. Electronic, Food, Clothing and Dunnhumby are unaffected.

All numbers are for the frozen model in this repository (no second hop), seeds 0, 1 and 2, as of 2026-11-09.

- **Metrics:** Recall@20 (R) and NDCG@20 (N), full-catalogue ranking with the seeds removed.
- **Clean queries:** on POG and Spotify, "clean" excludes test queries whose complete bundle also occurs in training
  or validation. The e-commerce splits have no such overlap, so all their queries are clean.
- **Statistics:** "Δ" is DIALS minus the other method on clean queries, with a 95% paired cluster-bootstrap interval;
  clusters are identical bundles. "p" is a paired sign-flip test, Holm-corrected within each table.
- **Baselines:** the outputs and comparison scripts for the baselines are kept outside this repository; this
  repository holds only the model.

## 1. DIALS

| Dataset | Items | Test queries (clean) | R clean | N clean | R all | N all |
|---|---:|---:|---:|---:|---:|---:|
| Electronic | 3,499 | 346 | .4691 | .3346 | .4691 | .3346 |
| Food | 3,767 | 353 | .5862 | .4388 | .5862 | .4388 |
| Clothing | 4,487 | 371 | .5338 | .3876 | .5338 | .3876 |
| POG_DENSE | 31,217 | 5,938 (5,547) | .3779 | .2808 | .4091 | .3093 |
| POG | 48,676 | 4,000 (3,995) | .0606 | .0360 | .0613 | .0367 |
| Spotify_sparse | 213,325 | 2,498 (2,497) | .1816 | .2546 | .1819 | .2549 |
| Spotify | 254,155 | 4,000 (3,995) | .1660 | .2816 | .1665 | .2822 |

## 2. Against the best simple baseline

For each dataset, the simple baselines are tuned on validation NDCG@20, and the best one on validation is compared:
popularity, co-occurrence, ItemKNN, RP3β, GF-CF, EASE, iALS, SGFCF, and a content+ItemKNN blend whose content
weight is tuned on validation. Holm is applied over all 14 tests.

| Dataset | Best simple baseline | Its R / N | DIALS R / N | Δ N [95% CI] | p (N) | p (R) |
|---|---|---|---|---|---:|---:|
| Electronic | content+ItemKNN blend | .4320 / .2986 | .4691 / .3346 | +.036 [.016, .057] | .002 | .027 |
| Food | content+ItemKNN blend | .5442 / .4252 | .5862 / .4388 | +.014 [−.006, .034] | **.21** | .014 |
| Clothing | content+ItemKNN blend | .4329 / .3303 | .5338 / .3876 | +.057 [.035, .080] | .001 | .001 |
| POG_DENSE | RP3β | .3447 / .2407 | .3779 / .2808 | +.040 [.036, .044] | .001 | .001 |
| POG | EASE | .0295 / .0198 | .0606 / .0360 | +.016 [.014, .019] | .001 | .001 |
| Spotify_sparse | RP3β | .1180 / .1488 | .1816 / .2546 | +.106 [.097, .115] | .001 | .001 |
| Spotify | content+ItemKNN blend | .1313 / .2209 | .1660 / .2816 | +.061 [.055, .067] | .001 | .001 |

DIALS is ahead everywhere except NDCG on Food, where the content+ItemKNN blend is **not separated** from it. Recall
on Food is separated.

## 3. Against neural baselines rerun on identical splits

RaMen (ECAI'25) and CLHE (WSDM'24) were run with their own code on exactly these splits, three seeds each, settings
chosen on validation.

- **On e-commerce, "clean"** means their user–item graph and collaborative features are built from training bundles
  only, which is the information DIALS has.
- **CLHE on POG and Spotify** used its shipped preset, on split files whose SHA-256 matches. On full Spotify it ran at
  batch size 128 instead of the preset's 256, because 256 does not fit on a 16 GB GPU.

Holm is applied over 20 planned tests, including the two still pending.

| Dataset | Method | R / N | DIALS R / N | Δ N [95% CI] | p (N and R) |
|---|---|---|---|---|---:|
| Electronic | RaMen | .3440 / .2537 | .4691 / .3346 | +.081 [.057, .107] | .002 |
| Electronic | CLHE | .3618 / .2600 | | +.075 [.049, .103] | .002 |
| Food | RaMen | .3523 / .2744 | .5862 / .4388 | +.164 [.133, .197] | .002 |
| Food | CLHE | .3701 / .2721 | | +.167 [.136, .200] | .002 |
| Clothing | RaMen | .2378 / .1905 | .5338 / .3876 | +.197 [.168, .229] | .002 |
| Clothing | CLHE | .2414 / .1859 | | +.202 [.173, .232] | .002 |
| POG_DENSE | CLHE | .3486 / .2476 | .3779 / .2808 | +.033 [.030, .036] | .002 |
| POG | CLHE | .0259 / .0172 | .0606 / .0360 | +.019 [.016, .022] | .002 |
| Spotify_sparse | CLHE | .0971 / .1153 | .1816 / .2546 | +.139 [.129, .149] | .002 |
| Spotify | CLHE | running | .1660 / .2816 | | |

**The leaky user–item graph.** RaMen's released user–item file for Electronic and Food contains every bundle's
items, test bundles included. A graph built from it links every test target to its seeds, against 26–38% for a graph
built from training bundles only. Run with that graph:

| Dataset | RaMen R / N | CLHE R / N |
|---|---|---|
| Electronic | .4985 / .3529 | .5193 / .3690 |
| Food | .4893 / .3501 | .4834 / .3597 |
| Clothing | .4826 / .3007 | .3015 / .2411 |

These runs are shown for reference and not tested.

## 4. Published numbers

These are copied from the papers, not rerun, and are for all test queries.

| Dataset | Published | R / N | DIALS R / N (all queries) | Comparable? |
|---|---|---|---|---|
| POG | RaMen | .0375 / .0226 | .0613 / .0367 | same split files |
| Spotify | RaMen | .1091 / .1882 | .1665 / .2822 | same split files |
| Spotify | CLHE | .1082 / .1806 | .1665 / .2822 | same split files |
| Electronic | RaMen | .8371 / .5373 | .4691 / .3346 | **no**: different, unreleased split, and its user–item graph contains the test bundles (section 3) |
| Food | RaMen | .8459 / .5242 | .5862 / .4388 | **no**: same reasons |

## 5. DDBC's protocol

DDBC (ICLR'26) uses our POG_DENSE and Spotify training files, and its test queries are windows of our test bundles,
with the same IDs.

- **Protocol:** each query's window is split 1:1 into seeds and targets. The candidate pool is the targets plus
  99·|targets| random items outside the window, and the prediction is the top-|targets| of the pool. That makes
  precision, recall and F1 equal; Jaccard is h/(2|targets| − h).
- **Setup:** 5 pool draws per query, the same draws for every method, single attempt as in DDBC's Table 1. DDBC's
  numbers are published, not rerun.

| Case | Queries | DIALS F1 / Jaccard | Best simple baseline F1 / Jaccard | Δ F1 [95% CI] | DDBC (published) | Best baseline in DDBC's paper |
|---|---:|---|---|---|---|---|
| POG k=4 | 2,521 | **.491 / .400** | RP3β .452 / .360 | +.039 [.033, .045] | .139 / .098 | TIGER .213 / .157 |
| Spotify k=30 | 2,744 | **.622 / .479** | blend .540 / .402 | +.082 [.078, .087] | .282 / .176 | BundleNAT .153 / .090 |
| Spotify k=60 | 1,582 | **.641 / .493** | blend .560 / .413 | +.081 [.076, .088] | .285 / .178 | TIGER .129 / .076 |
| Spotify k=90 | 969 | **.649 / .500** | blend .561 / .412 | +.088 [.079, .097] | .287 / .177 | TIGER .123 / .070 |

- **Without a candidate pool,** taking the top-|targets| from the whole catalogue, DIALS F1 is .192, .101, .153 and
  .193. That is a much harder setting and is not comparable with DDBC's pool numbers.
- **Queries whose bundle also occurs in training:** 89 of the POG queries, and 4 / 1 / 0 of the Spotify ones. DDBC
  keeps them. Dropping them gives DIALS pool F1 .475 on POG; Spotify does not change.
- **CIs** here are a plain per-query bootstrap.

## 6. Ablation: the second graph hop

An earlier version had a second graph hop (ARCHITECTURE.md, section 15). Retrained with and without it, three seeds
each, clean test queries, Holm over 14 tests:

| Dataset | With second hop N | Without (this model) N | Δ N (with − without) [95% CI] | p (N) |
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

## 7. Size

Trained parameters:

| Dataset | DIALS | CLHE | RaMen |
|---|---:|---:|---:|
| Electronic | 41,624 | 748,864 | 1.31 M |
| Food | 41,624 | 783,168 | 1.38 M |
| Clothing | 41,624 | 875,328 | 1.56 M |
| POG_DENSE | 41,624 | 7.25 M | 11.6 M |
| POG | 41,624 | 9.48 M | 16.1 M |
| Spotify_sparse | 41,978 | 28.98 M | 56.5 M |
| Spotify | 41,978 | | 67.0 M |

- **CLHE:** counted from the trained models.
- **RaMen:** counted from its code at its published settings; it learns 256 parameters per item.
- **DIALS** has no per-item parameters. Its inference cost per query is 0.1 ms on e-commerce, 0.5–0.6 ms on POG and
  3–4 ms on Spotify, on 4 CPU threads.
