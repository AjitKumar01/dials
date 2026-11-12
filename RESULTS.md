# Results

All numbers are for the frozen model in this repository (no second hop), seeds 0, 1 and 2. POG, POG_DENSE, Spotify_sparse and Spotify
were retrained on 2026-11-11 with the released item features only (BLIP image and text on POG, CLAP audio and text on
Spotify), the same files CLHE uses; the other datasets use TF-IDF of the item text, since nothing is released for them.

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
| Dunnhumby | 21,744 | 5,707 | .1490 | .1023 | .1490 | .1023 |
| POG_DENSE | 31,217 | 5,938 (5,547) | .3756 | .2802 | .4072 | .3080 |
| POG | 48,676 | 4,000 (3,995) | .0543 | .0313 | .0551 | .0320 |
| Spotify_sparse | 213,325 | 2,498 (2,497) | .1837 | .2562 | .1840 | .2565 |
| Spotify | 254,155 | 4,000 (3,995) | .1674 | .2824 | .1680 | .2831 |

**The earlier features.** Before the retrain, POG also used our own bge-small-zh embedding of the item titles, and
Spotify used the CLAP audio features without the CLAP text. With them, clean N was .0360 on POG (now .0313, −13%),
.2808 on POG_DENSE, .2546 on Spotify_sparse and .2816 on Spotify. Only POG moves by more than 1%: its titles are
Chinese, and BLIP's text encoder was trained on English.

## 2. Against the best simple baseline

For each dataset, the simple baselines are tuned on validation NDCG@20, and the best one on validation is compared:
popularity, co-occurrence, ItemKNN, RP3β, GF-CF, EASE, iALS, SGFCF, and a content+ItemKNN blend whose content
weight is tuned on validation. Holm is applied over all 16 tests.

| Dataset | Best simple baseline | Its R / N | DIALS R / N | Δ N [95% CI] | p (N) | p (R) |
|---|---|---|---|---|---:|---:|
| Electronic | content+ItemKNN blend | .4320 / .2986 | .4691 / .3346 | +.036 [.016, .057] | .002 | **.054** |
| Food | content+ItemKNN blend | .5442 / .4252 | .5862 / .4388 | +.014 [−.006, .034] | **.28** | .023 |
| Clothing | content+ItemKNN blend | .4329 / .3303 | .5338 / .3876 | +.057 [.035, .080] | .002 | .002 |
| Dunnhumby | content+ItemKNN blend | .1441 / .1044 | .1490 / .1023 | **−.002 [−.005, .001]** | **.28** | **.070** |
| POG_DENSE | RP3β | .3447 / .2407 | .3756 / .2802 | +.040 [.036, .043] | .002 | .002 |
| POG | EASE | .0295 / .0198 | .0543 / .0313 | +.012 [.009, .014] | .002 | .002 |
| Spotify_sparse | content+ItemKNN blend | .1166 / .1526 | .1837 / .2562 | +.104 [.095, .112] | .002 | .002 |
| Spotify | content+ItemKNN blend | .1324 / .2228 | .1674 / .2824 | +.060 [.054, .066] | .002 | .002 |

- **Separated on both metrics** on Clothing, POG_DENSE, POG and both Spotify sets.
- **Food:** recall is separated, NDCG is not.
- **Electronic:** NDCG is separated; recall misses the 0.05 level after Holm (p = .054).
- **Dunnhumby:** **not separated on either metric.** DIALS is slightly behind on NDCG (−.002) and slightly ahead on
  recall (+.005). Why is not yet established.
- **The blend on Spotify** now uses the released CLAP features too; it became the best simple baseline on
  Spotify_sparse (before, RP3β .1488).

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
| POG_DENSE | CLHE | .3486 / .2476 | .3756 / .2802 | +.033 [.030, .036] | .002 |
| POG | CLHE | .0259 / .0172 | .0543 / .0313 | +.014 [.012, .017] | .002 |
| Spotify_sparse | CLHE | .0971 / .1153 | .1837 / .2562 | +.141 [.131, .151] | .002 |
| Spotify | CLHE | 1 of 3 seeds | .1674 / .2824 | | |

On POG, POG_DENSE and Spotify, DIALS and CLHE now see exactly the same item features. Full Spotify: only one CLHE
seed finished within Kaggle's 12-hour limit (test N .1788 on all queries), so it is not tested.

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
| POG | RaMen | .0375 / .0226 | .0551 / .0320 | same split files |
| Spotify | RaMen | .1091 / .1882 | .1680 / .2831 | same split files |
| Spotify | CLHE | .1082 / .1806 | .1680 / .2831 | same split files |
| Electronic | RaMen | .8371 / .5373 | .4691 / .3346 | **no**: different, unreleased split, and its user–item graph contains the test bundles (section 3) |
| Food | RaMen | .8459 / .5242 | .5862 / .4388 | **no**: same reasons |

**Bundle-MLLM's protocol (HitRate@1).** Bundle-MLLM (KDD'25) asks one question per (test bundle, held-out item):
the held-out item and 9 items drawn uniformly from outside the bundle; a hit if the held-out item scores highest.
This follows its code (commit 7818c05) and its released split files. DIALS scores every question, 3 seeds × 5 draws;
the published numbers are from 2,048 sampled questions.

| | DIALS | Bundle-MLLM (published) | CLHE (published in Bundle-MLLM's paper) |
|---|---|---|---|
| POG | **.630** (seed sd .001) | .413 | .348 |
| POG_DENSE | **.781** (seed sd .002) | .668 | .637 |

The paper states an 8:1:1 split, but the only released files are 7:1:2 (14,000 / 2,000 / 4,000 on POG); these are
the files used here. Bundle-MLLM's checkpoints are released, so a rerun on exactly these questions is possible; it
has not been done.

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
| POG k=4 | 2,521 | **.488 / .396** | RP3β .452 / .360 | +.036 [.030, .042] | .139 / .098 | TIGER .213 / .157 |
| Spotify k=30 | 2,744 | **.625 / .482** | blend .540 / .402 | +.085 [.081, .090] | .282 / .176 | BundleNAT .153 / .090 |
| Spotify k=60 | 1,582 | **.643 / .495** | blend .560 / .413 | +.083 [.077, .089] | .285 / .178 | TIGER .129 / .076 |
| Spotify k=90 | 969 | **.651 / .501** | blend .561 / .412 | +.090 [.081, .099] | .287 / .177 | TIGER .123 / .070 |

- **Without a candidate pool,** taking the top-|targets| from the whole catalogue, DIALS F1 is .191, .101, .154 and
  .192. That is a much harder setting and is not comparable with DDBC's pool numbers.
- **Queries whose bundle also occurs in training:** 89 of the POG queries, and 4 / 1 / 0 of the Spotify ones. DDBC
  keeps them. Dropping them gives DIALS pool F1 .472 on POG; Spotify does not change.
- **CIs** here are a plain per-query bootstrap.

## 6. Ablation: the second graph hop

An earlier version had a second graph hop (ARCHITECTURE.md, section 15). Retrained with and without it, three seeds
each, clean test queries, Holm over 14 tests. These runs predate the feature change: on POG, POG_DENSE and both
Spotify sets they used the earlier features, so their "without" column matches the earlier numbers in section 1, not
the current ones.

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
