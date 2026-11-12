# Future work (deferred, not started)

Ideas agreed to postpone. None of them is implemented, and the frozen model is unchanged. SCALING.md holds the
measurements and the plan for large catalogues; this file lists every deferred item, with the training speed-up
analysis in full.

## 1. Faster training

### Why a 41.6 k-parameter model has slower epochs than 16 M-parameter RaMen

Parameter count is not compute. DIALS applies its shared maps to every item at every step, because the loss is a
softmax over the whole catalogue. RaMen's parameters are mostly embedding rows that a batch only looks up. It
also normalizes over the whole catalogue, but at batch 1024: on POG that is 14 steps per epoch for RaMen and 437 for
DIALS (batch 32).

Measured on POG (4 CPU threads, Apple M5 Pro, `bundlefuse/results/timing`):

| | Per epoch | Epochs | Training | Inference |
|---|---|---|---|---|
| DIALS | 26–32 s | 26–31 (early stopping) | 12–15 min | 0.64–0.70 ms/query |
| RaMen | 21 s | 200 (its config, not trained by us) | 71 min | 0.69 |
| CLHE | 102 s | 50 (our runs) | 85 min | 0.53 |

### Where a training step goes

One POG step, forward and backward, about 56 ms. Each part was timed on its own, so the parts overlap.

| Part | ms | Kind of work |
|---|---|---|
| Content term | 34 | dense over all $n$ items: encoder (9.5), keys (about 5), the $32\times n$ product, the leave-one-out loop |
| Label term | 12 | dense $32\times n$ broadcast and gather |
| Loss | 12 | logsumexp over all $n$ items; masks built in a Python loop |
| Graph term and prior | 5 | sparse: the seeds' rows of the CSR co-occurrence arrays |
| Gate | 0.3 | |

The sparse $n\times n$ co-occurrence matrix is about 9% of the step. The cost is the dense work over the catalogue.

### Exact rewrites, same model (estimated gain about 2×)

1. Map the query instead of the catalogue: $\Phi K^\top=(\Phi B_z)Z^\top+(\Phi B_h)H^\top$. Costs $q\,d\,n$ instead of
   $n\,d\,r$.
2. Score the low-rank label part per label ($q\times L$), then index by $g(c)$.
3. Score the shared-label diagonal only on items whose label occurs in the seeds, through an inverted index label → items.
4. Vectorize the leave-one-out corrections and the loss masks.
5. Precompute `history()`, which is a fixed buffer.

Acceptance: outputs match the frozen model, within a tolerance, on every dataset; then time an epoch on POG and Spotify.

### Removing the dependence on n (changes the objective)

Sampled softmax with a logQ correction (4–16 k popularity-sampled negatives), or a cache of stale keys refreshed every
few hundred steps. Acceptance: three Spotify seeds reproduce the frozen clean NDCG@20 (.2824) within noise.

### Memory, and where hashing helps

DIALS has no per-item parameters, so hashed embedding tables do not apply. Hashing helps when building the
co-occurrence counts from millions of baskets: a count-min sketch while counting, then the top $k$ partners per item,
stored as int32 indices and int16 counts. The features, histories and keys can be stored in float16 or product-quantized.

### What the small parameter count still buys

- A model whose size does not grow with the catalogue.
- New items scored from their features with no retraining.
- Fewer epochs to converge.

## 2. Large catalogues (millions of SKUs)

See SCALING.md: candidates, then exact re-ranking; sampled-softmax training; buffer memory; four validation experiments.

## 3. Personalization on dunnhumby

Add a household repurchase term $w_U\log(1+n_{u,c})$, where $n_{u,c}$ is household $u$'s purchase count of $c$ in its
training baskets.
- Give it its own gate weight.
- During training, subtract the training basket from its household's counts (leave-one-out).
- `data.load_baskets` would need to keep the household ID.

Earlier internal tests of this signal, with a different scorer, gave +37–44% on the same baskets.

For a fair comparison, give the baselines the same signal ("baselines + repurchase"). Report it as a separate,
personalized task: POG and Spotify have no owner per bundle, and curated bundles (Electronic, Food, Clothing) showed
no gain.

## 4. Item features: matched and modern encoders

The paper should report two feature settings, kept separate. Readers want both a fair comparison of methods and the
best result achievable in 2026.

**Matched features (the method comparison).**
- POG and Spotify use only the released features. This was applied on 2026-11-11, and the retrained results are in RESULTS.md:
  - POG and POG_DENSE: BLIP image and text;
  - Spotify and Spotify_sparse: CLAP audio and text.
- Electronic, Food, Clothing and Dunnhumby have no released features. The plan is BLIP text embeddings computed by us,
  labelled as ours, in place of TF-IDF.
  - Not started; it needs the go-ahead.
  - The TF-IDF results are reported alongside.
- Published numbers (RaMen, CLHE, Bundle-MLLM, DDBC) belong to this setting only.

What was checked on 2026-11-11:
- **The release's BLIP pipeline cannot be reproduced exactly.** Neither `Salesforce/blip-itm-base-coco` nor `-flickr`
  (text-encoder CLS or mean) matches the released POG text vectors. Same-item cosine is 0.35–0.38, against 0.29–0.35
  for different items.
- CLHE released no extraction script. So our BLIP features are the same family, not the same features.
- **Images are not available for these datasets.** Only 451 of 3,499 Electronic items have an image URL, and those
  tested did not resolve. Food, Clothing and Dunnhumby have none.

**Modern encoder (the state of the art).**
- **Encoder:** one current multilingual text encoder for all 8 datasets, such as Qwen3-Embedding-0.6B, BGE-M3 or the
  SigLIP 2 text tower. Check the current Hugging Face listings before choosing.
- **Other modalities:** keep the released image and audio features. POG images (URLs only) and Spotify audio (track IDs
  only) cannot be re-encoded.
- **Same features for every method:** DIALS, the content plus ItemKNN blend, CLHE and RaMen.

What this tests:
- **Is the gain the model's or the features'?** If DIALS's margin holds in both settings, the gain is the model's.
- **POG most of all.** Its titles are Chinese and BLIP's text encoder is English-trained, so the released text features
  are likely weak there. A multilingual encoder may help never-seen items most (59% of POG targets).
  Measured: replacing our Chinese bge-small-zh title embeddings with the released BLIP features cost POG 13%
  of its test NDCG@20 (.0360 to .0313); POG_DENSE and Spotify moved by under 1%.
- **Does a stronger signal survive DIALS's fixed SVD to 128 dimensions?** If not, that is a design weakness.
- **Do feature-heavy baselines catch up?** CLHE's MLP sees the full-dimensional features, so it may gain more than DIALS.
  Report it either way.

Cost: text embedding is minutes per dataset on CPU (Spotify has 254k tracks). Then retrain DIALS on all 8 datasets
(hours for Spotify), and rerun the blend, CLHE and RaMen (hours, on Kaggle for CLHE on POG and Spotify).

## 5. Open runs

- **CLHE on Spotify:** 1 of 3 seeds finished on Kaggle (test N@20 .1788, batch 128); the 12 h limit stopped the
  rest. Either report it with 1 seed or relaunch one kernel per seed (about 6.5 GPU-hours each).
- **Retrained ablations:** stopped on 2026-11-09. Finished: POG with the diagonal removed and with low-rank removed.
  The interrupted runs keep checkpoints and resume.
- **Dunnhumby:**
  - the extended baseline suite (blend, iALS, SGFCF test scores) was interrupted;
  - the significance test against the best simple baseline has not been run.
- **Inference timing on other datasets:** POG is done on one machine; the others are not.
