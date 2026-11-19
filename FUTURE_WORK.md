# Future work (deferred, not started)

Ideas agreed to postpone. None of them is implemented, and the frozen model is unchanged. SCALING.md holds the
measurements and the plan for large catalogues; this file lists every deferred item, with the training speed-up
analysis in full.

## 1. Faster training

### Why a 41.4 k-parameter model has slower epochs than 16 M-parameter RaMen

Parameter count is not compute. TRIAD applies its shared maps to every item at every step, because the loss is a
softmax over the whole catalogue. RaMen's parameters are mostly embedding rows that a batch only looks up. It
also normalizes over the whole catalogue, but at batch 1024: on POG that is 14 steps per epoch for RaMen and 437 for
TRIAD (batch 32).

Measured on POG (4 CPU threads, Apple M5 Pro, `bundlefuse/results/timing`), with DIALS, the previous version, which
also computed a low-rank label part; TRIAD drops that part, so its times can only be lower:

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
2. Score the shared-label term only on items whose label occurs in the seeds, through an inverted index label → items.
3. Vectorize the leave-one-out corrections and the loss masks.
4. Precompute `history()`, which is a fixed buffer.

Acceptance: outputs match the frozen model, within a tolerance, on every dataset; then time an epoch on POG and Spotify.

### Removing the dependence on n (changes the objective)

Sampled softmax with a logQ correction (4–16 k popularity-sampled negatives), or a cache of stale keys refreshed every
few hundred steps. Acceptance: three Spotify seeds reproduce the frozen clean NDCG@20 (.2824) within noise.

### Memory, and where hashing helps

TRIAD has no per-item parameters, so hashed embedding tables do not apply. Hashing helps when building the
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
- **Same features for every method:** TRIAD, the content plus ItemKNN blend, CLHE and RaMen.

What this tests:
- **Is the gain the model's or the features'?** If TRIAD's margin holds in both settings, the gain is the model's.
- **POG most of all.** Its titles are Chinese and BLIP's text encoder is English-trained, so the released text features
  are likely weak there. A multilingual encoder may help never-seen items most (59% of POG targets).
  Measured: replacing our Chinese bge-small-zh title embeddings with the released BLIP features cost POG 13%
  of its test NDCG@20 (.0360 to .0313); POG_DENSE and Spotify moved by under 1%.
- **Does a stronger signal survive TRIAD's fixed SVD to 128 dimensions?** If not, that is a design weakness.
- **Do feature-heavy baselines catch up?** CLHE's MLP sees the full-dimensional features, so it may gain more than TRIAD.
  Report it either way.

Cost: text embedding is minutes per dataset on CPU (Spotify has 254k tracks). Then retrain TRIAD on all 8 datasets
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

## 6. Deployment: adding items and bundles without retraining

**Goal.** A new item, or a new bundle, should be usable as soon as it arrives, with the learned parameters left as they
are. TRIAD has no per-item parameters, so this only means appending rows to its fixed tables.

**What a new item c★ needs** (a title, perhaps an image, a category; no bundles yet):
- its feature row: BLIP on the title, row-normalized, projected with the *saved* SVD basis, z★ = x★ V₁₂₈ᵀ, appended to X;
- zero bundle statistics: N★· = N·★ = 0, f★ = deg★ = 0, h★ = 0;
- its label: category g★, whose size goes up by one (a new category starts at size 1).

Scored as a candidate it gets the full content term (key = B_z z★), the label term if a seed shares g★, and the graph
term's constant ρ(0) for never-seen items. As a seed it adds its content query and its label, nothing to the graph
term, and it raises the gate's "share of seeds never seen", which moves weight towards content and labels.

**What a new bundle b needs** (indicator vector y): N += y yᵀ − diag(y), f += y, deg += (|b| − 1) y, and the partner
histories H of its |b| items recomputed. Only the |b|² pairs inside the bundle change.

Example: an Electronic bundle {charger, cable, new power bank}. Before it arrives, the power bank is scored by its
title and category only. After it, N gains the three pairs (charger, power bank), (cable, power bank), (charger, cable),
and the next query that contains the charger also finds the power bank through the graph term.

**How well new items do today** (test Recall@20 on targets never seen in training, BLIP, frozen model): Electronic
.288, Food .524, Clothing .503, Spotify_sparse .131, POG_DENSE .002, POG .021. Content carries new items on retail and
same-artist matches on Spotify; on POG, content alone does not find the right item among 31–49 k.

**To do:**
1. **Save the SVD basis.** triad/data.py refits the SVD on the whole catalogue at every load, so adding one item would
   shift every item's 128-d features and the trained content maps would no longer fit them. Save V₁₂₈ (and the
   normalization) with the model, and only transform new items.
2. **Add `add_item` and `add_bundle`.** Update X, the labels, N, f, deg and H in place.
3. **Test it.** Adding items and bundles one at a time must give the same scores as rebuilding the model from the
   enlarged data with the same parameters.
4. **Measure the inductive case.** The SVD is fitted on every item's title, test-only items included. It uses no
   bundles, so nothing leaks, but a truly new item is projected with a basis that never saw it. Refit the SVD on
   training items only and compare.
