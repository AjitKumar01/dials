# Scaling DIALS to large catalogues (notes for later)

Status: ideas and measurements only. Nothing here is implemented or validated, and the frozen model (README.md,
ARCHITECTURE.md) is unchanged. The largest catalogue tested so far is Spotify (254,155 items).

## 1. Where the cost is today

Inference scores every item for every query, so latency grows linearly with the catalogue size $n$
(CPU, frozen runs, `runs/summary.json`):

| Dataset | Items | ms / query |
|---|---|---|
| Electronic / Food / Clothing | 4–7 k | 0.11 |
| POG | 48,676 | 0.49 |
| Spotify | 254,155 | 3.24 |

These include the query-independent cache (keys, histories and prior computed once per evaluation). Extrapolated:
about 13 ms at 1 M items and about 130 ms at 10 M items, still scoring every item.

Training: one POG step (batch 32, forward plus backward, 4 CPU threads, Apple M5 Pro, median of 17) takes about 56 ms.
That is 437 steps, or about 24 s, per epoch. The parts, each timed on its own with its backward pass, so they overlap
and do not add up exactly:

| Part | ms | What it computes |
|---|---|---|
| Content term | 34 | of which: encoder $Z=\mathrm{normalize}(X+\mathrm{MLP}(X))$ over all $n$ items 9.5; keys $K=ZB_z^\top+HB_h^\top$ about 5 more; the per-bundle leave-one-out loop and the $32\times n$ product $\Phi K^\top$ the rest |
| Label term | 12 | $32\times n$ broadcast of the low-rank part, per fold, plus the $32\times n$ label gather for the diagonal |
| Loss | 12 | masks built in a Python loop, logsumexp over all $n$ items |
| Graph term and prior | 5 | the sparse part: only the seeds' edges of $N$ |
| Gate | 0.3 | |

So the sparse $n\times n$ co-occurrence matrix is already cheap: it is stored as CSR-style arrays (pointer, target,
count), and a step only reads the seeds' rows. Almost all of the time goes to dense work over all $n$ items:
encoding and keying the whole catalogue, $32\times n$ score matrices, and the full-catalogue softmax.

Parameter count is not compute. The 41.6 k shared parameters are applied to every item at every step. CLHE and
RaMen have millions of parameters, but mostly embedding rows that a batch only looks up. They also normalize over the
whole catalogue (a multinomial log-softmax; their `neg_num` setting is unused by the loss), but with large batches
(CLHE 256, RaMen 1024), so a RaMen epoch on POG is 14 steps against 437 for DIALS at batch 32.

## 2. Speeding up training without changing the model (constant factors, estimates)

1. Put the key maps on the query side. $\Phi K^\top=(\Phi B_z)Z^\top+(\Phi B_h)H^\top$ costs $q\,d\,n$ instead of
   $n\,d\,r$: half, at $q=32$, $d=128$, $r=64$. Exact.
2. Score the low-rank label part per label, then gather. Compute $\tfrac{t(S)}{|S|}P_lM_lQ_l^\top\in\mathbb{R}^{q\times L}$
   ($L$ labels), then index columns by $g(c)$. Exact; costs $q\cdot 16\cdot L$ instead of $q\cdot 16\cdot n$.
3. Score the diagonal only on items whose label occurs in the seeds, via an inverted index label → items, instead of
   gathering a dense $q\times n$ tally and calling `nonzero`. Exact.
4. Vectorize the leave-one-out corrections and the loss masks. The Python loops over the 32 bundles become
   scatter/gather operations. Exact.
5. Precompute `history()`. It is a fixed buffer but is recomputed on every call. Exact, and small.

Expected combined gain: roughly 2×. This is a guess, not measured. Each change needs a bit-for-bit or
tolerance check against the frozen model, as was done for the `dials` package.

## 3. Removing the dependence on n (changes the training objective)

As long as the loss normalizes over every item, each step costs at least order $q\cdot n$. Two standard ways out:

- Sampled softmax with a logQ correction. Normalize over the bundle's items plus 4–16 k negatives drawn by popularity,
  subtracting $\log Q(c)$ from each sampled logit. The encoder and keys then run only on the sampled items. With only
  41.6 k shared parameters the model does not need a gradient from every item at every step.
- A cache of stale keys (memory bank). Refresh $Z$ and $K$ for the whole catalogue every few hundred steps, and take
  gradients only through the items in the batch.

Either one must first be shown to reproduce the full-softmax results. Spotify is the natural test (clean NDCG@20 .2824).

## 4. Inference at millions of items: candidates, then exact re-ranking

The score is a sum of parts, and each part has a cheap index:

```math
s(c)=w_C\,\varphi(S)^\top k_c+\sum_l w_l\Big(\delta_l[\text{shared label}]+\tfrac{t(S)}{|S|}P_lM_lQ_l[g(c)]^\top\Big)+w_G\,\theta\log\Big(1+\sum_{s\in S}N_{sc}\cdots\Big)+\rho\big(\log(1+f_c)\big)
```

| Part | Index | Exact? |
|---|---|---|
| Content $\varphi(S)^\top k_c$ | keys $K\in\mathbb{R}^{n\times64}$ are fixed after training; nearest-neighbour search (FAISS, ScaNN) | approximate (search recall) |
| Labels | depends on $c$ only through $g(c)$: score the $L$ labels, then take the items of the top labels | exact |
| Graph | nonzero only for neighbours of the seeds: read their adjacency lists | exact |
| Prior | fixed per item: a precomputed popularity list | exact |

Stage 1 takes the union of these candidate lists, a few thousand items. Stage 2 computes the exact DIALS score on them.
Cost then depends on the number of candidates, not on $n$. Since the parts are weighted with non-negative gate
weights, a threshold algorithm over the per-part sorted lists (Fagin) can return the exact top 20.

New items need no history: $k_c$ comes from the item's features, with $H_c=0$, so a new SKU is scorable as soon as it
is listed.

## 5. Memory

DIALS has no per-item parameters, so the model itself does not grow with $n$. Only the fixed buffers do:

| Buffer | Size at 1 M items | Options |
|---|---|---|
| Features $X$, partner history $H$ ($n\times128$ each) | 512 MB each, float32 | float16 halves it; product quantization |
| Keys $K$ ($n\times64$), for serving | 256 MB | float16, or product quantization inside the index |
| Co-occurrence $N$ (CSR) | grows with distinct pairs, potentially $10^9$ | lossless gap and count coding, about 4× on Spotify (section 6); keep the top $k$ partners per item (as ItemKNN does); a count-min sketch (hashed, approximate counts) while counting |

Hashing helps with building $N$ from a stream of baskets, not with training speed. The hashing trick for embedding
tables does not apply, since DIALS has none.

## 6. Memory on an edge device: Spotify, measured

Measured on 2026-11-11 on the trained Spotify model (released CLAP features, seed 2; 254,155 items). The question:
the least memory that still gives exactly the same scores.

**As trained, DIALS holds 1.2 GB of buffers.** About 106 MB of that, the five cross-fitting folds of the label
factors, is used only in training. Most of the rest is the co-occurrence matrix $N$:
- 67.2 M stored entries: 33.6 M distinct pairs, each stored in both rows; 264 partners per track on average, 17,779
  at most;
- stored as int64 indices (537 MB) and float32 counts (269 MB).

**Exact inference, uncompressed.** Precompute everything that does not depend on the query:

| Part | Size |
|---|---|
| Per-item query rows $z_sA_z^\top+h_sA_h^\top$ and key rows $k_c$, $n\times64$ each, float32 | 130 MB |
| $N$ as CSR: int32 indices and uint16 counts (maximum count 299) | **403 MB** |
| Label factors $P_lM_l$ and $Q_l$, album (116,217) and artist (49,164), rank 16 | 21 MB |
| Labels $g_l(c)$, prior $\rho(\log(1+f_c))$, log-degrees | about 5 MB |
| Trained parameters (41,978) | 0.17 MB |
| **Total** | **about 560 MB** |

**$N$ compresses about 4× without loss.** Each row is sorted; the indices are stored as gaps, and the counts are coded
by their distribution. A row still decodes on its own, and a query reads only its seeds' rows (about 23 on Spotify).

| Part of $N$ | Plain | Compressed, lossless | How |
|---|---|---|---|
| Indices | 269 MB | 106 MB (1.57 bytes/entry) | gaps between sorted indices, as varints |
| | | **89 MB** (1.33 bytes/entry) | the same, after renumbering tracks by artist, then album, so partners get nearby ids |
| | | 65 MB | entropy limit of those gaps (information bound, not a format) |
| Counts | 134 MB | **about 17 MB** | 1-bit flag "count is 1", one extra byte otherwise: 86.9% of counts are 1, 99.7% are at most 15 |
| | | 7 MB | entropy limit, 0.87 bits per count |
| **Total** | **403 MB** | **about 106 MB** | |

Symmetry does not halve this. Keeping only the upper triangle means a row lookup also needs a column lookup, and that
costs the second orientation again.

**DIALS compared with CLHE and RaMen.** The CLHE and RaMen figures come from reading their scoring code (`evaluate`),
not from a device. Their minimum precomputes the item vectors that scoring uses:
- CLHE: a seed-side vector and a candidate-side vector of 64 dimensions per item, plus a small transformer;
- RaMen: about four 64-dimensional vectors per item: bundle view, graph view, hypergraph view and candidate view.

| | Trained parameters | As trained | Exact minimum, float32 | float16 vectors |
|---|---|---|---|---|
| DIALS, plain $N$ | 41,978 | 1.2 GB | about 560 MB | about 495 MB |
| **DIALS, compressed $N$** | 41,978 | | **about 260 MB** | **about 200 MB** |
| RaMen | 67.0 M | about 1 GB of raw features, 268 MB of parameters, its item–item graph | about 260 MB | about 130 MB |
| CLHE | about 34 M | about 1.1 GB of raw features, 137 MB of parameters | about 130 MB | about 65 MB |

**What has to be in RAM.** Every query scores all $n$ items, so the candidate keys must be in RAM. $N$ and the query
rows are read only for the seeds, so they can stay in flash and be memory-mapped:

| | RAM | Flash |
|---|---|---|
| DIALS | about 80 MB (keys in float16, plus label factors and small buffers) | about 140 MB |
| RaMen | about 65–130 MB | about 65 MB |
| CLHE | about 35–65 MB | about 30 MB |

On RAM the three methods are close. DIALS needs more flash because it keeps $N$ itself rather than folding it into
item vectors.

**Smaller still, but no longer exact:**
- keep only each item's top $k$ partners; $k=100$ gives about 150 MB plain, before compression;
- product-quantize the keys.

Both change the scores and need a retrain or an accuracy check. What DIALS keeps in either case: its parameters do not
grow with the catalogue, and a new track needs only its features and a row of $N$, with no retraining. CLHE learns
128 parameters per item and RaMen about 256.

## 7. Experiments to run before claiming any of this

1. Section 2 changes: check the outputs match the frozen model, then time an epoch on POG and Spotify.
2. Sampled softmax on Spotify: retrain three seeds and compare clean NDCG@20 with the frozen .2824.
3. Candidates plus re-ranking on Spotify: report the recall of the candidate set and the change in NDCG@20 against
   scoring every item.
4. Synthetic timing at 1 M and 10 M items: random features and a random sparse graph, latency and memory only.
5. Compressed $N$ on Spotify: confirm scores match the plain CSR bit for bit; measure the per-query decoding time
   and the resident memory (RSS) with $N$ memory-mapped.
6. float16 keys and query rows: report how many test top-20 lists change, and the change in NDCG@20.
