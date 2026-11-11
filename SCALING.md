# Scaling DIALS to large catalogues (notes for later)

Status: ideas and measurements only. Nothing here is implemented or validated, and the frozen model (README.md,
ARCHITECTURE.md) is unchanged. The largest catalogue tested so far is Spotify (254,155 items).

## 1. Where the cost is today

Inference scores every item for every query, so latency grows linearly with the catalogue size $n$
(CPU, frozen runs, `runs/summary.json`):

| Dataset | Items | ms / query |
|---|---|---|
| Electronic / Food / Clothing | 4–7 k | 0.11 |
| POG | 48,676 | 0.63 |
| Spotify | 254,155 | 4.25 |

Extrapolated: about 17 ms at 1 M items and about 170 ms at 10 M items, still scoring every item.

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
RaMen have millions of parameters, but mostly embedding rows that a batch only looks up, and they train with BPR
and one negative (two scored items per bundle, batch 1024), so their epochs are 14 steps of cheap work.

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

Either one must first be shown to reproduce the full-softmax results. Spotify is the natural test (clean NDCG@20 .2816).

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
| Co-occurrence $N$ (CSR) | grows with distinct pairs, potentially $10^9$ | keep the top $k$ partners per item (as ItemKNN does); int32 indices and int16 counts; a count-min sketch (hashed, approximate counts) while counting |

Hashing helps with building $N$ from a stream of baskets, not with training speed. The hashing trick for embedding
tables does not apply, since DIALS has none.

## 6. Experiments to run before claiming any of this

1. Section 2 changes: check the outputs match the frozen model, then time an epoch on POG and Spotify.
2. Sampled softmax on Spotify: retrain three seeds and compare clean NDCG@20 with the frozen .2816.
3. Candidates plus re-ranking on Spotify: report the recall of the candidate set and the change in NDCG@20 against
   scoring every item.
4. Synthetic timing at 1 M and 10 M items: random features and a random sparse graph, latency and memory only.
