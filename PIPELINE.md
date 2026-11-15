# DIALS, end to end: from raw files to test scores

This document follows one dataset through the code, from the files on disk to the numbers in `runs/summary.json`. At
every step it names the function that runs and the variable that holds the result, with its shape.

The example is **Electronic**, the same dataset as ARCHITECTURE.md's running example. It is small enough to show real
sizes. Where another dataset differs, that is said. ARCHITECTURE.md explains *why* each component has its form; this
document explains *what the code does, in order*. Section numbers like "ARCH §7" point to ARCHITECTURE.md.

**Shapes:** q = queries in a batch, n = catalogue size, d = 128 feature dimensions, r = 64 content rank, K = number of
label values at a level, k = 16 label rank. ❄️ marks fixed values built from data; 🔥 marks trained parameters.

---

## 0. The whole pipeline on one page

```
python -m dials train --dataset electronic --seed 0
│
├─ 1. import dials                      __init__.py   pin BLAS to 1 thread (reproducible SVDs)
├─ 2. data.load('electronic')           data.py       → ds: Dataset (bundles, splits, features, labels)
├─ 3. train.fit(ds, seed, run_dir)      train.py
│   ├─ 3a. train.build(ds)
│   │     ├─ train.spectral(ds)         ❄️ label factors P, Q (+5 fold copies), cached in cache/
│   │     └─ DIALS(...)                 ❄️ N, f, deg, NX as buffers; 🔥 parameters initialized
│   ├─ 3b. optimizer                    two AdamW groups
│   └─ 3c. epochs
│         ├─ batches of 32 bundles: hide → model(seeds, bundles) → loss → step
│         └─ evaluate(model, val)       keep best epoch, stop after 6 bad epochs
│   → runs/electronic/seed0/{model.pt, selected.json, history.json}
│
python -m dials test --datasets electronic
└─ 4. for seeds 0,1,2: train.load → evaluate(model, test) → test.json; means → runs/summary.json
```

---

## 1. Before anything runs: `dials/__init__.py`

Importing `dials` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to 1,
unless they are already set.
- **Why:** randomized SVDs give bit-identical results only at a fixed thread count. Apple's Accelerate library can't
  be limited after numpy has loaded it.
- **Consequence:** `dials` must be imported before numpy. If numpy came first, a warning says the SVD caches may not
  be bit-reproducible. Scripts in this repo start with `import dials` for this reason.

---

## 2. Data: `data.load(name)` → a `Dataset`

Every dataset ends up in the same container, `data.Dataset`:

| Field | Type | Electronic | Meaning |
|---|---|---|---|
| `name` | str | `'electronic'` | |
| `n_items` | int | 3,499 | catalogue size n; items are numbered 0..n−1 |
| `train` | list of lists | 1,211 bundles, sizes 2–10 (mean 3.5) | training bundles; e.g. `train[0] = [697, 826]` |
| `val_pairs` | list of (seeds, held-out) | 173 | e.g. `([1268], [2186])` |
| `test_pairs` | list of (seeds, held-out) | 346 | e.g. query 20 is `([2708, 1912], [943])` |
| `features` | float32 array, n × 128 | 3,499 × 128 | item features X (not yet normalized for e-commerce; the model normalizes them) |
| `levels` | list of int arrays, each length n | one level, 479 categories | label value of every item at each level |
| `clean_test` | list of int | all 346 | indices of test queries whose full bundle is in neither training nor validation |

`load(name)` sends each name to one of three loaders.

### 2a. `load_resplit`: Electronic, Food, Clothing

Files: `<DATA_ROOT>/BundleRec/dataset/<name>/`. These are the crowd-labelled bundles of Sun et al. (SIGIR 2022).

1. **Read the bundles.** `bundle_item.csv` gives (bundle, item) rows, collected into `raw: {bundle_id: [items]}`.
2. **Renumber the items.** Only items that occur in some bundle are kept, numbered 0..n−1 in sorted order of their
   original IDs (`index: {original_id: new_id}`).
3. **Clean the bundles.**
   - Repeated items inside a bundle are dropped, keeping order.
   - Bundles with fewer than 2 items are dropped.
   - Of identical bundles (same item set), only the first is kept, so no bundle appears in two splits.
   - Result: `bundles`.
4. **Features.**
   - `item_titles.csv` gives one title string per item (`titles`).
   - TF-IDF: at most 8,000 unigrams and bigrams, English stop words removed, sublinear term frequency, rows
     L2-normalized.
   - A truncated SVD to 128 dimensions gives `features`. It is not re-normalized here; the model does that.
5. **Labels.**
   - `item_categories.csv` gives each item's category string, numbered by first appearance into `labels`.
   - An item with no category row keeps the initial value 0, so it shares the first category seen.
   - The IDs are then renumbered 0..K−1 by `np.unique(..., return_inverse=True)`.
6. **Split.**
   - The bundle order is shuffled with `random.Random(0)`.
   - The first 70% are `train`, the next 10% validation, the last 20% test.
7. **Hide half of each held-out bundle** with `hide(bundles, seed)`, seed 1 for validation and 2 for test:
   - shuffle the bundle;
   - the first round(|b|/2) items are the seeds (at least one), the rest are held out;
   - if nothing would be held out, the last item is.

   The result is lists of `(seeds, held_out)`.
8. **Clean test queries:** duplicates were removed in step 3, so every test query is clean: `clean_test =
   range(len(test))`.

680 catalogue items never occur in a training bundle, so their training frequency is 0. They are the "cold"
items.

### 2b. `load_supplied`: POG, POG_DENSE, Spotify_sparse, Spotify

Files: `<DATA_ROOT>/bundlemllm/datasets/<name>/`, the released splits, used unchanged.

1. **Catalogue size:** `count.json["#I"]`.
2. **Splits.** `bi_train.txt`, `bi_valid_input.txt` / `bi_valid_gt.txt` and `bi_test_input.txt` / `bi_test_gt.txt`
   are read by `read_rows`. Each row is `bundle_id,item,item,...`.
3. **Checks.** Loading stops with an error if any of these fails:
   - every bundle ID appears once, no item repeats within a row, every item is in range, no row is empty;
   - input and ground-truth files have the same IDs;
   - no ID is shared between splits;
   - no seed also appears in its own ground truth.
4. **Labels** from `item_info.json`:
   - POG: one level, `cate` (or `cate_id`).
   - Spotify: two levels, `artist_uri` and `album_uri`.
   - IDs follow first appearance; the first level is then renumbered in sorted order.
5. **Clean test queries.** A test query is *not* clean when its full bundle (seeds ∪ truth) equals a training bundle
   or a full validation bundle. POG_DENSE has 5,547 clean of 5,938.
6. **Features** (`projected_features`, cached as `cache/<name>/features_released.npy`):
   - load the two released tensors: `content_feature.pt` (BLIP image on POG, CLAP audio on Spotify) and
     `description_feature.pt` (BLIP text, CLAP text);
   - check that both have n rows;
   - L2-normalize each, concatenate (media first, then text) and normalize again;
   - truncated SVD to 128 dimensions;
   - normalize the rows.

### 2c. `load_baskets`: Dunnhumby

File: `<DATA_ROOT>/dunnhumby/transaction_data.csv` and `product.csv`.

1. Group transactions into baskets by `BASKET_ID`, as sets of `PRODUCT_ID`.
2. Keep baskets with 3–20 distinct products, shuffle them with `random.Random(0)`, and keep the first 30,000.
3. Count products over those baskets. Drop products seen fewer than twice, then drop baskets left with fewer than 3
   products.
4. Renumber products 0..n−1 and drop duplicate baskets.
5. **Labels:** `SUB_COMMODITY_DESC` (blank becomes `'?'`). A product missing from `product.csv` gets a label of its
   own.
6. **Features:** TF-IDF of "department + commodity + sub-commodity + brand", then SVD to 128 dimensions, as in 2a.
7. Same 70/10/20 split and hiding as 2a. Households are not used.

---

## 3. Fixed structures ❄️: `train.build(ds)`

`build` calls `train.spectral(ds, cache)` for the label factors, then `DIALS(ds.features, ds.levels, ds.train,
spectral, rank=64, hidden=32)`. Everything below is computed from `ds.train` only.

### 3a. Co-occurrence: `model.pair_counts(train, n)`

The matrix N (ARCH §4) is never stored densely. It is n × n = 12M entries on Electronic, of which 13,788 are
non-zero.

1. Each bundle becomes a sorted array of distinct items (`rows`).
2. **Frequency:** `frequency[c]` = f_c, the number of bundles containing c. This is `np.bincount` over all rows.
3. **Pairs:** for each bundle, every ordered pair (s, c) with s ≠ c is encoded as one integer `s*n + c`. It uses
   `np.repeat` for the s column and `np.tile` for the c column, then drops the diagonal.
4. **Counts:** `np.unique(keys, return_counts=True)` gives each distinct pair and how many bundles contain it, so
   `counts[e]` = N_{s,c}. `np.unique` also sorts the keys, which means sorted by source, then target.
5. **Decode:** `source = keys // n`, `target = keys % n`.
6. **Partner index:** `pointer = searchsorted(source, 0..n)`. The partners of item s are `target[pointer[s] :
   pointer[s+1]]`, one contiguous slice. This is CSR format without a sparse-matrix object.

Both directions are stored, (s, c) and (c, s), so N is symmetric.

### 3b. Buffers registered in `DIALS.__init__`

All buffers are created with `persistent=False`. They are rebuilt from `ds.train` whenever a model is built, and are
**not saved in checkpoints**. `model.pt` holds only the 🔥 parameters.

| Buffer | Electronic shape | POG_DENSE shape | Value |
|---|---|---|---|
| `features` | 3499 × 128 | 31217 × 128 | X with L2-normalized rows (`F.normalize`) |
| `pointer` | 3500 | 31218 | partner index from 3a |
| `target` | 13,788 | 153,074 | partner c of each stored pair |
| `counts` | 13,788 | 153,074 | N_sc of each stored pair |
| `frequency` | 3499 | 31217 | f_c |
| `history_sum` | 3499 × 128 | 31217 × 128 | N X: for each item, the count-weighted sum of its partners' features |
| `history_mass` | 3499 | 31217 | deg_c = Σ_s N_sc, the row sums of N |
| `log_degree` | 3499 | 31217 | log max(deg_c, 1) |
| `label0` | 3499 | 31217 | g(c) at level 0 |
| `label_size0` | 479 | 67 | log(1 + number of items with each label value) |
| `label_left0`, `label_right0` | 479 × 16 | 67 × 16 | P, Q from all training bundles |
| `label_fold_left0`, `label_fold_right0` | 5 × 479 × 16 | 5 × 67 × 16 | P^(k), Q^(k) without fold k |

Spotify has a second level, so it also gets `label1`, `label_size1`, `label_left1` and so on.

On Electronic, 680 items have f = 0 and deg = 0 (6,786 on POG_DENSE). For them, `history()` returns a zero row, and
the graph term gives only the prior.

### 3c. Label factors: `train.spectral` → `model.cross_fitted_labels`

1. **Cache key.** A SHA-256 of the training bundles (each sorted) and every label array. The file is
   `cache/<name>/labels_r16_<first 16 hex digits>.npz`. A changed split or label set gives a new file name, so a
   stale cache is never reused.
2. **If cached,** the arrays `0_left`, `0_right`, `0_fold_left`, `0_fold_right` (and `1_*` for a second level) are
   read back into the nested structure.
3. **If not,** `cross_fitted_labels(train, n, levels, 16)` runs:
   - `fold_of(b)`: an MD5 hash of the bundle's sorted distinct items, mod 5. Identical bundles share a fold. On
     Electronic the folds hold 235, 239, 249, 241 and 247 bundles.
   - **Full set:** build N from all training bundles, then `label_factors` for every level.
   - **Fold sets:** for k = 0..4, build N from the bundles outside fold k, then `label_factors`.
   - The result is regrouped per level: `((P, Q), [(P^(0), Q^(0)), …, (P^(4), Q^(4))])`.
4. **`label_factors(N, labels, 16)`:**
   - Γ: the n × K one-hot matrix (sparse).
   - Label co-occurrence: N̄ = Γᵀ N Γ, which is K × K.
   - Label degrees: D = diag(N̄ 1), floored at 1.
   - Normalize: N̂ = D^-½ N̄ D^-½.
   - Randomized SVD of rank min(16, K−1), with 4 power iterations and `random_state=0`: N̂ ≈ U Σ Vᵀ.
   - Return P = U Σ^½ and Q = V Σ^½, as float32.

   Six SVDs per level in all: one full set and five fold sets.

---

## 4. Trained parameters 🔥 and their initial values

41,624 parameters on every one-level dataset (41,978 on Spotify, which has two levels).

| Parameter | Shape | Count | Initial value | Group |
|---|---|---|---|---|
| `encoder` (Linear 128→32, GELU, Linear 32→128) | | 8,352 | default init, last layer **zero** (so Z = X) | shared |
| `query_content` A_z | 64 × 128 | 8,192 | √10 · [I₆₄ 0] | shared |
| `key_content` B_z | 64 × 128 | 8,192 | √10 · [I₆₄ 0] | shared |
| `query_history` A_h | 64 × 128 | 8,192 | 0 | shared |
| `key_history` B_h | 64 × 128 | 8,192 | 0 | shared |
| `mixing[l]` M_l | 16 × 16 | 256 per level | 0 (low-rank label part off) | fast |
| `diagonal[l]` δ_l (Linear 3→16, GELU, Linear 16→1) | | 81 per level | default init | fast |
| `exponents` (α, β) | 2 | 2 | 0, 0 (raw counts) | fast |
| `theta` θ | scalar | 1 | 1 | fast |
| `prior` ρ (Linear 1→16, GELU, Linear 16→1) | | 49 | default init | fast |
| `gate` (Linear 3→16, GELU, Linear 16→2+L) | | 115 (one level) | last layer weight 0, bias log(e−1) (every weight starts at 1) | fast |

- **Fast group:** the small relation parameters, 504 in total, listed by `fast_parameters()`. They train at learning
  rate 10⁻².
- **Shared group:** everything else. It trains at 10⁻³.

At initialization the model is a plain sum with weight 1 per term:
- **Content:** 10 × the dot product of the first 64 coordinates of X (seed mean against candidate).
- **Labels:** only δ_l, since M_l = 0.
- **Graph:** log(1 + raw co-occurrence count) + ρ(f).

---

## 5. The forward pass: `model(seeds, bundles=None, cache=None)` → q × n scores

Two modes, set by the arguments:

| Mode | Called from | `bundles` | `cache` | Leave-one-out |
|---|---|---|---|---|
| **Training** | `train.fit` | the complete training bundles | None | yes |
| **Evaluation** | `evaluate.score_batches` | None | `model.cache()` | no |

Passing both raises an error, because the cache holds no leave-one-out corrections.

**Step 0: normalize the input.** Each seed list becomes a sorted tensor of distinct item IDs. In training, the bundles
are converted the same way.

**Step 1: content, `content_scores` (ARCH §6).**
1. **Item tables.**
   - Evaluation reads them from the cache.
   - Training computes Z = normalize(X + encoder(X)) (n × 128), the keys Z B_zᵀ + H B_hᵀ (n × 64), and
     H = `history()` = N X / deg, with zero rows where deg = 0.
2. **Seed side, for each query.**
   - Take the seeds' rows of Z and H.
   - *Training:* the seeds' history rows are replaced by `loo_history(seeds, bundle)`. It subtracts the other bundle
     members' features from N X and |b|−1 from deg.
   - query(s) = A_z z_s + A_h h_s; then φ(S) = the mean over seeds (a vector of 64 numbers).
3. **Scores:** stack φ into q × 64, then `scores = φ Keysᵀ` (q × n), one matrix product.
4. **Training correction:** the bundle's own members are also candidates, and their keys should use leave-one-out
   histories too. For each member, the code stores Δkey = (h_loo − h) B_hᵀ and adds φ(S)·Δkey at those (query,
   member) entries with `index_put(..., accumulate=True)`. This avoids rebuilding the full key matrix per query.

**Step 2: labels, `label_scores(i, …)` for each level i (ARCH §7).**
1. **Seed label counts:** `tally` (q × K), with `tally[j, g]` = number of seeds of query j with label g. It is built
   with `bincount`.
2. **Label mix:** `mean = tally / row sums` (q × K), which is t(S)/|S|.
3. **Low-rank part:** `mean @ P @ M` gives q × 16, then `@ Q[label].T`, where `Q[label]` is n × 16, one row per item,
   for q × n in total.
   - *Evaluation:* P, Q from all bundles.
   - *Training:* for each fold k present in the batch, the queries whose bundle is in fold k use P^(k), Q^(k).
4. **Diagonal part.**
   - `m = tally[:, label]` (q × n) is how many seeds share each candidate's label.
   - Only the non-zero entries are evaluated. For each, the input is (log(1+m), log|S|, log(1 + number of items with
     that label)), run through δ_l.
   - The outputs are added at those entries (sparse `index_put`).
5. **No leave-one-out here.** The label counts come from the seeds, and the label sizes from the catalogue, not from
   bundles. The only bundle-derived piece, P and Q, is handled by the folds.

**Step 3: graph, `graph_scores` (ARCH §8).**
1. **Gather the seeds' partners:** `expand(query_ids, seed_ids)` returns, for every stored pair whose source is a
   seed, the query index, the source and the pair index `edge`. It does this with `repeat_interleave` over the
   `pointer` slices, without a Python loop.
2. **Per pair:** target c = `target[edge]`, n_sc = `counts[edge]`, log deg_s and log deg_c.
3. **Training: leave-one-out.**
   - `member` (q × n, bool) marks each query's bundle items, and `inside` = 1 when the pair's target is in the bundle.
   - The count becomes n_sc − inside.
   - The seed's degree becomes deg_s − (|b|−1).
   - A target in the bundle has its degree reduced by |b|−1 too.
4. **Weights:** w = n_sc · exp(−α log deg_s − β log deg_c) = n_sc deg_s^-α deg_c^-β.
5. **Sum over seeds:** `scatter_add` into a q·n vector at position query·n + c, reshaped to q × n.
6. **Damping:** θ · log(1 + sum).
7. **Prior:** ρ(log(1 + f_c)) for every item (from the cache at evaluation), broadcast to q × n. In training, the
   bundle's own members use ρ(log(1 + f_c − 1)) instead.

**Step 4: gate, `gate_weights` (ARCH §9).**
1. For each query, f_s is the seeds' frequencies (f_s − 1 in training, floored at 0).
2. The three inputs are u = (log|S|, mean log(1+f_s), share of seeds with f_s = 0).
3. w = softplus(gate MLP(u)), which is q × (2 + levels), in the order content, label levels, graph.

**Step 5: total.**

  s = w[:,0]·content + Σ_l w[:,1+l]·label_l + w[:,last]·graph   (q × n)

Every item in the catalogue gets a score, the seeds included. Seeds are removed later, by the loss and by `top_items`.

---

## 6. Training: `train.fit(ds, seed, run_dir)` (ARCH §10)

**Set up.**
- `torch.manual_seed(seed)` and `rng = np.random.default_rng(seed)`.
- `model = build(ds)`.
- The training list `train` keeps only bundles with at least 2 distinct items.
- Optimizer: `AdamW` with the fast group at 10⁻², the shared group at 10⁻³, and weight decay 10⁻³ for both.
- `state` tracks `epoch`, `best` (validation NDCG@20), `best_epoch`, `stale` (epochs since an improvement), `history`
  and `elapsed`.

**Resume.**
- If `run_dir/last.pt` exists, the model, the optimizer, both random-generator states and `state` are restored.
- The saved config and seed must match, or training stops with "Resume mismatch".
- Because the random states are restored, a resumed run is identical to an uninterrupted one. A test checks this.

**One epoch.**
1. Stop if `stale ≥ 6`, or after 100 epochs.
2. Shuffle the order of the training bundles with `rng`.
3. **For each batch of 32 bundles:**
   - **Split:** `hide(rows, rng)`. Each bundle's distinct items are permuted. With probability ½ the cut is at
     max(1, |b|//2); otherwise it is uniform in 1..|b|−1. The first part is the seeds, the rest the hidden targets.
   - **Score:** `scores = model(seeds, bundles=[the full bundles])`, in training mode with leave-one-out (q × n).
   - **Loss**, `loss(scores, seeds, truth)`:
     - `allowed` (q × n) is True outside the bundle, `positive` is True at the hidden items.
     - LSE = logsumexp of the scores where allowed, one per query.
     - Per hidden item j: softplus(LSE − s_j). This is averaged within each query, then over the batch.
     - Every catalogue item outside the bundle is a negative; there is no negative sampling.
   - **Step:** backward, clip the gradient norm to 5 (stop on non-finite), AdamW step. A non-finite loss stops
     training.
4. **Validation:** `evaluate(model, ds.val_pairs)`.
   - If NDCG@20 beats `best`, save the parameters to `model.pt` and reset `stale` to 0.
   - Otherwise add 1 to `stale`.
5. **Record and checkpoint.**
   - Append the bundle-weighted mean loss and the validation metrics to `history`.
   - Save model, optimizer, random states, config, seed and `state` to `last.tmp`, then rename it to `last.pt`. The
     rename is atomic, so a crash never leaves a half-written checkpoint.

**Finish.**
1. Reload `model.pt` and evaluate validation again. It must reproduce the best score to within 10⁻⁹, or training
   stops with an error.
2. Write `history.json` (per epoch) and `selected.json`: dataset, seed, config, best epoch, epochs run, validation
   metrics, parameter count and seconds.
3. Delete `last.pt`.

**Determinism.** Fixed seeds, `torch.use_deterministic_algorithms(True)`, one thread by default (`--threads`), and
the single-threaded BLAS from step 1 make a run bit-reproducible.

---

## 7. Evaluation and test (ARCH §12)

**Loading a finished model,** `train.load(ds, run_dir)`:
1. Read the config from `selected.json`.
2. `build(ds, config)` recomputes every buffer from `ds.train`.
3. Load `model.pt`, which holds the parameters.

**Scoring,** `evaluate(model, pairs)`:
1. `model.cache()` computes the query-independent tables once: Z, H, the key matrix and the prior.
2. Queries go in batches of 32 through `model(seeds, cache=cache)`, giving 32 × n scores per batch. Non-finite scores
   stop the evaluation.
3. **Per query, `query_metrics`:**
   - `top_items(scores, seeds, 20)` sets the seeds' scores to −∞ and finds the 20th-largest score with
     `np.partition`.
   - It takes every item strictly above that score, then fills the rest with tied items in increasing ID order.
   - The result is sorted by (score descending, ID ascending). Exact ties therefore never depend on batching.
   - Recall@20 = hits / |held out|. NDCG@20 = Σ hit_k / log₂(k+1) over the ideal DCG with min(|held out|, 20) hits.
4. **Returns:** the mean recall20 and ndcg20, a record per query (recall, NDCG, top-20 list) and the seconds taken.

**`python -m dials test`:**
1. Uses 4 threads and deterministic algorithms.
2. Skips any dataset where not all three seeds have finished.
3. **For seeds 0, 1 and 2:** if `test.json` is missing, load the model and evaluate the test queries once, writing
   all per-query records. Otherwise reuse the file.
4. **`runs/summary.json[name]` holds:**
   - the query counts;
   - `all` and `clean`: recall and NDCG averaged over the queries and the three seeds;
   - per seed: the clean-query metrics (`evaluate.subset`), the best epoch and the validation metrics;
   - the parameter count;
   - milliseconds per query.

---

## 8. Files written

| Path | Written by | Content |
|---|---|---|
| `cache/<name>/features_released.npy` | `data.projected_features` | supplied datasets only: the 128-dimensional features |
| `cache/<name>/labels_r16_<hash>.npz` | `train.spectral` | P, Q and the five fold copies per level |
| `runs/<name>/seed<s>/last.pt` | `train.fit`, each epoch | resume state; deleted at the end |
| `runs/<name>/seed<s>/model.pt` | `train.fit` | parameters of the best epoch |
| `runs/<name>/seed<s>/history.json` | `train.fit` | loss and validation metrics per epoch |
| `runs/<name>/seed<s>/selected.json` | `train.fit` | chosen epoch, validation metrics, config, size, time |
| `runs/<name>/seed<s>/test.json` | `python -m dials test` | per-query test results |
| `runs/summary.json` | `python -m dials test` | three-seed means, all and clean queries |

---

## 9. How the datasets differ

| | Electronic / Food / Clothing | Dunnhumby | POG / POG_DENSE | Spotify_sparse / Spotify |
|---|---|---|---|---|
| Loader | `load_resplit` | `load_baskets` | `load_supplied` | `load_supplied` |
| Bundles | crowd-labelled, deduplicated | grocery baskets, sampled | released outfits | released playlists |
| Split | ours: 70/10/20, half hidden | ours: 70/10/20, half hidden | released | released |
| Features | TF-IDF of titles → SVD 128 | TF-IDF of department to brand → SVD 128 | BLIP image + text → SVD 128 | CLAP audio + text → SVD 128 |
| Label levels | 1: category | 1: sub-commodity | 1: category | 2: artist, album |
| Clean test | all queries | all queries | full bundle not in train/val | full bundle not in train/val |
| Parameters | 41,624 | 41,624 | 41,624 | 41,978 |

The model, the configuration (`train.CONFIG`) and the training loop are identical on every dataset.

---

## 10. In the code but not spelled out in ARCHITECTURE.md

- **Item IDs.** On Electronic, Food and Clothing, items are renumbered over those that occur in any bundle. Items
  without a category row share the first category seen (label 0 before renumbering).
- **Feature normalization.** The e-commerce and Dunnhumby features leave `data.py` unnormalized. The model's
  `F.normalize` on `features` puts every dataset on unit-length rows.
- **Training bundles with one distinct item** are skipped by `fit`; they can't give a seed and a target.
- **Seeds are scored too.** The forward pass scores the whole catalogue, seeds included. Seeds are excluded later:
  from the loss negatives by `allowed`, and from rankings by `top_items`.
- **The label term has no leave-one-out.** Only P and Q depend on bundles, and those are cross-fitted. The diagonal's
  inputs come from the seeds and the catalogue.
- **The members-as-candidates correction** in `content_scores` adjusts only the bundle members' keys, through
  `index_put`. It never rebuilds the key matrix.
- **The early-stopping rule** counts epochs without a *strict* improvement and is checked at the start of the next
  epoch. The selected checkpoint must replay its validation score exactly.
- **Checkpoints and caches.**
  - Buffers are non-persistent, so `model.pt` is parameters only, and the training split always rebuilds the data
    structures.
  - The label cache is keyed by a hash of the split, so a different split can't silently reuse old factors.
- **Determinism** depends on importing `dials` before numpy (step 1).
