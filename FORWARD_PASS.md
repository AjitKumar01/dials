# The forward pass of DIALS, one bundle at a time

This document follows **one real bundle** through `DIALS.forward` in `dials/model.py`, line by line. Every number
below comes from the trained Electronic model (`runs/electronic/seed0`, best epoch 14). We ran the code on this
bundle and printed each intermediate value. Adding the pieces up by hand gives exactly the model's output (the
largest difference is 0.0).

It covers only the forward pass: how a partial bundle becomes one score per catalogue item. Where the forward pass
uses something built earlier (the co-occurrence counts, the label factors, the trained weights), that piece is
explained at the point where it is first used. The loss and the optimizer are not covered.

**Contents**

1. [The bundle we follow](#1-the-bundle-we-follow)
2. [The idea in one equation](#2-the-idea-in-one-equation)
3. [What exists before the forward pass starts](#3-what-exists-before-the-forward-pass-starts)
4. [Entering `forward`](#4-entering-forward)
5. [Term 1: content](#5-term-1-content-content_scores)
6. [Term 2: labels](#6-term-2-labels-label_scores)
7. [Term 3: graph](#7-term-3-graph-graph_scores)
8. [The gate](#8-the-gate-gate_weights)
9. [Adding up, and ranking](#9-adding-up-and-ranking)
10. [Many bundles at once, and the evaluation cache](#10-many-bundles-at-once-and-the-evaluation-cache)
11. [The same forward pass during training](#11-the-same-forward-pass-during-training)
12. [Summary of shapes](#12-summary-of-shapes)

---

## 1. The bundle we follow

Electronic test query 20 has two visible items (the **seeds**) and one hidden item (the **target**). The model must
find the target among all 3,499 catalogue items.

**What the 3,499 items are.** They are every distinct item that occurs in any bundle of BundleRec's Electronic file
(`bundle_item.csv`), across training, validation and test. `data.load_resplit` renumbers them 0 to 3,498 *before* it
splits the bundles, so the catalogue is fixed first and the split happens inside it:

| Items | Count | What the model knows about them |
|---|---|---|
| in at least one training bundle | 2,819 (80.6%) | features, category, co-occurrence counts, partner history |
| only in validation or test bundles | 680 (19.4%) | features and category only: f = 0, no partners, h = 0 |
| of those 680: only in validation / only in test / in both | 200 / 438 / 42 | |

So the 3,499 are not just the training items. The 680 items never seen in training are still candidates in every
query, and the model has to score them from their title features and category alone. This is the "cold" case that
the content and label terms exist for. Every item in the catalogue occurs in at least one bundle. The catalogue holds
no item that was never bundled. That is a property of the benchmark and is the same for every compared method.

| Role | Item ID | Title (shortened) | Category | f (training bundles containing it) |
|---|---|---|---|---|
| seed | 1912 | Lite-On 24X SATA DVD±RW drive | Internal DVD Drives (label 131) | 4 |
| seed | 2708 | Samsung SATA optical drive SH-224DB | External DVD Drives (label 48) | 1 |
| **target** | **943** | **Seagate Barracuda 320 GB internal hard drive** | Internal Hard Drives (label 2) | 11 |

Someone is building a desktop PC: they have two optical drives and the model should suggest the hard drive. We also
follow three items that compete with the target. They are the model's ranks 2, 3 and 4:

| Item | Title (shortened) | Category | f |
|---|---|---|---|
| 717 | Corsair CX750 power supply | Chargers & Adapters (label 29) | 8 |
| 2775 | Corsair Vengeance 8 GB DDR3 memory | Memory (label 130) | 2 |
| 2982 | AMD Athlon X4 processor | CPU Processors (label 115) | 2 |

The target's category differs from both seeds' categories. Keep this in mind: in this bundle the label term can't
reward "same category as a seed", and we will see that it barely contributes.

## 2. The idea in one equation

For a set of seeds S and every catalogue item c, DIALS computes one number (ARCHITECTURE.md §5):

```
s(c | S) = w_C(S) · C(S, c)  +  w_L(S) · L(S, c)  +  w_G(S) · G(S, c)
```

In plain words:

- **C, content:** does c *look like* something that goes with the seeds? It uses the item's title features.
- **L, labels:** does c's *category* go with the seeds' categories?
- **G, graph:** has c *appeared in training bundles together with* the seeds?
- **w, the gate:** three positive weights, one per term. They are decided by the seeds themselves (how many there are,
  and how well known they are), so the model can lean on the graph for well-known seeds and on content for new ones.

Each term produces one row of 3,499 numbers, one per catalogue item. The forward pass computes the three rows and the
three weights, then adds the weighted rows. The highest-scoring items that aren't seeds are the recommendation.

## 3. What exists before the forward pass starts

The model object holds two kinds of things. Both are created in `DIALS.__init__` and neither changes during a forward
pass.

### 3.1 Fixed data, built from the training bundles (never trained)

These are *buffers*: tensors that belong to the model but receive no gradient. On Electronic, n = 3,499 items and
d = 128 feature dimensions.

| Name in code | Symbol | Shape | What it holds |
|---|---|---|---|
| `features` | X | 3499 × 128 | Each item's title features (TF-IDF projected to 128 numbers), rows scaled to length 1 |
| `target`, `counts` | N | 13,788 each | The co-occurrence matrix N stored as a list of pairs; see below |
| `pointer` | | 3,500 | Where each item's pairs start in that list |
| `frequency` | f | 3,499 | f_c = number of training bundles containing c |
| `history_mass` | deg | 3,499 | deg_c = Σ_s N_sc, the total co-occurrence count of c |
| `log_degree` | log deg | 3,499 | log(max(deg, 1)) |
| `history_sum` | N X | 3499 × 128 | For each item, the sum of its partners' features, each weighted by the pair count |
| `label0` | g(c) | 3,499 | Each item's category number (479 categories) |
| `label_size0` | log(1 + n_g) | 479 | log(1 + number of items in each category) |
| `label_left0`, `label_right0` | P, Q | 479 × 16 each | Fixed label factors from a spectral decomposition; see §6 |

**How N is stored.** N_sc = the number of training bundles that contain both s and c. Most pairs never meet, so only
the 13,788 non-zero (s, c) entries are kept. They are sorted by s. `target[e]` and `counts[e]` are the partner and the
count of entry e. `pointer[s]` is the first entry belonging to s, so the partners of s are entries
`pointer[s]` to `pointer[s+1] - 1`. This is the CSR (compressed sparse row) layout.

For our two seeds:

```
pointer[1912] = 7878, pointer[1913] = 7887   → entries 7878..7886, 9 partners
pointer[2708] = 11021, pointer[2709] = 11028 → entries 11021..11027, 7 partners

partners of 1912 (item, N): (7,1) (717,2) (943,2) (2124,1) (2230,1) (2708,1) (2775,2) (2982,2) (3292,1)
partners of 2708 (item, N): (7,1) (717,1) (943,1) (1912,1) (2230,1) (2775,1) (2982,1)
```

So the hard drive 943 appeared in 2 training bundles with seed 1912 and in 1 with seed 2708. The power supply, the
memory and the processor appear with the seeds as often. The graph alone can't easily tell these four apart.

Where these counts come from: the training bundles containing 1912 are

```
[2982, 2775, 717, 943, 1912]
[1912, 943, 2230, 717, 2775, 7, 2982, 2708]
[1912, 3292]
[1912, 2124]
```

The second one holds both seeds *and* the target. Our test bundle {1912, 2708, 943} isn't identical to any training
bundle, so it counts as a clean test query, but it is part of a larger training bundle. That is why the graph term
does well here.

### 3.2 Trained parameters (41,624 numbers)

| Name in code | Symbol | Shape | Used in |
|---|---|---|---|
| `encoder` | MLP | 128 → 32 → 128 (8,352 numbers) | content: refines X into Z |
| `query_content`, `query_history` | A_z, A_h | 64 × 128 each | content: seed side |
| `key_content`, `key_history` | B_z, B_h | 64 × 128 each | content: candidate side |
| `mixing[0]` | M | 16 × 16 | labels: which categories go together |
| `diagonal[0]` | δ | 3 → 16 → 1 | labels: bonus for sharing a seed's category |
| `exponents` | α, β | 2 | graph: degree discounts |
| `theta` | θ | 1 | graph: scale |
| `prior` | ρ | 1 → 16 → 1 | graph: popularity curve |
| `gate` | | 3 → 16 → 3 | the three term weights |

Trained values that matter in this walk-through: α = 0.0359, β = 0.1748, θ = 1.8922.

## 4. Entering `forward`

```python
def forward(self,seeds,bundles=None,cache=None):
    if cache is not None and bundles is not None:
        raise ValueError('The cache is for evaluation only')

    # Distinct, sorted item IDs as tensors.
    seeds=[torch.as_tensor(sorted(set(s)),dtype=torch.long) for s in seeds]
    if bundles is not None:
        bundles=[torch.as_tensor(sorted(set(b)),dtype=torch.long) for b in bundles]
```

- `seeds` is a **list of queries**, and each query is a list of item IDs. For our bundle the call is
  `model([[2708, 1912]], cache=...)`: a list holding one query. The forward pass always scores a batch. Here the batch
  size is q = 1.
- `bundles` is `None` at evaluation. During training it holds the complete bundles, so the model can remove each
  bundle's own contribution from the counts (§11).
- `cache` holds precomputed parts that don't depend on the query (§10).
- Each query is deduplicated and sorted: `[2708, 1912]` becomes `tensor([1912, 2708])`. Sorting doesn't change any
  score, because every term treats the seeds as a set.

```python
    parts=([self.content_scores(seeds,bundles,cache)]
           +[self.label_scores(i,seeds,bundles) for i in range(self.levels)]
           +[self.graph_scores(seeds,bundles,cache)])

    w=self.gate_weights(seeds,bundles)
    return sum(w[:,j,None]*v for j,v in enumerate(parts))
```

`parts` is a list of three q × n matrices: content, labels (Electronic has one label level, the category), and graph.
`w` is q × 3. The last line multiplies each matrix by its column of weights and adds them. The next four sections
open each call.

## 5. Term 1: content (`content_scores`)

**Question it answers:** "Given what the seeds look like, and what *their* usual partners look like, does c look like
it belongs?"

### 5.1 The linear algebra

Let z̄ be the mean over the seeds. Then

```
query(s) = A_z z_s + A_h h_s           (64 numbers)
φ(S)     = mean over s in S of query(s)
key(c)   = B_z z_c + B_h h_c           (64 numbers)
C(S, c)  = φ(S)ᵀ key(c)
```

Expanding the product, C is a **bilinear form** in the features:

```
C(S, c) = z̄_Sᵀ (A_zᵀ B_z) z_c  +  z̄_Sᵀ (A_zᵀ B_h) h_c  +  h̄_Sᵀ (A_hᵀ B_z) z_c  +  h̄_Sᵀ (A_hᵀ B_h) h_c
```

Each of the four 128 × 128 matrices has rank at most 64. So C learns *which directions of seed features go with which
directions of candidate features*, for example "optical-drive words" with "hard-drive words". It isn't a similarity:
the matrices aren't symmetric, so complements can score high even when their titles share no words.

The two inputs per item are:

- **z, the refined features:** Z = normalize(X + MLP(X)).
- **h, the partner history:** H = N X / deg, the count-weighted average of the features of the item's training
  partners. An item is described both by its own title and by the titles it usually appears with.

### 5.2 Building Z and H (used by every query)

```python
def content(self):
    return F.normalize(self.features+self.encoder(self.features),dim=1)

def history(self):
    return self.history_sum/self.history_mass.clamp_min(1)[:,None]*(self.history_mass>0)[:,None]
```

`content()` runs the small encoder MLP on every item's features, adds the result to the original features
(a *residual* connection), and rescales each row to length 1. The encoder's last layer starts at zero, so at
initialization Z = X exactly. After training it has moved a lot. For our seeds, ‖MLP(x)‖ is 1.30 and 1.49, larger
than ‖x‖ = 1. The cosine between x and z is 0.60 for 1912 and 0.62 for 2708: z is still related to the title
features, but quite different from them.

`history()` divides each row of N X by deg, giving the weighted mean of the partners' features. `clamp_min(1)` avoids
dividing by zero. The final multiplication by `(history_mass>0)` sets the row to zero for items with no partners: an
item never seen in a training bundle has no history. For our seeds, ‖h‖ = 0.44 and 0.46. These are averages of
several unit vectors pointing in different directions, so they are shorter than 1.

```python
def keys(self,z=None):
    z=self.content() if z is None else z
    return z@self.key_content.T+self.history()@self.key_history.T
```

`keys` computes key(c) = B_z z_c + B_h h_c **for all 3,499 items at once**, as a 3499 × 64 matrix:
`Z @ B_zᵀ + H @ B_hᵀ`.

### 5.3 Scoring one query

```python
def content_scores(self,seeds,bundles,cache=None):
    if cache is None:
        z=self.content()
        k=self.keys(z)
        h=self.history()
    else:
        z,k,h=cache['content'],cache['keys'],cache['history']
```

At evaluation, Z, H and the keys come from the cache (§10). They are the same tensors, just computed once.

```python
    phi=[]
    fixes=[]
    for i,s in enumerate(seeds):
        hs=h[s]
        if bundles is not None:
            ...                     # training only, §11
        phi.append((z[s]@self.query_content.T+hs@self.query_history.T).mean(0))
```

For query i with seeds `s = tensor([1912, 2708])`:

- `z[s]` is 2 × 128, the two seeds' rows of Z. `z[s] @ A_zᵀ` is 2 × 64. Its rows have length 2.70 and 2.70.
- `hs @ A_hᵀ` is 2 × 64. Its rows have length 0.22 each: the seeds' partner histories contribute about a tenth of
  what their own features contribute.
- The two rows are added and `.mean(0)` averages over the seeds: φ(S), 64 numbers, ‖φ‖ = 2.68.

```python
    phi=torch.stack(phi)
    scores=phi@k.T
```

`torch.stack` turns the list into a q × 64 matrix (here 1 × 64). `phi @ k.T` is (1 × 64)(64 × 3499) = 1 × 3499:
**one matrix product scores the whole catalogue.** The `if fixes:` block after it runs only in training (§11).

### 5.4 The numbers

| Item | φᵀ B_z z_c (own title) | φᵀ B_h h_c (partners' titles) | C |
|---|---|---|---|
| **943 hard drive** | 5.320 | −0.137 | **5.182** |
| 717 power supply | 4.457 | −0.133 | 4.324 |
| 2775 memory | 4.265 | −0.104 | 4.161 |
| 2982 processor | 3.589 | −0.130 | 3.459 |

Over all items, C has mean 2.20 and standard deviation 1.32. Its largest value (6.06) is the seed 2708 itself; seeds
are removed only at ranking time (§9). Counting only non-seeds, **content alone puts the hard drive at rank 27**.
Content scores the target above the three competitors, but 26 other items score higher still.

## 6. Term 2: labels (`label_scores`)

**Question it answers:** "Does c's category go with the seeds' categories?" Two parts:

- **Low rank:** *which categories go with which*, for example "DVD drives go with hard drives".
- **Diagonal:** a learned bonus (or penalty) when c's category is *the same as* a seed's category.

**Why "low rank" and "diagonal".** Picture a 479 × 479 table R whose entry R_gg' says how well a seed of category g
supports a candidate of category g'. DIALS writes it as R = P M Qᵀ + Δ. The first matrix has rank at most 16. The
second sits only on the main diagonal, the entries R_gg where the seed's category and the candidate's category are the
same. That is the "same category" case, so the bonus δ fills the diagonal. The two parts do different jobs. A rank-16
matrix cannot store 479 separate same-category effects, since it is built from just 16 directions. The diagonal can,
one entry per category. One difference from a fixed matrix: an entry of Δ is not a stored number. δ computes it from
m, |S| and the category size, so it changes from query to query. The name of the model (diagonal-plus-low-rank) comes
from this split.

```
L(S, c) = (t(S)/|S|) · P M Qᵀ [:, g(c)]  +  δ(log(1+m), log|S|, log(1+n_g(c))) · [m > 0]
```

### 6.1 Where P and Q come from (built before training, not trained)

They are computed once by `cross_fitted_labels` → `label_factors` and passed to the model when it is built:

1. **Γ** (n × K) is the one-hot matrix "item c has category g". **Γᵀ N Γ** (K × K, here 479 × 479) counts how often
   category g appears in a training bundle with category g'.
2. Each row and column is divided by the square root of its total: D^-½ Γᵀ N Γ D^-½. This stops big categories from
   dominating.

   **D** is the 479 × 479 diagonal matrix of the row sums of N̄ = Γᵀ N Γ, floored at 1:
   D = diag(max(N̄ 1, 1)), so D_g = Σ_g' N̄_gg'. N is symmetric, so N̄ is too, and row sums equal column sums. The
   same D therefore scales both sides. Since N̄ 1 = Γᵀ N Γ 1 = Γᵀ N 1 = Γᵀ deg (each row of Γ has a single 1), D_g is
   the total co-occurrence degree of category g: the sum of deg_c over the items c in g. In the code
   (`label_factors`) it is `scale = sqrt(max(row sums, 1))`, and the matrix is `diags(1/scale) @ N̄ @ diags(1/scale)`.

   On Electronic:

   | Category | Items | D_g |
   |---|---|---|
   | 2, Internal Hard Drives | 32 | 220 |
   | 131, Internal DVD Drives | 7 | 37 |
   | 48, External DVD Drives | 8 | 31 |

   Across categories D_g runs from 1 to 514, with a median of 14. 32 categories have no training pairs at all, and
   the floor keeps them at 1 instead of dividing by zero. Entry by entry, N̂_gg' = N̄_gg' / √(D_g D_g'). For example,
   N̄[131, 2] = 3 pairs becomes 3 / √(37 · 220) = 0.033, and N̄[48, 2] = 1 pair becomes 1 / √(31 · 220) = 0.012.
   Without the scaling, the large categories would set every direction of the SVD.
3. A rank-16 truncated SVD gives U Σ Vᵀ. Then P = U Σ^½ and Q = V Σ^½ (each 479 × 16), so P Qᵀ ≈ the normalized
   category co-occurrence matrix.

Each category thus becomes 16 numbers on the left (P, "as a seed") and 16 on the right (Q, "as a candidate"). Only the
16 × 16 matrix **M** between them is trained. It starts at zero, so the low-rank part starts switched off. In the
trained model ‖M‖ = 12.2 and the mean of its diagonal is 1.10.

The evaluation forward pass uses `label_left0` and `label_right0`, computed from all training bundles. Training uses
per-fold versions (§11).

### 6.2 The code, step by step

```python
labels=getattr(self,f'label{i}')   # g(c) for every item, length 3499
q=len(seeds)
size=int(labels.max())+1           # K = 479
out=torch.zeros(q,self.n)

tally=torch.zeros(q,size)
for j,s in enumerate(seeds):
    tally[j]=torch.bincount(labels[s],minlength=size).float()
```

**`tally` is t(S): how many seeds carry each category.** `labels[s]` is `[131, 48]` (the seeds' categories), and
`bincount` turns it into a length-479 vector that is 1 at positions 48 and 131 and 0 elsewhere.

```python
mean=tally/tally.sum(1,keepdim=True)
mix=self.mixing[i]
```

**`mean` is t(S)/|S|, the seeds' category mix:** 0.5 at 48, 0.5 at 131.

```python
if bundles is None:
    out=out+(mean@getattr(self,f'label_left{i}')@mix)@getattr(self,f'label_right{i}')[labels].T
```

Read it from left to right:

- `mean @ P`: (1 × 479)(479 × 16) → 1 × 16. This averages the P rows of categories 48 and 131. Those two rows are
  nearly parallel (first four entries: 0.029, 0.010, 0.001, −0.033 for 48; 0.031, 0.011, 0.001, −0.036 for 131). The
  two DVD-drive categories co-occur with the same other categories in the training bundles, so the SVD gives them
  similar rows.
- `@ M`: (1 × 16)(16 × 16) → 1 × 16, the query vector in "candidate" coordinates. Its length is 0.21.
- `Q[labels]`: this step turns 479 category rows into 3,499 item rows. Q itself is 479 × 16, one row per
  **category**. `labels` is a length-3,499 vector holding each **item's** category number: `labels[943] = 2`,
  `labels[717] = 29`, `labels[2775] = 130`. Indexing a matrix with a vector of row numbers (PyTorch "advanced
  indexing") returns one row per entry of the vector:

  ```
  Q[labels][c] = Q[labels[c]]      for c = 0 … 3498
  Q[labels][943] = Q[2]            (the hard drive gets the "Internal Hard Drives" row: 0.0765, 0.0264, 0.0023, …)
  ```

  So `Q[labels]` is 3499 × 16. Row c is the Q row of item c's category, and items in the same category get identical
  copies. In matrix terms it is Γ Q, where Γ is the 3499 × 479 one-hot item-to-category matrix of §6.1. Indexing gives
  the same result without building Γ. `.T` then makes it 16 × 3499.
- The product is (1 × 16)(16 × 3499) → 1 × 3499: the low-rank score of every item. Since items in the same category
  share a Q row, **all items in the same category get the same low-rank score.**

On this query the low-rank scores run from −0.057 to 0.104. They are small next to content (≈ 5) and graph (≈ 1.7).

```python
m=tally[:,labels]
row,col=m.nonzero(as_tuple=True)
```

**`m[j, c]` is the number of seeds that share item c's category.** This is the same indexing trick as `Q[labels]`,
applied to columns. `tally` is q × 479, one column per category. `tally[:, labels]` picks, for each of the 3,499
items, the column of its category, so `m` is q × 3499 and `m[j, c] = tally[j, labels[c]]`. For the hard drive,
`m[0, 943] = tally[0, 2] = 0`, because no seed is in category 2. Here m = 1 for the 15 items in category 48 or 131 (including the two seeds) and 0 for all
others. `nonzero` lists those 15 (query, item) positions. The diagonal is evaluated only there.

```python
if len(row):
    count=torch.tensor([float(len(s)) for s in seeds]).log()
    x=torch.stack((m[row,col].log1p(),
                   count[row],
                   getattr(self,f'label_size{i}')[labels[col]]),
                  1)
    out=out.index_put((row,col),self.diagonal[i](x).squeeze(1),accumulate=True)
return out
```

For each of the 15 positions, three inputs are built: log(1 + m), log|S| and log(1 + size of that category).
The category size uses the same indexing twice: `labels[col]` gives the category of each listed item, and
`label_size0[...]` (length 479, one entry per category) looks up that category's log(1 + n_g). The small MLP δ maps them to one number, and `index_put(..., accumulate=True)` **adds** it to the low-rank score at that
position.

What the trained δ learned (with |S| = 2):

| m (seeds sharing the category) | category size 1 | 5 | 20 | 50 |
|---|---|---|---|---|
| 1 | +0.85 | +0.50 | +0.01 | −0.16 |
| 2 | +0.92 | +0.60 | +0.09 | −0.10 |

Sharing a **small** category is strong evidence. Sharing a **large** category is worth nothing or slightly
negative, because a big category like "cases" gives no specific hint. The sign is learned, not assumed.

### 6.3 The numbers

| Item | Shares a seed's category? | Low rank | Diagonal | L |
|---|---|---|---|---|
| **943 hard drive** (label 2) | no | −0.0048 | 0 (m = 0, not evaluated) | **−0.0048** |
| 717 power supply (label 29) | no | −0.0098 | 0 (m = 0, not evaluated) | −0.0098 |
| 2775 memory (label 130) | no | −0.0076 | 0 (m = 0, not evaluated) | −0.0076 |
| 2982 processor (label 115) | no | −0.0037 | 0 (m = 0, not evaluated) | −0.0037 |
| 1697 Sony slim DVD drive (label 48) | yes, m = 1, size 8 | −0.0015 | +0.3151 | 0.3136 |
| 3317 LG DVD rewriter (label 131) | yes, m = 1, size 7 | −0.0025 | +0.3679 | 0.3653 |

Every item gets a low-rank score, because every item has a category. Only items whose category matches a seed's
get a diagonal score. For the other 3,484 items, m = 0 and δ is never evaluated, so that part is exactly 0. All
items in category 48 get the same L of 0.3136 (the seed 2708 included), and the same holds for category 131 at 0.3653.
The size-8 category earns a slightly smaller bonus than the size-7 one, as the δ table above predicts.

On its own, the label term would rank the hard drive **1,340th**: it prefers more DVD drives. In this bundle the
label term is close to silent. It moves the four items we follow by less than 0.02 after weighting, and the decision
is made by the other two terms. This is typical on Electronic, where the bundles mix categories.

## 7. Term 3: graph (`graph_scores`)

**Question it answers:** "How often has c been bundled with the seeds, after discounting items that are bundled with
everything?"

```
G(S, c) = θ · log(1 + Σ_{s ∈ S} N_sc · deg_s^-α · deg_c^-β)  +  ρ(log(1 + f_c))
```

In matrix terms, the inner sum is a row vector: 1_Sᵀ D^-α N D^-β, where 1_S is the indicator of the seeds and D is
the diagonal matrix of degrees. The code never builds that matrix. It touches only the stored partners of the seeds,
so the cost is the number of partners (here 16), not 3,499.

### 7.1 Finding the seeds' partners: `expand`

[SPARSE_GRAPH.md](SPARSE_GRAPH.md) walks through this whole section on a 5-item toy example.

```python
alpha,beta=self.exponents
q=len(seeds)

query,source,edge=self.expand(torch.cat([torch.full((len(s),),i) for i,s in enumerate(seeds)]),
                              torch.cat(seeds))
```

The first argument says which query each seed belongs to (here `[0, 0]`). The second lists all seeds in the batch
(`[1912, 2708]`).

```python
def expand(self,query,rows):
    start=self.pointer[rows]
    length=self.pointer[rows+1]-start
    offsets=torch.cumsum(length,0)-length
    edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
    return torch.repeat_interleave(query,length),torch.repeat_interleave(rows,length),edge
```

**What it is for.** In matrix terms, the graph term needs the rows N[1912, :] and N[2708, :] of the item-item count
matrix. N is stored in CSR form (§3): one long list `target`/`counts` of every stored pair, sorted by source item, with
`pointer` marking where each item's block begins. Row s is the slice `pointer[s] : pointer[s+1]` of that list. An
**edge** is a position in that list, so one edge is one nonzero N_sc. `expand` returns the positions of all nonzeros in
the seed rows, laid end to end:

```
edge = [7878, 7879, …, 7886,  11021, …, 11027]      (= arange(7878, 7887) followed by arange(11021, 11028))
```

A Python loop over seeds would build exactly this. `expand` builds it in one vector operation, which matters in
training, where a batch holds hundreds of seeds.

**How it does that, and what `offsets` is.** Number the output slots k = 0 … 15. Seed j owns a run of `length[j]`
consecutive slots, and **`offsets[j]` is the slot where that run begins**: the total length of all earlier runs, i.e.
an exclusive running sum (`cumsum(length) - length`). Inside seed j's run, slot k must hold

```
edge[k] = start[j] + (k - offsets[j])          (k - offsets[j] is the position within seed j's block)
        = (start[j] - offsets[j]) + k
```

The code computes the two terms of the last line separately. `repeat_interleave(start - offsets, length)` writes
`start[j] - offsets[j]` into each of seed j's slots, and `arange(16)` supplies k.

With our numbers:

| Variable | Value | Meaning |
|---|---|---|
| `start` | [7878, 11021] | first stored pair of each seed |
| `length` | [9, 7] | number of partners of each seed |
| `offsets` | [0, 9] | where each seed's block begins in the output |
| `start - offsets` | [7878, 11012] | repeated 9 and 7 times, then 0..15 is added |
| `edge` | 7878…7886, 11021…11027 | the 16 stored-pair indices |

Check slot by slot. Slots 0–8 give 7878 + k = 7878 … 7886. Slots 9–15 give 11012 + k = 11021 … 11027. Without
subtracting `offsets`, slot 9 would be 11021 + 9 = 11030, which lies past the end of seed 2708's block.

The outputs are three length-16 vectors: the query of each pair (all 0), its seed (1912 × 9, 2708 × 7) and its index
into `target`/`counts`. This is a loop-free way of writing "for each seed, for each partner".

```python
target=self.target[edge]    # c:  [7, 717, 943, 2124, 2230, 2708, 2775, 2982, 3292, 7, 717, 943, 1912, 2230, 2775, 2982]
n=self.counts[edge]         # N_sc
ds=self.log_degree[source]  # log deg_s
dt=self.log_degree[target]  # log deg_c
```

The `if bundles is not None:` block that follows runs only in training (§11).

### 7.2 Weighting and summing

```python
w=n*torch.exp(-alpha*ds-beta*dt)
first=torch.zeros(q*self.n).scatter_add(0,query*self.n+target,w).view(q,self.n)
out=self.theta*first.log1p()
```

- `w` is N_sc · deg_s^-α · deg_c^-β, written as exp(−α log deg_s − β log deg_c).
- `scatter_add` adds each pair's `w` into position (query, c) of a flat q·n vector. `query*self.n+target` is the flat
  index. If two seeds share a partner c, their two weights land in the same cell and are summed. That is the
  Σ over s ∈ S.
- `.view(q, n)` reshapes it to q × n. Only 10 items have a non-zero value here: the distinct partners of the two seeds.
- `log1p` damps large sums, and θ scales the result.

For the hard drive 943 (deg = 52, log deg = 3.951):

```
from seed 1912 (deg 13, log 2.565):  2 · exp(−0.0359·2.565 − 0.1748·3.951) = 2 · 0.457 = 0.914
from seed 2708 (deg 7,  log 1.946):  1 · exp(−0.0359·1.946 − 0.1748·3.951) = 1 · 0.467 = 0.467
sum = 1.382,   θ · log(1 + 1.382) = 1.8922 · 0.868 = 1.642
```

The learned α = 0.036 is close to 0: the seed's degree hardly matters. β = 0.175 discounts popular candidates
mildly. A candidate with deg 52 keeps exp(−0.175 · 3.95) = 50% of its count; one with deg 11 keeps 66%.

### 7.3 The popularity prior

```python
prior=(self.prior(self.frequency.log1p()[:,None]).squeeze(1) if cache is None else cache['prior']).expand(q,-1)
if bundles is not None:
    ...                         # training only, §11
return out+prior
```

ρ is a small MLP applied to log(1 + f_c). It is the same for every query, and `expand(q, -1)` copies it to every row
without copying memory. It is added to every item, including items that never met the seeds. What the trained ρ
learned:

| f_c | 0 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| ρ | +0.07 | −0.38 | −0.37 | −0.15 | +0.04 | +0.20 |

It isn't monotone. Items seen in 1 or 2 bundles are pushed down and frequent items are pushed up. Never-seen items
get a small positive value, and the gate (§8) has a separate input for them.

### 7.4 The numbers

| Item | Σ weights | θ log(1 + Σ) | ρ(f) | G |
|---|---|---|---|---|
| **943 hard drive** (deg 52, f 11) | 1.382 | 1.642 | +0.064 | **1.706** |
| 717 power supply (deg 46, f 8) | 1.412 | 1.666 | −0.016 | 1.650 |
| 2775 memory (deg 11, f 2) | 1.813 | 1.957 | −0.369 | 1.588 |
| 2982 processor (deg 11, f 2) | 1.813 | 1.957 | −0.369 | 1.588 |

The memory and the processor have the largest discounted counts, because they are less popular and β discounts them
less. The prior then pulls them below the hard drive, because items seen in only 2 bundles are less trusted. **Graph
alone ranks the hard drive 1st**, but only 0.056 ahead of the power supply.

## 8. The gate (`gate_weights`)

**Question it answers:** "For *these* seeds, how much should each term count?"

```python
def gate_weights(self,seeds,bundles):
    stats=[]
    for s in seeds:
        f=(self.frequency[s]-(1 if bundles is not None else 0)).clamp_min(0)
        stats.append(torch.stack((torch.tensor(float(len(s))).log(),
                                  f.log1p().mean(),
                                  (f==0).float().mean())))
    return F.softplus(self.gate(torch.stack(stats)))
```

Three numbers describe the query:

| Input | Formula | Our query |
|---|---|---|
| size | log\|S\| | log 2 = 0.693 |
| familiarity | mean over seeds of log(1 + f_s) | (log 5 + log 2)/2 = 1.151 |
| novelty | share of seeds with f_s = 0 | 0 |

The gate is an MLP 3 → 16 → 3, and `softplus` makes the three outputs positive. The last layer starts with zero
weights and a bias of log(e − 1), so at initialization every weight is exactly softplus(log(e − 1)) = 1.

The trained gate gives this query:

```
w_C = 1.228     w_L = 1.616     w_G = 4.441
```

The seeds are known items (f = 4 and 1), so the gate trusts the graph most, about 3.6 times as much as content.
For seeds never seen in training, the third input would be 1, the graph term would be nearly empty, and the gate
would shift weight to content.

## 9. Adding up, and ranking

```python
return sum(w[:,j,None]*v for j,v in enumerate(parts))
```

`w[:, j, None]` is q × 1, so it multiplies every column of the q × n matrix `v`. The result is q × n. For the items we
follow:

| Item | w_C · C | w_L · L | w_G · G | **Total** | Rank |
|---|---|---|---|---|---|
| **943 hard drive** | 6.364 | −0.008 | 7.578 | **13.934** | **1** |
| 717 power supply | 5.310 | −0.016 | 7.326 | 12.620 | 2 |
| 2775 memory | 5.110 | −0.012 | 7.051 | 12.149 | 3 |
| 2982 processor | 4.247 | −0.006 | 7.051 | 11.293 | 4 |

The 5th item is 2230, a Rosewill power supply, at 10.796.

**Why the target wins:**

- On the graph, the hard drive is only 0.25 ahead of the power supply after weighting.
- On content, it is 1.05 ahead: its title fits "optical drives" better than a power supply's does.

Neither term alone settles it as clearly. Content alone ranks the target 27th and graph alone ranks it 1st by a thin
margin. Together, the gap to rank 2 is 1.31.

**Ranking** happens outside the model, in `evaluate.top_items`:

```python
values=np.asarray(scores).copy()
values[list(seeds)]=-np.inf
```

The seeds are set to −∞ so they can never be recommended. This matters: content's highest score belonged to the
seed 2708. The top 20 are then taken, with exact ties broken by lower item ID. The top 20 for our query start
`943, 717, 2775, 2982, 2230, 7, 2124, …`. The target is first, so Recall@20 = 1 and NDCG@20 = 1.

## 10. Many bundles at once, and the evaluation cache

**Batches.** Evaluation calls `model(seeds[start:start+32], cache=cache)`, so q = 32 queries go through together.
Nothing in the walk-through changes:

- φ becomes 32 × 64, and `phi @ k.T` gives 32 × 3499.
- `tally` and `mean` become 32 × 479.
- `expand` tags every pair with its query, and `scatter_add` writes into row `query` of the 32 × 3499 matrix.
- The gate gets 32 × 3 inputs and returns 32 × 3 weights, so every query has its own weights.

Each query's scores depend only on its own seeds. Batching changes the speed, not the numbers.

**The cache.** Several pieces don't depend on the query at all: Z, H, the keys and ρ(f). `evaluate.score_batches`
computes them once with `model.cache()` and reuses them for every batch:

```python
def cache(self):
    z=self.content()
    h=self.history()
    return dict(content=z,                                       # Z, 3499 × 128
                history=h,                                       # H, 3499 × 128
                keys=z@self.key_content.T+h@self.key_history.T,  # 3499 × 64
                prior=self.prior(self.frequency.log1p()[:,None]).squeeze(1))  # 3499
```

These are the same formulas as without the cache, so the scores are identical. The cache is refused in training
(the `ValueError` in §4), because in training the parameters change after every batch and the pieces would go stale.

## 11. The same forward pass during training

During training, `train.fit` calls `model(seeds, bundles=bundles)`. Here `bundles` are the complete training
bundles, and `seeds` are part of each bundle (the rest are the targets).

**The problem.** That bundle is itself in the training data. Its pairs were counted in N, its items' frequencies
include it, and its items' partner histories include each other. If nothing were done, the model would learn "items
that were counted with the seeds score high" from counts that contain the answer. At test time the answer is never
in the counts, so it would learn the wrong lesson.

**The fix.** Every place that reads training counts subtracts this bundle's own contribution (leave-one-out). The
label factors P and Q can't be corrected by subtraction (they come from an SVD), so they are swapped for a version
computed without this bundle's fold (cross-fitting).

To show the effect, take the training bundle [2982, 2775, 717, 943, 1912] (fold 0), with seeds [2982, 2775] and
targets 717, 943, 1912. (The real training loop draws the split at random; we fixed one for the example.) The same
seeds are scored twice: in training mode with the bundle given, and in evaluation mode without it.

### 11.1 Content: leave-one-out partner histories

```python
if bundles is not None:
    b=bundles[i]
    hs=self.loo_history(s,b)
    fixes.append((i,b,(self.loo_history(b,b)-h[b])@self.key_history.T))
```

```python
def loo_history(self,items,bundle):
    mass=self.history_mass[items]-(len(bundle)-1)
    value=(self.history_sum[items]-(self.features[bundle].sum(0)-self.features[items]))/mass.clamp_min(1)[:,None]
    return value*(mass>0)[:,None]
```

This bundle made each of its items a partner of the other |b| − 1 = 4 members. So:

- 4 is taken off the item's degree.
- The other members' features (the sum over the bundle minus the item's own row) are taken off its N X row.

The seeds' histories `hs` are recomputed this way. The members are also candidates, so their keys change too. `fixes`
stores the change of their keys, and after `phi @ k.T` the code adds φᵀ(change) at exactly those (query, member)
cells:

```python
if fixes:
    rows=torch.cat([torch.full((len(b),),i) for i,b,_ in fixes])
    cols=torch.cat([b for _,b,_ in fixes])
    scores=scores.index_put((rows,cols),
                            torch.cat([delta@phi[i] for i,_,delta in fixes]),
                            accumulate=True)
```

The effect is small here: for the members, C moves by at most 0.02 (for example 943: 3.893 → 3.902), because
histories carry little weight in this trained model (§5.3).

### 11.2 Labels: the fold's factors

```python
fold=torch.tensor([fold_of(b.tolist()) for b in bundles])
for k in fold.unique().tolist():
    rows=(fold==k).nonzero(as_tuple=True)[0]
    part=(mean[rows]@getattr(self,f'label_fold_left{i}')[k]@mix)@getattr(self,f'label_fold_right{i}')[k][labels].T
    out=out.index_put((rows,),out[rows]+part)
```

`fold_of` hashes the bundle's sorted items to a number 0–4 (identical bundles always get the same fold). Queries
are grouped by fold, and each group uses P^(k), Q^(k), computed from the bundles *outside* fold k. Our bundle is in
fold 0, so its category pairs never entered the factors it is scored with. The diagonal part uses only the seeds'
own categories, so it needs no correction.

Because the factors differ, L changes for every item, not only the members: up to 0.38 here. For the members:
943 goes from −0.018 to +0.035, and 2775 from −0.207 to −0.141.

### 11.3 Graph: subtracting the bundle's pairs

```python
if bundles is not None:
    member=torch.zeros(q,self.n,dtype=torch.bool)
    for i,b in enumerate(bundles):
        member[i,b]=True
    inside=member[query,target].float()
    n=n-inside
    others=torch.tensor([float(len(b)-1) for b in bundles])[query]
    ds=(self.history_mass[source]-others).clamp_min(1).log()
    dt=(self.history_mass[target]-inside*others).clamp_min(1).log()
```

- `member[i, c]` marks the items of query i's bundle.
- `inside` is 1 for each (seed, partner) pair where the partner is also in the bundle. Those pairs were counted once
  by this bundle, so `n - inside` removes it.
- Every seed is a bundle member, so its degree loses |b| − 1. A partner's degree loses |b| − 1 only when the partner
  is in the bundle too (`inside*others`).

Then the prior:

```python
rows=torch.cat([torch.full((len(b),),i) for i,b in enumerate(bundles)])
cols=torch.cat(bundles)
left=(self.frequency[cols]-1).clamp_min(0)
prior=prior.index_put((rows,cols),
                      self.prior(left.log1p()[:,None]).squeeze(1)-prior[rows,cols],
                      accumulate=True)
```

For the members, ρ is re-evaluated at f − 1 and the difference is added, so their prior is ρ(f − 1).

The effect is large, and it is the reason for all of this. N for the pair (2775, 943) is 2, but one of those two
bundles is this one, so the honest count is 1. For the targets, G drops:

| Member | G, evaluation mode (answer counted) | G, training mode (answer removed) |
|---|---|---|
| 717 | 1.985 | 1.231 |
| 943 | 2.039 | 1.303 |
| 1912 | 2.071 | 1.261 |

Without the correction, every target would get about 0.75 extra graph score from its own bundle. The model would
learn to over-trust the graph.

### 11.4 Gate: f − 1

`gate_weights` subtracts 1 from each seed's frequency in training (`1 if bundles is not None else 0`), because the
bundle being scored is one of the f bundles. The seeds 2982 and 2775 have f = 2, so the familiarity input goes from
log 3 to log 2. The weights change from (1.24, 1.64, 4.60) in evaluation mode to (1.35, 1.90, 5.90) in training mode.

All three terms and the gate see training counts **as they would look if this bundle had never been seen**. That is
exactly the situation of a test bundle.

## 12. Summary of shapes

For a batch of q queries on Electronic (n = 3,499, d = 128, r = 64, K = 479, k = 16, P = number of seed–partner
pairs in the batch):

| Step | Code | Shape |
|---|---|---|
| refined features Z | `content()` | n × d |
| partner histories H | `history()` | n × d |
| keys | `keys()` | n × r |
| seed queries φ | `phi` | q × r |
| content scores C | `phi @ k.T` | q × n |
| category tally t(S) | `tally` | q × K |
| category mix | `mean` | q × K |
| label query | `mean @ P @ M` | q × k |
| candidate label factors | `Q[labels]` | n × k |
| label scores L | `out` | q × n |
| seed–partner pairs | `expand(...)` | P (here 16) |
| discounted counts | `first` | q × n |
| prior ρ(f) | `prior` | n, broadcast to q × n |
| graph scores G | `out + prior` | q × n |
| gate inputs | `stats` | q × 3 |
| gate weights | `w` | q × 3 |
| final scores | `forward(...)` | q × n |

To reproduce the numbers in this document, load the model with `dials.train.load(ds, 'runs/electronic/seed0',
'cache')` and call each method above on `[tensor([1912, 2708])]`.
