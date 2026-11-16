# The sparse graph computation

Code: `pair_counts`, `DIALS.expand` and `DIALS.graph_scores` in `dials/model.py`. Every number below was printed by
running that code.

## 0. The goal, and the plan

The graph term asks: *how often has candidate c been bundled with the seeds?* For query j with seed set S_j:

```
G(S_j, c) = θ · log(1 + A[j, c]) + ρ(log(1 + f_c)),     A[j, c] = Σ_{s ∈ S_j} N_sc · deg_s^-α · deg_c^-β
```

In matrix form, A[j, :] = 1_{S_j}ᵀ W with W_sc = N_sc deg_s^-α deg_c^-β. This is the sum of the seed rows of W.

**The obstacle.** W is n × n. On Spotify that is 254,155² = 64.6 B entries, or 258 GB as float32. But 99.9% of them
are 0, because most item pairs never share a bundle:

| | n | Dense entries | Non-zero | Mean non-zeros per row |
|---|---|---|---|---|
| Electronic | 3,499 | 12.2 M | 13,788 | 3.9 |
| Spotify | 254,155 | 64.6 B | 67.2 M | 264 |

A zero entry adds nothing to A. So the only work needed is: **for each query, visit the non-zeros in its seeds' rows,
and add each one into the right cell of A.** The plan has three steps, one per section:

1. **Store** only the non-zeros, arranged so that one item's row can be read as a single contiguous slice (§2).
2. **Collect** the non-zeros of all seed rows in the batch into flat lists. Each entry gets a tag saying which query
   and which candidate cell it belongs to (§3).
3. **Weight** each entry and **add** it into its (query, candidate) cell (§4).

§5 shows how training reuses the same lists to remove a bundle's own pairs. §6 runs the real example.

**Toy data used throughout:** n = 5 items and three training bundles.

```
b1 = [0, 1, 2]     b2 = [1, 2, 3]     b3 = [2, 4]

        c=0  1  2  3  4
N = s=0 [ 0  1  1  0  0 ]      N_sc = number of bundles containing both s and c
    s=1 [ 1  0  2  1  0 ]      (N_12 = 2: items 1 and 2 meet in b1 and b2)
    s=2 [ 1  2  0  1  1 ]
    s=3 [ 0  1  1  0  0 ]
    s=4 [ 0  0  1  0  0 ]

deg = N 1 = [2, 4, 5, 2, 1]    f = bundles containing each item = [1, 2, 3, 1, 1]
```

## 1. (Background) What a query batch looks like

Training processes 32 training bundles per step, each with its own seed set. Evaluation also scores 32 test
queries at a time (`evaluate.score_batches`). So the input is a list of seed sets, not one seed set. The toy batch has two queries:

```
query 0: seeds {1, 3}          query 1: seeds {4}
```

The result A is therefore a q × n matrix (here 2 × 5), with one row per query.

## 2. Store: keep only the non-zeros, grouped by row (`pair_counts`)

**Why.** Step 3 has to read "all non-zeros of row s" for every seed s. If the non-zeros are sorted by s, each row is
one contiguous block, and reading it is a slice. No search is needed.

**How.** For every bundle, list each ordered pair (s, c) with s ≠ c. Count identical pairs, which gives N_sc, and
sort by s, then c:

```
entry e   0  1  2  3  4  5  6  7  8  9 10 11
source    0  0  1  1  1  2  2  2  2  3  3  4
target    1  2  0  2  3  0  1  3  4  1  2  2       target[e] = the partner c
counts    1  1  1  2  1  1  2  1  1  1  1  1       counts[e] = N_sc
```

Each entry is called an **edge**: one non-zero N_sc. Both directions are stored, (1, 2) and (2, 1), so row s holds all
of s's partners.

**Finding a row's block.** `pointer[s]` is the first edge of row s, and `pointer[s+1]` is one past its last:

```
pointer = [0, 2, 5, 9, 11, 12]

row 1 = edges pointer[1] .. pointer[2]−1 = 2, 3, 4  → partners 0, 2, 3 with counts 1, 2, 1
row 3 = edges 9, 10                                 → partners 1, 2    with counts 1, 1
```

This is the standard CSR (compressed sparse row) layout. The model keeps `pointer`, `target`, `counts` and deg (as
`history_mass`). The `source` array is not kept, because `pointer` already says which row an edge belongs to.

In code, the counting is done by turning each pair into the integer key s·n + c. `np.unique(keys,
return_counts=True)` then counts and sorts in one call, and `pointer = searchsorted(source, arange(n+1))`.

## 3. Collect: gather the seeds' edges into flat lists (`expand`)

**What step 4 will need from each edge.** To add edge (s, c) into A, step 4 needs four facts about it:

| Fact | Why it is needed |
|---|---|
| its **query** j | which row of A it goes into |
| its **target** c | which column of A it goes into, and deg_c for the weight |
| its **source** s | deg_s for the weight (and, in training, the leave-one-out fix of §5) |
| its **count** N_sc | the value being added |

So the goal of `expand` is a table with one line per edge of every seed in the batch, holding those four facts.

**Why not a Python loop.** "For each query, for each seed, for each partner" works, but it is slow in Python, and the
batch holds many seeds. `expand` builds the same table with a few vector operations.

### 3.1 The edge list

Two lists are involved. Keep them apart:

- **The storage** (§2): all 12 edges of N, numbered e = 0..11. Each seed's edges sit together in one block.
- **The output**: the new list being built, holding only the seeds' edges. Its positions are numbered k = 0, 1, 2, …

The batch's seeds are 1, 3 and 4 (`rows = [1, 3, 4]`; "row" means a row of N, i.e. one seed). Their blocks in the
storage are:

```
storage e:  0  1 [ 2  3  4] 5  6  7  8 [ 9 10][11]
               seed 1's block          seed 3   seed 4
```

The output copies those blocks one after another:

```
output k:     0  1  2   3   4   5
holds edge:   2  3  4   9  10  11
from seed:    1  1  1   3   3   4
```

So the job is: **for each output position k, find which storage edge goes there.** The code knows two numbers per
seed:

```python
start=self.pointer[rows]           # [2, 9, 11]  where each seed's block begins in the storage
length=self.pointer[rows+1]-start  # [3, 2, 1]   how many edges each seed has
```

It needs one more: where each seed's block begins in the **output**. Seed 1 goes first, so it begins at k = 0. Seed 3
comes after seed 1's 3 edges, so it begins at k = 3. Seed 4 comes after 3 + 2 edges, so it begins at k = 5. That is
`offsets`, the running total of the earlier lengths:

```python
offsets=torch.cumsum(length,0)-length   # [0, 3, 5]
```

Now any output position k is easy to fill. Say k belongs to seed j's block:

```
how far into seed j's block k is:   k − offsets[j]
storage edge for position k:        start[j] + (k − offsets[j])
```

Worked through every k:

```
k   seed  start  offsets  k − offsets  edge = start + (k − offsets)
0   1     2      0        0            2
1   1     2      0        1            3
2   1     2      0        2            4
3   3     9      3        0            9
4   3     9      3        1            10
5   4     11     5        0            11
```

The code computes the same thing for all k at once. It rewrites start + (k − offsets) as (start − offsets) + k:

```python
edge=torch.repeat_interleave(start-offsets,length)+torch.arange(int(length.sum()))
#     start−offsets = [2, 6, 6], written once per output position: [2, 2, 2, 6, 6, 6]
#     arange(6)                                                  = [0, 1, 2, 3, 4, 5]
#     sum                                                        = [2, 3, 4, 9, 10, 11]
```

### 3.2 The tags

Each edge also needs its query and source, which is the repeat of `query` and `rows` by `length`. Reading
`target[edge]` and `counts[edge]` then fills in the rest of the table:

```
query    0  0  0  0  0  1      repeat_interleave([0, 0, 1], [3, 2, 1])
source   1  1  1  3  3  4      repeat_interleave([1, 3, 4], [3, 2, 1])
target   0  2  3  1  2  2      target[edge]
n        1  2  1  1  1  1      counts[edge]
```

This table holds all the work for the batch: 6 edges. Every other entry of N is zero for these seeds and is never
read. On the real Electronic query in §6 it is 16 edges, while a dense pass would read 2 × 3,499 entries.

## 4. Weight and add (`scatter_add`)

**Weight.** The weight is the formula from §0, with one value per line of the table:

```python
w=n*torch.exp(-alpha*log_degree[source]-beta*log_degree[target])      # N_sc · deg_s^-α · deg_c^-β
```

With α = β = 0.5 for illustration (trained Electronic values: α = 0.036, β = 0.175):

```
query  s→c   N  deg_s  deg_c   w
0      1→0   1  4      2       1/√8  = 0.354
0      1→2   2  4      5       2/√20 = 0.447
0      1→3   1  4      2       1/√8  = 0.354
0      3→1   1  2      4       1/√8  = 0.354
0      3→2   1  2      5       1/√10 = 0.316
1      4→2   1  1      5       1/√5  = 0.447
```

**Add.** Each w must go into cell A[query, target]. Two lines can share a cell: here (0, 2) gets 0.447 from seed 1
and 0.316 from seed 3. That sharing *is* the Σ over seeds. `scatter_add` adds values into given positions and
accumulates when positions repeat. It works on a 1-D vector, so A is held flat, with cell (j, c) at index j·n + c:

```python
A=torch.zeros(q*n).scatter_add(0,query*n+target,w).view(q,n)
```

```
index = query·5 + target = [0, 2, 3, 1, 2, 7]

A =      c=0    1      2                3      4
  q0  [ 0.354  0.354  0.447+0.316=0.763  0.354  0 ]
  q1  [ 0      0      0.447              0      0 ]
```

This equals the dense result 1_{S_j}ᵀ W. Then G = θ · log1p(A) plus the prior ρ(log(1 + f_c)).

How to read the result:
- Item 2 is a partner of both seeds, so it scores highest for query 0.
- Item 4 never met seeds 1 or 3. Its cell is 0, and it gets only the prior.
- Item 3 is a seed of query 0, yet it scores 0.354, because seed 1 is its partner. This is intended. Seeds are removed
  at ranking time (`evaluate.top_items` sets them to −∞).

## 5. Training: remove the bundle's own pairs

**Problem.** A training query is built from a training bundle b, and b's pairs are counted in N. If they stayed, the
target's score would contain the very co-occurrence the model is asked to predict, and the model would learn to read
the answer off N. At test time no such free count exists. So for each training query, N, deg and f must look as if b
had never been seen.

**Why this is cheap.** b's pairs that matter are exactly the edges between members of b. Those edges are already in
the table from §3. So the fix is a correction to the table's columns, and the stored arrays are never changed:

- **N_sc − 1** when c is in b, since b added exactly 1 to every pair inside it.
- **deg_s − (|b| − 1)** for the seed, since b gave each member |b| − 1 partners.
- **deg_c − (|b| − 1)** when c is in b.

Example: b = b2 = [1, 2, 3], with seeds {1, 3}:

```
s→c   N  c in b?  N'  deg_s'         deg_c'
1→0   1  no       1   4−2 = 2        2          (item 0 is outside b: unchanged)
1→2   2  yes      1   2              5−2 = 3
1→3   1  yes      0   2              2−2 = 0 → 1 (floored)
3→1   1  yes      0   2−2 = 0 → 1    4−2 = 2
3→2   1  yes      0   1              3
```

Check by recounting from b1 and b3 only: N_12 = 1, N_13 = 0, deg_1 = 2, deg_2 = 3 and deg_3 = 0. These match. Edges
whose count drops to 0 stay in the table with w = 0. For members of b, the prior uses f_c − 1 in place of f_c.

## 6. The real query

Electronic test query 20, with seeds 1912 and 2708 (see [FORWARD_PASS.md](FORWARD_PASS.md) §7):

```
start = [7878, 11021]   length = [9, 7]   offsets = [0, 9]   →  16 edges: 7878..7886, 11021..11027

target = [7, 717, 943, 2124, 2230, 2708, 2775, 2982, 3292,   7, 717, 943, 1912, 2230, 2775, 2982]
counts = [1, 2,   2,   1,    1,    1,    2,    2,    1,      1, 1,   1,   1,    1,    1,    1   ]
```

Six items (7, 717, 943, 2230, 2775 and 2982) are partners of both seeds, so `scatter_add` sums two weights into their
cells. The target 943 ends with G = 1.706, the highest of any non-seed item.
