# The task, how CLHE, RaMen and the simple baselines solve it, and how DIALS does

This is a companion to the guide deck (`slides/dials_guide.pdf`). It defines the task precisely, then describes each
compared method from its code, not from its abstract:
- CLHE: `models/CLHE.py` in the authors' released code;
- RaMen: `models/RaMen.py` in the authors' released code;
- the simple baselines: our own implementation, run with the shared evaluation;
- DIALS: `dials/model.py` and `dials/train.py`.

Every method is written in the same notation, so the differences are visible as differences in the formulas.

## 1. The task: bundle completion

### Data

| Symbol | Meaning |
|---|---|
| $n$ | items in the catalogue |
| $X \in \{0,1\}^{m\times n}$ | training bundles × items; $X_{bi}=1$ if item $i$ is in bundle $b$ |
| $C = X^\top X$, zero diagonal | co-occurrence counts; $C_{ij}$ is the number of training bundles holding both $i$ and $j$ |
| $f_i = \sum_b X_{bi}$ | training frequency of item $i$ |
| $Z \in \mathbb{R}^{n\times d}$ | item features: BLIP image + text on POG, CLAP audio + text on Spotify, TF-IDF of titles on Electronic/Food/Clothing and of the product hierarchy on Dunnhumby |
| $\Gamma_l \in \{0,1\}^{n\times L_l}$ | item labels at level $l$: category on POG and e-commerce; artist and album on Spotify; sub-commodity on Dunnhumby |

### A query

A held-out bundle $B$ is split into seeds $S$ (the half that is shown) and targets $T = B\setminus S$ (the half that is
hidden).
- On POG and Spotify the split is the one supplied in the released files.
- On the other datasets half of each bundle is hidden with a fixed seed.

As a matrix, a batch of $q$ queries is a seed indicator $Y \in \{0,1\}^{q\times n}$. Every method is a function
$Y \mapsto \text{scores} \in \mathbb{R}^{q\times n}$. The seeds are removed from the ranking, and the rest of the
catalogue is ranked.

### What counts as a success

The task is scored over the full catalogue, with no sampled negatives.

$$
\text{Recall@20} = \frac{|\text{top}_{20}\cap T|}{|T|},\qquad
\text{NDCG@20} = \frac{\sum_{r\le 20}[\text{item}_r\in T]/\log_2(r+1)}{\sum_{r\le \min(|T|,20)} 1/\log_2(r+1)}.
$$

- **Clean queries:** results are reported on test queries whose complete bundle appears in neither training nor
  validation.
- **Statistics:** comparisons use a paired cluster bootstrap with sign flips (identical bundles form one cluster),
  Holm-corrected within each family of tests.

### Why it is hard

The three kinds of evidence each fail somewhere:
- **History ($C$)** is empty for new or rare items. On POG, 96% of test queries have no seed that co-occurs with any
  target.
- **Content ($Z$)** measures resemblance, and resemblance is not compatibility. Two near-identical boots are
  substitutes, not a bundle.
- **Labels ($\Gamma$)** are coarse, and the right use of a shared label depends on the domain. In an outfit a second
  pair of boots is wrong; in a playlist a second song by the same artist is often right.

A method has to decide, query by query, how much to trust each kind.

## 2. The simple baselines: one item-to-item matrix

Every simple baseline is

$$
\text{scores} = Y W,\qquad W\in\mathbb{R}^{n\times n},
$$

that is, a candidate's score is the sum of its similarities to the seeds. A query is treated as a user whose
interactions are its seeds. Each method's setting is chosen on validation NDCG@20, and test is scored once.

| Method | $W$ | What it captures |
|---|---|---|
| Popularity | rank-one: every row is $\log(1+f)$ | how often an item is bought |
| Co-occurrence | $C$ | raw counts; favours hubs |
| ItemKNN | $C_{ij}/(\sqrt{f_if_j}+h)$, top-$k$ per row | cosine co-occurrence, with shrinkage $h$ |
| RP3β | $D_f^{-\alpha} X^\top D_b^{-\alpha} X D_f^{-\beta}$, zero diagonal | a 3-step random walk, popularity damped by $\beta$ |
| EASE | $I - P\,\mathrm{diag}(P)^{-1}$, $P=(X^\top X+\lambda I)^{-1}$ | ridge regression of each item on all others |
| GF-CF / SGFCF | $\tilde R^\top\tilde R$ plus a low-rank ideal filter from the SVD of $\tilde R = D_b^{-1/2} X D_f^{-1/2}$ | graph signal filtering on the normalized bundle-item matrix |
| iALS | $V$ from implicit ALS; a query folds in as $u = (V^\top V+\alpha V_S^\top V_S+\lambda I)^{-1}(1+\alpha)V_S^\top\mathbf 1$ | low-rank factorization |
| Content kNN | $\hat Z\hat Z^\top$ | feature similarity only |
| Blend | $(1-w)\,\mathrm{scaled}(Y W_{\text{knn}}) + w\,\mathrm{scaled}(Y\hat Z\hat Z^\top)$ | ItemKNN plus content, one weight $w$ per dataset |
| Label-aware rules | same label, label-to-label transition, blended with ItemKNN and content | category or artist evidence |

What every row shares is that **$W$ is fixed for the dataset**. The weighting cannot change with the query, an item with
$f_i=0$ only gets a score from content or labels, and a shared label can only add to a score (the label rules count it
positively).

**What these results show:**
- The best label-aware rule ties CLHE on POG_DENSE (.249 vs .248) and is ahead of it on POG, Spotify_sparse and
  Spotify, so labels carry a lot of the signal.
- DIALS is ahead of the best rule on all four large datasets, by +.008 to +.051 NDCG@20 (Holm $p=.0008$).
- On Dunnhumby (warm grocery baskets) the blend ties DIALS.

## 3. CLHE (Ma et al., WSDM 2024)

CLHE is the Contrastive Learning-enhanced Hierarchical Encoder. The whole method is one score:

$$
s(c\mid S) = \mathbf b(S)^\top \mathbf v_c,
$$

a bundle vector against an item vector, both 64-dimensional.

### Item encoder (`HierachicalEncoder`)

Each item $i$ gets three 64-dimensional **views**:
1. **Multimodal view:**
   $\mathbf m_i = \mathrm{norm}(\mathrm{MLP}_c(z^{\text{img}}_i)) + \mathrm{norm}(\mathrm{MLP}_t(z^{\text{txt}}_i))$. Each
   MLP has three layers ($d \to d \to 256 \to 64$, ReLU).
2. **ID view:** $\mathbf e_i$, a free learned row of an $n\times 64$ table.
3. **Collaborative view:** $W_{cf}\,\mathbf u_i$, a linear map of a precomputed item vector from the user–item graph.
   - The released files use LightGCN embeddings (`item_cf_feature.pt`).
   - On the e-commerce reruns, where no released file exists, we used a rank-64 SVD of the normalized user–item matrix.
   - An item absent from the user–item graph gets its multimodal view copied in.

The three views are stacked into a $3\times 64$ matrix $F_i$. One self-attention layer mixes them, and mean pooling gives
the item vector:

$$
\mathbf v_i = \mathrm{mean}_{\text{rows}}\Big(\mathrm{softmax}\big(\tfrac{F_iW_q (F_iW_k)^\top}{\sqrt{64}}\big)\,F_iW_v\Big).
$$

This is **early fusion**: by the time an item is scored, its content, ID and history have been merged into one vector.

### Bundle encoder

- The seeds' item vectors $\mathbf v_s$ are passed as tokens through a Transformer encoder (multi-head self-attention
  plus a feed-forward block, with padding masked), then pooled into $\mathbf b(S)$.
- The catalogue side uses a second item encoder with the same architecture but separate weights (the "decoder"). It
  gives the $\mathbf v_c$ that every candidate is scored against.

### Training

CLHE auto-encodes training bundles: the full bundle is the input, and the same bundle is the reconstruction target.

$$
\mathcal L = -\frac{1}{|B|}\sum_{i\in B}\log\mathrm{softmax}_i\big(\mathbf b(B)^\top V\big)
\;+\;\alpha_{\text{item}}\,\mathrm{InfoNCE}(\text{two augmented item views})
\;+\;\alpha_{\text{bundle}}\,\mathrm{InfoNCE}(\mathbf b(B),\,\mathbf b(\tilde B)).
$$

- **Reconstruction:** a softmax over the full catalogue.
- **Item contrastive term:** two dropout- or noise-augmented copies of each item vector are pulled together.
- **Bundle contrastive term:** the bundle is matched with a modified copy of itself, $\tilde B$, with items dropped or
  replaced.
- **At test time:** the seeds are the input.

### Parameters and what follows from them

- **Size:** two $n\times 64$ ID tables (one in each encoder), plus the MLPs, attention and Transformer. That is 9.48M
  parameters on POG and 29.0M on Spotify_sparse.
- **Cold items:** an item with no training bundles has an ID row that was never trained, and its collaborative view is
  a copy of its multimodal view. Its vector is therefore content only, passed through layers trained mostly on warm
  items.
- **No query-specific trust:** nothing in the score can say "these seeds have no history, so trust content more". That
  decision was already made inside each $\mathbf v_i$.
- **No labels:** category and artist are not inputs.
- **Measured:** on Spotify_sparse targets with no co-occurring seed CLHE gets NDCG@20 .003 (DIALS .263), and on POG's
  no-history queries .005 (DIALS .019).

## 4. RaMen (ECAI 2025)

*RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction.*

RaMen keeps CLHE's skeleton (an item encoder, a Transformer bundle encoder and a softmax reconstruction loss). It adds
two graph branches, and its score has two parts:

$$
s(c\mid S) = \underbrace{\big(\alpha\,\mathbf b_{\text{trans}}(S) + (1-\alpha)\,\mathbf b_{\text{graph}}(S)\big)^\top \mathbf v_c}_{\text{CLHE-like part}}
\;+\;\underbrace{\mathbf h_S^\top \mathbf h_c}_{\text{hypergraph part}}.
$$

### Branch 1: the item–item graph (asymmetric relations)

- **The graph:** $E = X_u^\top X_u$ from the user–item matrix $X_u$. There is an edge $i\to j$ if $E_{ij}\ge\varepsilon$.
  - The graph-building script is not in the released repository, so we built it from the paper (§3.1.3), with
    $\varepsilon$ = 5 on POG, 450 on Spotify and 1 on e-commerce.
- **Propagation:** a separate $n\times 64$ item table goes through 3 layers of a GATv2-style attention convolution
  (2 heads), averaged over layers, with a residual:
  $\mathbf g_i = \mathrm{GAT}(\mathbf e'_i) + \alpha_r\,\mathbf e'_i$.
- **Bundle vector:** the mean of its items' graph vectors, $\mathbf b_{\text{graph}} = D_B^{-1}X\,G$.
- **Item vector:** $\mathbf v_c$ is mixed with the graph vector by a constant,
  $\mathbf v_c \leftarrow \beta\,\mathbf v_c + (1-\beta)\,\mathbf g_c$.

### Branch 2: the multimodal hypergraph

- **Soft hyperedges:** for each modality (image, text), items are soft-assigned to $H$ hyperedges:
  $A = \mathrm{gumbel\text{-}softmax}(\mathrm{MLP}(z)\,W_{\text{hyper}})\in\mathbb{R}^{n\times H}$.
- **Bundle assignments:** bundles are assigned through their items, $A_B = \mathrm{gumbel\text{-}softmax}(X\,\mathrm{MLP}(z)\,W_{\text{hyper}})$.
- **Propagation:** one hypergraph layer, with dropout on the assignments:
  - each hyperedge gathers its items, $\Lambda = A^\top G$;
  - items and bundles read back, $\mathbf h_i = A\Lambda$ and $\mathbf h_B = A_B\Lambda$.
- **Score:** the image and text results are summed and normalized, giving the second score term $\mathbf h_S^\top\mathbf h_c$.

### The item encoder inside RaMen

- **Views:** two instead of CLHE's three: a multimodal view (image and text MLPs, concatenated, then a two-layer MLP)
  and an ID view.
- **Fusion:** the same self-attention fusion as CLHE.
- **No collaborative view:** the user–item collaborative view is dropped, because the item–item graph above replaces it.

### Training

$$
\mathcal L = \text{softmax reconstruction}
\;+\;\alpha_{\text{item}}\,\mathrm{InfoNCE}(\mathbf v_i,\mathbf h_i)
\;+\;\alpha_{\text{bundle}}\,\mathrm{InfoNCE}(\mathbf b,\mathbf h_B).
$$

The contrastive terms align the CLHE-like view with the hypergraph view, for items and for bundles. The mixing
constants $\alpha$, $\beta$, $\alpha_r$ and the edge threshold $\varepsilon$ are configuration values, not learned per
query.

### Parameters and what follows from them

- **Size:** four $n\times 64$ item tables (two per encoder, 256 numbers per item), plus the MLPs, GAT, hypergraph and
  Transformer. That is 16.1M parameters on POG and 56.5M on Spotify_sparse.
- **Same limits as CLHE:** the evidence is still fused into item and bundle vectors before scoring, labels are not
  used, and the trust between branches is a constant per dataset.
- **Graph dependence:** the graph branch depends on the user–item graph.
  - On RaMen's published e-commerce split that graph contains the test bundles. Its published R@20 of .84 comes from
    that setup, and the split is not released.
  - Our reruns use a clean graph built from training bundles only.

## 5. How DIALS solves it

DIALS stands for Diagonal-plus-low-rank label relations with spectral cross-fitting. It keeps the three kinds of
evidence apart until the last step, learns each one as a small function of shared statistics, and lets the query decide
how much each one counts:

$$
s(c\mid S) = w_C(S)\,C(S,c) + \sum_l w_l(S)\,L_l(S,c) + w_G(S)\,G(S,c).
$$

**Content:** $C(S,c) = \varphi(S)^\top\psi(c)$.
- $\psi$ is a rank-64 map of the candidate's features $z_c$ and its *partner history* $h_c$ (the co-occurrence-weighted
  mean feature of its training partners).
- $\varphi(S)$ is the same kind of map, averaged over the seeds.
- The maps are shared by all items. Nothing is stored per item, so a new item is scored from its features on day one.

**Labels:**

$$
L_l(S,c) = \delta_l\big(t_{g(c)}(S), |S|, |g(c)|\big)\,[t_{g(c)}(S)>0] \;+\; \tfrac{t(S)}{|S|}\,P_l M_l Q_l^\top\,e_{g(c)}.
$$

- $t(S)$ counts the seeds' labels, and $g(c)$ is the candidate's label.
- **The diagonal $\delta_l$** (a small MLP) scores a *shared* label, and its **sign is learned**. It comes out negative on
  POG (avoid the seed's category in an outfit) and positive on Spotify (same artist).
- **The low-rank part** says which labels go with which (boots → handbag).
  - $P_l$ and $Q_l$ are fixed factors from a rank-16 SVD of the normalized label-by-label co-occurrence
    $\Gamma_l^\top N\Gamma_l$.
  - Only the $16\times16$ matrix $M_l$ is learned.
  - Because the relation sits at the label level, it carries over to items with no history of their own.

**Graph:**

$$
G = \theta\log\Big(1+\sum_{s\in S} C_{sc}\,f_s^{-\alpha} f_c^{-\beta}\Big) + \rho(f_c).
$$

- These are the damped co-occurrence counts, with **learned** degree exponents $\alpha$ and $\beta$; the item-to-item
  baselines fix theirs.
- $\rho$ is a learned popularity prior.

**Gate:**

$$
w(S) = \mathrm{softplus}\big(\mathrm{MLP}(\log|S|,\ \text{mean seed }\log f,\ \text{share of seeds never seen})\big).
$$

It sets one weight per term for each query. On Spotify_sparse the graph-to-content weight falls from 0.43 to 0.16 as
the share of unseen seeds rises.

**Training** (`dials/train.py`):
- **Self-supervised queries:** each step hides a random part of 32 training bundles (half, or a uniform cut) and scores
  the full catalogue from the rest. Each hidden item competes against every item outside the bundle in a softmax, with
  no negative sampling.
- **Leave-one-out:** every count, degree, frequency and partner history is computed with the bundle's own contribution
  removed, so the model can't read its answer from its own counts.
- **Cross-fitting:** the label SVD can't be corrected that way, so it is cross-fitted over five folds.
- **Early stopping:** on validation NDCG@20.
- **Size:** 41.6k trained parameters (42.0k on Spotify_sparse), none of them per item.

## 6. Side by side

| | Simple baselines | CLHE | RaMen | DIALS |
|---|---|---|---|---|
| Score | $YW$ | $\mathbf b(S)^\top\mathbf v_c$ | $\mathbf b^\top\mathbf v_c + \mathbf h_S^\top\mathbf h_c$ | gated sum of content, label and graph terms |
| When evidence is combined | inside $W$, once per dataset | inside each item vector (early) | inside item and bundle vectors, plus a constant mix | at the end, per query (late) |
| History | $C$ or $X$ directly | LightGCN view (user–item) | item–item GAT, hypergraph | damped counts with learned exponents |
| Content | content kNN, blend | MLP view | MLP view, hypergraph | bilinear map with partner history |
| Labels | label rules (positive only) | not used | not used | signed diagonal + low-rank label relation |
| Cold item | content or label only | ID row never trained; content copied into the CF view | same | content and label terms, no per-item state |
| Per-item parameters | none | $2\times64$ | $4\times64$ | none |
| Trained parameters (POG) | 0 to a few | 9.48M | 16.1M | 41.6k |
| Training (POG, 4 CPU threads) | seconds to minutes | 85 min | about 71 min | 12–15 min |
| Leakage control | statistics from training only | — | — | leave-one-out + cross-fitting |

## 7. What the comparison shows

Test NDCG@20, on identical splits and features, with each method run from its own code over 3 seeds:

| Dataset | CLHE | RaMen | DIALS | Δ vs best neural |
|---|---|---|---|---|
| Electronic | .260 | .254 | **.335** | +.075 |
| Food | .272 | .274 | **.439** | +.164 |
| Clothing | .186 | .191 | **.388** | +.197 |
| POG_DENSE | .248 | | **.280** | +.033 |
| POG | .0172 | | **.0313** | +.014 |
| Spotify_sparse | .115 | | **.256** | +.141 |

- All 18 tests have $p=.002$ (Holm over 20). RaMen was rerun on the three e-commerce datasets only.
- **Where DIALS gains most:** on items and queries without history, because content and labels are separate terms
  that the gate can rely on.
- **It isn't just the gate:** with the per-query gate replaced by constant weights, most of the gain remains. The
  learned terms carry it.
- **Where it doesn't win:**
  - Warm grocery baskets (Dunnhumby), where the content + ItemKNN blend ties it.
  - The personalized Dunnhumby task, where every method gains the same amount from household repurchase counts.

## Sources

- **CLHE:** Ma et al., *Leveraging Multimodal Features and Item-level User Feedback for Bundle Construction*, WSDM 2024.
  The authors' released code.
- **RaMen:** *RaMen: Multi-Strategy Multi-Modal Learning for Bundle Construction*, ECAI 2025. The authors' released
  code; the item–item graph, missing from it, was rebuilt from the paper (§3.1.3).
- **Results:** `RESULTS.md` in this repository.
