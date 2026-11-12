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

### 2.1 The common form

Every simple baseline builds one item-to-item matrix $W\in\mathbb{R}^{n\times n}$ from the training bundles and scores
a batch of queries as

$$
\text{scores} = Y W,\qquad \text{score}(c\mid S) = \sum_{s\in S} W_{sc}.
$$

- **Reading a row:** row $s$ of $W$ is item $s$'s neighbourhood, meaning how strongly $s$ votes for every other item.
  A query's score is the sum of its seeds' votes.
- **Why this form:** these are the standard collaborative-filtering recommenders. A query is treated as a new user whose
  only interactions are its seeds, so a method built for users and items applies unchanged.
- **What differs between methods:** only how $W$ is built from $X$ (and, for some, from $Z$ or $\Gamma$).
- **Shared details:**
  - Item statistics come from training bundles only.
  - Ties are broken by $10^{-6}\log(1+f_c)$ for every method alike.
  - Each method's setting is chosen by grid search on validation NDCG@20, and test is scored once with that setting.

### 2.2 A worked example

Nine training bundles over five items:

| Bundle | Items |
|---|---|
| 1, 5 | tee, jeans |
| 2 | tee, boots |
| 3, 9 | tee, belt |
| 4, 6 | tee, scarf |
| 7 | jeans, boots, belt |
| 8 | jeans, boots |

The frequencies are $f$ = tee 7, jeans 4, boots 3, belt 3, scarf 2. The query is $S=\{\text{jeans}\}$. The row of
$C$ for jeans is tee 2, boots 2, belt 1, scarf 0.

Tee is a hub: it appears in 7 of 9 bundles. Boots is the item that specifically goes with jeans (2 of boots' 3 bundles
contain jeans). A good method should rank boots first. The scores each method gives (the seed itself excluded):

| Method | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| Popularity | tee 2.08 | boots 1.39 | belt 1.39 | scarf 1.10 |
| Co-occurrence | tee 2 | boots 2 | belt 1 | scarf 0 |
| ItemKNN (cosine) | **boots .577** | tee .378 | belt .289 | scarf 0 |
| RP3β, α = 1, β = 0 | tee .250 | boots .208 | belt .083 | scarf 0 |
| RP3β, α = 1, β = .5 | **boots .120** | tee .094 | belt .048 | scarf 0 |
| EASE, λ = 1 | **boots .372** | tee .322 | belt .025 | scarf **−.099** |

A second query uses two seeds, $S=\{\text{jeans},\text{belt}\}$. This shows how the seeds' votes add up. Boots is
in the one bundle that contains both seeds (bundle 7). Tee is bundled with each seed separately: twice with jeans and
twice with belt.

| Method | 1st | 2nd | 3rd |
|---|---|---|---|
| Popularity | tee 2.08 | boots 1.39 | scarf 1.10 |
| Co-occurrence | tee 4 (2 + 2) | boots 3 (2 + 1) | scarf 0 |
| ItemKNN (cosine) | **boots .911** (.577 + .333) | tee .814 (.378 + .436) | scarf 0 |
| RP3β, α = 1, β = 0 | tee .583 | boots .319 | scarf 0 |
| RP3β, α = 1, β = .5 | tee .220 | boots .184 | scarf 0 |
| EASE, λ = 1 | tee .746 | boots .534 | scarf **−.229** |

With two seeds the hub collects a vote from each one, so most methods now put tee first. ItemKNN is the exception,
because its normalization discounts each of tee's votes separately; §2.5 works through it step by step.

The rest of this section explains why each row of both tables comes out as it does.

### 2.3 Popularity

$$
W_{sc} = \log(1+f_c)\quad\text{for every } s,
$$

so $W = \mathbf 1\,\log(1+f)^\top$, a rank-one matrix that ignores the seeds.

- **What it captures:** that some items are simply bought more.
- **What it can't do:** anything query-specific. It ranks tee first for every query.
- **Its role:** the floor every other method has to beat, and the source of the prior every method's tie-break uses.

### 2.4 Co-occurrence

$$
W = C = X^\top X\ \text{with the diagonal zeroed},\qquad W_{sc} = \#\{\text{training bundles containing } s \text{ and } c\}.
$$

- **What it captures:** direct evidence that $c$ has been bundled with the seeds.
- **What goes wrong:** counts grow with popularity. A hub co-occurs with everything, so it ties or beats the specific
  partner. In the example, tee ties boots for $\{\text{jeans}\}$. For $\{\text{jeans},\text{belt}\}$ it wins
  outright, 4 against 3, because a hub collects a vote from every seed.
- **Setting:** none.

### 2.5 ItemKNN (Sarwar et al. 2001; Deshpande and Karypis 2004)

**The idea.** Two items are similar if they tend to appear in the same bundles. ItemKNN measures this with the cosine
between the items' columns of $X$, then lets each seed vote for its most similar items.

**Step 1: the cosine.** Column $\mathbf x_s = X_{:s}\in\{0,1\}^m$ marks the training bundles that contain item $s$. For
binary columns:
- the dot product counts shared bundles, $\mathbf x_s^\top\mathbf x_c = C_{sc}$;
- the squared norm counts the item's bundles, $\|\mathbf x_s\|^2 = f_s$.

So the cosine is

$$
\cos(s,c) = \frac{\mathbf x_s^\top\mathbf x_c}{\|\mathbf x_s\|\,\|\mathbf x_c\|} = \frac{C_{sc}}{\sqrt{f_s f_c}},
\qquad \text{in matrix form } \; D_f^{-1/2}\,C\,D_f^{-1/2}.
$$

Raw co-occurrence $C_{sc}$ grows with how popular both items are. Dividing by $\sqrt{f_sf_c}$ asks instead: of the
bundles these two items could have shared, how many did they? A hub appears in many bundles, so it pays for it in the
denominator.

**Step 2: shrinkage.** A cosine that rests on very few bundles is noisy. Two items that each appear once, in the same
bundle, have cosine $1/\sqrt{1\cdot1}=1$, the maximum, on the strength of a single bundle. Adding a constant $h$ to the
denominator damps these:

$$
\tilde W_{sc} = \frac{C_{sc}}{\sqrt{f_s f_c}+h}.
$$

With $h=10$:
- the single-bundle pair falls from 1 to $1/(1+10)=.09$;
- a well-supported pair, with $C=50$ and $f_s=f_c=100$, falls only from .50 to $50/(100+10)=.45$.

So $h$ penalizes thin evidence and leaves strong evidence almost untouched. At $h=0$ it is the plain cosine.

**Step 3: keep the top $k$ per row.** Row $s$ of $\tilde W$ is item $s$'s neighbourhood. ItemKNN keeps its $k$ largest
entries and sets the rest to zero:

$$
W_{s\cdot} = \mathrm{top}_k\big(\tilde W_{s\cdot}\big).
$$

- **Why:** the long tail of weak similarities is mostly noise, and dropping it also makes $W$ sparse and fast.
- **A side effect:** $W$ is no longer symmetric. $c$ can be among $s$'s top $k$ without $s$ being among $c$'s.
- **What scoring uses:** row $s$, that is, *the seed's* neighbourhood. A candidate is reached only if it is among some
  seed's $k$ nearest items.

**Step 4: score a query by summing the seeds' rows.**

$$
\text{score}(c\mid S) = \sum_{s\in S} W_{sc},\qquad \text{scores} = YW.
$$

Each seed votes for its neighbours with weight equal to their similarity. A candidate close to *several* seeds collects
several votes, so the query favours items that fit the whole partial bundle, not just one seed. The seeds themselves
are then removed from the ranking.

**The worked example, step by step.** Cosines from the toy data ($f$ = tee 7, jeans 4, boots 3, belt 3, scarf 2):

| $\cos(s,c)$ | tee | jeans | boots | belt | scarf |
|---|---|---|---|---|---|
| **jeans** (row used for seed jeans) | $2/\sqrt{28}$ = .378 | — | $2/\sqrt{12}$ = .577 | $1/\sqrt{12}$ = .289 | 0 |
| **belt** (row used for seed belt) | $2/\sqrt{21}$ = .436 | $1/\sqrt{12}$ = .289 | $1/\sqrt{9}$ = .333 | — | 0 |

- **One seed, $S=\{\text{jeans}\}$:** the scores are the jeans row. Boots .577 > tee .378 > belt .289.
  - Co-occurrence tied tee and boots at 2. The cosine breaks the tie, because tee's 2 shared bundles come out of 7
    while boots' come out of 3.
- **Two seeds, $S=\{\text{jeans},\text{belt}\}$:** add the two rows and drop the seeds.

  | Candidate | from jeans | from belt | score |
  |---|---|---|---|
  | boots | .577 | .333 | **.911** |
  | tee | .378 | .436 | .814 |
  | scarf | 0 | 0 | 0 |

  - **Tee is closer to belt** than boots is (.436 vs .333), because tee and belt share 2 bundles.
  - **But boots is much closer to jeans** (.577 vs .378), and that larger margin decides the sum.
  - **Raw co-occurrence ranks them the other way** (tee 4, boots 3). It counts tee's 4 shared bundles at face value,
    without discounting them by tee's 7 bundles in total.
- **Shrinkage:** with $h=1$ every entry shrinks (jeans–boots .577 → $2/(\sqrt{12}+1)$ = .448). The two-seed scores
  become boots .698 and tee .676: the same order, with a smaller gap. Here every pair rests on only 1–2 bundles, so
  $h$ scales all of them by similar factors. On real data it separates thin pairs from well-supported ones.
- **Top-$k$:** with $k=1$, jeans keeps only boots (.577) and belt keeps only tee (.436). The scores become boots .577
  and tee .436: still the same order, but each seed now votes for a single item.

**Settings searched:** $k\in\{50,200,1000\}$, $h\in\{0,10,100\}$, chosen on validation NDCG@20.

**How it did:** ItemKNN on its own was never the best method. It is the collaborative half of the blend (§2.11), which
was the best simple baseline on 6 of 8 datasets, at $k$ from 50 to 1000. On POG_DENSE it scored .208, against RP3β's
.241. There popularity is useful, and the cosine removes it.

**What it can't do:**
- **Cold items:** an item with $f_c=0$ has a zero column, so its cosine with everything is 0 (we use $\max(f,1)$ to
  avoid dividing by zero). It can never be recommended.
- **No penalties:** similarities are non-negative, so a candidate can only gain from a seed, never lose. It can't
  express "this seed makes that item *less* likely", which EASE (§2.7) can.
- **No interaction between seeds:** the sum treats seeds independently. Two seeds that each mildly suggest $c$ count
  the same whether or not they ever appeared together.

### 2.6 RP3β (Cooper et al. 2014; Paudel et al. 2017)

RP3β is a random walk on the bundle–item graph. Write $D_f=\mathrm{diag}(f)$ and $D_b=\mathrm{diag}(|b|)$ (bundle
sizes):
- $P_{ib} = X_{bi}/f_i$ is the step from item $i$ to one of its bundles;
- $P_{bj} = X_{bj}/|b|$ is the step from bundle $b$ to one of its items.

The two-step walk item → bundle → item is $D_f^{-1}X^\top D_b^{-1}X$. RP3β raises these normalizers to a power $\alpha$
and adds a popularity penalty $\beta$ on the destination:

$$
W = D_f^{-\alpha}\,X^\top D_b^{-\alpha} X\,D_f^{-\beta},\qquad \text{diagonal zeroed}.
$$

- **At α = 1, β = 0:** $W_{sc}$ is the probability that a walk from $s$ ends at $c$. That still favours hubs, because
  most walks pass through them. In the example tee wins, .250 against .208.
- **At β > 0:** each destination is divided by $f_c^\beta$, which demotes hubs. At β = .5, boots wins (.120 against
  .094).
- **Two seeds:** for $\{\text{jeans},\text{belt}\}$, tee wins even at β = .5 (.220 against .184). A β of .5 is not
  enough to offset two separate votes for the hub; this toy needs a larger β.
- **What β does:** it sets how hard popularity is penalized, a knob ItemKNN fixes at ½ through its cosine.
- **Setting searched:** $\alpha\in\{.5,1\}$, $\beta\in\{0,.25,.5,.75\}$.
- **Where it was best:** POG_DENSE, with .241 at α = 1, β = 0. There popularity helps, which is why β = 0 was chosen.
- **What it can't do:** score an item with no history.

### 2.7 EASE (Steck 2019)

EASE fits a linear auto-encoder: predict each item from all the others, by ridge regression, without letting an item
predict itself.

$$
\min_W \|X - XW\|_F^2 + \lambda\|W\|_F^2\quad\text{s.t. } \mathrm{diag}(W)=0.
$$

It has a closed form. With $P=(X^\top X+\lambda I)^{-1}$,

$$
W = I - P\,\mathrm{diag}(P)^{-1},\qquad W_{sc} = -P_{sc}/P_{cc}\ (s\neq c).
$$

- **How it differs from similarity:** $W$ comes from the *inverse* of the Gram matrix, so each weight is a partial
  regression coefficient: the effect of $s$ on $c$ after the other items are accounted for.
- **Negative weights:** that partial coefficient can be negative. In the example scarf gets −.099. Scarf appears only
  with tee, so once tee is accounted for, a jeans-only query is evidence *against* scarf. Similarity methods can't
  express this. With two seeds the penalty adds up: scarf gets −.229 for $\{\text{jeans},\text{belt}\}$. Tee comes
  first there (.746 against .534 for boots). Jeans' row favours boots (.372 vs .322), but belt's row favours tee by much
  more (.424 vs .162).
- **λ:** shrinks toward zero and stabilizes the inverse for rare items.
- **Cost:** a dense $n\times n$ inverse. It is run only when there are at most 40,000 training items, so it was skipped
  on Spotify_sparse and Spotify.
- **Setting searched:** $\lambda\in\{50,200,500,1000,3000\}$.
- **Where it was best:** POG, with .0198 at λ = 3000. The heavy regularization reflects how sparse POG is.

### 2.8 GF-CF (Shen et al., CIKM 2021)

GF-CF treats the bundle–item matrix as a graph signal and applies filters to it. Normalize it symmetrically:

$$
\tilde R = D_b^{-1/2}\,X\,D_f^{-1/2},\qquad \tilde R = U\Sigma V^\top\ \text{(SVD)}.
$$

The score has two parts:

$$
\text{scores} = Y\,\tilde R^\top\tilde R \;+\; w\,\big(Y D_f^{-1/2}V_k\big)\big(D_f^{1/2}V_k\big)^\top .
$$

- **The linear filter** $\tilde R^\top\tilde R$ is a degree-normalized co-occurrence matrix, close to ItemKNN without
  top-$k$.
- **The ideal low-pass filter** keeps only the top $k$ singular directions $V_k$, the smooth, large-scale structure of
  the graph. It connects items that never co-occurred directly but share partners.
- **Setting searched:** $k\in\{64,256\}$, $w\in\{0,.1,.3,1,3\}$.
- **What it can't do:** score an item with no history, whose row of $V$ is zero.

### 2.9 SGFCF (Peng et al., KDD 2024)

SGFCF generalizes GF-CF in two ways. Normalize with an exponent $\varepsilon$ instead of ½, and take the SVD:

$$
N = D_b^{-\varepsilon}\,X\,D_f^{-\varepsilon} = U\Sigma V^\top.
$$

- **Folding in the query:** $r = Y D_f^{-\varepsilon}$.
- **Reweighting the spectrum:** each singular direction $k$ is weighted by $(\sigma_k/\sigma_{\max})^{2\beta}/\sigma_k$,
  so larger β concentrates the score on the strongest directions.
- **Combining:** the low-rank result is normalized per query, and γ times the normalized full linear filter $rN^\top N$
  is added back.

$$
\text{scores} = \mathrm{unit}\Big(r\,V\,\mathrm{diag}\big(\tfrac{(\sigma/\sigma_{\max})^{2\beta}}{\sigma}\big)V^\top\Big) + \gamma\,\mathrm{unit}\big(rN^\top N\big).
$$

- **Setting searched:** rank $\in\{64,256\}$, $\varepsilon\in\{.3,.5\}$, $\beta\in\{1,2.5\}$, $\gamma\in\{0,.5,1\}$.
- **How it did:** close behind the best method on most datasets, but never the best.

### 2.10 iALS (Hu, Koren and Volinsky 2008)

iALS is matrix factorization for implicit data: $X\approx UV^\top$ with $U\in\mathbb{R}^{m\times k}$ and
$V\in\mathbb{R}^{n\times k}$. Observed entries are trusted more, with confidence $1+\alpha$ against 1 for unobserved
ones:

$$
\min_{U,V}\sum_{b,i} c_{bi}\,(X_{bi}-\mathbf u_b^\top\mathbf v_i)^2 + \lambda(\|U\|^2+\|V\|^2).
$$

- **Training:** alternate closed-form ridge solves for $U$ and $V$, 8 iterations.
- **A query** is folded in as a new row:
  $\mathbf u = (V^\top V+\alpha V_S^\top V_S+\lambda I)^{-1}(1+\alpha)V_S^\top\mathbf 1$. The scores are then $V\mathbf u$.
- **Setting searched:** rank $\in\{64,128\}$, $\alpha\in\{5,20\}$, $\lambda\in\{1,10\}$.
- **What it can't do:** an item with no history gets $\mathbf v_c=0$ and so a score of 0.
- **How it did:** below ItemKNN everywhere. With few items per bundle, a low-rank fit loses the specific pairs.

### 2.11 Content kNN and the blend

**Content kNN** uses the cosine of the item features:

$$
W = \hat Z\hat Z^\top,\qquad \hat Z = Z\ \text{with unit rows}.
$$

It is the only family member that scores an item with no history.

**The blend** puts content next to the collaborative evidence:

$$
\text{scores} = (1-w)\,\mathrm{scaled}(Y W_{\text{knn}}) + w\,\mathrm{scaled}(Y\hat Z\hat Z^\top).
$$

- **Construction:** each part is divided by its largest absolute value in the query, and ItemKNN is fixed at its own
  validation setting.
- **Setting searched:** $w\in\{0,.1,.2,.3,.5,.7,.9,1\}$.
- **Where it was best:** on 6 of 8 datasets (Electronic, Food, Clothing, Dunnhumby, Spotify_sparse, Spotify). It is the
  strongest simple baseline overall.
- **What it can't do:** one $w$ per dataset can't be right for both a query whose seeds are warm and one whose seeds are
  new.
- **Its main limit:** content enters as **resemblance**. The blend can say that a candidate looks like the seeds, but
  not that it *goes with* them.

### 2.12 Label-aware rules

These baselines use the same item labels DIALS reads, and nothing it doesn't:

| Dataset | Label levels $l$ |
|---|---|
| POG, POG_DENSE | category (one level) |
| Spotify, Spotify_sparse | artist and album (two levels) |

They ask one question at the level of labels: *given the labels of the seeds, which labels should the candidate have?*
The item's own identity enters only through a popularity prior.

**Step 1: labels as a matrix.** At level $l$ every item has exactly one label, $g_l(c)$. Write it as a one-hot matrix
$\Gamma_l\in\{0,1\}^{n\times L_l}$, with $(\Gamma_l)_{c,g}=1$ if item $c$ has label $g$.

**Step 2: the seeds' label profile.**

$$
t_l(S) = \mathbf y_S^\top\Gamma_l \in \mathbb{N}^{L_l},
$$

that is, how many seeds carry each label. For a batch of queries this is $Y\Gamma_l$.

**Step 3a: the "same label" rule.** Count how many seeds share the candidate's label:

$$
m_l(S,c) = t_l(S)_{g_l(c)} = (\mathbf y_S^\top\Gamma_l\Gamma_l^\top)_c,
$$

$$
\text{score}(c\mid S) = \sum_l a_l\log\big(1+m_l(S,c)\big) + \gamma\log(1+f_c).
$$

- $\Gamma_l\Gamma_l^\top$ is the item-by-item "same label" matrix, so this is again an item-to-item rule of the form
  $YW$, with $W=\Gamma\Gamma^\top$ inside a log.
- **Its weakness:** a shared label can only *raise* a score, because $\log(1+m)\ge 0$.

**Step 3b: the "transition" rule.** Learn from the training bundles which labels go together, then score the candidate's
label against the seeds' labels.
- **Label co-occurrence:** aggregate item co-occurrence to labels,
  $$L_l = \Gamma_l^\top C\,\Gamma_l\in\mathbb{R}^{L_l\times L_l}.$$
  Entry $(g,h)$ counts pairs of distinct items, one labelled $g$ and one labelled $h$, that share a training bundle.
  The diagonal is kept: two different items with the same label can co-occur.
- **Row-normalize:** $T_l = D^{-1}L_l$, so row $g$ is a distribution, "given an item labelled $g$, how often its bundle
  partner is labelled $h$".
- **Average over the seeds:** the seeds' labels predict the candidate's label through
  $$\frac{t_l(S)}{|S|}\,T_l\in\mathbb{R}^{L_l},$$
  the average over seeds of each seed's row of $T_l$.
- **Score:** the log of that average at the candidate's label, plus a within-label prior:
  $$\text{score}(c\mid S) = \sum_l a_l\log\Big(\epsilon+\big[\tfrac{t_l(S)}{|S|}T_l\big]_{g_l(c)}\Big) + \gamma\log P(c\mid g(c)),
  \qquad P(c\mid g)=\frac{f_c+1}{F_g+|g|}.$$
  - $F_g$ is the total training frequency of the items with label $g$, and $|g|$ is how many items have that label.
  - $P(c\mid g)$ is the item's add-one share of its own label. Every item of a label gets the same transition score, so
    this prior is the only thing that ranks items *within* a label.
  - On Spotify the prior uses the finer level (album).
  - $\epsilon$ keeps the log finite for label pairs never seen together.

**Step 4: blends.**
- **Label + ItemKNN:** whichever of the two rules scored better on validation is scaled to $[0,1]$ per query (min-max,
  because log scores are negative). It is blended with ItemKNN, which is scaled by its largest absolute value:
  $$(1-w)\,\mathrm{scaled}(\text{ItemKNN}) + w\,\mathrm{minmax}(\text{label}).$$
  ItemKNN is fixed at its own validation setting.
- **Label + ItemKNN + content:** the same, plus $v$ times the scaled content cosine.

**The worked example.** Give the toy items categories: tee = top, jeans = bottom, boots = shoes, belt and scarf =
accessory.

| $L$ (label pairs sharing a bundle) | top | bottom | shoes | accessory |
|---|---|---|---|---|
| top | 0 | 2 | 1 | 4 |
| bottom | 2 | 0 | 2 | 1 |
| shoes | 1 | 2 | 0 | 1 |
| accessory | 4 | 1 | 1 | 0 |

| $T$ (row-normalized) | top | bottom | shoes | accessory |
|---|---|---|---|---|
| bottom | .40 | 0 | .40 | .20 |
| accessory | .667 | .167 | .167 | 0 |

The accessory–accessory entry is 0 because belt and scarf never share a bundle.

**Query $S=\{\text{jeans},\text{belt}\}$:** the label profile is $t(S)$ = (top 0, bottom 1, shoes 0, accessory 1), with
$|S|=2$.

| Candidate (label) | Same label: $\log(1+m)$ | Transition: $\tfrac{t(S)}{\lvert S\rvert}T$ at its label | Transition score ($\gamma=0$) |
|---|---|---|---|
| tee (top) | $\log 1 = 0$ | (.40 + .667)/2 = **.533** | **−0.63** |
| boots (shoes) | $\log 1 = 0$ | (.40 + .167)/2 = .283 | −1.26 |
| scarf (accessory) | $\log 2 =$ **.693** | (.20 + 0)/2 = .100 | −2.30 |

- **The two rules disagree completely.**
  - "Same label" ranks scarf first, because it shares *accessory* with belt. In an outfit a second accessory is the
    wrong answer, and this is exactly the case that hurts the rule on POG.
  - "Transition" ranks scarf last: accessories are rarely bundled with accessories. It ranks tee first, because tops go
    with both bottoms and accessories.
- **The transition rule can't separate items with the same label.** Every top gets .533. On real data, with thousands of
  items per category, the prior $P(c\mid g)$ and the ItemKNN blend have to do that work.
- **With $\gamma=1$, "same label" becomes mostly popularity:** tee 2.08, scarf 1.79, boots 1.39. The label vote moves
  scarf above boots.

**Settings searched** (validation NDCG@20):
- same label: $a=(1)$ on POG; on Spotify $a\in\{(1,0),(0,1),(1,1),(1,.5),(.5,1)\}$ over (artist, album);
  $\gamma\in\{0,.1,.3,1\}$;
- transition: the same $a$; $\gamma\in\{0,.5,1,2\}$, $\epsilon\in\{10^{-4},10^{-2}\}$;
- blends: $w\in\{0,.1,.2,.3,.5,.7,.9,1\}$, $v\in\{0,.05,.1,.2,.3,.5\}$.

**What was chosen:**
- **POG_DENSE:** transition ($\gamma=.5$), blended at $w=.7$. Content added nothing ($v=0$).
- **Spotify_sparse:** same label with artist and album weighted equally ($\gamma=.3$), blended at $w=.7$, plus content at
  $v=.5$, the largest value searched. That lifted the score only from .203 to .205.

| Test NDCG@20 (clean) | POG_DENSE | POG | Spotify_sparse | Spotify |
|---|---|---|---|---|
| Same label | .140 | .0052 | .193 | .198 |
| Transition | .197 | .0117 | .122 | .161 |
| Label + ItemKNN + content | .249 | .0234 | .205 | .247 |
| DIALS | **.280** | **.0313** | **.256** | **.282** |

The table shows the problem DIALS is built for:
- **On POG,** "same label" is the weaker rule. A shared category should count *against* a candidate (one item per
  category in an outfit), but the rule can only add to its score.
- **On Spotify,** "same label" is the stronger rule, because a shared artist is good evidence.
- **The consequence:** the rule family has to be chosen per dataset. DIALS learns the sign of the shared-label term
  instead ($\delta_l$ in §5), and a low-rank label relation in place of the fixed transition table.

### 2.13 How the simple baselines did

Test NDCG@20 on clean queries, each method at its validation setting:

| Dataset | Pop. | Co-occ. | ItemKNN | RP3β | EASE | GF-CF | SGFCF | iALS | Blend | DIALS |
|---|---|---|---|---|---|---|---|---|---|---|
| Electronic | .026 | .231 | .248 | .255 | .254 | .254 | .246 | .236 | **.299** | .335 |
| Food | .016 | .249 | .262 | .274 | .267 | .269 | .272 | .250 | **.425** | .439 |
| Clothing | .003 | .177 | .193 | .194 | .201 | .192 | .191 | .175 | **.330** | .388 |
| Dunnhumby | .057 | .094 | .102 | .103 | .104 | .080 | .104 | .081 | **.104** | .102 |
| POG_DENSE | .156 | .240 | .208 | **.241** | .228 | .230 | .237 | .191 | .208 | .280 |
| POG | .0068 | .0195 | .0174 | .0193 | **.0198** | .0165 | .0183 | .0151 | .0174 | .0313 |
| Spotify_sparse | .006 | .139 | .148 | .149 | — | .099 | .138 | .130 | **.153** | .256 |
| Spotify | .022 | .160 | .219 | .210 | — | .173 | .195 | .210 | **.223** | .282 |

- **Bold** marks the best simple baseline by validation, the one DIALS is tested against.
- "—" means EASE was not run (more than 40,000 items).
- On Dunnhumby the blend, SGFCF and EASE all round to .104.
- **Content matters most on e-commerce.** Adding it lifts ItemKNN by .05 to .16. There the bundles are small and the
  catalogue is mostly cold.
- **On POG the best methods are the ones that keep popularity** (RP3β at β = 0, co-occurrence, EASE). In outfits,
  popular pieces are often right.
- **DIALS's lead over the best simple baseline:**
  - largest where history is thin and labels carry signal: Spotify_sparse, +.104;
  - none on warm grocery baskets (Dunnhumby, −.002, not significant).

### 2.14 What the family can't do, and what DIALS changes

| Limitation of $YW$ | Where it shows | DIALS's answer |
|---|---|---|
| One $W$ (and one blend weight) per dataset | a query with new seeds and one with warm seeds get the same mix | a gate $w(S)$ that sets the mix per query |
| Popularity correction fixed by a grid ($\alpha$, $\beta$, $h$, $\varepsilon$) | the right β differs between POG and e-commerce | degree exponents $\alpha$, $\beta$ learned jointly with the other terms |
| Items with $f_c=0$ get no collaborative score | POG: 96% of queries have no co-occurring seed | content and label terms that need no history |
| Content as resemblance ($\hat Z\hat Z^\top$) | near-duplicates rank first | a learned bilinear map $\varphi(S)^\top\psi(c)$ |
| A shared label can only add | "same label" is weaker on POG and stronger on Spotify | a shared-label term $\delta_l$ whose sign is learned |
| A fixed label-transition table | coarse, and blind to rare labels | a low-rank, learned label relation $P_lM_lQ_l^\top$ |

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
   - The released files use LightGCN embeddings (`item_cf_feature.pt`). LightGCN (He et al. 2020) propagates free
     user and item embeddings over the normalized user–item graph $\hat A = D^{-1/2}AD^{-1/2}$ and averages the layers,
     $E = \tfrac1{K+1}\sum_{k=0}^{K}\hat A^kE^{(0)}$. Without nonlinearities, this is a smoothed low-rank factorization
     of the user–item matrix.
   - On the e-commerce reruns, where no released file exists, we used a rank-64 SVD of the normalized user–item matrix.
   - An item absent from the user–item graph gets its multimodal view copied in.

The three views are stacked into a $3\times 64$ matrix $F_i$. One self-attention layer mixes them, and mean pooling gives
the item vector:

$$
\mathbf v_i = \mathrm{mean}_{\text{rows}}\Big(\mathrm{softmax}\big(\tfrac{F_iW_q (F_iW_k)^\top}{\sqrt{64}}\big)\,F_iW_v\Big).
$$

This is **early fusion**: by the time an item is scored, its content, ID and history have been merged into one vector.

### Bundle encoder

- The seeds' item vectors $\mathbf v_s$ form a $|S|\times64$ matrix $V_S$. This passes through a stack of self-attention
  layers, $V_S \leftarrow \mathrm{softmax}\big(\mathrm{LN}(V_S)W_q(\mathrm{LN}(V_S)W_k)^\top/\sqrt{64}\big)\,\mathrm{LN}(V_S)$,
  with padding masked.
  - The code's defaults are one head, layer norm, no feed-forward block and no residual. The number of layers is a
    configuration value.
  - The result is mean-pooled over the seeds into $\mathbf b(S)$.
  - There are no position encodings, so the bundle vector does not depend on the order of the seeds.
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
- **InfoNCE**, the contrastive loss used in both terms below: for a batch of paired vectors $(\mathbf a_i,\mathbf b_i)$,
  $-\frac1N\sum_i\log\frac{\exp(\cos(\mathbf a_i,\mathbf b_i)/\tau)}{\sum_j\exp(\cos(\mathbf a_i,\mathbf b_j)/\tau)}$. It pulls the two
  copies of the same item (or bundle) together and pushes the other items in the batch away; $\tau$ = 0.2 in both
  models.
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
- **Propagation:** a separate $n\times 64$ item table $\mathbf e'$ goes through 3 layers of a GATv2-style attention
  convolution (2 heads, averaged).
  - One layer updates each item from its graph neighbours $\mathcal N(i)$, with learned attention weights:
    $$\mathbf x_i' = \sum_{j\in\mathcal N(i)}a_{ij}\,W_l\mathbf x_j,\qquad
    a_{ij} = \mathrm{softmax}_j\big(\mathbf a^\top\,\mathrm{LeakyReLU}(W_l\mathbf x_j + W_r\mathbf x_i)\big).$$
  - The layer outputs (including the input) are averaged, and a residual is added:
    $\mathbf g_i = \mathrm{GAT}(\mathbf e'_i) + \alpha_r\,\mathbf e'_i$.
  - Because $a_{ij}\neq a_{ji}$, the learned relation is asymmetric: boots may point to a belt more strongly than the
    belt points back. RaMen's paper presents this as its "asymmetric" relation.
- **Bundle vector:** the mean of its items' graph vectors, $\mathbf b_{\text{graph}} = D_B^{-1}X\,G$.
- **Item vector:** $\mathbf v_c$ is mixed with the graph vector by a constant,
  $\mathbf v_c \leftarrow \beta\,\mathbf v_c + (1-\beta)\,\mathbf g_c$.

### Branch 2: the multimodal hypergraph

- **Hyperedges as learned clusters:** a hyperedge joins any number of items. Here the $H$ hyperedges are not given by the
  data; they are learned soft clusters of the item features.
- **Gumbel-softmax** (temperature $\tau$ = 0.2) turns each row of logits into a nearly one-hot but differentiable
  assignment, so each item belongs mostly to one cluster while gradients still flow.
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
