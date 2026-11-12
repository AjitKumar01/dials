# Introduction (draft)

Citation keys in brackets are placeholders for the bibliography.

---

A bundle is a set of items offered or consumed together: an outfit of a top, trousers, shoes and a bag; a playlist
built around a few artists; the contents of one shopping trip. Platforms rarely build bundles from nothing. A
merchandiser picks a hero product, or a listener adds a couple of songs, and the system is asked to fill in the rest.
This is **bundle completion**. Given a few seed items $S$ of an unseen bundle, rank the whole catalogue so that the
missing items come first [CLHE, RaMen]. The task sits between two familiar problems. Like item-to-item
recommendation, it starts from items rather than from a user. Like set prediction, the answer must fit the seeds as a
group, not just resemble each seed one at a time.

The current state of the art treats bundle completion as representation learning. CLHE [CLHE] encodes every item from
three views: an ID embedding, multimodal features from foundation models such as BLIP and CLAP, and a LightGCN
embedding of the user–item graph. It fuses the views with self-attention, pools the seeds with a second attention
encoder, and trains the result with reconstruction and two contrastive losses. RaMen [RaMen] keeps this skeleton. It
adds attention over an item–item graph and a hypergraph of learned latent intents. Both models score a candidate by an
inner product between a bundle vector and the candidate's item vector. Both learn that item vector from the bundles
the item appears in.

That design ties the model's capacity to the catalogue. On POG, CLHE has 9.5M trained parameters and RaMen 16.1M,
mostly embedding tables with a row per item. Each row is fitted only from the few training bundles that contain its
item. Bundle data is long-tailed: most items appear in a handful of bundles, and new items in none. Both papers
recognise this and answer it with more views, more graphs and more self-supervision. The basic constraint is still
there. What the model knows about an item is stored in that item's parameters, so rarely bundled items get weak rows,
and cold items have only their content view to fall back on. A second, quieter problem follows from the first. On the
splits we reran, CLHE scores below a tuned simple baseline, such as ItemKNN, EASE or a blend of content similarity
and ItemKNN, on five of six datasets. Large per-item models are hard to fit from bundle data, and their gains over
well-tuned simple baselines should not be taken for granted [Dacrema].

What makes items go together is largely shared across items, and it is visible in their labels. An outfit takes one
item from each of several categories and seldom two from the same one. A playlist does the opposite: it returns to the
same artists and albums. Which label sits next to which is a relation between a few hundred labels, not a property of
each of hundreds of thousands of items, so every bundle in the training set can be used to learn it. How much each
kind of evidence should count also depends on the query. A seed that appears in many training bundles has reliable
co-occurrence. A seed that appears in none has only its content and its labels. A model that weights these sources
once, for all queries, has to compromise between them.

In this work we propose **DIALS**, a bundle completion model whose parameters describe relations between labels and
features, never individual items. For seeds $S$ and a candidate $c$ it computes

$$
s(c\mid S) = w_C(S)\,C(S,c) + \sum_l w_l(S)\,L_l(S,c) + w_G(S)\,G(S,c),
$$

a gated sum of three kinds of evidence:

- **Content** $C(S,c)=\varphi(S)^\top\psi(c)$ is a rank-64 bilinear map of the items' encoded features and their
  partner history.
- **Labels.** $L_l$ splits each label level into two parts:
  - a learned *diagonal* that scores a shared label and can turn negative, so the model learns to avoid the seeds'
    category in fashion and to prefer their artist in music;
  - a *low-rank* term, $(t(S)/|S|)\,P_l M_l Q_l^\top e_{g(c)}$, which says which labels go together. $P_l$ and
    $Q_l$ are fixed factors from a cross-fitted SVD of the label co-occurrence matrix, and only the $16\times16$
    matrix $M_l$ is learned.
- **Graph** $G(S,c)$ adds damped co-occurrence counts with a frequency prior.

The gate $w(S)$ reads three statistics of the query: its size, how often its seeds were bundled, and how many of them
were never bundled. From these it decides how far to trust each term. The whole model has 41,624 parameters, the
same configuration on every dataset. None of the parameters belongs to an item, so an item that never appeared in a
training bundle is scored by exactly the same function as any other.

We evaluate DIALS on eight datasets from four domains: fashion outfits (POG, POG_DENSE), music playlists (Spotify,
Spotify_sparse), e-commerce bundles (Electronic, Food, Clothing) and grocery baskets (Dunnhumby). We rerun CLHE and
RaMen with their own code on identical splits, using the same released item features, and tune twelve simple and
label-aware baselines on validation. With three seeds and Holm-corrected paired tests:

- **Against the neural models:** DIALS is significantly ahead of CLHE on all six datasets where CLHE was rerun, and
  of RaMen on the three where RaMen was.
- **Against the best simple baseline:** DIALS is significantly ahead on NDCG@20 on six of eight datasets. On Food only
  recall separates. On Dunnhumby, grocery baskets, neither metric does, and we have not yet
  established why.
- **With fewer training bundles:** on POG_DENSE, DIALS trained on 1% of the bundles keeps an NDCG@20 of .201, above
  CLHE trained on the same 1% (.169). On the sparser POG, every method collapses at 1%.
- **With user histories instead of bundles:** no method learns useful completions. Histories pair substitutes, while
  bundles pair complements. So platforms without bundle data cannot simply substitute purchase logs, and this holds for
  CLHE's design as much as for ours.

Our contributions are as follows:

- **Model.** We formulate bundle completion as a query-gated combination of content, label-relation and co-occurrence
  evidence. Every parameter is shared across items, and label relations are learned through a cross-fitted low-rank
  factorisation.
- **Evaluation.** We rerun the strongest neural models under one protocol: identical splits, identical item features,
  and user–item graphs restricted to training bundles. We report them alongside a broad suite of tuned simple
  baselines, with paired significance tests.
- **Findings.** DIALS is significantly ahead of both neural models and of most simple baselines with about
  $1/230$ of CLHE's parameters and $1/390$ of RaMen's. It degrades gracefully as training bundles become scarce, and
  we identify where it does not help: grocery baskets, very sparse bundle data, and training without bundles.

The rest of the paper is organised as follows. Section 2 reviews bundle construction and item-to-item recommendation.
Section 3 defines the task and describes DIALS. Section 4 gives the protocol and baselines. Section 5 reports the
results, ablations and the bundle-budget study.

---

## Notes on the draft

- **Where each paper shaped the structure.**
  - *Attention Is All You Need* sets the arc: the established approach, its structural constraint, why existing fixes
    leave that constraint in place, and the proposed change.
  - CLHE sets the problem framing (partial bundle, missing items) and the challenges, then contributions layout.
  - RaMen's "intrinsic vs. extrinsic information" maps onto our labels and content vs. co-occurrence.
- **No wording is copied.** Each claim about CLHE and RaMen is from their papers or code, and each number is from
  RESULTS.md or the bundle-budget comparison.
- **Claims deliberately avoided:** "first", "completely", and any comparison or citation of BundleFuse.
- **To check before submission:**
  - The twelve baselines: popularity, co-occurrence, ItemKNN, RP3β, EASE, GF-CF, SGFCF, iALS, content kNN, the blend,
    same label, transition. The label+ItemKNN blends are counted as variants.
  - The Dunnhumby sentence says the cause is not yet established, matching RESULTS.md. Replace it once an ablation
    explains the result.
