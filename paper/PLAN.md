# Paper plan: DIALS for PAKDD 2027

## 0. Venue constraints

| Item | PAKDD 2027 research track |
|---|---|
| Format | Springer LNCS/LNAI (`llncs.cls` v2.26 and `splncs04.bst`, downloaded from CTAN into this folder) |
| Length | **12 pages including references and appendices** |
| Review | Double-blind: no author names, own prior work cited in the third person |
| Deadline | **Friday 20 November 2026, 23:59 AoE** (4 days from 16 Nov) |
| Notification / camera-ready | 26 Feb 2027 / 19 Mar 2027 |

Source: pakdd2027.org (authors kit and dates pages), EasyChair CFP.

**Anonymity issue.** The repository github.com/AjitKumar01/dials is public and names the author. The submission must
link an anonymised copy (for example anonymous.4open.science) or no code link at all.

## 1. Datasets

These are the 7 datasets with real training bundles. Dunnhumby is excluded (its "bundles" are shopping baskets).

| Domain | Datasets | Features |
|---|---|---|
| Fashion outfits | POG, POG_DENSE | released BLIP image + text |
| Music playlists | Spotify, Spotify_sparse | released CLAP audio + text |
| E-commerce bundles | Electronic, Food, Clothing | TF-IDF of item text (nothing released) |

## 2. The contribution

### The common problem it addresses

The data a bundle-completion system learns from has the same shape everywhere:

- few bundles, against a large and long-tailed catalogue;
- most items in a handful of bundles, and new items in none.

Current neural models (CLHE, RaMen) store what they know about an item in that item's own parameters: an embedding
row, fitted only from the bundles containing it. Model size therefore grows with the catalogue (9.5 M to 67 M
parameters here). Rarely bundled items get weak rows, and new items have only a content view to fall back on.

### The thesis (one sentence)

What makes items go together is mostly **shared across items**: which categories or artists sit together, how
similar the content is, and how often items have been bundled. A model can learn these relations once, with no
per-item parameters, and score every item with the same function.

### Contributions (three, each tied to the thesis)

1. **Method (DIALS).** A bundle-completion model whose 41,624 parameters describe relations, never items. It is a
   query-gated sum of three kinds of evidence:
   - content, a low-rank bilinear map;
   - label relations, split into a learned diagonal ("same category or artist", with a learned sign) plus a low-rank
     part on fixed, cross-fitted SVD factors;
   - degree-discounted co-occurrence.

   It is trained leakage-free: leave-one-out counts and cross-fitted label factors. One configuration is used on all 7
   datasets.
2. **A matched evaluation.** CLHE and RaMen are rerun with their own code on identical splits, identical released
   features and training-only graphs. They sit next to tuned simple and label-aware baselines, with paired tests and
   Holm correction. On the reruns, DIALS is ahead of CLHE and RaMen everywhere. It is ahead of the best simple
   baseline on NDCG@20 on 6 of 7 datasets (Food is not separated). The section also documents two evaluation
   pitfalls in public splits: duplicate bundles shared across train and test, and a user–item graph that contains the
   test bundles.
3. **Guidance for practitioners: when a learned model pays off.** A bundle-budget study (full down to 50 training
   bundles, 5 datasets plus POG fractions):
   - On playlists, DIALS is ahead from 50 bundles up.
   - On small-bundle e-commerce data it needs about 1,000 bundles, and below about 600 a training-free label + kNN +
     content scorer is better.
   - The failure is overfitting, not too few training steps.

   Learned parameters are also readable: the label diagonal is negative for fashion (avoid the seeds' category) and
   positive for music (same artist).

**Not claimed:** "first"; good accuracy on new items (on POG, features find unseen targets only at chance level);
any comparison with or citation of BundleFuse.

## 3. Sections and page budget (12 pages)

| § | Section | Pages | Content |
|---|---|---|---|
| | Abstract | 0.25 | problem → thesis → model → evaluation → main numbers → guidance |
| 1 | Introduction | 1.5 | the task; how current models work; the per-item constraint; the thesis; DIALS in one paragraph; contributions. **Fig. 1** |
| 2 | Related work | 1.0 | bundle generation and completion (CLHE, RaMen, Bundle-MLLM, DDBC, …); category-aware compatibility; simple item-to-item baselines and the reproducibility literature; leakage in evaluation |
| 3 | Method | 3.0 | 3.1 task and notation; 3.2 overview (**Fig. 2**); 3.3 content; 3.4 labels: diagonal + low-rank + cross-fitting; 3.5 graph; 3.6 gate; 3.7 training (leave-one-out, softmax loss); 3.8 cost |
| 4 | Experiments | 4.5 | 4.1 datasets and protocol (Table 1); 4.2 baselines; 4.3 main results (Table 2: R/N, Δ, params); 4.4 how much training data (**Fig. 3**); 4.5 ablations (Table 3); 4.6 what DIALS learns (**Fig. 4**) |
| 5 | Conclusion | 0.5 | findings, limitations (Food, small retail data, new items, catalogue-linear inference), future work |
| | References | 1.25 | |

### Figures

- **Fig. 1 (introduction, in the spirit of CLHE's and RaMen's first figures).**
  - Left: a partial outfit and a partial playlist, with the missing items to fill.
  - Right: where knowledge is stored. A per-item model keeps an embedding table with one row per item, so a new item
    has an empty row. DIALS keeps shared relations (a category × category matrix, a content map, co-occurrence
    counts), so a new item is scored by the same function.
- **Fig. 2 (method).** The architecture: three evidence branches, the gate and the sum. Adapted from
  `figures/architecture.tex`.
- **Fig. 3 (bundle budget).** NDCG@20 against the number of training bundles (log scale): DIALS vs the best simple
  baseline, one panel per domain.
- **Fig. 4 (what is learned).** The learned diagonal δ as a function of label size, for fashion, e-commerce and music,
  with the sign flip visible. A possible second panel: gate weights against how often the seeds were bundled.

## 4. Gaps to close before writing the experiments section

1. **Ablations do not exist yet for the frozen model.** The retrained ablation queue (`bundlefuse/experiments/
   dials_ablations`, 8 parts) stopped on 9 Nov with almost no finished runs. A paper needs Table 3. Proposal: no
   diagonal, no low-rank, no cross-fitting, no leave-one-out, no gate (fixed weights), each term alone, and
   untrained.
   - Cost: 3 seeds on the retail sets and POG_DENSE take minutes to under an hour each.
   - Spotify takes about 1–1.5 h per run uncontended, so it is probably limited to Spotify_sparse, or one seed.
2. **One baseline pool, one selection rule.** RESULTS.md §2 (CF + blend) and the budget test (which adds the
   label-aware baselines) pick different "best simple" methods on Food and POG_DENSE. The paper should use one pool
   (all baselines), picked on validation, everywhere.
3. **How to report RaMen's published e-commerce numbers.** The facts: a different, unreleased split, and a released
   user–item file that contains the test bundles' items. Proposal: state both neutrally in §4.1, report our reruns
   with a training-only graph, and give the leaky-graph runs in one sentence. No accusation, and no claim that the
   leak explains their number (not reproduced).
4. **DDBC and Bundle-MLLM protocols.** Probably one short paragraph with numbers, not tables, given the page limit.
