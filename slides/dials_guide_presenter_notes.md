# DIALS deck: presenter notes and prepared answers

Deck: `slides/dials_guide.tex` (22 slides; the architecture figure is read from `../figures/architecture.pdf`). Build: run `pdflatex dials_guide.tex` twice from `slides/`.

## The argument in four sentences

1. Using history and descriptive evidence together is not new. CLHE, RaMen and hybrid recommenders all do it.
2. What differs is where the evidence is combined. They fuse it into one item vector before scoring. DIALS keeps
   separate, learned terms (content, labels, co-occurrence) and a per-query weight on each, and it uses label metadata
   that neither neural model uses.
3. Its final step is a weighted sum. The gain is in what is summed:
   - the best tuned sum of standard scorers loses to DIALS;
   - DIALS's own terms with fixed weights keep most of the gain.
4. With 41.6k parameters it beats CLHE and RaMen, rerun on identical splits and features. It does not win on warm
   grocery baskets.

## Slide by slide

**1. Title.** The task in one line, the model in one line, and the four questions the talk answers. Saying the
questions up front shows you expect them.

**2. From BundleFuse to DIALS.** The link to last time, in one slide.
- **Recap:** BundleFuse had no trained scorer. It beat published RaMen on POG and Spotify (+45% and +43% Recall@20).
  That showed how evidence is combined matters more than model size.
- **Walk the table:** each choice BundleFuse fixed by hand, what that could not do, and what DIALS learns instead.
  The row to stress is the category one: similarity can only reward a shared category, but outfits need the
  opposite.
- **Close with:** DIALS keeps BundleFuse's evidence and learns its hand-set choices.
- **If asked for a head-to-head number:** say the two were not run on identical features (BundleFuse used our own
  Chinese title embeddings on POG). The comparisons in this talk are against CLHE, RaMen and simple baselines on
  identical splits and features.

**3. Why DIALS.** The pitch, before any detail. Say the bold line first: bundle completion that still works when
the items have no purchase history, at a fraction of the size of neural models.
- **Without history:** POG, 96% of queries have no co-occurrence evidence: .019 vs CLHE .005. Spotify_sparse
  no-history targets: .263 vs .003.
- **Accuracy:** ahead of CLHE on all 6 rerun datasets and of RaMen on the 3 where it was rerun, +.014 to +.197
  NDCG@20.
- **Cost:** 41.6k parameters, none per item, CPU training in minutes; new items scored on day one.
- **Readable:** named terms; the sign of the category term flips by domain.
- **Say the limit yourself:** warm grocery (Dunnhumby) is a tie. The case is new, sparse or changing catalogues.

**4. The task and three kinds of evidence.** Each kind fails somewhere:
- history fails for new or rare items;
- content fails because resemblance is not compatibility;
- labels fail when categories are coarse.

The real question is how much to trust each one, per query.

**5. Is combining evidence new? No.** Say it plainly; it disarms the question.
- **Hybrid CF:** one fixed weight per dataset.
- **CLHE:** attention over an ID embedding, a multimodal vector and a LightGCN vector from the user–item graph, giving
  one vector per item.
- **RaMen:** an item–item graph plus a hypergraph, mixed with fixed constants into one vector.
- **Neither neural model uses labels.**

**6. Separate terms, not one item vector.**
- **Early fusion:** $s=\mathrm{enc}(S)^\top v_c$, where $v_c$ already mixes everything. When the seeds have no
  history, nothing can fall back to content.
- **The evidence, from Spotify_sparse:** on targets with no co-occurring seed, CLHE scores NDCG .003 and DIALS .263.
- **Read the table down a column, not across.** Each column holds different queries; the dip in the middle columns
  is in the queries (larger, mixed bundles), and it is there with co-occurrence switched off too.
- **On the same queries, co-occurrence helps once most targets have history** (+.009, +.044) and costs about .008
  where few do: it promotes items that co-occur with the seeds, and the gate sees only the seeds. Per-candidate
  weighting is the natural next step.

**7. Labels.** Simple rules on category or artist beat CLHE on all four large datasets, so labels are a strong
signal both neural models leave out. DIALS still beats the best such rule:
- +.008 to +.051 NDCG;
- p = .0008 after Holm;
- that rule is tuned on validation and also blends ItemKNN and content.

**8. Each design choice, and its evidence.** This slide answers "why not something simpler?" in advance. Every row
pairs a choice with the simpler alternative and the measurement that rejected it. The two to point at:
- **free label embeddings** memorized the bundles of rare labels;
- **without leave-one-out** a history term drove validation NDCG to .06.

**9. The architecture.** Read the figure bottom to top.
- **Blue boxes (frozen):** computed once from the training data: features $X$, partner history $H=D^{-1}NX$, label
  factors $P_l,Q_l$ from the SVD of $\Gamma_l^\top N\Gamma_l$, and the counts $N$.
- **Orange boxes (trained):** the three terms, each a score for every catalogue item.
- **Gate:** takes three numbers about the seeds and outputs one weight per term.
- **Top:** the weighted sum gives $s(c\mid S)$ for all items; top 20 at evaluation, softmax loss in training.
- **Point to make:** nothing in the orange boxes is a per-item table. That is why new items are scored on day one and
  why the parameter count does not grow with the catalogue.

**10. The score.** One equation, three terms, a gate.
- Point at the sign of $\delta$ (learned) and $M_l$ (16×16).
- One configuration on all 8 datasets.

**11. Leakage-free training.** Leave-one-out, cross-fitting, full-catalogue softmax (no negative sampling, so no
mismatch between training and evaluation).

**12. Against the best simple baseline.**
- **Separated:** DIALS beats the strongest of nine tuned baselines on 6 of 8 datasets.
- **Not separated:** Food (NDCG) and Dunnhumby. They're shown in colour; don't hide them.

**13. Against CLHE and RaMen.**
- **Rerun with their code** on identical split files and the same released features, 3 seeds. All 18 tests
  p = .002.
- **Parameters:** 41.6k against 0.75M–56.5M.
- **RaMen's published e-commerce numbers (.84 R@20)** use an unreleased split and a user–item graph that contains the
  test bundles. Say this only if asked.

**14. With fewer training bundles, and with none.** CLHE's paper motivates itself with "I don't have many bundles",
but never trains on fewer. We did, on POG_DENSE and POG, at 10% and 1% of the training bundles, and with user
histories in place of bundles.
- **Read the POG_DENSE column first:** DIALS keeps its lead at every budget. At 1% it still scores .201, close to
  CLHE trained on everything (.248). Labels and content carry it when co-occurrence is thin.
- **Then POG at 1%:** every method is near zero. Say plainly that DIALS falls slightly behind there; it is a limit.
- **Then user histories:** nobody learns bundles from them, CLHE included. Histories pair substitutes (two dresses);
  bundles pair complements (dress and shoes). So the "no bundles" scenario is open for every method, not just ours.
- **If asked whether the histories leak test bundles:** removing the users who overlap test bundles changes nothing.

**15. Is it just an ensemble?** The main slide for that question.
- **"Standard scorers, tuned weighted sum"** is an ensemble with the same three kinds of evidence. It loses .008–.051.
- **"DIALS terms, fixed weights"** is DIALS's own learned terms with constant weights. It keeps most of the gain.
- **So:** the final step is a sum in every case. What makes DIALS different is that the terms are learned for
  completion.
- **The gate** adds +.016 on Spotify_sparse, where history availability varies most across queries, and little on
  Clothing or POG.

**16. The sign of δ.** Outfits need one item per category, so a shared category counts against a candidate on POG
(δ < 0). A shared artist counts for one on Spotify (δ > 0).
- Nothing tells the model this; one configuration learns both.
- A sum of similarity scorers can only add "same category" with a positive weight.

**17. Where the lead comes from.**
- **POG is a no-history problem:** 96% of queries have zero co-occurrence evidence. There DIALS is 4× CLHE.
- **POG_DENSE:** the lead over CLHE grows as evidence grows (+.014 at 0, +.062 at 4+).
- **The caveat:** on the few POG queries with evidence, EASE beats DIALS.

**18. Knockouts.** Every term matters somewhere, and which one matters depends on the domain:
- the prior on POG_DENSE;
- labels on Spotify_sparse;
- content and labels together on POG.

**19. Practical value.**
- **Size and speed:** 41.6k parameters, none per item. Trains in 12–15 min on CPU for POG; 0.5–3 ms per query.
- **New items:** scored from features and category on day one.
- **Explainable:** every recommendation splits into named terms.
- **Memory, stated honestly:** the co-occurrence matrix dominates, about 260–560 MB on Spotify. That is not smaller
  than CLHE.
- **Training cost of the alternative:** full Spotify CLHE needed about 6.5 GPU-hours per seed, and 2 of 3 seeds did
  not finish in Kaggle's 12-hour limit.

**20. Where DIALS does not win.**
- **Dunnhumby ties.**
- **Personalization helps every method alike:** the household's repurchase counts add about 45% to every method.
- **Food:** NDCG not separated.
- **POG:** the released-feature drop.
- **No bundles:** trained on user histories, every method is near zero.
- **Open runs:** CLHE on full Spotify, and RaMen on POG and Spotify.

**21. Contribution.** Four items, then the disclaimer: the components exist separately; the claim is the combination
and its measured effect. Never say "first". Close on the one line: bundle completion that works without purchase history, with 41.6k parameters.

**22. Backup.** The Bundle-MLLM and DDBC protocols, if asked about LLM or diffusion methods.

## Prepared answers

### "Combining history and description has been done. What is different?"

Agree first: hybrid recommenders, CLHE and RaMen all combine them. The difference is three design choices, each with a
test:
1. **Separate learned terms instead of one fused item vector.**
   - On Spotify_sparse targets with no co-occurring seed: CLHE .003, DIALS .263.
   - On POG's 96% of zero-evidence queries: CLHE .005, DIALS .019.
2. **Labels as a learned relation with a learned sign.**
   - Labels alone beat CLHE.
   - Removing labels drops Spotify_sparse from .256 to .137.
   - δ is negative on outfits and positive on playlists.
3. **Leakage-free training** (leave-one-out, cross-fitting), so the model can use the training bundles' own counts as
   features without memorizing them.

### "Is it just an ensemble?"

In form, the last step is a weighted sum; concede that. What separates it from an ensemble of existing scorers
(slide 15):
- **Same evidence, standard scorers:** the best tuned weighted sum loses by .008–.051.
- **DIALS's learned terms, fixed weights:** keeps most of the gain. The gain is in the terms.
- **Sign flip:** the learned label relation changes sign by domain, which no positive-weighted sum of similarity
  scorers can do.

### "Does such a small model have practical value against CLHE or RaMen?"

On the measured axes, yes:
- more accurate on every dataset where both were run (18 of 18 tests);
- 18–1,350× fewer parameters (Electronic against CLHE up to Spotify_sparse against RaMen);
- trains on a CPU, and scores new items without retraining;
- each score decomposes into named terms.

Be honest that its memory is not smaller, because of the co-occurrence matrix. Also, it ties strong simple baselines
on warm grocery data.

### "So the gate isn't the contribution?"

It is a modest part. It adds up to 6% (Spotify_sparse), where history availability varies most. Most of the gain is
the learned terms: the label relation and its sign, the content maps with partner history, and the degree exponents.

### "Why not a bigger or deeper model?"

The second graph hop was tested and removed:
- level on 5 of 7 datasets;
- .005 NDCG lower on Spotify;
- saved 2.5 minutes and 0.8 GB.

Free label embeddings overfit rare labels. The evidence so far is that the right structure matters more than
capacity. Not tested: a larger content encoder.

### "What about personalization?"

CLHE and RaMen use user data only to build item vectors; neither takes a user at prediction time.
- **Grocery (Dunnhumby):** adding household repurchase helps every method by about 45%, and DIALS + R ties the best
  baseline + R.
- **The literature agrees:** repeat purchases are over 97% of next-basket performance (Reality Check, TOIS 2023).
  See `docs/grocery_benchmarks.md`.

### "What if the basket is empty?"

With no seeds, only the prior (popularity) is left, plus repurchase if the user is known. Today the code returns NaN
for zero seeds; the fix is a small guard.

### "Are the comparisons fair?"

- **Same data:** CLHE and RaMen were run with their own code on byte-identical split files and the same released
  features, settings chosen on validation, 3 seeds.
- **Same statistics:** a paired cluster bootstrap and sign-flip tests with Holm correction.
- **Simple baselines:** tuned on validation, then scored once on test.
- **Open:** full Spotify has only 1 CLHE seed.

## Sources in the repository

- `dials/RESULTS.md`: sections 1–7, all main tables.
- `dials/ARCHITECTURE.md`: section 14 (learned δ, exponents, gate) and section 15 (design decisions).
- `bundlefuse/results/story_checks/`: `summary.json` (buckets, gate), `knockout.json`, `fixed_gate.json`.
- `bundlefuse/results/label_baselines/comparison.json`: DIALS vs the label rules.
- `bundlefuse/results/dials_user/`: the Dunnhumby personalization and leakage runs.
