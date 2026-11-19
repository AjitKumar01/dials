# Bundling and basket completion on Dunnhumby and Instacart: what has been published

Searched on 2026-11-14. "Instamart" (Swiggy) has no public dataset, so this covers **Instacart**, the public grocery
dataset that is the usual partner of Dunnhumby.

**Verification levels:**
- **read:** we read the paper's tables ourselves.
- **abstract:** from the abstract or search snippets; check before citing.
- **unconfirmed:** the dataset use is not confirmed.

## The short answer

- **Bundle completion in our sense** (rank the catalogue to fill a partial bundle, with bundles as the unit of
  training): no published paper on either dataset turned up. The bundle-construction literature (CLHE, RaMen,
  Bundle-MLLM) uses POG, Spotify and the BundleRec e-commerce sets. Bundle-recommendation surveys list Dunnhumby and
  Instacart only as next-basket datasets.
- **Within-basket recommendation** is the nearest published task. A partial basket is given and the next item is
  predicted. It treats the basket as a bundle, and several papers evaluate it on Dunnhumby and Instacart.
- **Next-basket recommendation**, which predicts a user's whole next basket, dominates the use of both datasets. Its
  main lesson for us is that **repurchase drives almost all accuracy** on these datasets. That is the same effect we
  found when we added household repurchase counts (+45% for DIALS, the previous version of TRIAD, and +42% for the blend).
- **Bundle pricing** on Dunnhumby: we found no paper. Pricing work uses the dataset for personalized price or
  willingness to pay, not for bundles.

## 1. Within-basket (basket completion) work: the closest to our task

| Work | Datasets | Task and method | What it reports | Level |
|---|---|---|---|---|
| Wan, Wang, Melchior, McAuley. *Representing and Recommending Shopping Baskets with Complementarity, Compatibility and Loyalty* (triple2vec), CIKM 2018 [1] | **Dunnhumby** (26,780 items, 2,500 users), **Instacart** (42,987 items, 206,209 users), two private grocery sets | Embeddings from (item, item, user) triples within a basket; an "adaLoyal" repurchase layer on top. Next-basket and within-basket recommendation. | Within-basket AUC/NDCG: triple2vec .864/.145 on Dunnhumby and .960/.127 on Instacart; with adaLoyal .879/.185 and .970/.279. **User-wise popularity alone gets NDCG .174 and .272**, above triple2vec without the repurchase layer. | read |
| Warlop, Mary, Gartrell. *Multi-Task Determinantal Point Processes for Recommendation* (basket completion), 2018 [2] | **Instacart** (it records the order items were added), plus other basket data | Directed basket completion: predict the next item given the items already in the basket, with a multi-task DPP. | Mean percentile rank and precision@K. | abstract |
| Gartrell, Paquet, Koenigstein. *Low-rank factorization of DPPs* (basket completion), AAAI 2017 [3]; Gartrell et al., *Deep DPPs*, 2018 [4]; nonsymmetric DPP sampling, 2022 [5] | **Instacart** (3.2M baskets, 49,677 products, baskets >100 items dropped; a 10k-basket sample) | Basket completion as next-item prediction from a set model. | Set-model likelihood and completion metrics. | abstract |
| Liu, Wan, Guo, Achan, Yu. *BasConv*, SDM 2020 [6] | **Instacart**, Walmart | User–basket–item graph; basket recommendation as basket–item link prediction with a graph convolutional network. | Recall, NDCG and HR gains over baselines on Instacart (+7.3%, +1.7%, +9.9%). Code public. | abstract |
| Mantha et al. (Walmart). *A Large-Scale Deep Architecture for Personalized Grocery Basket Recommendations*, 2020 [7] | **Instacart**, Walmart | triple2vec-style embeddings served in real time (RTT2Vec) for within-basket suggestions. | Production-oriented; offline Instacart metrics. | abstract |
| Entezari, Papalexakis et al. *Tensor-based Complementary Product Recommendation*, IEEE BigData 2021 [8] | **Instacart** | RESCAL tensor factorization of same-order co-purchases. | Recall@10 .192 vs .149 for triple2vec. Notes that recommendations often ignore the current basket. | abstract |
| Xu, Yu, Sun, Guo, Yao. *Modeling Within-Basket Auxiliary Item Recommendation with Matchability and Ubiquity* (Matc4Rec), ACM TIST 2023 [9] | Not confirmed (search results pair it with Instacart and Dunnhumby, but the abstract doesn't name datasets) | Directed "primary → auxiliary" items within a basket; models matchability and ubiquity; claims cold-start gains. | Unknown until read. | unconfirmed |
| *Diversifying complementary recommendations for online grocery*, Frontiers in Big Data 2023 [10] | **Instacart** | Complementary-item recommendation with diversity. | Unknown until read. | abstract |

## 2. Next-basket recommendation (the dominant use of both datasets)

| Work | What matters for us | Level |
|---|---|---|
| Li, Jullien, Ariannezhad, de Rijke. *A Next Basket Recommendation Reality Check*, TOIS 2023 [11] | On Dunnhumby and Instacart, **repeat items account for most of the accuracy**; methods are often not better than simple repeat baselines on explore items. Dunnhumby is restricted to its first two months. | abstract |
| Shao et al. *An empirical study of next-basket recommendations*, 2023 [12] | **UP-CF@r and TIFU-KNN** are the best methods on all metrics on Instacart and Dunnhumby. These two are on our list of personalized baselines. | abstract |
| Hu et al. *TIFU-KNN: Modeling Personalized Item Frequency Information for Next-basket Recommendation*, SIGIR 2020 [13] | A simple item-frequency kNN that beats neural next-basket models on these datasets. | abstract |
| Li et al. *Masked and Swapped Sequence Modeling for Next Novel Basket (BTBR)*, RecSys 2023 [14] | "Next novel basket": only items the user has never bought. This is the **explore** case, closest to completing a bundle from items new to the user. | abstract |
| *Are We Really Achieving Better Beyond-Accuracy Performance in NBR?*, 2024 [15]; *Mitigating Frequency Bias in NBR via Deconfounders*, 2022 [16] | Beyond-accuracy and frequency bias on the same datasets. | abstract |

## 3. Bundle surveys and bundle construction

- **The 2024 survey on bundle recommendation** [17] lists Instacart as an online grocery basket dataset. It groups
  Ta-Feng, Dunnhumby and Instacart under *next-basket* studies, not under bundle construction.
- **Bundle construction** [CLHE, RaMen, Bundle-MLLM, 18] defines the task as completing seed items, but none of the
  bundle-construction papers we checked evaluate on Dunnhumby or Instacart.
- **Industry: Instacart's own LLM pipeline** [19] generates "bought together" and substitute lists per search query
  offline. Instacart reports that it misses natural complements, such as soy sauce for "sushi".

## 4. Pricing and promotions on Dunnhumby

- **Personalized pricing.** Recent work on price personalization and willingness to pay uses Dunnhumby's price and
  discount fields [20, 21]. It is not about bundles.
- **A caveat for any price or promotion analysis.** In the Complete Journey user guide, `sales_value` is what the
  retailer received after loyalty and coupon discounts, not the price the shopper paid [22].

## What this means for TRIAD

1. **Dunnhumby as a bundle-completion benchmark is our own construction.** We treat each basket as a bundle,
   sample 30,000 baskets of 3–20 products, and hide half of each held-out basket. No published method has numbers on
   it, so the paper must describe it as a derived benchmark, not a standard one.
2. **The closest published comparators are within-basket models:** triple2vec, the DPP basket completers and
   BasConv. They use different splits and metrics (AUC, MPR, per-user splits), so their numbers can't be compared
   with ours directly. A fair comparison means rerunning one of them on our split; triple2vec and BasConv have public
   code.
3. **Our Dunnhumby limit matches the literature.** On grocery data, repurchase and plain co-occurrence carry most of
   the signal:
   - user-wise popularity beats triple2vec's NDCG [1];
   - repeats dominate next-basket accuracy [11];
   - simple frequency kNNs win [12, 13].

   This fits DIALS (the previous version) tying the content+ItemKNN blend on Dunnhumby, and every method gaining about 45% from household
   repurchase counts.
4. **Instacart adds what Dunnhumby lacks:** product names, aisles and departments (a two-level label hierarchy, like
   artist/album on Spotify), and the order items were added to the basket. That order would allow a directed
   "first items → later items" protocol like the DPP papers use. It remains the natural next dataset.

## Sources

1. Wan et al., CIKM 2018 (triple2vec): https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm18a.pdf ; slides: https://mengtingwan.github.io/paper/cikm18_mwan_slides.pdf
2. Multi-Task Determinantal Point Processes for Recommendation: https://arxiv.org/pdf/1805.09916 ; slides: https://www.slideshare.net/slideshow/multi-task-dpp-for-basket-completion-by-romain-warlop-fifty-five/117316072
3. Low-rank DPP factorization (basket completion): https://arxiv.org/pdf/1608.04245v1
4. Deep Determinantal Point Processes: https://arxiv.org/pdf/1811.07245
5. Scalable MCMC Sampling for Nonsymmetric DPPs: https://arxiv.org/pdf/2207.00486
6. BasConv: https://arxiv.org/pdf/2001.09900 ; code: https://www.github.com/JimLiu96/basConv
7. Large-Scale Deep Architecture for Personalized Grocery Basket Recommendations: https://arxiv.org/pdf/1910.12757v3.pdf
8. Tensor-based Complementary Product Recommendation: https://www.cs.ucr.edu/~epapalex/papers/IEEEBigData2021-complementary-recommendation.pdf
9. Matc4Rec, ACM TIST 2023: https://dl.acm.org/doi/10.1145/3574157
10. Diversifying complementary recommendations, online grocery: https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2023.974072/full
11. A Next Basket Recommendation Reality Check: https://arxiv.org/pdf/2109.14233
12. An empirical study of next-basket recommendations: https://arxiv.org/html/2312.02550
13. TIFU-KNN: https://arxiv.org/pdf/2006.00556
14. BTBR, next novel basket: https://arxiv.org/pdf/2308.01308
15. Beyond-accuracy in NBR: https://arxiv.org/pdf/2405.01143
16. Frequency bias in NBR: https://arxiv.org/pdf/2211.09072
17. A Survey on Bundle Recommendation: https://arxiv.org/html/2411.00341
18. Fine-tuning Multimodal LLMs for Product Bundling (Bundle-MLLM): https://arxiv.org/pdf/2407.11712
19. Instacart LLM complements/substitutes: https://www.zenml.io/llmops-database/using-llms-to-enhance-search-discovery-and-recommendations
20. Price personalization with deep RL (Dunnhumby): https://www.researchgate.net/publication/397313115
21. Transformer-based price personalization: https://doi.org/10.3390/jtaer21080241
22. Dunnhumby source files and Complete Journey guide: https://www.dunnhumby.com/source-files/
