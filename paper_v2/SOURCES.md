# Where every number in paper_v2 comes from

Build: `pdflatex main.tex` twice. Figures: `python paper_v2/figdata/make_figs.py` from the repository root.

| Paper element | Source |
|---|---|
| TRIAD R@20 / N@20 (Table 2) | `runs/summary.json` (`python -m triad test`), RESULTS.md §1 |
| Simple baselines, CLHE, RaMen rows and their parameter counts (Table 2); RaMen user-item-file numbers | the earlier paper's tables (`paper/main.tex`, Table 2 and Results), unchanged runs: baselines from `triad/baselines.py` (reproduces them exactly), CLHE/RaMen reruns kept outside this repository (`bundle-dpo/clhe_out`) |
| TRIAD minus best simple (Table 3) | `bundlefuse/results/small_data/table_final.json` (tests with `versus = best_simple`), Holm recomputed over the 7 datasets |
| Content term alone (§5) | `paper_v2/figdata/content_only.py` (printed output recorded in its docstring) |
| Training-set size (Fig. 3, §5) | `results/training_size.json` (`python -m analysis.tables training_size`) |
| Knock-outs (Table 4) | `results/readings.json` (`python -m analysis.readings`), validation only |
| Learned values in "What TRIAD learns" | `results/readings.json` (label-rule ratios from `label_rule[0].m1`, the three runs), `results/readings_extra.json`; TRIAD_DESIGN.md §3 |
| Datasets (Table 1) | `triad/data.py`; the earlier paper's Table 1 |
| Halved learning rates, Food sd .027 to .004 | RESULTS.md §5 |
| Parameter count 74,136 | `TRIAD.parameter_count()`, `tests/test_triad.py` |
| Fig. data properties (a)-(c), "What the data shows", abstract shares, Table 1 Unseen | `results/data_facts_test.json` (`python -m analysis.data_facts_test`, clean test queries); figure drawn by `figdata/make_figs.py` |
| RaMen with the paper's settings on Electronic (.334 against .537) | bundlefuse `results/ramen_retail/electronic_published_blip/lr0.001_gat2_gamma0.2_batch1024_seed*` (gamma 0.2, H 4, N 2, lr 1e-3, batch 1024; RaMen Table 2 for .537) |

Not reported for this model: paired tests of TRIAD against CLHE and RaMen (the paper reports their scores, not
p-values).
| LINK row (Table 2) and the LINK sentence in Baselines | `runs/baselines/{electronic,food,clothing}.json`, key `methods.link` (`python -m triad link`; `triad/baselines.py` `run_link`), the variant chosen on validation (with the prior on all three) |
| Table "Warm and cold clean test queries" (TRIAD, simple, $\Delta$, significance, counts) and its paragraph | bundlefuse `results/small_data/cold_sparse_check.json` (`experiments/small_data/cold_sparse_check.py`); RESULTS.md §6 |
| CLHE columns of that table, and the 87/72/73% vs 60/42/29% ratios | bundlefuse `results/small_data/clhe_cold.json` (`experiments/small_data/clhe_cold.py`), divided by the TRIAD warm/cold values above |
