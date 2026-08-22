# Paper 1 — Locked Results: Item 3 (Dirichlet α / Client-Count Sensitivity)

**Status: LOCKED 2026-08-22.** This document freezes the scientific narrative and all tables/figures for Item 3. It does not alter or reopen any number in `../RESULTS_ITEM1_ITEM2_LOCKED.md` (Item 1+2, locked 2026-08-20, integrity-tracked in `../LOCK_MANIFEST.json`). Item 3 tests how the erosion phenomenon documented in Item 1+2 *varies* with Dirichlet α and client count — it does not re-derive or re-test whether the phenomenon exists.

---

## 1. Research Question

**RQ3.** How does rare-attack-class knowledge erosion (established at one implicit α, one fixed 15-client partition in Item 1+2) vary with the degree of Non-IID heterogeneity (Dirichlet α) and client count? At what point/condition does the phenomenon begin to appear, and how does it relate to class concentration, holder count, and client weight? This is a **sensitivity analysis, not an exhaustive confirmatory test** — the goal is to characterize the onset condition, not to prove the phenomenon occurs in every configuration.

## 2. Design Summary

Full design frozen before any execution in `DESIGN_FROZEN.md` (2026-08-20); analysis plans frozen *before* opening results in `ANALYSIS_PLAN_FROZEN.md` (exploratory, 2026-08-21) and `CONFIRMATION_ANALYSIS_PLAN_FROZEN.md` (confirmation, 2026-08-22).

- **Data:** ISOT Drone Dataset, same fixed validation (18 sessions) and test (28 sessions) pools as Item 1+2, from the original frozen manifest — **never redistributed**. Only the 91 training-pool sessions are re-partitioned into new client sets. **Item 3 never touches `held_out_test`** — it is a validation-only sensitivity analysis end-to-end.
- **Seed separation (binding throughout):** `partition_seed` controls the Dirichlet draw only; `model_seed` controls network initialization and batch shuffling only, on an already-fixed partition. No file in Item 3 has an ambiguous single "seed" column.
- **Two phases:**
  1. **Exploratory grid** — α ∈ {0.1, 0.3, 1.0} × clients ∈ {10, 15, 30} = 9 combinations × 3 `partition_seed`s (101/102/103) × 2 algorithms (FedAvg-SGD lr=0.5, FedNova-SGD lr=0.3) = **54 runs**, `model_seed=11` fixed throughout (isolates partition-driven variance from init variance). 45 rounds each, validation only.
  2. **Confirmation phase** — 3 anchor partitions selected from exploratory diagnostics only (procedure in §5), × 2 algorithms × 5 `model_seed`s (11 repeated for a determinism check + 22/33/44/55 new) = **30 planned runs, 29 completed** (one divergence, §7).
- **Algorithm scope, locked justification (verbatim, used in the manuscript):**
  > Following the locked Item-2 findings and before observing any Item-3 results, the exploratory sensitivity grid was restricted to FedAvg-SGD and FedNova-SGD. These methods provide a matched-optimizer, matched-communication comparison between standard parameter averaging and the most stable heterogeneity-aware baseline. FedProx and SCAFFOLD were excluded from the full factorial grid to avoid optimizer and communication-cost confounding.
- **Class definition, not fixed in advance:** unlike Item 1+2 (Manipulation/Replay fixed), Item 3 does not pre-designate a "rare" class. All 10 classes are scored at every partition; rarity is a *consequence* of that partition's Dirichlet draw, not a manual choice.

## 3. A Documented Design Deviation, Corrected in the Confirmation Phase

`run_exploratory_grid.py` recorded only round-level, post-aggregation global performance. The original design promised a local-to-global diagnostic ("is local recall positive while global recall is zero") that was never implemented for the 54 exploratory runs — a real deviation from the original promise, not merely unavailable data. **The exploratory grid was not rerun to fix this.** Instead, the confirmation phase's 6 `model_seed=11` reruns (identical seed, on the same 3 anchor partitions already selected from diagnostics-only exploratory evidence) added the missing mechanistic diagnostic prospectively, and their round-45 macro-F1 was required to match the exploratory value **exactly** before being trusted (§6).

## 4. Two Independently Caught Data-Integrity Issues (Corrected Before Locking)

Both were caught and fixed **after** first-pass analysis and **before** this document was written — no scientific conclusion below rests on the uncorrected numbers.

### 4.1 Password Cracking has zero validation support

`Password Cracking` has **0 rows** in the fixed validation split (independently verified: `np.bincount` over `yval` = 0 for this class). `evaluate()`'s `per_class_recall` (and the confirmation phase's `all_class_predictions_and_margins()`) compute recall over **all 10 classes**, not just the label set used for macro-F1 — so this class's recorded recall is **exactly 0.0 by sklearn construction, in literally every run**, regardless of what was learned locally. Any statistic keyed on "recall == 0" therefore miscounted it as erosion/zero-recall unconditionally.

This affected **both phases**, independently confirmed in each:
- **Confirmation phase:** every one of 29 runs' Password Cracking row had `global_recall=0`, `margin=NaN`, `strict_erosion=1`.
- **Exploratory phase:** all 54 round-45 rows had Password Cracking recall exactly 0.0, contaminating `zero_recall` in Q3, Q3b, Q4, Q5, Q5b, Q6.

**Fix (identical logic in both `analyze_confirmation.py` and `analyze_exploratory.py`):** a class is **evaluable** only if it has nonzero true support in validation, computed from data directly (not hardcoded by name — one class fails this test in this dataset: Password Cracking). Excluded from every erosion-rate denominator, correlation, and rescue-count computation in both phases. `Q1` (realized concentration), `Q2` (holder counts), and the confirmation-anchor composite-score selection **do not use recall and are unaffected** — the 3 selected anchor partitions are unchanged after the fix (verified: identical `partition_seed`s 102/103/101 re-selected).

| Metric | Before fix | After fix |
|---|---:|---:|
| Confirmation strict erosion, heavy anchor | 10/50, 11/50, 4/40 (FedAvg); 18/50, 14/50, 8/50 (FedNova) | **5/45, 6/45, 0/36 (FedAvg); 13/45, 9/45, 3/45 (FedNova)** |
| Confirmation FedAvg-zero rescue count | 0/26 | **0/12** |
| Exploratory Q4 (zero despite holders) | FedAvg 79/270=0.2926, FedNova 68/270=0.2519 | **FedAvg 52/243=0.2140, FedNova 41/243=0.1687** |
| Exploratory Q3b partition-level (n=27, real independent units) | gini ρ=0.652, holder_fraction ρ=-0.631, hhi ρ=0.097 (p=0.63, not significant) | **gini ρ=0.668, holder_fraction ρ=-0.630, hhi ρ=0.097 — direction/conclusion unchanged**, since the excluded class's concentration metrics enter a partition mean already dominated by the other 9 classes |

Password Cracking is still reported in Table 3 (confirmation) labeled `class_evaluable=False`: its local retention is a valid observation (models do fit it on the client's own rows), only its global generalization is unmeasurable.

### 4.2 Non-independence in pooled significance tests

Two related pseudo-replication errors, caught on review before locking:

- **Class-row pooling (both phases):** rows for the 10 classes within one partition, and rows for the 2 algorithms sharing a partition, are not independent observations. Any p-value computed on the pooled table (270/243 exploratory rows, 261 confirmation rows) is **descriptive context only** — never cited as a significance result. The exploratory table's p-value column is labeled `p_value_descriptive_only`; the confirmation table's is labeled `naive_unadjusted_p_do_not_report`.
- **Seed-repetition across anchors (confirmation phase only):** the confirmation phase's unified "all anchors combined" paired test initially pooled 14 (anchor × seed) rows as if independent, when the same 5 `model_seed`s repeat across up to 3 anchors each (p=0.00037, invalid). Corrected to average each seed's diff **only across seeds present in all 3 anchors** (`model_seed` 33 excluded — its matching FedAvg run diverged in the light anchor, so it has only 2/3 anchors and is not a like-for-like average with the rest): **n=4 (11, 22, 44, 55), Wilcoxon exact two-sided floor p=0.125**, all 4 seeds same direction.

Both fixes are documented in full, with exact before/after numbers, in `ANALYSIS_PLAN_FROZEN.md`'s and `CONFIRMATION_ANALYSIS_PLAN_FROZEN.md`'s changelog sections. **No new training. `item3_exploratory_round_metrics.csv` and `item3_confirmation_mechanistic_round45.csv` (the two locked source files) were never modified** — both fixes are re-derivations from the same source data, verifiable by re-running `analyze_exploratory.py` / `analyze_confirmation.py`.

## 5. Exploratory Grid Results (27 partitions, `model_seed=11` fixed)

**Q1 — does lower α raise realized concentration?** Yes, monotonically within each client count (`analysis_q1_alpha_vs_realized_concentration.csv`): mean HHI at 10 clients falls from 0.622 (α=0.1) → 0.560 (α=0.3) → 0.378 (α=1.0); at 30 clients from 0.483 → 0.333 → 0.308. **Realized concentration is not a clean function of nominal α alone** — HHI range across the 3 partition_seeds at fixed (α, n_clients) is as wide as 0.21 (α=0.1, 10 clients), confirming the design's premise that nominal α is an insufficient descriptor and realized diagnostics must be reported alongside it.

**Q2 — does more clients reduce holder count?** Mean `n_holders` per class rises slightly with client count in absolute terms (e.g. α=0.1: 2.4 at 10 clients → 4.3 at 30 clients) but **`holder_fraction` (=n_holders/n_clients) falls**, e.g. α=0.1: 0.243 → 0.142 — more clients means a smaller *share* of the network holds any given class, even though the raw count rises (`analysis_q2_nclients_vs_holders.csv`).

**Q3 — predictor ranking for zero-recall (exploratory pooled, descriptive only; n=243 class-runs, not independent):** `holder_weight_share` (ρ=-0.481), `holder_fraction` (ρ=-0.375), `gini` (ρ=0.311), `n_holders` (ρ=-0.267), `hhi` (ρ=0.187), `n_clients` (ρ=0.122), `alpha` (ρ=-0.083, weakest). **Q3b — partition-level check, n=27 real independent units (the trustworthy version):** `gini` (ρ=0.668, p=0.0001), `holder_fraction` (ρ=-0.630, p=0.0004), `holder_weight_share` (ρ=-0.524, p=0.005) are robust; **`hhi` is NOT significant at the correct unit of analysis** (ρ=0.097, p=0.63) — its apparent pooled-level signal (ρ=0.187) was pseudo-replication, not a real partition-level effect. **This is the headline methodological finding of the exploratory phase**: nominal α (ρ=-0.083, weakest of all 7 predictors even pooled) is a poor proxy for the phenomenon; gini and holder_fraction/holder_weight_share are the reliable realized-concentration signals.

**Q4 — given a class has ≥1 local holder, is global recall still zero at round 45?** FedAvg-SGD: 52/243 = **21.4%**. FedNova-SGD: 41/243 = **16.9%**. Having a local holder does not guarantee the class survives aggregation for roughly 1 in 5 (FedAvg) or 1 in 6 (FedNova) class×partition combinations.

**Q5/Q5b — does FedNova reduce zero-recall relative to FedAvg on identical partitions?** Paired by (partition, class), n=243: both zero 18, FedAvg-only zero (candidate rescue) 34, FedNova-only zero (candidate hurt) 23, neither zero 168. Correct unit of analysis — per-partition rate difference, n=27 partitions: mean diff (FedAvg−FedNova) = **+0.0453**, median 0.0, cluster-bootstrap 95% CI **[-0.0082, 0.0989]** (includes zero), paired sign-flip permutation p=0.1314. **Locked framing:** *"observed modest net reduction in zero-recall cases"* — descriptive, not a confirmed effect (CI includes zero).

**Q6 — is FedNova's rescue pattern concentration-dependent?** Restricted to FedAvg-zero cases only (n=52, per the corrected Q4 above): rescued (n=34, FedNova recall>0) mean HHI=**0.376**; both-remained-zero (n=18) mean HHI=**0.584**. **Locked framing (exact wording, binding for the manuscript):** *"FedNova rescue was descriptively concentrated in moderately concentrated class allocations, while highly concentrated cases were rarely rescued."* Gini/HHI/holder_fraction/holder_weight_share are never called causes anywhere in this document — only "strongest stable exploratory associations."

## 6. Confirmation-Anchor Selection (Diagnostics-Only, No Cherry-Picking)

Composite heterogeneity z-score computed **from diagnostics only** (JS-divergence mean/max, mean HHI, client-size CV, negative mean n_holders — never from model performance), pre-registered before any result was seen. For each of the 3 pre-registered (α, n_clients) cases, the **median**-scoring `partition_seed` of its 3 available draws was selected (avoiding both extremes deliberately):

| α | Clients | Selected `partition_seed` | Composite score (of 3) |
|---:|---:|---:|---:|
| 0.1 | 30 | **102** | 0.006 (median of −0.883, 0.006, 0.877) |
| 0.3 | 15 | **103** | −0.112 (median of −0.461, −0.112, 0.573) |
| 1.0 | 10 | **101** | −0.269 (median of −0.384, −0.269, 0.653) |

Labeled `heavy_a0.1_nc30`, `moderate_a0.3_nc15`, `light_a1.0_nc10` throughout. Re-verified unchanged after the Password Cracking fix (§4.1).

## 7. Confirmation Phase: Reproducibility, Divergence, Performance

**Determinism check (mandatory safety gate, not a scientific finding):** the 6 `model_seed=11` reruns matched the exploratory grid's round-45 macro-F1 **exactly** (0.0 abs diff) in all 6 (algorithm × anchor) combinations — proof the full pipeline (data loading, partitioning, training, evaluation) is deterministic under fixed seeds, and that adding the mechanistic-diagnostic instrumentation did not alter training dynamics.

**One divergence (reported, not rescued):** `fedavg_sgd, α=1.0, n_clients=10, partition_seed=101, model_seed=33` diverged at round 16. Per the project's standing rule, **hyperparameters were not adjusted**; the run is excluded from every denominator and explicitly reported. 29/30 planned runs completed. FedAvg-SGD: 1/15 = 6.7% divergence overall, **entirely confined to the light anchor (1/5 there; 0/10 in the other two anchors)** — not generalized as a property of FedAvg-SGD independent of the Non-IID regime.

**Validation macro-F1, round 45, mean ± SD across available `model_seed`s** (`confirmation_table1_performance_by_anchor_algo_seed.csv`):

| Anchor | FedAvg-SGD | FedNova-SGD |
|---|---|---|
| heavy_a0.1_nc30 (n=5/5) | 0.4544 ± 0.0119 | 0.4047 ± 0.0137 |
| moderate_a0.3_nc15 (n=5/5) | 0.4491 ± 0.0231 | 0.4487 ± 0.0335 |
| light_a1.0_nc10 (n=4/5 FedAvg, conditional) | 0.5622 ± 0.0368 | 0.5106 ± 0.0417 |

**Important scope caveat:** unlike Item 2's headline "FedNova has the strongest overall performance," Item 3 only measures **validation** macro-F1 (test is never touched, by design — §2). On this validation-only measure, **FedAvg-SGD's mean is at or above FedNova-SGD's in all three anchors**, opposite of Item 2's *test*-set ranking. Item 3 does not reproduce or contradict Item 2's test-performance finding — the two measure different splits by design. The "erosion despite strong aggregate performance" framing (§9) rests specifically on **Item 2's separately-locked test results**, not on any performance ranking established within Item 3 itself.

## 8. Confirmation Phase: Strict and Practical Erosion (Password-Cracking-Excluded)

**Definitions (frozen in `CONFIRMATION_ANALYSIS_PLAN_FROZEN.md`):** strict erosion = 𝟙[max_local_recall > 0 ∧ global_recall = 0]; practical erosion = 𝟙[max_local_recall ≥ 0.05 ∧ global_recall = 0] (guards against treating a negligible local recall as "real" local knowledge). Both always reported together. Unit of analysis: (partition, algorithm, model_seed, class) — holder rows aggregated within that unit before any statistic.

| Anchor | Algorithm | n seeds | Strict erosion | Practical erosion |
|---|---|---:|---:|---:|
| heavy_a0.1_nc30 | FedAvg-SGD | 5 | **5/45 = 11.1%** | 4/45 = 8.9% |
| heavy_a0.1_nc30 | FedNova-SGD | 5 | **13/45 = 28.9%** | 13/45 = 28.9% |
| moderate_a0.3_nc15 | FedAvg-SGD | 5 | **6/45 = 13.3%** | 6/45 = 13.3% |
| moderate_a0.3_nc15 | FedNova-SGD | 5 | **9/45 = 20.0%** | 9/45 = 20.0% |
| light_a1.0_nc10 | FedAvg-SGD | 4 (conditional) | **0/36 = 0%** | 0/36 = 0% |
| light_a1.0_nc10 | FedNova-SGD | 5 | **3/45 = 6.7%** | 3/45 = 6.7% |

**FedNova-SGD shows a higher strict-erosion rate than FedAvg-SGD in all three anchors.** This is the central, corrected confirmation-phase finding — see §9 for the statistical framing.

**Rescue analysis (restricted to FedAvg-zero cases, evaluable classes only):** of 12 FedAvg zero-global-recall cases with a FedNova match, **0 were rescued (0/12)**. The exploratory phase's modest net-reduction pattern (§5, Q5) and its concentration-dependent rescue pattern (§5, Q6) **did not transfer to the three confirmation anchors** — the confirmation phase found no rescue events at all to characterize by concentration.

## 9. The Central Confirmation-Phase Statistical Finding

Paired FedAvg-vs-FedNova comparison on identical (partition, model_seed) pairs, strict-erosion rate as the paired unit, evaluable classes only (`confirmation_table4_paired_fedavg_vs_fednova.csv`):

| Comparison | n | Mean diff (FedAvg−FedNova) | Wilcoxon p |
|---|---:|---:|---:|
| heavy_a0.1_nc30 | 5 | −0.178 | 0.125 (exact floor at n=5) |
| moderate_a0.3_nc15 | 5 | −0.067 | 0.25 |
| light_a1.0_nc10 (common seeds only) | 4 | −0.083 | 0.25 |
| **Unified, full-case seeds only (11,22,44,55 present in all 3 anchors)** | **4** | **−0.102** | **0.125 (exact two-sided floor at n=4)** |

All 4 full-case seeds show FedAvg < FedNova in strict-erosion rate — a unanimous direction — but at n=4 independent seeds the exact two-sided Wilcoxon floor is 2/2⁴=0.125, so this does not clear a conventional significance threshold. `model_seed=33` is shown separately (present in only 2/3 anchors, since its matching FedAvg run diverged) and excluded from the unified test as not a like-for-like average.

**Locked summary wording for the manuscript (final, supersedes an earlier draft in `CONFIRMATION_ANALYSIS_PLAN_FROZEN.md` that used the pre-fix n=5/p=0.0625 version):**

> Across the three prespecified confirmation anchors, FedNova showed descriptively higher local-to-global erosion than FedAvg. All available seed-level contrasts pointed in the same direction; however, the small number of independent seeds and one incomplete FedAvg condition preclude a confirmatory significance claim.

## 10. Local Fitting vs. Global Generalization — A Binding Interpretive Distinction

`own_client_rows`-based recall/margin (the source of every "local" quantity in Table 3) is evaluated on **the same rows the client just trained on that round** — it is **local fitting/retention evidence**, not independent local-generalization evidence. It answers "did the client's own model still fit its own data," not "would this model generalize to unseen data from a similar distribution." Only `validation`-based recall/margin (source of every "global" quantity) constitutes generalization evidence, because validation sessions are held out from all training partitions by design. This distinction must be preserved in the manuscript wherever local vs. global recall is compared — a local-recall number is never cited as evidence of the model's generalization ability, only of what the aggregation step subsequently did or did not preserve.

## 11. Full List of Corrections and Deviations from the Original Design (Consolidated)

1. Local-to-global diagnostic was missing from the 54-run exploratory grid (§3) — a genuine deviation, not merely unavailable data; corrected prospectively in the confirmation phase, not by rerunning the exploratory grid.
2. Confirmation phase expanded from the originally-sketched 24 runs to 30 (24 new + 6 `model_seed=11` reruns) specifically to recover the missing diagnostic without invalidating exploratory-vs-confirmation seed comparability.
3. Password Cracking's 0-validation-support artifact (§4.1) — caught independently in both phases, fixed identically, verified to leave the 3 anchor-selection partitions unchanged.
4. Pooled-row pseudo-replication in both phases' significance tests (§4.2) — p-values relabeled as non-reportable/descriptive; partition-level (exploratory) and seed-level, full-case-only (confirmation) analyses substituted as the trustworthy versions.
5. The confirmation phase's original "all anchors pooled" test (n=14, p=0.00037) was itself non-independent (same seeds repeating across anchors) and was replaced by the seed-level, full-case-only test (n=4, p=0.125) reported in §9.
6. One run diverged (`fedavg_sgd`, light anchor, `model_seed=33`, round 16) — reported per the standing no-rescue rule, never retried with adjusted hyperparameters.

## 12. Limitations and Threats to Validity

- **Construct validity.** "Erosion" is operationalized identically to Item 1 (strict: local>0 ∧ global=0) plus a practical variant with a 0.05 local-recall floor; both reported together throughout.
- **Internal validity.** The confirmation phase's central finding (§9) is directionally consistent (4/4 full-case seeds) but not statistically confirmatory at n=4 — reported explicitly as descriptive, per the locked wording in §9.
- **External validity.** One dataset (ISOT), 3 confirmation anchor partitions chosen by a pre-registered diagnostics-only procedure from 27 exploratory partitions — not an exhaustive characterization of the (α, n_clients) space; the exploratory grid's 27-partition evidence (§5) is the appropriate basis for concentration-related claims (Gini, holder_fraction), not the 3-partition confirmation Table 5, which is explicitly descriptive/illustrative only for those same predictors within just 3 partitions.
- **Statistical conclusion validity.** All p-values from pooled class-row tables (both phases) are non-reportable by design; only partition-level (exploratory, n=27) and seed-level full-case (confirmation, n=4) tests are treated as informative about direction, and neither claims conventional significance at n=4.
- **Validation-only scope.** Item 3 never touches `held_out_test`; §7's performance comparison and its scope caveat make clear that Item 3's own validation-macro-F1 ranking does not reproduce Item 2's test-set ranking, and the two must not be conflated in the manuscript.
- **Password Cracking.** Permanently unmeasurable on this validation split; retained in Table 3 as local-only evidence, excluded from every global/erosion/correlation statistic in both phases, in both this document and all locked CSVs.

## 13. Locked Summary Statement

> Rare-attack-class knowledge erosion is not a fixed property of one partition — it varies systematically with realized class concentration (Gini, holder fraction), not with nominal Dirichlet α alone. Within three prespecified confirmation anchors spanning heavy, moderate, and light Non-IID regimes, FedNova — the algorithm with the strongest previously-documented test-set performance and stability (Item 2) — showed consistently *higher* local-to-global erosion than plain FedAvg, a directionally unanimous but not statistically confirmatory pattern at the available seed count. A federated algorithm can achieve strong aggregate validation/test performance while erasing rare-attack knowledge that some of its clients demonstrably learned — and neither nominal α nor macro-F1 alone is sufficient to detect this failure mode.
