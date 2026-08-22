# Item 2 — Five-Seed Validation Summary (LOCKED 2026-08-20)

Source: `item2_five_seed_final_round_metrics.csv` (round 45, the fixed primary metric — no best-checkpoint substitution). Metric is **9-class validation macro-F1** (Password Cracking excluded — absent from the validation split entirely; see Phase-1 audit). 95% CIs use Student's t with n−1 degrees of freedom (n=5 seeds → t=2.776; FedAvg-SGD's n=3 successful seeds → t=4.303), not a normal-approximation 1.96 factor.

## Primary results (round 45, 9-class validation macro-F1)

| Algorithm | n (successful seeds) | Mean ± SD | 95% t-CI |
|---|---:|---:|---|
| FedNova-SGD | 5/5 | 0.5179 ± 0.0145 | [0.5000, 0.5359] |
| SCAFFOLD-weighted | 5/5 | 0.5209 ± 0.0232 | [0.4920, 0.5497] |
| SCAFFOLD-uniform | 5/5 | 0.4741 ± 0.0249 | [0.4432, 0.5050] |
| FedAdam | 5/5 | 0.4293 ± 0.0322 | [0.3893, 0.4694] |
| **FedAvg-SGD** | **3/5** | **0.5238 ± 0.0402 (conditional)** | **[0.4240, 0.6236] (conditional)** |

> **FedAvg-SGD (lr=0.5): divergence rate 2/5 (40%) — seeds 44 and 55 diverged at round 2 (client 0 and client 9 respectively), a documented, reproducible instability, not a missing-at-random dropout. Conditional macro-F1 among the three successful runs: 0.524 ± 0.040, 95% t-CI [0.424, 0.624]. This conditional statistic is NEVER reported alongside the other algorithms' unconditional means without the word "conditional" and the 40% divergence rate stated in the same breath — the two failed seeds are themselves part of this algorithm's true performance profile, not excluded noise.

Naming: this run's `fedavg_sgd` / `fednova_sgd` / `scaffold_*` / `fedadam` all use SGD (or SGD client-steps + Adam server-step for FedAdam) locally. Phase-1's frozen `fedavg` / `fedprox` / `client_uniform` (`results_phase1_frozen/`) used Adam locally — always labeled **FedAvg-Adam** etc. in any comparison table mixing the two runs.

## Rare-class detail (round 45, per seed — see full table in chat log / CSV)

- **Manipulation**: FedNova-SGD is the only algorithm with nonzero recall in all 5/5 seeds (range 0.017–0.601, high variance but never exactly zero). SCAFFOLD-uniform: nonzero in 2/5 seeds. FedAdam, FedAvg-SGD (conditional), SCAFFOLD-weighted: recall ≈ 0 in nearly every seed.
- **Replay**: SCAFFOLD-uniform is the most consistently strong (0.963–0.994 across all 5 seeds). FedNova-SGD strong in 4/5 seeds (0.945–0.971) with one seed-sensitive exception. FedAdam highly variable (0.0–0.683).
- **Seed 22, FedNova-SGD**: recall for Replay drops to 0.245 (vs. 0.945–0.971 in the other four seeds) — but this is **NOT** a general-performance outlier: seed 22's overall 9-class macro-F1 (0.5348) is in fact the *highest* of FedNova-SGD's five seeds. Correct description: **a seed-sensitive failure specific to the Replay class**, not an outlier run. No outlier statistical test has been applied to "seed 22" as a run; none is claimed.

## Stability / cost (round 45)

| Algorithm | CV% of macro-F1 across seeds | Mean wall time/seed | Comm. multiplier |
|---|---:|---:|---:|
| FedNova-SGD | 2.79% (lowest) | ~870s | 1× |
| SCAFFOLD-weighted | 4.46% | ~836s | 2× |
| SCAFFOLD-uniform | 5.25% | ~832s | 2× |
| FedAvg-SGD (conditional, n=3) | 7.67% | ~902s | 1× |
| FedAdam | 7.51% | ~850s | 1× |

## Checkpoints available for the final test pass

23 of 25 planned (algorithm × seed) round-45 checkpoints exist in `checkpoints_five_seed/` — the two FedAvg-SGD divergence cases (seeds 44, 55) have **no round-45 checkpoint** (training stopped at round 2) and are **excluded from the final held-out test evaluation** entirely — no partial/diverged model is ever tested.

## Status: LOCKED

This document is the frozen validation-phase analysis. No hyperparameters, thresholds, or seed exclusions may be changed after this point. The next and final step is a single held-out test evaluation pass using these 23 checkpoints, with no retraining and no post-hoc selection.
