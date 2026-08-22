# Paper 1 — Evidence Map (RQ → Item → Locked Source)

Frozen 2026-08-22, after Item 3 lock. Binding reference for manuscript writing — every claim in the manuscript's Results section must trace to one row below; no number enters the manuscript without a locked source file behind it.

| RQ | Question | Item | Locked source | Status |
|---|---|---|---|---|
| **RQ1** | Does parameter averaging erase rare-attack knowledge that clients demonstrably learn locally, and through what mechanism? | Item 1 | `RESULTS_ITEM1_ITEM2_LOCKED.md` §3 | LOCKED 2026-08-20 |
| **RQ2** | Does erosion persist under heterogeneity-aware (SCAFFOLD) or adaptive (FedNova, FedAdam) algorithms, or is it specific to plain FedAvg? | Item 2 | `RESULTS_ITEM1_ITEM2_LOCKED.md` §4–5 | LOCKED 2026-08-20 |
| **RQ3** | How does the phenomenon vary with Non-IID degree (Dirichlet α) and client count — when does it start, and what does it correlate with? | Item 3 | `item3_sensitivity/RESULTS_ITEM3_LOCKED.md` | LOCKED 2026-08-22 |

## Integrity chain

- Item 1+2: `LOCK_MANIFEST.json` (SHA-256, 2026-08-20).
- Item 3: `item3_sensitivity/ITEM3_LOCK_MANIFEST.json` (SHA-256, 2026-08-22, independent of and never overlapping the Item 1+2 manifest).

## Per-RQ evidence detail

### RQ1 — mechanism (Item 1)
- Local-to-global recall gap, all 300 (seed×round×class×holder×eval-set) observations: `RESULTS_ITEM1_ITEM2_LOCKED.md` §3, Figure 1.
- Directional-cancellation mechanism (cosine similarity, non-holder dominance ~9–18×): §3, Figure 2.
- Locked causal framing (softmax-implicit-suppression, non-exclusive): §3, final blockquote.

### RQ2 — algorithm dependence (Item 2)
- 5-algorithm, 5-seed validation + held-out-test comparison, three corrected test metrics: `RESULTS_ITEM1_ITEM2_LOCKED.md` §4.4–4.5, Figures 3–5.
- Paired statistical comparisons (FedNova significantly ahead on pooled/session-balanced; SCAFFOLD-weighted ahead on hierarchical recall, marginal vs FedNova): §4.6.
- FedAvg-SGD's 40% divergence rate at its own frozen LR (Item 2), reported conditionally throughout: §4.3, §4.4.
- Locked summary statement: §5.

### RQ3 — sensitivity (Item 3)
- Realized-concentration vs nominal-α decoupling (Q1) and holder-fraction dilution with client count (Q2): `item3_sensitivity/RESULTS_ITEM3_LOCKED.md` §5.
- Correct partition-level predictor ranking (Q3b, n=27): gini and holder_fraction/holder_weight_share robust; HHI not significant at the correct unit — §5.
- Zero-recall-despite-holders base rate (Q4) and FedNova's non-significant net reduction (Q5/Q5b, Q6 concentration-dependence): §5.
- Confirmation-anchor selection procedure (diagnostics-only, no cherry-picking): §6.
- Confirmation-phase strict/practical erosion by anchor, Password-Cracking-corrected: §8.
- **Central RQ3 finding** — FedNova shows consistently higher local-to-global erosion than FedAvg across all 3 confirmation anchors, directionally unanimous (4/4 full-case seeds) but not statistically confirmatory (n=4, exact Wilcoxon floor p=0.125): §9.
- Local-fitting-vs-global-generalization interpretive distinction (binding for how any "local recall" number may be described in the manuscript): §10.
- Locked summary statement: §13.

## Cross-RQ framing note (binding)

Item 2's "FedNova has the strongest test-set performance/stability" finding and Item 3's "FedNova shows higher local-to-global erosion" finding are **not measured on the same split** (Item 2: held-out test; Item 3: validation only, by design — test is never touched in Item 3). They are not contradictory, but they must never be merged into a single split-agnostic claim. The manuscript's central message — strong aggregate performance can coexist with rare-class erosion — is exactly the juxtaposition of these two separately-locked, separately-scoped findings, not a single unified metric.
