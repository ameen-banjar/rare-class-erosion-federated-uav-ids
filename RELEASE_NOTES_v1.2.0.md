## What's new in v1.2.0

This release adds two new pre-registered experimental items to the v1.1.1 reproducibility package, developed for the manuscript's revision to *Computer Networks* (Elsevier). **v1.1.1 (Items 1–3) is unchanged and remains independently citable** at its own DOI (`10.5281/zenodo.22236554`).

### Item 4 — Mechanistic Localization (`item4_mechanistic_localization/`)

A pre-registered counterfactual-aggregation design testing *where* within the model rare-class knowledge is lost: the target classifier row, the full classifier head, or the shared representation. Design frozen in writing (`DESIGN_FROZEN.md`) before any execution, per this project's established discipline.

- **Determinism gate:** passed exactly — 150/150 checks (5 seeds × 6 rounds × 5 holder/class units), maximum absolute parameter difference `0.0` against the locked Item 1 checkpoints, at every single logged round.
- **Result:** a disciplined negative one. No single tested component (target-row-only, full-head-only, representation-only) rescues retention at a majority-of-seeds, valid round. The primary continuous endpoint (ΔMargin) is negative in 179 of 180 tested cells.
- **Engineering note:** the determinism gate initially failed at round 5 across six independent single-pass fix attempts, each producing a bit-for-bit identical failure magnitude — evidence the cause was not a logic bug but an MPS-backend sensitivity to any host-interleaved computation during live training. Resolved by a two-pass execution architecture (`run_item4.py`): Pass A replays the trajectory with zero Item-4-specific computation interleaved; Pass B computes all counterfactual interventions afterward, from checkpointed state, with no live trajectory left to perturb. Full diagnosis documented in the design file's change log.

### Item 5 — Cross-Domain External Validation (`item5_cross_domain_validation/`)

Tests whether the retention-measurement framework — not only the UAV-specific finding — transfers to an independent, real-device IoT intrusion-detection benchmark (CICIoT2023), under a different (row-level, synthetic non-IID) federation protocol.

- **Dataset provenance:** official UNB/CIC CICIoT2023 CSV release, SHA-256-verified archive and full per-file manifest (`dataset_audit/ciciot2023_csv_manifest.json`, 309 files).
- **Cleaning pipeline (frozen before any training):** raw 46,776,700 rows → drop NaN/Inf (1,040 rows) → remove cross-label-conflicting feature vectors (533,448 fingerprints, 14,144,617 rows — overwhelmingly DDoS/DoS same-mechanism pairs sharing an identical 39-feature vector) → exact within-class deduplication (12,029,808 further rows) → cleaned corpus of 20,601,235 rows across all 34 classes.
- **Rare-tail selection:** pre-registered bottom-5-by-cleaned-prevalence rule (not chosen by inspecting any model outcome); membership and rank order confirmed identical before and after cleaning.
- **Protocol:** row-level synthetic Dirichlet label-skew federation (α=0.1, 15 clients — no device/session identifier exists in this release), FedAvg + FedNova, 5 independent partition draws × 3 nested model seeds × 2 algorithms = 30 runs, 20 rounds each. Zero crashes across the full ~39-hour run.
- **Result:** the Recall Retention Gap (RRG) is positive in at least 4 of 5 independent partition draws for **all 50** tested (algorithm, class, round) combinations — 41/50 positive in all 5 partitions. Conditional retention (whether global recall collapses to exactly zero) varies substantially by partition draw, unlike ISOT's near-uniform erosion — reported as a genuine cross-setting difference, not smoothed over.
- **Integrity check:** a post-hoc fingerprint audit confirmed zero train/validation/test leakage for all five rare-tail classes specifically (a small float32-precision artifact, fully documented, is confined to high-volume majority classes and does not affect any reported retention result).

### What's excluded from this release (and why)

Large derived/intermediate artifacts are not committed, consistent with this package's existing policy of not redistributing large third-party-derived data:
- CICIoT2023 cleaned/split parquet files (622MB, including one 435MB file over GitHub's 100MB hard limit) — reproducible byte-for-byte from the included cleaning script (`item5_cross_domain_validation/prepare_ciciot.py`) and the official CICIoT2023 archive (hash-verified in the manifest).
- Item 4's intermediate per-round Pass-A checkpoints (32MB) — reproducible from `run_item4.py`; the scientific results themselves (`item4_raw.csv`, `item4_determinism_gate.csv`) are included in full.

### Citation

See `CITATION.cff` (GitHub renders a "Cite this repository" button automatically). This release corresponds to the manuscript *Rare-Class Knowledge Retention in Federated Intrusion Detection: A Session-Level UAV Diagnostic with Mechanistic Localization and Cross-Domain Validation*, prepared for submission to *Computer Networks* (Elsevier).
