# Session-Level Non-IID Federated UAV Intrusion Detection — Rare-Class Knowledge Retention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22236554.svg)](https://doi.org/10.5281/zenodo.22236554)

Reproducibility package for the manuscript below. This repository receives its own independent release history and Zenodo DOI, version-addressable and separate from the manuscript text itself.

## Manuscript

**Rare-Class Knowledge Retention in Federated Intrusion Detection: A Session-Level UAV Diagnostic with Mechanistic Localization and Cross-Domain Validation**

Federated parameter averaging can erase rare-attack-class knowledge that individual clients demonstrably learn locally, even while the aggregated model shows strong overall accuracy. This package provides:

- **Items 1–3** (mechanism, five-algorithm comparison, heterogeneity sensitivity): the full session-level (never row-level) non-IID partition protocol over the ISOT Drone Dataset, five federated aggregation algorithm implementations (FedAvg, FedNova, SCAFFOLD-uniform, SCAFFOLD-weighted, FedAdam), a mechanism-targeted FedRS restricted-softmax intervention (`fedrs_baseline_mechanism.py`), and a Dirichlet-α/client-count sensitivity study.
- **Item 4** (mechanistic localization): a pre-registered counterfactual-aggregation design (`item4_mechanistic_localization/`) testing whether rare-class erosion localizes to the target classifier row, the full classifier head, or the shared representation, with a mandatory determinism gate passed exactly (150/150 checks, parameter-wise identity) before any result is interpreted.
- **Item 5** (cross-domain external validation): the retention-measurement framework re-tested on CICIoT2023 (`item5_cross_domain_validation/`) — an independent, real-device IoT intrusion-detection benchmark — under a row-level synthetic non-IID federation protocol, including the full dataset audit, deduplication/cleaning pipeline, and Dirichlet partitioning code.

Every reported number is traceable to a SHA-256-hashed, locked results document.

→ See [`paper1_rare_class_erosion/`](paper1_rare_class_erosion/) — start with [`paper1_rare_class_erosion/REPRODUCE.md`](paper1_rare_class_erosion/REPRODUCE.md).

Locked scientific results (do not treat as a preprint — the manuscript itself is not published here, only the code/data/results package. See below):
- [`RESULTS_ITEM1_ITEM2_LOCKED.md`](paper1_rare_class_erosion/RESULTS_ITEM1_ITEM2_LOCKED.md) — mechanism (RQ1) and five-algorithm comparison (RQ2).
- [`item3_sensitivity/RESULTS_ITEM3_LOCKED.md`](paper1_rare_class_erosion/item3_sensitivity/RESULTS_ITEM3_LOCKED.md) — Dirichlet α / client-count sensitivity (RQ3).
- [`item4_mechanistic_localization/DESIGN_FROZEN.md`](paper1_rare_class_erosion/item4_mechanistic_localization/DESIGN_FROZEN.md) — mechanistic localization design (frozen pre-registration, change log, and result summary).
- [`item5_cross_domain_validation/DESIGN_FROZEN.md`](paper1_rare_class_erosion/item5_cross_domain_validation/DESIGN_FROZEN.md) — CICIoT2023 cross-domain validation design (frozen pre-registration, dataset audit, change log).
- [`EVIDENCE_MAP.md`](paper1_rare_class_erosion/EVIDENCE_MAP.md) — traces every manuscript claim to its locked source.

## Repository layout

```
uav-federated-ids/
├── paper1_rare_class_erosion/   # this release
│   ├── item4_mechanistic_localization/
│   └── item5_cross_domain_validation/
├── shared/                      # dataset prep + core FL algorithm code
├── README.md
├── CITATION.cff
├── LICENSE
└── environment.yml
```

## What's here vs. what's not

**Included:** ISOT dataset preparation code, the frozen session-level client/validation/test partition manifest, all five federated algorithm implementations, every experiment-runner script (Items 1–5), all derived/aggregated results (CSV/JSON), all figures, model checkpoints (small — a 3-layer MLP), SHA-256 integrity manifests (including a full per-file manifest for the CICIoT2023 release used in Item 5), and instructions to reproduce every table and figure.

**Not included:** the raw ISOT Drone Dataset itself (redistribution rights not confirmed — see [`shared/data_prep/README.md`](shared/data_prep/README.md) for the official source and a checksum-verification script against the exact copy used here); the raw or cleaned CICIoT2023 CSV release (already public at its official source — see `item5_cross_domain_validation/DESIGN_FROZEN.md` §0 for the exact archive hash and download instructions; the cleaned/split parquet files are large derived intermediates, reproducible byte-for-byte from the included cleaning script); and the manuscript text/PDF.

## Environment

```bash
conda env create -f environment.yml
conda activate uav-federated-ids
```

See `paper1_rare_class_erosion/REPRODUCE.md` for dataset setup, smoke test, full experiment commands, and approximate runtime/disk requirements.

## Citation

See [`CITATION.cff`](CITATION.cff) (GitHub renders this automatically as a "Cite this repository" button). If you use this code, partition protocol, or derived results, please cite both the software release (DOI above) and the associated paper once published.

## License

Code, configuration, partition manifests, and derived results: MIT (see [`LICENSE`](LICENSE)). The ISOT Drone Dataset and CICIoT2023 are not included and remain the property of their original owners under their own terms.
