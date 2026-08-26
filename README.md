# Paper 1 — Rare-Class Knowledge Erosion

Standalone reproducibility package for the first paper in the federated UAV intrusion-detection research program. This repository contains **Paper 1 only** and will receive its own independent release history and Zenodo DOI.

## Paper 1 (this release)

**Diagnosing Rare-Class Knowledge Erosion Under Session-Level Non-IID Federated UAV Intrusion Detection**

Federated parameter averaging can erase rare-attack-class knowledge that individual clients demonstrably learn locally, even while the aggregated model shows strong overall accuracy. This package provides the full session-level (never row-level) non-IID partition protocol over the ISOT Drone Dataset, five federated aggregation algorithm implementations (FedAvg, FedNova, SCAFFOLD-uniform, SCAFFOLD-weighted, FedAdam), and a Dirichlet-α/client-count sensitivity study, with every reported number traceable to a SHA-256-hashed, locked results document.

→ See [`paper1_rare_class_erosion/`](paper1_rare_class_erosion/) — start with [`paper1_rare_class_erosion/REPRODUCE.md`](paper1_rare_class_erosion/REPRODUCE.md).

Locked scientific results (do not treat as a preprint — the manuscript itself is not published here, only the code/data/results package. See below):
- [`RESULTS_ITEM1_ITEM2_LOCKED.md`](paper1_rare_class_erosion/RESULTS_ITEM1_ITEM2_LOCKED.md) — mechanism (RQ1) and five-algorithm comparison (RQ2).
- [`item3_sensitivity/RESULTS_ITEM3_LOCKED.md`](paper1_rare_class_erosion/item3_sensitivity/RESULTS_ITEM3_LOCKED.md) — Dirichlet α / client-count sensitivity (RQ3).
- [`EVIDENCE_MAP.md`](paper1_rare_class_erosion/EVIDENCE_MAP.md) — traces every manuscript claim to its locked source.

## Repository layout

```
uav-federated-ids/
├── paper1_rare_class_erosion/   # this release
├── shared/                      # dataset prep + core FL algorithm code, shared across papers in the series
├── README.md
├── CITATION.cff
├── LICENSE
└── environment.yml
```

Later papers in the research program are maintained in separate repositories. Their plans and unfinished materials are intentionally excluded from this repository.

## What's here vs. what's not

**Included:** ISOT dataset preparation code, the frozen session-level client/validation/test partition manifest, all five federated algorithm implementations, every experiment-runner script, all derived/aggregated results (CSV/JSON), all figures, model checkpoints (small — a 3-layer MLP), SHA-256 integrity manifests, and instructions to reproduce every table and figure.

**Not included:** the raw ISOT Drone Dataset itself (redistribution rights not confirmed — see [`shared/data_prep/README.md`](shared/data_prep/README.md) for the official source and a checksum-verification script against the exact copy used here), the manuscript text/PDF, and materials for any paper beyond the one(s) actually released.

## Environment

```bash
conda env create -f environment.yml
conda activate uav-federated-ids
```

See `paper1_rare_class_erosion/REPRODUCE.md` for dataset setup, smoke test, full experiment commands, and approximate runtime/disk requirements.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code or its derived results, please cite the associated paper once published; citation metadata (including a Zenodo DOI for this specific code release) will be finalized at that point.

## License

Code, configuration, partition manifests, and derived results: MIT (see [`LICENSE`](LICENSE)). The ISOT Drone Dataset is not included and remains the property of its original owners under their own terms.
