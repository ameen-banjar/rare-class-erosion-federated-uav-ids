# ISOT Drone Dataset — Acquisition (not bundled)

The raw dataset is **not included in this repository**. Redistribution rights for the ISOT Drone Dataset CSVs were not confirmed with ISOT Research Lab, so only the code, partition manifests, and derived (already-aggregated) results are published here. This is a standard, expected setup for a reproducibility package built on a third-party dataset — see the Data Availability Statement in the paper.

## Official source

- ISOT Research Lab, University of Victoria: <https://onlineacademiccommunity.uvic.ca/isot/2024/12/05/drone-datasets/>
- Real DJI Tello flight captures (not simulated): ~14 hours attack traffic + ~10 hours normal traffic, extracted into 137 per-session feature CSVs across 10 attack-family categories.
- Follow the official page's instructions to obtain the dataset archive and the extraction password documented there.

## Expected local layout

Once obtained, arrange the 137 session CSVs into 10 category subfolders matching the original archive structure:

```
isot_drone_dataset/
├── DoS/*.csv
├── Injection/*.csv
├── Ip Spoofing/*.csv
├── MITM/*.csv
├── Manipulation/*.csv
├── Password Cracking/*.csv
├── Regular/*.csv
├── Replay/*.csv
├── Unauth/*.csv
└── Video/*.csv
```

Point the code at this directory via the `ISOT_DATA_DIR` environment variable (or place it at `<repo-root>/isot_drone_dataset/`, the default `prepare_isot.py` looks for):

```bash
export ISOT_DATA_DIR=/path/to/isot_drone_dataset
```

## Verifying your copy matches ours

`isot_dataset_file_manifest.json` in this folder records the per-file size and CRC32 checksum of the exact 137 CSVs used to produce every result in this repository (extracted once, 2026-08-15). After placing your copy, verify:

```bash
python3 - <<'EOF'
import json, zlib
from pathlib import Path

manifest = json.load(open("isot_dataset_file_manifest.json"))["files"]
root = Path("isot_drone_dataset")  # or your ISOT_DATA_DIR
mismatches = 0
for entry in manifest:
    p = root / entry["local_path"]
    if not p.is_file():
        print("MISSING:", p); mismatches += 1; continue
    data = p.read_bytes()
    if len(data) != entry["size_bytes"] or (zlib.crc32(data) & 0xFFFFFFFF) != entry["crc32"]:
        print("MISMATCH:", p); mismatches += 1
print(f"{len(manifest) - mismatches}/{len(manifest)} files verified OK")
EOF
```

## Known data-quality issue (already handled in code)

Six columns in the official feature-extraction output (`ts`, `min_duration`, `max_duration`, `sum_duration`, `average_duration`, `flow_idle_time`) are raw Unix epoch timestamps, not duration statistics — since each attack category was captured on a different calendar day, these columns trivially leak session/category identity if left in. `prepare_isot.py` drops them unconditionally (`TIMESTAMP_LEAK_COLUMNS`); every downstream script inherits this fix automatically. `isot_learnability_audit_v2.py` documents the discovery and re-measures genuine (post-fix) learnability: binary attack/normal ROC-AUC=1.0000 driven by legitimate features (payload entropy, rate statistics), 10-class GroupKFold macro-F1 without any timestamp leakage.

## Dataset rights

The ISOT Drone Dataset is the property of ISOT Research Lab, University of Victoria. This repository provides only the code to prepare and partition it, plus checksums to verify a copy obtained from the official source — never the raw data itself.
