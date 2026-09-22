"""
CICIoT2023 dataset audit -- Item 5 prerequisite (per user's explicit
instructions: no model fitting, no training, read-only inventory of the
official UNB CSV release before any design freeze).

For every CSV file under the extracted CSV/ tree, computes: row count,
column schema, missing/inf counts, exact-duplicate row count, and a
SHA-256 checksum. Aggregates per-class (folder-name) row totals and a
global column-schema consistency check. Writes a full manifest (JSON)
and a human-readable summary (this script's stdout, redirected to
audit_report.txt by the caller).
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

DATA_ROOT = Path("/Users/abanjar/All Project/UAV-FL/uav-federated-ids/development-workspace/ciciot2023_dataset/CSV")
OUT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = OUT_DIR / "ciciot2023_csv_manifest.json"


def sha256_of(path, bufsize=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files under {DATA_ROOT}")

    manifest = {}
    schema_reference = None
    schema_mismatches = []
    per_class_rows = defaultdict(int)
    per_class_files = defaultdict(int)
    global_row_count = 0
    global_missing_cells = 0
    global_inf_cells = 0
    global_duplicate_rows = 0
    identity_like_cols_found = set()
    IDENTITY_KEYWORDS = ["device", "ip", "mac", "src", "dst", "source", "destination",
                          "addr", "client_id", "node", "host_id", "session_id"]

    t0 = time.time()
    for i, path in enumerate(csv_files, 1):
        rel = str(path.relative_to(DATA_ROOT.parent))
        class_name = path.parent.name

        digest = sha256_of(path)
        size_bytes = path.stat().st_size

        df = pd.read_csv(path, low_memory=False)
        cols = list(df.columns)

        if schema_reference is None:
            schema_reference = cols
        elif cols != schema_reference:
            schema_mismatches.append({"file": rel, "columns": cols})

        for c in cols:
            cl = c.lower()
            if any(k in cl for k in IDENTITY_KEYWORDS):
                identity_like_cols_found.add(c)

        n_rows = len(df)
        n_missing = int(df.isna().sum().sum())
        numeric_df = df.select_dtypes(include=[np.number])
        n_inf = int(np.isinf(numeric_df.to_numpy(dtype="float64", na_value=0.0)).sum()) if not numeric_df.empty else 0
        n_dupes = int(df.duplicated().sum())

        per_class_rows[class_name] += n_rows
        per_class_files[class_name] += 1
        global_row_count += n_rows
        global_missing_cells += n_missing
        global_inf_cells += n_inf
        global_duplicate_rows += n_dupes

        manifest[rel] = {
            "sha256": digest,
            "size_bytes": size_bytes,
            "n_rows": n_rows,
            "n_columns": len(cols),
            "n_missing_cells": n_missing,
            "n_inf_cells": n_inf,
            "n_duplicate_rows_within_file": n_dupes,
        }

        if i % 25 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}] processed, elapsed={time.time()-t0:.0f}s")

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Total CSV files: {len(csv_files)}")
    print(f"Total classes (folders): {len(per_class_rows)}")
    print(f"Total rows (all files): {global_row_count}")
    print(f"Total missing cells: {global_missing_cells}")
    print(f"Total +/-inf cells: {global_inf_cells}")
    print(f"Total duplicate rows (within-file only, not cross-file): {global_duplicate_rows}")
    print(f"Schema (columns, n={len(schema_reference)}): {schema_reference}")
    print(f"Schema mismatches across files: {len(schema_mismatches)}")
    for m in schema_mismatches[:10]:
        print(f"  MISMATCH: {m['file']}")
    print(f"Identity/provenance-like column names found: {sorted(identity_like_cols_found) or 'NONE'}")

    print("\nPer-class row counts (folder = class label):")
    for cls in sorted(per_class_rows, key=lambda c: -per_class_rows[c]):
        print(f"  {cls:30s}  files={per_class_files[cls]:3d}  rows={per_class_rows[cls]:>10d}  "
              f"share={per_class_rows[cls]/global_row_count*100:.4f}%")

    print(f"\nManifest written -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
