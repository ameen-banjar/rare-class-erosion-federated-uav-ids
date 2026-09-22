"""
Item 5 data preparation (per DESIGN_FROZEN.md sections 2, 3, 6): builds the
cleaned CICIoT2023 corpus EXACTLY once (drop NaN/Inf rows -> remove
cross-label-conflicting fingerprints -> exact within-class dedup), then a
stratified 70/15/15 train/validation/test split (SPLIT_SEED=2023), and
caches all three splits to disk so run_item5.py never repeats this
expensive step. Read/clean logic mirrors dataset_audit/run_fingerprint_audit.py
exactly (same fingerprint definition) so the resulting counts match the
frozen design's audited numbers precisely.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("/Users/abanjar/All Project/UAV-FL/uav-federated-ids/development-workspace/ciciot2023_dataset/CSV")
OUT_DIR = Path(__file__).resolve().parent / "prepared_data"
SPLIT_SEED = 2023
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def main():
    OUT_DIR.mkdir(exist_ok=True)
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))

    frames = []
    t0 = time.time()
    for i, path in enumerate(csv_files, 1):
        class_name = path.parent.name
        df = pd.read_csv(path, low_memory=False)
        feat_cols = list(df.columns)  # schema verified identical across all files (dataset_audit)

        feat64 = df.to_numpy(dtype=np.float64)
        bad = np.isnan(feat64).any(axis=1) | np.isinf(feat64).any(axis=1)
        df = df.loc[~bad].copy()

        # Fingerprint computed at the SAME precision pandas read the CSV at
        # (float64), matching dataset_audit/run_fingerprint_audit.py exactly
        # -- this must byte-match the frozen design's audited cleaned-corpus
        # counts (DESIGN_FROZEN.md Section 2). Casting to float32 BEFORE
        # fingerprinting (an earlier version of this script did) creates
        # extra false collisions among nearly-identical high-volume flood
        # rows and undercounts the cleaned corpus by ~2.6% -- confirmed by
        # comparing against the audit's own numbers before this fix.
        df["fp"] = pd.util.hash_pandas_object(df[feat_cols], index=False).to_numpy(dtype=np.uint64).astype(np.int64)
        df["cls"] = class_name
        df[feat_cols] = df[feat_cols].astype(np.float32)
        frames.append(df)
        if i % 50 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}] read, elapsed={time.time()-t0:.0f}s")

    full = pd.concat(frames, ignore_index=True)
    del frames
    print(f"Rows after NaN/Inf drop: {len(full)}  elapsed={time.time()-t0:.0f}s")

    # Cross-label conflicts: same fingerprint under >1 class -> remove entirely
    nunique_cls = full.groupby("fp")["cls"].transform("nunique")
    conflict_mask = nunique_cls > 1
    n_conflict_rows = int(conflict_mask.sum())
    full = full.loc[~conflict_mask].copy()
    print(f"Removed {n_conflict_rows} cross-label-conflicting rows. Remaining: {len(full)}")

    # Exact within-class dedup
    before = len(full)
    full = full.drop_duplicates(subset=["fp", "cls"]).copy()
    print(f"Removed {before - len(full)} exact within-class duplicate rows. Cleaned corpus: {len(full)}")

    full = full.drop(columns=["fp"]).reset_index(drop=True)
    class_counts = full["cls"].value_counts()
    print("\nCleaned per-class counts (must match dataset_audit/cleaned_class_prevalence.csv):")
    print(class_counts.sort_values(ascending=False).to_string())

    # Stratified 70/15/15 split
    train_df, temp_df = train_test_split(full, train_size=TRAIN_FRAC, stratify=full["cls"],
                                          random_state=SPLIT_SEED)
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(temp_df, train_size=rel_val, stratify=temp_df["cls"],
                                        random_state=SPLIT_SEED)

    print(f"\nSplit sizes: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUT_DIR / "validation.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)

    with open(OUT_DIR / "feature_columns.txt", "w") as f:
        f.write("\n".join(feat_cols))
    with open(OUT_DIR / "classes.txt", "w") as f:
        f.write("\n".join(sorted(full["cls"].unique())))

    print(f"\nSaved -> {OUT_DIR}/{{train,validation,test}}.parquet")
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
