"""
CICIoT2023 global fingerprint audit (Item 5 prerequisite, per user's explicit
request after reviewing run_audit.py's within-file duplicate finding of
~39%). Read-only, no model fitting.

For every row across all 309 CSV files, computes a row fingerprint (hash of
the 39 feature columns as parsed) and its (class, file) provenance. Then,
globally (across all files, not per-file):
  1. Rows with any NaN or +/-Inf feature value (to be dropped, per the
     frozen cleaning rule, before deduplication).
  2. Exact-duplicate fingerprints WITHIN the same class (whether from the
     same file or across different files of that class).
  3. Cross-label conflicts: the same fingerprint appearing under more than
     one class -- these rows are structurally ambiguous for a classifier
     restricted to these 39 features and must be removed entirely from
     the modeling corpus, per the frozen rule (no arbitrary label choice).
  4. The resulting CLEANED per-class row counts (after NaN/Inf drop +
     exact within-class dedup + full removal of cross-label-conflicting
     fingerprints), which is the only prevalence figure the rare-tail
     selection rule may use.

Memory strategy: only (fingerprint:int64, class:category-code:int8,
file:category-code:int16) are kept per row, not the raw feature values,
so ~46.7M rows fit comfortably as three numpy arrays.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("/Users/abanjar/All Project/UAV-FL/uav-federated-ids/development-workspace/ciciot2023_dataset/CSV")
OUT_DIR = Path(__file__).resolve().parent


def main():
    csv_files = sorted(DATA_ROOT.rglob("*.csv"))
    classes = sorted({p.parent.name for p in csv_files})
    class_to_code = {c: i for i, c in enumerate(classes)}

    fp_list, class_code_list, file_code_list = [], [], []
    nan_inf_rows_total = 0
    total_rows_seen = 0

    t0 = time.time()
    for i, path in enumerate(csv_files, 1):
        class_name = path.parent.name
        df = pd.read_csv(path, low_memory=False)
        total_rows_seen += len(df)

        feat = df.select_dtypes(include=[np.number])
        bad_mask = feat.isna().any(axis=1) | np.isinf(feat.to_numpy(dtype="float64", na_value=0.0)).any(axis=1)
        nan_inf_rows_total += int(bad_mask.sum())

        # Fingerprint computed over ALL 39 columns as parsed (order fixed by
        # the verified-identical schema across every file). Rows with
        # NaN/Inf are still fingerprinted here (for completeness of the raw
        # audit) but excluded from the "cleaned" counts below via bad_mask.
        row_hash = pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64).astype(np.int64)

        keep = ~bad_mask.to_numpy()
        fp_list.append(row_hash[keep])
        class_code_list.append(np.full(keep.sum(), class_to_code[class_name], dtype=np.int16))
        file_code_list.append(np.full(keep.sum(), i, dtype=np.int32))

        if i % 25 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}] elapsed={time.time()-t0:.0f}s")

    fp = np.concatenate(fp_list)
    cls = np.concatenate(class_code_list)
    filecode = np.concatenate(file_code_list)
    del fp_list, class_code_list, file_code_list

    print(f"\nTotal rows seen (raw): {total_rows_seen}")
    print(f"Rows with NaN or +/-Inf (dropped before dedup): {nan_inf_rows_total}")
    print(f"Rows remaining after NaN/Inf drop: {len(fp)}")

    df_fp = pd.DataFrame({"fp": fp, "cls": cls, "file": filecode})

    # ---- Cross-label conflicts: same fingerprint, >1 distinct class ----
    grp = df_fp.groupby("fp")["cls"].nunique()
    conflict_fps = grp[grp > 1].index
    n_conflict_fps = len(conflict_fps)
    conflict_rows = df_fp["fp"].isin(conflict_fps)
    n_conflict_rows = int(conflict_rows.sum())
    print(f"\nCross-label-conflicting fingerprints (same feature vector, >1 class): {n_conflict_fps}")
    print(f"Total rows affected by cross-label conflicts (all removed from modeling corpus): {n_conflict_rows}")
    if n_conflict_fps > 0:
        conflict_detail = (df_fp[conflict_rows].groupby("fp")["cls"]
                            .apply(lambda s: sorted(classes[c] for c in set(s))))
        print("Sample conflicting class-sets (up to 10):")
        for fpid, clsset in list(conflict_detail.items())[:10]:
            print(f"  fingerprint={fpid}: classes={clsset}")

    # ---- Clean corpus: drop NaN/Inf rows (done) + drop conflicting rows + dedup ----
    clean = df_fp[~conflict_rows].copy()
    n_before_dedup = len(clean)
    clean_dedup = clean.drop_duplicates(subset=["fp", "cls"])
    n_after_dedup = len(clean_dedup)
    print(f"\nRows after removing cross-label conflicts: {n_before_dedup}")
    print(f"Rows after exact within-class deduplication (final cleaned corpus): {n_after_dedup}")
    print(f"Total duplicate rows removed by dedup (within-class, any file): {n_before_dedup - n_after_dedup}")

    print("\n" + "=" * 90)
    print("CLEANED per-class row counts (NaN/Inf dropped + cross-label conflicts removed + exact dedup)")
    print("=" * 90)
    counts = clean_dedup["cls"].value_counts()
    total_clean = counts.sum()
    rows_out = []
    for code, n in counts.sort_values(ascending=False).items():
        cname = classes[code]
        rows_out.append((cname, int(n)))
        print(f"  {cname:30s}  rows={n:>10d}  share={n/total_clean*100:.4f}%")

    print(f"\nTotal cleaned corpus rows: {total_clean}")
    print(f"Total classes represented in cleaned corpus: {counts.shape[0]} (of {len(classes)})")

    out_csv = OUT_DIR / "cleaned_class_prevalence.csv"
    pd.DataFrame(rows_out, columns=["class", "cleaned_rows"]).to_csv(out_csv, index=False)
    print(f"\nCleaned prevalence table written -> {out_csv}")


if __name__ == "__main__":
    main()
