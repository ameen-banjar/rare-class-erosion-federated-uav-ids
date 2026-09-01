"""
Construct-validity fix for the RQ3 confirmation-phase erosion table (Table 3),
added post-JNCA-audit. analyze_confirmation.py's `max_local_recall` was
computed from eval_data=="own_client_rows" (the holder's own TRAINING rows),
then compared against `global_recall` measured on VALIDATION rows -- a
mismatched surface relative to RQ1 (Section 5), which measures both local
and global recall on the SAME validation rows.

The raw confirmation-phase mechanistic CSV already contains a matching
eval_data=="validation" row for the local (post-local, pre-aggregation)
model at round 45 (recorded by run_confirmation.py's `local_cache_val`),
so this is a REANALYSIS of already-collected data, not a new experiment --
no retraining, no new checkpoints.

Reproduces exactly the anchor x algorithm strict/practical erosion table
(the source of the manuscript's Table 3), substituting the local side for
validation-matched local recall. Held-out test is never touched (unaffected
either way -- this script never reads it).
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

ANCHORS = [
    (0.1, 30, 102, "heavy_a0.1_nc30"),
    (0.3, 15, 103, "moderate_a0.3_nc15"),
    (1.0, 10, 101, "light_a1.0_nc10"),
]
ANCHOR_LABEL = {(a, n, p): lbl for a, n, p, lbl in ANCHORS}
ALGOS = ["fedavg_sgd", "fednova_sgd"]
PRACTICAL_THRESH = 0.05
KEYS = ["algorithm", "alpha", "n_clients", "partition_seed", "model_seed", "class"]

mech = pd.read_csv(RESULTS_DIR / "item3_confirmation_mechanistic_round45.csv")

_data = ISOTFederatedData()
_data.fit_preprocessing()
_, _yval = _data.get_validation_data()
_val_support = np.bincount(_yval.numpy(), minlength=len(ALL_CATEGORIES))
EVALUABLE_CLASSES = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if _val_support[i] > 0}
NON_EVALUABLE_CLASSES = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if _val_support[i] == 0}
print(f"Evaluable classes ({len(EVALUABLE_CLASSES)}/{len(ALL_CATEGORIES)}): {sorted(EVALUABLE_CLASSES)}")

# ---- KEY CHANGE: local side now comes from eval_data=="validation" (matched
#      rows), not "own_client_rows" (training rows). ----
val_local = mech[mech["eval_data"] == "validation"].copy()

def agg_val_local(g):
    w = g["client_fedavg_weight"].to_numpy()
    wsum = w.sum()
    return pd.Series({
        "n_holders": len(g),
        "max_local_recall_matched": g["recall_post_local"].max(),
        "weighted_local_recall_post_matched": np.average(g["recall_post_local"], weights=w) if wsum > 0 else g["recall_post_local"].mean(),
    })

local_agg = val_local.groupby(KEYS, as_index=False).apply(agg_val_local, include_groups=False)

global_agg = val_local.groupby(KEYS, as_index=False).first()[
    KEYS + ["recall_post_agg"]
].rename(columns={"recall_post_agg": "global_recall"})

unit = local_agg.merge(global_agg, on=KEYS, how="inner")
assert len(unit) == len(local_agg) == len(global_agg), "unit mismatch"

unit["class_evaluable"] = unit["class"].isin(EVALUABLE_CLASSES)
unit["strict_erosion"] = np.where(
    unit["class_evaluable"],
    ((unit["max_local_recall_matched"] > 0) & (unit["global_recall"] == 0)).astype(float), np.nan)
unit["practical_erosion"] = np.where(
    unit["class_evaluable"],
    ((unit["max_local_recall_matched"] >= PRACTICAL_THRESH) & (unit["global_recall"] == 0)).astype(float), np.nan)
unit["anchor"] = unit.apply(lambda r: ANCHOR_LABEL[(r["alpha"], r["n_clients"], r["partition_seed"])], axis=1)

unit.to_csv(RESULTS_DIR / "confirmation_unit_table_MATCHED_VALIDATION.csv", index=False)

rows = []
for anchor in [a[3] for a in ANCHORS]:
    for algo in ALGOS:
        sub = unit[(unit["anchor"] == anchor) & (unit["algorithm"] == algo) & unit["class_evaluable"]]
        k_strict, n = int(sub["strict_erosion"].sum()), len(sub)
        k_prac = int(sub["practical_erosion"].sum())
        rows.append({"anchor": anchor, "algorithm": algo,
                      "strict_erosion_matched": f"{k_strict}/{n}", "strict_rate_matched": k_strict / n,
                      "practical_erosion_matched": f"{k_prac}/{n}", "practical_rate_matched": k_prac / n})
table = pd.DataFrame(rows)
table.to_csv(RESULTS_DIR / "confirmation_table3_MATCHED_VALIDATION.csv", index=False)
print("\n=== Table 3, recomputed with matched validation-side local recall ===")
print(table.to_string(index=False))

# ---- comparison against the original (own_client_rows) version, for the record ----
orig_unit = pd.read_csv(RESULTS_DIR / "confirmation_unit_table_run_class.csv")
orig_unit["anchor"] = orig_unit.apply(lambda r: ANCHOR_LABEL[(r["alpha"], r["n_clients"], r["partition_seed"])], axis=1)
print("\n=== Original (own_client_rows / training-side) Table 3, for comparison ===")
orows = []
for anchor in [a[3] for a in ANCHORS]:
    for algo in ALGOS:
        sub = orig_unit[(orig_unit["anchor"] == anchor) & (orig_unit["algorithm"] == algo) & orig_unit["class_evaluable"]]
        k_strict, n = int(sub["strict_erosion"].sum()), len(sub)
        k_prac = int(sub["practical_erosion"].sum())
        orows.append({"anchor": anchor, "algorithm": algo,
                       "strict_erosion_orig": f"{k_strict}/{n}", "practical_erosion_orig": f"{k_prac}/{n}"})
print(pd.DataFrame(orows).to_string(index=False))

print(f"\nSaved -> {RESULTS_DIR / 'confirmation_table3_MATCHED_VALIDATION.csv'}")
print(f"Saved -> {RESULTS_DIR / 'confirmation_unit_table_MATCHED_VALIDATION.csv'}")
