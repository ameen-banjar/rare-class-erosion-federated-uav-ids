"""
Item 2 -- SCAFFOLD fallback confirmation. Both scaffold_uniform and
scaffold_weighted FAILED the pre-registered 45-round stability check at their
15-round-screening winner (lr=0.2): scaffold_uniform diverged outright at
round 31 (client 6), scaffold_weighted's rounds-36-45 slope (-0.00232) missed
the -0.002 threshold. Per the pre-registered fallback rule, we step DOWN to
the nearest lower tested LR (0.1) rather than extending the grid further.
Same per-client divergence detection, same stability criteria, seed=11,
validation-only, 45 rounds, held_out_test never touched.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import csv
from data import ISOTFederatedData
from item2_final_confirmation import run_scaffold, stability_report, RESULTS_DIR

def main():
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    uniform_weights = [1.0 / len(client_ids)] * len(client_ids)
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_scaffold_fallback_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds",
                     "diverged", "diverged_client_id", "diverged_note"])
        run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, 0.1, "uniform", w); f.flush()
        run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, 0.1, "weighted", w); f.flush()

    df = pd.read_csv(out_path)
    df["diverged"] = df["diverged"].astype(bool)
    stability_report(df)
    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
