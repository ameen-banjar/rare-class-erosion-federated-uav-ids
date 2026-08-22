"""
Item 2, steps 6-7 (user protocol): pick the winning config per algorithm
family from the 15-round screening (highest mean val_macro_f1 over the last
3 logged rounds -- NOT Replay-specific), then re-run that single winning
config for 45 rounds, seed=11, validation-only, to confirm convergence
before any 5-seed freeze. held_out_test never touched.
"""
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import item2_optimizer_screening as scr
from data import ISOTFederatedData

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def pick_winners():
    df = pd.read_csv(RESULTS_DIR / "item2_optimizer_screening_round_metrics.csv")
    last3 = df[df["round"].isin([11, 13, 15])]
    means = last3.groupby(["algorithm", "config"])["val_macro_f1"].mean().reset_index()
    winners = means.loc[means.groupby("algorithm")["val_macro_f1"].idxmax()]
    print("=== Screening winners (mean val_macro_f1, rounds 11/13/15) ===")
    print(winners.to_string(index=False))
    return winners


def main():
    winners = pick_winners()
    scr.SCREEN_ROUNDS = 45
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    uniform_weights = [1.0 / len(client_ids)] * len(client_ids)
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_confirm_convergence_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds"])

        for _, row in winners.iterrows():
            algo, config = row["algorithm"], row["config"]
            print(f"\n--- Confirming {algo} ({config}) for 45 rounds ---")
            if algo == "fedavg_sgd":
                lr = float(config.split("=")[1])
                scr.run_fedavg_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w)
            elif algo in ("scaffold_weighted", "scaffold_uniform"):
                lr = float(config.split("=")[1])
                variant = "weighted" if algo == "scaffold_weighted" else "uniform"
                scr.run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, variant, w)
            elif algo == "fednova_sgd":
                lr = float(config.split("=")[1])
                scr.run_fednova_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w)
            elif algo == "fedadam":
                parts = dict(kv.split("=") for kv in config.split(","))
                scr.run_fedadam(data, input_dim, client_ids, row_weights, Xval, yval,
                                 float(parts["client_lr"]), float(parts["server_lr"]), w)
            f.flush()

    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
