"""
Item 2 -- second (final) boundary extension. All four SGD-based algorithms
picked lr=0.1, the top of the first extension's grid, in the merged
13-15-round screening -- per protocol, a boundary winner blocks freezing.
Tests lr in {0.2, 0.3} for fedavg_sgd, scaffold_weighted, scaffold_uniform,
fednova_sgd only (FedAdam was already bracketed on both axes and is excluded).
15 rounds, seed=11, validation-only. held_out_test never touched.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import item2_optimizer_screening as scr
from data import ISOTFederatedData

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXTENSION2_LRS = [0.2, 0.3]


def main():
    print(f"Device: {scr.DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    uniform_weights = [1.0 / len(client_ids)] * len(client_ids)
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_boundary_extension2_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds"])

        for lr in EXTENSION2_LRS:
            scr.run_fedavg_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()
        for lr in EXTENSION2_LRS:
            scr.run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, "weighted", w); f.flush()
        for lr in EXTENSION2_LRS:
            scr.run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, "uniform", w); f.flush()
        for lr in EXTENSION2_LRS:
            scr.run_fednova_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()

    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
