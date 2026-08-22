"""
Item 2 -- boundary extension (per review: every SGD-based winner picked the
top of the tested LR grid (0.03), and FedAdam picked the extremes of both its
grids (client_lr=0.003 min, server_lr=0.03 max) -- a classic sign the
screening grid did not yet bracket the optimum. This is NOT a freeze; it is
one more 15-round, seed=11, validation-only extension to check whether
performance keeps rising past 0.03 or starts falling (which would bracket
the optimum and justify a freeze).

Extension grid:
  - fedavg_sgd, scaffold_weighted, scaffold_uniform, fednova_sgd: lr in {0.05, 0.1}
  - fedadam: client_lr in {0.001, 0.003} x server_lr in {0.03, 0.1}
    (beta1=0.9, beta2=0.99, tau=1e-3 unchanged)
"""
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import item2_optimizer_screening as scr
from data import ISOTFederatedData

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXTENSION_LRS = [0.05, 0.1]
FEDADAM_CLIENT_LRS = [0.001, 0.003]
FEDADAM_SERVER_LRS = [0.03, 0.1]


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

    out_path = RESULTS_DIR / "item2_boundary_extension_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds"])

        for lr in EXTENSION_LRS:
            scr.run_fedavg_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()
        for lr in EXTENSION_LRS:
            scr.run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, "weighted", w); f.flush()
        for lr in EXTENSION_LRS:
            scr.run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, "uniform", w); f.flush()
        for lr in EXTENSION_LRS:
            scr.run_fednova_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()
        for client_lr in FEDADAM_CLIENT_LRS:
            for server_lr in FEDADAM_SERVER_LRS:
                scr.run_fedadam(data, input_dim, client_ids, row_weights, Xval, yval, client_lr, server_lr, w); f.flush()

    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
