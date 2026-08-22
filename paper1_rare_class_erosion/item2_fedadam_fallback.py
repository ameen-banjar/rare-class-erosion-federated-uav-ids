"""
Item 2 -- FedAdam fallback. The screening winner (client_lr=0.003,
server_lr=0.03) FAILED the pre-registered 45-round stability check
(gap=0.03719 > 0.03; slope=-0.00393 < -0.002) -- it peaked at round 22 and
regressed afterward. Testing two direct backup configs at 45 rounds,
seed=11, validation-only, same per-client divergence detection and
stability criteria as item2_final_confirmation.py:
  1. client_lr=0.003, server_lr=0.01  (lower server LR, same client LR)
  2. client_lr=0.001, server_lr=0.03  (lower client LR, same server LR)
If both pass, the higher mean(41-45) wins. If only one passes, it is frozen.
held_out_test never touched.
"""
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import item2_optimizer_screening as scr
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from advanced_algorithms import local_train_generic, aggregate_fedadam
from item2_final_confirmation import log_round, log_diverged, client_diverged, stability_report, RESULTS_DIR, NUM_ROUNDS, SEED, BATCH_SIZE, DEVICE

FEDADAM_BETA1, FEDADAM_BETA2, FEDADAM_TAU = 0.9, 0.99, 1e-3
CANDIDATES = [(0.003, 0.01), (0.001, 0.03)]


def run_fedadam_guarded(data, input_dim, client_ids, weights, Xval, yval, client_lr, server_lr, writer):
    config = f"client_lr={client_lr},server_lr={server_lr}"
    scr.set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    m_state = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in global_model.state_dict().items()}
    v_state = {k: torch.full_like(v, FEDADAM_TAU ** 2, dtype=torch.float32, device="cpu") for k, v in global_model.state_dict().items()}
    t0 = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states = []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=client_lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            if client_diverged(state):
                log_diverged(writer, "fedadam", config, rnd, cid, t0)
                return
            client_states.append(state)
        agg_state, m_state, v_state = aggregate_fedadam(client_states, global_state_start, weights,
                                                          m_state, v_state, server_lr=server_lr,
                                                          beta1=FEDADAM_BETA1, beta2=FEDADAM_BETA2, tau=FEDADAM_TAU)
        global_model.load_state_dict(agg_state)
        log_round(writer, "fedadam", config, rnd, global_model, Xval, yval, data, t0)


def main():
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_fedadam_fallback_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds",
                     "diverged", "diverged_client_id", "diverged_note"])
        for client_lr, server_lr in CANDIDATES:
            run_fedadam_guarded(data, input_dim, client_ids, row_weights, Xval, yval, client_lr, server_lr, w)
            f.flush()

    df = pd.read_csv(out_path)
    df["diverged"] = df["diverged"].astype(bool)
    df["algorithm"] = df["algorithm"] + "_" + df["config"]  # so stability_report separates the two candidates
    stability_report(df)
    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
