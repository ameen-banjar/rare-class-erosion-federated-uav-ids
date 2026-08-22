"""
Item 2, phase A (per user instruction): smoke test + single-seed (seed=11),
45-round, VALIDATION-ONLY run for SCAFFOLD, FedNova, FedAdam -- verifies
implementation correctness and convergence behavior BEFORE freezing
hyperparameters and committing to the full 5-seed suite. held_out_test is
never touched here.
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from algorithms import local_train, aggregate_weighted, evaluate
from advanced_algorithms import (zeros_like_state, local_train_scaffold, aggregate_scaffold,
                                  local_train_fednova, aggregate_fednova, aggregate_fedadam)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 11
NUM_ROUNDS = 45
BATCH_SIZE = 4096
LR = 1e-3
SCAFFOLD_LR = 1e-2  # SCAFFOLD conventionally uses a larger local LR with plain SGD (no Adam momentum)
FEDADAM_SERVER_LR = 0.01
FEDADAM_BETA1, FEDADAM_BETA2, FEDADAM_TAU = 0.9, 0.99, 1e-3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
VAL_LABEL_INDICES = [i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def log_row(writer, algo, seed, rnd, m, comm_multiplier, elapsed):
    writer.writerow([algo, seed, rnd, m["macro_f1"], m["loss"],
                      m["per_class_predicted_count"]["Manipulation"],
                      m["per_class_predicted_count"]["Replay"],
                      m["per_class_recall"]["Manipulation"], m["per_class_recall"]["Replay"],
                      m["per_class_f1"]["Manipulation"], m["per_class_f1"]["Replay"],
                      comm_multiplier, elapsed])


def run_scaffold(data, input_dim, client_ids, weights, Xval, yval, writer):
    set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    c_global = zeros_like_state(global_model)
    c_local = {cid: zeros_like_state(global_model) for cid in client_ids}
    t0 = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states, client_delta_c = [], []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, delta_c, new_c_i, n_steps, n, loss = local_train_scaffold(
                local_model, Xc, yc, DEVICE, c_global, c_local[cid], global_state_start,
                epochs=1, batch_size=BATCH_SIZE, lr=SCAFFOLD_LR,
                class_weights=data.class_weights_full_balanced)
            client_states.append(state); client_delta_c.append(delta_c)
            c_local[cid] = new_c_i
        agg_state, c_global = aggregate_scaffold(client_states, client_delta_c, weights, c_global)
        global_model.load_state_dict(agg_state)
        m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                     class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
        log_row(writer, "scaffold", SEED, rnd, m, 2.0, time.time() - t0)  # 2x comm: model + control variate
        if rnd % 5 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
            print(f"[scaffold] round {rnd}/{NUM_ROUNDS}  macro_f1={m['macro_f1']:.4f}  "
                  f"Manip_pred={m['per_class_predicted_count']['Manipulation']}  "
                  f"Replay_pred={m['per_class_predicted_count']['Replay']}  elapsed={time.time()-t0:.0f}s")


def run_fednova(data, input_dim, client_ids, weights, Xval, yval, writer):
    set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states, taus = [], []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_fednova(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=LR,
                                                            class_weights=data.class_weights_full_balanced)
            client_states.append(state); taus.append(n_steps)
        agg_state = aggregate_fednova(client_states, global_state_start, weights, taus)
        global_model.load_state_dict(agg_state)
        m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                     class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
        log_row(writer, "fednova", SEED, rnd, m, 1.0, time.time() - t0)
        if rnd % 5 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
            print(f"[fednova] round {rnd}/{NUM_ROUNDS}  macro_f1={m['macro_f1']:.4f}  "
                  f"Manip_pred={m['per_class_predicted_count']['Manipulation']}  "
                  f"Replay_pred={m['per_class_predicted_count']['Replay']}  elapsed={time.time()-t0:.0f}s")


def run_fedadam(data, input_dim, client_ids, weights, Xval, yval, writer):
    set_seed(SEED)
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
            state, n, loss = local_train(local_model, Xc, yc, DEVICE, epochs=1, batch_size=BATCH_SIZE, lr=LR,
                                          class_weights=data.class_weights_full_balanced)
            client_states.append(state)
        agg_state, m_state, v_state = aggregate_fedadam(client_states, global_state_start, weights,
                                                          m_state, v_state, server_lr=FEDADAM_SERVER_LR,
                                                          beta1=FEDADAM_BETA1, beta2=FEDADAM_BETA2, tau=FEDADAM_TAU)
        global_model.load_state_dict(agg_state)
        m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                     class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
        log_row(writer, "fedadam", SEED, rnd, m, 1.0, time.time() - t0)
        if rnd % 5 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
            print(f"[fedadam] round {rnd}/{NUM_ROUNDS}  macro_f1={m['macro_f1']:.4f}  "
                  f"Manip_pred={m['per_class_predicted_count']['Manipulation']}  "
                  f"Replay_pred={m['per_class_predicted_count']['Replay']}  elapsed={time.time()-t0:.0f}s")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(weights); weights = [w / s for w in weights]
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_smoke_single_seed_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "manipulation_f1", "replay_f1",
                     "communication_multiplier_vs_fedavg", "elapsed_seconds"])
        run_scaffold(data, input_dim, client_ids, weights, Xval, yval, w)
        f.flush()
        run_fednova(data, input_dim, client_ids, weights, Xval, yval, w)
        f.flush()
        run_fedadam(data, input_dim, client_ids, weights, Xval, yval, w)
        f.flush()

    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
