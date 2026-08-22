"""
Item 2 -- optimizer-matched screening (per review: the exploratory single-seed
run mixed SGD/lr=0.01 for SCAFFOLD against Adam/lr=0.001 for FedNova/FedAdam/
FedAvg, so Replay's recovery under SCAFFOLD/FedNova could not yet be
attributed to their heterogeneity-correction mechanism rather than the
optimizer/LR change itself).

Phase 1 (this script): 15-round, seed=11, VALIDATION-ONLY screening grid:
  - fedavg_sgd            x local_lr in {0.003, 0.01, 0.03}
  - scaffold_weighted     x local_lr in {0.003, 0.01, 0.03}   (row-weighted aggregation, our convention)
  - scaffold_uniform      x local_lr in {0.003, 0.01, 0.03}   (equal-client averaging, the standard SCAFFOLD baseline)
  - fednova_sgd           x local_lr in {0.003, 0.01, 0.03}
  - fedadam               x local_lr in {0.003, 0.01, 0.03} x server_lr in {0.003, 0.01, 0.03}
    (beta1=0.9, beta2=0.99, tau=1e-3 fixed, per Reddi et al. 2020 defaults)
All local training uses SGD (the optimizer-matched control) except FedAdam's
server-side step, which stays FedAdam's own optimizer by definition.

Selection: for each algorithm family, the config with the highest mean
val_macro_f1 averaged over the LAST 3 logged rounds (11,13,15) -- not
Replay-specific -- wins. held_out_test never touched.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from algorithms import aggregate_weighted, evaluate
from advanced_algorithms import (zeros_like_state, local_train_scaffold, aggregate_scaffold,
                                  local_train_fednova, aggregate_fednova, aggregate_fedadam,
                                  local_train_generic)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 11
SCREEN_ROUNDS = 15
LOCAL_LRS = [0.003, 0.01, 0.03]
SERVER_LRS = [0.003, 0.01, 0.03]
FEDADAM_BETA1, FEDADAM_BETA2, FEDADAM_TAU = 0.9, 0.99, 1e-3
BATCH_SIZE = 4096
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
VAL_LABEL_INDICES = [i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_fedavg_sgd(data, input_dim, client_ids, weights, Xval, yval, lr, writer):
    set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, SCREEN_ROUNDS + 1):
        client_states = []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            client_states.append(state)
        agg_state = aggregate_weighted(client_states, weights)
        global_model.load_state_dict(agg_state)
        log_round(writer, "fedavg_sgd", f"lr={lr}", rnd, global_model, Xval, yval, data, t0)


def run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, variant, writer):
    weights = row_weights if variant == "weighted" else uniform_weights
    set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    c_global = zeros_like_state(global_model)
    c_local = {cid: zeros_like_state(global_model) for cid in client_ids}
    t0 = time.time()
    for rnd in range(1, SCREEN_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states, client_delta_c = [], []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, delta_c, new_c_i, n_steps, n, loss = local_train_scaffold(
                local_model, Xc, yc, DEVICE, c_global, c_local[cid], global_state_start,
                epochs=1, batch_size=BATCH_SIZE, lr=lr, class_weights=data.class_weights_full_balanced)
            client_states.append(state); client_delta_c.append(delta_c)
            c_local[cid] = new_c_i
        agg_state, c_global = aggregate_scaffold(client_states, client_delta_c, weights, c_global)
        global_model.load_state_dict(agg_state)
        log_round(writer, f"scaffold_{variant}", f"lr={lr}", rnd, global_model, Xval, yval, data, t0)


def run_fednova_sgd(data, input_dim, client_ids, weights, Xval, yval, lr, writer):
    set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, SCREEN_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states, taus = [], []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_fednova(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            client_states.append(state); taus.append(n_steps)
        agg_state = aggregate_fednova(client_states, global_state_start, weights, taus)
        global_model.load_state_dict(agg_state)
        log_round(writer, "fednova_sgd", f"lr={lr}", rnd, global_model, Xval, yval, data, t0)


def run_fedadam(data, input_dim, client_ids, weights, Xval, yval, client_lr, server_lr, writer):
    set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    m_state = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in global_model.state_dict().items()}
    v_state = {k: torch.full_like(v, FEDADAM_TAU ** 2, dtype=torch.float32, device="cpu") for k, v in global_model.state_dict().items()}
    t0 = time.time()
    for rnd in range(1, SCREEN_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states = []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=client_lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            client_states.append(state)
        agg_state, m_state, v_state = aggregate_fedadam(client_states, global_state_start, weights,
                                                          m_state, v_state, server_lr=server_lr,
                                                          beta1=FEDADAM_BETA1, beta2=FEDADAM_BETA2, tau=FEDADAM_TAU)
        global_model.load_state_dict(agg_state)
        log_round(writer, "fedadam", f"client_lr={client_lr},server_lr={server_lr}", rnd, global_model, Xval, yval, data, t0)


def log_round(writer, algo, config, rnd, global_model, Xval, yval, data, t0):
    m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                 class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
    writer.writerow([algo, config, SEED, rnd, m["macro_f1"], m["loss"],
                      m["per_class_predicted_count"]["Manipulation"], m["per_class_predicted_count"]["Replay"],
                      m["per_class_recall"]["Manipulation"], m["per_class_recall"]["Replay"],
                      time.time() - t0])
    if rnd % 5 == 0 or rnd == 1 or rnd == SCREEN_ROUNDS:
        print(f"[{algo} {config}] round {rnd}/{SCREEN_ROUNDS}  macro_f1={m['macro_f1']:.4f}  "
              f"Manip_rec={m['per_class_recall']['Manipulation']:.3f}  Replay_rec={m['per_class_recall']['Replay']:.3f}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    uniform_weights = [1.0 / len(client_ids)] * len(client_ids)
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_optimizer_screening_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds"])

        for lr in LOCAL_LRS:
            run_fedavg_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()
        for lr in LOCAL_LRS:
            run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, "weighted", w); f.flush()
        for lr in LOCAL_LRS:
            run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, "uniform", w); f.flush()
        for lr in LOCAL_LRS:
            run_fednova_sgd(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()
        for client_lr in LOCAL_LRS:
            for server_lr in SERVER_LRS:
                run_fedadam(data, input_dim, client_ids, row_weights, Xval, yval, client_lr, server_lr, w); f.flush()

    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
