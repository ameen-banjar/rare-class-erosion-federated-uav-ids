"""
Item 2 -- CORRECTED final 45-round confirmation, seed=11, validation-only,
for the true winning configs after full boundary search:
  - fedavg_sgd:        lr=0.5
  - fednova_sgd:       lr=0.3
  - scaffold_uniform:  lr=0.2
  - scaffold_weighted: lr=0.2
(FedAdam's winner, client_lr=0.003/server_lr=0.03, was already confirmed at
45 rounds in item2_confirm_convergence.py and is reused as-is -- not rerun.)

Per-CLIENT divergence check (every one of the 15 clients' local output, not
just the last one processed -- extension3's smoke test showed a mid-loop
client can diverge while the last-processed client looks fine). A diverged
config is logged with round + client_id and the round loop stops; no
gradient clipping, no silent drop. Round 45's result is always the reported
one -- never substituted with a better earlier checkpoint.

Pre-registered stability criteria (computed after the run, decision is NOT
made mid-run):
  1. no numerical divergence
  2. |best 5-round moving average - mean(rounds 41-45)| <= 0.03 macro-F1
  3. slope of rounds 36-45 >= -0.002 per round

The item2_confirm_convergence.py run (lr=0.03 for all four) is NOT deleted
or overwritten -- kept as a valid sensitivity/comparison artifact, per
instruction, just no longer the final-config confirmation.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import item2_optimizer_screening as scr
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from algorithms import aggregate_weighted, evaluate
from advanced_algorithms import (zeros_like_state, local_train_scaffold, aggregate_scaffold,
                                  local_train_fednova, aggregate_fednova, local_train_generic)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 11
NUM_ROUNDS = 45
BATCH_SIZE = 4096
DEVICE = scr.DEVICE
VAL_LABEL_INDICES = scr.VAL_LABEL_INDICES

WINNERS = [
    ("fedavg_sgd", {"lr": 0.5}),
    ("fednova_sgd", {"lr": 0.3}),
    ("scaffold_uniform", {"lr": 0.2}),
    ("scaffold_weighted", {"lr": 0.2}),
]


def client_diverged(state):
    return any(not torch.isfinite(v).all() for v in state.values())


def log_round(writer, algo, config, rnd, global_model, Xval, yval, data, t0):
    m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                 class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
    writer.writerow([algo, config, SEED, rnd, m["macro_f1"], m["loss"],
                      m["per_class_predicted_count"]["Manipulation"], m["per_class_predicted_count"]["Replay"],
                      m["per_class_recall"]["Manipulation"], m["per_class_recall"]["Replay"],
                      time.time() - t0, False, "", ""])
    if rnd % 5 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
        print(f"[{algo} {config}] round {rnd}/{NUM_ROUNDS}  macro_f1={m['macro_f1']:.4f}  "
              f"Manip_rec={m['per_class_recall']['Manipulation']:.3f}  Replay_rec={m['per_class_recall']['Replay']:.3f}")


def log_diverged(writer, algo, config, rnd, client_id, t0):
    writer.writerow([algo, config, SEED, rnd, None, None, None, None, None, None,
                      time.time() - t0, True, client_id, "per-client NaN/Inf in local_train output"])
    print(f"[{algo} {config}] round {rnd}/{NUM_ROUNDS}  DIVERGED at client {client_id} -- stopping this config")


def run_fedavg_sgd(data, input_dim, client_ids, weights, Xval, yval, lr, writer):
    config = f"lr={lr}"
    scr.set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        client_states = []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            if client_diverged(state):
                log_diverged(writer, "fedavg_sgd", config, rnd, cid, t0)
                return
            client_states.append(state)
        agg_state = aggregate_weighted(client_states, weights)
        global_model.load_state_dict(agg_state)
        log_round(writer, "fedavg_sgd", config, rnd, global_model, Xval, yval, data, t0)


def run_fednova_sgd(data, input_dim, client_ids, weights, Xval, yval, lr, writer):
    config = f"lr={lr}"
    scr.set_seed(SEED)
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
                                                            batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            if client_diverged(state):
                log_diverged(writer, "fednova_sgd", config, rnd, cid, t0)
                return
            client_states.append(state); taus.append(n_steps)
        agg_state = aggregate_fednova(client_states, global_state_start, weights, taus)
        global_model.load_state_dict(agg_state)
        log_round(writer, "fednova_sgd", config, rnd, global_model, Xval, yval, data, t0)


def run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, lr, variant, writer):
    config = f"lr={lr}"
    algo = f"scaffold_{variant}"
    weights = row_weights if variant == "weighted" else uniform_weights
    scr.set_seed(SEED)
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
                epochs=1, batch_size=BATCH_SIZE, lr=lr, class_weights=data.class_weights_full_balanced)
            if client_diverged(state):
                log_diverged(writer, algo, config, rnd, cid, t0)
                return
            client_states.append(state); client_delta_c.append(delta_c)
            c_local[cid] = new_c_i
        agg_state, c_global = aggregate_scaffold(client_states, client_delta_c, weights, c_global)
        global_model.load_state_dict(agg_state)
        log_round(writer, algo, config, rnd, global_model, Xval, yval, data, t0)


def stability_report(df):
    print("\n=== Pre-registered stability check ===")
    results = {}
    for algo in df.algorithm.unique():
        sub = df[(df.algorithm == algo) & (~df.diverged)].sort_values("round")
        diverged = df[(df.algorithm == algo) & (df.diverged)]
        if len(diverged) > 0:
            print(f"{algo}: FAILED -- diverged at round {diverged['round'].iloc[0]}, "
                  f"client {diverged['diverged_client_id'].iloc[0]}")
            results[algo] = "FAILED_DIVERGED"
            continue
        f1 = sub.val_macro_f1.values
        rounds = sub["round"].values
        ma5 = pd.Series(f1).rolling(5).mean()
        best_ma5 = ma5.max()
        mean_41_45 = sub[sub["round"].between(41, 45)].val_macro_f1.mean()
        gap = best_ma5 - mean_41_45
        last10 = sub[sub["round"].between(36, 45)]
        slope = np.polyfit(last10["round"], last10.val_macro_f1, 1)[0]
        pass_gap = gap <= 0.03
        pass_slope = slope >= -0.002
        verdict = "PASS" if (pass_gap and pass_slope) else "FAIL"
        print(f"{algo}: best_MA5={best_ma5:.4f}  mean(41-45)={mean_41_45:.4f}  gap={gap:.4f} "
              f"({'OK' if pass_gap else 'FAIL'})  slope(36-45)={slope:+.5f} ({'OK' if pass_slope else 'FAIL'})  "
              f"-> {verdict}")
        results[algo] = verdict
    return results


def main():
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    uniform_weights = [1.0 / len(client_ids)] * len(client_ids)
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_final_confirmation_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds",
                     "diverged", "diverged_client_id", "diverged_note"])

        run_fedavg_sgd(data, input_dim, client_ids, row_weights, Xval, yval, 0.5, w); f.flush()
        run_fednova_sgd(data, input_dim, client_ids, row_weights, Xval, yval, 0.3, w); f.flush()
        run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, 0.2, "uniform", w); f.flush()
        run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, 0.2, "weighted", w); f.flush()

    df = pd.read_csv(out_path)
    df["diverged"] = df["diverged"].astype(bool)
    stability_report(df)
    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
