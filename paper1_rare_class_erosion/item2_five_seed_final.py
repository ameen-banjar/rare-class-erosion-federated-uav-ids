"""
Item 2 -- FINAL five-seed run, frozen configs (DESIGN_FROZEN.md, 2026-08-18):
  fedavg_sgd:        lr=0.5,  optimizer=SGD,  comm_multiplier=1x
  fednova_sgd:       lr=0.3,  optimizer=SGD,  comm_multiplier=1x
  scaffold_uniform:  lr=0.1,  comm_multiplier=2x (model + control variate)
  scaffold_weighted: lr=0.1,  comm_multiplier=2x
  fedadam:           client_lr=0.001, server_lr=0.03 (beta1=0.9, beta2=0.99, tau=1e-3), comm_multiplier=1x

Seeds: 11, 22, 33, 44, 55. 45 rounds FIXED -- no early stopping, no best-
checkpoint substitution. Round-45 checkpoint saved for every (algorithm,
seed) pair = 25 checkpoints. VALIDATION ONLY during this run; held_out_test
is never read here (a single separate evaluation pass runs later, after
Validation analysis is fully locked, reusing these exact checkpoints without
retraining).

Divergence (NaN/Inf) checked after EVERY client's local training AND after
aggregation. A diverged (algorithm, seed) is logged with the failing round
and client (or "aggregation" if the collapse appears only post-aggregate)
and its round loop stops -- the seed is NOT swapped, NOT deleted, and
hyperparameters are NOT adjusted to rescue it. Divergence is a reportable
result.

Naming convention (per review): this run's FedAvg/FedNova/SCAFFOLD/FedAdam
use SGD (or SGD+server-Adam for FedAdam) locally; Phase 1's frozen "fedavg"/
"fedprox"/"client_uniform" baselines (results_phase1_frozen/) used Adam
locally. Tables must label Phase-1's as "FedAvg-Adam" etc. to avoid an
apples-to-oranges optimizer mismatch reading as an algorithm difference.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP, save_checkpoint
from algorithms import aggregate_weighted, evaluate
from advanced_algorithms import (zeros_like_state, local_train_scaffold, aggregate_scaffold,
                                  local_train_fednova, aggregate_fednova, local_train_generic,
                                  aggregate_fedadam)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints_five_seed"
SEEDS = [11, 22, 33, 44, 55]
NUM_ROUNDS = 45
BATCH_SIZE = 4096
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
VAL_LABEL_INDICES = [i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"]

FROZEN_CONFIGS = {
    "fedavg_sgd":        {"lr": 0.5, "comm_multiplier": 1.0},
    "fednova_sgd":       {"lr": 0.3, "comm_multiplier": 1.0},
    "scaffold_uniform":  {"lr": 0.1, "comm_multiplier": 2.0},
    "scaffold_weighted": {"lr": 0.1, "comm_multiplier": 2.0},
    "fedadam":           {"client_lr": 0.001, "server_lr": 0.03, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3,
                           "comm_multiplier": 1.0},
}


def set_seed(seed):
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)


def state_finite(state):
    return all(torch.isfinite(v).all() for v in state.values())


def code_version_hash():
    files = ["item2_five_seed_final.py", "../shared/fl_pipeline/advanced_algorithms.py",
              "../shared/fl_pipeline/algorithms.py", "../shared/fl_pipeline/data.py", "../shared/fl_pipeline/model.py"]
    h = hashlib.sha256()
    for f in files:
        p = Path(__file__).resolve().parent / f
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def log_row(writer, algo, config_str, seed, rnd, m, comm_mult, t0, diverged=False,
            diverged_stage="", diverged_client_id=""):
    if diverged:
        writer.writerow([algo, config_str, seed, rnd, None, None, "{}", "{}", "{}", "{}",
                          comm_mult, time.time() - t0, True, diverged_stage, diverged_client_id])
        return
    writer.writerow([algo, config_str, seed, rnd, m["macro_f1"], m["loss"],
                      json.dumps(m["per_class_precision"]), json.dumps(m["per_class_recall"]),
                      json.dumps(m["per_class_f1"]), json.dumps(m["per_class_predicted_count"]),
                      comm_mult, time.time() - t0, False, "", ""])


def eval_and_log(writer, algo, config_str, seed, rnd, global_model, Xval, yval, data, comm_mult, t0):
    m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                 class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
    log_row(writer, algo, config_str, seed, rnd, m, comm_mult, t0)
    if rnd % 5 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
        print(f"[{algo} seed={seed}] round {rnd}/{NUM_ROUNDS}  macro_f1={m['macro_f1']:.4f}  "
              f"Manip_rec={m['per_class_recall']['Manipulation']:.3f}  Replay_rec={m['per_class_recall']['Replay']:.3f}")
    return m


def run_fedavg_sgd(data, input_dim, client_ids, weights, Xval, yval, seed, cfg, writer):
    algo, config_str = "fedavg_sgd", f"lr={cfg['lr']}"
    set_seed(seed)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        client_states = []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=cfg["lr"], optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            if not state_finite(state):
                log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                        diverged=True, diverged_stage="local_train", diverged_client_id=cid)
                print(f"[{algo} seed={seed}] round {rnd} DIVERGED at client {cid} (local_train)")
                return None
            client_states.append(state)
        agg_state = aggregate_weighted(client_states, weights)
        if not state_finite(agg_state):
            log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                    diverged=True, diverged_stage="aggregation", diverged_client_id="")
            print(f"[{algo} seed={seed}] round {rnd} DIVERGED at aggregation")
            return None
        global_model.load_state_dict(agg_state)
        eval_and_log(writer, algo, config_str, seed, rnd, global_model, Xval, yval, data, cfg["comm_multiplier"], t0)
    return global_model


def run_fednova_sgd(data, input_dim, client_ids, weights, Xval, yval, seed, cfg, writer):
    algo, config_str = "fednova_sgd", f"lr={cfg['lr']}"
    set_seed(seed)
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
                                                            batch_size=BATCH_SIZE, lr=cfg["lr"], optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            if not state_finite(state):
                log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                        diverged=True, diverged_stage="local_train", diverged_client_id=cid)
                print(f"[{algo} seed={seed}] round {rnd} DIVERGED at client {cid} (local_train)")
                return None
            client_states.append(state); taus.append(n_steps)
        agg_state = aggregate_fednova(client_states, global_state_start, weights, taus)
        if not state_finite(agg_state):
            log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                    diverged=True, diverged_stage="aggregation", diverged_client_id="")
            print(f"[{algo} seed={seed}] round {rnd} DIVERGED at aggregation")
            return None
        global_model.load_state_dict(agg_state)
        eval_and_log(writer, algo, config_str, seed, rnd, global_model, Xval, yval, data, cfg["comm_multiplier"], t0)
    return global_model


def run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, seed, cfg, variant, writer):
    algo, config_str = f"scaffold_{variant}", f"lr={cfg['lr']}"
    weights = row_weights if variant == "weighted" else uniform_weights
    set_seed(seed)
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
                epochs=1, batch_size=BATCH_SIZE, lr=cfg["lr"], class_weights=data.class_weights_full_balanced)
            if not state_finite(state):
                log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                        diverged=True, diverged_stage="local_train", diverged_client_id=cid)
                print(f"[{algo} seed={seed}] round {rnd} DIVERGED at client {cid} (local_train)")
                return None
            client_states.append(state); client_delta_c.append(delta_c)
            c_local[cid] = new_c_i
        agg_state, c_global = aggregate_scaffold(client_states, client_delta_c, weights, c_global)
        if not state_finite(agg_state):
            log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                    diverged=True, diverged_stage="aggregation", diverged_client_id="")
            print(f"[{algo} seed={seed}] round {rnd} DIVERGED at aggregation")
            return None
        global_model.load_state_dict(agg_state)
        eval_and_log(writer, algo, config_str, seed, rnd, global_model, Xval, yval, data, cfg["comm_multiplier"], t0)
    return global_model


def run_fedadam(data, input_dim, client_ids, weights, Xval, yval, seed, cfg, writer):
    algo = "fedadam"
    config_str = f"client_lr={cfg['client_lr']},server_lr={cfg['server_lr']}"
    set_seed(seed)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    m_state = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in global_model.state_dict().items()}
    v_state = {k: torch.full_like(v, cfg["tau"] ** 2, dtype=torch.float32, device="cpu") for k, v in global_model.state_dict().items()}
    t0 = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states = []
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=cfg["client_lr"], optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            if not state_finite(state):
                log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                        diverged=True, diverged_stage="local_train", diverged_client_id=cid)
                print(f"[{algo} seed={seed}] round {rnd} DIVERGED at client {cid} (local_train)")
                return None
            client_states.append(state)
        agg_state, m_state, v_state = aggregate_fedadam(client_states, global_state_start, weights,
                                                          m_state, v_state, server_lr=cfg["server_lr"],
                                                          beta1=cfg["beta1"], beta2=cfg["beta2"], tau=cfg["tau"])
        if not state_finite(agg_state):
            log_row(writer, algo, config_str, seed, rnd, None, cfg["comm_multiplier"], t0,
                    diverged=True, diverged_stage="aggregation", diverged_client_id="")
            print(f"[{algo} seed={seed}] round {rnd} DIVERGED at aggregation")
            return None
        global_model.load_state_dict(agg_state)
        eval_and_log(writer, algo, config_str, seed, rnd, global_model, Xval, yval, data, cfg["comm_multiplier"], t0)
    return global_model


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    uniform_weights = [1.0 / len(client_ids)] * len(client_ids)
    Xval, yval = data.get_validation_data()

    metadata = {
        "seeds": SEEDS, "num_rounds": NUM_ROUNDS, "batch_size": BATCH_SIZE,
        "frozen_configs": FROZEN_CONFIGS, "code_version_sha256_16": code_version_hash(),
        "device": str(DEVICE), "run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Naming: fedavg_sgd/fednova_sgd/scaffold_*/fedadam here use SGD (or SGD+server-Adam) "
                "locally. Phase-1 frozen fedavg/fedprox/client_uniform (results_phase1_frozen/) used "
                "Adam locally -- label Phase-1's as FedAvg-Adam etc. in comparison tables.",
        "held_out_test_touched": False,
    }
    with open(RESULTS_DIR / "item2_five_seed_final_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    out_path = RESULTS_DIR / "item2_five_seed_final_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "per_class_precision_json", "per_class_recall_json", "per_class_f1_json",
                     "per_class_predicted_count_json", "comm_multiplier_vs_fedavg", "elapsed_seconds",
                     "diverged", "diverged_stage", "diverged_client_id"])

        t0 = time.time()
        for seed in SEEDS:
            print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")

            gm = run_fedavg_sgd(data, input_dim, client_ids, row_weights, Xval, yval, seed, FROZEN_CONFIGS["fedavg_sgd"], w)
            if gm is not None:
                save_checkpoint(gm, CKPT_DIR / f"fedavg_sgd_seed{seed}_round45.pt")

            gm = run_fednova_sgd(data, input_dim, client_ids, row_weights, Xval, yval, seed, FROZEN_CONFIGS["fednova_sgd"], w)
            if gm is not None:
                save_checkpoint(gm, CKPT_DIR / f"fednova_sgd_seed{seed}_round45.pt")

            gm = run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, seed,
                               FROZEN_CONFIGS["scaffold_uniform"], "uniform", w)
            if gm is not None:
                save_checkpoint(gm, CKPT_DIR / f"scaffold_uniform_seed{seed}_round45.pt")

            gm = run_scaffold(data, input_dim, client_ids, row_weights, uniform_weights, Xval, yval, seed,
                               FROZEN_CONFIGS["scaffold_weighted"], "weighted", w)
            if gm is not None:
                save_checkpoint(gm, CKPT_DIR / f"scaffold_weighted_seed{seed}_round45.pt")

            gm = run_fedadam(data, input_dim, client_ids, row_weights, Xval, yval, seed, FROZEN_CONFIGS["fedadam"], w)
            if gm is not None:
                save_checkpoint(gm, CKPT_DIR / f"fedadam_seed{seed}_round45.pt")

            f.flush()
            print(f"Seed {seed} done. Elapsed {time.time()-t0:.0f}s total.")

    print(f"\nSaved -> {out_path}\nCheckpoints -> {CKPT_DIR}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
