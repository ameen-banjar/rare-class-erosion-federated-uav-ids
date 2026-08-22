"""
Item 2 -- boundary extension 3 (targeted), corrected per review: only
FedAvg-SGD and FedNova-SGD still sat at the top of the tested LR range (0.3)
after extension 2. Tests lr in {0.5, 1.0, 2.0} -- 2.0 added specifically so
1.0 winning cannot itself become a new untested boundary requiring a fourth
run. Explicit pre-registered divergence detection (NaN/Inf in loss or model
parameters): a diverged config is LOGGED as such and its round loop stops
early -- never silently dropped, never masked with gradient clipping (which
would change the algorithm being tested). 15 rounds, seed=11,
validation-only, same init/client order/batch size as prior screening runs.
held_out_test never touched.
"""
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
import item2_optimizer_screening as scr
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from algorithms import aggregate_weighted, evaluate
from advanced_algorithms import local_train_fednova, aggregate_fednova, local_train_generic

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 11
SCREEN_ROUNDS = 15
EXTENSION_LRS = [0.5, 1.0, 2.0]
BATCH_SIZE = 4096
DEVICE = scr.DEVICE
VAL_LABEL_INDICES = scr.VAL_LABEL_INDICES


def is_diverged(loss_value, model):
    if loss_value is None or not torch.isfinite(torch.tensor(loss_value)):
        return True
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


def run_fedavg_sgd_guarded(data, input_dim, client_ids, weights, Xval, yval, lr, writer):
    scr.set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, SCREEN_ROUNDS + 1):
        client_states = []
        last_loss = None
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            client_states.append(state)
            last_loss = loss
        agg_state = aggregate_weighted(client_states, weights)
        global_model.load_state_dict(agg_state)

        diverged = is_diverged(last_loss, global_model)
        if diverged:
            writer.writerow(["fedavg_sgd", f"lr={lr}", SEED, rnd, None, last_loss, None, None, None, None,
                              time.time() - t0, True])
            print(f"[fedavg_sgd lr={lr}] round {rnd}/{SCREEN_ROUNDS}  DIVERGED (loss={last_loss})  -- stopping this config")
            return
        m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                     class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
        writer.writerow(["fedavg_sgd", f"lr={lr}", SEED, rnd, m["macro_f1"], m["loss"],
                          m["per_class_predicted_count"]["Manipulation"], m["per_class_predicted_count"]["Replay"],
                          m["per_class_recall"]["Manipulation"], m["per_class_recall"]["Replay"],
                          time.time() - t0, False])
        if rnd % 5 == 0 or rnd == 1 or rnd == SCREEN_ROUNDS:
            print(f"[fedavg_sgd lr={lr}] round {rnd}/{SCREEN_ROUNDS}  macro_f1={m['macro_f1']:.4f}")


def run_fednova_sgd_guarded(data, input_dim, client_ids, weights, Xval, yval, lr, writer):
    scr.set_seed(SEED)
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    for rnd in range(1, SCREEN_ROUNDS + 1):
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
        client_states, taus = [], []
        last_loss = None
        for cid in client_ids:
            Xc, yc = data.get_client_data(cid)
            local_model = AttackFamilyMLP(input_dim).to(DEVICE)
            local_model.load_state_dict(global_model.state_dict())
            state, n_steps, n, loss = local_train_fednova(local_model, Xc, yc, DEVICE, epochs=1,
                                                            batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                            class_weights=data.class_weights_full_balanced)
            client_states.append(state); taus.append(n_steps)
            last_loss = loss
        agg_state = aggregate_fednova(client_states, global_state_start, weights, taus)
        global_model.load_state_dict(agg_state)

        diverged = is_diverged(last_loss, global_model)
        if diverged:
            writer.writerow(["fednova_sgd", f"lr={lr}", SEED, rnd, None, last_loss, None, None, None, None,
                              time.time() - t0, True])
            print(f"[fednova_sgd lr={lr}] round {rnd}/{SCREEN_ROUNDS}  DIVERGED (loss={last_loss})  -- stopping this config")
            return
        m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                     class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
        writer.writerow(["fednova_sgd", f"lr={lr}", SEED, rnd, m["macro_f1"], m["loss"],
                          m["per_class_predicted_count"]["Manipulation"], m["per_class_predicted_count"]["Replay"],
                          m["per_class_recall"]["Manipulation"], m["per_class_recall"]["Replay"],
                          time.time() - t0, False])
        if rnd % 5 == 0 or rnd == 1 or rnd == SCREEN_ROUNDS:
            print(f"[fednova_sgd lr={lr}] round {rnd}/{SCREEN_ROUNDS}  macro_f1={m['macro_f1']:.4f}")


def main():
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    row_weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(row_weights); row_weights = [w / s for w in row_weights]
    Xval, yval = data.get_validation_data()

    out_path = RESULTS_DIR / "item2_boundary_extension3_round_metrics.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "config", "seed", "round", "val_macro_f1", "val_loss",
                     "manipulation_predicted_count", "replay_predicted_count",
                     "manipulation_recall", "replay_recall", "elapsed_seconds", "diverged"])

        for lr in EXTENSION_LRS:
            run_fedavg_sgd_guarded(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()
        for lr in EXTENSION_LRS:
            run_fednova_sgd_guarded(data, input_dim, client_ids, row_weights, Xval, yval, lr, w); f.flush()

    print(f"\nSaved -> {out_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
