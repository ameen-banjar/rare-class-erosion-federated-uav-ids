"""
Paper 1, Item 1 (frozen design: DESIGN_FROZEN.md) -- classifier-head /
update-conflict analysis. FedAvg only, all 5 seeds, logged rounds
[1,5,10,20,30,45]. held_out_test never touched.

For each logged round and each rare-class holder client (Manipulation: 2,13;
Replay: 1,4,10), measures the full mechanistic chain: pre-local-train ->
post-local-train (pre-agg) -> post-aggregation performance and logit margin
on both that client's own rows and the fixed validation set, PLUS the classifier-head update vector
for that class from every one of the 15 clients (not just holders), so we
can compute cross-client cosine similarity, each client's FedAvg-weighted
contribution, and the aggregate cancellation ratio.
"""
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP, save_checkpoint
from algorithms import local_train, aggregate_weighted, evaluate

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
SEEDS = [11, 22, 33, 44, 55]
LOGGED_ROUNDS = [1, 5, 10, 20, 30, 45]
NUM_ROUNDS = 45
BATCH_SIZE = 4096
LR = 1e-3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

RARE_CLASS_HOLDERS = {"Manipulation": [2, 13], "Replay": [1, 4, 10]}
ALL_HOLDER_CLIENTS = sorted({c for v in RARE_CLASS_HOLDERS.values() for c in v})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def head_layer(model):
    return model.net[-1]  # final nn.Linear(64, 10)


def class_update_vector(local_state, global_state, class_idx):
    """65-dim: 64 hidden-unit weight-row delta + 1 bias delta, for one class."""
    dw = (local_state["net.6.weight"][class_idx] - global_state["net.6.weight"][class_idx]).cpu().numpy()
    db = (local_state["net.6.bias"][class_idx] - global_state["net.6.bias"][class_idx]).cpu().numpy()
    return np.concatenate([dw, [db]])


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@torch.no_grad()
def logit_margin(model, X, y, class_idx, device, batch_size=4096):
    """mean(logit[true_class] - max(logit[other_classes])) over rows where y==class_idx."""
    mask = (y == class_idx)
    if mask.sum() == 0:
        return float("nan")
    Xc = X[mask]
    margins = []
    for i in range(0, len(Xc), batch_size):
        xb = Xc[i:i + batch_size].to(device)
        logits = model(xb)
        true_logit = logits[:, class_idx]
        other = logits.clone()
        other[:, class_idx] = -1e9
        max_other = other.max(dim=1).values
        margins.append((true_logit - max_other).cpu())
    return float(torch.cat(margins).mean())


def find_layer_names(model):
    """net.6 is the final Linear if net = [Linear,ReLU,Dropout]*2 + Linear (indices 0..6)."""
    names = [n for n, _ in model.named_parameters()]
    assert "net.6.weight" in names and "net.6.bias" in names, f"unexpected layer names: {names}"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    find_layer_names(AttackFamilyMLP(input_dim))

    client_ids = data.client_ids()
    client_data = {cid: data.get_client_data(cid) for cid in client_ids}
    weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(weights); weights = [w / s for w in weights]
    class_idx_of = {c: ALL_CATEGORIES.index(c) for c in RARE_CLASS_HOLDERS}

    mech_path = RESULTS_DIR / "update_conflict_mechanistic.csv"
    pair_path = RESULTS_DIR / "update_conflict_pairwise_cosine.csv"
    pred_path = RESULTS_DIR / "update_conflict_prediction_counts.csv"
    contrib_path = RESULTS_DIR / "update_conflict_client_contributions.csv"

    with open(mech_path, "w", newline="") as mf, open(pair_path, "w", newline="") as pf, \
         open(pred_path, "w", newline="") as prf, open(contrib_path, "w", newline="") as cf:
        mw = csv.writer(mf)
        mw.writerow(["seed", "round", "class", "holder_client_id", "eval_data",
                     "n_eval_rows", "n_positive_rows",
                     "precision_pre_local_train", "recall_pre_local_train", "f1_pre_local_train",
                     "precision_post_local_pre_agg", "recall_post_local_pre_agg", "f1_post_local_pre_agg",
                     "precision_post_aggregation", "recall_post_aggregation", "f1_post_aggregation",
                     "logit_margin_pre_local_train", "logit_margin_post_local_pre_agg",
                     "logit_margin_post_aggregation", "local_margin_gain",
                     "aggregation_margin_change", "net_margin_change",
                     "update_norm_this_client", "cosine_holder_update_vs_aggregate_update",
                     "fedavg_weighted_contribution_norm", "cancellation_ratio",
                     "holder_weighted_sum_norm", "nonholder_weighted_sum_norm",
                     "cosine_holder_sum_vs_nonholder_sum"])
        pw = csv.writer(pf)
        pw.writerow(["seed", "round", "class", "client_a", "client_b", "role_a", "role_b", "cosine_similarity"])
        prw = csv.writer(prf)
        prw.writerow(["seed", "round", "predicted_class", "predicted_count"] +
                     [f"true_count_{c}" for c in ALL_CATEGORIES])
        cw = csv.writer(cf)
        cw.writerow(["seed", "round", "class", "client_id", "role", "fedavg_weight",
                     "update_norm", "weighted_contribution_norm", "cosine_update_vs_aggregate"])

        Xval, yval = data.get_validation_data()

        t0 = time.time()
        for seed in SEEDS:
            set_seed(seed)
            global_model = AttackFamilyMLP(input_dim).to(DEVICE)

            for rnd in range(1, NUM_ROUNDS + 1):
                log_this_round = rnd in LOGGED_ROUNDS
                global_state_start = {k: v.clone() for k, v in global_model.state_dict().items()}

                # Same-data causal comparison plus fixed validation generalization check.
                pre_local = {}
                if log_this_round:
                    for cls, holders in RARE_CLASS_HOLDERS.items():
                        ci = class_idx_of[cls]
                        for cid in holders:
                            Xc, yc = client_data[cid]
                            for eval_name, (Xe, ye) in {
                                "own_client_rows": (Xc, yc), "validation": (Xval, yval)
                            }.items():
                                m = evaluate(global_model, Xe, ye, DEVICE, ALL_CATEGORIES,
                                             class_weights=data.class_weights_full_balanced,
                                             eval_label_indices=[ci])
                                margin = logit_margin(global_model, Xe, ye, ci, DEVICE)
                                pre_local[(cls, cid, eval_name)] = (
                                    m["per_class_precision"][cls], m["per_class_recall"][cls],
                                    m["per_class_f1"][cls], margin)

                client_states = {}
                local_perf = {}
                for cid in client_ids:
                    Xc, yc = client_data[cid]
                    local_model = AttackFamilyMLP(input_dim).to(DEVICE)
                    local_model.load_state_dict(global_state_start)
                    state, n, loss = local_train(local_model, Xc, yc, DEVICE, epochs=1,
                                                  batch_size=BATCH_SIZE, lr=LR,
                                                  class_weights=data.class_weights_full_balanced)
                    client_states[cid] = state

                    if log_this_round and cid in ALL_HOLDER_CLIENTS:
                        for cls, holders in RARE_CLASS_HOLDERS.items():
                            if cid in holders:
                                ci = class_idx_of[cls]
                                for eval_name, (Xe, ye) in {
                                    "own_client_rows": (Xc, yc), "validation": (Xval, yval)
                                }.items():
                                    m = evaluate(local_model, Xe, ye, DEVICE, ALL_CATEGORIES,
                                                 class_weights=data.class_weights_full_balanced,
                                                 eval_label_indices=[ci])
                                    local_perf[(cls, cid, eval_name)] = (
                                        m["per_class_precision"][cls], m["per_class_recall"][cls],
                                        m["per_class_f1"][cls], logit_margin(local_model, Xe, ye, ci, DEVICE))

                ordered_states = [client_states[cid] for cid in client_ids]
                agg_state = aggregate_weighted(ordered_states, weights)
                global_model.load_state_dict(agg_state)

                if log_this_round:
                    for cls, holders in RARE_CLASS_HOLDERS.items():
                        ci = class_idx_of[cls]
                        # update vector for THIS class from every client (not just holders)
                        upd = {cid: class_update_vector(client_states[cid], global_state_start, ci)
                               for cid in client_ids}
                        weighted_upd = {cid: weights[j] * upd[cid] for j, cid in enumerate(client_ids)}
                        aggregate_update = sum(weighted_upd.values())
                        denominator = sum(np.linalg.norm(weighted_upd[cid]) for cid in client_ids)
                        cancellation_ratio = (1.0 - np.linalg.norm(aggregate_update) / denominator
                                              if denominator > 1e-12 else float("nan"))
                        holder_sum = sum(weighted_upd[cid] for cid in holders)
                        nonholder_sum = sum(weighted_upd[cid] for cid in client_ids if cid not in holders)

                        for j, cid in enumerate(client_ids):
                            cw.writerow([seed, rnd, cls, cid,
                                         "holder" if cid in holders else "non_holder", weights[j],
                                         float(np.linalg.norm(upd[cid])),
                                         float(np.linalg.norm(weighted_upd[cid])),
                                         cosine(upd[cid], aggregate_update)])

                        # pairwise cosine among ALL 15 clients' updates for this class
                        for cid_a, cid_b in itertools.combinations(client_ids, 2):
                            role_a = "holder" if cid_a in holders else "non_holder"
                            role_b = "holder" if cid_b in holders else "non_holder"
                            pw.writerow([seed, rnd, cls, cid_a, cid_b, role_a, role_b,
                                         cosine(upd[cid_a], upd[cid_b])])

                        for cid in holders:
                            Xc, yc = client_data[cid]
                            for eval_name, (Xe, ye) in {
                                "own_client_rows": (Xc, yc), "validation": (Xval, yval)
                            }.items():
                                m_global = evaluate(global_model, Xe, ye, DEVICE, ALL_CATEGORIES,
                                                    class_weights=data.class_weights_full_balanced,
                                                    eval_label_indices=[ci])
                                margin_global = logit_margin(global_model, Xe, ye, ci, DEVICE)
                                p_pre, rec_pre, f1_pre, margin_pre = pre_local[(cls, cid, eval_name)]
                                p_local, rec_local, f1_local, margin_local = local_perf[(cls, cid, eval_name)]
                                mw.writerow([
                                    seed, rnd, cls, cid, eval_name, len(ye), int((ye == ci).sum()),
                                    p_pre, rec_pre, f1_pre, p_local, rec_local, f1_local,
                                    m_global["per_class_precision"][cls],
                                    m_global["per_class_recall"][cls], m_global["per_class_f1"][cls],
                                    margin_pre, margin_local, margin_global,
                                    margin_local - margin_pre, margin_global - margin_local,
                                    margin_global - margin_pre, float(np.linalg.norm(upd[cid])),
                                    cosine(upd[cid], aggregate_update),
                                    float(np.linalg.norm(weighted_upd[cid])), cancellation_ratio,
                                    float(np.linalg.norm(holder_sum)), float(np.linalg.norm(nonholder_sum)),
                                    cosine(holder_sum, nonholder_sum),
                                ])

                    # prediction counts on validation, for the sanity check (not a threshold artifact)
                    m_val = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                                      class_weights=data.class_weights_full_balanced,
                                      eval_label_indices=[i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"])
                    true_counts = {c: int((yval.numpy() == ALL_CATEGORIES.index(c)).sum()) for c in ALL_CATEGORIES}
                    for c in ALL_CATEGORIES:
                        prw.writerow([seed, rnd, c, m_val["per_class_predicted_count"][c]] +
                                     [true_counts[cc] for cc in ALL_CATEGORIES])

                    ckpt_path = CKPT_DIR / f"fedavg_seed{seed}_round{rnd}.pt"
                    save_checkpoint(global_model, ckpt_path)
                    print(f"[seed={seed}] round {rnd}/{NUM_ROUNDS} logged  "
                          f"Manip_pred_count={m_val['per_class_predicted_count']['Manipulation']}  "
                          f"Replay_pred_count={m_val['per_class_predicted_count']['Replay']}  "
                          f"elapsed={time.time()-t0:.0f}s")

            mf.flush(); pf.flush(); prf.flush(); cf.flush()
            print(f"Seed {seed} done. Elapsed {time.time()-t0:.0f}s total.")

    print(f"\nSaved -> {mech_path}\nSaved -> {pair_path}\nSaved -> {pred_path}"
          f"\nSaved -> {contrib_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
