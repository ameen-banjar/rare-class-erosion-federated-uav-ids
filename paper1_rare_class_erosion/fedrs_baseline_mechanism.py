"""
Mechanism-targeted baseline test, added post-AEJ-rejection to strengthen the
RQ1 mechanistic claim before resubmission to JNCA.

Section 5 of the manuscript proposes a "plausible, non-exclusive mechanistic
candidate": with 12-13 of 15 clients holding zero examples of a rare class,
standard softmax cross-entropy's gradient for that class's absent logit is
positive for every one of those clients' rows, implicitly and collectively
suppressing it. This was stated as a plausible account, never tested as an
intervention.

FedRS (Li & Zhan, KDD 2021, "FedRS: Federated Learning with Restricted
Softmax for Label Distribution Non-IID Data") proposes restricted softmax
specifically for this failure mode. The paper's Eq. 8 states restriction
formally as a binary observed/missing split (alpha_c = 1 for c in O, = alpha
for c in M), and multiplies the raw logit for missing classes by alpha
before the softmax exponential. The OFFICIAL reference implementation
(github.com/lxcnju/FedRepo, algorithms/fedrs.py) computes something more
specific than that binary rule: a continuous, per-client, per-class
restriction r_c = alpha + (1-alpha) * n_c / max_j(n_j), from each client's
own local class row counts n_c. r_c = alpha exactly when n_c = 0 (matching
the paper's binary case), but a class present at low frequency gets an
INTERMEDIATE value close to alpha, not r_c = 1. This distinction is not
academic here: empirically, all 5 Manipulation/Replay holder clients in
this dataset hold their rare class at only 1.3-4.3% of their own largest
class's row count, giving r ~ 0.51-0.52 at alpha=0.5 -- i.e. holders are
almost as restricted on their held rare class as non-holders are, under the
official schedule. An earlier version of this script used the paper's
binary rule (r_c = 1.0 for any locally observed class, regardless of how
rare) and was corrected to the official continuous schedule after this was
checked directly against holder-client class counts.

alpha=1.0 reduces to standard softmax; alpha=0.5 is evaluated directly in
the original paper (Table 1, Fig. 5), selected here a priori and not tuned
on this dataset. This is a single frozen configuration, not a
hyperparameter search: a small, targeted mechanistic intervention test, not
a new item.

Everything else -- the fixed 15-client partition, the 5 seeds, the logged
rounds, the class-weight scheme, the evaluation protocol -- is copied
UNCHANGED from update_conflict_analysis.py (the frozen RQ1/Item-1 script) so
that FedRS and FedAvg results are directly comparable, round for round, seed
for seed. held_out_test is never touched.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP, get_state_dict_copy, save_checkpoint
from algorithms import make_loader, aggregate_weighted, evaluate

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints_fedrs"
SEEDS = [11, 22, 33, 44, 55]
LOGGED_ROUNDS = [1, 5, 10, 20, 30, 45]
NUM_ROUNDS = 45
BATCH_SIZE = 4096
LR = 1e-3
ALPHA = 0.5  # prespecified, frozen -- taken from FedRS paper's explored range, not tuned here
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

RARE_CLASS_HOLDERS = {"Manipulation": [2, 13], "Replay": [1, 4, 10]}
ALL_HOLDER_CLIENTS = sorted({c for v in RARE_CLASS_HOLDERS.values() for c in v})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def logit_margin(model, X, y, class_idx, device, batch_size=4096):
    mask = (y == class_idx)
    if mask.sum() == 0:
        return float("nan")
    Xc = X[mask]
    margins = []
    with torch.no_grad():
        for i in range(0, len(Xc), batch_size):
            xb = Xc[i:i + batch_size].to(device)
            logits = model(xb)
            true_logit = logits[:, class_idx]
            other = logits.clone()
            other[:, class_idx] = -1e9
            max_other = other.max(dim=1).values
            margins.append((true_logit - max_other).cpu())
    return float(torch.cat(margins).mean())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def class_update_vector(local_state, global_state, class_idx):
    dw = (local_state["net.6.weight"][class_idx] - global_state["net.6.weight"][class_idx]).cpu().numpy()
    db = (local_state["net.6.bias"][class_idx] - global_state["net.6.bias"][class_idx]).cpu().numpy()
    return np.concatenate([dw, [db]])


def local_train_fedrs(model, X, y, device, epochs, batch_size, lr, class_weights, restriction_scale):
    """FedRS restricted softmax, matching the OFFICIAL reference
    implementation (lxcnju/FedRepo, algorithms/fedrs.py) rather than only
    the paper's simplified binary Eq. 8. The official code computes a
    per-class, per-client CONTINUOUS restriction from local class frequency:
    r_c = alpha + (1 - alpha) * (n_c / max_j n_j), where n_c is this
    client's row count for class c. r_c = alpha exactly when n_c = 0
    (matches the paper's "missing class" case); r_c = 1 exactly when class c
    is this client's single largest class; classes present at low frequency
    (the common case for a rare-class holder, whose held class is nearly
    always a small minority of its own rows) get an intermediate value close
    to alpha, NOT r_c = 1 -- verified empirically for this dataset: all 5
    Manipulation/Replay holder clients have rare-class row shares of
    1.3-4.3% of their largest class, giving r ~ 0.51-0.52 at alpha=0.5, not
    the r=1.0 an earlier binary-restriction version of this script
    (observed-class -> 1.0, missing-class -> alpha) incorrectly assumed.
    restriction_scale is this precomputed per-client r vector, applied as
    elementwise multiplication of the model's output logits before
    cross-entropy."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = class_weights.to(device) if class_weights is not None else None
    scale = restriction_scale.to(device)
    loader = make_loader(X, y, batch_size)
    last_loss = None
    for _ in range(epochs):
        total_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb) * scale.unsqueeze(0)
            loss = nn.functional.cross_entropy(logits, yb, weight=w)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        last_loss = total_loss / max(n_batches, 1)
    return get_state_dict_copy(model), len(X), last_loss


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}  |  FedRS alpha={ALPHA} (prespecified, frozen)")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    n_classes = len(ALL_CATEGORIES)

    client_ids = data.client_ids()
    client_data = {cid: data.get_client_data(cid) for cid in client_ids}
    weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(weights); weights = [w / s for w in weights]
    class_idx_of = {c: ALL_CATEGORIES.index(c) for c in RARE_CLASS_HOLDERS}

    restriction_scale = {}
    for cid in client_ids:
        _, yc = client_data[cid]
        counts = torch.bincount(yc, minlength=n_classes).float()
        max_count = counts.max().clamp(min=1.0)
        restriction_scale[cid] = ALPHA + (1.0 - ALPHA) * (counts / max_count)
        n_absent = int((counts == 0).sum())
        r_manip = restriction_scale[cid][class_idx_of.get("Manipulation", 0)].item()
        r_replay = restriction_scale[cid][class_idx_of.get("Replay", 0)].item()
        print(f"  client {cid}: {n_absent}/{n_classes} classes absent; "
              f"r(Manipulation)={r_manip:.3f} r(Replay)={r_replay:.3f}")

    mech_path = RESULTS_DIR / "fedrs_mechanistic.csv"
    contrib_path = RESULTS_DIR / "fedrs_client_contributions.csv"

    with open(mech_path, "w", newline="") as mf, open(contrib_path, "w", newline="") as cf:
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
                    state, n, loss = local_train_fedrs(
                        local_model, Xc, yc, DEVICE, epochs=1, batch_size=BATCH_SIZE, lr=LR,
                        class_weights=data.class_weights_full_balanced,
                        restriction_scale=restriction_scale[cid])
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

                    ckpt_path = CKPT_DIR / f"fedrs_seed{seed}_round{rnd}.pt"
                    save_checkpoint(global_model, ckpt_path)
                    m_val = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                                      class_weights=data.class_weights_full_balanced,
                                      eval_label_indices=[i for i in range(n_classes)
                                                           if ALL_CATEGORIES[i] != "Password Cracking"])
                    print(f"[seed={seed}] round {rnd}/{NUM_ROUNDS} logged  "
                          f"Manip_recall={m_val['per_class_recall']['Manipulation']:.3f}  "
                          f"Replay_recall={m_val['per_class_recall']['Replay']:.3f}  "
                          f"elapsed={time.time()-t0:.0f}s")

            mf.flush(); cf.flush()
            print(f"Seed {seed} done. Elapsed {time.time()-t0:.0f}s total.")

    print(f"\nSaved -> {mech_path}\nSaved -> {contrib_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
