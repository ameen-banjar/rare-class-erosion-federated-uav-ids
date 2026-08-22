"""
Item 3 -- confirmation phase. 3 anchor partitions (fixed, selected from
exploratory diagnostics only): (alpha=0.1,n_clients=30,partition_seed=102),
(0.3,15,103), (1.0,10,101).

30 runs total:
  - 24 NEW: 3 partitions x 2 algorithms x 4 NEW model_seeds (22,33,44,55).
  - 6 REPRODUCIBILITY reruns: 3 partitions x 2 algorithms x model_seed=11
    (the exploratory seed). Their round-45 val_macro_f1 MUST match the
    exploratory grid's value for the same (algorithm, alpha, n_clients,
    partition_seed) exactly. Purpose: attach the local-to-global mechanistic
    diagnostic the exploratory grid omitted (documented deviation,
    DESIGN_FROZEN.md). Any mismatch is flagged, not silently accepted.

Round-45-only mechanistic diagnostic for every (class, client) holder pair,
on both the client's own rows and the fixed validation set: global recall/
precision/F1 pre-local-train, local post-local (pre-agg), global post-agg,
logit margin at all three stages, predicted-count, client weight, n_holders,
holder weight share.

PERFORMANCE NOTE (fixed after a slow first attempt): naive per-(class,client)
evaluation re-ran a full validation forward pass (245k rows) per class per
client per stage -- ~264 validation passes for one 10-client partition,
impractically slow. Rewritten to compute per-class metrics/margins for ALL
10 classes in ONE forward pass per (model, eval-set), cached and reused
across every class that shares that client -- ~12 validation passes for a
10-client partition instead of ~132.

Validation only. held_out_test is never touched.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP, save_checkpoint
from algorithms import aggregate_weighted, evaluate, make_loader
from advanced_algorithms import local_train_fednova, aggregate_fednova, local_train_generic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_partition import build_partition

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints_confirmation"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
BATCH_SIZE = 4096
NUM_ROUNDS = 45
ALL_LABELS = list(range(10))
ANCHOR_PARTITIONS = [(0.1, 30, 102), (0.3, 15, 103), (1.0, 10, 101)]
MODEL_SEEDS_NEW = [22, 33, 44, 55]
MODEL_SEED_REPRO = 11
ALGO_CONFIGS = {"fedavg_sgd": {"lr": 0.5}, "fednova_sgd": {"lr": 0.3}}
VAL_LABEL_INDICES = [i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"]


def set_model_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def state_finite(state):
    return all(torch.isfinite(v).all() for v in state.values())


@torch.no_grad()
def all_class_predictions_and_margins(model, X, y, device, batch_size=4096):
    """ONE forward pass -> predictions (for precision/recall/F1) AND per-class
    mean logit margin for every class present in y. Reused across every
    (class, client) pair that shares this (model, eval-set)."""
    model.eval()
    preds, margins_by_class_sum, margins_by_class_n = [], {}, {}
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        logits = model(xb)
        p = logits.argmax(dim=1)
        preds.append(p.cpu())
        for ci in yb.unique().tolist():
            mask = yb == ci
            if mask.sum() == 0:
                continue
            true_logit = logits[mask, ci]
            other = logits[mask].clone()
            other[:, ci] = -1e9
            max_other = other.max(dim=1).values
            m = (true_logit - max_other).sum().item()
            margins_by_class_sum[ci] = margins_by_class_sum.get(ci, 0.0) + m
            margins_by_class_n[ci] = margins_by_class_n.get(ci, 0) + int(mask.sum())
    preds = torch.cat(preds).numpy()
    y_np = y.numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y_np, preds, labels=ALL_LABELS,
                                                                 average=None, zero_division=0)
    pred_counts = np.bincount(preds, minlength=10)
    margins = {ci: margins_by_class_sum[ci] / margins_by_class_n[ci] for ci in margins_by_class_sum}
    return {
        "precision": precision, "recall": recall, "f1": f1, "pred_counts": pred_counts,
        "margins": margins,  # dict: class_idx -> mean margin, only for classes present in y
    }


def get(cache, class_idx):
    p = cache["precision"][class_idx]
    r = cache["recall"][class_idx]
    f1 = cache["f1"][class_idx]
    margin = cache["margins"].get(class_idx, float("nan"))
    return p, r, f1, margin


def client_data_from_partition(data, client_sessions):
    out = {}
    for cid, sessions in client_sessions.items():
        if not sessions:
            continue
        df = data._load_sessions(sessions)
        out[cid] = data.tensors_from_df(df)
    return out


def holder_pairs(diag):
    pairs = []
    for cid_str, cdiag in diag["client_diagnostics"].items():
        cid = int(cid_str) if isinstance(cid_str, str) else cid_str
        for cls, n in cdiag["row_counts_by_category"].items():
            if n > 0:
                pairs.append((cls, cid))
    return pairs


def log_diverged(round_writer, algo, meta, rnd, where):
    round_writer.writerow([algo, meta["alpha"], meta["n_clients"], meta["partition_seed"], meta["model_seed"],
                            rnd, None, None, True, str(where)])
    print(f"    DIVERGED at round {rnd}, {where}")


def run_one(data, input_dim, algo, lr, client_data, weights, meta, diag, Xval, yval, round_writer, mech_writer):
    cids = list(client_data.keys())
    w = [weights[c] for c in cids]
    set_model_seed(meta["model_seed"])
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    pairs = holder_pairs(diag)
    holder_clients = sorted({cid for _, cid in pairs})
    total_rows_all = sum(v["n_rows"] for v in diag["client_diagnostics"].values())

    for rnd in range(1, NUM_ROUNDS + 1):
        record_mechanistic = (rnd == NUM_ROUNDS)

        pre_cache_val = pre_cache_own = None
        if record_mechanistic:
            pre_cache_val = all_class_predictions_and_margins(global_model, Xval, yval, DEVICE)
            pre_cache_own = {cid: all_class_predictions_and_margins(global_model, *client_data[cid], DEVICE)
                              for cid in holder_clients}

        client_states = []
        local_models = {}
        if algo == "fedavg_sgd":
            for cid in cids:
                Xc, yc = client_data[cid]
                local_model = AttackFamilyMLP(input_dim).to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                                batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                                class_weights=data.class_weights_full_balanced)
                if not state_finite(state):
                    log_diverged(round_writer, algo, meta, rnd, cid); return None
                client_states.append(state)
                if record_mechanistic and cid in holder_clients:
                    local_model.load_state_dict(state)
                    local_models[cid] = local_model
            agg_state = aggregate_weighted(client_states, w)
        else:
            global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
            taus = []
            for cid in cids:
                Xc, yc = client_data[cid]
                local_model = AttackFamilyMLP(input_dim).to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                state, n_steps, n, loss = local_train_fednova(local_model, Xc, yc, DEVICE, epochs=1,
                                                                batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                                class_weights=data.class_weights_full_balanced)
                if not state_finite(state):
                    log_diverged(round_writer, algo, meta, rnd, cid); return None
                client_states.append(state); taus.append(n_steps)
                if record_mechanistic and cid in holder_clients:
                    local_model.load_state_dict(state)
                    local_models[cid] = local_model
            agg_state = aggregate_fednova(client_states, global_state_start, w, taus)

        if not state_finite(agg_state):
            log_diverged(round_writer, algo, meta, rnd, "aggregation"); return None
        global_model.load_state_dict(agg_state)

        m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                     class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
        round_writer.writerow([algo, meta["alpha"], meta["n_clients"], meta["partition_seed"], meta["model_seed"],
                                rnd, m["macro_f1"], m["loss"], False, ""])

        if record_mechanistic:
            post_cache_val = all_class_predictions_and_margins(global_model, Xval, yval, DEVICE)
            post_cache_own = {cid: all_class_predictions_and_margins(global_model, *client_data[cid], DEVICE)
                               for cid in holder_clients}
            local_cache_val = {cid: all_class_predictions_and_margins(local_models[cid], Xval, yval, DEVICE)
                                for cid in holder_clients}
            local_cache_own = {cid: all_class_predictions_and_margins(local_models[cid], *client_data[cid], DEVICE)
                                for cid in holder_clients}

            for cls, cid in pairs:
                ci = ALL_CATEGORIES.index(cls)
                n_holders = diag["n_holders_per_class"][cls]
                holder_weight_share = sum(v["n_rows"] for v in diag["client_diagnostics"].values()
                                          if v["row_counts_by_category"].get(cls, 0) > 0) / total_rows_all
                for eval_name, (pre_c, loc_c, post_c) in {
                    "own_client_rows": (pre_cache_own[cid], local_cache_own[cid], post_cache_own[cid]),
                    "validation": (pre_cache_val, local_cache_val[cid], post_cache_val),
                }.items():
                    p_pre, r_pre, f1_pre, margin_pre = get(pre_c, ci)
                    p_loc, r_loc, f1_loc, margin_loc = get(loc_c, ci)
                    p_post, r_post, f1_post, margin_post = get(post_c, ci)
                    mech_writer.writerow([
                        algo, meta["alpha"], meta["n_clients"], meta["partition_seed"], meta["model_seed"],
                        rnd, cls, cid, eval_name,
                        p_pre, r_pre, f1_pre, p_loc, r_loc, f1_loc, p_post, r_post, f1_post,
                        margin_pre, margin_loc, margin_post,
                        w[cids.index(cid)], n_holders, holder_weight_share,
                        int(post_cache_val["pred_counts"][ci]),
                    ])

        if rnd % 15 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
            print(f"    round {rnd}/{NUM_ROUNDS}  macro_f1={m['macro_f1']:.4f}  elapsed={time.time()-t0:.0f}s")

    return global_model


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    Xval, yval = data.get_validation_data()

    round_path = RESULTS_DIR / "item3_confirmation_round_metrics.csv"
    mech_path = RESULTS_DIR / "item3_confirmation_mechanistic_round45.csv"

    with open(round_path, "w", newline="") as rf, open(mech_path, "w", newline="") as mf:
        rw = csv.writer(rf)
        rw.writerow(["algorithm", "alpha", "n_clients", "partition_seed", "model_seed", "round",
                     "val_macro_f1", "val_loss", "diverged", "diverged_at"])
        mw = csv.writer(mf)
        mw.writerow(["algorithm", "alpha", "n_clients", "partition_seed", "model_seed", "round",
                     "class", "client_id", "eval_data",
                     "precision_pre_local", "recall_pre_local", "f1_pre_local",
                     "precision_post_local", "recall_post_local", "f1_post_local",
                     "precision_post_agg", "recall_post_agg", "f1_post_agg",
                     "logit_margin_pre_local", "logit_margin_post_local", "logit_margin_post_agg",
                     "client_fedavg_weight", "n_holders_this_class", "holder_weight_share_this_class",
                     "global_predicted_count_this_class"])

        t_all0 = time.time()
        run_idx = 0
        total_runs = len(ANCHOR_PARTITIONS) * 2 * (1 + len(MODEL_SEEDS_NEW))
        for (alpha, n_clients, partition_seed) in ANCHOR_PARTITIONS:
            client_sessions, diag = build_partition(alpha, n_clients, partition_seed)
            client_data = client_data_from_partition(data, client_sessions)
            total = sum(len(y) for _, y in client_data.values())
            weights = {cid: len(y) / total for cid, (X, y) in client_data.items()}

            for algo, cfg in ALGO_CONFIGS.items():
                for model_seed in [MODEL_SEED_REPRO] + MODEL_SEEDS_NEW:
                    run_idx += 1
                    kind = "REPRODUCIBILITY (seed=11)" if model_seed == MODEL_SEED_REPRO else "NEW"
                    print(f"\n--- [{run_idx}/{total_runs}] {kind}  {algo}  alpha={alpha} n_clients={n_clients} "
                          f"partition_seed={partition_seed} model_seed={model_seed} ---  "
                          f"total_elapsed={time.time()-t_all0:.0f}s")
                    meta = {"alpha": alpha, "n_clients": n_clients, "partition_seed": partition_seed,
                            "model_seed": model_seed}
                    gm = run_one(data, input_dim, algo, cfg["lr"], client_data, weights, meta, diag,
                                 Xval, yval, rw, mw)
                    rf.flush(); mf.flush()
                    if gm is not None:
                        save_checkpoint(gm, CKPT_DIR / f"{algo}_a{alpha}_nc{n_clients}_ps{partition_seed}_ms{model_seed}_round45.pt")

    print(f"\nSaved -> {round_path}\nSaved -> {mech_path}\n(held_out_test never touched)")


if __name__ == "__main__":
    main()
