"""
Item 4 (frozen design: DESIGN_FROZEN.md) -- mechanistic localization,
head vs representation. Reuses Item 1's exact FedAvg trajectory
(round 1->45, single manual_seed per seed, same 15-client manifest,
same 5 seeds, same logged rounds [1,5,10,20,30,45]) via a full
deterministic replay -- NOT by reloading Item 1's saved checkpoints,
which are post-aggregation snapshots and cannot serve as a re-entry
point (see DESIGN_FROZEN.md's "tashih hasim" section for why).

TWO-PASS ARCHITECTURE (empirically required -- see below for why):

Pass A (run_pass_a) replays Item 1's trajectory with ZERO Item-4-specific
computation interleaved mid-round: only the exact operations Item 1's own
script performs (proven bit-exact against Item 1's own locked checkpoints
via an isolated diagnostic). At each logged round it snapshots everything
Item 4 needs (client_states, global_state_start, agg0_state, local_perf,
post_agg_perf, macro_f1_intervention0) to a per-(seed,round) checkpoint
file on disk, then continues the live trajectory untouched.

Pass B (run_pass_b) runs strictly AFTER Pass A has finished the ENTIRE
45-round trajectory for a seed (no interleaving with any live MPS
training loop is possible at that point). It loads each saved checkpoint
and, working entirely from CPU tensors, performs the mandatory
determinism gate and the A/B/C intervention construction/evaluation.

Why two passes: six independent single-pass fix attempts (RNG-consuming
model reconstruction eliminated; A/B/C evaluation isolated to CPU; A/B/C
*construction* (aggregate_weighted) also isolated to CPU; per-class
upd-before-evaluate reordering to match Item 1's exact interleaving;
Item 1's pre_local logit_margin() call restored; an explicit
torch.mps.synchronize() at Item 1's own checkpoint-save position) all
left the round-5 divergence from Item 1's locked checkpoint completely
unchanged in magnitude (bit-for-bit identical: 0.1501685380935669) or,
in one bisection variant that changed the *volume* of extra work instead
of its device/order, changed to a different but still nonzero magnitude
(0.0777...). A from-scratch verbatim replay of Item 1's own code (no
Item 4 additions at all) reproduced Item 1's own round-5 checkpoint
exactly (0.0 diff), and a bisected version of Item 4's own loop that
stops immediately after the Item-1-matching post-aggregation block (no
gate check, no A/B/C at all) also reproduced it exactly. Only adding the
A/B/C intervention-construction/evaluation block back in reintroduced a
nonzero, work-volume-dependent divergence. This is consistent with the
MPS backend's async kernel scheduling being sensitive to *any* extra
host-interleaved computation at a logged round, regardless of which
device that computation itself runs on -- not a logic bug fixable by
reordering or device placement within a single process pass. Physically
separating "advance the live trajectory" from "compute interventions"
into two passes (the second starting only once the first has moved on)
removes the interleaving entirely.

Determinism gate (mandatory, checked at every logged round before any
A/B/C interpretation is meaningful): intervention 0's aggregated
state_dict must match the corresponding Item 1 checkpoint
parameter-wise (torch.allclose) and its recall/margin must match the
locked Item 1 mechanistic CSV for the same (seed, round, class, holder,
eval_data='validation').
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from algorithms import local_train, aggregate_weighted, evaluate

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
PASS_A_DIR = HERE / "pass_a_checkpoints"
ITEM1_DIR = HERE.parent
ITEM1_CKPT_DIR = ITEM1_DIR / "checkpoints"
ITEM1_MECH_CSV = ITEM1_DIR / "results" / "update_conflict_mechanistic.csv"

SEEDS = [11, 22, 33, 44, 55]
LOGGED_ROUNDS = [1, 5, 10, 20, 30, 45]
NUM_ROUNDS = 45
BATCH_SIZE = 4096
LR = 1e-3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
CPU_DEVICE = torch.device("cpu")

RARE_CLASS_HOLDERS = {"Manipulation": [2, 13], "Replay": [1, 4, 10]}
ALL_HOLDER_CLIENTS = {c for hs in RARE_CLASS_HOLDERS.values() for c in hs}
INTERVENTIONS = ["0", "A", "B", "C"]
HEAD_W, HEAD_B = "net.6.weight", "net.6.bias"
REPR_KEYS = ["net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias"]

# Determinism-gate tolerances (frozen in DESIGN_FROZEN.md -- not tuned post hoc)
PARAM_ATOL, PARAM_RTOL = 1e-6, 1e-5
MAX_ABS_DIFF_TOL = 1e-5
MARGIN_ATOL = 1e-4


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def class_update_vector(local_state, global_state, class_idx):
    dw = (local_state[HEAD_W][class_idx] - global_state[HEAD_W][class_idx]).cpu().numpy()
    db = (local_state[HEAD_B][class_idx] - global_state[HEAD_B][class_idx]).cpu().numpy()
    return np.concatenate([dw, [db]])


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@torch.no_grad()
def logit_margin(model, X, y, class_idx, device, batch_size=4096):
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


def holder_renormalized_weights(client_ids, weights, holders):
    """w~_i = weight_i / sum_{j in holders} weight_j, for i in holders. Frozen
    in DESIGN_FROZEN.md to remove the weighting-vs-component confound between
    interventions A/B/C -- never a plain 1/|H_c| arithmetic mean."""
    idx_of = {cid: j for j, cid in enumerate(client_ids)}
    denom = sum(weights[idx_of[h]] for h in holders)
    return {h: weights[idx_of[h]] / denom for h in holders}


def build_intervention_state(client_states, client_ids, weights, holders, mode):
    """mode in {'0','A','B','C'}. Returns the counterfactual global
    state_dict. Representation/head keys not touched by a given mode follow
    standard full FedAvg (all 15 clients, original weights) -- see
    DESIGN_FROZEN.md's four aggregation-rule definitions."""
    ordered_states = [client_states[cid] for cid in client_ids]
    agg = aggregate_weighted(ordered_states, weights)
    if mode == "0":
        return agg
    w_tilde = holder_renormalized_weights(client_ids, weights, holders)
    if mode == "B":
        agg[HEAD_W] = sum(w_tilde[h] * client_states[h][HEAD_W] for h in holders)
        agg[HEAD_B] = sum(w_tilde[h] * client_states[h][HEAD_B] for h in holders)
        return agg
    if mode == "C":
        for k in REPR_KEYS:
            agg[k] = sum(w_tilde[h] * client_states[h][k] for h in holders)
        return agg
    raise ValueError(mode)


def apply_intervention_A_row(agg_state, client_states, client_ids, weights, holders, class_idx):
    w_tilde = holder_renormalized_weights(client_ids, weights, holders)
    agg_state[HEAD_W] = agg_state[HEAD_W].clone()
    agg_state[HEAD_B] = agg_state[HEAD_B].clone()
    agg_state[HEAD_W][class_idx] = sum(w_tilde[h] * client_states[h][HEAD_W][class_idx] for h in holders)
    agg_state[HEAD_B][class_idx] = sum(w_tilde[h] * client_states[h][HEAD_B][class_idx] for h in holders)
    return agg_state


def load_item1_mechanistic_lookup():
    """(seed,round,class,holder_client_id) -> (recall_post_aggregation,
    logit_margin_post_aggregation) on validation, from Item 1's locked CSV --
    the second half of the mandatory determinism gate for intervention 0."""
    lookup = {}
    with open(ITEM1_MECH_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["eval_data"] != "validation":
                continue
            key = (int(row["seed"]), int(row["round"]), row["class"], int(row["holder_client_id"]))
            lookup[key] = (float(row["recall_post_aggregation"]), float(row["logit_margin_post_aggregation"]))
    return lookup


def load_data():
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    client_ids = data.client_ids()
    client_data = {cid: data.get_client_data(cid) for cid in client_ids}
    weights = [data.client_weight_rows(cid) for cid in client_ids]
    s = sum(weights); weights = [w / s for w in weights]
    class_idx_of = {c: ALL_CATEGORIES.index(c) for c in RARE_CLASS_HOLDERS}
    Xval, yval = data.get_validation_data()
    return data, input_dim, client_ids, client_data, weights, class_idx_of, Xval, yval


# ============================== PASS A ==============================
# Pure Item-1-identical trajectory replay. No gate check, no A/B/C, no
# torch.load of Item 1's checkpoints, no CPU copies beyond what's needed
# to snapshot state to disk for Pass B -- nothing that isn't already part
# of Item 1's own script, so the live MPS trajectory is never perturbed.

def run_pass_a():
    PASS_A_DIR.mkdir(exist_ok=True)
    print(f"[Pass A] Device: {DEVICE}")
    data, input_dim, client_ids, client_data, weights, class_idx_of, Xval, yval = load_data()

    t0 = time.time()
    for seed in SEEDS:
        set_seed(seed)
        global_model = AttackFamilyMLP(input_dim).to(DEVICE)

        for rnd in range(1, NUM_ROUNDS + 1):
            log_this_round = rnd in LOGGED_ROUNDS
            global_state_start = {k: v.clone() for k, v in global_model.state_dict().items()}

            # Verbatim from Item 1 (update_conflict_analysis.py): both evaluate()
            # AND logit_margin() per (class, holder, eval_name). Proven bit-exact
            # against Item 1's own checkpoints via isolated diagnostic replay.
            if log_this_round:
                for cls, holders in RARE_CLASS_HOLDERS.items():
                    ci = class_idx_of[cls]
                    for cid in holders:
                        Xc, yc = client_data[cid]
                        for eval_name, (Xe, ye) in {
                            "own_client_rows": (Xc, yc), "validation": (Xval, yval)
                        }.items():
                            evaluate(global_model, Xe, ye, DEVICE, ALL_CATEGORIES,
                                     class_weights=data.class_weights_full_balanced,
                                     eval_label_indices=[ci])
                            logit_margin(global_model, Xe, ye, ci, DEVICE)

            local_perf = {}
            client_states = {}
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
                                    m["per_class_recall"][cls],
                                    logit_margin(local_model, Xe, ye, ci, DEVICE))

            ordered_states = [client_states[cid] for cid in client_ids]
            agg0_state = aggregate_weighted(ordered_states, weights)
            global_model.load_state_dict(agg0_state)  # this IS the continuing trajectory

            if log_this_round:
                post_agg_perf = {}
                for cls, holders in RARE_CLASS_HOLDERS.items():
                    ci = class_idx_of[cls]
                    for cid in holders:
                        Xc, yc = client_data[cid]
                        for eval_name, (Xe, ye) in {
                            "own_client_rows": (Xc, yc), "validation": (Xval, yval)
                        }.items():
                            m_global = evaluate(global_model, Xe, ye, DEVICE, ALL_CATEGORIES,
                                                 class_weights=data.class_weights_full_balanced,
                                                 eval_label_indices=[ci])
                            margin_global = logit_margin(global_model, Xe, ye, ci, DEVICE)
                            post_agg_perf[(cls, cid, eval_name)] = (
                                m_global["per_class_recall"][cls], margin_global)

                m_val_item1_pattern = evaluate(
                    global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                    class_weights=data.class_weights_full_balanced,
                    eval_label_indices=[i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"])
                macro_f1_intervention0 = m_val_item1_pattern["macro_f1"]

                # Snapshot everything Pass B needs, to CPU, for disk. This happens
                # AFTER the trajectory has already been advanced (load_state_dict
                # above already committed round rnd's result), and Pass B will not
                # run until Pass A finishes seed's entire 45-round loop -- no live
                # MPS trajectory is left to perturb once these files are read back.
                torch.save({
                    "global_state_start": {k: v.detach().cpu() for k, v in global_state_start.items()},
                    "agg0_state": {k: v.detach().cpu() for k, v in agg0_state.items()},
                    "client_states": {cid: {k: v.detach().cpu() for k, v in s.items()}
                                      for cid, s in client_states.items()},
                    "local_perf": local_perf,
                    "post_agg_perf": post_agg_perf,
                    "macro_f1_intervention0": macro_f1_intervention0,
                }, PASS_A_DIR / f"pass_a_seed{seed}_round{rnd}.pt")

                print(f"[Pass A][seed={seed}] round {rnd}/{NUM_ROUNDS} snapshotted  "
                      f"elapsed={time.time()-t0:.0f}s")

        print(f"[Pass A] Seed {seed} done. Elapsed {time.time()-t0:.0f}s total.")

    print("[Pass A] Complete -- all seeds' full 45-round trajectories snapshotted.")


# ============================== PASS B ==============================
# Runs entirely after Pass A. No live global_model, no training, no MPS
# trajectory to protect -- gate-check and A/B/C computation happen freely.

def run_pass_b():
    RESULTS_DIR.mkdir(exist_ok=True)
    data, input_dim, client_ids, client_data, weights, class_idx_of, Xval, yval = load_data()
    item1_lookup = load_item1_mechanistic_lookup()
    Xval_cpu, yval_cpu = Xval.cpu(), yval.cpu()
    scratch_model = AttackFamilyMLP(input_dim).to(CPU_DEVICE)

    out_path = RESULTS_DIR / "item4_raw.csv"
    gate_path = RESULTS_DIR / "item4_determinism_gate.csv"

    with open(out_path, "w", newline="") as of, open(gate_path, "w", newline="") as gf:
        ow = csv.writer(of)
        ow.writerow(["intervention", "seed", "round", "target_class", "holder_client_id",
                     "local_recall_holder_own_rows", "local_recall_holder_validation",
                     "logit_margin_holder_post_local",
                     "global_recall_post_intervention_validation",
                     "conditional_retention",
                     "logit_margin_post_intervention", "delta_margin",
                     "logit_margin_retention_ratio_secondary",
                     "holder_update_norm", "cosine_holder_vs_realized_aggregate",
                     "holder_weighted_sum_norm_baseline", "nonholder_weighted_sum_norm_baseline",
                     "nonholder_over_holder_contribution_ratio_baseline",
                     "intervention_c_macro_f1_all_classes", "intervention_c_flagged_invalid"])
        gw = csv.writer(gf)
        gw.writerow(["seed", "round", "max_abs_param_diff", "params_allclose",
                     "class", "holder_client_id",
                     "recall_match", "margin_abs_diff", "margin_match", "gate_pass"])

        gate_failed = False
        for seed in SEEDS:
            if gate_failed:
                break
            for rnd in LOGGED_ROUNDS:
                snap = torch.load(PASS_A_DIR / f"pass_a_seed{seed}_round{rnd}.pt", map_location="cpu")
                global_state_start = snap["global_state_start"]
                agg0_state = snap["agg0_state"]
                client_states = snap["client_states"]
                local_perf = snap["local_perf"]
                post_agg_perf = snap["post_agg_perf"]
                macro_f1_intervention0 = snap["macro_f1_intervention0"]

                ckpt_state = torch.load(ITEM1_CKPT_DIR / f"fedavg_seed{seed}_round{rnd}.pt",
                                         map_location="cpu")
                max_abs_diff = max(
                    (agg0_state[k] - ckpt_state[k]).abs().max().item() for k in agg0_state
                )
                params_allclose = all(
                    torch.allclose(agg0_state[k], ckpt_state[k], atol=PARAM_ATOL, rtol=PARAM_RTOL)
                    for k in agg0_state
                )

                for cls, holders in RARE_CLASS_HOLDERS.items():
                    for cid in holders:
                        recall0, margin0 = post_agg_perf[(cls, cid, "validation")]
                        locked_recall, locked_margin = item1_lookup[(seed, rnd, cls, cid)]
                        recall_match = abs(recall0 - locked_recall) < 1e-9
                        margin_abs_diff = abs(margin0 - locked_margin)
                        margin_match = margin_abs_diff < MARGIN_ATOL
                        gate_pass = (params_allclose and max_abs_diff < MAX_ABS_DIFF_TOL
                                     and recall_match and margin_match)
                        gw.writerow([seed, rnd, max_abs_diff, params_allclose, cls, cid,
                                     recall_match, margin_abs_diff, margin_match, gate_pass])
                        if not gate_pass:
                            gate_failed = True

                if gate_failed:
                    print(f"!!! DETERMINISM GATE FAILED at seed={seed} round={rnd} -- "
                          f"stopping Item 4 for investigation, per DESIGN_FROZEN.md.")
                    break

                idx_of = {cid: j for j, cid in enumerate(client_ids)}
                for cls, holders in RARE_CLASS_HOLDERS.items():
                    ci = class_idx_of[cls]

                    upd = {cid: class_update_vector(client_states[cid], global_state_start, ci)
                           for cid in client_ids}
                    weighted_upd = {cid: weights[idx_of[cid]] * upd[cid] for cid in client_ids}
                    holder_sum = sum(weighted_upd[cid] for cid in holders)
                    nonholder_sum = sum(weighted_upd[cid] for cid in client_ids if cid not in holders)
                    nonholder_over_holder = (float(np.linalg.norm(nonholder_sum)) /
                                              float(np.linalg.norm(holder_sum))
                                              if np.linalg.norm(holder_sum) > 1e-12 else float("nan"))

                    variants = {
                        "0": agg0_state,
                        "A": apply_intervention_A_row(
                            {k: v.clone() for k, v in agg0_state.items()},
                            client_states, client_ids, weights, holders, ci),
                        "B": build_intervention_state(client_states, client_ids, weights, holders, "B"),
                        "C": build_intervention_state(client_states, client_ids, weights, holders, "C"),
                    }

                    scratch_model.load_state_dict(variants["C"])
                    m_val_all_c = evaluate(scratch_model, Xval_cpu, yval_cpu, CPU_DEVICE, ALL_CATEGORIES,
                                            class_weights=data.class_weights_full_balanced,
                                            eval_label_indices=[i for i in range(10)
                                                                 if ALL_CATEGORIES[i] != "Password Cracking"])
                    macro_f1_c = m_val_all_c["macro_f1"]
                    c_flagged_invalid = (macro_f1_c < macro_f1_intervention0 - 0.15)

                    for mode in INTERVENTIONS:
                        scratch_model.load_state_dict(variants[mode])
                        m_val = evaluate(scratch_model, Xval_cpu, yval_cpu, CPU_DEVICE, ALL_CATEGORIES,
                                          class_weights=data.class_weights_full_balanced,
                                          eval_label_indices=[ci])
                        global_recall = m_val["per_class_recall"][cls]
                        global_margin = logit_margin(scratch_model, Xval_cpu, yval_cpu, ci, CPU_DEVICE)

                        realized_row_w = variants[mode][HEAD_W][ci] - global_state_start[HEAD_W][ci]
                        realized_row_b = variants[mode][HEAD_B][ci] - global_state_start[HEAD_B][ci]
                        realized_update = np.concatenate([
                            realized_row_w.numpy(), [realized_row_b.numpy().item()]])

                        for cid in holders:
                            local_recall_own, local_margin_own = local_perf[(cls, cid, "own_client_rows")]
                            local_recall_val, local_margin_val = local_perf[(cls, cid, "validation")]
                            cond_retention = int(local_recall_val > 0 and global_recall > 0)
                            delta_margin = global_margin - local_margin_val
                            ratio = (global_margin / local_margin_val
                                     if local_margin_val > 0 else float("nan"))
                            ow.writerow([
                                mode, seed, rnd, cls, cid,
                                local_recall_own, local_recall_val, local_margin_val,
                                global_recall, cond_retention,
                                global_margin, delta_margin, ratio,
                                float(np.linalg.norm(upd[cid])),
                                cosine(upd[cid], realized_update),
                                float(np.linalg.norm(holder_sum)), float(np.linalg.norm(nonholder_sum)),
                                nonholder_over_holder,
                                macro_f1_c if mode == "C" else "",
                                c_flagged_invalid if mode == "C" else "",
                            ])

                print(f"[Pass B][seed={seed}] round {rnd} gate_pass=True")

            of.flush(); gf.flush()
            print(f"[Pass B] Seed {seed} done.")

    print(f"\nSaved -> {out_path}\nSaved -> {gate_path}")
    if gate_failed:
        print("\n!!! Item 4 STOPPED due to determinism-gate failure. Investigate before any A/B/C interpretation.")


def main():
    run_pass_a()
    run_pass_b()


if __name__ == "__main__":
    main()
