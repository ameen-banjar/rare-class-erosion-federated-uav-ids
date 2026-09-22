"""
Item 5 (frozen design: DESIGN_FROZEN.md) -- cross-domain external
validation on CICIoT2023. Reuses local_train/aggregate_weighted/evaluate
(FedAvg) and local_train_fednova/aggregate_fednova (FedNova) from
shared/fl_pipeline unchanged. 5 independent Dirichlet partition draws x
3 model seeds x 2 algorithms = 30 runs. Partition draw is the independent
statistical unit; model seeds are nested within partition (Section 9).

Endpoints (Section 10): primary = RRG (Recall Retention Gap); secondary =
strict/practical erosion, conditional retention; context = per-class
recall (all 34 classes) and aggregate macro-F1. Computed only for the
frozen rare-tail subset (Section 4) at each logged round, holder status
determined per partition draw (Section 7), not a fixed manifest.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))

from ciciot_data import CICIoTFederatedData
from model import AttackFamilyMLP
from algorithms import local_train, aggregate_weighted, evaluate
from advanced_algorithms import local_train_fednova, aggregate_fednova

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

RARE_TAIL = ["Uploading_Attack", "Recon-PingSweep", "Backdoor_Malware", "XSS", "SqlInjection"]
ALGORITHMS = ["FedAvg", "FedNova"]
PARTITION_SEEDS = [101, 102, 103, 104, 105]
MODEL_SEEDS = [11, 22, 33]
N_CLIENTS = 15
ALPHA = 0.1
NUM_ROUNDS = 20
LOGGED_ROUNDS = [1, 5, 10, 15, 20]
# Amendment (DESIGN_FROZEN.md changelog, 2026-09-20): the initial draft used
# 256 under the mistaken assumption it matched Item 1/4's protocol; code
# audit confirmed Item 1's own update_conflict_analysis.py explicitly sets
# BATCH_SIZE=4096 (256 is only local_train()'s generic default, never the
# value Item 1/4 actually ran with). Corrected to 4096 for consistency with
# the primary ISOT protocol -- not a new tuning choice.
BATCH_SIZE = 4096
LR = 1e-3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

MIN_SUPPORT = 50  # Section 4 fallback rule: insufficient_evaluation_support if val/test < this


def set_model_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_one(algorithm, partition_seed, model_seed, data, class_idx_of, Xval, yval, ow, lw):
    client_data, weights, holder_map = data.make_dirichlet_partition(partition_seed, ALPHA, N_CLIENTS)
    client_ids = [c for c in range(N_CLIENTS) if weights[c] > 0]
    input_dim = len(data.feature_cols)
    n_classes = len(data.all_categories)

    # Leakage check (Section 13-b): TRAIN partition rows vs VAL/TEST must
    # never share a fingerprint. Client data comes from data.train_df only,
    # by construction (make_dirichlet_partition indexes train_df exclusively),
    # so no runtime check is needed here; this is a structural guarantee, not
    # a per-run test -- documented in DESIGN_FROZEN.md Section 13.

    set_model_seed(model_seed)
    global_model = AttackFamilyMLP(input_dim, n_classes=n_classes).to(DEVICE)

    for rnd in range(1, NUM_ROUNDS + 1):
        log_this_round = rnd in LOGGED_ROUNDS
        # CPU, matching Item 2's established FedNova/SCAFFOLD/FedAdam pattern
        # (item2_five_seed_final.py) exactly -- aggregate_fednova combines
        # this with client_states, which local_train_fednova already returns
        # on CPU; local_model.load_state_dict() copies cross-device safely,
        # so FedAvg's path (device-resident client_states via
        # get_state_dict_copy) is unaffected by this being CPU-resident.
        global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

        client_states = {}
        taus = {}
        local_perf = {}  # (cls, cid) -> recall on validation, post-local-training
        for cid in client_ids:
            Xc, yc = client_data[cid]
            local_model = AttackFamilyMLP(input_dim, n_classes=n_classes).to(DEVICE)
            local_model.load_state_dict(global_state_start)

            if algorithm == "FedAvg":
                state, n, loss = local_train(local_model, Xc, yc, DEVICE, epochs=1,
                                              batch_size=BATCH_SIZE, lr=LR,
                                              class_weights=data.class_weights_full_balanced)
            else:
                state, n_steps, n, loss = local_train_fednova(local_model, Xc, yc, DEVICE, epochs=1,
                                                                batch_size=BATCH_SIZE, lr=LR,
                                                                optimizer="sgd",
                                                                class_weights=data.class_weights_full_balanced)
                taus[cid] = n_steps
            client_states[cid] = state

            if log_this_round:
                holder_classes_this_client = [cls for cls in RARE_TAIL if cid in holder_map[cls]]
                if holder_classes_this_client:
                    # per_class_recall already covers every class in one call
                    # (eval_label_indices only affects macro_f1 averaging) --
                    # one evaluate() per holder-client, not one per class,
                    # avoids redundant full-validation forward passes.
                    m = evaluate(local_model, Xval, yval, DEVICE, data.all_categories,
                                 class_weights=data.class_weights_full_balanced)
                    for cls in holder_classes_this_client:
                        local_perf[(cls, cid)] = m["per_class_recall"][cls]

        active_weights = [weights[c] for c in client_ids]
        s = sum(active_weights)
        active_weights = [w / s for w in active_weights]
        ordered_states = [client_states[c] for c in client_ids]

        if algorithm == "FedAvg":
            agg_state = aggregate_weighted(ordered_states, active_weights)
        else:
            ordered_taus = [taus[c] for c in client_ids]
            agg_state = aggregate_fednova(ordered_states, global_state_start, active_weights, ordered_taus)
        global_model.load_state_dict(agg_state)

        if log_this_round:
            m_val_all = evaluate(global_model, Xval, yval, DEVICE, data.all_categories,
                                  class_weights=data.class_weights_full_balanced)
            macro_f1_all = m_val_all["macro_f1"]

            for cls in RARE_TAIL:
                holders = holder_map[cls]
                global_recall = m_val_all["per_class_recall"][cls]

                if not holders:
                    lw.writerow([algorithm, partition_seed, model_seed, rnd, cls,
                                 "no_holder_in_this_draw", "", "", "", ""])
                    continue

                local_recalls = [local_perf.get((cls, cid), 0.0) for cid in holders]
                best_local = max(local_recalls)
                rrg = best_local - global_recall
                strict_erosion = int(best_local > 0 and global_recall == 0)
                practical_erosion = int(best_local >= 0.05 and global_recall == 0)
                conditional_retention = int(best_local > 0 and global_recall > 0)

                ow.writerow([
                    algorithm, partition_seed, model_seed, rnd, cls,
                    len(holders), best_local, global_recall, rrg,
                    strict_erosion, practical_erosion, conditional_retention,
                    macro_f1_all,
                ])

            print(f"  [{algorithm} p{partition_seed} s{model_seed}] round {rnd}/{NUM_ROUNDS} logged")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}")

    data = CICIoTFederatedData()
    data.fit_preprocessing()
    class_idx_of = {c: data.all_categories.index(c) for c in RARE_TAIL}
    Xval, yval = data.get_validation_data()

    # Report evaluation support for the rare tail against Section 4's
    # MIN_SUPPORT fallback rule (val/test >= 50 examples), before any run.
    val_counts = data.val_df["cls"].value_counts()
    test_counts = data.test_df["cls"].value_counts()
    print("\nRare-tail evaluation support check (Section 4 fallback rule):")
    for cls in RARE_TAIL:
        vc, tc = int(val_counts.get(cls, 0)), int(test_counts.get(cls, 0))
        status = "OK" if vc >= MIN_SUPPORT and tc >= MIN_SUPPORT else "INSUFFICIENT_EVALUATION_SUPPORT"
        print(f"  {cls:20s} val={vc:6d} test={tc:6d} -> {status}")

    out_path = RESULTS_DIR / "item5_raw.csv"
    leak_path = RESULTS_DIR / "item5_no_holder_log.csv"

    t0 = time.time()
    with open(out_path, "w", newline="") as of, open(leak_path, "w", newline="") as lf:
        ow = csv.writer(of)
        ow.writerow(["algorithm", "partition_seed", "model_seed", "round", "class",
                     "n_holders_this_draw", "best_holder_local_recall_validation",
                     "global_recall_validation", "RRG",
                     "strict_erosion", "practical_erosion", "conditional_retention",
                     "macro_f1_all_classes_validation"])
        lw = csv.writer(lf)
        lw.writerow(["algorithm", "partition_seed", "model_seed", "round", "class", "note", "", "", "", ""])

        for partition_seed in PARTITION_SEEDS:
            for model_seed in MODEL_SEEDS:
                for algorithm in ALGORITHMS:
                    run_one(algorithm, partition_seed, model_seed, data, class_idx_of, Xval, yval, ow, lw)
                    of.flush()
            print(f"[partition_seed={partition_seed}] all model_seeds/algorithms done. "
                  f"elapsed={time.time()-t0:.0f}s")

    print(f"\nSaved -> {out_path}\nSaved -> {leak_path}")


if __name__ == "__main__":
    main()
