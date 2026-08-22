"""
Item 2 -- SINGLE final held-out test evaluation pass. No retraining, no
threshold/hyperparameter changes, no post-hoc selection. Evaluates the 23
available round-45 checkpoints (25 planned minus the 2 FedAvg-SGD seeds
that diverged at round 2 and have no round-45 checkpoint -- those seeds are
excluded entirely, never evaluated on a partial/diverged model).

Test data uses ALL 10 classes (Password Cracking IS present in the 28
held-out test sessions, unlike validation where it was absent -- this is the
first evaluation in Item 2 where that class is scored at all).

Saved per checkpoint:
  - pooled (row-level) precision/recall/F1/predicted-count for all 10 classes
    + pooled macro-F1
  - confusion matrix (10x10)
  - per-session results (each of the 28 test sessions individually)
  - session-level MACRO summary (mean of each session's own macro-F1 -- an
    unweighted average across sessions so the largest sessions cannot
    dominate the pooled row-level number)
Then, per algorithm: 95% Student's-t CI across successful seeds (pooled
metric), AND a cluster-bootstrap (resampling the 28 test sessions with
replacement, 2000 resamples) to quantify session-selection uncertainty
separately from seed uncertainty.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP, load_checkpoint

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints_five_seed"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
SEEDS = [11, 22, 33, 44, 55]
ALGORITHMS = ["fedavg_sgd", "fednova_sgd", "scaffold_uniform", "scaffold_weighted", "fedadam"]
ALL_LABELS = list(range(10))
N_BOOT = 2000


@torch.no_grad()
def predict(model, X, batch_size=4096):
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(DEVICE)
        preds.append(model(xb).argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def per_class_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=ALL_LABELS,
                                                                 average=None, zero_division=0)
    pred_counts = np.bincount(y_pred, minlength=10)
    macro_f1 = f1_score(y_true, y_pred, labels=ALL_LABELS, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        "per_class_precision": {ALL_CATEGORIES[i]: float(precision[i]) for i in ALL_LABELS},
        "per_class_recall": {ALL_CATEGORIES[i]: float(recall[i]) for i in ALL_LABELS},
        "per_class_f1": {ALL_CATEGORIES[i]: float(f1[i]) for i in ALL_LABELS},
        "per_class_predicted_count": {ALL_CATEGORIES[i]: int(pred_counts[i]) for i in ALL_LABELS},
    }


def main():
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)

    test_sessions = data.manifest["held_out_test_sessions"]
    print(f"Held-out test sessions: {len(test_sessions)}")

    # per-session tensors, built once (session identity preserved), reused for every checkpoint
    session_tensors = {}
    for sess in test_sessions:
        cat = data._cat_of[sess]
        df = data._load_sessions([sess])
        Xs, ys = data.tensors_from_df(df)
        session_tensors[sess] = (Xs, ys, cat)
    print(f"Loaded {len(session_tensors)} test sessions.")

    # pooled tensors for the full-test-set pass
    Xtest, ytest = data.get_final_test_data()
    print(f"Pooled test rows: {len(Xtest)}")

    pooled_path = RESULTS_DIR / "item2_final_test_pooled_results.csv"
    session_path = RESULTS_DIR / "item2_final_test_per_session_results.csv"
    cm_path = RESULTS_DIR / "item2_final_test_confusion_matrices.json"

    pooled_rows = []
    session_rows = []
    confusion_matrices = {}
    fedavg_sgd_available_seeds = []

    for algo in ALGORITHMS:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"{algo}_seed{seed}_round45.pt"
            if not ckpt_path.exists():
                print(f"[{algo} seed={seed}] no round-45 checkpoint -- SKIPPED (diverged before round 45)")
                continue
            if algo == "fedavg_sgd":
                fedavg_sgd_available_seeds.append(seed)

            model = AttackFamilyMLP(input_dim).to(DEVICE)
            load_checkpoint(model, ckpt_path, map_location=DEVICE)
            model.eval()

            # pooled
            preds = predict(model, Xtest)
            m = per_class_metrics(ytest.numpy(), preds)
            cm = confusion_matrix(ytest.numpy(), preds, labels=ALL_LABELS)
            confusion_matrices[f"{algo}_seed{seed}"] = cm.tolist()
            pooled_rows.append({"algorithm": algo, "seed": seed, "n_rows": len(ytest), **{
                "macro_f1": m["macro_f1"],
                "per_class_precision_json": json.dumps(m["per_class_precision"]),
                "per_class_recall_json": json.dumps(m["per_class_recall"]),
                "per_class_f1_json": json.dumps(m["per_class_f1"]),
                "per_class_predicted_count_json": json.dumps(m["per_class_predicted_count"]),
            }})

            # per-session
            session_macro_f1s = []
            for sess, (Xs, ys, cat) in session_tensors.items():
                sp = predict(model, Xs)
                sm = per_class_metrics(ys.numpy(), sp)
                session_macro_f1s.append(sm["macro_f1"])
                session_rows.append({"algorithm": algo, "seed": seed, "session": sess, "category": cat,
                                      "n_rows": len(ys), "macro_f1": sm["macro_f1"],
                                      "per_class_recall_json": json.dumps(sm["per_class_recall"])})
            session_macro_mean = float(np.mean(session_macro_f1s))

            print(f"[{algo} seed={seed}] pooled_macro_f1={m['macro_f1']:.4f}  "
                  f"session_level_macro_mean={session_macro_mean:.4f}  "
                  f"Manip_recall={m['per_class_recall']['Manipulation']:.3f}  "
                  f"Replay_recall={m['per_class_recall']['Replay']:.3f}  "
                  f"PwdCracking_recall={m['per_class_recall']['Password Cracking']:.3f}")

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df["session_level_macro_f1_mean"] = pooled_df.apply(
        lambda r: np.mean([sr["macro_f1"] for sr in session_rows
                            if sr["algorithm"] == r["algorithm"] and sr["seed"] == r["seed"]]), axis=1)
    pooled_df.to_csv(pooled_path, index=False)
    pd.DataFrame(session_rows).to_csv(session_path, index=False)
    with open(cm_path, "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    print(f"\nSaved -> {pooled_path}\nSaved -> {session_path}\nSaved -> {cm_path}")

    # --- summary: t-CI across seeds (pooled), + cluster-bootstrap over sessions ---
    print("\n=== Final test summary: 95% t-CI across seeds (pooled macro-F1, 10-class) ===")
    summary_rows = []
    for algo in ALGORITHMS:
        sub = pooled_df[pooled_df.algorithm == algo]
        n = len(sub)
        vals = sub.macro_f1.values
        mean, sd = vals.mean(), vals.std(ddof=1) if n > 1 else float("nan")
        if n > 1:
            tcrit = stats.t.ppf(0.975, df=n - 1)
            se = sd / np.sqrt(n)
            lo, hi = mean - tcrit * se, mean + tcrit * se
        else:
            lo, hi = float("nan"), float("nan")
        conditional = " (CONDITIONAL -- see divergence note)" if algo == "fedavg_sgd" else ""
        print(f"{algo:20s} n={n}  mean={mean:.4f}  sd={sd:.4f}  95%t-CI=[{lo:.4f},{hi:.4f}]{conditional}")
        summary_rows.append({"algorithm": algo, "n_seeds": n, "mean_macro_f1": mean, "sd": sd,
                              "ci_lo": lo, "ci_hi": hi, "conditional": algo == "fedavg_sgd"})

    print("\n=== Cluster-bootstrap over the 28 test sessions (session-selection uncertainty), per algorithm, seed=11 ===")
    rng = np.random.default_rng(0)
    boot_rows = []
    for algo in ALGORITHMS:
        ckpt_path = CKPT_DIR / f"{algo}_seed11_round45.pt"
        if not ckpt_path.exists():
            continue
        model = AttackFamilyMLP(input_dim).to(DEVICE)
        load_checkpoint(model, ckpt_path, map_location=DEVICE)
        model.eval()
        sess_true_pred = {}
        for sess, (Xs, ys, cat) in session_tensors.items():
            sp = predict(model, Xs)
            sess_true_pred[sess] = (ys.numpy(), sp)
        boots = []
        sess_list = list(sess_true_pred.keys())
        for _ in range(N_BOOT):
            sample = rng.choice(sess_list, size=len(sess_list), replace=True)
            yt = np.concatenate([sess_true_pred[s][0] for s in sample])
            yp = np.concatenate([sess_true_pred[s][1] for s in sample])
            boots.append(f1_score(yt, yp, labels=ALL_LABELS, average="macro", zero_division=0))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        print(f"{algo:20s} (seed=11)  point={np.mean(boots):.4f}  95% cluster-bootstrap CI=[{lo:.4f},{hi:.4f}]")
        boot_rows.append({"algorithm": algo, "seed": 11, "boot_mean": float(np.mean(boots)),
                           "boot_ci_lo": float(lo), "boot_ci_hi": float(hi)})

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "item2_final_test_seed_ci_summary.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(RESULTS_DIR / "item2_final_test_session_bootstrap_ci.csv", index=False)
    print(f"\nfedavg_sgd evaluated seeds (excludes diverged 44,55): {fedavg_sgd_available_seeds}")
    print("\nDONE. held_out_test evaluated exactly once, no retraining, no post-hoc selection.")


if __name__ == "__main__":
    main()
