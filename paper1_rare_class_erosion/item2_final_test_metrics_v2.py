"""
Item 2 -- corrected final test metrics. The original "session_level_macro_f1
_mean" (naive average of each session's own 10-class macro-F1) was a
methodological bug: most test sessions are near-single-category, so 9
absent classes get zero-recall-by-definition penalties that have nothing to
do with model performance. That number is DISCARDED as a headline metric
(kept only as a documented "present-label per-session macro-F1" side
analysis, never called "session-level macro-F1").

Three frozen metric definitions, per review:
  1. Pooled macro-F1 (10-class, all test rows together) -- the primary
     metric, unchanged from the original run. Every prediction is re-run
     here from the SAME frozen checkpoints (inference only, no retraining,
     no hyperparameter change) and its pooled macro-F1 is hash/value
     -checked against the original run's saved number to prove the
     predictions are identical (determinism check), not a new "look" that
     could change any decision.
  2. Session-balanced macro-F1: every ROW gets sample_weight = 1/n_s (its
     own session's row count), so every session contributes equal total
     weight regardless of size, computed via sklearn's sample_weight= over
     the full pooled 10-class prediction set -- keeps precision meaningful
     (unlike per-session-only F1) while preventing large sessions from
     dominating.
  3. Hierarchical session-macro recall: R_{c,s} = recall of class c within
     session s (only for sessions containing >=1 true row of c); R_c =
     mean over those sessions; R_session-macro = mean over the 10 R_c
     values. Equal weight per class, then equal weight per session within
     each class -- the primary metric for rare-class-across-sessions
     analysis.

Also saves: per-session accuracy (median/IQR/min across the 28 sessions),
per-class-per-session recall table, t-CI across seeds (pooled AND
session-balanced), and a cluster-bootstrap over the 28 sessions that
recomputes session-balanced macro-F1 within each resample (not the naive
pooled metric) to quantify session-selection uncertainty consistently with
metric #2.

No retraining. No hyperparameter change. No new algorithm decision made
from this recomputation -- it corrects how an already-fixed set of
predictions is summarized.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, f1_score

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


def main():
    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    test_sessions = data.manifest["held_out_test_sessions"]

    # build pooled arrays WITH session id + row-weight (1/n_s), reused for every checkpoint
    y_true_all, session_id_all = [], []
    for sess in test_sessions:
        df = data._load_sessions([sess])
        _, ys = data.tensors_from_df(df)
        y_true_all.append(ys.numpy())
        session_id_all.extend([sess] * len(ys))
    y_true_all = np.concatenate(y_true_all)
    session_id_all = np.array(session_id_all)
    session_sizes = pd.Series(session_id_all).value_counts().to_dict()
    row_weights = np.array([1.0 / session_sizes[s] for s in session_id_all])
    print(f"Pooled rows: {len(y_true_all)}  sessions: {len(test_sessions)}  "
          f"session size range: {min(session_sizes.values())}-{max(session_sizes.values())}")

    Xtest, ytest_check = data.get_final_test_data()
    assert np.array_equal(ytest_check.numpy(), y_true_all), "row order mismatch vs original pooled test load"

    pooled_prev = pd.read_csv(RESULTS_DIR / "item2_final_test_pooled_results.csv")

    results = []
    per_class_session_recall_rows = []
    per_session_accuracy_rows = []
    all_preds_hash = {}

    for algo in ALGORITHMS:
        for seed in SEEDS:
            ckpt_path = CKPT_DIR / f"{algo}_seed{seed}_round45.pt"
            if not ckpt_path.exists():
                continue
            model = AttackFamilyMLP(input_dim).to(DEVICE)
            load_checkpoint(model, ckpt_path, map_location=DEVICE)
            model.eval()
            preds = predict(model, Xtest)

            h = hashlib.sha256(preds.tobytes()).hexdigest()[:16]
            all_preds_hash[f"{algo}_seed{seed}"] = h

            # 1. pooled macro-F1 (determinism check against original run)
            pooled_f1 = f1_score(y_true_all, preds, labels=ALL_LABELS, average="macro", zero_division=0)
            prev_row = pooled_prev[(pooled_prev.algorithm == algo) & (pooled_prev.seed == seed)]
            prev_f1 = float(prev_row.macro_f1.iloc[0]) if len(prev_row) else None
            match = (prev_f1 is not None and abs(pooled_f1 - prev_f1) < 1e-9)

            # 2. session-balanced macro-F1 (sample_weight = 1/n_s per row)
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                y_true_all, preds, labels=ALL_LABELS, average=None, zero_division=0, sample_weight=row_weights)
            session_balanced_macro_f1 = float(np.mean(f1_w))

            # 3. hierarchical session-macro recall
            per_class_recall = {}
            for ci in ALL_LABELS:
                cls_name = ALL_CATEGORIES[ci]
                sessions_with_c = np.unique(session_id_all[y_true_all == ci])
                r_c_s = []
                for s in sessions_with_c:
                    mask = (session_id_all == s) & (y_true_all == ci)
                    n_c_s = mask.sum()
                    if n_c_s == 0:
                        continue
                    correct = (preds[mask] == ci).sum()
                    r_c_s.append(correct / n_c_s)
                    per_class_session_recall_rows.append({"algorithm": algo, "seed": seed, "class": cls_name,
                                                            "session": s, "n_rows": int(n_c_s),
                                                            "recall": float(correct / n_c_s)})
                per_class_recall[cls_name] = float(np.mean(r_c_s)) if r_c_s else float("nan")
            hierarchical_session_macro_recall = float(np.nanmean(list(per_class_recall.values())))

            # per-session accuracy
            session_acc = []
            for s in test_sessions:
                mask = session_id_all == s
                acc = (preds[mask] == y_true_all[mask]).mean()
                session_acc.append(acc)
                per_session_accuracy_rows.append({"algorithm": algo, "seed": seed, "session": s,
                                                   "n_rows": int(mask.sum()), "accuracy": float(acc)})
            session_acc = np.array(session_acc)

            results.append({
                "algorithm": algo, "seed": seed, "pred_hash": h,
                "pooled_macro_f1": pooled_f1, "pooled_matches_original_run": match,
                "session_balanced_macro_f1": session_balanced_macro_f1,
                "hierarchical_session_macro_recall": hierarchical_session_macro_recall,
                "session_accuracy_median": float(np.median(session_acc)),
                "session_accuracy_q1": float(np.quantile(session_acc, 0.25)),
                "session_accuracy_q3": float(np.quantile(session_acc, 0.75)),
                "session_accuracy_min": float(session_acc.min()),
                "per_class_hierarchical_recall_json": json.dumps(per_class_recall),
            })
            print(f"[{algo} seed={seed}] pooled={pooled_f1:.4f} (matches_orig={match})  "
                  f"session_balanced={session_balanced_macro_f1:.4f}  "
                  f"hierarchical_session_macro_recall={hierarchical_session_macro_recall:.4f}  "
                  f"Manip_R_c={per_class_recall['Manipulation']:.3f}  Replay_R_c={per_class_recall['Replay']:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "item2_final_test_metrics_v2.csv", index=False)
    pd.DataFrame(per_class_session_recall_rows).to_csv(RESULTS_DIR / "item2_final_test_per_class_session_recall.csv", index=False)
    pd.DataFrame(per_session_accuracy_rows).to_csv(RESULTS_DIR / "item2_final_test_per_session_accuracy.csv", index=False)

    print(f"\nAll pooled macro-F1 match original run: {res_df.pooled_matches_original_run.all()}")

    # --- t-CI across seeds, both metrics ---
    print("\n=== 95% t-CI across seeds ===")
    summary = []
    for algo in ALGORITHMS:
        sub = res_df[res_df.algorithm == algo]
        n = len(sub)
        for metric in ["pooled_macro_f1", "session_balanced_macro_f1", "hierarchical_session_macro_recall"]:
            vals = sub[metric].values
            mean, sd = vals.mean(), vals.std(ddof=1) if n > 1 else float("nan")
            if n > 1:
                tcrit = stats.t.ppf(0.975, df=n - 1)
                se = sd / np.sqrt(n)
                lo, hi = mean - tcrit * se, mean + tcrit * se
            else:
                lo, hi = float("nan"), float("nan")
            cond = " CONDITIONAL(n=3,40% divergence)" if algo == "fedavg_sgd" else ""
            print(f"{algo:20s} {metric:32s} n={n}  mean={mean:.4f}  sd={sd:.4f}  95%CI=[{lo:.4f},{hi:.4f}]{cond}")
            summary.append({"algorithm": algo, "metric": metric, "n": n, "mean": mean, "sd": sd,
                             "ci_lo": lo, "ci_hi": hi, "conditional": algo == "fedavg_sgd"})
    pd.DataFrame(summary).to_csv(RESULTS_DIR / "item2_final_test_ci_summary_v2.csv", index=False)

    # --- cluster-bootstrap over sessions, recomputing SESSION-BALANCED macro-F1 within each resample ---
    print("\n=== Cluster-bootstrap over 28 test sessions (session-balanced macro-F1), seed=11 ===")
    rng = np.random.default_rng(0)
    boot_rows = []
    for algo in ALGORITHMS:
        ckpt_path = CKPT_DIR / f"{algo}_seed11_round45.pt"
        if not ckpt_path.exists():
            continue
        model = AttackFamilyMLP(input_dim).to(DEVICE)
        load_checkpoint(model, ckpt_path, map_location=DEVICE)
        model.eval()
        preds = predict(model, Xtest)

        boots = []
        for _ in range(N_BOOT):
            sample_sessions = rng.choice(test_sessions, size=len(test_sessions), replace=True)
            mask_idx = []
            w = []
            for s in sample_sessions:
                idx = np.where(session_id_all == s)[0]
                mask_idx.append(idx)
                w.append(np.full(len(idx), 1.0 / len(idx)))
            mask_idx = np.concatenate(mask_idx)
            w = np.concatenate(w)
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                y_true_all[mask_idx], preds[mask_idx], labels=ALL_LABELS, average=None,
                zero_division=0, sample_weight=w)
            boots.append(float(np.mean(f1_w)))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        print(f"{algo:20s} (seed=11)  point={np.mean(boots):.4f}  95% cluster-bootstrap CI=[{lo:.4f},{hi:.4f}]")
        boot_rows.append({"algorithm": algo, "seed": 11, "metric": "session_balanced_macro_f1",
                           "boot_mean": float(np.mean(boots)), "boot_ci_lo": float(lo), "boot_ci_hi": float(hi)})
    pd.DataFrame(boot_rows).to_csv(RESULTS_DIR / "item2_final_test_session_bootstrap_ci_v2.csv", index=False)

    with open(RESULTS_DIR / "item2_final_test_prediction_hashes.json", "w") as f:
        json.dump(all_preds_hash, f, indent=2)

    print("\nDONE. Inference-only recomputation from frozen checkpoints -- no retraining, "
          "no hyperparameter change, no new algorithm-selection decision.")


if __name__ == "__main__":
    main()
