"""
Reproducible Item 3 exploratory-grid analysis, implementing
ANALYSIS_PLAN_FROZEN.md exactly, AS AMENDED 2026-08-22 (evaluable-class
correction -- see ANALYSIS_PLAN_FROZEN.md changelog). Round 45 only,
model_seed=11 fixed throughout (partition-variance only). Never touches
held_out_test.

AMENDMENT: the confirmation-phase analysis (analyze_confirmation.py) caught
that Password Cracking has 0 rows in the validation split, so its recorded
recall is 0 by sklearn construction in every single run regardless of what
was locally learned -- unconditionally miscounted as "zero recall" wherever
recall==0 is used as a target/indicator. This exploratory script uses the
IDENTICAL `zero_recall = int(recall==0)` pattern (feeding Q3/Q3b/Q4/Q5/Q5b/
Q6), inherited from `evaluate()`'s per-class recall which is computed over
ALL classes, not just VAL_LABEL_INDICES -- so it has the SAME defect,
independently reconfirmed here (all 54 round-45 rows: Password Cracking
recall == 0.0 exactly). Fixed the same way: a class is "evaluable" only if
it has nonzero true support in validation (computed from data, not
hardcoded by name); excluded from every zero-recall-derived denominator/
correlation. Q1 (realized concentration) and Q2 (holder counts) and the
anchor-selection composite score do not use recall and are UNAFFECTED --
the 3 confirmation anchor partitions selected from them remain valid, no
re-selection or retraining needed.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
ALL_CATEGORIES = ["DoS", "Injection", "Ip Spoofing", "MITM", "Manipulation",
                   "Password Cracking", "Regular", "Replay", "Unauth", "Video"]
ANCHOR_CASES = [(0.1, 30), (0.3, 15), (1.0, 10)]

sys.path.insert(0, str(ROOT.parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData


def evaluable_classes():
    d = ISOTFederatedData()
    d.fit_preprocessing()
    _, yval = d.get_validation_data()
    support = np.bincount(yval.numpy(), minlength=len(ALL_CATEGORIES))
    evaluable = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if support[i] > 0}
    non_evaluable = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if support[i] == 0}
    return evaluable, non_evaluable


def load_round45():
    df = pd.read_csv(RESULTS_DIR / "item3_exploratory_round_metrics.csv")
    df = df[(df["round"] == 45) & (~df.diverged)].copy()
    df["recall"] = df.per_class_recall_json.apply(json.loads)
    return df


def load_diagnostics():
    diag = json.load(open(RESULTS_DIR / "item3_exploratory_partition_diagnostics.json"))
    by_key = {(d["alpha"], d["n_clients"], d["partition_seed"]): d for d in diag}
    return diag, by_key


def holder_weight_share(diag_entry, cls):
    cdiag = diag_entry["client_diagnostics"]
    total_rows = sum(v["n_rows"] for v in cdiag.values())
    if total_rows == 0:
        return 0.0
    holder_rows = sum(v["n_rows"] for v in cdiag.values() if v["row_counts_by_category"].get(cls, 0) > 0)
    return holder_rows / total_rows  # NOTE: this is holders' OWN total-row share, not class-specific weight


def main():
    df = load_round45()
    diag_list, diag_by_key = load_diagnostics()
    print(f"Loaded {len(df)} round-45 rows (model_seed=11 fixed throughout).")
    print(f"Loaded {len(diag_list)} partition diagnostics.\n")

    EVALUABLE, NON_EVALUABLE = evaluable_classes()
    print(f"Validation-evaluable classes ({len(EVALUABLE)}/{len(ALL_CATEGORIES)}): {sorted(EVALUABLE)}")
    print(f"NON-evaluable classes (0 validation support -- excluded from zero-recall Q3/Q3b/Q4/Q5/Q5b/Q6): {sorted(NON_EVALUABLE)}\n")

    # ---- Q1: does lower alpha raise realized HHI/JS at every n_clients? ----
    rows = []
    for d in diag_list:
        mean_hhi = np.mean(list(d["hhi_per_class"].values()))
        rows.append({"alpha": d["alpha"], "n_clients": d["n_clients"], "partition_seed": d["partition_seed"],
                     "mean_hhi_10class": mean_hhi, "js_mean": d["js_divergence_mean_across_clients"],
                     "js_max": d["js_divergence_max_across_clients"],
                     "client_size_cv": d["client_size_disparity"]["cv"],
                     "mean_n_holders": np.mean(list(d["n_holders_per_class"].values()))})
    diag_df = pd.DataFrame(rows)
    q1 = diag_df.groupby(["alpha", "n_clients"]).agg(
        mean_hhi=("mean_hhi_10class", "mean"), hhi_range=("mean_hhi_10class", lambda x: x.max() - x.min()),
        mean_js=("js_mean", "mean"), js_range=("js_mean", lambda x: x.max() - x.min()),
    ).round(4)
    print("=== Q1: realized HHI/JS by (alpha, n_clients) -- mean and range across 3 partition_seeds ===")
    print(q1.to_string())
    q1.to_csv(RESULTS_DIR / "analysis_q1_alpha_vs_realized_concentration.csv")

    # ---- Q2: does more clients reduce holder count per class? report n_holders AND holder_fraction ----
    q2_rows = []
    for d in diag_list:
        for cls, n_h in d["n_holders_per_class"].items():
            q2_rows.append({"alpha": d["alpha"], "n_clients": d["n_clients"],
                             "partition_seed": d["partition_seed"], "class": cls, "n_holders": n_h,
                             "holder_fraction": n_h / d["n_clients"]})
    q2_df = pd.DataFrame(q2_rows)
    q2 = q2_df.groupby(["n_clients", "alpha"])[["n_holders", "holder_fraction"]].agg(["mean", "min", "max"]).round(3)
    print("\n=== Q2: n_holders (absolute) AND holder_fraction (=n_holders/n_clients), by n_clients x alpha ===")
    print(q2.to_string())
    q2.to_csv(RESULTS_DIR / "analysis_q2_nclients_vs_holders.csv")

    # ---- build per (run, class) table with realized diagnostics joined in ----
    long_rows = []
    for _, r in df.iterrows():
        key = (r.alpha, r.n_clients, r.partition_seed)
        d = diag_by_key[key]
        for cls in ALL_CATEGORIES:
            n_h = d["n_holders_per_class"][cls]
            long_rows.append({
                "run_id": r.run_id, "algorithm": r.algorithm, "alpha": r.alpha, "n_clients": r.n_clients,
                "partition_seed": r.partition_seed,
                "partition_id": f"{r.alpha}_{r.n_clients}_{r.partition_seed}",
                "class": cls, "recall": r.recall[cls],
                "zero_recall": int(r.recall[cls] == 0.0),
                "n_holders": n_h, "holder_fraction": n_h / r.n_clients,
                "hhi": d["hhi_per_class"][cls], "gini": d["gini_per_class"][cls],
                "holder_weight_share": holder_weight_share(d, cls),
                "class_evaluable": cls in EVALUABLE,
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(RESULTS_DIR / "analysis_long_table_run_class.csv", index=False)
    # Q3/Q3b/Q4/Q5/Q5b/Q6 all key off `zero_recall`, which is 0-by-construction (not a
    # measurable event) for classes absent from validation -- exclude them here.
    long_df_all = long_df
    long_df = long_df[long_df["class_evaluable"]].copy()
    print(f"({len(long_df_all) - len(long_df)} rows excluded from zero-recall analysis below: "
          f"{sorted(NON_EVALUABLE)} x {long_df_all['run_id'].nunique()} runs)")

    # ---- Q3: EXPLORATORY POOLED Spearman associations (rows are NOT independent --
    #      10 classes share a partition, 2 algorithms share a partition -- p-values
    #      on the pooled table are descriptive only, never a significance claim) ----
    print("\n=== Q3: Exploratory POOLED Spearman associations with zero-recall indicator ===")
    print("    (rows are NOT independent observations -- classes and algorithms share partitions;")
    print("     p-values below are descriptive context only, not a significance test)")
    predictors = ["alpha", "n_clients", "n_holders", "holder_fraction", "hhi", "gini", "holder_weight_share"]
    corr_rows = []
    for p in predictors:
        rho, pval = spearmanr(long_df[p], long_df["zero_recall"])
        corr_rows.append({"predictor": p, "spearman_rho_pooled_exploratory": rho,
                           "p_value_descriptive_only": pval, "abs_rho": abs(rho)})
    corr_df = pd.DataFrame(corr_rows).sort_values("abs_rho", ascending=False)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(RESULTS_DIR / "analysis_q3_spearman_ranking.csv", index=False)

    # partition-level consistency check: mean zero-recall rate per partition (27 points, real N)
    part_agg = long_df.groupby("partition_id").agg(
        mean_zero_recall=("zero_recall", "mean"), mean_hhi=("hhi", "mean"),
        mean_gini=("gini", "mean"), mean_holder_fraction=("holder_fraction", "mean"),
        mean_holder_weight_share=("holder_weight_share", "mean"),
    ).reset_index()
    print(f"\n--- Partition-level consistency check (n={len(part_agg)} partitions, real independent units) ---")
    part_corr_rows = []
    for p in ["mean_hhi", "mean_gini", "mean_holder_fraction", "mean_holder_weight_share"]:
        rho, pval = spearmanr(part_agg[p], part_agg["mean_zero_recall"])
        part_corr_rows.append({"predictor": p, "spearman_rho_partition_level": rho, "p_value": pval})
    part_corr_df = pd.DataFrame(part_corr_rows).sort_values("spearman_rho_partition_level", key=abs, ascending=False)
    print(part_corr_df.to_string(index=False))
    part_corr_df.to_csv(RESULTS_DIR / "analysis_q3b_partition_level_spearman.csv", index=False)
    part_agg.to_csv(RESULTS_DIR / "analysis_q3b_partition_level_aggregates.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(corr_df.predictor, corr_df.spearman_rho_pooled_exploratory,
            color=["firebrick" if v < 0 else "steelblue" for v in corr_df.spearman_rho_pooled_exploratory])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman rho (exploratory pooled association) vs. zero-recall indicator")
    ax.set_title("Item 3 Q3: predictor ranking, exploratory pooled associations\n(rows not independent -- see partition-level check in text)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "item3_fig_q3_spearman_ranking.png")
    plt.close(fig)

    # ---- Q4 (adapted): if n_holders > 0, is recall_45 > 0? ----
    with_holders = long_df[long_df.n_holders > 0]
    q4 = with_holders.groupby(["algorithm"]).apply(
        lambda g: pd.Series({"n_class_runs_with_holders": len(g),
                              "n_still_zero_recall": (g.recall == 0).sum(),
                              "fraction_zero_despite_holders": (g.recall == 0).mean()})
    ).round(4)
    print("\n=== Q4 (adapted -- local-to-global gap NOT available from this data): "
          "given n_holders>0, does global recall stay zero at round 45? ===")
    print(q4.to_string())
    q4.to_csv(RESULTS_DIR / "analysis_q4_holders_vs_zero_recall.csv")

    # ---- Q5: paired FedNova vs FedAvg on the SAME partition -- but the correct
    #      unit of analysis is the PARTITION (n=27), not the class-row (n=270).
    #      "observed modest net reduction" language only -- no "real improvement"
    #      claim until partition-clustered uncertainty is shown. ----
    piv = long_df.pivot_table(index=["alpha", "n_clients", "partition_seed", "class"],
                               columns="algorithm", values="zero_recall")
    piv = piv.dropna()
    piv["partition_id"] = [f"{a}_{n}_{p}" for a, n, p, c in piv.index]
    both_zero = ((piv.fedavg_sgd == 1) & (piv.fednova_sgd == 1)).sum()
    fedavg_only_zero = ((piv.fedavg_sgd == 1) & (piv.fednova_sgd == 0)).sum()
    fednova_only_zero = ((piv.fedavg_sgd == 0) & (piv.fednova_sgd == 1)).sum()
    neither_zero = ((piv.fedavg_sgd == 0) & (piv.fednova_sgd == 0)).sum()
    print(f"\n=== Q5: paired zero-recall, FedNova-SGD vs FedAvg-SGD on IDENTICAL partitions (n={len(piv)} class-partitions, {piv.partition_id.nunique()} partitions) ===")
    print(f"  raw counts -- both zero: {both_zero}  FedAvg-only zero (candidate 'rescue'): {fedavg_only_zero}  "
          f"FedNova-only zero (candidate 'hurt'): {fednova_only_zero}  neither zero: {neither_zero}")
    print(f"  pooled zero-recall rate: FedAvg-SGD={piv.fedavg_sgd.mean():.4f}  FedNova-SGD={piv.fednova_sgd.mean():.4f}")
    piv.to_csv(RESULTS_DIR / "analysis_q5_paired_zero_recall.csv")

    # correct unit of analysis: per-partition zero-recall rate, n=27
    per_partition = piv.groupby("partition_id")[["fedavg_sgd", "fednova_sgd"]].mean()
    per_partition["diff_fedavg_minus_fednova"] = per_partition.fedavg_sgd - per_partition.fednova_sgd
    diffs = per_partition["diff_fedavg_minus_fednova"].values
    print(f"\n--- Per-partition zero-recall-rate difference (FedAvg - FedNova), n={len(diffs)} partitions ---")
    print(f"  mean={diffs.mean():.4f}  median={np.median(diffs):.4f}  "
          f"IQR=[{np.percentile(diffs,25):.4f}, {np.percentile(diffs,75):.4f}]")

    rng = np.random.default_rng(0)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(5000)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    print(f"  cluster-bootstrap (resample partitions, 5000x) 95% CI of mean diff: [{ci_lo:.4f}, {ci_hi:.4f}]")

    n_perm = 5000
    perm_means = []
    for _ in range(n_perm):
        flips = rng.choice([-1, 1], size=len(diffs))
        perm_means.append((diffs * flips).mean())
    perm_p = float(np.mean(np.abs(perm_means) >= np.abs(diffs.mean())))
    print(f"  paired sign-flip permutation test (partition-level, {n_perm}x): p={perm_p:.4f}")
    verdict = ("observed modest net reduction in zero-recall cases" if diffs.mean() > 0
               else "no net reduction observed")
    print(f"  LOCKED FRAMING: \"{verdict}\" (CI {'excludes' if ci_lo > 0 or ci_hi < 0 else 'includes'} zero)")
    per_partition.to_csv(RESULTS_DIR / "analysis_q5b_per_partition_diff.csv")

    # ---- Q6: HHI comparison restricted to FedAvg-zero cases ONLY: rescued vs both-remained-zero ----
    merge_cols = long_df[long_df.algorithm == "fedavg_sgd"][
        ["alpha", "n_clients", "partition_seed", "class", "hhi", "n_holders", "holder_fraction"]]
    q6 = piv.reset_index().merge(merge_cols, on=["alpha", "n_clients", "partition_seed", "class"])
    q6["rescued"] = (q6.fedavg_sgd == 1) & (q6.fednova_sgd == 0)
    fedavg_zero_only = q6[q6.fedavg_sgd == 1].copy()  # restrict to FedAvg-zero cases only, per review
    print(f"\n=== Q6 (corrected): within FedAvg-zero cases only (n={len(fedavg_zero_only)}), "
          f"'rescued' vs 'both remained zero' ===")
    rescued_hhi = fedavg_zero_only[fedavg_zero_only.rescued].hhi
    stuck_hhi = fedavg_zero_only[~fedavg_zero_only.rescued].hhi
    print(f"  rescued (n={len(rescued_hhi)}): mean HHI={rescued_hhi.mean():.4f}")
    print(f"  both remained zero (n={len(stuck_hhi)}): mean HHI={stuck_hhi.mean():.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = fedavg_zero_only.rescued.map({True: "green", False: "gray"})
    ax.scatter(fedavg_zero_only.hhi, fedavg_zero_only.n_holders, c=colors, alpha=0.7, s=50)
    ax.set_xlabel("HHI (class concentration)")
    ax.set_ylabel("n_holders")
    ax.set_title("Item 3 Q6 (corrected): within FedAvg-zero-recall cases only\ngreen=FedNova rescued, gray=both remained zero")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "item3_fig_q6_fednova_rescue_vs_concentration.png")
    plt.close(fig)
    q6.to_csv(RESULTS_DIR / "analysis_q6_fednova_rescue_detail.csv", index=False)

    # ---- anchor-point composite heterogeneity score, diagnostics-only ----
    print("\n=== Confirmation-phase partition_seed selection (diagnostics-only composite score, median picked) ===")
    anchor_rows = []
    for alpha, n_clients in ANCHOR_CASES:
        sub = diag_df[(diag_df.alpha == alpha) & (diag_df.n_clients == n_clients)].copy()
        for col in ["js_mean", "js_max", "mean_hhi_10class", "client_size_cv"]:
            sub[f"z_{col}"] = (sub[col] - sub[col].mean()) / sub[col].std(ddof=0) if sub[col].std(ddof=0) > 0 else 0.0
        sub["z_neg_mean_n_holders"] = -((sub.mean_n_holders - sub.mean_n_holders.mean()) / sub.mean_n_holders.std(ddof=0)) if sub.mean_n_holders.std(ddof=0) > 0 else 0.0
        sub["composite"] = sub[["z_js_mean", "z_js_max", "z_mean_hhi_10class", "z_client_size_cv", "z_neg_mean_n_holders"]].mean(axis=1)
        sub = sub.sort_values("composite").reset_index(drop=True)
        median_row = sub.iloc[len(sub) // 2]  # n=3 -> index 1 -> true median
        print(f"alpha={alpha} n_clients={n_clients}:")
        print(sub[["partition_seed", "composite"]].to_string(index=False))
        print(f"  -> SELECTED for confirmation: partition_seed={int(median_row.partition_seed)} "
              f"(composite={median_row.composite:.3f}, median of the 3)\n")
        anchor_rows.append({"alpha": alpha, "n_clients": n_clients,
                             "selected_partition_seed": int(median_row.partition_seed),
                             "composite_score": float(median_row.composite),
                             "all_composites": sub[["partition_seed", "composite"]].to_dict("records")})
    with open(RESULTS_DIR / "analysis_confirmation_partition_selection.json", "w") as f:
        json.dump(anchor_rows, f, indent=2, default=str)

    print("\nDONE. All tables in results/analysis_*.csv, figures in figures/item3_fig_*.png")
    print("held_out_test never touched.")


if __name__ == "__main__":
    main()
