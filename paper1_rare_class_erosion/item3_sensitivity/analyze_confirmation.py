"""
Item 3 -- confirmation-phase analysis, implementing
CONFIRMATION_ANALYSIS_PLAN_FROZEN.md exactly, AS AMENDED 2026-08-22 (see the
"تصحيح ما بعد المراجعة الثانية" section appended to that file). Round 45
mechanistic diagnostics only. Never touches held_out_test. No new training.

Unit of analysis: (partition_seed, algorithm, model_seed, class).
Holder-level (own_client_rows) and validation-level rows are aggregated
into that unit BEFORE any statistic or correlation is computed.

AMENDMENT: a class is "evaluable" only if it has nonzero true support in the
validation split (checked directly from data, not hardcoded by name). A
class with a local holder but zero validation support gets recall=0/
margin=NaN by sklearn construction regardless of what the model actually
learned -- it is unconditionally miscounted as "erosion" otherwise. Such
classes are EXCLUDED from every erosion-rate denominator, the paired
FedAvg-vs-FedNova comparison, the rescue analysis, and the concentration
correlations. They are still reported (labeled non-evaluable) in Table 3's
local-only columns, since local retention is still a valid observation.
"""
import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist, ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

ANCHORS = [
    (0.1, 30, 102, "heavy_a0.1_nc30"),
    (0.3, 15, 103, "moderate_a0.3_nc15"),
    (1.0, 10, 101, "light_a1.0_nc10"),
]
ANCHOR_LABEL = {(a, n, p): lbl for a, n, p, lbl in ANCHORS}
ALGOS = ["fedavg_sgd", "fednova_sgd"]
PRACTICAL_THRESH = 0.05

rng = np.random.default_rng(12345)  # bootstrap only; no effect on any trained model

# ---------------------------------------------------------------- load ----
mech = pd.read_csv(RESULTS_DIR / "item3_confirmation_mechanistic_round45.csv")
perf = pd.read_csv(RESULTS_DIR / "item3_confirmation_round_metrics.csv")
with open(RESULTS_DIR / "item3_exploratory_partition_diagnostics.json") as f:
    diag = json.load(f)
diag_by_key = {(d["alpha"], d["n_clients"], d["partition_seed"]): d for d in diag}

# ---------------------------------------------- evaluable-class determination
# A class is evaluable iff it has nonzero TRUE support in the validation
# split. Computed from data directly (not hardcoded by name) so this stays
# correct if the dataset or split ever changes.
_data_for_support = ISOTFederatedData()
_data_for_support.fit_preprocessing()
_, _yval = _data_for_support.get_validation_data()
_val_support = np.bincount(_yval.numpy(), minlength=len(ALL_CATEGORIES))
EVALUABLE_CLASSES = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if _val_support[i] > 0}
NON_EVALUABLE_CLASSES = {ALL_CATEGORIES[i] for i in range(len(ALL_CATEGORIES)) if _val_support[i] == 0}
print(f"Validation-evaluable classes ({len(EVALUABLE_CLASSES)}/{len(ALL_CATEGORIES)}): {sorted(EVALUABLE_CLASSES)}")
print(f"NON-evaluable classes (0 validation support -- excluded from all erosion/rescue/correlation denominators): {sorted(NON_EVALUABLE_CLASSES)}")

KEYS = ["algorithm", "alpha", "n_clients", "partition_seed", "model_seed", "class"]

own = mech[mech["eval_data"] == "own_client_rows"].copy()
val = mech[mech["eval_data"] == "validation"].copy()

# ---------------------------------------------------- per-unit aggregation
def agg_own(g):
    w = g["client_fedavg_weight"].to_numpy()
    wsum = w.sum()
    return pd.Series({
        "n_holders": len(g),
        "max_local_recall": g["recall_post_local"].max(),
        "weighted_local_recall_pre": np.average(g["recall_pre_local"], weights=w) if wsum > 0 else g["recall_pre_local"].mean(),
        "weighted_local_recall_post": np.average(g["recall_post_local"], weights=w) if wsum > 0 else g["recall_post_local"].mean(),
        "max_local_margin_post": g["logit_margin_post_local"].max(),
        "holder_weight_share_this_class": g["holder_weight_share_this_class"].iloc[0],
    })

own_agg = own.groupby(KEYS, as_index=False).apply(agg_own, include_groups=False)

val_agg = val.groupby(KEYS, as_index=False).first()[
    KEYS + ["recall_post_agg", "logit_margin_post_agg", "recall_pre_local", "logit_margin_pre_local"]
].rename(columns={
    "recall_post_agg": "global_recall",
    "logit_margin_post_agg": "global_margin_post_agg",
    "recall_pre_local": "global_recall_pre",
    "logit_margin_pre_local": "global_margin_pre",
})

unit = own_agg.merge(val_agg, on=KEYS, how="inner")
assert len(unit) == len(own_agg) == len(val_agg), "own/validation unit mismatch -- investigate before proceeding"

unit["class_evaluable"] = unit["class"].isin(EVALUABLE_CLASSES)
unit["local_learning_gain"] = unit["weighted_local_recall_post"] - unit["weighted_local_recall_pre"]
# aggregation_recall_loss / margin_change are only meaningful against a real
# global ground truth; NaN them for non-evaluable classes rather than
# computing a number against a definitionally-zero recall.
unit["aggregation_recall_loss"] = np.where(unit["class_evaluable"],
                                            unit["max_local_recall"] - unit["global_recall"], np.nan)
unit["aggregation_margin_change"] = np.where(unit["class_evaluable"],
                                              unit["global_margin_post_agg"] - unit["max_local_margin_post"], np.nan)
unit["strict_erosion"] = np.where(unit["class_evaluable"],
                                   ((unit["max_local_recall"] > 0) & (unit["global_recall"] == 0)).astype(float), np.nan)
unit["practical_erosion"] = np.where(unit["class_evaluable"],
                                      ((unit["max_local_recall"] >= PRACTICAL_THRESH) & (unit["global_recall"] == 0)).astype(float), np.nan)
unit["holder_fraction"] = unit["n_holders"] / unit["n_clients"]
unit["anchor"] = unit.apply(lambda r: ANCHOR_LABEL[(r["alpha"], r["n_clients"], r["partition_seed"])], axis=1)

def lookup_diag(row, field):
    d = diag_by_key[(row["alpha"], row["n_clients"], row["partition_seed"])]
    return d[field][row["class"]]

unit["gini"] = unit.apply(lambda r: lookup_diag(r, "gini_per_class"), axis=1)
unit["hhi"] = unit.apply(lambda r: lookup_diag(r, "hhi_per_class"), axis=1)

unit.to_csv(RESULTS_DIR / "confirmation_unit_table_run_class.csv", index=False)
print(f"unit table: {len(unit)} rows (expect <=290; 29 runs x 10 classes)")

# ============================================================ TABLE 1 =====
# performance by anchor x algorithm x seed (val_macro_f1 @ round 45)
r45 = perf[(perf["round"] == 45) & (perf["diverged"] == False)].copy()
r45["anchor"] = r45.apply(lambda r: ANCHOR_LABEL[(r["alpha"], r["n_clients"], r["partition_seed"])], axis=1)
t1_rows = []
for anchor in [a[3] for a in ANCHORS]:
    for algo in ALGOS:
        sub = r45[(r45["anchor"] == anchor) & (r45["algorithm"] == algo)].sort_values("model_seed")
        vals = sub["val_macro_f1"].to_numpy()
        n = len(vals)
        mean, sd = vals.mean(), vals.std(ddof=1) if n > 1 else 0.0
        if n > 1:
            ci = t_dist.ppf(0.975, n - 1) * sd / np.sqrt(n)
        else:
            ci = float("nan")
        for _, row in sub.iterrows():
            t1_rows.append({"anchor": anchor, "algorithm": algo, "model_seed": row["model_seed"],
                             "val_macro_f1": row["val_macro_f1"], "n_seeds_in_cell": n,
                             "cell_mean_macro_f1": mean, "cell_sd_macro_f1": sd, "cell_95pct_tCI_halfwidth": ci})
table1 = pd.DataFrame(t1_rows)
table1.to_csv(RESULTS_DIR / "confirmation_table1_performance_by_anchor_algo_seed.csv", index=False)

# ============================================================ TABLE 2 =====
# erosion rates: strict + practical, seed-first, per anchor x algorithm
t2_rows = []
for anchor in [a[3] for a in ANCHORS]:
    for algo in ALGOS:
        sub = unit[(unit["anchor"] == anchor) & (unit["algorithm"] == algo)]
        seeds = sorted(sub["model_seed"].unique())
        strict_rates, practical_rates = [], []
        for s in seeds:
            ss = sub[(sub["model_seed"] == s) & sub["class_evaluable"]]
            n_eval = len(ss)  # evaluable classes only (nonzero validation support)
            strict_rates.append(ss["strict_erosion"].sum() / n_eval)
            practical_rates.append(ss["practical_erosion"].sum() / n_eval)
        strict_rates, practical_rates = np.array(strict_rates), np.array(practical_rates)
        sub_eval = sub[sub["class_evaluable"]]
        k_strict, n_strict = int(sub_eval["strict_erosion"].sum()), len(sub_eval)
        k_prac, n_prac = int(sub_eval["practical_erosion"].sum()), len(sub_eval)
        t2_rows.append({
            "anchor": anchor, "algorithm": algo, "n_seeds": len(seeds),
            "n_evaluable_classes_per_seed": n_eval,
            "excluded_classes": ", ".join(sorted(NON_EVALUABLE_CLASSES)) if NON_EVALUABLE_CLASSES else "",
            "strict_erosion_k_of_N_pooled": f"{k_strict}/{n_strict}",
            "strict_erosion_rate_pooled": k_strict / n_strict,
            "strict_erosion_rate_seed_mean": strict_rates.mean(),
            "strict_erosion_rate_seed_sd": strict_rates.std(ddof=1) if len(seeds) > 1 else 0.0,
            "practical_erosion_k_of_N_pooled": f"{k_prac}/{n_prac}",
            "practical_erosion_rate_pooled": k_prac / n_prac,
            "practical_erosion_rate_seed_mean": practical_rates.mean(),
            "practical_erosion_rate_seed_sd": practical_rates.std(ddof=1) if len(seeds) > 1 else 0.0,
            "conditional_on_n_runs": len(seeds),
        })
table2 = pd.DataFrame(t2_rows)
table2.to_csv(RESULTS_DIR / "confirmation_table2_erosion_rates_strict_practical.csv", index=False)

# ============================================================ TABLE 3 =====
# recall/margin pipeline pre-local -> post-local -> post-agg, by anchor x algo x class
t3 = unit.groupby(["anchor", "algorithm", "class"], as_index=False).agg(
    n_seeds=("model_seed", "nunique"),
    class_evaluable=("class_evaluable", "first"),
    mean_recall_pre_local=("weighted_local_recall_pre", "mean"),
    mean_recall_post_local=("weighted_local_recall_post", "mean"),
    mean_max_local_recall=("max_local_recall", "mean"),
    mean_global_recall=("global_recall", "mean"),
    mean_local_learning_gain=("local_learning_gain", "mean"),
    mean_aggregation_recall_loss=("aggregation_recall_loss", "mean"),
    median_aggregation_recall_loss=("aggregation_recall_loss", "median"),
    mean_aggregation_margin_change=("aggregation_margin_change", "mean"),
    median_aggregation_margin_change=("aggregation_margin_change", "median"),
)
table3 = t3
with open(RESULTS_DIR / "confirmation_table3_recall_margin_pipeline.csv", "w") as f:
    f.write("# mean_recall_pre_local / mean_recall_post_local / mean_max_local_recall are LOCAL "
            "FITTING/RETENTION EVIDENCE on the holder's own training rows (own_client_rows), not "
            "independent local-generalization evidence -- they are evaluated on the same rows the "
            "client just trained on this round.\n")
    f.write("# mean_global_recall and everything downstream of it (aggregation_recall_loss, "
            "aggregation_margin_change) are NaN where class_evaluable=False: that class has zero "
            "true support in the validation split, so global recall is 0/NaN by sklearn construction "
            "regardless of what was learned locally, and is not a measurable erosion event.\n")
    table3.to_csv(f, index=False)

# ============================================================ TABLE 4 =====
# paired FedAvg vs FedNova on common (anchor, model_seed) -- seed-level erosion rate as the paired unit
unit_eval = unit[unit["class_evaluable"]].copy()  # excludes classes with 0 validation support

t4_rows = []
for anchor in [a[3] for a in ANCHORS]:
    fa = unit_eval[(unit_eval["anchor"] == anchor) & (unit_eval["algorithm"] == "fedavg_sgd")]
    fn = unit_eval[(unit_eval["anchor"] == anchor) & (unit_eval["algorithm"] == "fednova_sgd")]
    common_seeds = sorted(set(fa["model_seed"]) & set(fn["model_seed"]))
    for s in common_seeds:
        fa_s, fn_s = fa[fa["model_seed"] == s], fn[fn["model_seed"] == s]
        t4_rows.append({
            "anchor": anchor, "model_seed": s,
            "fedavg_strict_erosion_rate": fa_s["strict_erosion"].mean(),
            "fednova_strict_erosion_rate": fn_s["strict_erosion"].mean(),
            "fedavg_practical_erosion_rate": fa_s["practical_erosion"].mean(),
            "fednova_practical_erosion_rate": fn_s["practical_erosion"].mean(),
            "fedavg_mean_recall_loss": fa_s["aggregation_recall_loss"].mean(),
            "fednova_mean_recall_loss": fn_s["aggregation_recall_loss"].mean(),
        })
table4_pairs = pd.DataFrame(t4_rows)

t4_summary = []
for anchor in [a[3] for a in ANCHORS] + ["ALL_ANCHORS_POOLED"]:
    sub = table4_pairs if anchor == "ALL_ANCHORS_POOLED" else table4_pairs[table4_pairs["anchor"] == anchor]
    n = len(sub)
    if n >= 2:
        d = sub["fedavg_strict_erosion_rate"].to_numpy() - sub["fednova_strict_erosion_rate"].to_numpy()
        try:
            w_stat, w_p = wilcoxon(d) if np.any(d != 0) else (float("nan"), 1.0)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p = ttest_rel(sub["fedavg_strict_erosion_rate"], sub["fednova_strict_erosion_rate"])
    else:
        d, w_stat, w_p, t_stat, t_p = np.array([]), float("nan"), float("nan"), float("nan"), float("nan")
    note = ""
    if anchor == "light_a1.0_nc10":
        note = "light anchor uses common seeds only (33 excluded from FedAvg side: diverged; FedNova-33 excluded from this paired comparison)"
    elif anchor == "ALL_ANCHORS_POOLED":
        note = ("CAUTION: these 14 rows are NOT independent -- the same 5 model_seeds repeat across "
                "up to 3 anchors each. Do not report this p-value as evidence of statistical "
                "significance. See ALL_ANCHORS_SEED_AGGREGATED below for the analysis that respects "
                "seed as the true independent unit.")
    t4_summary.append({
        "anchor": anchor, "n_paired_seeds": n,
        "mean_diff_strict_erosion_rate_fedavg_minus_fednova": d.mean() if n else float("nan"),
        "paired_t_stat": t_stat, "paired_t_pvalue": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
        "note": note,
    })

# per-seed diff averaged across the anchors that seed appears in -- kept for display,
# but NOT all seeds average over the same number of anchors (seed 33 is missing from the
# light anchor on the FedAvg side, since that run diverged), so these 5 rows are not the
# same underlying quantity and must not all be pooled into one "n=5" test.
seed_agg = table4_pairs.copy()
seed_agg["diff"] = seed_agg["fedavg_strict_erosion_rate"] - seed_agg["fednova_strict_erosion_rate"]
seed_level = seed_agg.groupby("model_seed", as_index=False).agg(
    n_anchors_averaged=("anchor", "nunique"), mean_diff=("diff", "mean"))

# unified full-case test: restricted to seeds present in ALL three anchors (here: 11,22,44,55 --
# seed 33 excluded because FedAvg diverged in the light anchor, so it only has 2/3 anchors and is
# not a like-for-like average with the rest). This is the correct "all anchors combined" unit.
full_case = seed_level[seed_level["n_anchors_averaged"] == len(ANCHORS)]
excluded_partial_seeds = sorted(set(seed_level["model_seed"]) - set(full_case["model_seed"]))
n_sa = len(full_case)
if n_sa >= 2:
    d_sa = full_case["mean_diff"].to_numpy()
    try:
        w_stat_sa, w_p_sa = wilcoxon(d_sa) if np.any(d_sa != 0) else (float("nan"), 1.0)
    except ValueError:
        w_stat_sa, w_p_sa = float("nan"), float("nan")
    full_case_seeds = set(full_case["model_seed"])
    fc_pairs = seed_agg[seed_agg["model_seed"].isin(full_case_seeds)]
    t_stat_sa, t_p_sa = ttest_rel(fc_pairs.groupby("model_seed")["fedavg_strict_erosion_rate"].mean(),
                                   fc_pairs.groupby("model_seed")["fednova_strict_erosion_rate"].mean())
else:
    w_stat_sa, w_p_sa, t_stat_sa, t_p_sa = float("nan"), float("nan"), float("nan"), float("nan")
n_same_direction = int((d_sa < 0).sum()) if n_sa else 0
t4_summary.append({
    "anchor": "ALL_ANCHORS_SEED_AGGREGATED", "n_paired_seeds": n_sa,
    "mean_diff_strict_erosion_rate_fedavg_minus_fednova": d_sa.mean() if n_sa else float("nan"),
    "paired_t_stat": t_stat_sa, "paired_t_pvalue": t_p_sa,
    "wilcoxon_stat": w_stat_sa, "wilcoxon_pvalue": w_p_sa,
    "note": (f"Unified full-case unit: one row per model_seed present in all {len(ANCHORS)} anchors "
             f"(n={n_sa}: seeds {sorted(full_case['model_seed'].tolist())}). Seed(s) "
             f"{excluded_partial_seeds} excluded from this row -- present in only 2/{len(ANCHORS)} "
             f"anchors because the matching FedAvg run diverged (light anchor), so not the same "
             f"averaged quantity; shown separately in the per-seed table above. "
             f"{n_same_direction}/{n_sa} full-case seeds show fedavg<fednova; descriptive only -- at "
             f"n={n_sa} the exact two-sided Wilcoxon floor is 2/2^n -- do not call this "
             f"'statistically significant'."),
})
table4 = pd.DataFrame(t4_summary)
with open(RESULTS_DIR / "confirmation_table4_paired_fedavg_vs_fednova.csv", "w") as f:
    f.write("# per-seed paired values (evaluable classes only -- classes with 0 validation support excluded)\n")
    table4_pairs.to_csv(f, index=False)
    f.write("\n# per-seed diff averaged across the anchors that seed appears in (display only -- "
            "n_anchors_averaged differs for seed 33, see note on ALL_ANCHORS_SEED_AGGREGATED below "
            "for the correct like-for-like unified test)\n")
    seed_level.to_csv(f, index=False)
    f.write("\n# summary (seed-level paired test, seed = statistical unit)\n")
    table4.to_csv(f, index=False)

# ============================================================ TABLE 5 =====
# erosion / recall-loss vs concentration correlates (class-run level, exploratory-style Spearman)
concentration_vars = ["gini", "hhi", "holder_fraction", "holder_weight_share_this_class"]
target_vars = ["strict_erosion", "practical_erosion", "aggregation_recall_loss"]
corr_rows = []
for target in target_vars:
    for cvar in concentration_vars:
        rho, p = spearmanr(unit_eval[cvar], unit_eval[target])
        corr_rows.append({"target": target, "concentration_var": cvar, "spearman_rho": rho,
                           "naive_unadjusted_p_do_not_report": p, "n": len(unit_eval)})
table5_corr = pd.DataFrame(corr_rows)

# partition-level aggregate (mean per partition; descriptive only -- 3 confirmation
# partitions is FAR too few to support a "robustness check" label. Concentration
# conclusions should be drawn from the exploratory grid (27 partitions), not this table.
part_agg = unit_eval.groupby(["anchor", "partition_seed"], as_index=False).agg(
    mean_gini=("gini", "mean"), mean_hhi=("hhi", "mean"),
    mean_holder_fraction=("holder_fraction", "mean"),
    strict_erosion_rate=("strict_erosion", "mean"),
    practical_erosion_rate=("practical_erosion", "mean"),
)

# FedNova-rescue vs HHI, restricted to FedAvg-zero cases, cluster-bootstrap by partition_id
# (evaluable classes only -- a non-evaluable class can never be "rescued", its global recall
# is 0 by construction regardless of algorithm)
fa_zero = unit_eval[(unit_eval["algorithm"] == "fedavg_sgd") & (unit_eval["global_recall"] == 0)][
    ["anchor", "partition_seed", "model_seed", "class", "hhi"]
]
fn_lookup = unit_eval[unit_eval["algorithm"] == "fednova_sgd"].set_index(["partition_seed", "model_seed", "class"])["global_recall"]
fa_zero = fa_zero.copy()
fa_zero["fednova_rescued"] = fa_zero.apply(
    lambda r: fn_lookup.get((r["partition_seed"], r["model_seed"], r["class"]), np.nan) > 0
    if (r["partition_seed"], r["model_seed"], r["class"]) in fn_lookup.index else np.nan, axis=1)
fa_zero = fa_zero.dropna(subset=["fednova_rescued"])

rescued_hhi = fa_zero[fa_zero["fednova_rescued"]]["hhi"]
stuck_hhi = fa_zero[~fa_zero["fednova_rescued"]]["hhi"]
observed_gap = rescued_hhi.mean() - stuck_hhi.mean() if len(rescued_hhi) and len(stuck_hhi) else float("nan")

partition_ids = fa_zero["partition_seed"].unique()
boot_gaps = []
if len(partition_ids) >= 2 and len(rescued_hhi) and len(stuck_hhi):
    for _ in range(5000):
        sampled_parts = rng.choice(partition_ids, size=len(partition_ids), replace=True)
        boot_df = pd.concat([fa_zero[fa_zero["partition_seed"] == p] for p in sampled_parts], ignore_index=True)
        r_hhi = boot_df[boot_df["fednova_rescued"]]["hhi"]
        s_hhi = boot_df[~boot_df["fednova_rescued"]]["hhi"]
        if len(r_hhi) and len(s_hhi):
            boot_gaps.append(r_hhi.mean() - s_hhi.mean())
boot_gaps = np.array(boot_gaps)
if len(boot_gaps) > 0:
    ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
else:
    ci_lo, ci_hi = float("nan"), float("nan")

n_rescued = int(fa_zero["fednova_rescued"].sum())
if n_rescued == 0:
    confirmation_wording = ("Within the three confirmation anchors (evaluable classes only), FedNova rescued "
                             "ZERO of the 12 FedAvg zero-global-recall cases (0/12). The exploratory grid's "
                             "modest net reduction in zero-recall cases did NOT transfer to these three "
                             "prespecified anchors -- the exploratory HHI/rescue pattern is not reproduced here. "
                             "The exploratory-phase wording ('FedNova rescue was descriptively concentrated in "
                             "moderately concentrated class allocations...') describes the EXPLORATORY grid only "
                             "and must not be applied to the confirmation anchors.")
else:
    confirmation_wording = ("FedNova rescue was descriptively concentrated in moderately concentrated class "
                             "allocations, while highly concentrated cases were rarely rescued.")

rescue_summary = pd.DataFrame([{
    "n_fedavg_zero_cases_with_fednova_match": len(fa_zero),
    "n_rescued": n_rescued,
    "n_stuck_both_zero": int((~fa_zero["fednova_rescued"]).sum()),
    "mean_hhi_rescued": rescued_hhi.mean() if len(rescued_hhi) else float("nan"),
    "mean_hhi_stuck": stuck_hhi.mean() if len(stuck_hhi) else float("nan"),
    "observed_gap_rescued_minus_stuck": observed_gap,
    "cluster_bootstrap_by_partition_n_resamples": len(boot_gaps),
    "cluster_bootstrap_95pct_CI_lo": ci_lo,
    "cluster_bootstrap_95pct_CI_hi": ci_hi,
    "approved_wording_confirmation_scope": confirmation_wording,
    "caveat": "Gini/HHI/holder_fraction/holder_weight_share are NOT causes; strongest stable exploratory associations only.",
}])

with open(RESULTS_DIR / "confirmation_table5_erosion_vs_concentration.csv", "w") as f:
    f.write("# class-run-level Spearman rho DESCRIPTIVE ONLY (n=%d evaluable units -- %s excluded for 0 "
            "validation support -- rows correlated within only 3 partitions). The p-value column is "
            "naive/unadjusted for that clustering and MUST NOT be reported or cited as a significance "
            "result -- it is retained only so the raw naive value is visible, not to license its use.\n"
            % (len(unit_eval), ", ".join(sorted(NON_EVALUABLE_CLASSES)) if NON_EVALUABLE_CLASSES else "none"))
    table5_corr.to_csv(f, index=False)
    f.write("\n# partition-level aggregate (mean per partition). NOT a robustness check -- only 3 "
            "confirmation partitions exist, far too few to assess robustness. Concentration/HHI/Gini "
            "conclusions should be drawn from the exploratory grid (27 partitions, see "
            "analysis_q3b_partition_level_spearman.csv), not from this table.\n")
    part_agg.to_csv(f, index=False)
    f.write("\n# FedNova-rescue vs HHI, restricted to FedAvg-zero cases, cluster-bootstrap by partition_id\n")
    rescue_summary.to_csv(f, index=False)

# ============================================================ TABLE 6 =====
# divergence report, kept separate from all performance tables
planned = []
for a, n, p, lbl in ANCHORS:
    for algo in ALGOS:
        planned.append({"anchor": lbl, "algorithm": algo, "planned_seeds": 5})
planned = pd.DataFrame(planned)
completed_counts = unit.groupby(["anchor", "algorithm"])["model_seed"].nunique().reset_index(name="completed_seeds")
div = planned.merge(completed_counts, on=["anchor", "algorithm"], how="left").fillna({"completed_seeds": 0})
div["completed_seeds"] = div["completed_seeds"].astype(int)
div["diverged_seeds"] = div["planned_seeds"] - div["completed_seeds"]
div["divergence_rate"] = div["diverged_seeds"] / div["planned_seeds"]

overall = pd.DataFrame([
    {"scope": "FedAvg-SGD, all anchors pooled", "diverged": int(div[div.algorithm == "fedavg_sgd"]["diverged_seeds"].sum()),
     "planned": int(div[div.algorithm == "fedavg_sgd"]["planned_seeds"].sum())},
    {"scope": "FedNova-SGD, all anchors pooled", "diverged": int(div[div.algorithm == "fednova_sgd"]["diverged_seeds"].sum()),
     "planned": int(div[div.algorithm == "fednova_sgd"]["planned_seeds"].sum())},
    {"scope": "FedAvg-SGD, light anchor (a1.0/nc10) only", "diverged": int(div[(div.algorithm == "fedavg_sgd") & (div.anchor == "light_a1.0_nc10")]["diverged_seeds"].sum()),
     "planned": int(div[(div.algorithm == "fedavg_sgd") & (div.anchor == "light_a1.0_nc10")]["planned_seeds"].sum())},
])
overall["rate"] = overall["diverged"] / overall["planned"]

diverged_detail = pd.DataFrame([{
    "algorithm": "fedavg_sgd", "alpha": 1.0, "n_clients": 10, "partition_seed": 101,
    "model_seed": 33, "diverged_at_round": 16,
    "note": "excluded from every performance/erosion denominator; no round-45 diagnostics exist for this run",
}])

with open(RESULTS_DIR / "confirmation_table6_divergence_report.csv", "w") as f:
    f.write("# per anchor x algorithm\n")
    div.to_csv(f, index=False)
    f.write("\n# overall / light-anchor-specific\n")
    overall.to_csv(f, index=False)
    f.write("\n# diverged run detail\n")
    diverged_detail.to_csv(f, index=False)

# ============================================================ FIGURE 1 ====
# heatmap: classes x anchors, mean global_recall (averaged over algo+seed) -- separately per algorithm
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
anchor_order = [a[3] for a in ANCHORS]
class_order = sorted(unit_eval["class"].unique())  # non-evaluable classes excluded (global_recall=0 by construction, not a finding)
for ax, algo in zip(axes, ALGOS):
    piv = unit_eval[unit_eval["algorithm"] == algo].pivot_table(
        index="class", columns="anchor", values="global_recall", aggfunc="mean"
    ).reindex(index=class_order, columns=anchor_order)
    im = ax.imshow(piv.values, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(anchor_order))); ax.set_xticklabels(anchor_order, rotation=30, ha="right")
    ax.set_yticks(range(len(class_order))); ax.set_yticklabels(class_order)
    ax.set_title(algo)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.4 else "black")
fig.colorbar(im, ax=axes, label="mean global_recall (round 45, post-aggregation)")
excl_note = f" (excludes {', '.join(sorted(NON_EVALUABLE_CLASSES))}: 0 validation support)" if NON_EVALUABLE_CLASSES else ""
fig.suptitle(f"Item 3 confirmation -- global recall by class x anchor (mean over available seeds){excl_note}")
fig.savefig(FIG_DIR / "confirmation_fig_heatmap_class_x_anchor.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================ FIGURE 2 ====
# local-to-global path: pre-local -> post-local (weighted) -> post-agg, per anchor, per algorithm, classes as lines
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
stage_x = [0, 1, 2]
stage_labels = ["pre-local", "post-local", "post-agg"]
for ax, (a, n, p, lbl) in zip(axes, ANCHORS):
    for algo, ls in zip(ALGOS, ["-", "--"]):
        sub = unit[(unit["anchor"] == lbl) & (unit["algorithm"] == algo)]
        cls_mean = sub.groupby("class")[["weighted_local_recall_pre", "weighted_local_recall_post", "global_recall"]].mean()
        for cls in class_order:
            if cls in cls_mean.index:
                y = cls_mean.loc[cls, ["weighted_local_recall_pre", "weighted_local_recall_post", "global_recall"]].to_numpy()
                ax.plot(stage_x, y, ls, alpha=0.6, linewidth=1.2,
                        label=f"{algo}" if cls == class_order[0] else None)
    ax.set_xticks(stage_x); ax.set_xticklabels(stage_labels, rotation=15)
    ax.set_title(lbl); ax.set_ylim(-0.02, 1.02)
axes[0].set_ylabel("recall (weighted-local pre/post, global post-agg)")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Item 3 confirmation -- local-to-global recall path per class (solid=FedAvg, dashed=FedNova)")
fig.savefig(FIG_DIR / "confirmation_fig_local_to_global_path.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================ seed-11 check
c11 = perf[(perf["model_seed"] == 11) & (perf["round"] == 45)][["algorithm", "alpha", "n_clients", "partition_seed", "val_macro_f1"]]
expl = pd.read_csv(RESULTS_DIR / "item3_exploratory_round_metrics.csv")
e11 = expl[(expl["model_seed"] == 11) & (expl["round"] == 45)][["algorithm", "alpha", "n_clients", "partition_seed", "val_macro_f1"]]
seed11_check = c11.merge(e11, on=["algorithm", "alpha", "n_clients", "partition_seed"], suffixes=("_confirmation", "_exploratory"))
seed11_check["abs_diff"] = (seed11_check["val_macro_f1_confirmation"] - seed11_check["val_macro_f1_exploratory"]).abs()
seed11_check.to_csv(RESULTS_DIR / "confirmation_seed11_determinism_check.csv", index=False)

print("\n=== SEED-11 DETERMINISM CHECK ===")
print(seed11_check.to_string(index=False))
print(f"\nall match exactly: {bool((seed11_check['abs_diff'] == 0).all())}")

print("\n=== TABLE 2 (erosion rates) ===")
print(table2.to_string(index=False))

print("\n=== TABLE 6 (divergence, per anchor x algo) ===")
print(div.to_string(index=False))
print(overall.to_string(index=False))

print("\nDone. All 8 required outputs + 1 intermediate unit table + 1 determinism check written to results/ and figures/.")
