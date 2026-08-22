"""Generates the six figures for the locked Item 1+2 results document."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10})

RARE_HOLDERS = {"Manipulation": [2, 13], "Replay": [1, 4, 10]}
ALGO_ORDER = ["fedavg_sgd", "fednova_sgd", "scaffold_uniform", "scaffold_weighted", "fedadam"]
ALGO_LABELS = {"fedavg_sgd": "FedAvg-SGD", "fednova_sgd": "FedNova-SGD",
               "scaffold_uniform": "SCAFFOLD-uniform", "scaffold_weighted": "SCAFFOLD-weighted",
               "fedadam": "FedAdam"}


# ---------------------------------------------------------------------------
# Figure 1: local -> post-local -> post-aggregation recall for rare classes (Item 1, FedAvg, round 45, validation)
# ---------------------------------------------------------------------------
def fig1():
    df = pd.read_csv(RESULTS_DIR / "update_conflict_mechanistic.csv")
    sub = df[(df.eval_data == "validation") & (df["round"] == 45) & (df.seed == 11)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, cls in zip(axes, ["Manipulation", "Replay"]):
        s = sub[sub["class"] == cls]
        stages = ["recall_pre_local_train", "recall_post_local_pre_agg", "recall_post_aggregation"]
        stage_labels = ["Pre-local\n(global,\nstart of round)", "Post-local\n(pre-aggregation)", "Post-\naggregation\n(global)"]
        for _, row in s.iterrows():
            vals = [row[c] for c in stages]
            ax.plot(stage_labels, vals, marker="o", label=f"client {int(row.holder_client_id)}")
        ax.set_title(f"{cls} (round 45, FedAvg, validation, seed 11)")
        ax.set_ylabel("Recall")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8)
    fig.suptitle("Item 1: recall collapses at aggregation despite genuine local learning (seed 11)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_local_to_global_recall.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: cosine alignment + holder/non-holder contribution norms (Item 1, FedAvg, round 45, validation-based classes)
# ---------------------------------------------------------------------------
def fig2():
    df = pd.read_csv(RESULTS_DIR / "update_conflict_mechanistic.csv")
    sub = df[(df.eval_data == "validation") & (df["round"] == 45) & (df.seed == 11)].drop_duplicates(subset=["class", "holder_client_id"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    labels = [f"{r['class'][:4]}.\nclient {int(r.holder_client_id)}" for _, r in sub.iterrows()]
    ax.bar(labels, sub.cosine_holder_update_vs_aggregate_update, color="firebrick")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("cosine(holder update, aggregate update)")
    ax.set_title("Holder update vs. aggregate direction")
    ax.tick_params(axis="x", labelsize=8)

    ax = axes[1]
    cls_level = sub.drop_duplicates(subset="class")
    x = np.arange(len(cls_level))
    w = 0.35
    ax.bar(x - w/2, cls_level.holder_weighted_sum_norm, w, label="holders (weighted sum)")
    ax.bar(x + w/2, cls_level.nonholder_weighted_sum_norm, w, label="non-holders (weighted sum)")
    ax.set_xticks(x); ax.set_xticklabels(cls_level["class"])
    ax.set_ylabel("FedAvg-weighted contribution norm")
    ax.set_title("Non-holders' aggregate pull dominates holders'")
    ax.legend(fontsize=8)
    fig.suptitle("Item 1: directional cancellation mechanism (round 45, FedAvg, seed 11)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cosine_and_contributions.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: heatmap of Manipulation/Replay hierarchical recall by algorithm x seed (final test)
# ---------------------------------------------------------------------------
def fig3():
    df = pd.read_csv(RESULTS_DIR / "item2_final_test_metrics_v2.csv")
    df["rec"] = df["per_class_hierarchical_recall_json"].apply(json.loads)
    df["Manip"] = df["rec"].apply(lambda d: d["Manipulation"])
    df["Replay"] = df["rec"].apply(lambda d: d["Replay"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cls in zip(axes, ["Manip", "Replay"]):
        piv = df.pivot(index="algorithm", columns="seed", values=cls).reindex(ALGO_ORDER)
        piv.index = [ALGO_LABELS[a] for a in piv.index]
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=8)
        ax.set_title(f"{'Manipulation' if cls=='Manip' else 'Replay'} hierarchical recall (test)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Item 2: rare-class recall by algorithm x seed (held-out test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_rare_class_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: pooled vs session-balanced vs hierarchical, with 95% t-CI error bars
# ---------------------------------------------------------------------------
def fig4():
    df = pd.read_csv(RESULTS_DIR / "item2_final_test_ci_summary_v2.csv")
    metrics = ["pooled_macro_f1", "session_balanced_macro_f1", "hierarchical_session_macro_recall"]
    metric_labels = ["Pooled\nmacro-F1", "Session-balanced\nmacro-F1", "Hierarchical\nsession-macro recall"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ALGO_ORDER))
    w = 0.25
    for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
        sub = df[df.metric == m].set_index("algorithm").reindex(ALGO_ORDER)
        err = np.abs(np.vstack([sub["mean"] - sub.ci_lo, sub.ci_hi - sub["mean"]]))
        err = np.nan_to_num(err)
        ax.bar(x + (i - 1) * w, sub["mean"], w, yerr=err, capsize=3, label=ml)
    ax.set_xticks(x); ax.set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Item 2: three test metrics, 95% t-CI across seeds\n(FedAvg-SGD conditional on 3/5 seeds, 40% divergence)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_three_metrics_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: performance vs communication cost and stability
# ---------------------------------------------------------------------------
def fig5():
    df = pd.read_csv(RESULTS_DIR / "item2_final_test_ci_summary_v2.csv")
    sb = df[df.metric == "session_balanced_macro_f1"].set_index("algorithm").reindex(ALGO_ORDER)
    comm = {"fedavg_sgd": 1, "fednova_sgd": 1, "scaffold_uniform": 2, "scaffold_weighted": 2, "fedadam": 1}
    cv = sb["sd"] / sb["mean"] * 100

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALGO_ORDER)))
    for algo, c in zip(ALGO_ORDER, colors):
        marker = "x" if algo == "fedavg_sgd" else "o"
        ax.scatter(comm[algo], sb.loc[algo, "mean"], s=300 / max(cv.loc[algo], 1) * 15, color=c,
                   marker=marker, edgecolor="black", zorder=3)
        label = ALGO_LABELS[algo] + (" (conditional)" if algo == "fedavg_sgd" else "")
        ax.annotate(label, (comm[algo], sb.loc[algo, "mean"]), xytext=(8, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Communication multiplier vs. FedAvg (x)")
    ax.set_ylabel("Session-balanced macro-F1 (test, mean across seeds)")
    ax.set_xticks([1, 2])
    ax.set_title("Item 2: performance vs. communication cost\n(bubble size ~ inverse CV%, i.e. bigger = more stable)")
    ax.set_xlim(0.7, 2.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_performance_vs_cost.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: convergence curves + divergence events
# ---------------------------------------------------------------------------
def fig6():
    df = pd.read_csv(RESULTS_DIR / "item2_five_seed_final_round_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ALGO_ORDER)))
    for algo, c in zip(ALGO_ORDER, colors):
        sub = df[(df.algorithm == algo) & (~df.diverged)]
        for seed in sub.seed.unique():
            s = sub[sub.seed == seed].sort_values("round")
            ax.plot(s["round"], s.val_macro_f1, color=c, alpha=0.5, lw=1)
        mean_curve = sub.groupby("round").val_macro_f1.mean()
        ax.plot(mean_curve.index, mean_curve.values, color=c, lw=2.5, label=ALGO_LABELS[algo])

    div = df[df.diverged == True]
    for _, row in div.iterrows():
        ax.scatter(row["round"], 0, color="red", marker="X", s=120, zorder=5)
    if len(div):
        ax.scatter([], [], color="red", marker="X", s=120, label="divergence event (FedAvg-SGD)")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("9-class validation macro-F1")
    ax.set_title("Item 2: five-seed convergence curves (thin=per-seed, thick=mean)\nred X = divergence (round logged, y=0 marker only)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_convergence_and_divergence.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 done")
    fig2(); print("fig2 done")
    fig3(); print("fig3 done")
    fig4(); print("fig4 done")
    fig5(); print("fig5 done")
    fig6(); print("fig6 done")
    print(f"\nAll figures saved to {FIG_DIR}")
