"""
ISOT Drone Dataset learnability audit v2 -- corrected for a real bug found in
the official CSV feature-extraction pipeline: several columns advertised as
duration statistics are actually raw Unix epoch timestamps (~1.7e9), not
deltas. Since each attack category was captured on a different calendar day,
these columns trivially leak session/category identity and produced a fake
ROC-AUC=1.0000 in v1. They are excluded here; the genuine signal is re-measured.

Confirmed contaminated (epoch-scale, >1e9, or arithmetic on epoch values):
  ts, min_duration, max_duration, sum_duration, average_duration, flow_idle_time
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

from prepare_isot import DATA_DIR, AUDIT_DIR

OUTDIR = AUDIT_DIR
RNG = 42

TIMESTAMP_LEAK_COLUMNS = ["ts", "min_duration", "max_duration", "sum_duration",
                          "average_duration", "flow_idle_time"]

csv_files = sorted(p for p in DATA_DIR.rglob("*.csv"))
frames = []
for f in csv_files:
    category = f.relative_to(DATA_DIR).parts[0]
    d = pd.read_csv(f)
    d["_category"] = category
    d["_session"] = f.stem
    d["_is_attack"] = 0 if category == "Regular" else 1
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
print(f"Total rows: {len(df)}  sessions: {df['_session'].nunique()}")

# sanity-check every remaining numeric column isn't ALSO epoch-scale before trusting it
FEATURE_COLS = [c for c in df.columns if c not in ("_category", "_session", "_is_attack") + tuple(TIMESTAMP_LEAK_COLUMNS)]
X_all = df[FEATURE_COLS].select_dtypes(include=[np.number]).fillna(0)
still_large = [c for c in X_all.columns if X_all[c].abs().median() > 1e8]
print(f"Columns dropped as timestamp-scale leakage: {TIMESTAMP_LEAK_COLUMNS}")
print(f"Remaining columns still epoch-scale (median > 1e8) -- would also need dropping: {still_large}")
X_all = X_all.drop(columns=still_large)
print(f"Final clean feature count: {X_all.shape[1]}  columns: {list(X_all.columns)}")

y_binary = df["_is_attack"].values
groups = df["_session"].values
report = {"n_rows": int(len(df)), "n_sessions": int(df["_session"].nunique()),
          "dropped_leak_columns": TIMESTAMP_LEAK_COLUMNS + still_large,
          "clean_feature_columns": list(X_all.columns)}

print("\n=== A. Naive row-level random 80/20 split (clean features) ===")
Xtr, Xte, ytr, yte = train_test_split(X_all, y_binary, test_size=0.2, random_state=RNG, stratify=y_binary)
m = HistGradientBoostingClassifier(random_state=RNG, max_depth=6, max_iter=200)
m.fit(Xtr, ytr)
prob = m.predict_proba(Xte)[:, 1]
print(f"ROC-AUC={roc_auc_score(yte,prob):.4f}  AUPRC={average_precision_score(yte,prob):.4f}  F1={f1_score(yte,(prob>=0.5).astype(int)):.4f}")
report["row_level_random_split_clean"] = {
    "roc_auc": float(roc_auc_score(yte, prob)), "auprc": float(average_precision_score(yte, prob)),
    "f1": float(f1_score(yte, (prob >= 0.5).astype(int))),
}

print("\n=== B. GroupKFold by session (clean features) -- the real leakage-safe estimate ===")
gkf = GroupKFold(n_splits=5)
fold_aucs, fold_auprcs, fold_f1s = [], [], []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_binary, groups)):
    m = HistGradientBoostingClassifier(random_state=RNG, max_depth=6, max_iter=200)
    m.fit(X_all.iloc[tr_idx], y_binary[tr_idx])
    prob = m.predict_proba(X_all.iloc[te_idx])[:, 1]
    yte_fold = y_binary[te_idx]
    auc = roc_auc_score(yte_fold, prob) if len(set(yte_fold)) > 1 else float("nan")
    auprc = average_precision_score(yte_fold, prob)
    f1 = f1_score(yte_fold, (prob >= 0.5).astype(int))
    fold_aucs.append(auc); fold_auprcs.append(auprc); fold_f1s.append(f1)
    print(f"fold {fold}: ROC-AUC={auc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  (n_test_sessions={len(set(groups[te_idx]))})")
print(f"Mean: ROC-AUC={np.nanmean(fold_aucs):.4f}  AUPRC={np.mean(fold_auprcs):.4f}  F1={np.mean(fold_f1s):.4f}")
report["group_kfold_clean"] = {
    "fold_roc_auc": [float(x) for x in fold_aucs], "fold_auprc": [float(x) for x in fold_auprcs],
    "fold_f1": [float(x) for x in fold_f1s], "mean_roc_auc": float(np.nanmean(fold_aucs)),
    "mean_auprc": float(np.mean(fold_auprcs)), "mean_f1": float(np.mean(fold_f1s)),
}

print("\n=== C. Multi-class attack-category (GroupKFold, clean features, macro-F1) ===")
le = LabelEncoder()
y_cat = le.fit_transform(df["_category"].values)
fold_macro_f1 = []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_cat, groups)):
    m = HistGradientBoostingClassifier(random_state=RNG, max_depth=6, max_iter=200)
    m.fit(X_all.iloc[tr_idx], y_cat[tr_idx])
    pred = m.predict(X_all.iloc[te_idx])
    mf1 = f1_score(y_cat[te_idx], pred, average="macro", zero_division=0)
    fold_macro_f1.append(mf1)
    print(f"fold {fold}: macro-F1={mf1:.4f}")
print(f"Mean macro-F1: {np.mean(fold_macro_f1):.4f}")
report["multiclass_group_kfold_clean"] = {"fold_scores": [float(x) for x in fold_macro_f1], "mean": float(np.mean(fold_macro_f1))}

print("\n=== D. Feature importance on clean feature set (sanity check for new giveaways) ===")
from sklearn.inspection import permutation_importance
tr_idx, te_idx = next(gkf.split(X_all, y_binary, groups))
m = HistGradientBoostingClassifier(random_state=RNG, max_depth=6, max_iter=150)
m.fit(X_all.iloc[tr_idx], y_binary[tr_idx])
sub = np.random.RandomState(0).choice(te_idx, size=min(20000, len(te_idx)), replace=False)
r = permutation_importance(m, X_all.iloc[sub], y_binary[sub], n_repeats=3, random_state=0, n_jobs=-1)
order = np.argsort(r.importances_mean)[::-1][:15]
top_features = [{"feature": X_all.columns[i], "importance_mean": float(r.importances_mean[i]),
                 "importance_std": float(r.importances_std[i])} for i in order]
for tf in top_features:
    print(f"  {tf['feature']:25s} {tf['importance_mean']:.4f} +- {tf['importance_std']:.4f}")
report["top_feature_importance_clean"] = top_features

with open(OUTDIR / "isot_learnability_audit_v2_results.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nSaved -> {OUTDIR/'isot_learnability_audit_v2_results.json'}")
