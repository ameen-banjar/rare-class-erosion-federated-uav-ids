"""
Prepares (but does NOT run) a federated client split for ISOT, v2.

Adds, per review feedback:
  - A VALIDATION session pool, carved out of the training pool (never the
    held-out test sessions, never the sole Password Cracking training
    session), for hyperparameter/threshold/round-count selection. The 28
    held-out test sessions stay untouched by anything except the final report.
  - Row-level (not just session-count) category breakdown per client:
    row counts per category, Regular row share, Shannon label entropy, and
    Jensen-Shannon divergence from the global category row-distribution --
    because sessions vary from 177 to 205,370 rows, session-count Non-IID and
    row-count Non-IID are not the same thing.
  - Terminology: this is "controlled session-level Non-IID partitioning over
    real UAV capture sessions using Dirichlet allocation", not an
    unqualified claim of "real" Non-IID.
"""
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from prepare_isot import load_all_sessions, session_table, AUDIT_DIR

RNG = 42
N_CLIENTS = 15
TEST_HOLDOUT_FRACTION = 0.2
VALIDATION_FRACTION_OF_POOL = 0.15
DIRICHLET_ALPHA = 0.5
ALL_CATEGORIES = ["DoS", "Injection", "Ip Spoofing", "MITM", "Manipulation",
                   "Password Cracking", "Regular", "Replay", "Unauth", "Video"]


def category_row_distribution(sessions, n_rows_lookup, cat_lookup):
    counts = {c: 0 for c in ALL_CATEGORIES}
    for s in sessions:
        counts[cat_lookup[s]] += n_rows_lookup[s]
    total = sum(counts.values())
    dist = {c: counts[c] / total if total else 0.0 for c in ALL_CATEGORIES}
    return counts, dist


def main():
    rng = np.random.RandomState(RNG)
    df = load_all_sessions()
    st = session_table(df)
    n_rows_lookup = dict(zip(st["_session"], st["n_rows"]))
    cat_lookup = dict(zip(st["_session"], st["_category"]))

    global_counts, global_dist = category_row_distribution(st["_session"].tolist(), n_rows_lookup, cat_lookup)
    global_dist_vec = np.array([global_dist[c] for c in ALL_CATEGORIES])

    # ---- 1. held-out test sessions, stratified by category ----
    held_out, pool = [], []
    for cat, g in st.groupby("_category"):
        sessions = list(g["_session"])
        rng.shuffle(sessions)
        n_hold = max(1, round(len(sessions) * TEST_HOLDOUT_FRACTION)) if len(sessions) >= 2 else 0
        held_out += sessions[:n_hold]
        pool += sessions[n_hold:]
    print(f"Held-out TEST sessions: {len(held_out)} / {len(st)}")

    # ---- 2. validation sessions carved from the pool, never touching the sole
    #         Password-Cracking training session or the held-out test set ----
    pool_st = st[st["_session"].isin(pool)]
    validation, remaining_pool = [], []
    for cat, g in pool_st.groupby("_category"):
        sessions = list(g["_session"])
        rng.shuffle(sessions)
        if len(sessions) <= 1:
            remaining_pool += sessions  # never move a category's only training session to validation
            continue
        n_val = max(1, round(len(sessions) * VALIDATION_FRACTION_OF_POOL))
        n_val = min(n_val, len(sessions) - 1)  # always leave >=1 session for client training
        validation += sessions[:n_val]
        remaining_pool += sessions[n_val:]
    print(f"VALIDATION sessions (carved from pool, for hyperparameter/threshold/round selection only): {len(validation)}")
    print(f"Remaining pool for CLIENT training: {len(remaining_pool)}\n")

    # ---- 3. assign remaining_pool sessions to clients ----
    remaining_st = pool_st[pool_st["_session"].isin(remaining_pool)]
    client_sessions = {i: [] for i in range(N_CLIENTS)}
    regular_pool = remaining_st[remaining_st["_category"] == "Regular"]["_session"].tolist()
    rng.shuffle(regular_pool)
    for i in range(N_CLIENTS):
        if regular_pool:
            client_sessions[i].append(regular_pool.pop())
    leftover_regular = regular_pool

    attack_cats = [c for c in remaining_st["_category"].unique() if c != "Regular"]
    for cat in attack_cats:
        cat_sessions = remaining_st[remaining_st["_category"] == cat]["_session"].tolist()
        rng.shuffle(cat_sessions)
        weights = rng.dirichlet([DIRICHLET_ALPHA] * N_CLIENTS)
        for s in cat_sessions:
            c = rng.choice(N_CLIENTS, p=weights)
            client_sessions[c].append(s)
    for s in leftover_regular:
        c = rng.choice(N_CLIENTS)
        client_sessions[c].append(s)

    # ---- 4. summarize with row-level stats, entropy, JS-divergence ----
    manifest = {
        "partition_description": "Controlled session-level Non-IID partitioning over real UAV capture sessions using Dirichlet allocation (not an unqualified claim of naturally-occurring Non-IID).",
        "held_out_test_sessions": held_out,
        "validation_sessions": validation,
        "validation_note": "For hyperparameter/round-count/threshold selection ONLY. Never used to pick D2Guard's server-side decision parameters as if it were a trusted root dataset, and never touches held_out_test_sessions.",
        "n_clients": N_CLIENTS, "dirichlet_alpha": DIRICHLET_ALPHA, "seed": RNG,
        "global_category_row_distribution": global_dist, "clients": {},
    }

    total_rows_assigned = sum(n_rows_lookup[s] for sessions in client_sessions.values() for s in sessions)
    for c, sessions in client_sessions.items():
        n_rows = sum(n_rows_lookup[s] for s in sessions)
        cat_row_counts, cat_row_dist = category_row_distribution(sessions, n_rows_lookup, cat_lookup)
        dist_vec = np.array([cat_row_dist[cc] for cc in ALL_CATEGORIES])
        nonzero = dist_vec[dist_vec > 0]
        label_entropy = float(-(nonzero * np.log2(nonzero)).sum()) if len(nonzero) else 0.0
        js_div = float(jensenshannon(dist_vec, global_dist_vec, base=2) ** 2)  # JS divergence (squared distance = divergence)

        session_cat_counts = pd.Series([cat_lookup[s] for s in sessions]).value_counts().to_dict()
        manifest["clients"][c] = {
            "sessions": sessions, "n_sessions": len(sessions), "n_rows": int(n_rows),
            "session_counts_by_category": session_cat_counts,
            "row_counts_by_category": cat_row_counts,
            "regular_row_share": cat_row_dist["Regular"],
            "label_entropy_bits": label_entropy,
            "js_divergence_from_global": js_div,
            "fedavg_weight_by_rows": n_rows / total_rows_assigned,
            "uniform_weight": 1.0 / N_CLIENTS,
        }

    print(f"{'client':6s} {'sessions':9s} {'rows':10s} {'row-wt':8s} {'Regular%':9s} {'entropy':8s} {'JS-div':7s}")
    for c, info in manifest["clients"].items():
        print(f"{c:6d} {info['n_sessions']:9d} {info['n_rows']:10d} {info['fedavg_weight_by_rows']:8.4f} "
              f"{info['regular_row_share']*100:8.2f}% {info['label_entropy_bits']:8.3f} {info['js_divergence_from_global']:7.4f}")

    with open(AUDIT_DIR / "federated_clients_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nSaved -> {AUDIT_DIR/'federated_clients_manifest.json'}  (split + validation prepared only -- no FL training run)")


if __name__ == "__main__":
    main()
