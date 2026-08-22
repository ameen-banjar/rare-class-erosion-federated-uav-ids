"""
Item 3 partition builder. Validation (18 sessions) and held-out test (28
sessions) are read directly from the FROZEN
`shared/data_prep/federated_clients_manifest.json` and never touched. Only the 91
training-pool sessions (the union of that manifest's 15 clients' session
lists) are redistributed, per (alpha, n_clients, partition_seed), into a NEW
client partition -- independent of and never overwriting the Item 1+2
manifest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))
from data import ALL_CATEGORIES, DATA_DIR

FROZEN_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "shared" / "data_prep" / "federated_clients_manifest.json"


def _load_frozen_manifest():
    import json
    with open(FROZEN_MANIFEST_PATH) as f:
        return json.load(f)


def get_fixed_validation_and_test():
    m = _load_frozen_manifest()
    return m["validation_sessions"], m["held_out_test_sessions"]


def get_training_pool_sessions():
    """The 91 sessions distributed across Item 1+2's 15 frozen clients --
    i.e. everything NOT validation and NOT held-out test."""
    m = _load_frozen_manifest()
    pool = []
    for c in m["clients"].values():
        pool += c["sessions"]
    return pool


def _session_category_and_rows():
    """category + row-count lookup for every session in the dataset (built
    once from the raw CSVs' row counts, cheap: just line counts)."""
    cat_of, rows_of = {}, {}
    for cat_dir in DATA_DIR.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.glob("*.csv"):
                cat_of[f.stem] = cat_dir.name
                with open(f) as fh:
                    rows_of[f.stem] = sum(1 for _ in fh) - 1
    return cat_of, rows_of


_CAT_OF, _ROWS_OF = None, None


def _ensure_lookups():
    global _CAT_OF, _ROWS_OF
    if _CAT_OF is None:
        _CAT_OF, _ROWS_OF = _session_category_and_rows()


def category_row_distribution(sessions):
    _ensure_lookups()
    counts = {c: 0 for c in ALL_CATEGORIES}
    for s in sessions:
        counts[_CAT_OF[s]] += _ROWS_OF[s]
    total = sum(counts.values())
    dist = {c: (counts[c] / total if total else 0.0) for c in ALL_CATEGORIES}
    return counts, dist


def hhi(shares):
    return float(np.sum(np.square(shares)))


def gini(shares):
    x = np.sort(np.array(shares, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def build_partition(alpha, n_clients, partition_seed):
    """Returns (client_sessions: dict[int, list[str]], diagnostics: dict).
    Only the 91 training-pool sessions are used. Never splits a session.
    Empty-client fallback: steal one session from the currently-largest
    client (by row count) and document it -- no re-drawing to avoid an
    empty client."""
    _ensure_lookups()
    rng = np.random.RandomState(partition_seed)
    pool_sessions = get_training_pool_sessions()
    pool_df = pd.DataFrame({"session": pool_sessions})
    pool_df["category"] = pool_df.session.map(_CAT_OF)
    pool_df["n_rows"] = pool_df.session.map(_ROWS_OF)

    client_sessions = {i: [] for i in range(n_clients)}
    for cat, g in pool_df.groupby("category"):
        sessions = list(g.session)
        rng.shuffle(sessions)
        weights = rng.dirichlet([alpha] * n_clients)
        for s in sessions:
            c = rng.choice(n_clients, p=weights)
            client_sessions[c].append(s)

    # --- empty-client fallback, documented ---
    reallocation_events = []
    empty_clients_before = [c for c, ss in client_sessions.items() if len(ss) == 0]
    for c in empty_clients_before:
        sizes = {cc: sum(_ROWS_OF[s] for s in ss) for cc, ss in client_sessions.items() if len(ss) > 1}
        if not sizes:
            break  # cannot rescue -- documented as a genuinely empty client below
        donor = max(sizes, key=sizes.get)
        donor_sessions = sorted(client_sessions[donor], key=lambda s: _ROWS_OF[s])
        stolen = donor_sessions[0]  # steal the donor's smallest session, minimizing disruption
        client_sessions[donor].remove(stolen)
        client_sessions[c].append(stolen)
        reallocation_events.append({"empty_client": c, "donor_client": donor, "session_moved": stolen,
                                     "session_rows": _ROWS_OF[stolen]})

    empty_clients_after = [c for c, ss in client_sessions.items() if len(ss) == 0]

    # --- global training-pool distribution (fixed regardless of partition) ---
    global_counts, global_dist = category_row_distribution(pool_sessions)
    global_vec = np.array([global_dist[c] for c in ALL_CATEGORIES])

    client_diag = {}
    js_divs = []
    for c, sessions in client_sessions.items():
        row_counts, row_dist = category_row_distribution(sessions)
        dist_vec = np.array([row_dist[cc] for cc in ALL_CATEGORIES])
        js = float(jensenshannon(dist_vec, global_vec, base=2) ** 2) if sessions else float("nan")
        js_divs.append(js)
        client_diag[c] = {
            "n_sessions": len(sessions), "n_rows": sum(_ROWS_OF[s] for s in sessions),
            "row_counts_by_category": row_counts, "js_divergence_from_pool": js,
        }

    n_holders_per_class, hhi_per_class, gini_per_class = {}, {}, {}
    total_rows_per_class = {cat: global_counts[cat] for cat in ALL_CATEGORIES}
    for cat in ALL_CATEGORIES:
        shares = []
        n_holders = 0
        for c, sessions in client_sessions.items():
            rows_c = sum(_ROWS_OF[s] for s in sessions if _CAT_OF[s] == cat)
            if rows_c > 0:
                n_holders += 1
            shares.append(rows_c / total_rows_per_class[cat] if total_rows_per_class[cat] else 0.0)
        n_holders_per_class[cat] = n_holders
        hhi_per_class[cat] = hhi(shares)
        gini_per_class[cat] = gini(shares)

    sizes = np.array([client_diag[c]["n_rows"] for c in range(n_clients)])
    size_disparity = {
        "min_rows": int(sizes.min()), "max_rows": int(sizes.max()),
        "ratio_max_min": float(sizes.max() / max(sizes.min(), 1)),
        "cv": float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else float("nan"),
    }

    diagnostics = {
        "alpha": alpha, "n_clients": n_clients, "partition_seed": partition_seed,
        "n_empty_clients_before_reallocation": len(empty_clients_before),
        "n_empty_clients_after_reallocation": len(empty_clients_after),
        "empty_client_ids_after": empty_clients_after,
        "reallocation_applied": len(reallocation_events) > 0,
        "reallocation_events": reallocation_events,
        "n_holders_per_class": n_holders_per_class,
        "hhi_per_class": hhi_per_class, "gini_per_class": gini_per_class,
        "js_divergence_mean_across_clients": float(np.nanmean(js_divs)),
        "js_divergence_max_across_clients": float(np.nanmax(js_divs)),
        "client_size_disparity": size_disparity,
        "client_diagnostics": client_diag,
    }
    return client_sessions, diagnostics


if __name__ == "__main__":
    val, test = get_fixed_validation_and_test()
    pool = get_training_pool_sessions()
    print(f"Fixed validation sessions: {len(val)}  Fixed test sessions: {len(test)}  Training pool: {len(pool)}")
    cs, diag = build_partition(alpha=0.1, n_clients=30, partition_seed=101)
    print(f"\nalpha=0.1, n_clients=30, partition_seed=101:")
    print(f"  empty clients before/after: {diag['n_empty_clients_before_reallocation']}/{diag['n_empty_clients_after_reallocation']}")
    print(f"  reallocation applied: {diag['reallocation_applied']} ({len(diag['reallocation_events'])} events)")
    print(f"  n_holders_per_class: {diag['n_holders_per_class']}")
    print(f"  HHI per class: { {k: round(v,3) for k,v in diag['hhi_per_class'].items()} }")
    print(f"  Gini per class: { {k: round(v,3) for k,v in diag['gini_per_class'].items()} }")
    print(f"  JS-div mean/max across clients: {diag['js_divergence_mean_across_clients']:.4f} / {diag['js_divergence_max_across_clients']:.4f}")
    print(f"  size disparity: {diag['client_size_disparity']}")
    total_sessions = sum(len(s) for s in cs.values())
    print(f"  total sessions distributed: {total_sessions} (should be {len(pool)})")
