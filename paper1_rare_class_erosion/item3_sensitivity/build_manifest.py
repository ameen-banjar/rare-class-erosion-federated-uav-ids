"""
Builds the static, resumable manifest for the 54-run Item 3 exploratory
grid. Fixed execution order: partition_seed (batch) outer, then the 9
(alpha, n_clients) combinations in a fixed order, then algorithm alternating
{fedavg_sgd, fednova_sgd} within each combination -- so batch 1
(partition_seed=101) is 18 runs, batch 2 (102) is 18, batch 3 (103) is 18.

Each run has a stable run_id and a session_partition_hash (sha256 of the
realized client->sessions assignment) computed once per (alpha, n_clients,
partition_seed) triple and shared by its two algorithm runs -- proves both
algorithms in a triple trained on the IDENTICAL partition.
"""
import hashlib
import json
from pathlib import Path

from build_partition import build_partition

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MANIFEST_PATH = RESULTS_DIR / "item3_run_manifest.json"

MODEL_SEED = 11
PARTITION_SEEDS = [101, 102, 103]
ALPHAS = [0.1, 0.3, 1.0]
N_CLIENTS_GRID = [10, 15, 30]
ALGO_CONFIGS = {"fedavg_sgd": {"lr": 0.5}, "fednova_sgd": {"lr": 0.3}}
NUM_ROUNDS = 45


def session_partition_hash(client_sessions):
    payload = json.dumps({str(k): sorted(v) for k, v in sorted(client_sessions.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = []
    run_idx = 0
    for batch_num, partition_seed in enumerate(PARTITION_SEEDS, start=1):
        for alpha in ALPHAS:
            for n_clients in N_CLIENTS_GRID:
                client_sessions, _ = build_partition(alpha, n_clients, partition_seed)
                phash = session_partition_hash(client_sessions)
                for algo in ["fedavg_sgd", "fednova_sgd"]:  # fixed alternation order
                    run_idx += 1
                    runs.append({
                        "run_id": run_idx, "batch": batch_num,
                        "algorithm": algo, "config": ALGO_CONFIGS[algo],
                        "alpha": alpha, "n_clients": n_clients,
                        "partition_seed": partition_seed, "model_seed": MODEL_SEED,
                        "num_rounds": NUM_ROUNDS,
                        "session_partition_hash": phash,
                        "status": "pending",
                        "rounds_completed": 0,
                        "diverged": False, "diverged_at_round": None, "diverged_where": None,
                        "result_round_metrics_path": "results/item3_exploratory_round_metrics.csv",
                    })
        print(f"Batch {batch_num} (partition_seed={partition_seed}): 18 runs planned")

    with open(MANIFEST_PATH, "w") as f:
        json.dump({"total_runs": len(runs), "runs": runs}, f, indent=2)
    print(f"\nTotal runs planned: {len(runs)}")
    print(f"Saved -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
