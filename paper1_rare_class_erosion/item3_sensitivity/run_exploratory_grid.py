"""
Item 3 -- exploratory grid, resumable, manifest-driven, run in three batches
by partition_seed (per review). Usage:

    python3 run_exploratory_grid.py --batch 1   # partition_seed=101, 18 runs
    python3 run_exploratory_grid.py --batch 2   # partition_seed=102, 18 runs
    python3 run_exploratory_grid.py --batch 3   # partition_seed=103, 18 runs

All three batches are committed to running regardless of what earlier
batches show -- no run is skipped based on interim results. On resume after
an interruption, only runs still marked "pending"/"running" in the manifest
are executed; "completed" runs (successful OR diverged -- both are terminal,
reported outcomes) are never re-run. Round-level checkpointing is not used,
so a resumed run restarts from round 1 -- redoing one ~800s run costs far
less than losing the whole batch, and no run that reached "completed" status
is ever repeated.

model_seed is FIXED at 11 for the entire exploratory grid (isolates
partition-driven variance from model-init variance; the confirmation phase
varies model_seed on fixed partitions). Validation only -- held_out_test is
never touched.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "fl_pipeline"))
from data import ISOTFederatedData, ALL_CATEGORIES
from model import AttackFamilyMLP
from algorithms import aggregate_weighted, evaluate
from advanced_algorithms import local_train_fednova, aggregate_fednova, local_train_generic
from build_partition import build_partition

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MANIFEST_PATH = RESULTS_DIR / "item3_run_manifest.json"
ROUND_METRICS_PATH = RESULTS_DIR / "item3_exploratory_round_metrics.csv"
DIAG_PATH = RESULTS_DIR / "item3_exploratory_partition_diagnostics.json"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
BATCH_SIZE = 4096
VAL_LABEL_INDICES = [i for i in range(10) if ALL_CATEGORIES[i] != "Password Cracking"]

CSV_HEADER = ["run_id", "algorithm", "alpha", "n_clients", "partition_seed", "model_seed", "round",
              "val_macro_f1", "val_loss", "per_class_recall_json", "per_class_precision_json",
              "per_class_f1_json", "per_class_predicted_count_json", "elapsed_seconds",
              "diverged", "diverged_at"]


def set_model_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def state_finite(state):
    return all(torch.isfinite(v).all() for v in state.values())


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def client_data_from_partition(data, client_sessions):
    out = {}
    for cid, sessions in client_sessions.items():
        if not sessions:
            continue
        df = data._load_sessions(sessions)
        out[cid] = data.tensors_from_df(df)
    return out


def client_weights_from_partition(client_data):
    total = sum(len(y) for _, y in client_data.values())
    return {cid: len(y) / total for cid, (X, y) in client_data.items()}


def eval_and_log(csv_writer, run, rnd, global_model, Xval, yval, data, t0):
    m = evaluate(global_model, Xval, yval, DEVICE, ALL_CATEGORIES,
                 class_weights=data.class_weights_full_balanced, eval_label_indices=VAL_LABEL_INDICES)
    csv_writer.writerow([run["run_id"], run["algorithm"], run["alpha"], run["n_clients"],
                          run["partition_seed"], run["model_seed"], rnd,
                          m["macro_f1"], m["loss"],
                          json.dumps(m["per_class_recall"]), json.dumps(m["per_class_precision"]),
                          json.dumps(m["per_class_f1"]), json.dumps(m["per_class_predicted_count"]),
                          time.time() - t0, False, ""])
    return m


def log_diverged(csv_writer, run, rnd, where, t0):
    csv_writer.writerow([run["run_id"], run["algorithm"], run["alpha"], run["n_clients"],
                          run["partition_seed"], run["model_seed"], rnd,
                          None, None, "{}", "{}", "{}", "{}", time.time() - t0, True, str(where)])
    print(f"  DIVERGED at round {rnd}, {where}")


def execute_run(run, data, input_dim, client_data, weights, Xval, yval, csv_writer):
    cids = list(client_data.keys())
    w = [weights[c] for c in cids]
    lr = run["config"]["lr"]
    set_model_seed(run["model_seed"])
    global_model = AttackFamilyMLP(input_dim).to(DEVICE)
    t0 = time.time()
    rounds_completed = 0

    for rnd in range(1, run["num_rounds"] + 1):
        if run["algorithm"] == "fedavg_sgd":
            client_states = []
            for cid in cids:
                Xc, yc = client_data[cid]
                local_model = AttackFamilyMLP(input_dim).to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                state, n_steps, n, loss = local_train_generic(local_model, Xc, yc, DEVICE, epochs=1,
                                                                batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                                class_weights=data.class_weights_full_balanced)
                if not state_finite(state):
                    log_diverged(csv_writer, run, rnd, f"client_{cid}", t0)
                    return rounds_completed, "diverged"
                client_states.append(state)
            agg_state = aggregate_weighted(client_states, w)
            if not state_finite(agg_state):
                log_diverged(csv_writer, run, rnd, "aggregation", t0)
                return rounds_completed, "diverged"
            global_model.load_state_dict(agg_state)
        else:  # fednova_sgd
            global_state_start = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}
            client_states, taus = [], []
            for cid in cids:
                Xc, yc = client_data[cid]
                local_model = AttackFamilyMLP(input_dim).to(DEVICE)
                local_model.load_state_dict(global_model.state_dict())
                state, n_steps, n, loss = local_train_fednova(local_model, Xc, yc, DEVICE, epochs=1,
                                                                batch_size=BATCH_SIZE, lr=lr, optimizer="sgd",
                                                                class_weights=data.class_weights_full_balanced)
                if not state_finite(state):
                    log_diverged(csv_writer, run, rnd, f"client_{cid}", t0)
                    return rounds_completed, "diverged"
                client_states.append(state); taus.append(n_steps)
            agg_state = aggregate_fednova(client_states, global_state_start, w, taus)
            if not state_finite(agg_state):
                log_diverged(csv_writer, run, rnd, "aggregation", t0)
                return rounds_completed, "diverged"
            global_model.load_state_dict(agg_state)

        m = eval_and_log(csv_writer, run, rnd, global_model, Xval, yval, data, t0)
        rounds_completed = rnd
        if rnd % 15 == 0 or rnd == 1 or rnd == run["num_rounds"]:
            print(f"  round {rnd}/{run['num_rounds']}  macro_f1={m['macro_f1']:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    return rounds_completed, "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()

    manifest = load_manifest()
    batch_runs = [r for r in manifest["runs"] if r["batch"] == args.batch]
    todo = [r for r in batch_runs if r["status"] not in ("completed",)]
    print(f"Batch {args.batch}: {len(batch_runs)} total runs, {len(todo)} to execute "
          f"({len(batch_runs)-len(todo)} already completed)")

    if not todo:
        print("Nothing to do for this batch.")
        return

    print(f"Device: {DEVICE}")
    data = ISOTFederatedData()
    data.fit_preprocessing()
    input_dim = len(data.feature_cols)
    Xval, yval = data.get_validation_data()

    file_exists = ROUND_METRICS_PATH.exists()
    diagnostics = json.load(open(DIAG_PATH)) if DIAG_PATH.exists() else []
    partition_cache = {}

    with open(ROUND_METRICS_PATH, "a", newline="") as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(CSV_HEADER)

        t_batch0 = time.time()
        for i, run in enumerate(todo, 1):
            run["status"] = "running"
            save_manifest(manifest)
            print(f"\n--- [{i}/{len(todo)}] run_id={run['run_id']}  {run['algorithm']}  "
                  f"alpha={run['alpha']} n_clients={run['n_clients']} partition_seed={run['partition_seed']} "
                  f"model_seed={run['model_seed']} ---  batch_elapsed={time.time()-t_batch0:.0f}s")

            pkey = (run["alpha"], run["n_clients"], run["partition_seed"])
            if pkey not in partition_cache:
                client_sessions, diag = build_partition(*pkey)
                client_data = client_data_from_partition(data, client_sessions)
                weights = client_weights_from_partition(client_data)
                partition_cache[pkey] = (client_data, weights)
                diagnostics.append(diag)
                with open(DIAG_PATH, "w") as df_:
                    json.dump(diagnostics, df_, indent=2, default=str)
            client_data, weights = partition_cache[pkey]

            rounds_completed, outcome = execute_run(run, data, input_dim, client_data, weights, Xval, yval, csv_writer)
            f.flush()

            run["status"] = "completed"  # both success and divergence are terminal, reported outcomes
            run["rounds_completed"] = rounds_completed
            run["diverged"] = (outcome == "diverged")
            save_manifest(manifest)
            print(f"  -> {outcome}, rounds_completed={rounds_completed}")

    print(f"\nBatch {args.batch} done. Elapsed {time.time()-t_batch0:.0f}s.")
    print(f"Saved -> {ROUND_METRICS_PATH}\nSaved -> {DIAG_PATH}\nManifest -> {MANIFEST_PATH}")
    print("(held_out_test never touched; only training-pool sessions redistributed)")


if __name__ == "__main__":
    main()
