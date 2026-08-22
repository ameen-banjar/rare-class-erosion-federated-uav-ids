# Reproducing Paper 1

All locked numbers in `RESULTS_ITEM1_ITEM2_LOCKED.md` and `item3_sensitivity/RESULTS_ITEM3_LOCKED.md` were produced by the scripts in this folder against the dataset described in `../shared/data_prep/README.md`. Every derived result (CSVs, JSON diagnostics, figures, checkpoints) needed to regenerate the locked documents' tables/figures is already committed — you do **not** need to retrain anything to check the analysis. Retraining from scratch is only needed to verify the numbers themselves.

## 1. Environment

```bash
conda env create -f ../environment.yml
conda activate uav-federated-ids
```

Python 3.11, PyTorch 2.10 (Apple Silicon MPS or CPU — see `../environment.yml` for CUDA notes), NumPy 1.26, pandas 2.3, SciPy 1.16, scikit-learn 1.7, matplotlib 3.10. All results in this repository were produced on Apple Silicon (MPS backend).

## 2. Dataset

Not bundled — see `../shared/data_prep/README.md` for the official source, expected layout, and a checksum-verification script. Once placed:

```bash
export ISOT_DATA_DIR=/path/to/isot_drone_dataset
```

Build the frozen client/validation/test partition manifest once (already committed at `../shared/data_prep/federated_clients_manifest.json` — only needed if verifying from a clean dataset copy):

```bash
cd ../shared/data_prep && python3 build_federated_clients.py
```

## 3. Smoke test (verifies the pipeline end-to-end before committing to full runs)

```bash
cd paper1_rare_class_erosion
python3 item2_smoke_single_seed.py
```

**Not a quick toy check** — this is a full 45-round, single-seed (seed=11), validation-only run of SCAFFOLD, FedNova, and FedAdam (order-of-magnitude similar cost to one row of the Item 3 table above, so expect roughly 20–40 minutes on Apple Silicon). It exists to verify data loading, partitioning, and all three algorithm implementations converge sanely *before* committing to the full LR-search-plus-5-seed program in §4, not as a fast smoke check. If you only want to confirm the environment and data path are wired correctly, interrupt it after round 1–2 finishes printing — a clean per-round macro-F1 line for all three algorithms is sufficient evidence the pipeline runs end-to-end.

## 4. Full experiments and approximate cost

All figures below are wall-clock, single-machine (Apple Silicon, MPS), sequential — no parallelism across runs. Treat as order-of-magnitude, hardware-dependent estimates, not guarantees.

| Stage | Script(s) | Runs | Approx. time |
|---|---|---:|---|
| Item 1 mechanism diagnostic | `update_conflict_analysis.py` | 5 seeds × 6 rounds sampled | ~30–60 min |
| Item 2 LR search + stability checks | `item2_optimizer_screening.py`, `item2_boundary_extension*.py`, `item2_confirm_convergence.py`, `item2_fedadam_fallback.py`, `item2_scaffold_fallback.py` | staged, ~40 runs total | several hours (see `RESULTS_ITEM1_ITEM2_LOCKED.md` §4.1 for the full staged history) |
| Item 2 final 5-seed, 5-algorithm run | `item2_five_seed_final.py` | 25 (2 diverged) | ~2–4 hours |
| Item 2 final test evaluation | `item2_final_test_metrics_v2.py` | 23 checkpoints, inference only | ~10–20 min |
| Item 3 exploratory grid | `item3_sensitivity/run_exploratory_grid.py --batch {1,2,3}` | 54 (18/batch) | ~800s/run observed → **~4 hours/batch, ~12 hours total**; resumable per batch, safe to run batches on separate days |
| Item 3 confirmation phase | `item3_sensitivity/run_confirmation.py` | 30 planned, 29 completed | similar per-run cost to exploratory plus one extra mechanistic-diagnostic pass at round 45 (~45–55s/algorithm) → **~7–8 hours** |

Disk: this repository (code + committed derived results + figures + checkpoints, no dataset) is **~14MB**. The dataset itself (not included) is ~1.3GB of CSVs once extracted. Peak working memory during a run: dataset-loading + one client's session batch in memory at a time — a few GB is sufficient; no distributed/multi-GPU setup is used or required.

## 5. Regenerating tables, figures, and lock manifests

```bash
# Item 1+2
python3 generate_figures.py
python3 generate_lock_manifest.py       # recomputes SHA-256 over this folder's current files

# Item 3
cd item3_sensitivity
python3 analyze_exploratory.py          # regenerates analysis_q1..q6*.csv + 2 figures
python3 analyze_confirmation.py         # regenerates confirmation_table1..6*.csv + 2 figures
python3 generate_item3_lock_manifest.py
```

`generate_lock_manifest.py` / `generate_item3_lock_manifest.py` hash **this repository's current files**, not the private research copy's originals — a handful of scripts here were edited for path portability (personal absolute paths → `ISOT_DATA_DIR` env var / repo-relative paths) when this reproduction package was prepared. No algorithmic, numeric, or training-logic line was changed in that edit; only `sys.path`/`DATA_DIR`/`AUDIT_DIR` plumbing. If you re-run the manifest generators here, the resulting hashes will match what's actually in this repository, which is the correct thing for a reproduction package to assert.

## 6. Verifying determinism

Every script pins both `partition_seed` (Dirichlet draw) and `model_seed` (network init + batch shuffling) explicitly and separately — never a single ambiguous "seed". Re-running any individual (algorithm, α, n_clients, partition_seed, model_seed) configuration should reproduce the committed round-45 macro-F1 exactly; this was verified during Item 3's confirmation phase (`item3_sensitivity/results/confirmation_seed11_determinism_check.csv` — 6/6 exact matches against the independently-run exploratory grid).

## 7. Held-out test protocol (do not modify)

`../shared/data_prep/federated_clients_manifest.json` fixes 28 test sessions and 18 validation sessions permanently across every item in this paper. Item 3 never touches the test sessions at all (see `item3_sensitivity/DESIGN_FROZEN.md`). If you modify the partition protocol for your own experiments, do not reuse the locked results in this repository as a baseline comparison without re-verifying the split is unchanged.
