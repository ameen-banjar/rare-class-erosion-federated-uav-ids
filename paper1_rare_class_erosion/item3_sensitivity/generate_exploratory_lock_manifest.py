"""SHA-256 lock manifest for the completed Item 3 exploratory grid (54 runs,
2,430 rows), taken BEFORE any results analysis, so the analysis cannot be
suspected of selectively re-running or editing inputs after seeing numbers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256_of(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def hash_files(paths):
    out = {}
    for p in sorted(paths):
        if p.is_file():
            out[str(p.relative_to(ROOT))] = {"sha256": sha256_of(p), "size_bytes": p.stat().st_size}
    return out


def main():
    manifest = {"locked_at": "2026-08-21", "stage": "exploratory_grid_pre_analysis"}
    manifest["result_files"] = hash_files([
        ROOT / "results" / "item3_exploratory_round_metrics.csv",
        ROOT / "results" / "item3_exploratory_partition_diagnostics.json",
        ROOT / "results" / "item3_run_manifest.json",
    ])
    manifest["code"] = hash_files([
        ROOT / "build_partition.py", ROOT / "build_manifest.py", ROOT / "run_exploratory_grid.py",
    ])
    out_path = ROOT / "EXPLORATORY_LOCK_MANIFEST.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Locked files:")
    for cat, files in [("result_files", manifest["result_files"]), ("code", manifest["code"])]:
        for name in files:
            print(f"  [{cat}] {name}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
