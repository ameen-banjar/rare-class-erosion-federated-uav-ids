"""
Generates LOCK_MANIFEST.json -- SHA-256 of every artifact behind the locked
Item 1+2 results (RESULTS_ITEM1_ITEM2_LOCKED.md). Run once at lock time; if
this script is ever re-run and a hash differs from the committed manifest,
that file was modified after the lock and the discrepancy must be
investigated before trusting any number derived from it.

Categories: locked results document, figures, raw result files, analysis/
evaluation scripts (this folder + the shared fl_pipeline modules they
import -- if those change, Item 1+2 results are no longer reproducible from
the current codebase even though the saved CSVs are untouched), metadata,
and checkpoints (Item 1's per-round diagnostic checkpoints + Item 2's
five-seed round-45 checkpoints).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FL_PIPELINE = ROOT.parent / "shared" / "fl_pipeline"


def sha256_of(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def hash_files(paths):
    out = {}
    for p in sorted(paths):
        if p.is_file():
            out[str(p.relative_to(ROOT.parent))] = {
                "sha256": sha256_of(p),
                "size_bytes": p.stat().st_size,
            }
    return out


def main():
    manifest = {"locked_at": "2026-08-20", "root": str(ROOT.relative_to(ROOT.parent))}

    manifest["locked_results_document"] = hash_files([ROOT / "RESULTS_ITEM1_ITEM2_LOCKED.md"])
    manifest["design_frozen_document"] = hash_files([ROOT / "DESIGN_FROZEN.md"])
    manifest["figures"] = hash_files((ROOT / "figures").glob("*.png"))

    manifest["raw_result_files"] = hash_files((ROOT / "results").glob("*"))

    scripts = list(ROOT.glob("*.py"))
    fl_pipeline_deps = [FL_PIPELINE / n for n in
                         ["algorithms.py", "advanced_algorithms.py", "data.py", "model.py"]]
    manifest["analysis_and_evaluation_scripts"] = hash_files(scripts + fl_pipeline_deps)

    manifest["metadata"] = hash_files((ROOT / "results").glob("*metadata*.json"))

    manifest["checkpoints_item1_update_conflict"] = hash_files((ROOT / "checkpoints").glob("*.pt"))
    manifest["checkpoints_item2_five_seed"] = hash_files((ROOT / "checkpoints_five_seed").glob("*.pt"))

    manifest["counts"] = {k: len(v) for k, v in manifest.items() if isinstance(v, dict) and k != "counts"
                           and all(isinstance(x, dict) for x in v.values())}

    out_path = ROOT / "LOCK_MANIFEST.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Category counts:")
    for k, v in manifest["counts"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
