"""Generates ITEM3_LOCK_MANIFEST.json -- SHA-256 of every artifact behind
RESULTS_ITEM3_LOCKED.md. Independent of, and never modifies, the Item 1+2
manifest (../LOCK_MANIFEST.json). Run once at lock time; if re-run later and
a hash differs from the committed manifest, that file was modified after
the lock and any number derived from it must be re-verified before trust.

Categories: locked results document, the three frozen design/analysis-plan
documents (with their amendment sections), figures, raw result files
(exploratory + confirmation, including the two never-modified source CSVs
this whole document is derived from), analysis/evaluation scripts (this
folder + the shared fl_pipeline modules they import), and confirmation-phase
checkpoints (the exploratory grid saved no checkpoints by design).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER1_ROOT = ROOT.parent
FL_PIPELINE = PAPER1_ROOT.parent / "shared" / "fl_pipeline"


def sha256_of(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def hash_files(paths):
    out = {}
    for p in sorted(set(paths)):
        if p.is_file():
            out[str(p.relative_to(PAPER1_ROOT.parent))] = {
                "sha256": sha256_of(p),
                "size_bytes": p.stat().st_size,
            }
    return out


def main():
    manifest = {"locked_at": "2026-08-22", "root": str(ROOT.relative_to(PAPER1_ROOT.parent)),
                "note": "Item 3 only. Independent of and does not modify ../LOCK_MANIFEST.json (Item 1+2)."}

    manifest["locked_results_document"] = hash_files([ROOT / "RESULTS_ITEM3_LOCKED.md"])
    manifest["design_and_analysis_plan_documents"] = hash_files([
        ROOT / "DESIGN_FROZEN.md",
        ROOT / "ANALYSIS_PLAN_FROZEN.md",
        ROOT / "CONFIRMATION_ANALYSIS_PLAN_FROZEN.md",
    ])
    manifest["figures"] = hash_files((ROOT / "figures").glob("*.png"))
    manifest["raw_result_files"] = hash_files((ROOT / "results").glob("*"))

    scripts = list(ROOT.glob("*.py"))
    fl_pipeline_deps = [FL_PIPELINE / n for n in
                         ["algorithms.py", "advanced_algorithms.py", "data.py", "model.py"]]
    manifest["analysis_and_evaluation_scripts"] = hash_files(scripts + fl_pipeline_deps)

    manifest["checkpoints_confirmation"] = hash_files((ROOT / "checkpoints_confirmation").glob("*.pt"))

    manifest["counts"] = {k: len(v) for k, v in manifest.items()
                           if isinstance(v, dict) and k not in ("counts",)
                           and all(isinstance(x, dict) for x in v.values())}

    out_path = ROOT / "ITEM3_LOCK_MANIFEST.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Category counts:")
    for k, v in manifest["counts"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
