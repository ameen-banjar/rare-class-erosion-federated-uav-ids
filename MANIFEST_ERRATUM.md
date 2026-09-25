# Lock-manifest erratum (documentation-only)

Five entries in the lock manifests do not match the current bytes of four files. The discrepancy was introduced after v1.1.1 (v1.1.1 still matches), is present in v1.2.0, and was found while preparing v1.2.1; v1.2.1 leaves the manifests and these files untouched so the discrepancy stays visible.

**Cause:** commit `e81edcb` (2026-09-16, "Remove \"Paper 1\" framing from visible docs and code comments") changed one line in each file: it deleted the words "Paper 1" from a title or docstring. No result, number, code statement or logic changed. The manifests were locked earlier (2026-08-20 and 2026-08-22) and were not regenerated.

To confirm: `git diff e81edcb^ e81edcb -- <file>` shows a single-line wording change per file.

| File | Manifest | Locked SHA-256 (manifest) | Current SHA-256 |
|---|---|---|---|
| `paper1_rare_class_erosion/RESULTS_ITEM1_ITEM2_LOCKED.md` | `LOCK_MANIFEST.json` | `6ac114daa78de0260cd6a10898f3e38702db83d11c464bf8c98debd6f6208b7f` | `7fd1a383839fc6de7346423a815dd53aea1e993a0e19f20231d4bbcf5d24c8b1` |
| `paper1_rare_class_erosion/update_conflict_analysis.py` | `LOCK_MANIFEST.json` | `a9fd3178e7ed911d7f024f403d6d340ea9e3b844e8664666ab7892d6cb8d30b1` | `56c2a20f8f419dc4f7ee5b4d7623b9460c7d6c5fb3cb4dcfaf7646a156f1ed33` |
| `shared/fl_pipeline/advanced_algorithms.py` | `LOCK_MANIFEST.json` | `0212c36f9fd3f659a07bc228b8a06365e8d0d0239a95444bd5fa2a0b2ef5a88a` | `ed1c0a2da19ab83884c1d598be5f34f7538e34430300cc1b6b8e8415411f82c0` |
| `shared/fl_pipeline/advanced_algorithms.py` | `ITEM3_LOCK_MANIFEST.json` | `0212c36f9fd3f659a07bc228b8a06365e8d0d0239a95444bd5fa2a0b2ef5a88a` | `ed1c0a2da19ab83884c1d598be5f34f7538e34430300cc1b6b8e8415411f82c0` |
| `paper1_rare_class_erosion/item3_sensitivity/RESULTS_ITEM3_LOCKED.md` | `ITEM3_LOCK_MANIFEST.json` | `74579a70f53d756c45cbc8a66c984a86dbd4c51d21976d5430150ad144046c8b` | `c6649b2d25c7d7c4eb619c8caf053b2a00549a2df92585eb9d67c28a0abcf3fa` |

Every other path-keyed manifest entry (212 of 217 checked) matches the current files.
