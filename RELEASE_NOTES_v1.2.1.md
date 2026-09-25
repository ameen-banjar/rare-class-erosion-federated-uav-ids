## v1.2.1 - metadata and documentation only

**No code, data, result, checkpoint or integrity manifest changed relative to v1.2.0.** All numbers in the manuscript *Rare-Class Knowledge Retention in Federated Intrusion Detection: A Session-Level UAV Diagnostic with Mechanistic Localization Tests and Cross-Domain Validation* remain traceable to the same immutable evidence releases:

- v1.1.1 (`10.5281/zenodo.22236554`): Items 1-3 (manuscript Sections 5, 7, 8)
- v1.2.0 (`10.5281/zenodo.22899158`): Items 4-5 (manuscript Sections 6 and 9)

### What changed
- `CITATION.cff`: title aligned with the manuscript's terminology (retention rather than erosion), version 1.2.1, concept DOI as the primary `doi`, earlier version DOIs listed under `identifiers`, preferred-citation status set to "submitted".
- `README.md`: wording aligned with the manuscript.

### Terminology clarification
Earlier text in this repository and in the v1.2.0 release notes calls the Item 4 and Item 5 designs "pre-registered". Precisely, each design was written and frozen in version control (`DESIGN_FROZEN.md`, commit dates in the git history) before the corresponding experiments were executed; there is no external registry entry, and the public archive was created after execution. The historical `DESIGN_FROZEN.md` files are unchanged (they are covered by the integrity manifests).

### Item 4 counts
The v1.2.0 notes report the logit-margin endpoint as negative in 179 of the 180 tested cells. The manuscript reports the same result both ways: 179/180 over the full grid, and 129/129 restricted to acquisition-supported cells (43 acquired units x 3 interventions); the single positive cell is in a non-acquired unit.
