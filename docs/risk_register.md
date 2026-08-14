# VisionLab Risk Register

Status: T1 accepted; Phase 1 not started.

This register tracks project risks that should be reviewed at phase checks and updated when implementation evidence changes their likelihood, impact, or control plan.

| ID | Risk | Current control | Status |
| --- | --- | --- | --- |
| R-001 | Scope expands into a broad survey of classification, detection, segmentation, simulation, edge deployment, or unrelated domains. | Classification-first core, deferred applied-domain gate, explicit non-goals, approval for major changes. | Open |
| R-002 | Broad phases hide multiple approvals, training boundaries, or artifact-contract decisions. | Use bounded phase plans, approved subphases when needed, phase checks, and closeouts. | Open |
| R-003 | Correlated images leak across train, validation, test, OOD, or real-world boundaries. | Future group-aware manifests, source-aware split review, and visual inspection before material training. | Open |
| R-004 | Models learn shortcuts from backgrounds, compression, source artifacts, or generation metadata. | Future source controls, failure analysis, cross-source evaluation, and cautious interpretation. | Open |
| R-005 | Training ambitions exceed local or free-tier compute. | Smoke paths before material runs, bounded models, no uncontrolled search, explicit training approval. | Open |
| R-006 | Experimental integrity is weakened by tuning toward a preferred result. | Preserve null and mixed results; separate engineering success from experimental outcome. | Open |
| R-007 | Real-data sample count is overstated when many images come from few objects or sessions. | Future object/session identity tracking and effective independent sample reporting. | Open |
| R-008 | Diagnostics such as Grad-CAM are overstated as proof of model reasoning. | Diagnostics limitations must be documented near diagnostic outputs and claims. | Open |
| R-009 | AI-generated implementation outpaces builder understanding and review. | Concept briefings, approval gates, phase checks, and journaled AI recommendations. | Open |
| R-010 | Portfolio language overstates evidence before artifacts exist. | README remains status-focused; numerical claims must trace to preserved artifacts. | Open |
| R-011 | Current Python 3.14.5 environment may not be compatible with the intended PyTorch stack. | T1 verified an ignored local `.venv` with `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, CPU tensor ops, and a tiny convolution check. Keep dependency range unchanged until a later dependency-pinning phase. | Reduced; monitor |
| R-012 | Test commands are ambiguous because raw `unittest` does not read `pyproject.toml` `pythonpath` settings. | Use `scripts/test.ps1` as the deterministic local test command; it sets `PYTHONPATH=src` explicitly before running unittest. | Controlled |

## Review Notes

- 2026-08-13: T0 initialized this register from the specification's initial risks and added the observed Python 3.14.5 compatibility risk.
- 2026-08-14: T1 reduced R-011 with local CPU PyTorch evidence and added R-012 for the now-documented test invocation behavior.
