# T0 — Project Bootstrap and Baseline Capture

Date: 2026-08-13

Status: Complete.

Builder decision: Accepted on 2026-08-13.

## Objective

Establish VisionLab's truthful repository baseline, operating documents, version-control boundaries, and a minimal local smoke path before any substantive vision implementation begins.

## What Changed

- Initialized Git for the repository.
- Moved the project specification to `docs/project_specs.md`.
- Added `docs/risk_register.md`, `docs/requirement_change_log.md`, and `docs/phase_catalog.md`.
- Added a minimal Python package skeleton under `src/visionlab/`.
- Added a local smoke command and deterministic smoke test.
- Added `.env.example` with non-secret local defaults.
- Preserved T1 and later work as explicitly out of scope.

## Baseline Evidence

- No dataset has been selected or registered.
- No applied domain has been selected.
- No model, training, evaluation, diagnostic, inference, or artifact pipeline has been implemented.
- No material training or external data acquisition has been approved or run.
- The observed local Python version is 3.14.5 and is recorded as a T1 compatibility risk for the intended PyTorch stack.

## Verification

- `python -m visionlab.smoke` with `PYTHONPATH=src` passed.
- `python -m unittest discover -s tests` with `PYTHONPATH=src` passed: 1 test.
- `git status --short` succeeded after `git init`; all files remain untracked because no commit has been requested.
- Repository references were searched for `project_specs.md`; active project references now point to `docs/project_specs.md`.

Smoke output recorded:

```text
visionlab_version: 0.0.0
python_version: 3.14.5
repo_root: C:\codex_workspace\VisionLab
project_spec_exists: True
```

## Requirement Impact

No material project requirement change was made during T0. The repository structure was corrected to match the documented project setup.

## Known Limitations

- Dependency decisions remain unresolved.
- PyTorch compatibility with the current Python version remains unresolved until T1.
- The smoke path validates only repository wiring, not computer-vision functionality.

## Next Boundary

T1 has not started. T1 may be planned next with a separate concept briefing and approval boundary. It should address vision-foundation concepts, local environment feasibility, and PyTorch/device compatibility without selecting the applied domain.
