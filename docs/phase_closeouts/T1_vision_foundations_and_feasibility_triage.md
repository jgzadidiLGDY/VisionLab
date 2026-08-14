# T1 - Vision Foundations and Feasibility Triage

Date: 2026-08-14

Status: Complete.

Builder decision: Accepted on 2026-08-14, with final builder visual review of generated T1 foundation artifacts recorded as the remaining manual review condition.

## Objective

Develop the minimum conceptual foundation needed to evaluate early vision work, verify the local Python/PyTorch path, compare low-friction development datasets, and prepare for Phase 1 without beginning dataset registration or material training.

## What Changed

- Added T1 concept notes at `docs/vision_foundations.md`.
- Added dependency-light convolution and tiny image helpers under `src/visionlab/foundations/`.
- Added an optional-dependency and PyTorch device probe at `src/visionlab/environment.py`.
- Added deterministic tests for foundation helpers and environment probe behavior.
- Added unambiguous local scripts:
  - `scripts/test.ps1`
  - `scripts/probe_environment.ps1`
  - `scripts/make_t1_foundation_artifacts.ps1`
- Added compute feasibility evidence at `docs/compute_feasibility.md`.
- Added development dataset comparison at `docs/development_dataset_candidates.md`.
- Updated project status, risk, phase catalog, and journal records for T1 review.

## Evidence

Automated verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result:

```text
Ran 8 tests in 0.066s
OK
```

Base interpreter probe:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\probe_environment.ps1
```

Result summary:

- Python `3.14.5`
- no `torch`, `torchvision`, `PIL`, `numpy`, or `matplotlib` in the base interpreter
- torch device status: `torch_not_installed`

T1 `.venv` PyTorch probe:

- `torch 2.13.0+cpu`
- `torchvision 0.28.0+cpu`
- CPU tensor op succeeded
- tiny `torch.nn.functional.conv2d` check produced shape `(1, 1, 3, 3)`
- CUDA unavailable locally

Visual/manual verification artifact generation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_t1_foundation_artifacts.ps1
```

Generated ignored local artifacts:

- `outputs/t1_foundations/synthetic_step_image.pgm`
- `outputs/t1_foundations/vertical_edge_feature_map.pgm`
- `outputs/t1_foundations/README.md`

These demonstrate convolution behavior only. They are not dataset samples, model outputs, or diagnostics.

Visual review record:

- Codex visual QA created temporary PNG previews from the ignored PGM artifacts because the local viewer could not render PGM directly.
- `synthetic_step_image.pgm` displays the intended dark/light vertical step image.
- `vertical_edge_feature_map.pgm` displays the expected vertical edge response, including padding-related border response.
- Builder accepted T1 on 2026-08-14 subject to final builder visual review of these generated artifacts.

## Dataset Recommendation

T1 recommends CIFAR-10 as the provisional core development dataset for Phase 1 planning because it is RGB, balanced, low-friction, small enough for bounded training, and does not select the applied domain.

Fashion-MNIST remains a useful fallback for tiny smoke checks. STL-10 and EuroSAT RGB are deferred because they add complexity or risk premature domain coupling.

## Requirement and Governance Impact

No material requirement was changed during T1 implementation.

Approved clarifications:

- clarify that the supported local development path is currently Python 3.14.5 with a local `.venv` and CPU PyTorch wheels;
- keep `requires-python = ">=3.11"` unchanged until a later dependency-pinning phase decides the exact supported range;
- document `scripts/test.ps1` as the deterministic local test command;
- require CPU-compatible default tests and mark GPU/full-dataset/material-training checks separately;
- record CIFAR-10 as a provisional Phase 1 candidate, not an applied-domain decision.

## Known Limitations

- Local CUDA was not available.
- Remote GPU or Colab workflow was not verified.
- No dataset was downloaded, registered, split, or visually inspected.
- No model or trainer was implemented.
- No material training occurred.
- The generated PGM artifacts are deliberately synthetic and tiny.

## Phase Check

Intended scope versus implementation:

- T1 stayed within foundations, probes, and feasibility.
- Phase 1 dataset contract work did not begin.
- Applied-domain selection remains deferred.

Learning objective:

- Builder now has concise reference notes for tensors, normalization, convolution, pooling, feature maps, receptive fields, and split roles.

Test and verification:

- Eight deterministic unittest checks pass through `scripts/test.ps1`.
- Base and `.venv` environment probes recorded the current dependency and device state.
- Tiny visual convolution artifacts were generated for manual inspection.

Data and split integrity:

- No dataset was acquired or split.
- Development dataset recommendation remains provisional.

Readiness:

- Ready to plan Phase 1 after final builder visual review of the T1 foundation artifacts, with CIFAR-10 as the recommended provisional candidate.
