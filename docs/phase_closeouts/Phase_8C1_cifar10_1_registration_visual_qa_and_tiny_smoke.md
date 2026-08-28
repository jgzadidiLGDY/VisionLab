# Phase 8C-1 Closeout - CIFAR-10.1 v6 Registration, Visual QA, and Tiny Smoke

Date: 2026-08-26

Status: Complete and accepted.

Phase 8C-1 acquired, registered, and visually inspected CIFAR-10.1 v6 as VisionLab's first cross-source/evaluation-only dataset. This closeout does not close Phase 8C as a whole because Phase 8C-2 has not started.

## Scope

Phase 8C-1 was limited to external acquisition, dataset registration, visual QA, tiny smoke verification, and artifact preservation for CIFAR-10.1 v6.

The phase did not evaluate any model checkpoint and did not produce cross-source/OOD model-performance conclusions.

## Dataset Identity

- Dataset ID: `cifar10-1`
- Version: `v6`
- Split identity: `cross_source_test`
- Usage boundary: cross-source evaluation-only; never train, tune, or select checkpoints
- Expected sample count: `2,000`
- Image structure: `32 x 32 x 3`
- Raw representation: unnormalized RGB unit tensor `C x H x W` in `[0, 1]` before any later model-specific preprocessing
- Class mapping: exact CIFAR-10 class order: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Acquisition Source

CIFAR-10.1 v6 was acquired only from the declared official source files:

- `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy`
- `https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy`

No substitute dataset or alternate version was used.

## Integrity Evidence

The accepted registration artifact records:

- sample count: `2,000`
- image shape: `[32, 32, 3]`
- label range: CIFAR-10 class IDs `0` through `9`
- class count: `10`
- class distribution: `200` examples for each CIFAR-10 class
- data SHA-256: `2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6`
- labels SHA-256: `ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448`
- sample-label digest: `2afa813c387e578086d1f0aeeb1b9674e352c73c4690b89d69385aedca3e8b75`

## Visual QA

The generated visual QA grid is preserved at:

`outputs/phase8c1-cifar10-1-registration-visual-qa-tiny-smoke/artifacts/phase8c1_cifar10_1_v6_visual_grid.png`

The visual sample manifest used one fixed sample per CIFAR-10 class:

- `cifar10-1-v6-00000`: `airplane`
- `cifar10-1-v6-00200`: `automobile`
- `cifar10-1-v6-00400`: `bird`
- `cifar10-1-v6-00600`: `cat`
- `cifar10-1-v6-00800`: `deer`
- `cifar10-1-v6-01000`: `dog`
- `cifar10-1-v6-01200`: `frog`
- `cifar10-1-v6-01400`: `horse`
- `cifar10-1-v6-01600`: `ship`
- `cifar10-1-v6-01800`: `truck`

Builder visual review: accepted. The builder reviewed the CIFAR-10.1 v6 visual QA grid and confirmed that it looks correct.

Visual QA is qualitative inspection only. It does not establish label correctness, OOD detection capability, degradation robustness, cross-source robustness, or deployment reliability.

## Artifacts

Phase 8C-1 artifacts are preserved under ignored `outputs/phase8c1-cifar10-1-registration-visual-qa-tiny-smoke/`.

Key artifacts:

- `phase8c1_result.json`
- `phase8c1_inspection_note.md`
- `artifacts/phase8c1_cifar10_1_contract.json`
- `artifacts/phase8c1_local_availability.json`
- `artifacts/phase8c1_tiny_fixture_smoke.json`
- `artifacts/phase8c1_tiny_fixture_manifest.json`
- `artifacts/phase8c1_cifar10_1_v6_registration.json`
- `artifacts/phase8c1_cifar10_1_v6_manifest.json`
- `artifacts/phase8c1_visual_sample_manifest.json`
- `artifacts/phase8c1_cifar10_1_v6_visual_grid.png`

The local dataset files are ignored under `data/cifar10.1/`.

## Verification

Focused Phase 8C-1 tests passed with `12` tests.

Canonical deterministic suite passed after implementation with `132` tests and `1` skipped.

After documentation-only closeout updates, the canonical deterministic suite passed with `132` tests and `1` skipped via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1`.

## Explicit Exclusions

Phase 8C-1 did not include:

- evaluation of Phase 4B, Phase 6B-2, or Phase 6C-2 checkpoints;
- official CIFAR-10 test evaluation;
- material cross-source/OOD evaluation;
- training;
- tuning;
- model selection;
- checkpoint modification;
- Phase 8C-2A or Phase 8C-2B implementation or execution;
- Phase 9 failure analysis;
- any OOD-detection claim;
- any robustness or deployment-reliability claim.

## Accepted State

Phase 8C-1 is complete and accepted. CIFAR-10.1 v6 is now registered as a cross-source/evaluation-only dataset for later separately approved Phase 8C-2 work. Phase 8C as a whole is not complete because Phase 8C-2 has not started.
