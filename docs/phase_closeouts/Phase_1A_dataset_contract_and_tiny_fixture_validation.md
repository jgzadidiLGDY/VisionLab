# Phase 1A - Dataset Contract and Deterministic Tiny-Fixture Validation

Date: 2026-08-14

Status: Complete.

Builder decision: Accepted on 2026-08-14. No further Phase 1A expansion requested.

## Objective

Define a focused dataset contract and deterministic tiny-fixture validation path before registering a real development dataset.

## What Changed

- Added minimal dataset contract objects under `src/visionlab/data/`.
- Added manifest validation for identity-adjacent consistency, declared split membership, labels, relative paths, sample file presence, tiny image metadata, and class counts.
- Added committed tiny RGB ASCII Netpbm fixtures under `data/fixtures/phase1_tiny_rgb/`.
- Added deterministic unit tests in `tests/test_data_contract.py`.
- Added the dataset-contract reference at `docs/dataset_contract.md`.
- Updated README, phase catalog, and builder journal status.

## Evidence

Automated verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result:

```text
Ran 18 tests in 0.071s
OK
```

## Resulting Contract

The Phase 1A contract records:

- dataset identity: `dataset_id`, `version`, `source`, `license_or_usage`, optional `description`;
- ordered class names and derived class-to-index mapping;
- declared split names;
- deterministic preprocessing metadata: image size, color mode, value range, normalization mean/std, deterministic flag;
- sample records: `sample_id`, `split`, `label`, `relative_path`, optional `source_id`, `group_id`, and `checksum`.

## Validation Behavior

Phase 1A validation checks:

- duplicate sample IDs;
- undeclared split names;
- undeclared labels;
- unsafe relative paths;
- missing sample files;
- invalid tiny fixture images;
- image size mismatch;
- color-mode mismatch;
- empty split warnings;
- class counts by split.

## Known Limitations

- The tiny image parser is intentionally limited to ASCII Netpbm fixtures for dependency-free tests.
- No CIFAR-10 data has been downloaded or registered.
- No manifest JSON/YAML loader has been added yet.
- No checksum computation has been added yet.
- `group_id` is optional; datasets without correlated-group metadata must document that limitation.

## Phase Check

Intended scope versus implementation:

- Phase 1A stayed within contract, validation, tiny fixtures, tests, and documentation.
- No model, dataloader, training, material dataset registration, or visual inspection artifact generation was added.

Learning objective:

- The builder has an inspectable contract for how dataset identity, classes, splits, sample membership, labels, and deterministic preprocessing will be represented before training.

Data and split integrity:

- Tiny committed fixtures validate the contract mechanics only.
- CIFAR-10 remains provisional and unregistered.

Readiness:

- Ready to plan Phase 1B CIFAR-10 registration and visual data inspection.
- Phase 1B must define the CIFAR-10 source/version record, train/validation/test split policy, stable sample-ID strategy, deterministic preprocessing profile, class-count report, visual inspection artifacts, and group/leakage limitations.
