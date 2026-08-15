# Phase 1 - Dataset Contract and Visual Data Inspection

Date: 2026-08-14

Status: Complete.

Builder decision: Phase 1A and Phase 1B accepted. Phase 1 is closed; Phase 2 has not started.

## Objective

Establish a trustworthy, inspectable dataset foundation before model implementation or training.

## Completed Subphases

- Phase 1A - Dataset Contract and Deterministic Tiny-Fixture Validation.
- Phase 1B - CIFAR-10 Registration and Visual Data Inspection.

## What Phase 1 Established

Dataset contract:

- minimal reusable dataset identity, class, split, sample, and deterministic preprocessing contract;
- validation behavior for split membership, labels, paths, tiny fixture image metadata, and class counts;
- committed tiny fixtures for deterministic contract tests.

Registered provisional core development dataset:

- CIFAR-10 acquired locally through `torchvision.datasets.CIFAR10`;
- source/version/provenance record documented;
- stable upstream-based sample identity defined;
- deterministic train/validation/test split policy defined;
- class-count report generated;
- raw and deterministic-preprocessing sample grids generated and reviewed;
- group/leakage and license/provenance limitations documented.

## Phase 1 Dataset State

Registered provisional core development dataset:

- dataset: CIFAR-10;
- VisionLab train: 45,000 images, 4,500 per class;
- VisionLab validation: 5,000 images, 500 per class;
- VisionLab test: 10,000 images, 1,000 per class;
- validation seed: `20260814`;
- validation policy: exact stratified selection from upstream train only;
- stable sample IDs: `cifar10-{upstream_partition}-{upstream_index:05d}`;
- preprocessing: RGB 32x32, `[0.0, 1.0]` value range, `(0.5, 0.5, 0.5)` mean/std, no augmentation.

Generated local data and inspection artifacts are ignored and not committed.

## Verification

Deterministic suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result:

```text
Ran 22 tests in 0.068s
OK
```

Visual verification:

- Codex inspected raw and preprocessed train/validation/test CIFAR-10 grids.
- Builder completed visual review and accepted Phase 1B.

## Requirement and Governance Impact

No material requirement change is recommended.

Approved phase split:

- Phase 1A established reusable contract and deterministic tiny-fixture validation.
- Phase 1B registered CIFAR-10 and completed visual data inspection.

Risk-register impact:

- CIFAR-10 group/leakage limitation is documented.
- License/provenance limitation is documented because license metadata differs by source route.

## Known Limitations

- CIFAR-10 does not expose rich correlated-group metadata through the standard torchvision interface.
- The validation split is deterministic and stratified, but not group-aware.
- Visual inspection covered deterministic grids, not all samples.
- The original University of Toronto CIFAR-10 page does not visibly state a license; UCI metadata currently lists `CC BY 4.0`.
- CIFAR-10 remains a provisional core development dataset and is not the applied-domain selection.
- No model, dataloader, training loop, checkpoint, evaluation metric, or inference behavior has been implemented.

## Readiness

Phase 1 satisfies its approved objective. The repository is ready for a separate Phase 2 concept briefing and implementation plan for the custom CNN and shape-safe forward path.
