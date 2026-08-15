# Phase 1B - CIFAR-10 Registration and Visual Data Inspection

Date: 2026-08-14

Status: Complete.

Builder decision: Accepted on 2026-08-14 after builder visual review of the Phase 1B CIFAR-10 inspection artifacts.

## Objective

Register CIFAR-10 as VisionLab's provisional core development dataset with a reproducible split policy, stable sample identity, deterministic preprocessing metadata, class-count reporting, visual inspection artifacts, and explicit limitations.

## What Changed

- Added deterministic stratified validation split helper at `src/visionlab/data/splits.py`.
- Added Phase 1B CIFAR-10 registration and inspection script at `scripts/register_cifar10_phase1b.py`.
- Added deterministic split tests at `tests/test_data_splits.py`.
- Added CIFAR-10 registration and visual-inspection documentation at `docs/cifar10_phase1b_registration.md`.
- Updated current project status, dataset contract notes, phase catalog, builder journal, and risk register.

## Evidence

Automated deterministic verification:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result:

```text
Ran 22 tests in 0.068s
OK
```

Live CIFAR-10 registration command:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python scripts\register_cifar10_phase1b.py
```

Observed result:

```text
record_count: 60000
split_counts: train 45000, val 5000, test 10000
class_count_min_max_by_split: train 4500/4500, val 500/500, test 1000/1000
```

Generated ignored outputs:

- `outputs/phase1b_cifar10_inspection/cifar10_phase1b_manifest_summary.json`
- `outputs/phase1b_cifar10_inspection/cifar10_phase1b_class_counts.json`
- `outputs/phase1b_cifar10_inspection/cifar10_phase1b_manifest_records.jsonl`
- `outputs/phase1b_cifar10_inspection/raw_train_grid.png`
- `outputs/phase1b_cifar10_inspection/raw_val_grid.png`
- `outputs/phase1b_cifar10_inspection/raw_test_grid.png`
- `outputs/phase1b_cifar10_inspection/preprocessed_train_grid.png`
- `outputs/phase1b_cifar10_inspection/preprocessed_val_grid.png`
- `outputs/phase1b_cifar10_inspection/preprocessed_test_grid.png`

Generated data and outputs remain ignored and are not committed.

## Dataset Identity

Registered provisional core development dataset:

- dataset: CIFAR-10;
- source route: University of Toronto CIFAR-10 Python archive via `torchvision.datasets.CIFAR10`;
- source archive MD5: `c58f30108f718f92721af3b95e74349a`;
- citation request: Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.

License/provenance limitation:

- The original University of Toronto CIFAR-10 page requests citation but does not visibly state a license.
- UCI's current CIFAR-10 repository metadata lists license `CC BY 4.0`.
- VisionLab records the difference as a source-route provenance limitation rather than treating the UCI metadata as a statement from the original download page.

## Split Policy

Upstream partitions:

- upstream train: 50,000 images, 5,000 per class;
- upstream test: 10,000 images, 1,000 per class.

VisionLab splits:

- train: 45,000 upstream-train images;
- validation: 5,000 upstream-train images;
- test: 10,000 upstream-test images.

Validation policy:

- exact stratification by class;
- 500 validation samples per class;
- seed `20260814`;
- selection unit: upstream train source index;
- upstream test remains untouched.

## Stable Sample Identity

Sample IDs are derived from immutable upstream partition and original upstream index:

```text
cifar10-{upstream_partition}-{upstream_index:05d}
```

The VisionLab split assignment is recorded separately. Moving an upstream-train sample between train and validation in a future approved split policy would not change its sample identity.

## Preprocessing Profile

Deterministic Phase 1B inspection preprocessing:

- image size: 32x32;
- color mode: RGB;
- input value range: `[0.0, 1.0]`;
- normalization mean: `(0.5, 0.5, 0.5)`;
- normalization std: `(0.5, 0.5, 0.5)`;
- augmentation: none;
- deterministic: true.

The normalization mirrors the simple PyTorch CIFAR-10 tutorial convention and is not computed from test-set statistics. Future training phases may propose a separate train-only empirical normalization profile, but no such training configuration has been approved yet.

## Visual Inspection

Codex inspected all six generated grids. The builder also completed visual review and accepted Phase 1B.

Accepted findings:

- Raw and deterministic-preprocessing grids are consistent with the registered dataset and preprocessing behavior.
- All ten classes appear in the inspected train, validation, and test grids.
- The 32x32 images are readable enough for the provisional development-dataset role.
- Some examples are visually ambiguous or small, especially among animal classes; this is acceptable and should inform later failure analysis.
- No blank or corrupt samples were observed in the inspected grids.

Limitations:

- Visual review covered deterministic grids, not all 60,000 samples.
- Spot-check visual agreement is not proof of perfect label quality.
- Generated preprocessed PNG grids are viewable renderings of the preprocessing behavior, not persisted model inputs.

## Group and Leakage Limitations

- CIFAR-10 does not expose subject, photographer, web source, capture-session, duplicate, or near-duplicate group identifiers through the standard torchvision interface.
- Phase 1B preserves upstream partition and source index, but cannot prove correlated samples are absent across splits.
- Validation is selected only from upstream train, and upstream test remains untouched.
- Group-aware split validation is therefore limited and must not be overstated in later results.
- CIFAR-10 is web-image-derived and has broader Tiny Images lineage concerns. It is registered only as the provisional core development dataset, not as an applied-domain selection.

## Phase Check

Intended scope versus implementation:

- Phase 1B stayed within dataset acquisition/registration, deterministic split construction, validation, class-count reporting, preprocessing inspection, visual review, and documentation.
- No model implementation, training loop, augmentation experiment, material training run, or Phase 2 work was started.

Readiness:

- CIFAR-10 is ready to serve as the registered provisional core development dataset for Phase 2 planning.
- Phase 2 may use this dataset contract and split identity only after a separate Phase 2 concept briefing and approved implementation plan.
