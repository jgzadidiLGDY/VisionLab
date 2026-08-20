# Phase 5A - Augmentation Profile and Smoke Verification

Status: Accepted and closed as Phase 5A.

Builder decision: Phase 5A implementation and phase-check review were accepted. The phase is closed with a strict boundary around augmentation-profile infrastructure, smoke verification, visual inspection, and explicit Phase 5B proposal preparation only. Material CIFAR-10 training, official test evaluation, augmentation-policy search, dropout, weight decay, scheduler changes, and Phase 5B remain out of scope until separately reviewed and approved.

## Purpose

Phase 5A establishes an explicit train-time augmentation layer before any augmented material baseline run.

The phase verifies that VisionLab can:

- preserve the Phase 4 no-augmentation configuration as a versioned control profile;
- define candidate augmentation profiles with stable IDs, versions, and exact parameters;
- keep augmentation strictly training-only;
- preserve deterministic validation/test preprocessing;
- generate reproducible visual inspection artifacts from fixed sample IDs;
- verify shape and normalized range contracts through smoke tests.

## Implemented Scope

Code:

- `src/visionlab/data/augmentation.py`
  - `AugmentationProfile`;
  - `TransformSpec`;
  - `PHASE4_NO_AUGMENTATION_PROFILE`;
  - `PHASE5A_CANDIDATE_FLIP_CROP_PROFILE`;
  - `profile_registry_dict`;
  - `get_augmentation_profile`;
  - `apply_augmentation_profile`.
- `src/visionlab/data/cifar10.py`
  - separates unit tensor conversion from normalization;
  - supports an optional train-only augmentation profile;
  - rejects augmentation profiles attached to validation or test splits;
  - preserves `to_normalized_tensor` for the existing deterministic preprocessing path.
- `scripts/inspect_phase5a_augmentations.py`
  - writes the machine-readable profile registry;
  - selects one fixed registered train sample per CIFAR-10 class;
  - writes the visual augmentation grid;
  - writes a Markdown inspection note with sample IDs, seed, artifact paths, observations, and recommendation.
  - requires Pillow for grid rendering; the dependency is now declared in `pyproject.toml`.
- `tests/test_phase5_augmentation.py`
  - verifies profile registry identity and parameters;
  - verifies unknown-profile rejection;
  - verifies no-augmentation control behavior;
  - verifies candidate profile shape and unit-range contract;
  - verifies augmentation is train-only;
  - verifies validation/test remain deterministic.

## Implemented Profiles

Control profile:

- Profile ID: `phase4-control-no-augmentation`
- Version: `1.0`
- Train-only: `true`
- Transforms: none
- Purpose: preserve the Phase 4 no-augmentation configuration as a versioned control.

Candidate profile:

- Profile ID: `phase5a-candidate-horizontal-flip-random-crop`
- Version: `1.0`
- Train-only: `true`
- Transform 1: `random_horizontal_flip`
  - probability: `0.5`
- Transform 2: `random_crop_with_padding`
  - output size: `[32, 32]`
  - padding: `4`
  - padding mode: `constant`
  - fill: `0.0`

The candidate remains pending builder review for Phase 5B. Phase 5A does not prove that this profile improves generalization.

## Visual Inspection Artifacts

Generated under ignored local output directory:

- `outputs/phase5a_augmentation_inspection/augmentation_profile_registry.json`
- `outputs/phase5a_augmentation_inspection/phase5a_candidate_augmentation_grid.png`
- `outputs/phase5a_augmentation_inspection/phase5a_visual_inspection_note.md`

Inspection seed:

- `20260819`

Fixed sample IDs:

- `cifar10-train-00029`: `airplane`
- `cifar10-train-00004`: `automobile`
- `cifar10-train-00006`: `bird`
- `cifar10-train-00009`: `cat`
- `cifar10-train-00003`: `deer`
- `cifar10-train-00027`: `dog`
- `cifar10-train-00000`: `frog`
- `cifar10-train-00007`: `horse`
- `cifar10-train-00008`: `ship`
- `cifar10-train-00001`: `truck`

Recorded observations:

- all rendered normalized outputs have shape `(3, 32, 32)`;
- normalized rendered outputs stayed within observed range `-1.000000` to `1.000000`;
- the control column visually matches the raw sample content after display conversion;
- candidate columns show small translations and occasional horizontal flips while retaining visible class evidence in the fixed grid;
- some candidate cells show visible black padded margins at crop edges, expected from zero padding;
- CIFAR-10 images remain low-resolution and several classes are visually ambiguous, so the inspection supports plausibility rather than proving label preservation for every sample.

Approval recommendation:

- Codex recommends treating `phase5a-candidate-horizontal-flip-random-crop` version `1.0` as appropriate for a single controlled Phase 5B material-run proposal, pending builder review of the visual grid.
- This is a recommendation only. The builder has not yet approved Phase 5B material training.

## Train/Validation/Test Transform Behavior

- Train split: may receive an approved augmentation profile through `train_augmentation_profile`.
- Validation split: deterministic Phase 4 preprocessing only.
- Test split: deterministic Phase 4 preprocessing only.
- Attaching an augmentation profile to validation or test raises a `ValueError`.

## Smoke-Test Results

Targeted Phase 5A tests:

```text
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m unittest tests.test_phase5_augmentation
......
Ran 6 tests in 0.005s
OK
```

Canonical deterministic suite:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
....................................................
Ran 52 tests in 1.377s
OK
```

Inspection artifact generation:

```text
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\inspect_phase5a_augmentations.py
```

Result: completed successfully.

## Proposed Phase 5B Material-Run Configuration

This is proposed for review only. Do not run until the builder explicitly approves Phase 5B.

- Phase: `5B`
- Purpose: single controlled custom-CNN augmentation comparison against the accepted Phase 4 baseline.
- Dataset: Phase 1B registered CIFAR-10.
- Split policy: unchanged `45,000` train, `5,000` validation, `10,000` official test.
- Model: `CustomCNNConfig(input_channels=3, image_size=(32, 32), num_classes=10, feature_channels=(32, 64, 128), dropout=0.0)`.
- Augmentation profile: `phase5a-candidate-horizontal-flip-random-crop` version `1.0`.
- Validation/test preprocessing: deterministic Phase 4 preprocessing only.
- Optimizer: Adam.
- Learning rate: `0.001`.
- Weight decay: `0.0`.
- Scheduler: none.
- Batch size: `128`.
- Epoch budget: `10`.
- Seed: `20260818` for closest comparability to the Phase 4B single-run baseline.
- DataLoader policy: seeded train shuffle, unshuffled validation/test, `num_workers=0`, `drop_last=false`.
- Checkpoint selection: minimum validation loss.
- Test use: evaluate official test split once only after restoring the selected best checkpoint.

Proposed artifacts:

- preflight report;
- run contract including dataset, model, training config, seed, DataLoader policy, and augmentation profile registry reference;
- profile registry snapshot;
- training metadata/history;
- best and terminal checkpoints;
- validation and official test summaries after best-checkpoint restore;
- validation and official test prediction records;
- history/curve artifacts;
- comparison report against `phase4b-cifar10-custom-cnn-baseline-001`;
- baseline-decision note recording whether the augmented profile is adopted, rejected, or left inconclusive.

## Explicit Non-Scope

Phase 5A did not:

- run material CIFAR-10 training;
- evaluate the official test split;
- tune augmentation parameters;
- search over multiple augmentation policies;
- add dropout;
- add weight decay;
- change the scheduler;
- introduce transfer learning, calibration, robustness, OOD, diagnostics, inference, or applied-domain behavior.

## Handoff

Phase 5A is accepted and closed.

Builder review that informed closeout inspected:

- the profile registry JSON;
- the augmentation visual grid;
- the separated observations and recommendation in the Markdown inspection note;
- test results and transform-split behavior;
- the proposed Phase 5B material-run configuration.

Phase 5B must not begin until the builder approves the exact material-run configuration.
