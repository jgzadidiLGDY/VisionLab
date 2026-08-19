# Phase 4A - Baseline Experiment Plumbing and Smoke Verification

Status: Implemented as Phase 4A; superseded by accepted Phase 4 closeout.

Builder decision: Phase 4A approved as plumbing and smoke verification only. Phase 4A proves the experiment route, not the experiment result. Material CIFAR-10 baseline training, official test evaluation, tuning, Phase 5 augmentation work, and official baseline performance claims remain out of scope.

## Scope

Phase 4A implemented the bounded plumbing needed before the first custom-CNN CIFAR-10 baseline:

- registered CIFAR-10 dataset/split contract constants and split loader construction;
- explicit DataLoader reproducibility policy;
- integration with `CustomCNN` and the Phase 3 `fit` training engine;
- validation-based best-checkpoint selection through the existing `selection_metric="val_loss"` path;
- minimal Phase 4 evaluation artifacts, prediction records, per-class data, and confusion data;
- machine-readable history and curve data artifacts;
- a deliberately tiny end-to-end smoke workflow using synthetic CIFAR-shaped data.

## Implementation

New code:

- `src/visionlab/data/cifar10.py`
  - records Phase 1B CIFAR-10 class order, identity, preprocessing, split seed, and validation count;
  - builds registered train/validation/test split datasets;
  - constructs train, validation, test, validation-prediction, and test-prediction loaders;
  - enforces shuffled train batches and non-shuffled eval batches through `DataLoaderPolicy`;
  - verifies the material CIFAR-10 class order and expected `45,000` train, `5,000` validation, and `10,000` test counts before Phase 4B training.
- `src/visionlab/evaluation/classification.py`
  - records prediction rows with `sample_id`, `split`, `true_label`, `predicted_label`, confidence, correctness, and `source_id`;
  - writes summary JSON and prediction CSV artifacts;
  - writes history JSON and curve-data CSV artifacts.
- `src/visionlab/experiments/phase4a.py`
  - runs a tiny synthetic CIFAR-shaped smoke workflow;
  - trains for one epoch on non-material data;
  - restores the validation-selected best checkpoint before writing validation artifacts;
  - writes a run contract that explicitly states the smoke is pipeline evidence only and not official test evaluation.
- `scripts/run_phase4a_smoke.py`
  - local entry point for regenerating ignored Phase 4A smoke artifacts.

Dependency update:

- `pyproject.toml` now declares `torchvision>=0.28` because the registered CIFAR-10 acquisition path uses `torchvision.datasets.CIFAR10`.

## Verification

Automated verification:

```text
.\scripts\test.ps1
44 tests passed
```

Note: `44 tests passed` records the historical Phase 4A implementation check. Follow-up tests added before Phase 4B increased the deterministic suite to `46 tests passed`, as recorded in the final Phase 4 closeout.

New tests cover:

- registered split construction and upstream test isolation;
- material CIFAR-10 preflight count validation;
- deterministic train-loader shuffling under a fixed seed;
- non-shuffled prediction/evaluation loader behavior;
- rejection of an incompatible CIFAR-10 class order;
- prediction records, per-class data, and confusion data;
- machine-readable evaluation/history artifact writing;
- tiny Phase 4A end-to-end smoke execution without official test evaluation.

Smoke command:

```text
$env:PYTHONPATH='src'
python scripts/run_phase4a_smoke.py
```

Smoke artifacts were generated under ignored `outputs/phase4a_smoke/`.

Key generated files:

- `outputs/phase4a_smoke/run_contract.json`
- `outputs/phase4a_smoke/metadata.json`
- `outputs/phase4a_smoke/checkpoints/best.pt`
- `outputs/phase4a_smoke/checkpoints/terminal.pt`
- `outputs/phase4a_smoke/artifacts/history.json`
- `outputs/phase4a_smoke/artifacts/curve_data.csv`
- `outputs/phase4a_smoke/artifacts/val_smoke_summary.json`
- `outputs/phase4a_smoke/artifacts/val_smoke_predictions.csv`

## Interpretation

The smoke run completed and exercised the baseline route end to end:

- split construction;
- train/validation/test loader construction;
- validation-based checkpoint selection;
- checkpoint writing;
- validation prediction artifact writing from the restored best checkpoint;
- history and curve-data artifact writing.

Any smoke loss, accuracy, predictions, or per-class values are pipeline evidence only. They are not VisionLab baseline results and should not be compared, tuned, or reported as model performance.

The smoke workflow constructed a test loader and recorded its count, but did not run official test evaluation. The official CIFAR-10 test evaluation remains part of Phase 4B after validation-based checkpoint selection.

## Phase 4A Check

Intended versus implemented scope:

- Implemented the approved plumbing and smoke verification scope.
- Did not run a material CIFAR-10 baseline.
- Did not tune optimizer, batch size, epoch budget, scheduler, seed, or device based on smoke results.
- Did not begin Phase 5 or broader Phase 7 evaluation harness work.

Data and split integrity:

- Train/validation split uses the Phase 1B deterministic stratified upstream-train policy.
- Test samples are sourced from the upstream test partition only.
- Evaluation loaders are non-shuffled by policy.

Model and checkpoint identity:

- Smoke route uses `CustomCNN` and the existing Phase 3 checkpoint helpers.
- Best checkpoint selection uses validation loss.
- Prediction/evaluation artifacts are written only after restoring the checkpoint tagged `best`.

Artifact completeness:

- Smoke artifacts include run contract, metadata, best/terminal checkpoints, history JSON, curve CSV, validation summary JSON, and validation prediction CSV.

Documentation drift:

- README, phase catalog, and builder journal have been updated to show Phase 4A as implemented and awaiting builder review.

## Remaining Limitations

- The default deterministic tests use tiny in-memory data and do not require a full CIFAR-10 download.
- Phase 4A does not prove material CIFAR-10 runtime, convergence, accuracy, or generalization.
- The current `TrainingConfig` still accepts CPU only; GPU material training would require a bounded approved device-contract update.
- Evaluation remains intentionally minimal and should not be mistaken for the later Phase 7 evaluation/calibration harness.
- No official test evaluation exists yet.

## Proposed Phase 4B Material-Run Configuration

This is a proposal for builder approval, not an executed run.

- Run ID: `phase4b-cifar10-custom-cnn-baseline-001`
- Dataset: Phase 1B registered CIFAR-10
- Splits:
  - train: upstream train excluding deterministic validation indices
  - validation: 5,000 upstream-train samples, 500 per class, seed `20260814`
  - test: upstream test partition, untouched until final evaluation
- Model: `CustomCNNConfig(input_channels=3, image_size=(32, 32), num_classes=10, feature_channels=(32, 64, 128), dropout=0.0)`
- Preprocessing: Phase 1B deterministic profile, `[0, 1]` tensor range normalized with mean/std `(0.5, 0.5, 0.5)`
- Augmentation: none
- Device: CPU for local reproducibility unless builder approves a bounded GPU device-contract change
- Seed: `20260818`
- DataLoader policy:
  - batch size: `128`
  - train shuffle: `true`
  - validation/test shuffle: `false`
  - num workers: `0` for first material run reproducibility
  - seeded train `torch.Generator`
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `0.0`
- Scheduler: none
- Epoch budget: `10`
- Loss: `torch.nn.CrossEntropyLoss`
- Checkpoint-selection rule: lowest validation loss
- Checkpoints: preserve best and terminal
- Test evaluation: run once after best-checkpoint selection

Expected runtime and compute:

- Expected local CPU runtime should be treated as uncertain until a short timed dry run is performed; a first estimate is tens of minutes for 10 epochs on CPU.
- If CPU runtime is too slow for review cadence, propose and approve a small GPU device-contract update before training.

Stop and failure conditions:

- stop if CIFAR-10 is missing and download approval is needed;
- stop if smoke-to-material config expansion changes more than device/runtime plumbing;
- stop on non-finite training or validation loss;
- stop if train/validation/test counts or class mapping do not match Phase 1B registration;
- stop if best-checkpoint or metadata artifacts cannot be written;
- stop if final validation/test artifacts cannot be generated from the restored best checkpoint;
- do not evaluate the test split until after checkpoint selection.

Artifact preservation plan:

- `metadata.json`
- `run_contract.json`
- training history JSON
- curve-data CSV
- material preflight report
- best checkpoint
- terminal checkpoint
- validation summary JSON
- validation prediction CSV
- test summary JSON after best-checkpoint restoration
- test prediction CSV after best-checkpoint restoration
- baseline report documenting setup, metrics, limitations, and smoke-versus-material distinction

## Readiness

Phase 4A implementation was accepted as part of the completed Phase 4 review.

Phase 4B has since completed and is summarized in the final Phase 4 closeout.
