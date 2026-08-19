# Phase 4 - Custom CNN Baseline Experiment

Status: Complete; accepted.

Builder decision: Phase 4 is accepted based on the completed phase-check report. Phase 4A established baseline experiment plumbing and smoke verification. Phase 4B ran the approved material custom-CNN CIFAR-10 baseline once. Phase 5 remains out of scope until separately briefed, planned, and approved.

## Scope

Phase 4 established VisionLab's first official custom CNN baseline experiment.

Approved boundaries:

- use the Phase 1B registered CIFAR-10 dataset, class order, split policy, and preprocessing profile;
- use the Phase 2 custom CNN;
- use the Phase 3 training engine and checkpoint helpers;
- run the approved fixed Phase 4B configuration once;
- select the best checkpoint by validation loss only;
- restore the selected best checkpoint before final validation and official test evaluation;
- evaluate the official test split once after checkpoint selection;
- preserve run identity, configuration, preflight evidence, metrics, checkpoints, prediction artifacts, and a cautious baseline report.

Out of scope:

- tuning or restarting to improve the result;
- using test results for model selection;
- augmentation or Phase 5 generalization controls;
- pretrained/transfer-learning comparison;
- calibration, robustness, OOD, diagnostics, inference, or applied-domain claims.

## Implementation Summary

Phase 4A added the experiment route:

- `src/visionlab/data/cifar10.py` for CIFAR-10 identity, split construction, DataLoader policy, preprocessing, and material preflight validation;
- `src/visionlab/evaluation/classification.py` for minimal Phase 4 classification summaries, prediction records, per-class counts, confusion data, and machine-readable history/curve artifacts;
- `src/visionlab/experiments/phase4a.py` for the tiny non-material smoke route and restored-best-checkpoint evaluation helper;
- `scripts/run_phase4a_smoke.py` for regenerating Phase 4A smoke artifacts.

Phase 4B added and executed the material run:

- `src/visionlab/experiments/phase4b.py` for the approved material-run orchestration and baseline report writer;
- `scripts/run_phase4b_material_baseline.py` for the material-run entry point.

## Approved Phase 4B Configuration

- Run ID: `phase4b-cifar10-custom-cnn-baseline-001`
- Device: CPU
- Model: `CustomCNNConfig(input_channels=3, image_size=(32, 32), num_classes=10, feature_channels=(32, 64, 128), dropout=0.0)`
- Dataset: Phase 1B registered CIFAR-10
- Preprocessing: RGB `[0, 1]` tensors normalized with mean/std `(0.5, 0.5, 0.5)`
- Augmentation: none
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `0.0`
- Scheduler: none
- Batch size: `128`
- Epoch budget: `10`
- Seed: `20260818`
- DataLoader policy: seeded train shuffle, unshuffled validation/test, `num_workers=0`, `drop_last=false`
- Checkpoint selection: minimum validation loss

The approved fixed configuration was used without tuning or restarts.

## Dataset and Split Integrity

Material preflight passed before training began and was preserved as `preflight_report.json`.

Verified contract:

- class order: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`;
- train: `45,000`;
- validation: `5,000`;
- test: `10,000`;
- validation indices: `5,000`;
- test partition: upstream `test`;
- preflight status: `passed`.

## Checkpoint and Evaluation Sequence

The material run followed the approved sequence:

```text
train
  -> select best checkpoint by minimum validation loss
  -> restore selected best checkpoint
  -> generate final validation artifacts
  -> evaluate official test split once
  -> generate official test artifacts
```

Best checkpoint:

- selected epoch: `10`;
- selection metric: minimum validation loss;
- selected checkpoint: `outputs/phase4b-cifar10-custom-cnn-baseline-001/checkpoints/best.pt`;
- terminal checkpoint: `outputs/phase4b-cifar10-custom-cnn-baseline-001/checkpoints/terminal.pt`.

For this run, the selected best checkpoint and terminal checkpoint both correspond to epoch 10 because validation loss improved through the final epoch.

## Results

Restored-best validation:

- validation loss: `1.017542`;
- validation accuracy: `0.630200`.

Official test evaluation:

- test loss: `1.024515`;
- test accuracy: `0.635900`.

The official test split was evaluated once after checkpoint selection and was not used for model selection.

## Basic Class and Error Observations

Test per-class accuracy:

- airplane: `0.743`
- automobile: `0.854`
- bird: `0.412`
- cat: `0.200`
- deer: `0.612`
- dog: `0.640`
- frog: `0.697`
- horse: `0.781`
- ship: `0.724`
- truck: `0.696`

Initial observations:

- strongest test class: automobile;
- weakest test class: cat;
- visible confusion patterns in the preserved confusion matrix include cat-to-dog, truck-to-automobile, and deer-to-horse errors.

These are basic Phase 4 observations only. They are not a full Phase 9 failure analysis.

## Artifact Inventory

Material run artifacts are preserved under ignored `outputs/phase4b-cifar10-custom-cnn-baseline-001/`.

Required artifacts:

- `outputs/phase4b-cifar10-custom-cnn-baseline-001/preflight_report.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/run_contract.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/metadata.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/history.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/curve_data.csv`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/checkpoints/best.pt`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/checkpoints/terminal.pt`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/phase4b_result.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/val_summary.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/val_predictions.csv`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/test_summary.json`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/artifacts/test_predictions.csv`
- `outputs/phase4b-cifar10-custom-cnn-baseline-001/baseline_report.md`

Supporting Phase 4A smoke artifacts remain under ignored `outputs/phase4a_smoke/` and are pipeline evidence only.

## Verification

Deterministic suite:

```text
.\scripts\test.ps1
46 tests passed
```

The suite covers:

- registered split construction and upstream test isolation;
- material CIFAR-10 preflight count validation;
- deterministic train-loader shuffling under a fixed seed;
- non-shuffled prediction/evaluation loader behavior;
- rejection of incompatible CIFAR-10 class order;
- prediction records, per-class data, and confusion data;
- machine-readable evaluation/history artifact writing;
- tiny Phase 4A end-to-end smoke execution without official test evaluation.

## Phase Check

The accepted Phase 4 phase-check found Phase 4 ready with small non-blocking documentation and inventory follow-ups.

Resolved follow-ups:

- top README status now reflects that Phase 4 has completed and is being closed;
- the Phase 4A verification note distinguishes its original historical `44 tests` result from the current `46-test` suite;
- this final Phase 4 closeout contains the complete required artifact inventory;
- unrelated deleted root `phase_briefing.md` and `phase_check.md` worktree entries are noted below and excluded from Phase 4 scope.

## Worktree Note

`git status` continues to show deleted root files:

- `phase_briefing.md`
- `phase_check.md`

These were pre-existing worktree changes unrelated to Phase 4 implementation, training, evaluation, or closeout. They are excluded from the Phase 4 commit scope unless the builder gives separate direction.

## Conclusions

Phase 4 produced VisionLab's first official custom CNN baseline:

- the approved fixed configuration was used without tuning or restarts;
- Phase 1B dataset/class/split preflight passed;
- checkpoint selection used minimum validation loss only;
- the selected best checkpoint was explicitly restored before final validation and official test evaluation;
- the official test split was evaluated once after checkpoint selection;
- restored-best official test loss was `1.024515`;
- restored-best official test accuracy was `0.635900`.

This is a single-run custom CNN baseline. It is not:

- a tuned best result;
- an estimate of run-to-run variance;
- a calibration result;
- a robustness result;
- an OOD result;
- a transfer-learning comparison;
- a broad generalization claim;
- a failure-analysis conclusion.

## Readiness

Phase 4 is closed and accepted.

The Phase 4 baseline artifacts and configuration should be preserved as the reference point for later controlled comparisons.

The next bounded step is a separate Phase 5 concept briefing and implementation plan for Augmentation and Generalization Controls. Phase 5 should not begin until that briefing and plan are reviewed and approved.
