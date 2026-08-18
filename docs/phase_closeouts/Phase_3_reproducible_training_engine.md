# Phase 3 - Reproducible Training Engine

Date: 2026-08-17

Status: Complete; accepted after phase-check review.

Builder decision: Phase 3 implementation and phase-check report accepted. Phase 3 is closed with a strict boundary around CPU synthetic/tiny-data verification, reproducibility-focused metadata, bounded checkpoint compatibility, validation/no-grad behavior, optional minimal scheduler support, and inspectable non-finite-loss failure handling. Material CIFAR-10 baseline training and model-performance claims remain out of scope until Phase 4.

## Objective

Build and verify reusable training infrastructure without optimizing headline performance.

Learning objective:

- explain the difference between training mechanics and model-performance evidence;
- verify controlled optimization, validation, checkpointing, and restore behavior;
- preserve enough run identity to support later material training review;
- keep test data and baseline interpretation out of this infrastructure phase.

## What Phase 3 Built

Training package:

- `src/visionlab/training/config.py`;
- `src/visionlab/training/reproducibility.py`;
- `src/visionlab/training/checkpoints.py`;
- `src/visionlab/training/engine.py`;
- `src/visionlab/training/__init__.py`.

Training configuration and metadata:

- `TrainingConfig` records run ID, seed, max epochs, CPU device, optimizer, optional scheduler, checkpoint flags, and selection metric;
- `OptimizerConfig` supports bounded Adam and SGD configuration;
- `SchedulerConfig` supports minimal optional `StepLR` configuration;
- `EpochMetrics` records epoch-level train/validation loss, train/validation accuracy, and learning rate;
- `TrainingRunMetadata` records run ID, configuration, seed, environment summary, status, epoch history, checkpoint references, stop reason, and failure reason.

Engine behavior:

- `train_one_epoch` runs model training mode, computes loss and accuracy, backpropagates, steps the optimizer, and rejects non-finite training loss;
- `validate` runs evaluation mode under `torch.no_grad()`, computes loss and accuracy, rejects non-finite validation loss, and restores training mode when validation was called from a training model;
- `fit` applies the seed controls, builds or accepts optimizer/scheduler objects, records epoch history, optionally writes checkpoints and `metadata.json`, and returns inspectable completed or failed run metadata.

Checkpoint behavior:

- checkpoint payloads include checkpoint version, tag, run ID, seed, epoch, model identity, optimizer class, scheduler class, model state, optimizer state, scheduler state, and metrics;
- restore rejects unsupported checkpoint versions, mismatched run IDs, incompatible model identity, incompatible optimizer class, and incompatible scheduler class;
- restore is intentionally bounded and does not introduce a generalized experiment registry.

Reproducibility controls:

- Phase 3 applies Python `random.seed`, `torch.manual_seed`, and `torch.use_deterministic_algorithms(True)`;
- environment summaries record Python version, platform, PyTorch version, requested device, CUDA availability, and deterministic-algorithm state;
- this is a CPU smoke reproducibility control, not a guarantee of bit-for-bit reproducibility across all machines, PyTorch versions, devices, or future DataLoader worker configurations.

## Verification

Deterministic suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result:

```text
Ran 38 tests
OK
```

New Phase 3 tests cover:

- trainable parameters update during a training epoch;
- validation uses evaluation/no-grad semantics and does not mutate parameters or gradients in the tested path;
- deterministic synthetic in-memory data can be deliberately overfit with final training accuracy `1.0` and final training loss below `0.02`;
- epoch metadata records learning-rate history when a scheduler is present;
- best and terminal checkpoint references are produced in a temporary run directory;
- compatible checkpoint restore reproduces model outputs in the tested path;
- incompatible model identity is rejected;
- non-finite training loss leaves failed metadata with an inspectable failure reason.

## Scope Discipline

Phase 3 did not add:

- material CIFAR-10 training;
- baseline model-performance results;
- test-set evaluation;
- pretrained models or transfer learning;
- augmentation experiments;
- evaluation reports, calibration, robustness, OOD analysis, or failure galleries;
- inference surface;
- applied-domain behavior;
- broad experiment-tracking framework.

## Phase Check Summary

Overall status from phase-check review:

- ready with small follow-ups.

Findings:

- training and validation semantics match the approved Phase 3 boundary;
- validation uses `eval()` and `torch.no_grad()` and is tested for no parameter/gradient mutation;
- checkpoint compatibility is useful but bounded;
- run metadata is reproducibility-focused and does not expand into later evaluation/reporting infrastructure;
- deterministic tiny-overfit evidence verifies training mechanics only and is not a CIFAR-10 performance claim;
- documentation correctly states that no material CIFAR-10 baseline, test-set evaluation, pretrained model, inference surface, or applied-domain behavior exists.

Blocking follow-ups:

- none for Phase 3 closure.

Non-blocking Phase 4 entry considerations:

- define the material-run DataLoader shuffle and worker seed policy explicitly;
- define the validation-based checkpoint-selection metric explicitly;
- continue isolating the test split from model selection.

## Requirement and Governance Impact

No material requirement change is recommended.

Phase 3 clarifies the initial checkpoint and training-run metadata contract for later baseline planning, while preserving the training approval boundary for Phase 4.

## Known Limitations

- The engine has been verified only with CPU synthetic/tiny data.
- No CIFAR-10 training result exists.
- No model-performance, generalization, calibration, robustness, OOD, or inference claim exists.
- Reproducibility controls are sufficient for the current CPU smoke path but must be revisited before material training with shuffled data, multiple workers, GPU, or external compute.
- Scheduler support is intentionally minimal and currently limited to optional `StepLR`.
- Checkpoint compatibility is bounded to the current model/config and optimizer/scheduler identity checks.

## Readiness

Phase 3 is closed and accepted.

The repository is ready for a separate Phase 4 concept briefing and implementation plan for the Custom CNN Baseline Experiment. Phase 4 must preserve dataset and split identity, use validation data for checkpoint selection, keep the test split out of model selection, and obtain approval before any material training run.
