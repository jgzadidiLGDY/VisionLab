# Phase 2 - Custom CNN and Shape-Safe Forward Path

Date: 2026-08-15

Status: Complete; awaiting builder review.

Builder decision: Phase 2 implementation plan approved with a strict boundary around the custom CNN, forward/loss smoke path, concise shape inspection, parameter counting, invalid-input/configuration tests, deterministic verification, and Phase 2 check. Phase 3 training infrastructure and material training remain out of scope.

## Objective

Implement the foundational custom vision model before building a trainer.

Learning objective:

- explain how a CIFAR-10-shaped input batch moves through convolutional blocks, downsampling, pooling, flattening, and a linear classification head;
- distinguish raw logits from probabilities;
- verify shape and configuration failures before training is introduced.

## What Phase 2 Built

Model package:

- `src/visionlab/models/custom_cnn.py`;
- `src/visionlab/models/__init__.py`.

Custom model:

- `CustomCNNConfig` records input channels, image size, class count, feature channels, and dropout;
- `CustomCNN` is a compact PyTorch `nn.Module`;
- `count_parameters` reports total and trainable parameter counts.

Architecture:

```text
input N x 3 x 32 x 32
  -> Conv2d(3, 32, 3x3, padding=1) + ReLU + MaxPool2d
  -> Conv2d(32, 64, 3x3, padding=1) + ReLU + MaxPool2d
  -> Conv2d(64, 128, 3x3, padding=1) + ReLU + MaxPool2d
  -> AdaptiveAvgPool2d(1 x 1)
  -> Flatten
  -> Linear(128, 10)
  -> logits N x 10
```

Concise intermediate shapes for `batch_size=1`:

```text
input:     (1, 3, 32, 32)
block1:    (1, 32, 16, 16)
block2:    (1, 64, 8, 8)
block3:    (1, 128, 4, 4)
pooled:    (1, 128, 1, 1)
flattened: (1, 128)
logits:    (1, 10)
```

Parameter counts:

```text
total:     94,538
trainable: 94,538
```

## Logits and Loss Semantics

`CustomCNN.forward` returns raw logits only. It does not apply softmax.

The Phase 2 smoke test verifies that `torch.nn.CrossEntropyLoss` accepts the model output and integer class labels. Probability conversion remains a later evaluation or inference concern, not model-forward behavior.

## Invalid-Input and Configuration Behavior

Forward validation rejects:

- tensors that are not rank 4;
- channel counts that do not match `config.input_channels`;
- spatial sizes that do not match `config.image_size`.

Configuration validation rejects:

- non-positive input channels;
- non-positive image dimensions;
- fewer than two classes;
- empty or non-positive feature-channel values;
- dropout outside `[0.0, 1.0)`.

Explicit dtype validation was intentionally not added because it would mostly duplicate PyTorch behavior at this phase boundary.

## Verification

Deterministic suite:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result:

```text
Ran 32 tests in 1.069s
OK
```

New tests cover:

- valid CIFAR-10-shaped forward pass;
- cross-entropy loss smoke path;
- concise intermediate-shape inspection;
- total and trainable parameter counts;
- eval/no-grad CPU forward path;
- invalid rank, channel count, and spatial size;
- invalid model configuration.

## Dependency and Command Clarification

Phase 2 turns PyTorch from a T1 feasibility probe into an implemented model dependency. `pyproject.toml` now declares `torch>=2.13`.

The system `python` on this machine does not have `torch` installed. The verified local CPU environment is `.\.venv\Scripts\python.exe` with `torch 2.13.0+cpu`. `scripts/test.ps1` now uses that local venv interpreter when present and falls back to `python` otherwise, while still setting `PYTHONPATH=src`.

## Scope Discipline

Phase 2 did not add:

- trainer, optimizer, scheduler, or checkpoint logic;
- model registry;
- dataset loading;
- pretrained models or transfer learning;
- evaluation harness or inference surface;
- Phase 3 infrastructure;
- material training.

## Requirement and Governance Impact

No material requirement change is recommended.

The only implementation clarification is that PyTorch is now an explicit package dependency for model work. This is consistent with the existing project specification and the approved Phase 2 plan.

## Known Limitations

- The model has not been trained.
- No accuracy, calibration, robustness, OOD, or failure-analysis claims exist.
- The architecture is a compact learning baseline, not an optimized CIFAR-10 model.
- Dtype validation is left to PyTorch errors unless a later phase exposes a practical contract need.
- The default local deterministic test command depends on the ignored local `.venv` for PyTorch unless the active system Python also has `torch`.

## Phase Check

Intended scope versus implementation:

- matched the approved custom-CNN and forward-contract boundary;
- did not begin Phase 3 or material training.

Concept and learning objective:

- the builder can inspect the data-to-logits path and explain the shape changes through each block.

Tests and verification:

- deterministic suite passes with 32 tests;
- new tests cover happy path, loss semantics, configuration failures, and invalid input failures.

Data and split integrity:

- Phase 2 references the CIFAR-10 shape and class-count contract but does not load or alter dataset splits.

Model identity:

- custom model configuration and parameter count are inspectable;
- no pretrained backbone replaced the learning objective.

Experiment controls:

- no experiment or training run occurred, so no metric or checkpoint claims are made.

Documentation drift:

- README, phase catalog, risk register, and builder journal were updated to reflect Phase 2 completion and PyTorch dependency clarification.

Readiness:

- Phase 2 is ready for builder review.
- After acceptance, the repository is ready for a separate Phase 3 concept briefing and implementation plan for a reproducible training engine.
