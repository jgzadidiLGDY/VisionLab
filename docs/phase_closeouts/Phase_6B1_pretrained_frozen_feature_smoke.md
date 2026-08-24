# Phase 6B-1 - Pretrained Frozen-Feature Smoke

Status: Accepted and closed as Phase 6B-1.

Builder decision: Phase 6B-1 implementation and phase-check review were accepted. This subphase is closed with pretrained-weight availability, actual ResNet-18 preprocessing application, and cached-weight frozen-feature mechanics verified. No material CIFAR-10 training, validation performance result, official test evaluation, fine-tuning, Phase 6B-2 material run, or Phase 6C work is included here.

## Purpose

Phase 6B-1 verified that the Phase 6A transfer-model contract can run with the actual approved pretrained weights and the actual Torchvision preprocessing path before any material frozen-feature training is proposed.

The approved boundary was:

- download only the exact approved pretrained checkpoint;
- verify cache availability for that exact checkpoint;
- load the pretrained weights;
- apply the selected Torchvision preprocessing transform;
- run a tiny CPU pretrained frozen-feature smoke path;
- preserve smoke artifacts and evidence boundaries;
- stop before material CIFAR-10 training.

## Selected Model and Weights

- Architecture: `torchvision.models.resnet18`
- Weight identity: `ResNet18_Weights.IMAGENET1K_V1`
- Checkpoint filename: `resnet18-f37072fd.pth`
- Classifier replacement: `Linear(512, 10)`
- Output semantics: raw logits with shape `N x 10`
- Freeze mode: `frozen_backbone_head_only`
- Comparison reference: `phase4b-cifar10-custom-cnn-baseline-001`

The exact checkpoint download was approved by the builder and completed for Phase 6B-1. Cache availability was verified after download.

## Preprocessing Verification

Phase 6B-1 verified actual preprocessing application through:

```text
ResNet18_Weights.IMAGENET1K_V1.transforms()
```

The preserved smoke contract records:

- raw tiny input shape: `2 x 3 x 32 x 32`
- preprocessed model input shape: `2 x 3 x 224 x 224`
- resize: `256`
- crop: `224`
- interpolation: bilinear
- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

This preprocessing remains separate from the Phase 4B CustomCNN preprocessing contract.

## Parameter Counts

Measured from the instantiated ResNet-18 model with the 10-class classifier head:

- total parameters: `11,181,642`
- trainable parameters: `5,130`
- frozen parameters: `11,176,512`

Only the replacement classifier head is trainable. The pretrained backbone remains frozen.

## Smoke Evidence

Phase 6B-1 executed a tiny pretrained frozen-feature smoke path using synthetic CIFAR-shaped data only.

Preserved evidence:

- `pretrained_weights_loaded: true`
- logits shape verified as `2 x 10`
- finite cross-entropy loss
- frozen backbone parameters did not receive trainable gradients
- frozen backbone parameters remained unchanged after an optimizer step
- classifier-head parameters updated after the optimizer step
- checkpoint/config identity was preserved
- Phase 4B baseline reference was recorded and left unchanged

This is pretrained frozen-feature mechanics smoke evidence only. It is not material CIFAR-10 training evidence, validation performance evidence, official test evidence, or a pretrained-versus-custom performance comparison.

## Preserved Boundaries

Phase 6B-1 did not:

- run material CIFAR-10 training;
- produce a validation performance result;
- perform official test evaluation;
- implement fine-tuning;
- implement partial backbone unfreezing;
- add differential learning-rate parameter groups;
- run Phase 6B-2 material training;
- begin Phase 6C work;
- modify the Phase 4B baseline.

The Phase 4B baseline `phase4b-cifar10-custom-cnn-baseline-001` remains unchanged and remains the comparison reference for later transfer-learning work.

## Artifact and Code Inventory

Phase 6B-1 smoke artifacts are preserved under ignored `outputs/phase6b1-resnet18-pretrained-frozen-smoke/`:

- `outputs/phase6b1-resnet18-pretrained-frozen-smoke/run_contract.json`
- `outputs/phase6b1-resnet18-pretrained-frozen-smoke/artifacts/metadata.json`
- `outputs/phase6b1-resnet18-pretrained-frozen-smoke/artifacts/preprocessing_contract.json`
- `outputs/phase6b1-resnet18-pretrained-frozen-smoke/checkpoints/pretrained_smoke.pt`
- `outputs/phase6b1-resnet18-pretrained-frozen-smoke/phase6b1_smoke_result.json`

Phase 6B-1 implementation files:

- `src/visionlab/data/transfer_preprocessing.py`
- `src/visionlab/data/__init__.py`
- `src/visionlab/experiments/phase6b.py`
- `src/visionlab/training/checkpoints.py`

Phase 6B-1 tests:

- `tests/test_phase6b_preprocessing.py`
- `tests/test_phase6b_smoke.py`

## Verification

Focused Phase 6B-1 and transfer-related verification:

- `23` tests passed;
- `1` test skipped.

Canonical deterministic suite after closeout updates:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result: passed at `71` tests with `1` skipped.

## Worktree Note

`git status` continues to show deleted root files:

- `phase_briefing.md`
- `phase_check.md`

These were pre-existing worktree changes unrelated to Phase 6B-1 implementation, verification, or closeout. They remain untouched and excluded from the Phase 6B-1 closeout scope.

## Conclusions

Phase 6B-1 is closed and accepted.

The preserved outcome is:

- the exact approved ResNet-18 checkpoint `resnet18-f37072fd.pth` was downloaded and verified in cache;
- `ResNet18_Weights.IMAGENET1K_V1` was loaded with `pretrained_weights_loaded: true`;
- actual Torchvision preprocessing application was verified;
- synthetic CIFAR-shaped `32 x 32` input was transformed to `224 x 224` model input;
- frozen-backbone/head-only mechanics were verified with actual pretrained weights;
- no material training, validation result, official test evaluation, fine-tuning, or later-phase work occurred.

Phase 6B-2 has not begun and requires a separate approved material-run plan before any material pretrained training.
