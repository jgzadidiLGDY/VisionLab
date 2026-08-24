# Phase 6B-2 - Material Frozen-Feature Transfer Run

Status: Accepted and closed as Phase 6B-2.

Builder decision: Phase 6B-2 implementation, material run, artifact inspection, and phase-check review were accepted. This subphase is closed as a single frozen-feature transfer-learning reference run. Phase 6C fine-tuning has not started and must be planned separately.

## Purpose

Phase 6B-2 ran one approved material CIFAR-10 frozen-feature transfer-learning experiment using the Phase 6A/6B-1 ResNet-18 contract.

The phase objective was to establish a fixed pretrained frozen-feature reference point for later comparison, not to tune for the best possible pretrained result.

## Exact Experiment Configuration

- Run ID: `phase6b2-cifar10-resnet18-frozen-feature-001`
- Dataset: Phase 1B registered CIFAR-10
- Split: `45,000` train / `5,000` validation / `10,000` official test
- Model: `torchvision.models.resnet18`
- Weights: `ResNet18_Weights.IMAGENET1K_V1`
- Cached checkpoint: `resnet18-f37072fd.pth`
- Classifier: `Linear(512, 10)`
- Freeze mode: `frozen_backbone_head_only`
- Total parameters: `11,181,642`
- Trainable parameters: `5,130`
- Frozen parameters: `11,176,512`
- Preprocessing: `ResNet18_Weights.IMAGENET1K_V1.transforms()`
- Input path: CIFAR-10 `32 x 32` RGB input transformed to ImageNet-preprocessed `224 x 224` model input
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `0.0`
- Scheduler: none
- Batch size: `64`
- Epoch budget: `5`
- Seed: `20260820`
- Device: CPU
- Checkpoint selection: minimum validation loss
- Augmentation: none
- Fine-tuning: none
- Partial unfreezing: none
- Differential learning-rate groups: none

## Pretrained-Weight Identity

The selected pretrained model identity is exactly:

- architecture: `torchvision.models.resnet18`
- weight enum: `ResNet18_Weights.IMAGENET1K_V1`
- checkpoint file: `resnet18-f37072fd.pth`

The Phase 6B-2 preflight verified that the checkpoint was already present in the Torch cache and did not attempt a download during the material run.

## Checkpoint and Results

The best checkpoint was selected by minimum validation loss.

- Best checkpoint: epoch `4`
- Restored-best validation loss: `0.3983015718460083`
- Restored-best validation accuracy: `0.8646`
- Official test loss: `0.41368580923080445`
- Official test accuracy: `0.8561`

The official test split was evaluated once, after restoring the selected best checkpoint.

## Comparison to Phase 4B

Phase 4B remains the accepted custom-CNN comparison reference:

- Baseline run ID: `phase4b-cifar10-custom-cnn-baseline-001`
- Phase 4B official test loss: `1.024515`
- Phase 4B official test accuracy: `0.635900`

Observed Phase 6B-2 deltas versus Phase 4B:

- Official test loss delta: `-0.610829`
- Official test accuracy delta: `+0.220200`

These are single-run comparison observations only.

## Asymmetry Limitations

The Phase 6B-2 comparison is intentionally asymmetric:

- ResNet-18 uses ImageNet source pretraining; the Phase 4B CustomCNN was trained from scratch.
- ResNet-18 uses a much larger parameter scale.
- ResNet-18 uses `224 x 224` ImageNet-preprocessed inputs; Phase 4B uses CIFAR-native `32 x 32` preprocessing.
- ResNet-18 trained only the replacement classifier head; the Phase 4B CustomCNN trained end to end.

The Phase 6B-2 result must not be attributed to architecture alone.

## Evidence Boundary

Phase 6B-2 establishes one fixed frozen-feature transfer-learning reference result.

It does not establish:

- fine-tuning performance;
- calibration;
- robustness or OOD behavior;
- seed variance;
- architecture-only superiority.

The Phase 6B-2 result should remain a fixed reference point for later work, just as Phase 4B remained the reference for Phase 5. Phase 6C should not automatically inherit `0.8561` test accuracy as a target or begin tuning around it.

## Preserved Artifacts

Material run artifacts are preserved under ignored `outputs/phase6b2-cifar10-resnet18-frozen-feature-001/`:

- `preflight_report.json`
- `run_contract.json`
- `metadata.json`
- `artifacts/preprocessing_contract.json`
- `artifacts/history.json`
- `artifacts/curve_data.csv`
- `checkpoints/best.pt`
- `checkpoints/terminal.pt`
- `artifacts/val_summary.json`
- `artifacts/val_predictions.csv`
- `artifacts/test_summary.json`
- `artifacts/test_predictions.csv`
- `phase6b2_comparison_report.md`
- `phase6b2_result.json`

## Verification

Phase 6B-2 material run completed successfully before the approved runtime guard.

Canonical deterministic suite after closeout updates:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result: passed at `73` tests with `1` skipped.

## Worktree Note

`git status` continues to show deleted root files:

- `phase_briefing.md`
- `phase_check.md`

These were pre-existing worktree changes unrelated to Phase 6B-2 implementation, verification, or closeout. They remain untouched and excluded from the Phase 6B-2 closeout scope.

## Conclusions

Phase 6B-2 is closed and accepted.

The project now has:

- the accepted Phase 4B custom-CNN baseline reference;
- the accepted Phase 5B augmentation comparison result;
- the accepted Phase 6B-2 single-run pretrained frozen-feature transfer-learning reference.

Phase 6C may be planned next as a separate fine-tuning experiment with its own approval boundary. It should treat the Phase 6B-2 result as a fixed reference, not as an optimization target.
