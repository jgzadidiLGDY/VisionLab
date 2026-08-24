# Phase 6C-2 Closeout - Material Layer4 Fine-Tuning Run

Status: Complete; accepted by builder.

## Phase Objective

Phase 6C-2 executed one approved material fine-tuning run to test the single intervention defined in Phase 6C-1: continue from the accepted Phase 6B-2 frozen-feature best checkpoint and unfreeze only ResNet-18 `layer4` plus the existing CIFAR-10 classifier head.

This phase did not search for the best fine-tuning recipe. It produced one fixed-reference material result for the approved `finetune_layer4_head` configuration.

## Exact Experiment Configuration

- Run ID: `phase6c-cifar10-resnet18-layer4-finetune-001`
- Dataset: Phase 1B registered CIFAR-10
- Split: `45,000` train / `5,000` validation / `10,000` official test
- Model: `torchvision.models.resnet18`
- Pretrained weights identity: `ResNet18_Weights.IMAGENET1K_V1`
- Cached checkpoint: `resnet18-f37072fd.pth`
- Initialization source run: `phase6b2-cifar10-resnet18-frozen-feature-001`
- Initialization checkpoint: `best`, epoch `4`
- Initialization checkpoint SHA-256: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`
- Classifier: existing `Linear(512, 10)` CIFAR-10 head from the Phase 6B-2 best checkpoint
- Fine-tuning mode: `finetune_layer4_head`
- Trainable scope: ResNet-18 `layer4` + `fc`
- Frozen scope: `conv1`, `bn1`, `layer1`, `layer2`, `layer3`
- Parameter counts: total `11,181,642`, trainable `8,398,858`, frozen `2,782,784`
- Preprocessing: `ResNet18_Weights.IMAGENET1K_V1.transforms()`
- Input path: CIFAR-10 `32 x 32` RGB -> ImageNet-preprocessed `224 x 224` tensors
- Optimizer: Adam
- Learning rate: `0.0001`
- Weight decay: `0.0`
- Scheduler: none
- Batch size: `64`
- Device: CPU
- Seed: `20260820`
- Epoch budget: `3`
- Augmentation: none
- Checkpoint selection: minimum validation loss

## Material Result

Phase 6C-2 completed the approved 3-epoch CPU fine-tuning run. The best checkpoint was selected using validation loss only.

- Best checkpoint: epoch `2`
- Restored-best validation loss: `0.246512`
- Restored-best validation accuracy: `0.925800`
- Official test loss: `0.272485`
- Official test accuracy: `0.914700`
- Official test evaluation count: one
- Official test timing: after restoring the selected best checkpoint

The terminal epoch was epoch `3`, with validation loss `0.258056` and validation accuracy `0.925400`; it was not selected because epoch `2` had the lower validation loss.

## Comparison References

Phase 6C-2 should be read against three distinct references:

1. **Phase 6B-2 frozen-feature reference**: `phase6b2-cifar10-resnet18-frozen-feature-001` used the same ResNet-18/ImageNet preprocessing path but trained only the classifier head. Its official test accuracy was `0.856100`.
2. **Phase 6C-2 fine-tuning intervention**: `phase6c-cifar10-resnet18-layer4-finetune-001` initialized from the Phase 6B-2 best checkpoint and trained `layer4 + fc`. Its official test accuracy was `0.914700`.
3. **Phase 4B CustomCNN historical comparison**: `phase4b-cifar10-custom-cnn-baseline-001` remains the from-scratch custom-CNN reference with official test accuracy `0.635900`.

Relative to the Phase 6B-2 frozen-feature reference, Phase 6C-2 observed:

- Test accuracy delta: `+0.058600` (`+5.86` percentage points)
- Test loss delta: `-0.141201`

Relative to Phase 4B, the comparison remains asymmetric because Phase 6C-2 uses ImageNet pretraining, ResNet-18 model scale, `224 x 224` ImageNet preprocessing, and a fine-tuning continuation from Phase 6B-2 rather than a from-scratch custom CNN trained on `32 x 32` CIFAR-10 tensors.

## Metadata Correction Note

The Phase 6C-2 phase check identified a stale metadata/documentation issue in `outputs/phase6c-cifar10-resnet18-layer4-finetune-001/run_contract.json`: the file had reused the Phase 6C-1 preflight contract labels and still said `phase: 6C-1` with preflight-only scope even though the material result, comparison report, checkpoints, metadata, and prediction artifacts correctly identified the completed Phase 6C-2 run.

That top-level contract metadata was corrected after the material run. The correction did not rerun training, did not perform another official test evaluation, did not change the experimental configuration, and did not generate a new experimental result. The corrected contract now records Phase 6C-2, `material_fine_tuning: true`, and one official test evaluation performed after best-checkpoint restoration.

## Artifact Inventory

All material artifacts are preserved under `outputs/phase6c-cifar10-resnet18-layer4-finetune-001/`:

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
- `phase6c_comparison_report.md`
- `phase6c_result.json`

The validation prediction artifact contains `5,000` validation predictions plus a header row. The official test prediction artifact contains `10,000` test predictions plus a header row.

## Verification

Phase 6C-2 verification after the material run and metadata correction:

- Focused Phase 6C-related tests before closeout: passed when run after the material result (`18` tests).
- Canonical deterministic suite after closeout metadata/documentation updates: passed with `81` tests and `1` skipped.

The phase-check review found the material evidence valid and identified the stale contract label as a metadata/documentation issue rather than an experimental-result issue.

## Boundaries Preserved

Phase 6C-2 did not perform:

- additional training reruns;
- seed sweeps;
- augmentation changes;
- hyperparameter search;
- additional unfreezing strategies;
- differential learning-rate groups;
- calibration;
- robustness/OOD evaluation;
- diagnostics;
- inference work;
- applied-domain work;
- Phase 7 work.

## Limitations

This result is a single material fine-tuning run. It does not establish:

- seed or run-to-run variance;
- optimal unfreezing depth;
- optimal hyperparameters;
- architecture-only superiority;
- calibration;
- robustness/OOD behavior;
- generalization beyond the evaluated CIFAR-10 experiment.

The Phase 6B-2 result remains a fixed frozen-feature reference, not a target that Phase 6C tuned against. Checkpoint selection in Phase 6C-2 used validation loss only.

## Closeout Decision

Phase 6C-2 is formally closed and accepted. Phase 6 is complete through the approved transfer-learning and fine-tuning sequence: Phase 6A contract, Phase 6B-1 pretrained smoke, Phase 6B-2 frozen-feature material reference, Phase 6C-1 fine-tuning preflight, and Phase 6C-2 material layer4 fine-tuning result.

The next project phase is Phase 7 planning for evaluation harness and calibration. Phase 7 has not begun and requires a separate plan and approval boundary.
