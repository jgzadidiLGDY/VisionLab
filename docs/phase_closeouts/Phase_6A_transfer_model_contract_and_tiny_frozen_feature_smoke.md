# Phase 6A - Transfer Model Contract and Tiny Frozen-Feature Smoke

Status: Accepted and closed as Phase 6A.

Builder decision: Phase 6A implementation and phase-check review were accepted. The phase is closed with the preserved Phase 4B custom-CNN baseline retained as the comparison reference. No pretrained-weight download, material CIFAR-10 training, official test evaluation, fine-tuning, additional backbone, or Phase 6B work is included here.

## Purpose

Phase 6A established the pretrained-model contract and tiny non-material frozen-feature mechanics path needed before any material transfer-learning run.

The approved boundary was:

- select exactly one pretrained backbone and weight identity;
- preserve a separate pretrained preprocessing contract;
- replace the pretrained classifier with a 10-class CIFAR-10 head;
- verify frozen-backbone/head-only mechanics on synthetic in-memory data;
- probe pretrained-weight cache availability without downloading;
- distinguish random-initialized mechanics evidence from actual pretrained-weight evidence;
- preserve the accepted Phase 4B baseline unchanged.

## Selected Transfer Model Identity

- Architecture: `torchvision.models.resnet18`
- Weight identity: `ResNet18_Weights.IMAGENET1K_V1`
- Expected checkpoint filename: `resnet18-f37072fd.pth`
- Classifier replacement: `Linear(512, 10)`
- Output semantics: raw logits with shape `N x 10`
- Freeze mode: `frozen_backbone_head_only`
- Comparison reference: `phase4b-cifar10-custom-cnn-baseline-001`

`ResNet18_Weights.DEFAULT` is not used for this phase. The explicit enum identity and checkpoint filename are the reproducible Phase 6A weight identity in the current Torchvision environment.

## Preprocessing Contract

Phase 6A records a separate ImageNet preprocessing contract for the selected ResNet-18 weights:

- model input: `N x 3 x 224 x 224`
- resize: `256`
- crop: `224`
- interpolation: bilinear
- mean: `[0.485, 0.456, 0.406]`
- std: `[0.229, 0.224, 0.225]`

This contract is intentionally separate from the Phase 4B CustomCNN preprocessing contract. Any later custom-CNN versus ResNet-18 comparison is asymmetric because the ResNet-18 route differs by ImageNet source pretraining, architecture scale, parameter count, input resolution, and preprocessing.

## Parameter Counts

Measured from the instantiated Phase 6A ResNet-18 model with the 10-class classifier head:

- total parameters: `11,181,642`
- trainable parameters: `5,130`
- frozen parameters: `11,176,512`

Only the replacement classifier head is trainable in Phase 6A. The backbone remains frozen.

## Cache Probe and Weight Availability

Phase 6A includes a non-download cache probe for the exact selected weights.

Observed implementation-time cache state:

- expected checkpoint filename: `resnet18-f37072fd.pth`
- expected local cache path: `C:\Users\jgzad\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth`
- cached locally: `false`
- download attempted: `false`
- pretrained weights loaded during executed smoke: `false`

The actual pretrained frozen-feature smoke path exists, but it was not executed because the exact selected weights were unavailable locally and no download approval was part of Phase 6A.

## Mechanics Smoke Evidence

The executed tiny CPU smoke path used synthetic in-memory data and `weights=None`.

Recorded evidence:

- forward pass produced raw logits with shape `N x 10`;
- cross-entropy loss was finite;
- classifier head replacement was verified as `Linear(512, 10)`;
- frozen parameters did not receive trainable gradients;
- frozen parameters remained unchanged after an optimizer step;
- classifier-head parameters updated after the optimizer step;
- checkpoint/config identity was preserved and incompatible transfer configuration was rejected.

This is model-mechanics evidence only. It is not transfer-learning evidence, not pretrained performance evidence, and not evidence about CIFAR-10 generalization.

## Preserved Boundaries

Phase 6A did not:

- download pretrained weights;
- run material CIFAR-10 training;
- perform official test evaluation;
- implement fine-tuning;
- implement partial backbone unfreezing;
- add differential learning-rate parameter groups;
- add another pretrained backbone;
- modify the accepted Phase 4B baseline;
- begin Phase 6B.

The Phase 4B baseline `phase4b-cifar10-custom-cnn-baseline-001` remains unchanged and remains the comparison reference for later transfer-learning work.

## Artifact and Code Inventory

Phase 6A implementation files:

- `src/visionlab/models/transfer.py`
- `src/visionlab/models/__init__.py`
- `src/visionlab/experiments/phase6a.py`

Phase 6A tests:

- `tests/test_transfer_model.py`
- `tests/test_phase6a_smoke.py`

Documentation updated during closeout:

- `README.md`
- `docs/phase_catalog.md`
- `AI_native_builder_journal.md`
- `docs/phase_closeouts/Phase_6A_transfer_model_contract_and_tiny_frozen_feature_smoke.md`

## Verification

Focused Phase 6A verification before closeout:

- `12` focused tests passed.

Canonical deterministic suite after closeout updates:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result: passed at `66` tests.

## Phase-Check Follow-Ups

Future Phase 6B entry requirements:

- resolve pretrained-weight availability or obtain explicit download approval for `resnet18-f37072fd.pth`;
- verify the actual preprocessing application path before material pretrained training;
- prepare a separate material frozen-feature run plan with dataset/split identity, compute estimate, checkpoint policy, artifact inventory, and official-test-use boundary.

These are not Phase 6A blockers. They are required before material frozen-feature training.

## Worktree Note

`git status` continues to show deleted root files:

- `phase_briefing.md`
- `phase_check.md`

These were pre-existing worktree changes unrelated to Phase 6A implementation, verification, or closeout. They remain untouched and excluded from the Phase 6A closeout scope.

## Conclusions

Phase 6A is closed and accepted.

The preserved outcome is:

- one exact transfer backbone and weight identity were selected: `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`;
- the separate ImageNet preprocessing contract was recorded;
- the classifier head was replaced with `Linear(512, 10)`;
- raw-logit output shape `N x 10` was verified;
- measured parameter counts matched the approved contract;
- frozen-backbone/head-only mechanics were verified on synthetic data;
- pretrained weights were not cached and were not downloaded;
- executed smoke used `weights=None` and is mechanics evidence only;
- the pretrained frozen-feature smoke path exists but was not executed;
- the Phase 4B baseline remains unchanged and remains the comparison reference.

Phase 6B has not begun and requires a separate approved plan before any material pretrained training.
