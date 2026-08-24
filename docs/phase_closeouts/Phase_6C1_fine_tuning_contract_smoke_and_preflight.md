# Phase 6C-1 - Fine-Tuning Contract, Smoke, and Preflight

Status: Accepted and closed as Phase 6C-1.

Builder decision: Phase 6C-1 implementation and phase-check review were accepted. This subphase is closed with the fine-tuning contract, Phase 6B-2 checkpoint initialization, `layer4 + fc` trainability, optimizer-scope verification, mechanics smoke, and material-run preflight/timing evidence preserved. No material fine-tuning, official test evaluation, Phase 6C-2 work, or later-phase work is included here.

## Purpose

Phase 6C-1 established the fine-tuning path that can later test limited adaptation relative to the accepted Phase 6B-2 frozen-feature reference.

The approved boundary was:

- initialize from the accepted Phase 6B-2 best checkpoint, not from a fresh pretrained model;
- define a single fine-tuning mode, `finetune_layer4_head`;
- verify that only ResNet-18 `layer4` and `fc` are trainable;
- verify that the optimizer contains exactly the trainable parameters;
- run tiny CPU mechanics verification;
- run material CIFAR-10 split/preprocessing preflight and timing probe;
- stop before material fine-tuning and official test evaluation.

## Initialization Identity

Phase 6C-1 records the exact Phase 6B-2 checkpoint source:

- initialization source run: `phase6b2-cifar10-resnet18-frozen-feature-001`
- initialization checkpoint tag: `best`
- initialization checkpoint epoch: `4`
- initialization checkpoint path: `outputs/phase6b2-cifar10-resnet18-frozen-feature-001/checkpoints/best.pt`
- initialization checkpoint SHA-256: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`
- source weight identity: `ResNet18_Weights.IMAGENET1K_V1`

The implementation verifies the source run, checkpoint tag, epoch, and compatible source model identity before loading the checkpoint state into the Phase 6C fine-tuning model. This prevents silently substituting a fresh pretrained initialization for the accepted Phase 6B-2 best checkpoint.

## Model and Preprocessing Identity

The model identity remains:

- architecture: `torchvision.models.resnet18`
- weight identity: `ResNet18_Weights.IMAGENET1K_V1`
- checkpoint filename: `resnet18-f37072fd.pth`
- classifier: `Linear(512, 10)`
- output semantics: raw logits with shape `N x 10`
- preprocessing: `ResNet18_Weights.IMAGENET1K_V1.transforms()`

The ImageNet preprocessing contract remains unchanged from Phase 6B-1/6B-2:

- resize `256`
- crop `224`
- bilinear interpolation
- mean `[0.485, 0.456, 0.406]`
- std `[0.229, 0.224, 0.225]`
- model input `N x 3 x 224 x 224`

The preflight preprocessing probe confirms CIFAR-shaped `3 x 32 x 32` input is transformed to `3 x 224 x 224`.

## Fine-Tuning Scope

The exact fine-tuning mode is:

```text
finetune_layer4_head
```

Trainable parameters are limited to:

- `model.layer4.*`
- `model.fc.*`

Frozen parameter groups remain:

- `conv1`
- `bn1`
- `layer1`
- `layer2`
- `layer3`

Measured parameter counts:

- total parameters: `11,181,642`
- trainable parameters: `8,398,858`
- frozen parameters: `2,782,784`

## Optimizer Integrity

The Phase 6C-1 optimizer contract is:

- optimizer: Adam
- learning rate: `0.0001`
- weight decay: `0.0`
- scheduler: none
- optimizer parameters: exactly the parameters marked trainable by `finetune_layer4_head`

Preserved optimizer-scope evidence:

- optimizer matches trainable scope: `true`
- frozen parameters in optimizer parameter groups: `0`

Tiny mechanics smoke verifies:

- logits shape: `2 x 10`
- loss is finite
- frozen gradients are blocked
- frozen parameters remain unchanged
- trainable parameters update
- no material fine-tuning occurred
- no official test evaluation occurred

## Preflight and Runtime

Phase 6C-1 preflight preserves the registered CIFAR-10 split:

- train: `45,000`
- validation: `5,000`
- official test: `10,000`

The material-run configuration prepared for later approval is:

- run ID: `phase6c-cifar10-resnet18-layer4-finetune-001`
- seed: `20260820`
- batch size: `64`
- device: CPU
- epoch budget: `3`
- checkpoint selection: minimum validation loss
- augmentation: none
- differential learning-rate groups: none
- seed sweep: none
- hyperparameter search: none

Timing probe evidence:

- timed batches: `2`
- timed examples: `128`
- estimated epoch runtime: `1361.8469919972122` seconds
- estimated 3-epoch runtime: `4085.5409759916365` seconds, about `68.1` minutes
- selected batch size: `64`
- selected device: CPU

This runtime is reasonable enough for a separately approved CPU material run with a runtime guard, but it does not approve or launch that run.

## Experimental Boundary

Phase 6C-1 records the intervention as a training-regime change:

> Change the training regime from frozen-backbone/head-only training to `layer4 + fc` fine-tuning while preserving dataset, initialization, preprocessing, augmentation, seed, and evaluation protocol.

The Phase 6B-2 frozen-feature result remains a fixed reference point:

- Phase 6B-2 official test loss: `0.413686`
- Phase 6B-2 official test accuracy: `0.856100`

Phase 6C-1 does not use the Phase 6B-2 test result for training decisions. Any future material Phase 6C-2 run must select its checkpoint by validation loss only and evaluate the official test split once after best-checkpoint restoration.

## Explicit Non-Claims

Phase 6C-1 does not establish:

- material fine-tuning performance;
- an official test result;
- calibration;
- robustness or OOD behavior;
- seed variance;
- architecture-only superiority;
- inference behavior;
- diagnostics or applied-domain behavior.

## Preserved Artifacts

Phase 6C-1 preflight artifacts are preserved under ignored `outputs/phase6c-cifar10-resnet18-layer4-finetune-001-preflight/`:

- `run_contract.json`
- `preflight_report.json`
- `artifacts/mechanics_smoke.json`
- `artifacts/timing_probe.json`

## Tests and Verification

Focused Phase 6C-1 and transfer-related verification:

- `16` tests passed.

Canonical deterministic suite after closeout updates:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

Result: passed at `79` tests with `1` skipped.

## Worktree Note

`git status` continues to show deleted root files:

- `phase_briefing.md`
- `phase_check.md`

These were pre-existing worktree changes unrelated to Phase 6C-1 implementation, verification, or closeout. They remain untouched and excluded from the Phase 6C-1 closeout scope.

## Conclusions

Phase 6C-1 is closed and accepted.

The preserved outcome is:

- the Phase 6C fine-tuning path is initialized from the accepted Phase 6B-2 best checkpoint;
- `finetune_layer4_head` trainability is verified;
- optimizer membership is proven to contain exactly trainable `layer4 + fc` parameters;
- tiny mechanics smoke verifies frozen/trainable update behavior;
- material split/preprocessing/timing preflight is complete;
- no material fine-tuning or official test evaluation occurred.

Phase 6C-2 may be proposed next as a separate material fine-tuning run. It requires explicit builder approval before launch and must preserve the Phase 6B-2 result as a fixed reference point, not a tuning target.
