# AI-Native Builder Journal

VisionLab is being built through an incremental AI-native learning and development workflow.

The purpose of this file is to preserve a visible record of how the project, its requirements, its experiments, and the builder’s understanding of computer vision evolve over time.

The README explains what VisionLab is intended to become and reports its current public status.

This journal explains how VisionLab is being built by an AI-native builder: a development partnership in which the human builder retains learning, product, experiment, and architecture ownership; Codex serves as an integrated tutor, implementation partner, debugger, and reviewer; and ChatGPT supports specification, conceptual discussion, and broader review where useful.

---

## Current Journal State

Status: **Phase 7 complete / accepted; Phase 8 not started**

The T0 bootstrap closeout has been accepted. T1 has been accepted with final builder visual review of generated foundation artifacts recorded as the remaining manual review condition. Phase 1A and Phase 1B have been accepted, Phase 1 is complete, Phase 3 is complete and accepted, Phase 4A is implemented, the approved Phase 4B material baseline run has completed, Phase 4 is accepted and closed, Phase 5 is complete through accepted Phase 5A and Phase 5B closeout, Phase 6A is complete and accepted, Phase 6B-1 is complete and accepted, Phase 6B-2 is complete and accepted, Phase 6C-1 is complete and accepted, Phase 6C-2 is complete and accepted, Phase 7 is complete and accepted, and Phase 8 has not started.

At this point, the main project artifacts are planning documents, governance records, a local smoke path, T1 concept notes, environment probes, tiny foundation exercises, a development-dataset candidate comparison, Phase 1A dataset-contract scaffolding, accepted Phase 1B CIFAR-10 registration artifacts, the Phase 2 custom CNN forward-contract implementation, the Phase 3 bounded training-engine implementation, Phase 4A baseline experiment plumbing with tiny smoke artifacts, the Phase 4B single-run custom CNN baseline artifacts, accepted Phase 5A augmentation profile/inspection artifacts, the accepted Phase 5B single-run augmentation comparison artifacts, the Phase 6A ResNet-18 transfer-model contract with tiny mechanics smoke coverage, the accepted Phase 6B-1 pretrained frozen-feature smoke artifacts, the accepted Phase 6B-2 material frozen-feature transfer-learning run artifacts, the accepted Phase 6C-1 fine-tuning contract/preflight artifacts, and the accepted Phase 6C-2 material layer4 fine-tuning artifacts. The project identity, fundamentals-to-applied progression, AI-native workflow, requirements, phase boundaries, evaluation principles, and closure tiers have been drafted.

Current boundary note: CIFAR-10 is the registered provisional core development dataset, with ignored local data under `data/` and ignored inspection outputs under `outputs/`. Phase 2 implemented a compact custom CNN, shape-safe forward path, logits/loss smoke test, parameter counting, and concise intermediate-shape inspection. Phase 3 implemented and closed CPU-only training loops, validation, optional scheduler stepping and learning-rate history, checkpoint save/restore with compatibility checks, minimal reproducibility/environment metadata, and non-finite loss handling against synthetic/tiny verification data. Phase 4A implemented loader, evaluation-artifact, prediction-record, history, and tiny smoke plumbing. Phase 4B produced a single-run custom CNN CIFAR-10 baseline result: restored-best official test loss `1.024515` and test accuracy `0.635900`. Phase 5A added explicit train-only augmentation profiles, visual inspection artifacts, smoke tests, and declared the Pillow dependency needed for augmentation-grid rendering. Phase 5B then ran one approved augmentation comparison using `phase5a-candidate-horizontal-flip-random-crop` version `1.0`, yielding restored-best validation loss `1.055734`, validation accuracy `0.620600`, official test loss `1.056135`, and official test accuracy `0.630800`. Relative to the Phase 4B baseline, that single run regressed by `+0.031620` test loss and `-0.005100` test accuracy, so the candidate profile is not adopted as the new baseline. Phase 6A added an explicit `torchvision.models.resnet18` transfer-model contract bound to `ResNet18_Weights.IMAGENET1K_V1`, a separate ImageNet preprocessing contract, frozen-backbone/head-only parameter inspection, a non-download cache probe, and tiny random-initialized mechanics smoke coverage. Phase 6B-1 then downloaded the exact approved checkpoint `resnet18-f37072fd.pth`, verified cache availability, loaded the selected weights with `pretrained_weights_loaded: true`, applied `ResNet18_Weights.IMAGENET1K_V1.transforms()`, and completed tiny pretrained frozen-feature mechanics smoke. Phase 6B-2 ran one approved material frozen-feature transfer-learning experiment, `phase6b2-cifar10-resnet18-frozen-feature-001`, using the same exact pretrained identity, selected epoch `4` by minimum validation loss, and produced restored-best validation loss `0.398302`, validation accuracy `0.864600`, official test loss `0.413686`, and official test accuracy `0.856100`. Relative to Phase 4B, this is a single-run test accuracy delta of `+0.220200` and loss delta of `-0.610829`. Phase 6C-1 then implemented the fine-tuning contract/preflight path initialized from the accepted Phase 6B-2 best checkpoint at epoch `4`, checkpoint SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`, with `finetune_layer4_head`, trainable `layer4 + fc`, optimizer scope matching only trainable parameters, and a 3-epoch CPU timing estimate of about `68.1` minutes at batch size `64`. The Phase 6B-2 result is a fixed reference point and should not become an automatic target for Phase 6C. Phase 6C-2 then ran the approved material fine-tuning experiment `phase6c-cifar10-resnet18-layer4-finetune-001`, selected epoch `2` by minimum validation loss, and produced restored-best validation loss `0.246512`, validation accuracy `0.925800`, official test loss `0.272485`, and official test accuracy `0.914700`. Relative to Phase 6B-2, this is a single-run test accuracy delta of `+0.058600`, or `+5.86` percentage points. The stale Phase 6C-1/preflight label in the material-run `run_contract.json` was corrected after phase check without rerunning training or changing experimental evidence. Phase 7 evaluation/calibration work is complete and accepted. Phase 8 has not started. No inference surface, robustness, OOD, diagnostics, or applied-domain behavior exists yet.

The applied domain remains intentionally undecided. It will be selected later through the implementation-stage feasibility gate rather than assumed at project start.

This journal should not imply that planned models, commands, experiments, checkpoints, results, or application behavior have already been implemented or validated unless they are explicitly tied to preserved artifacts. CIFAR-10 is registered only as the provisional core development dataset, not as an applied-domain decision.

Update this section as the project advances. Preserve historical detail in dated entries and phase closeouts rather than accumulating an outdated narrative here.

---
## 2026-08-24 - Phase 7 Evaluation Harness and Calibration Closeout

### Context

The builder approved Phase 7 implementation after reviewing the plan and amendments requiring fixed-checkpoint evaluation only, hard sample identity alignment, explicit checkpoint identity, separate validation/test semantics, no retraining, and no automatic closeout. After the formal phase check, the builder requested closeout documentation and status updates.

### AI Contribution

Codex implemented the metric/calibration harness, enriched prediction artifacts, lightweight SVG/CSV visual artifacts, and the fixed-checkpoint Phase 7 runner. Codex ran the deterministic suite and the Phase 7 evaluation pass over the accepted Phase 4B, Phase 6B-2, and Phase 6C-2 best checkpoints, then created the formal Phase 7 closeout.

### Evidence

- Phase check: `docs/phase_checks/Phase_7_evaluation_harness_and_calibration_check.md`.
- Phase closeout: `docs/phase_closeouts/Phase_7_evaluation_harness_and_calibration.md`.
- Phase 7 artifacts: `outputs/phase7-evaluation-harness-and-calibration/`.
- Sample alignment passed for `5,000` validation and `10,000` official test samples across all three compared runs.
- Canonical deterministic suite passed with `94` tests and `1` skipped after closeout documentation updates.

### Project Impact

VisionLab now has an accepted fixed-checkpoint evaluation and calibration layer. Phase 7 preserves the exact 10-bin ECE configuration and fixed-checkpoint results. Robustness/OOD, failure analysis, diagnostics, inference, and applied-domain selection remain later boundaries.

### Next Boundary

Phase 8 has not started. The next boundary is a separate Phase 8 concept briefing and implementation plan for robustness and OOD evaluation.

---

## 2026-08-22 - Phase 6C-2 Material Layer4 Fine-Tuning Run Closeout

### Context

The builder approved the Phase 6C-2 material run after Phase 6C-1 established the fine-tuning contract, initialization identity, optimizer scope, and CPU timing preflight.

### Concept or Hypothesis

Phase 6C-2 tested one specific intervention relative to Phase 6B-2: continue from the accepted frozen-feature best checkpoint and train only ResNet-18 `layer4 + fc`, while preserving dataset, preprocessing, seed, batch size, optimizer family, no augmentation, and validation-loss checkpoint selection.

### AI Contribution

Codex resumed the approved CPU material run after runner issues, preserved the fixed configuration, restored the selected best checkpoint before final validation and official test evaluation, inspected artifacts, ran verification, performed the phase-check review, corrected the stale Phase 6C-1/preflight top-level label in `run_contract.json`, and created the formal Phase 6C-2 closeout.

### Evidence

- Run ID: `phase6c-cifar10-resnet18-layer4-finetune-001`.
- Initialization checkpoint: Phase 6B-2 `best`, epoch `4`, SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- Fine-tuning mode: `finetune_layer4_head`, trainable `layer4 + fc`.
- Best checkpoint: epoch `2`.
- Restored-best validation loss `0.246512`, validation accuracy `0.925800`.
- Official test loss `0.272485`, official test accuracy `0.914700`, evaluated once after best-checkpoint restoration.
- Canonical deterministic suite passed with `81` tests and `1` skipped after closeout updates.

### Project Impact

Phase 6 now has both a frozen-feature transfer reference and a bounded fine-tuning reference. Phase 6C-2 improved over the Phase 6B-2 fixed reference by `+0.058600` test accuracy in this single run, but it does not establish seed variance, optimal unfreezing depth, optimal hyperparameters, architecture-only superiority, calibration, robustness/OOD behavior, or broader generalization.

### Next Boundary

Phase 7 has not started. The next step should be a separate Phase 7 concept briefing and implementation plan for the evaluation harness and calibration.

---

## 2026-08-13 - T0 Bootstrap Implementation

### Context

The builder approved T0 - Project Bootstrap and Baseline Capture, including `git init` and correction of the project specification path to `docs/project_specs.md`.

### AI Contribution

Codex inspected the repository, identified that the folder was not yet a Git repository, found that documentation referenced `docs/project_specs.md` while the specification lived at the repository root, proposed a bounded T0 plan, and implemented the approved bootstrap changes.

### Builder Review and Decision

The builder approved the T0 plan and requested that PyTorch and Python-version decisions remain deferred to T1. The observed Python 3.14.5 environment is recorded as a compatibility risk rather than resolved in T0.

### Evidence

- Git was initialized locally.
- The project specification now resides at `docs/project_specs.md`.
- T0 governance files exist at `docs/risk_register.md`, `docs/requirement_change_log.md`, and `docs/phase_catalog.md`.
- A minimal import smoke path exists under `src/visionlab/`.
- The T0 closeout draft is at `docs/phase_closeouts/T0_project_bootstrap_and_baseline_capture.md`.

### Project Impact

T0 establishes repository wiring and governance, not computer-vision behavior. Dataset selection, applied-domain selection, PyTorch installation, model implementation, training, evaluation, diagnostics, and inference remain out of scope until later approved phases.

### Next Boundary

T0 was accepted by the builder on 2026-08-13. T1 has not started and requires a separate concept briefing and approval boundary.

---

## 2026-08-14 - T1 Foundations and Feasibility Triage Implementation

### Context

The builder approved T1 with a boundary around vision foundations, Python/PyTorch feasibility, tiny tensor/image/convolution exercises, deterministic test invocation clarity, and low-friction development-dataset comparison.

### Concept or Hypothesis

T1 tested whether VisionLab can proceed toward Phase 1 with a practical local CPU development path and a suitable provisional development dataset, without beginning dataset registration or material training.

### AI Contribution

Codex added T1 concept notes, dependency-light convolution and PGM helpers, an environment/device probe, deterministic tests, local scripts, compute feasibility documentation, a dataset candidate comparison, and a T1 closeout draft.

### Evidence

- `scripts/test.ps1` passed 8 unittest checks.
- Base Python `3.14.5` had no ML dependencies installed.
- An ignored `.venv` installed `torch 2.13.0+cpu` and `torchvision 0.28.0+cpu`.
- CPU tensor and tiny PyTorch `conv2d` probes passed.
- CUDA was not available locally.
- Synthetic PGM artifacts were generated under ignored `outputs/t1_foundations/`.

### Learning

T1 reduced the Python 3.14/PyTorch risk for local CPU smoke work while keeping material training and GPU workflow approval separate. It also clarified why RGB data is preferable for the first core development dataset despite Fashion-MNIST being simpler.

### Project Impact

Recommended builder-review clarifications are recorded in `docs/requirement_change_log.md`. CIFAR-10 is recommended as the provisional Phase 1 development dataset, but no dataset has been downloaded, registered, split, or trained on.

### Builder Review and Decision

The builder accepted T1 on 2026-08-14, approved the recorded T1 requirement clarifications, and kept Phase 1 explicitly out of scope for this step. Final builder visual review of the generated T1 foundation artifacts remains recorded as the manual review condition.

Approved clarifications:

- Python 3.14.5 with the verified local `.venv` CPU PyTorch stack is the current tested development/smoke path; `requires-python` is not narrowed at this time.
- `scripts/test.ps1` is the canonical deterministic local test command.
- CIFAR-10 is the provisional Phase 1 development-dataset candidate only, pending dataset-contract, provenance/license, split-policy, validation, and visual-inspection work.

### Next Boundary

After final builder visual review of the T1 foundation artifacts, Phase 1 may be planned around dataset contract and visual data inspection. Phase 1 should still establish source, license, class mapping, validation split policy, sample grids, and data limitations before material training.

---

## 2026-08-15 - Phase 2 Custom CNN and Shape-Safe Forward Path

### Context

The builder approved Phase 2 after a concept briefing and implementation plan, with a strict boundary around the custom CNN, forward/loss smoke path, concise intermediate-shape inspection, parameter counting, invalid-input/configuration tests, deterministic verification, and Phase 2 check.

### Concept or Hypothesis

Phase 2 tested whether VisionLab could establish an explainable custom CNN data-to-logits path for CIFAR-10-shaped tensors before introducing a trainer or any material run.

### AI Contribution

Codex implemented `CustomCNNConfig`, `CustomCNN`, and `count_parameters`; added CPU-only unittest coverage; updated the deterministic test script to use the verified local `.venv` when available; declared `torch>=2.13` as a model-work dependency; and created the Phase 2 closeout.

### Builder Review and Decision

Implementation is complete and awaiting builder review. Phase 3 remained out of scope for this entry and required a separate concept briefing and plan.

### Evidence

- `scripts/test.ps1` passed 32 tests.
- The custom model returns raw logits shaped `N x 10`.
- `torch.nn.CrossEntropyLoss` accepts the model output and integer class labels in the smoke test.
- Intermediate shapes and parameter counts are documented in the Phase 2 closeout.

### Project Impact

VisionLab now has its first custom model skeleton and shape-safe forward contract. It still has no training engine, checkpointing, material experiment, model metrics, inference path, transfer learning, or applied-domain behavior.

### Next Boundary

After builder acceptance of Phase 2, the next step is a separate Phase 3 concept briefing and implementation plan for the reproducible training engine.

---

## 2026-08-17 - Phase 3 Reproducible Training Engine Implementation

### Context

The builder approved Phase 3 after concept briefing and plan review, with a strict boundary around CPU synthetic/tiny-data verification, reproducibility-focused metadata, bounded checkpoint compatibility, validation/no-grad behavior, optional minimal scheduler support, and non-finite loss failure status.

### Concept or Hypothesis

Phase 3 tested whether VisionLab can run controlled optimization infrastructure, preserve enough state to restore a compatible checkpoint, and leave failed runs inspectable before any material CIFAR-10 baseline experiment.

### AI Contribution

Codex implemented `visionlab.training` modules for configuration, reproducibility helpers, training/validation loops, fit orchestration, optional scheduler construction, checkpoint save/restore, and run metadata. Codex added CPU-only tests for parameter updates, validation no-grad behavior, deterministic tiny overfit, learning-rate history, checkpoint round trip, incompatible checkpoint rejection, and non-finite loss failure metadata.

### Evidence

- `scripts/test.ps1` passed 38 tests.
- Tiny synthetic in-memory data reaches an explicit overfit criterion with final training accuracy `1.0` and final training loss below `0.02`.
- Validation preserves model training mode and does not mutate parameters or gradients in the tested path.
- Checkpoint restore verifies model, optimizer, scheduler, run, seed, epoch, and metric identity at the bounded Phase 3 level.

### Project Impact

VisionLab now has a reusable training-engine smoke path and checkpoint contract suitable for reviewing a later Phase 4 baseline training plan. It still has no material CIFAR-10 baseline, test-set evaluation, pretrained model, inference surface, or applied-domain behavior.

### Next Boundary

The builder accepted Phase 3 based on the completed phase-check report. Phase 3 is closed at `docs/phase_closeouts/Phase_3_reproducible_training_engine.md`.

Phase 4 should begin only after a separate concept briefing and implementation plan. Entry considerations carried forward from Phase 3 are explicit DataLoader shuffle/worker seed policy, explicit validation-based checkpoint-selection metric, continued test-split isolation from model selection, and approval before material training.

---

## 2026-08-18 - Phase 4A Baseline Experiment Plumbing and Smoke Verification

### Context

The builder approved Phase 4A as a split from Phase 4, with a strict boundary around baseline experiment plumbing and tiny smoke verification only. Phase 4A proves the experiment route, not the experiment result.

### Concept or Hypothesis

Phase 4A tested whether VisionLab can construct the registered train/validation/test data path, run the custom CNN through the existing training engine, select a best checkpoint by validation loss, and preserve minimal prediction/history artifacts before approving any material CIFAR-10 baseline run.

### AI Contribution

Codex implemented CIFAR-10 split and DataLoader plumbing, explicit loader reproducibility policy, minimal classification evaluation artifacts, prediction records, history/curve artifact writing, and a tiny synthetic CIFAR-shaped end-to-end smoke workflow. Codex added focused tests for split isolation, loader determinism, evaluation records, artifact writing, and smoke execution.

### Evidence

- `scripts/test.ps1` passed 44 tests.
- `scripts/run_phase4a_smoke.py` completed against tiny non-material data.
- Ignored smoke artifacts were generated under `outputs/phase4a_smoke/`.
- The smoke run contract records `official_test_evaluation: false`.

### Project Impact

VisionLab now has the baseline route needed for Phase 4B review, including loader construction, validation-based checkpoint selection mechanics, prediction records, per-class/confusion data, and history/curve artifacts. It still has no material CIFAR-10 baseline result.

### Next Boundary

Phase 4B requires builder approval of one exact material-run configuration, expected runtime/compute path, DataLoader seed policy, validation checkpoint-selection rule, stop conditions, and artifact-preservation plan before training.

---

## 2026-08-18 - Phase 4B Custom CNN Material Baseline Run

### Context

The builder approved the exact Phase 4B material-run configuration after Phase 4A follow-up checks verified the restored-best-checkpoint evaluation route and material CIFAR-10 preflight contract.

### Concept or Hypothesis

Phase 4B established the first official single-run custom CNN CIFAR-10 baseline for VisionLab. The run was intended as a baseline reference, not a tuned best result or an estimate of training variance.

### AI Contribution

Codex added a bounded Phase 4B material-run entry point, wrote `preflight_report.json` before training, ran the approved CPU configuration once, restored the checkpoint selected by minimum validation loss, generated final validation artifacts, evaluated the official test split once, generated test artifacts, and wrote a cautious baseline report.

### Evidence

- `scripts/test.ps1` passed 46 tests before the material run.
- Run ID: `phase4b-cifar10-custom-cnn-baseline-001`.
- Preflight passed with 45,000 train, 5,000 validation, and 10,000 test samples.
- Best checkpoint selected at epoch 10 by validation loss.
- Official test accuracy from the restored best checkpoint: `0.6359`.
- Artifacts are preserved under ignored `outputs/phase4b-cifar10-custom-cnn-baseline-001/`.

### Project Impact

VisionLab now has its first official custom CNN baseline result and prediction-level artifacts. The result remains a single run without augmentation, calibration, robustness, OOD evaluation, transfer-learning comparison, diagnostics, inference, or applied-domain claims.

### Next Boundary

Phase 4 was accepted and closed after the separate phase-check review. Phase 5 should begin only after a separate concept briefing and implementation plan.

---

## 2026-08-19 - Phase 4 Closeout

### Context

The builder accepted Phase 4 based on the completed phase-check report and requested final closeout documentation.

### AI Contribution

Codex finalized the Phase 4 closeout, clarified the Phase 4A historical `44 tests` note against the current `46-test` suite, updated README and phase catalog status, recorded the complete Phase 4 artifact inventory, and noted unrelated deleted root `phase_briefing.md` and `phase_check.md` worktree entries as excluded from Phase 4 scope.

### Evidence

- Final Phase 4 closeout: `docs/phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md`.
- Phase 4B run artifacts: `outputs/phase4b-cifar10-custom-cnn-baseline-001/`.
- Restored-best official test loss: `1.024515`.
- Restored-best official test accuracy: `0.635900`.

### Project Impact

VisionLab now has a closed custom-CNN baseline phase and a reference artifact set for later controlled comparisons.

### Next Boundary

Phase 5A - Augmentation Profile and Smoke Verification has since been implemented for builder review. Phase 5B material training remains unapproved.

---

## 2026-08-19 - Phase 5A Augmentation Profile and Smoke Verification

### Context

The builder approved Phase 5A only, with a strict boundary around augmentation profiles, smoke tests, and visual inspection before any material augmented CIFAR-10 training.

### AI Contribution

Codex implemented a versioned augmentation-profile registry, preserved the Phase 4 no-augmentation control profile, added one candidate train-only horizontal-flip/crop profile, kept validation and test preprocessing deterministic, generated fixed-sample visual inspection artifacts, added smoke tests, and drafted a Phase 5A closeout with a proposed Phase 5B material-run configuration.

### Evidence

- Phase 5A closeout: `docs/phase_closeouts/Phase_5A_augmentation_profile_and_smoke_verification.md`.
- Profile registry artifact: `outputs/phase5a_augmentation_inspection/augmentation_profile_registry.json`.
- Visual grid artifact: `outputs/phase5a_augmentation_inspection/phase5a_candidate_augmentation_grid.png`.
- Inspection note: `outputs/phase5a_augmentation_inspection/phase5a_visual_inspection_note.md`.
- Targeted tests: `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m unittest tests.test_phase5_augmentation` passed with `6` tests.

### Project Impact

VisionLab now has an explicit augmentation contract and visual inspection path. The candidate augmentation profile is ready for builder review, but no material augmented training result exists.

### Next Boundary

Phase 5A is accepted and closed. The next step is builder review and approval of the exact Phase 5B material-run configuration. Phase 5B must not begin until that approval is given.

---

## 2026-08-20 - Phase 5B Material Augmentation Comparison and Closeout

### Context

The builder approved the exact Phase 5B material-run contract after accepting the separate Phase 5B implementation/plumbing review and the completed Phase 5B phase check.

### Concept or Hypothesis

Phase 5B tested whether one explicit train-time augmentation candidate, `phase5a-candidate-horizontal-flip-random-crop` version `1.0`, would improve or at least hold the Phase 4B custom-CNN baseline under an otherwise fixed CIFAR-10 training configuration.

### AI Contribution

Codex executed the approved Phase 5B material run exactly once, preserved the run contract and artifact set, restored the checkpoint selected by minimum validation loss, evaluated the official test split once after checkpoint selection, produced the comparison report against `phase4b-cifar10-custom-cnn-baseline-001`, completed the phase-check review, and finalized the Phase 5B closeout and project-status documentation.

### Evidence

- Phase 5B closeout: `docs/phase_closeouts/Phase_5B_material_augmentation_comparison.md`.
- Run ID: `phase5b-cifar10-custom-cnn-augmentation-candidate-001`.
- Comparison report: `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/phase5b_comparison_report.md`.
- Run artifacts: `outputs/phase5b-cifar10-custom-cnn-augmentation-candidate-001/`.
- Restored-best validation loss: `1.055734`.
- Restored-best validation accuracy: `0.620600`.
- Official test loss: `1.056135`.
- Official test accuracy: `0.630800`.
- Delta versus `phase4b-cifar10-custom-cnn-baseline-001`: `+0.031620` test loss and `-0.005100` test accuracy.
- Canonical deterministic suite passed after closeout updates via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1`.

### Project Impact

VisionLab now has a closed Phase 5 controlled augmentation comparison with preserved negative single-run evidence. The candidate train-time horizontal-flip/crop profile is not adopted as the new baseline, and the accepted Phase 4B no-augmentation run remains the comparison reference for later work.

### Interpretation Boundary

This result records an observed regression in one controlled run only. It should not be described as proof that augmentation generally hurts CIFAR-10 performance, or that all crop/flip policies would regress under other approved conditions.

### Next Boundary

Phase 5 is closed. Phase 6 has not begun and requires a separate concept briefing, plan, and approval boundary.

---

## 2026-08-20 - Phase 6A Transfer Model Contract and Closeout

### Context

The builder approved Phase 6A after the Phase 6 briefing, tightened implementation plan, completed implementation, and Phase 6A phase-check review.

### Concept or Boundary

Phase 6A established a frozen-feature transfer-learning contract without running material transfer training. The selected backbone is exactly `torchvision.models.resnet18`, bound to `ResNet18_Weights.IMAGENET1K_V1`. The executed smoke path used `weights=None` because `resnet18-f37072fd.pth` was not cached and no download was approved.

### AI Contribution

Codex implemented the transfer-model contract, separate ImageNet preprocessing contract, non-download cache probe, frozen-backbone/head-only parameter inspection, tiny synthetic mechanics smoke path, guarded pretrained frozen-feature smoke path, focused tests, phase-check review, and formal Phase 6A closeout documentation.

### Evidence

- Phase 6A closeout: `docs/phase_closeouts/Phase_6A_transfer_model_contract_and_tiny_frozen_feature_smoke.md`.
- Transfer model identity: `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`.
- Preprocessing contract: resize `256`, crop `224`, bilinear interpolation, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, input `N x 3 x 224 x 224`.
- Classifier replacement: `Linear(512, 10)`.
- Measured parameter counts: total `11,181,642`, trainable `5,130`, frozen `11,176,512`.
- Mechanics smoke verified frozen parameters remained unchanged, frozen parameters did not receive trainable gradients, and classifier-head parameters updated.
- Focused Phase 6A tests passed at `12` tests.
- Canonical deterministic suite passed after closeout updates via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1` at `66` tests.

### Project Impact

VisionLab now has an accepted transfer-model contract and mechanics smoke foundation for later frozen-feature material training. This closes Phase 6A but does not complete Phase 6 as a whole.

### Interpretation Boundary

Phase 6A smoke is mechanics evidence only. It is not transfer-learning evidence, not pretrained performance evidence, and not CIFAR-10 generalization evidence. The Phase 4B baseline `phase4b-cifar10-custom-cnn-baseline-001` remains unchanged and remains the comparison reference. Any later custom-CNN versus ResNet-18 comparison is asymmetric because of ImageNet source pretraining, model size, input resolution, preprocessing, and parameter count.

### Next Boundary

Phase 6B has not begun. Before material pretrained training, VisionLab must resolve pretrained-weight availability or download approval for `resnet18-f37072fd.pth` and verify the actual preprocessing application path.

---

## 2026-08-20 - Phase 6B-1 Pretrained Frozen-Feature Smoke and Closeout

### Context

The builder approved downloading the exact pretrained checkpoint required for Phase 6B-1 and accepted the completed Phase 6B-1 phase-check review.

### Concept or Boundary

Phase 6B-1 verified the actual pretrained frozen-feature smoke path before any material CIFAR-10 training. The selected model remained exactly `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`, using checkpoint `resnet18-f37072fd.pth`.

### AI Contribution

Codex downloaded only the approved checkpoint, verified cache availability, implemented actual Torchvision preprocessing application, ran tiny pretrained frozen-feature smoke with `pretrained_weights_loaded: true`, preserved smoke artifacts, ran focused and canonical tests, completed the phase-check review, and formalized the Phase 6B-1 closeout.

### Evidence

- Phase 6B-1 closeout: `docs/phase_closeouts/Phase_6B1_pretrained_frozen_feature_smoke.md`.
- Smoke artifacts: `outputs/phase6b1-resnet18-pretrained-frozen-smoke/`.
- Exact checkpoint: `resnet18-f37072fd.pth`.
- Preprocessing source: `ResNet18_Weights.IMAGENET1K_V1.transforms()`.
- Raw smoke input shape: `2 x 3 x 32 x 32`.
- Preprocessed model input shape: `2 x 3 x 224 x 224`.
- `pretrained_weights_loaded: true`.
- Frozen gradients blocked, frozen parameters unchanged, classifier head updated.
- Measured parameter counts: total `11,181,642`, trainable `5,130`, frozen `11,176,512`.
- Focused verification passed at `23` tests with `1` skipped.
- Canonical deterministic suite passed after closeout updates via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1` at `71` tests with `1` skipped.

### Project Impact

VisionLab now has accepted evidence that the exact pretrained ResNet-18 weights are available locally, the actual selected preprocessing path works, and frozen-feature mechanics hold with actual pretrained weights.

### Interpretation Boundary

Phase 6B-1 is pretrained frozen-feature mechanics smoke evidence only. It is not material CIFAR-10 training evidence, validation performance evidence, official test evidence, fine-tuning evidence, or a pretrained-versus-custom comparison. The Phase 4B baseline remains unchanged and remains the comparison reference.

### Next Boundary

Phase 6B-2 has since completed one approved material frozen-feature transfer run. Phase 6C fine-tuning remains separate and unstarted.

---

## 2026-08-21 - Phase 6B-2 Material Frozen-Feature Run and Closeout

### Context

The builder approved one exact material frozen-feature transfer-learning run after Phase 6B-2 implementation, preflight, timing probe, and phase-check review.

### Concept or Boundary

Phase 6B-2 tested one frozen-feature pretrained ResNet-18 reference on the registered CIFAR-10 split. It was designed as a fixed reference point, not a fine-tuning run, hyperparameter search, seed sweep, or architecture-only comparison.

### AI Contribution

Codex ran the approved material preflight, launched the exact CPU material run, monitored the runtime guard, restored the best checkpoint selected by minimum validation loss, evaluated the official test split once after restoration, inspected the preserved artifacts, ran the canonical deterministic suite, completed the phase-check review, and formalized the Phase 6B-2 closeout.

### Evidence

- Phase 6B-2 closeout: `docs/phase_closeouts/Phase_6B2_material_frozen_feature_run.md`.
- Material artifacts: `outputs/phase6b2-cifar10-resnet18-frozen-feature-001/`.
- Exact model: `torchvision.models.resnet18`.
- Exact weights: `ResNet18_Weights.IMAGENET1K_V1`.
- Exact checkpoint: `resnet18-f37072fd.pth`.
- Classifier: `Linear(512, 10)`.
- Preprocessing: `ResNet18_Weights.IMAGENET1K_V1.transforms()`.
- Configuration: Adam, learning rate `0.001`, weight decay `0.0`, no scheduler, batch size `64`, CPU, seed `20260820`, 5 epochs, no augmentation.
- Best checkpoint: epoch `4`.
- Restored-best validation loss: `0.398302`.
- Restored-best validation accuracy: `0.864600`.
- Official test loss: `0.413686`.
- Official test accuracy: `0.856100`.
- Canonical deterministic suite passed after closeout updates via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1` at `73` tests with `1` skipped.

### Project Impact

VisionLab now has a fixed single-run frozen-feature pretrained comparison reference against the accepted Phase 4B custom-CNN baseline. Relative to Phase 4B, Phase 6B-2 observed a single-run official test loss delta of `-0.610829` and test accuracy delta of `+0.220200`.

### Interpretation Boundary

The Phase 6B-2 result is asymmetric because of ImageNet source pretraining, ResNet-18 parameter scale, `224 x 224` inputs, ImageNet preprocessing, and frozen-feature head-only training. It does not establish fine-tuning performance, calibration, robustness/OOD behavior, seed variance, or architecture-only superiority.

The `0.856100` official test accuracy should remain a fixed Phase 6B-2 reference point. Phase 6C should not automatically inherit it as a target or begin tuning around it.

### Next Boundary

Phase 6C-1 has since completed the fine-tuning contract, smoke, and preflight boundary. Phase 6C-2 material fine-tuning remains separate and unstarted.

---

## 2026-08-21 - Phase 6C-1 Fine-Tuning Contract, Smoke, and Preflight

### Context

The builder approved Phase 6C-1 only after refining the plan to require explicit Phase 6B-2 checkpoint initialization, careful wording of the experimental intervention, and optimizer-scope verification.

### Concept or Boundary

Phase 6C-1 established the mechanics and preflight path for later limited fine-tuning. It did not launch material fine-tuning or evaluate the official CIFAR-10 test split.

The intended intervention for a future material run is changing the training regime from frozen-backbone/head-only training to `layer4 + fc` fine-tuning while preserving the dataset, initialization, preprocessing, augmentation, seed, and evaluation protocol.

### AI Contribution

Codex added the `finetune_layer4_head` transfer-model mode, implemented a Phase 6C-1 preflight module, restored the Phase 6B-2 best checkpoint as the required initialization source, verified `layer4 + fc` trainability, proved optimizer membership exactly matches the trainable parameters, ran tiny mechanics smoke, wrote preflight/timing artifacts, ran focused and canonical tests, completed the phase-check review, and formalized the Phase 6C-1 closeout.

### Evidence

- Phase 6C-1 closeout: `docs/phase_closeouts/Phase_6C1_fine_tuning_contract_smoke_and_preflight.md`.
- Preflight artifacts: `outputs/phase6c-cifar10-resnet18-layer4-finetune-001-preflight/`.
- Initialization source run: `phase6b2-cifar10-resnet18-frozen-feature-001`.
- Initialization checkpoint: `best`, epoch `4`.
- Initialization checkpoint SHA-256: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- Fine-tuning mode: `finetune_layer4_head`.
- Trainable scope: ResNet-18 `layer4 + fc`.
- Parameter counts: total `11,181,642`, trainable `8,398,858`, frozen `2,782,784`.
- Optimizer scope: exactly trainable parameters; frozen parameters in optimizer groups = `0`.
- Mechanics smoke verified finite loss, logits shape `2 x 10`, frozen gradients blocked, frozen parameters unchanged, and trainable parameters updated.
- Timing probe estimated a 3-epoch CPU material run at `4085.54` seconds, about `68.1` minutes, using batch size `64`.
- Focused verification passed at `16` tests.
- Canonical deterministic suite passed after closeout updates via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1` at `79` tests with `1` skipped.

### Project Impact

VisionLab now has a verified fine-tuning contract and material-run preflight for the selected ResNet-18 pathway. Phase 6C-1 closes the mechanics/readiness boundary and leaves Phase 6C-2 as a separate compute and experiment approval decision.

### Interpretation Boundary

Phase 6C-1 is mechanics and preflight evidence only. It is not material fine-tuning performance evidence, not an official test result, not calibration, not robustness/OOD, not seed variance, and not architecture-only evidence.

The Phase 6B-2 frozen-feature result remains a fixed reference point and should not be used as a tuning target.

### Next Boundary

Phase 6C-2 may be proposed as a separate material fine-tuning run. It requires explicit builder approval before launch and must preserve validation-loss checkpoint selection plus one official test evaluation after best-checkpoint restoration.

---

## Project Context

VisionLab is an AI-native computer-vision engineering laboratory.

The intended project will:

- establish trustworthy dataset and split identities;
- review and apply image and convolution fundamentals;
- implement and train a custom CNN using PyTorch primitives;
- build a reproducible training and checkpointing path;
- compare the custom model with one pretrained vision backbone;
- evaluate models beyond aggregate accuracy;
- measure calibration, confidence, degradation robustness, and domain shift;
- inspect representative and high-confidence failures;
- generate model-appropriate diagnostics with explicit limitations;
- support bounded image inference;
- select one applied domain through a feasibility decision;
- measure a controlled/synthetic-to-real or equivalent domain gap;
- test one evidence-supported intervention and re-evaluate it.

The project is intentionally bounded.

It is not intended to become a broad survey of classification, detection, segmentation, video, vision-language models, simulation, and edge deployment. It is also not intended to become a production medical, safety, inspection, or forensic authority.

The intended maturity progression is:

```text
Learning Foundation
  → Model-Engineering MVP
  → Evaluation-Centered MVP+
  → Applied Domain-Transfer Capstone
```

The strong MVP+ core should be independently closable before the applied capstone begins.

---

## Why This Is an AI-Native Builder Project

VisionLab is meaningful not only because of the intended software and experimental outputs, but also because of the learning and evaluation challenge.

The builder begins with general AI, Python, and PyTorch experience, but does not treat computer-vision engineering judgment as already complete. The project therefore requires progress in four connected areas:

- improving understanding of vision and model-training concepts;
- building a credible implementation path;
- learning to design and control experiments;
- improving the judgment needed to evaluate data, models, failures, and claims.

The core working idea is:

> The builder develops the vision system, while the system-building process develops the builder’s capacity to understand, challenge, and improve it.

This is different from using AI only as a code generator.

Codex should contribute through bounded roles:

- tutor before unfamiliar implementation;
- planner after repository inspection;
- implementer after approval;
- test and fixture author;
- debugging partner;
- experiment-design assistant;
- reviewer of data, configuration, and result integrity;
- documentation and closeout partner.

The human builder remains responsible for:

- understanding the important concepts;
- reviewing and approving substantive plans;
- controlling scope and phase boundaries;
- approving material training runs;
- examining actual images, plots, errors, and artifacts;
- separating observations from plausible explanations;
- selecting the applied domain;
- approving requirement changes;
- deciding whether phases and maturity boundaries are complete;
- determining the public portfolio narrative.

---

## Core Technical Question

VisionLab is organized around one unifying question:

> How do model architecture and training-data distribution affect generalization, robustness, confidence, and failure behavior when a vision model moves from controlled training conditions toward less-controlled real-world images?

This question connects the project’s major stages:

- the custom CNN establishes foundational architectural ownership;
- transfer learning introduces pretrained representations;
- controlled comparison examines architectural and training differences;
- calibration examines whether confidence matches correctness;
- degradation testing examines sensitivity to altered inputs;
- OOD testing examines cross-source generalization;
- failure analysis examines recurring model weaknesses;
- applied domain transfer examines the gap between controlled training data and real images;
- one intervention tests whether data or adaptation changes the observed gap.

The project does not assume that the custom model, pretrained model, stronger augmentation, synthetic data, domain randomization, or another intervention will produce the preferred result.

---

## Starting Assumptions

The initial direction is based on the following working assumptions:

- a custom CNN should precede transfer learning so the project develops real vision-model fundamentals;
- a low-friction development dataset can support the early learning and engineering phases;
- data identity and split integrity should be established before material training;
- test, OOD, and real-world data should remain outside routine model selection;
- training success must be judged through artifacts and evaluation, not process completion alone;
- aggregate accuracy is insufficient for a serious model assessment;
- calibration, degradation robustness, OOD behavior, and failure analysis can reveal weaknesses hidden by clean test accuracy;
- Grad-CAM and related methods are diagnostics, not proof of model understanding;
- negative or mixed experimental results are valid project outcomes;
- the applied domain should be selected through feasibility evidence during implementation;
- classification is the default applied-task boundary;
- controlled or synthetic data should be treated as a data-engineering hypothesis, not automatically realistic training data;
- one diagnosed intervention is more informative than several uncontrolled changes;
- the strong MVP+ core should be protected from applied-capstone risk;
- requirements and phase divisions may evolve when implementation evidence justifies change.

These are not conclusions. Triage, implementation, tests, visual inspection, and training evidence may confirm, refine, or reject them.

---

## Current Architectural Direction

The current reference laboratory workflow is:

```text
Dataset and Split Validation
  → Model and Configuration Identity
  → Custom CNN Training or Checkpoint Restore
  → In-Distribution Evaluation
  → Transfer-Learning Comparison
  → Calibration and Confidence Analysis
  → Degradation Robustness Sweep
  → OOD / Cross-Source Evaluation
  → Failure Analysis and Diagnostics
  → Experiment Artifact and Model Comparison
  → Bounded Inference
```

The current applied-capstone loop is:

```text
Applied-Domain Feasibility and Selection
  → Controlled or Synthetic Training Data
  → Independent Real Evaluation Data
  → Domain-Gap Baseline
  → Failure Diagnosis
  → One Approved Intervention
  → Controlled Retraining
  → Before/After Re-Evaluation
  → Honest Technical Report
```

The central architectural principle is **data and experiment identity before claims**.

The system should keep clear boundaries among:

- dataset source and split identity;
- preprocessing and augmentation;
- model configuration and weights;
- training configuration and run state;
- prediction-level records;
- aggregate evaluation;
- visual observations;
- explanatory hypotheses;
- interventions;
- public conclusions.

The exact repository structure may evolve. The project should preserve focused responsibility boundaries without constructing empty architecture merely to match a proposed directory tree.

---

## Incremental AI-Native Workflow

The normal phase cadence is:

```text
Concept Briefing
  → Builder Questions and Boundary Review
  → Bounded Implementation Plan
  → Builder Approval
  → Implementation
  → Automated Verification
  → Visual and Manual Verification
  → Result Interpretation
  → Phase Check and Context Synchronization
  → Phase Closeout
```

The concept briefing should function as tutoring. It should explain the concepts the builder needs to understand and evaluate the phase without becoming a substitute project review.

The phase check should determine whether the work has the intended conceptual and technical shape and should synchronize the builder and Codex before the next phase.

Broad phases may be divided into `A/B/C` subphases when:

- data feasibility should precede implementation;
- a schema or artifact contract should be stabilized first;
- a smoke path should precede material training;
- training and post-training interpretation require separate approvals;
- visual inspection creates a human-review boundary;
- compute or external access creates a natural pause;
- verification exposes a bounded repair;
- the phase is too broad to review safely as one change.

Phase splitting is a workflow refinement, not a failure.

---

## Material Training Workflow

Training is an experimental action, not merely a command.

The expected material-run sequence is:

```text
Pipeline and Tiny-Data Smoke Verification
  → Training Hypothesis and Configuration
  → Dataset / Split / Model Identity Review
  → Compute and Artifact Plan
  → Builder Approval
  → Material Training Run
  → Checkpoint and Metric Inspection
  → Evaluation and Failure Review
  → Interpretation Approval
```

Before a material run, the journal or associated phase artifact should identify:

- experiment purpose and hypothesis;
- dataset and split version;
- model and pretrained-weight identity, if any;
- preprocessing and augmentation profile;
- seed or seed set;
- optimizer, scheduler, and training budget;
- checkpoint and early-stop rules;
- metrics and prediction artifacts;
- expected environment and runtime;
- stop and failure conditions;
- outputs to preserve.

After the run, record:

- actual environment and duration;
- terminal status;
- selected checkpoint;
- learning-curve observations;
- evaluation sample coverage;
- warnings or anomalies;
- facts directly shown by artifacts;
- interpretations or hypotheses;
- decisions and follow-up.

Do not treat a falling training loss, successful process exit, or attractive graph as proof that an experiment succeeded.

---

## Initial Triage Plan

### T0 — Project Bootstrap and Baseline Capture

Purpose:

- establish the repository and operating documents;
- preserve a truthful starting state;
- define version-control boundaries for data, checkpoints, runs, outputs, private images, and secrets;
- establish initial risks and requirement governance;
- create a minimal environment or smoke path.

Expected artifacts may include:

- repository skeleton;
- `AGENTS.md`;
- `README.md`;
- `docs/project_specs.md`;
- this builder journal;
- `.gitignore` and environment guidance;
- initial risk register;
- requirement change log;
- initial phase catalog.

Status: **Complete**

### T1 — Vision Foundations and Feasibility Triage

Purpose:

- review image tensors, channels, normalization, convolution, pooling, receptive fields, and feature maps;
- verify the local and GPU execution path;
- complete small tensor and image-loading exercises;
- inspect one convolution or feature-map visualization;
- compare low-friction development-dataset candidates;
- establish provisional data and compute decisions.

Expected artifacts may include:

- vision glossary or concept notes;
- environment and device probe;
- tiny image-loading spike;
- feature-map visualization;
- candidate dataset comparison;
- provisional development-dataset decision;
- compute feasibility note;
- updated risks or requirements.

Status: **Accepted; final builder visual review condition recorded**

---

## Reference Implementation Roadmap

The current project specification proposes these phases:

1. Dataset Contract and Visual Data Inspection
2. Custom CNN and Shape-Safe Forward Path
3. Reproducible Training Engine
4. Custom CNN Baseline Experiment
5. Augmentation and Generalization Controls
6. Transfer Learning and Fine-Tuning
7. Evaluation Harness and Calibration
8. Robustness and OOD Evaluation
9. Failure Analysis and Interpretability
10. Inference Surface and Core Stabilization
11. Applied-Domain Feasibility and Selection
12. Applied Data Pipeline and Real Evaluation Set
13. Domain-Gap Baseline and Diagnosis
14. Data-Centric Intervention and Re-Evaluation
15. Final Integration, Portfolio Polish, and Closure Review

This roadmap is a starting hypothesis.

It may change when learning, data access, tests, visual inspection, compute limits, training behavior, or phase checks expose better boundaries or sequencing.

The project should preserve its maturity progression even when phase numbering changes.

---

## Maturity Trail

### Learning Foundation

Expected evidence:

- trustworthy development-dataset contract;
- custom CNN;
- reliable trainer;
- genuine baseline experiment;
- basic evaluation artifacts;
- builder explanation of the data-to-logits path.

Status: **Not reached**

### Model-Engineering MVP

Expected evidence:

- custom and pretrained model paths;
- frozen-feature and controlled fine-tuning runs;
- compatible experiment artifacts;
- preliminary model comparison.

Status: **Not reached**

### Evaluation-Centered MVP+

Expected evidence:

- correct class-wise evaluation;
- calibration and confidence analysis;
- degradation robustness;
- OOD or cross-source testing;
- systematic failure analysis;
- model-appropriate diagnostics;
- bounded inference;
- core stabilization review.

Status: **Not reached**

### Applied Domain-Transfer Capstone

Expected evidence:

- approved domain decision;
- controlled or synthetic training source;
- independent real evaluation source;
- measured domain gap;
- diagnosed failure patterns;
- one approved intervention;
- before/after re-evaluation;
- honest capstone report.

Status: **Not reached**

---

## Requirement Evolution

Requirement changes are expected, but they should be explicit, evidence-based, and reviewable.

Potential reasons include:

- inaccessible or poorly licensed data;
- class or label ambiguity;
- group-level leakage risk;
- corrupt or insufficient samples;
- compute limits;
- training instability;
- metric or calibration flaws;
- comparison incompatibility;
- diagnostics that do not support the intended claim;
- an applied domain that fails feasibility review;
- a simpler implementation that better preserves the learning objective;
- phase boundaries that prove too broad.

Examples of healthy future changes might include:

- replacing a development dataset after a documented feasibility problem;
- splitting training infrastructure from the first material run;
- narrowing augmentation after visual inspection shows label distortion;
- replacing a pretrained backbone due to compute or preprocessing constraints;
- choosing controlled photography instead of Blender;
- keeping classification rather than promoting detection;
- replacing physical collection with a valid cross-source public evaluation when collection is unsafe or infeasible;
- closing at the strong MVP+ boundary if no applied domain responsibly passes the gate.

Approved substantive changes should be recorded in:

- `docs/requirement_change_log.md`

Major architecture or applied-domain decisions should also use an ADR or dedicated decision report where appropriate.

---

## What the Project Should Preserve

Even if features, datasets, and sequencing change, the project should preserve these principles unless deliberately reconsidered:

- custom CNN before transfer learning;
- data and split identity before material training;
- separation of training, validation, test, OOD, and real-world evaluation roles;
- group-aware leakage prevention;
- reproducible configurations and material-run artifacts;
- controlled model and intervention comparisons;
- evaluation beyond aggregate accuracy;
- calibration and confidence visibility;
- degradation and domain-shift measurement;
- representative failure analysis;
- cautious interpretation of diagnostics;
- valid negative and mixed results;
- training and domain-selection approval gates;
- an independently closable strong MVP+ core;
- human approval over substantive project changes;
- honest documentation of current state.

---

## Journal Entry Convention

The journal should remain useful rather than becoming a raw activity log.

Add entries for material moments such as:

- phase entry and closeout;
- important concept review;
- dataset or compute feasibility decision;
- material training approval and result;
- unexpected model or data behavior;
- requirement or architecture change;
- accepted, modified, or rejected AI recommendation;
- meaningful debugging or verification finding;
- applied-domain selection;
- maturity-boundary or project-closure decision.

A recommended entry format is:

```markdown
## YYYY-MM-DD — Phase/Decision Title

### Context
What question or phase boundary was active?

### Concept or Hypothesis
What did the builder need to understand or test?

### AI Contribution
What did Codex or ChatGPT propose, explain, implement, or review?

### Builder Review and Decision
What was accepted, modified, rejected, or left unresolved?

### Evidence
What code, tests, images, plots, metrics, artifacts, or manual checks support the result?

### Learning
What changed in the builder's understanding?

### Project Impact
What changed in requirements, architecture, sequencing, risk, or readiness?

### Next Boundary
What is approved next, and what remains out of scope?
```

Short entries may omit sections that add no value. Do not fabricate disagreement, learning, or AI error to make the journal look more dramatic.

---

## Experiment Entry Convention

For a material training or intervention run, record or link to:

- experiment/run ID;
- hypothesis;
- dataset and split identity;
- model and weight identity;
- configuration and seed;
- compute environment;
- approval point;
- expected and actual runtime;
- checkpoint rule and selected checkpoint;
- primary metrics;
- calibration, robustness, or OOD result when relevant;
- important failures or warnings;
- preserved artifact locations;
- factual observations;
- interpretation and uncertainty;
- builder decision.

The journal may summarize these items and link to a detailed experiment artifact rather than duplicating it.

Do not enter performance numbers that cannot be traced to a preserved artifact.

---

## Phase Closeout Trail

Phase closeout files should become the primary evidence of the project’s development history.

Each closeout should record, at a useful level:

- phase learning and engineering objectives;
- what was learned;
- what was built, trained, evaluated, or investigated;
- important files, configurations, runs, and artifacts;
- tests and manual or visual verification;
- observations and interpretations;
- assumptions confirmed, narrowed, or rejected;
- requirement or architecture changes;
- known limitations;
- readiness for the next phase.

Expected closeout trail:

- T0 — [Project Bootstrap and Baseline Capture](docs/phase_closeouts/T0_project_bootstrap_and_baseline_capture.md)
- T1 — [Vision Foundations and Feasibility Triage](docs/phase_closeouts/T1_vision_foundations_and_feasibility_triage.md)
- Phase 1A — [Dataset Contract and Deterministic Tiny-Fixture Validation](docs/phase_closeouts/Phase_1A_dataset_contract_and_tiny_fixture_validation.md)
- Phase 1B — [CIFAR-10 Registration and Visual Data Inspection](docs/phase_closeouts/Phase_1B_cifar10_registration_and_visual_inspection.md)
- Phase 1 — [Dataset Contract and Visual Data Inspection](docs/phase_closeouts/Phase_1_dataset_contract_and_visual_data_inspection.md)
- Phase 2 — [Custom CNN and Shape-Safe Forward Path](docs/phase_closeouts/Phase_2_custom_cnn_and_shape_safe_forward_path.md)
- Phase 3 — [Reproducible Training Engine](docs/phase_closeouts/Phase_3_reproducible_training_engine.md)
- Phase 4 — [Custom CNN Baseline Experiment](docs/phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md)
- Phase 4A — [Baseline Experiment Plumbing and Smoke Verification](docs/phase_closeouts/Phase_4A_baseline_experiment_plumbing_and_smoke_verification.md)
- Phase 4B — summarized in [Custom CNN Baseline Experiment](docs/phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md)
- Phase 5A — [Augmentation Profile and Smoke Verification](docs/phase_closeouts/Phase_5A_augmentation_profile_and_smoke_verification.md)
- Phase 5B — [Material Augmentation Comparison](docs/phase_closeouts/Phase_5B_material_augmentation_comparison.md)
- Phase 5 — closed through accepted Phase 5A and Phase 5B subphases
- Phase 6A — [Transfer Model Contract and Tiny Frozen-Feature Smoke](docs/phase_closeouts/Phase_6A_transfer_model_contract_and_tiny_frozen_feature_smoke.md)
- Phase 6B-1 — [Pretrained Frozen-Feature Smoke](docs/phase_closeouts/Phase_6B1_pretrained_frozen_feature_smoke.md)
- Phase 6B-2 — [Material Frozen-Feature Transfer Run](docs/phase_closeouts/Phase_6B2_material_frozen_feature_run.md)
- Phase 6C-1 — [Fine-Tuning Contract, Smoke, and Preflight](docs/phase_closeouts/Phase_6C1_fine_tuning_contract_smoke_and_preflight.md)
- Phase 6C-2 — [Material Layer4 Fine-Tuning Run](docs/phase_closeouts/Phase_6C2_material_layer4_fine_tuning_run.md)
- Phase 6 — complete through accepted Phase 6C-2 subphase
- Phase 7 — [Evaluation Harness and Calibration](docs/phase_closeouts/Phase_7_evaluation_harness_and_calibration.md)
- Phase 8 — to be added; not started
- Phase 9 — to be added
- Phase 10 — to be added
- Phase 11 — to be added
- Phase 12 — to be added
- Phase 13 — to be added
- Phase 14 — to be added
- Phase 15 — to be added

If a phase is split, list and link each closeout explicitly. Do not hide meaningful subphase history behind one retroactive summary.

---

## AI Recommendation Trail

The journal should make material AI involvement reviewable without becoming a transcript archive.

For important recommendations, record whether the builder:

- accepted the recommendation;
- modified it;
- rejected it;
- deferred it pending evidence.

Useful examples include:

- dataset or backbone selection;
- proposed phase split;
- augmentation change;
- suspected leakage;
- explanation of a training anomaly;
- failure-cluster hypothesis;
- proposed intervention;
- applied-domain selection;
- closure or expansion recommendation.

The journal should not claim human verification merely because AI produced code or an explanation. Verification should identify tests, artifacts, visual review, or reasoning performed by the builder.

---

## Initial Evaluation Questions

As the project develops, the builder should improve the ability to evaluate questions such as:

- Are image shapes, channels, ranges, and normalization correct?
- Do augmentations preserve label meaning?
- Are train, validation, test, OOD, and real-world boundaries clean?
- Are correlated subjects, objects, videos, generators, or capture sessions kept together?
- Can the custom CNN’s feature path and parameter scale be explained?
- Does the training curve indicate underfitting, overfitting, instability, or a pipeline error?
- Is checkpoint selection based on the intended validation metric?
- Are model comparisons controlled and compatible?
- Which classes fail, and how do those failures differ between models?
- Does confidence track correctness?
- How quickly does performance degrade under blur, compression, rescaling, noise, or lighting shift?
- Does OOD performance collapse, and is the model confidently wrong?
- Are failure galleries representative rather than cherry-picked?
- Do Grad-CAM or other diagnostics reveal stable patterns, and what can they not establish?
- Is a suspected failure caused by model architecture, data distribution, labels, background shortcuts, or source artifacts?
- Does the selected applied domain have sufficient independent real examples?
- Does controlled or synthetic data resemble the relevant aspects of reality?
- Is the intervention supported by diagnosed evidence?
- Does the before/after result exceed likely training variation?
- Are public claims proportionate to the experiment’s strength?
- When should the project abstain from a conclusion, descope, or close at the strong MVP+ boundary?

The quality of these questions—and the evidence used to answer them—should improve throughout the project.

---

## How to Read the Project Trail

For a quick project view and current status:

- [README.md](README.md)

For current scope, requirements, phase definitions, and closure boundaries:

- [Project Specification](docs/project_specs.md)

For durable Codex working rules:

- [AGENTS.md](AGENTS.md)

For requirement changes:

- [Requirement Change Log](docs/requirement_change_log.md) once created

For the development history:

- read this journal;
- follow phase closeouts in order;
- inspect linked experiment artifacts for material results;
- consult ADRs for major architecture and domain decisions.

---

## Why This Matters

This repository is intended to preserve more than final code and attractive plots.

It should show evidence of a disciplined human-AI learning and development process in which:

- the vision problem is refined rather than assumed complete;
- concepts are reviewed before unfamiliar implementation;
- Codex expands implementation and analysis capacity;
- AI-generated work remains inspectable and contestable;
- data and split integrity precede performance claims;
- material compute is planned and approved;
- automated tests are paired with visual verification;
- experiments preserve configurations and artifacts;
- observations remain distinct from explanations;
- negative results remain visible;
- requirements evolve through evidence rather than uncontrolled drift;
- the applied domain is selected responsibly;
- the human builder retains ownership of learning, direction, quality, and final responsibility.

VisionLab will be successful not merely if it achieves strong accuracy or presents an impressive demo, but if the repository makes clear:

- how the system became credible;
- what experiments actually established;
- where the models failed;
- what remains uncertain;
- why important decisions were made;
- how the builder learned to evaluate and improve the work.

> The final model is one project artifact. The builder’s improved capacity to reason about vision systems is another.
