# VisionLab Phase Catalog

Status: Phase 7 complete; accepted. Phase 8 has not started.

This catalog tracks the current phase sequence and phase status. The project specification remains the source for detailed scope, exit criteria, and maturity boundaries.

| Phase | Title | Status | Closeout |
| --- | --- | --- | --- |
| T0 | Project Bootstrap and Baseline Capture | Complete | [T0 closeout](phase_closeouts/T0_project_bootstrap_and_baseline_capture.md) |
| T1 | Vision Foundations and Feasibility Triage | Complete | [T1 closeout](phase_closeouts/T1_vision_foundations_and_feasibility_triage.md) |
| 1A | Dataset Contract and Deterministic Tiny-Fixture Validation | Complete | [Phase 1A closeout](phase_closeouts/Phase_1A_dataset_contract_and_tiny_fixture_validation.md) |
| 1B | CIFAR-10 Registration and Visual Data Inspection | Complete | [Phase 1B closeout](phase_closeouts/Phase_1B_cifar10_registration_and_visual_inspection.md) |
| 1 | Dataset Contract and Visual Data Inspection | Complete | [Phase 1 closeout](phase_closeouts/Phase_1_dataset_contract_and_visual_data_inspection.md) |
| 2 | Custom CNN and Shape-Safe Forward Path | Complete; awaiting builder review | [Phase 2 closeout](phase_closeouts/Phase_2_custom_cnn_and_shape_safe_forward_path.md) |
| 3 | Reproducible Training Engine | Complete; accepted | [Phase 3 closeout](phase_closeouts/Phase_3_reproducible_training_engine.md) |
| 4 | Custom CNN Baseline Experiment | Complete; accepted | [Phase 4 closeout](phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md) |
| 4A | Baseline Experiment Plumbing and Smoke Verification | Complete; accepted as Phase 4 subphase | [Phase 4A closeout](phase_closeouts/Phase_4A_baseline_experiment_plumbing_and_smoke_verification.md) |
| 4B | Custom CNN Material Baseline Run and Report | Complete; accepted as Phase 4 subphase | [Phase 4 closeout](phase_closeouts/Phase_4_custom_cnn_baseline_experiment.md) |
| 5 | Augmentation and Generalization Controls | Complete; accepted | [Phase 5B closeout](phase_closeouts/Phase_5B_material_augmentation_comparison.md) |
| 5A | Augmentation Profile and Smoke Verification | Complete; accepted | [Phase 5A closeout](phase_closeouts/Phase_5A_augmentation_profile_and_smoke_verification.md) |
| 5B | Material Augmentation Comparison | Complete; accepted | [Phase 5B closeout](phase_closeouts/Phase_5B_material_augmentation_comparison.md) |
| 6 | Transfer Learning and Fine-Tuning | Complete; accepted through Phase 6C-2 | [Phase 6C-2 closeout](phase_closeouts/Phase_6C2_material_layer4_fine_tuning_run.md) |
| 6A | Transfer Model Contract and Tiny Frozen-Feature Smoke | Complete; accepted | [Phase 6A closeout](phase_closeouts/Phase_6A_transfer_model_contract_and_tiny_frozen_feature_smoke.md) |
| 6B-1 | Pretrained Frozen-Feature Smoke | Complete; accepted | [Phase 6B-1 closeout](phase_closeouts/Phase_6B1_pretrained_frozen_feature_smoke.md) |
| 6B-2 | Material Frozen-Feature Transfer Run | Complete; accepted | [Phase 6B-2 closeout](phase_closeouts/Phase_6B2_material_frozen_feature_run.md) |
| 6C-1 | Fine-Tuning Contract, Smoke, and Preflight | Complete; accepted | [Phase 6C-1 closeout](phase_closeouts/Phase_6C1_fine_tuning_contract_smoke_and_preflight.md) |
| 6C-2 | Material Fine-Tuning Run | Complete; accepted | [Phase 6C-2 closeout](phase_closeouts/Phase_6C2_material_layer4_fine_tuning_run.md) |
| 7 | Evaluation Harness and Calibration | Complete; accepted | [Phase 7 closeout](phase_closeouts/Phase_7_evaluation_harness_and_calibration.md) |
| 8 | Robustness and OOD Evaluation | Not started | To be added |
| 9 | Failure Analysis and Interpretability | Not started | To be added |
| 10 | Inference Surface and Core Stabilization | Not started | To be added |
| 11 | Applied-Domain Feasibility and Selection | Not started | To be added |
| 12 | Applied Data Pipeline and Real Evaluation Set | Not started | To be added |
| 13 | Domain-Gap Baseline and Diagnosis | Not started | To be added |
| 14 | Data-Centric Intervention and Re-Evaluation | Not started | To be added |
| 15 | Final Integration, Portfolio Polish, and Closure Review | Not started | To be added |

## Phase Boundary Notes

- The applied domain remains deferred until Phase 11 unless a project-level requirement change is approved.
- Material training requires a separate compute and artifact approval boundary.
- Phases may be split into approved subphases when evidence shows a safer review boundary.
- CIFAR-10 is the registered provisional core development dataset; this is not an applied-domain selection.
- Phase 1 registered a deterministic train/validation/test split, stable upstream-based sample IDs, and a deterministic preprocessing profile.
- Phase 2 implemented a compact custom CNN and CPU forward/loss smoke path.
- Phase 3 implemented and closed a bounded CPU training engine with synthetic tiny-data verification, checkpoint save/restore, minimal reproducibility metadata, optional scheduler support, and non-finite loss failure status.
- Phase 4A implemented baseline experiment plumbing and a tiny smoke route. Its smoke metrics are pipeline evidence only, not official baseline results.
- Phase 4B produced the first single-run custom CNN CIFAR-10 baseline result: restored-best official test loss `1.024515` and test accuracy `0.635900`. It is not a tuned best result and not an estimate of run-to-run variance.
- Phase 5A implemented and closed a versioned no-augmentation control profile, one candidate train-only horizontal-flip/crop profile, machine-readable profile registry output, visual inspection artifacts, and smoke tests.
- Phase 5B executed one approved material comparison against the preserved Phase 4B baseline, changing only the train-time augmentation profile to `phase5a-candidate-horizontal-flip-random-crop` version `1.0`.
- Phase 5B observed a single-run regression relative to the Phase 4B baseline: official test loss `1.056135` versus `1.024515`, and official test accuracy `0.630800` versus `0.635900`.
- That observed regression is preserved as single-run comparison evidence only; it is not a broader claim that augmentation generally hurts performance.
- `phase5a-candidate-horizontal-flip-random-crop` version `1.0` is not adopted as the new baseline, and the Phase 4B no-augmentation run remains the reference baseline.
- Phase 6A implemented and closed an explicit ResNet-18 transfer-model contract bound to `torchvision.models.resnet18` and `ResNet18_Weights.IMAGENET1K_V1`, a separate ImageNet preprocessing contract, frozen-backbone/head-only parameter inspection, non-download weight-cache probing, and tiny synthetic mechanics smoke coverage.
- The approved Phase 6A implementation did not download pretrained weights; `resnet18-f37072fd.pth` was not present in the local Torch cache at implementation time.
- Phase 6A mechanics smoke is random-initialized model-mechanics evidence only; it is not transfer-learning performance evidence.
- Phase 6B-1 downloaded the exact approved checkpoint `resnet18-f37072fd.pth`, verified cache availability, loaded `ResNet18_Weights.IMAGENET1K_V1` with `pretrained_weights_loaded: true`, applied the actual Torchvision preprocessing path, and completed tiny pretrained frozen-feature mechanics smoke.
- Phase 6B-1 is smoke evidence only; it is not material CIFAR-10 training evidence, validation performance evidence, official test evidence, fine-tuning evidence, or a pretrained-versus-custom comparison.
- Phase 6B-2 executed one approved material frozen-feature run, `phase6b2-cifar10-resnet18-frozen-feature-001`, using `torchvision.models.resnet18`, `ResNet18_Weights.IMAGENET1K_V1`, checkpoint `resnet18-f37072fd.pth`, `Linear(512, 10)`, frozen-backbone/head-only training, ImageNet preprocessing, Adam learning rate `0.001`, batch size `64`, CPU, seed `20260820`, no augmentation, and a 5-epoch budget.
- Phase 6B-2 selected epoch `4` as the best checkpoint by validation loss. Restored-best validation loss was `0.398302`, validation accuracy was `0.864600`, official test loss was `0.413686`, and official test accuracy was `0.856100`.
- Relative to Phase 4B `phase4b-cifar10-custom-cnn-baseline-001`, Phase 6B-2 observed a single-run official test loss delta of `-0.610829` and test accuracy delta of `+0.220200`.
- The Phase 6B-2 comparison is asymmetric because of ImageNet pretraining, ResNet-18 model scale, `224 x 224` inputs, ImageNet preprocessing, and frozen-feature head-only training versus the from-scratch Phase 4B CustomCNN. It must not be read as architecture-only superiority.
- Phase 6B-2 does not establish fine-tuning performance, calibration, robustness/OOD behavior, seed variance, inference behavior, diagnostics, applied-domain behavior, or a tuned pretrained target.
- Phase 6C-1 implemented and closed the fine-tuning contract, Phase 6B-2 checkpoint initialization, `layer4 + fc` trainability, optimizer-scope verification, tiny mechanics smoke, and material preflight/timing path for `phase6c-cifar10-resnet18-layer4-finetune-001`.
- Phase 6C-1 initializes from the accepted Phase 6B-2 best checkpoint at epoch `4`, with checkpoint SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- Phase 6C-1 records measured parameter counts: total `11,181,642`, trainable `8,398,858`, frozen `2,782,784`.
- Phase 6C-1 timing estimates a future 3-epoch CPU material run at about `4085.54` seconds, or `68.1` minutes, using batch size `64`.
- Phase 6C-1 is mechanics/preflight evidence only. It does not include material fine-tuning, official test evaluation, fine-tuning performance evidence, calibration, robustness/OOD, seed variance, diagnostics, inference, applied-domain behavior, or Phase 6C-2 work.
- Phase 6C-2 executed one approved material fine-tuning run, `phase6c-cifar10-resnet18-layer4-finetune-001`, initialized from the accepted Phase 6B-2 best checkpoint at epoch `4` with SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- Phase 6C-2 used `finetune_layer4_head`, training only ResNet-18 `layer4 + fc` with Adam learning rate `0.0001`, weight decay `0.0`, no scheduler, batch size `64`, CPU, seed `20260820`, no augmentation, and a 3-epoch budget.
- Phase 6C-2 selected epoch `2` as the best checkpoint by minimum validation loss, restored that checkpoint, produced validation loss `0.246512`, validation accuracy `0.925800`, official test loss `0.272485`, and official test accuracy `0.914700`.
- Relative to the Phase 6B-2 frozen-feature reference test accuracy `0.856100`, Phase 6C-2 observed a single-run test accuracy delta of `+0.058600`, or `+5.86` percentage points.
- Phase 6C-2 phase check identified a stale Phase 6C-1/preflight label in the material-run `run_contract.json`; the top-level metadata was corrected to Phase 6C-2 without rerunning training, changing configuration, or generating a new experimental result.
- Phase 6C-2 is a single-run fine-tuning result. It does not establish seed/run-to-run variance, optimal unfreezing depth, optimal hyperparameters, architecture-only superiority, calibration, robustness/OOD behavior, diagnostics, inference, applied-domain behavior, or generalization beyond the evaluated CIFAR-10 experiment.
- Phase 7 is complete and accepted. It generated fixed-checkpoint metrics/calibration artifacts under ignored `outputs/phase7-evaluation-harness-and-calibration/`, preserved a 10-bin ECE configuration, verified validation/test sample alignment, and closed without training or checkpoint mutation.
- Phase 8 has not started and requires separate briefing, planning, and approval.
- Phase 4 is closed and accepted. The Phase 4 baseline artifacts and configuration remain the historical custom-CNN reference point for later controlled comparisons.
