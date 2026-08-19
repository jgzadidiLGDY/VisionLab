# VisionLab Phase Catalog

Status: Phase 4 complete; accepted.

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
| 5 | Augmentation and Generalization Controls | Not started | To be added |
| 6 | Transfer Learning and Fine-Tuning | Not started | To be added |
| 7 | Evaluation Harness and Calibration | Not started | To be added |
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
- No inference surface, pretrained model, augmentation experiment, calibration, robustness, OOD, diagnostics, or applied-domain behavior exists.
- Phase 4 is closed and accepted. The Phase 4 baseline artifacts and configuration are the reference point for later controlled comparisons.
