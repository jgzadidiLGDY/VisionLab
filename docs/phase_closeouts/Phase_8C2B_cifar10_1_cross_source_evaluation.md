# Phase 8C-2B Closeout - CIFAR-10.1 v6 Cross-Source Evaluation

Date: 2026-08-27

Status: Complete and accepted.

Phase 8C-2B executed the approved single material CIFAR-10.1 v6 cross-source evaluation for VisionLab. This closeout does not close Phase 8 as a whole; the overall Phase 8 closeout remains a separate review step.

## Scope

Phase 8C-2B was limited to fixed-checkpoint cross-source evaluation on the registered CIFAR-10.1 v6 dataset.

The run used:

- run ID `phase8c2b-cifar10-1-v6-fixed-checkpoint-cross-source-evaluation`;
- dataset ID `cifar10-1`;
- version `v6`;
- split `cross_source_test` only;
- exactly `2,000` CIFAR-10.1 v6 samples per checkpoint;
- exactly `3` fixed accepted checkpoint references;
- unchanged Phase 7 metric/calibration semantics;
- the previously accepted Phase 7 official CIFAR-10 test metrics as historical fixed-reference summaries only.

The official CIFAR-10 test split was not rerun.

## Dataset Identity

The accepted Phase 8C-2B artifacts preserve the CIFAR-10.1 v6 identity:

- Dataset ID: `cifar10-1`
- Version: `v6`
- Split identity: `cross_source_test`
- Usage boundary: cross-source evaluation-only; never train, tune, or select checkpoints
- Sample count: `2,000`
- Image structure: `32 x 32 x 3`
- Class mapping: exact CIFAR-10 class order: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- Class distribution: `200` examples per class
- Data SHA-256: `2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6`
- Labels SHA-256: `ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448`
- Sample-label digest: `2afa813c387e578086d1f0aeeb1b9674e352c73c4690b89d69385aedca3e8b75`

No substitute dataset or alternate CIFAR-10.1 version was used.

## Fixed Checkpoints

The three fixed checkpoint identities were preserved:

- `phase4b-cifar10-custom-cnn-baseline-001`, checkpoint SHA-256 `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`.
- `phase6b2-cifar10-resnet18-frozen-feature-001`, checkpoint SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- `phase6c-cifar10-resnet18-layer4-finetune-001`, checkpoint SHA-256 `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`.

No checkpoint was modified, replaced, reselected, or re-exported.

## Preprocessing and Alignment

The approved raw-input invariant was preserved:

- each model receives the same raw CIFAR-10.1 unnormalized RGB unit tensor in `C x H x W` format before model-specific preprocessing;
- CustomCNN applies the existing Phase 4 CIFAR-10 normalization after the raw unit tensor;
- ResNet-18 applies the existing ImageNet preprocessing after the raw unit tensor.

Sample, label, and source alignment passed across all three fixed checkpoints. The alignment artifact records:

- checkpoint count: `3`;
- sample count per checkpoint: `2,000`;
- sample IDs and labels aligned across checkpoints: `true`;
- source IDs aligned across checkpoints: `true`;
- sample-label digest: `2afa813c387e578086d1f0aeeb1b9674e352c73c4690b89d69385aedca3e8b75`.

## Cross-Source Metrics

The generated metrics artifact contains exactly `3` cross-source metric rows, one per fixed checkpoint. Each row contains exactly `2,000` evaluated CIFAR-10.1 v6 examples.

| Checkpoint | Loss | Accuracy | Balanced Accuracy | Macro F1 | ECE | Average Confidence | Incorrect Average Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase4b-cifar10-custom-cnn-baseline-001` | `1.427107` | `0.512000` | `0.512000` | `0.503911` | `0.074457` | `0.586457` | `0.498562` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `0.741785` | `0.745500` | `0.745500` | `0.744516` | `0.052823` | `0.798323` | `0.639411` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `0.587721` | `0.825000` | `0.825000` | `0.824147` | `0.090583` | `0.915423` | `0.759730` |

## Historical-Reference Deltas

The generated delta artifact contains exactly `3` historical-reference delta rows, one per fixed checkpoint.

Deltas are defined as:

`CIFAR-10.1 v6 cross-source metric - previously accepted Phase 7 official CIFAR-10 test metric for the same fixed checkpoint`

These deltas are not paired-sample deltas. The Phase 7 official CIFAR-10 test metrics are historical fixed-reference summaries only and were not rerun in Phase 8C-2B.

| Checkpoint | Accuracy Delta | Balanced Accuracy Delta | Macro F1 Delta | ECE Delta | Average Confidence Delta | Incorrect Average Confidence Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase4b-cifar10-custom-cnn-baseline-001` | `-0.123900` | `-0.123900` | `-0.118489` | `0.068700` | `-0.053646` | `-0.008255` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `-0.110600` | `-0.110600` | `-0.111004` | `0.041419` | `-0.053385` | `0.030323` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `-0.089700` | `-0.089700` | `-0.089842` | `0.052895` | `-0.036965` | `0.008507` |

## Artifacts

Phase 8C-2B artifacts are preserved under ignored `outputs/phase8c2b-cifar10-1-v6-fixed-checkpoint-cross-source-evaluation/`.

Key artifacts:

- `phase8c2b_material_run_contract.json`;
- `phase8c2b_result.json`;
- `phase8c2_cross_source_report.md`;
- `artifacts/phase8c2_cross_source_metrics.csv`;
- `artifacts/phase8c2_cross_source_deltas.csv`;
- `artifacts/phase8c2_sample_alignment.json`;
- `artifacts/phase8c2_artifact_validation.json`;
- `artifacts/phase8c2b_cifar10_1_dataset_identity.json`;
- `artifacts/phase8c2b_checkpoint_manifest.json`;
- `artifacts/phase8c2b_historical_phase7_test_reference.json`;
- `artifacts/phase8c2b_preprocessing_verification.json`;
- `artifacts/phase8c2b_runtime_projection.json`.

Artifact validation passed with `3` cross-source metric rows, `3` historical-reference delta rows, and `2,000` samples per model row.

## Audit Note

The Phase 8C-2B phase check recorded one non-blocking audit note: full raw-input hashes were not stored for every material sample/model pair. The raw-input invariant is preserved by the dataset/view contract and verified by preprocessing evidence on a fixed subset, while the material artifacts verify evaluated counts and sample/label/source alignment. The experiment was not rerun to add full per-sample/model raw-input hashes.

## Interpretation Boundary

Phase 8C-2B provides CIFAR-10.1 v6 cross-source/distribution-shift evidence only for the three fixed accepted checkpoints.

This phase does not establish:

- general OOD detection;
- deployment reliability;
- real-world robustness;
- general model superiority;
- label correctness for CIFAR-10.1 v6;
- seed or run-to-run variance;
- causal explanations for the observed cross-source metric changes.

## Explicit Exclusions

Phase 8C-2B did not include:

- official CIFAR-10 test evaluation rerun;
- additional OOD dataset evaluation;
- training;
- tuning;
- model selection;
- checkpoint mutation;
- checkpoint replacement or re-export;
- Phase 9 work.

## Verification

Pre-run Phase 8C-2A focused tests passed with `11` tests.

Pre-run canonical deterministic suite passed with `150` tests and `1` skipped.

Post-run focused Phase 8C tests passed with `18` tests.

Post-run canonical deterministic suite passed with `150` tests and `1` skipped.

The formal Phase 8C-2B phase-check report was accepted by the builder. After documentation-only closeout updates, the canonical deterministic suite passed with `150` tests and `1` skipped via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1`.

## Accepted State

Phase 8C-2B is complete and accepted. Phase 8C cross-source evaluation is complete through the registered Phase 8C-1 dataset boundary, the Phase 8C-2A preflight implementation boundary, and the accepted Phase 8C-2B material-run boundary. Overall Phase 8 is not closed; the Phase 8 closeout remains a separate review step.
