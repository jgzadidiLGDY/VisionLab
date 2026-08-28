# Phase 8 Closeout - Robustness and OOD Evaluation

Date: 2026-08-27

Status: Complete and accepted.

Phase 8 established VisionLab's robustness and cross-source evaluation layer through accepted Phase 8A, Phase 8B, and Phase 8C subphases. The phase produced versioned degradation contracts, CIFAR-10 validation degradation robustness evidence, and CIFAR-10.1 v6 cross-source/distribution-shift evidence for the three fixed accepted checkpoints.

No additional experiment, evaluation, model inference, training, tuning, or dataset acquisition was performed as part of this closeout.

## Scope Summary

Phase 8 included three bounded tracks:

- Phase 8A: degradation registry, deterministic transform contracts, visual QA, and tiny smoke.
- Phase 8B: fixed-checkpoint CIFAR-10 validation degradation robustness evaluation across clean and registered degraded conditions.
- Phase 8C: CIFAR-10.1 v6 registration and fixed-checkpoint cross-source evaluation.

Phase 8 did not include training, tuning, model selection, checkpoint modification, additional OOD/cross-source datasets, applied-domain behavior, inference-surface work, failure analysis, diagnostics, or Phase 9 work.

## Phase 8A Degradation Contract

Phase 8A established registry `visionlab-phase8a-degradation-profiles` version `1.0`.

The frozen degradation profiles are:

| Profile ID | Version | S1 | S2 | S3 | S4 | S5 |
| --- | --- | --- | --- | --- | --- | --- |
| `phase8a-gaussian-noise` | `1.0` | `std=0.03` | `std=0.06` | `std=0.09` | `std=0.12` | `std=0.15` |
| `phase8a-gaussian-blur` | `1.0` | `kernel_size=3`, `sigma=0.4` | `kernel_size=3`, `sigma=0.7` | `kernel_size=5`, `sigma=1.0` | `kernel_size=5`, `sigma=1.3` | `kernel_size=7`, `sigma=1.6` |
| `phase8a-brightness-shift` | `1.0` | `delta=-0.08` | `delta=-0.16` | `delta=-0.24` | `delta=-0.32` | `delta=-0.40` |
| `phase8a-contrast-reduction` | `1.0` | `factor=0.90` | `factor=0.80` | `factor=0.70` | `factor=0.60` | `factor=0.50` |

The transform contract uses unnormalized RGB unit tensors in `C x H x W` format in `[0, 1]` before model-specific preprocessing. Outputs preserve shape, RGB channel count, finite values, and `[0, 1]` range without mutating the input tensor.

Gaussian noise is stochastic but deterministic under the registered seed policy. Its effective seed is derived from profile identity, version, severity, base seed, `sample_id`, and `source_id`. Deterministic transforms ignore the seed.

The degraded sample wrapper preserves `sample_id`, `label`, `split`, and `source_id`, and propagates degradation profile/version/severity/seed identity. Existing Phase 4B, Phase 6B-2, and Phase 6C-2 preprocessing contracts were not modified.

Builder visual QA: accepted. The builder manually reviewed the four generated degradation grids and confirmed they looked visually correct with sensible `S1` to `S5` progression. Visual QA remains qualitative only and does not establish semantic label preservation or robustness.

## Phase 8B Validation Robustness Evidence

Phase 8B evaluated fixed checkpoints on the registered CIFAR-10 validation split only.

The accepted material run was `phase8b2b-fixed-checkpoint-validation-robustness-sweep` and used:

- CIFAR-10 validation split `val` only;
- exactly `5,000` validation samples;
- exactly `3` fixed accepted checkpoints;
- exactly `21` conditions: `clean` plus the `20` frozen Phase 8A v`1.0` degraded conditions;
- exactly `63` model-condition metric rows;
- exactly `63` clean-delta rows;
- exactly `60` severity-curve rows excluding clean;
- unchanged Phase 7 metric/calibration semantics.

The official CIFAR-10 test split was not evaluated for Phase 8B robustness.

### Phase 8B Clean Validation Baselines

| Checkpoint | Accuracy | Balanced Accuracy | Macro F1 | ECE | Avg Confidence | Incorrect Avg Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase4b-cifar10-custom-cnn-baseline-001` | `0.630200` | `0.630200` | `0.618902` | `0.018083` | `0.638256` | `0.495998` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `0.864600` | `0.864600` | `0.864046` | `0.013265` | `0.851645` | `0.608570` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `0.925800` | `0.925800` | `0.925510` | `0.027550` | `0.953350` | `0.744843` |

### Phase 8B S5 Accuracy Deltas From Clean

These rows summarize the strongest registered severity for each degradation family. Full `S1` through `S5` curves are preserved in the Phase 8B artifacts.

| Checkpoint | Profile | S5 Accuracy Delta | S5 Macro F1 Delta | S5 ECE Delta |
| --- | --- | ---: | ---: | ---: |
| `phase4b-cifar10-custom-cnn-baseline-001` | `phase8a-gaussian-noise` | `-0.308000` | `-0.377936` | `0.349056` |
| `phase4b-cifar10-custom-cnn-baseline-001` | `phase8a-gaussian-blur` | `-0.297400` | `-0.333087` | `0.134989` |
| `phase4b-cifar10-custom-cnn-baseline-001` | `phase8a-brightness-shift` | `-0.248000` | `-0.247660` | `0.118380` |
| `phase4b-cifar10-custom-cnn-baseline-001` | `phase8a-contrast-reduction` | `-0.214800` | `-0.211723` | `0.072607` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `phase8a-gaussian-noise` | `-0.743200` | `-0.777399` | `0.430458` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `phase8a-gaussian-blur` | `-0.632800` | `-0.703195` | `0.274203` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `phase8a-brightness-shift` | `-0.197000` | `-0.194238` | `0.011816` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `phase8a-contrast-reduction` | `-0.090600` | `-0.088330` | `-0.004161` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `phase8a-gaussian-noise` | `-0.754200` | `-0.809673` | `0.434951` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `phase8a-gaussian-blur` | `-0.664200` | `-0.685110` | `0.444857` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `phase8a-brightness-shift` | `-0.178200` | `-0.171110` | `0.090074` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `phase8a-contrast-reduction` | `-0.058600` | `-0.058052` | `0.027419` |

Phase 8B findings are controlled CIFAR-10 validation degradation evidence only. They support condition-specific observations for the three fixed checkpoints under the registered degradation contract. They do not establish official test robustness, OOD robustness, semantic label preservation under degradation, production reliability, universal model superiority, or seed/run-to-run robustness variance.

## Phase 8C Cross-Source Evidence

Phase 8C registered CIFAR-10.1 v6 and evaluated the three fixed checkpoints on that dataset only.

The accepted material run was `phase8c2b-cifar10-1-v6-fixed-checkpoint-cross-source-evaluation` and used:

- dataset ID `cifar10-1`;
- version `v6`;
- split `cross_source_test` only;
- exactly `2,000` samples per checkpoint;
- exactly `3` fixed accepted checkpoints;
- exactly `3` cross-source metric rows;
- exactly `3` historical-reference delta rows;
- unchanged Phase 7 metric/calibration semantics.

CIFAR-10.1 v6 identity:

- data SHA-256: `2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6`;
- labels SHA-256: `ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448`;
- sample-label digest: `2afa813c387e578086d1f0aeeb1b9674e352c73c4690b89d69385aedca3e8b75`;
- class mapping: exact CIFAR-10 class order;
- class distribution: `200` examples per class.

### Phase 8C Cross-Source Metrics

| Checkpoint | Loss | Accuracy | Balanced Accuracy | Macro F1 | ECE | Avg Confidence | Incorrect Avg Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase4b-cifar10-custom-cnn-baseline-001` | `1.427107` | `0.512000` | `0.512000` | `0.503911` | `0.074457` | `0.586457` | `0.498562` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `0.741785` | `0.745500` | `0.745500` | `0.744516` | `0.052823` | `0.798323` | `0.639411` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `0.587721` | `0.825000` | `0.825000` | `0.824147` | `0.090583` | `0.915423` | `0.759730` |

### Phase 8C Historical-Reference Deltas

Phase 8C deltas are defined as:

`CIFAR-10.1 v6 cross-source metric - previously accepted Phase 7 official CIFAR-10 test metric for the same fixed checkpoint`

These are not paired-sample deltas. The Phase 7 official CIFAR-10 test metrics were used only as historical fixed-reference summaries and were not rerun in Phase 8C.

| Checkpoint | Accuracy Delta | Balanced Accuracy Delta | Macro F1 Delta | ECE Delta | Avg Confidence Delta | Incorrect Avg Confidence Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `phase4b-cifar10-custom-cnn-baseline-001` | `-0.123900` | `-0.123900` | `-0.118489` | `0.068700` | `-0.053646` | `-0.008255` |
| `phase6b2-cifar10-resnet18-frozen-feature-001` | `-0.110600` | `-0.110600` | `-0.111004` | `0.041419` | `-0.053385` | `0.030323` |
| `phase6c-cifar10-resnet18-layer4-finetune-001` | `-0.089700` | `-0.089700` | `-0.089842` | `0.052895` | `-0.036965` | `0.008507` |

Phase 8C findings are CIFAR-10.1 v6 cross-source/distribution-shift evidence only. They do not establish general OOD detection, deployment reliability, real-world robustness, label correctness, universal model superiority, or causal explanations for the observed distribution-shift behavior.

## Fixed Checkpoint Identities

All Phase 8 material evaluations used the same three fixed accepted checkpoint identities:

- `phase4b-cifar10-custom-cnn-baseline-001`, SHA-256 `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`.
- `phase6b2-cifar10-resnet18-frozen-feature-001`, SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- `phase6c-cifar10-resnet18-layer4-finetune-001`, SHA-256 `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`.

No checkpoint was modified, replaced, reselected, or re-exported during Phase 8.

## Interpretation Guide

Phase 8 separates related but distinct model behaviors:

- Clean discrimination: performance on clean in-distribution CIFAR-10 validation or official CIFAR-10 test references.
- Corruption robustness: performance changes under registered degradations on the CIFAR-10 validation split.
- Calibration: confidence-quality summaries such as 10-bin ECE, average confidence, and incorrect average confidence.
- Cross-source distribution shift: performance on CIFAR-10.1 v6, compared against previously accepted Phase 7 official CIFAR-10 test summaries.

These are complementary measurements, not interchangeable proof of one another. Strong clean accuracy does not automatically imply corruption robustness, calibration quality, or cross-source stability. Lower degradation loss does not prove general OOD detection. CIFAR-10.1 v6 deltas do not prove real-world reliability.

## Audit Note

Phase 8C-2B preserved the non-blocking audit note that full raw-input hashes were not stored for every material sample/model pair. The raw-input invariant is preserved by the dataset/view contract and verified by preprocessing evidence on a fixed subset, while the material artifacts verify evaluated counts and sample/label/source alignment. The experiment was not rerun to add full per-sample/model raw-input hashes.

## Artifacts

Key Phase 8 closeouts:

- `docs/phase_closeouts/Phase_8A_degradation_registry_visual_qa_and_tiny_smoke.md`;
- `docs/phase_closeouts/Phase_8B1_robustness_plumbing_validation_smoke.md`;
- `docs/phase_closeouts/Phase_8B2A_validation_robustness_runner_preflight.md`;
- `docs/phase_closeouts/Phase_8B2B_fixed_checkpoint_validation_robustness_sweep.md`;
- `docs/phase_closeouts/Phase_8C1_cifar10_1_registration_visual_qa_and_tiny_smoke.md`;
- `docs/phase_closeouts/Phase_8C2B_cifar10_1_cross_source_evaluation.md`.

Key Phase 8 output directories:

- `outputs/phase8a-degradation-registry-visual-qa-tiny-smoke/`;
- `outputs/phase8b1-robustness-plumbing-validation-smoke/`;
- `outputs/phase8b2a-validation-robustness-runner-preflight/`;
- `outputs/phase8b2b-fixed-checkpoint-validation-robustness-sweep/`;
- `outputs/phase8c1-cifar10-1-registration-visual-qa-tiny-smoke/`;
- `outputs/phase8c2a-cifar10-1-v6-cross-source-preflight/`;
- `outputs/phase8c2b-cifar10-1-v6-fixed-checkpoint-cross-source-evaluation/`.

## Explicit Non-Claims

Phase 8 does not prove:

- general real-world robustness;
- general OOD detection;
- deployment or production reliability;
- universal model superiority;
- semantic label preservation for every degraded sample;
- label correctness for every CIFAR-10.1 v6 sample;
- seed or run-to-run variance;
- that Phase 8 results should be used for model selection.

## Preserved Exclusions

Across Phase 8:

- no training occurred;
- no tuning occurred;
- no checkpoint was modified;
- no model selection was performed using Phase 8 results;
- no official CIFAR-10 test robustness evaluation occurred;
- no additional OOD/cross-source datasets were evaluated;
- no applied-domain selection or applied-domain implementation occurred;
- no Phase 9 failure analysis or interpretability work occurred.

## Verification

Phase 8A focused tests and canonical suite passed at closeout.

Phase 8B focused tests and canonical suite passed at subphase closeouts. Phase 8B-2B post-run canonical suite passed with `120` tests and `1` skipped.

Phase 8C focused tests and canonical suite passed at subphase closeouts. Phase 8C-2B post-run canonical suite passed with `150` tests and `1` skipped.

After this documentation-only Phase 8 closeout, the canonical deterministic suite passed with `150` tests and `1` skipped via:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
```

## Accepted State

Phase 8 - Robustness and OOD Evaluation is complete and accepted.

Phase 9 has not started. The next boundary is separate Phase 9 planning and approval.
