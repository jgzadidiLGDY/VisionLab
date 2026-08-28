# Phase 8B-2B Closeout - Fixed-Checkpoint Validation Robustness Sweep

Date: 2026-08-26

Status: Complete and accepted.

Phase 8B-2B executed the approved material validation-only fixed-checkpoint robustness sweep for VisionLab. It does not close Phase 8 as a whole because Phase 8C has not started.

## Scope

Phase 8B-2B was limited to fixed-checkpoint robustness evaluation on the registered CIFAR-10 validation split.

The run used:

- run ID `phase8b2b-fixed-checkpoint-validation-robustness-sweep`;
- CIFAR-10 validation split `val` only;
- exactly `5,000` validation samples;
- exactly `3` fixed accepted checkpoint references;
- exactly `21` conditions;
- frozen Phase 8A degradation registry version `1.0`;
- unchanged Phase 7 metric/calibration semantics.

The official CIFAR-10 test split was not evaluated.

## Condition Contract

The Phase 8B-2B condition manifest preserved the exact accepted Phase 8A v`1.0` degradation registry. The `21` conditions were:

- `clean`;
- `phase8a-gaussian-noise` v`1.0`, severities S1-S5: `std=0.03`, `0.06`, `0.09`, `0.12`, `0.15`;
- `phase8a-gaussian-blur` v`1.0`, severities S1-S5: `kernel_size=3, sigma=0.4`; `kernel_size=3, sigma=0.7`; `kernel_size=5, sigma=1.0`; `kernel_size=5, sigma=1.3`; `kernel_size=7, sigma=1.6`;
- `phase8a-brightness-shift` v`1.0`, severities S1-S5: `delta=-0.08`, `-0.16`, `-0.24`, `-0.32`, `-0.40`;
- `phase8a-contrast-reduction` v`1.0`, severities S1-S5: `factor=0.90`, `0.80`, `0.70`, `0.60`, `0.50`.

No degradation parameter was modified.

## Fixed Checkpoints

The three fixed checkpoint identities were preserved:

- `phase4b-cifar10-custom-cnn-baseline-001`, checkpoint SHA-256 `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`.
- `phase6b2-cifar10-resnet18-frozen-feature-001`, checkpoint SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- `phase6c-cifar10-resnet18-layer4-finetune-001`, checkpoint SHA-256 `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`.

No checkpoint was modified.

## Artifacts

Phase 8B-2B artifacts are preserved under ignored `outputs/phase8b2b-fixed-checkpoint-validation-robustness-sweep/`.

Key artifacts:

- `phase8b2b_material_run_contract.json`;
- `phase8b2b_result.json`;
- `phase8b2_validation_robustness_report.md`;
- `artifacts/phase8b2_condition_manifest.json`;
- `artifacts/phase8b2_checkpoint_manifest.json`;
- `artifacts/phase8b2_validation_metrics.csv`;
- `artifacts/phase8b2_clean_delta_metrics.csv`;
- `artifacts/phase8b2_severity_curves.csv`;
- `artifacts/phase8b2_sample_alignment.json`;
- `artifacts/phase8b2_artifact_validation.json`.

Artifact validation passed.

## Evidence Summary

The accepted Phase 8B-2B artifacts record:

- `63` model-condition metric rows;
- `63` clean-delta rows;
- `60` severity-curve rows excluding clean;
- `3` clean baseline rows;
- `5,000` validation examples for every model-condition row;
- `21` unique conditions;
- `3` fixed checkpoint references.

Sample alignment passed across all three fixed checkpoints and all 21 conditions. The preserved preflight sample-alignment digest is:

`8182140619d7359ac287d1496b2e75415ffb5b2f26a042636ec075fa68beeb9e`

The generated validation robustness report includes condition-specific metrics and clean deltas.

## Interpretation Boundary

Phase 8B-2B provides validation-only fixed-checkpoint robustness observations under the registered CIFAR-10 validation split and accepted Phase 8A degradations.

Allowed interpretation is limited to condition-specific validation metrics and clean deltas for the three fixed checkpoints.

This phase does not establish:

- official test robustness;
- OOD or cross-source robustness;
- semantic label preservation under degradation;
- production reliability;
- model superiority in general;
- seed or run-to-run robustness variance.

## Explicit Exclusions

Phase 8B-2B did not include:

- official test robustness evaluation;
- OOD or cross-source evaluation;
- training;
- tuning;
- model selection;
- checkpoint modification;
- degradation parameter modification;
- Phase 8C implementation or execution.

## Verification

Pre-run canonical deterministic suite passed with `119` tests and `1` skipped.

Post-run focused Phase 8B tests passed with `19` tests.

Post-run canonical deterministic suite passed with `120` tests and `1` skipped.

After documentation-only closeout updates, the canonical deterministic suite passed with `120` tests and `1` skipped via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1`.

## Accepted State

Phase 8B-2B is complete and accepted. Phase 8B validation robustness evaluation is complete through the accepted Phase 8B-1, Phase 8B-2A, and Phase 8B-2B boundaries. Phase 8 as a whole is not complete because Phase 8C has not started.
