# Phase 8B-2A Closeout - Validation Robustness Runner Preflight

Date: 2026-08-26

Status: Complete and accepted.

Phase 8B-2A implemented the validation-only runner/preflight contract for the future Phase 8B-2B fixed-checkpoint validation robustness sweep. It did not execute the full validation sweep and did not close Phase 8B as a whole.

## Scope

Phase 8B-2A was implementation/preflight only. It added:

- validation-only split enforcement;
- future Phase 8B-2B material-run contract generation;
- condition and checkpoint manifests;
- expected artifact schema;
- validation split count verification;
- sample-alignment preflight digest;
- output isolation checks;
- artifact validation;
- focused tests for the Phase 8B-2A invariants.

## Preserved Degradation Contract

Phase 8B-2A preserved the accepted Phase 8A degradation registry exactly. The registry version remains `1.0`.

The future material-run condition set is fixed at `21` conditions: one `clean` condition plus the 20 frozen Phase 8A v`1.0` degradation conditions.

Frozen profiles:

- `phase8a-gaussian-noise` v`1.0`: S1 `std=0.03`, S2 `std=0.06`, S3 `std=0.09`, S4 `std=0.12`, S5 `std=0.15`.
- `phase8a-gaussian-blur` v`1.0`: S1 `kernel_size=3, sigma=0.4`, S2 `kernel_size=3, sigma=0.7`, S3 `kernel_size=5, sigma=1.0`, S4 `kernel_size=5, sigma=1.3`, S5 `kernel_size=7, sigma=1.6`.
- `phase8a-brightness-shift` v`1.0`: S1 `delta=-0.08`, S2 `delta=-0.16`, S3 `delta=-0.24`, S4 `delta=-0.32`, S5 `delta=-0.40`.
- `phase8a-contrast-reduction` v`1.0`: S1 `factor=0.90`, S2 `factor=0.80`, S3 `factor=0.70`, S4 `factor=0.60`, S5 `factor=0.50`.

## Future Material-Run Contract

The generated Phase 8B-2B material-run contract is preserved under `outputs/phase8b2a-validation-robustness-runner-preflight/artifacts/phase8b2b_material_run_contract.json`.

It requires:

- run ID `phase8b2b-fixed-checkpoint-validation-robustness-sweep`;
- CIFAR-10 validation split `val` only;
- exactly `5,000` validation samples;
- exactly `3` fixed accepted checkpoints;
- exactly `21` conditions;
- exactly `63` expected model-condition rows;
- unchanged Phase 7 metric/calibration semantics.

Official test split requests are rejected in code by the Phase 8B-2 split gate.

## Fixed Checkpoint Identities

The three fixed checkpoint identities are:

- `phase4b-cifar10-custom-cnn-baseline-001`, checkpoint SHA-256 `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`.
- `phase6b2-cifar10-resnet18-frozen-feature-001`, checkpoint SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- `phase6c-cifar10-resnet18-layer4-finetune-001`, checkpoint SHA-256 `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`.

## Preflight Evidence

Generated preflight artifacts are preserved under ignored `outputs/phase8b2a-validation-robustness-runner-preflight/`.

Key artifacts:

- `phase8b2a_preflight_contract.json`;
- `phase8b2a_preflight_report.md`;
- `phase8b2a_result.json`;
- `artifacts/phase8b2_condition_manifest.json`;
- `artifacts/phase8b2_checkpoint_manifest.json`;
- `artifacts/phase8b2_expected_artifact_schema.json`;
- `artifacts/phase8b2_sample_alignment_preflight.json`;
- `artifacts/phase8b2_validation_preflight.json`;
- `artifacts/phase8b2a_artifact_validation.json`;
- `artifacts/phase8b2b_material_run_contract.json`.

The recorded validation sample-alignment digest is:

`8182140619d7359ac287d1496b2e75415ffb5b2f26a042636ec075fa68beeb9e`

Focused Phase 8B tests passed with `18` tests. The canonical deterministic suite passed after documentation-only closeout with `119` tests and `1` skipped via `powershell -ExecutionPolicy Bypass -File scripts\test.ps1`.

## Interpretation Boundary

Phase 8B-2A produced no robustness conclusion.

It did not execute the full validation robustness sweep. It did not produce model performance robustness curves, validation robustness findings, official test robustness findings, OOD findings, or model-selection evidence.

## Explicit Exclusions

Phase 8B-2A did not include:

- full validation robustness sweep execution;
- official test robustness evaluation;
- material robustness results;
- OOD or cross-source evaluation;
- training;
- tuning;
- model selection;
- checkpoint modification;
- Phase 8B-2B execution;
- Phase 8C implementation or execution.

## Accepted State

Phase 8B-2A is complete and accepted. Phase 8B as a whole is not complete. Phase 8B-2B has not started and requires separate material-run approval. Phase 8C has not started.
