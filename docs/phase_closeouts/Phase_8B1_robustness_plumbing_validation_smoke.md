# Phase 8B-1 Closeout - Robustness Plumbing Validation Smoke

Date: 2026-08-25

Status: Complete and accepted.

Phase 8B-1 implemented the robustness plumbing and validation-smoke boundary needed before any material Phase 8B-2 robustness sweep. It did not close Phase 8B as a whole.

## Scope

Phase 8B-1 was limited to:

- condition contracts for clean and degraded evaluation conditions;
- reuse of the accepted Phase 8A degradation registry;
- tiny validation-only smoke over fixed checkpoint references;
- sample and label alignment verification;
- raw degraded tensor equivalence verification before model-specific preprocessing;
- preprocessing-order verification for CustomCNN and ResNet-18;
- clean-vs-degraded delta plumbing;
- artifact validation;
- approximate runtime estimation for a future validation sweep.

## Frozen Degradation Contract

Phase 8B-1 preserved the accepted Phase 8A degradation registry exactly. The registry version remains `1.0`.

The Phase 8B-1 condition set contains `21` conditions: one clean condition plus all four Phase 8A profiles across severities `S1` through `S5`.

Frozen profiles and parameters:

- `phase8a-gaussian-noise` v`1.0`: S1 `std=0.03`, S2 `std=0.06`, S3 `std=0.09`, S4 `std=0.12`, S5 `std=0.15`.
- `phase8a-gaussian-blur` v`1.0`: S1 `kernel_size=3, sigma=0.4`, S2 `kernel_size=3, sigma=0.7`, S3 `kernel_size=5, sigma=1.0`, S4 `kernel_size=5, sigma=1.3`, S5 `kernel_size=7, sigma=1.6`.
- `phase8a-brightness-shift` v`1.0`: S1 `delta=-0.08`, S2 `delta=-0.16`, S3 `delta=-0.24`, S4 `delta=-0.32`, S5 `delta=-0.40`.
- `phase8a-contrast-reduction` v`1.0`: S1 `factor=0.90`, S2 `factor=0.80`, S3 `factor=0.70`, S4 `factor=0.60`, S5 `factor=0.50`.

No degradation parameter or condition definition was changed based on model output.

## Fixed References

Phase 8B-1 preserved the fixed checkpoint identities used for comparison plumbing:

- `phase4b-cifar10-custom-cnn-baseline-001`, checkpoint SHA-256 `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`.
- `phase6b2-cifar10-resnet18-frozen-feature-001`, checkpoint SHA-256 `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`.
- `phase6c-cifar10-resnet18-layer4-finetune-001`, checkpoint SHA-256 `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`.

## Verification Evidence

The formal phase check is preserved at `docs/phase_checks/Phase_8B1_robustness_plumbing_validation_smoke_check.md`.

Generated smoke artifacts are preserved under ignored `outputs/phase8b1-robustness-plumbing-validation-smoke/`.

Key verification results:

- sample IDs and labels remained aligned across all three fixed references and all `21` conditions;
- raw degraded `[0,1]` unit tensors were identical before model-specific preprocessing for the same sample, profile, severity, and seed policy;
- CustomCNN applied its existing `phase4-cifar10-normalization` preprocessing after degradation;
- ResNet-18 applied its existing `phase6a-resnet18-imagenet1k-v1-preprocessing` after degradation;
- clean and degraded condition identities were distinct and clean-vs-degraded delta plumbing was verified;
- focused Phase 8B-1 tests passed: `8` passed;
- canonical deterministic suite passed at phase check: `109` passed, `1` skipped;
- the final documentation-only closeout suite passed: `109` passed, `1` skipped.

## Runtime Estimate

The Phase 8B-1 smoke estimated a future 5,000-sample validation sweep at `4675.07` seconds, or approximately `77.92` minutes, for `21` conditions and `3` models.

This is an estimate only, derived from tiny validation-smoke timing. It is not a compute commitment and should be reconfirmed before any Phase 8B-2 material run.

## Interpretation Boundary

Phase 8B-1 produced no robustness conclusion.

The smoke metrics are plumbing evidence only. They are not official robustness results, not validation robustness findings, not official test robustness findings, not semantic label-preservation evidence, and not model-selection evidence.

## Explicit Exclusions

Phase 8B-1 did not include:

- official test robustness evaluation;
- material robustness sweep;
- OOD or cross-source evaluation;
- training;
- tuning;
- model selection;
- checkpoint modification;
- Phase 8B-2 implementation or execution;
- Phase 8C implementation or execution.

## Accepted State

Phase 8B-1 is complete and accepted. Phase 8B as a whole is not complete. Phase 8B-2 and Phase 8C have not started and require separate planning and approval.
