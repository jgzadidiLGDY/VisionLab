# Phase 8C-2A Phase-Check Report - Cross-Source Evaluation Preflight

Date: 2026-08-27

Status: Implemented; awaiting builder review. This phase-check report is documentation/preflight evidence only and is not Phase 8C-2B approval.

## 1. Overall Status

Phase 8C-2A implemented the approved cross-source evaluation runner contracts, preflight artifacts, tiny smoke, artifact validation, and focused tests. It stopped at the review boundary.

No material CIFAR-10.1 v6 evaluation was run. No official CIFAR-10 test split rerun occurred. No model checkpoint inference, training, tuning, model selection, checkpoint mutation, Phase 8C-2B work, or Phase 9 work occurred.

## 2. Intended-Shape Assessment

The intended future Phase 8C-2B material run is represented, but not executed:

- future run ID: `phase8c2b-cifar10-1-v6-fixed-checkpoint-cross-source-evaluation`;
- dataset: `cifar10-1` version `v6`;
- split: `cross_source_test`;
- sample count: `2,000`;
- sample-label digest: `2afa813c387e578086d1f0aeeb1b9674e352c73c4690b89d69385aedca3e8b75`;
- fixed checkpoint references: `3`;
- expected future model metric rows: `3`;
- expected future cross-source delta rows: `3`.

The Phase 7 official CIFAR-10 test metrics are used only as previously accepted historical reference summaries for future delta contracts. They were not rerun or regenerated.

## 3. Key Findings

- CIFAR-10.1 v6 identity and digest checks passed.
- CIFAR-10 class-map compatibility is preserved exactly.
- Evaluation-only usage is enforced in the dataset contract.
- Checkpoint manifest preserves the three accepted fixed checkpoint identities and SHA-256 values.
- Preprocessing verification passed on a six-sample tiny subset: CustomCNN uses Phase 4 CIFAR-10 normalization after the raw CIFAR-10.1 unit tensor, and ResNet-18 uses ImageNet preprocessing after the same raw unit tensor.
- Historical Phase 7 official-test reference rows are loaded from `outputs/phase7-evaluation-harness-and-calibration/artifacts/phase7_comparison_table.csv` and labeled as accepted summaries, not new paired predictions.
- Tiny smoke metrics are explicitly marked non-material and non-conclusive, and no model checkpoint evaluation is performed by the smoke.
- Artifact validation passed for the generated Phase 8C-2A artifacts.

## 4. Builder-Codex Context Check

Phase 8C-2A is plumbing and preflight only. It establishes readiness for a later material CIFAR-10.1 v6 cross-source evaluation plan, but it does not itself answer how any model performs on CIFAR-10.1 v6.

The correct interpretation remains narrow: no OOD-detection claim, robustness claim, deployment-reliability claim, or general model-superiority claim is supported by Phase 8C-2A.

## 5. Required Follow-Ups

Before Phase 8C-2B, the builder should approve a material-run prompt that fixes:

- the exact output directory and run ID;
- the full `2,000`-sample CIFAR-10.1 v6 evaluation boundary;
- the three fixed checkpoints and SHA-256 values;
- the metrics and delta columns;
- the historical Phase 7 official-test reference semantics;
- runtime guardrails and stop conditions.

No repair is required before review.

## 6. Next-Phase Readiness

Phase 8C-2A is ready for builder review. Phase 8C-2B should not start without separate material-run approval.

## 7. Verification

Focused Phase 8C-2A tests passed:

```text
Ran 11 tests in 0.760s
OK
```

Canonical deterministic suite passed:

```text
Ran 143 tests in 10.220s
OK (skipped=1)
```

## 8. Artifact Inventory

Generated under ignored `outputs/phase8c2a-cifar10-1-v6-cross-source-preflight/`:

- `phase8c2a_result.json`;
- `phase8c2a_preflight_report.md`;
- `artifacts/phase8c2a_material_contract.json`;
- `artifacts/phase8c2a_cifar10_1_dataset_identity.json`;
- `artifacts/phase8c2a_checkpoint_manifest.json`;
- `artifacts/phase8c2a_historical_phase7_test_reference.json`;
- `artifacts/phase8c2a_preprocessing_verification.json`;
- `artifacts/phase8c2a_tiny_smoke.json`;
- `artifacts/phase8c2a_expected_artifact_schema.json`;
- `artifacts/phase8c2a_artifact_validation.json`.
