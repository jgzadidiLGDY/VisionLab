# Phase-Check Report - Phase 7 Evaluation Harness and Calibration

## 1. Overall Status

**Ready with small follow-ups.**

Phase 7 is in the intended implementation shape: it evaluates fixed accepted checkpoints, preserves prediction-level evidence, computes class-wise metrics and calibration diagnostics, verifies hard sample alignment, and keeps validation/test semantics separate. The remaining follow-up is builder review of the generated visual and report artifacts before formal closeout. No blocking implementation issue was found in this formal phase-check pass.

## 2. Intended-Shape Assessment

Phase 7 achieved the approved objective of replacing headline accuracy with a structured, reusable fixed-checkpoint evaluation system. It stayed inside the approved boundary: no training, tuning, augmentation change, checkpoint regeneration, robustness/OOD sweep, failure-gallery work, diagnostics, inference work, or applied-domain selection was performed.

The repository remains coherent. The evaluation package now owns metric, calibration, prediction-record, and lightweight plot helpers; the Phase 7 experiment module owns the fixed-checkpoint orchestration and comparison invariants; status documents correctly say Phase 7 is implemented/evaluated but pending builder review rather than accepted or closed.

## 3. Key Findings

### Vision or ML Correctness

- Metrics are computed from prediction-level evidence rather than only summary artifacts.
- Phase 7 preserves logits, probabilities, true/predicted indices, confidence, correctness, sample ID, split, and source ID in enriched prediction records.
- Calibration is explicitly based on maximum predicted class probability and a `10`-bin ECE configuration recorded in `phase7_contract.json`.
- ECE bin semantics are documented: `min(int(confidence * num_bins), num_bins - 1)`, with the final bin inclusive of `1.0`.
- One-vs-rest ROC-AUC and PR average precision are implemented in local project code, with undefined cases returning warnings/nulls rather than misleading zeros.
- The current interpretation is appropriately cautious: Phase 6C-2 is the strongest discriminator among the fixed checkpoints, while the CustomCNN has the lowest measured ECE under this Phase 7 configuration.

### Architecture

- Evaluation concerns are separated into focused modules: `classification.py`, `metrics.py`, `calibration.py`, and `plots.py`.
- The Phase 7 runner lives in `src/visionlab/experiments/phase7.py` and does not leak robustness, OOD, diagnostics, inference, or applied-domain behavior into the evaluation layer.
- The script `scripts/run_phase7_evaluation.py` is a thin entry point and now reports per-run/per-split progress during CPU inference.
- The approach avoids adding `scikit-learn` or `matplotlib`; the metric definitions remain explicit project code.

### Experimental Evidence

- Phase 7 evaluates exactly the accepted fixed references:
  - `phase4b-cifar10-custom-cnn-baseline-001`
  - `phase6b2-cifar10-resnet18-frozen-feature-001`
  - `phase6c-cifar10-resnet18-layer4-finetune-001`
- The Phase 7 contract records checkpoint paths, tags, and SHA-256 identities:
  - Phase 4B CustomCNN best checkpoint: `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`
  - Phase 6B-2 frozen-feature best checkpoint: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`
  - Phase 6C-2 layer4 + fc fine-tuned best checkpoint: `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`
- The sample-alignment report passed for identical registered sample IDs and true labels across all three compared runs:
  - validation: `5,000` samples
  - official test: `10,000` samples
- The official test comparison table reports:

| Run | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC Macro | PR-AUC Macro | ECE | Avg Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4B CustomCNN | `0.635900` | `0.635900` | `0.622400` | `0.940403` | `0.708830` | `0.005757` | `0.640103` |
| Phase 6B-2 frozen ResNet-18 | `0.856100` | `0.856100` | `0.855520` | `0.988789` | `0.932195` | `0.011405` | `0.851709` |
| Phase 6C-2 fine-tuned ResNet-18 | `0.914700` | `0.914700` | `0.913989` | `0.995926` | `0.972716` | `0.037688` | `0.952388` |

### Tests and Verification

- Canonical deterministic suite passed:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
94 tests passed, 1 skipped
```

- Tests cover metric definitions, macro/weighted/micro behavior, undefined metric paths, grouped tied-score AUC behavior, ECE bin boundaries, empty bins, invalid confidence rejection, enriched prediction export, and hard sample/label alignment failures.
- Artifact inspection confirmed the Phase 7 output tree contains JSON, CSV, and SVG artifacts: `19` JSON files, `13` CSV files, and `18` SVG files under `outputs/phase7-evaluation-harness-and-calibration/artifacts/`.
- A representative reliability SVG contains axes, bars, points, labels, and calibration summary text rather than an empty placeholder.
- A representative per-class metrics JSON contains support, precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC, warnings, and metric definitions.

### Documentation and Context

- README, phase catalog, and builder journal now describe Phase 7 as implemented/evaluated with builder review pending.
- The comparison report avoids saying “best model” or “most reliable model.” Its language keeps the result framed as fixed-checkpoint evidence and avoids operational reliability claims.
- The phase catalog points to this Phase 7 check rather than a closeout, which matches the current review boundary.
- The pre-existing deleted root files `phase_briefing.md` and `phase_check.md` remain unrelated worktree changes and were not touched by Phase 7.

## 4. Builder-Codex Context Check

Phase 7 now establishes that VisionLab can compare fixed model checkpoints using sample-aligned prediction records, class-wise metrics, confidence summaries, calibration diagnostics, and lightweight visual artifacts. It also establishes the central Phase 7 lesson: discrimination and calibration are different model properties.

What remains provisional:

- The ECE result is tied to the Phase 7 10-bin configuration and should not be generalized as broad reliability evidence.
- ROC-AUC and PR-AUC are one-vs-rest summaries and should remain secondary to class-wise metrics and confusion patterns.
- SVG diagrams are adequate for project inspection but not publication-polished visuals.

What is explicitly deferred:

- robustness and degradation evaluation;
- OOD or cross-source evaluation;
- high-confidence failure galleries and systematic failure analysis;
- diagnostics/interpretability;
- inference surface work;
- applied-domain selection.

No material mismatch was found between repository artifacts and the current claim that Phase 7 is implemented/evaluated but not formally accepted or closed.

## 5. Required Follow-Ups

**Blocking:** None found for Phase 7 acceptance.

**Non-blocking:**

- Builder should review the Phase 7 comparison report and representative visual artifacts before approving closeout.
- Future phases should preserve the local metric definition tests, especially for calibration binning, undefined metrics, and AUC tie behavior.
- Phase 8 should reuse Phase 7 metric/calibration helpers while keeping robustness and OOD evidence separate from clean in-distribution Phase 7 artifacts.

## 6. Next-Phase Readiness

Phase 7 may proceed to builder review and, if accepted, formal closeout. Phase 8 planning should wait until the builder has reviewed and accepted the Phase 7 check/closeout boundary.

The most important context to carry forward is:

- all three fixed references are sample-aligned on the registered CIFAR-10 validation and official test splits;
- Phase 6C-2 is strongest on clean fixed-checkpoint discrimination metrics;
- the CustomCNN has the lowest measured test ECE under the Phase 7 10-bin configuration;
- calibration evidence does not replace robustness, OOD, or operational reliability evaluation.

## 7. Proposed Phase Closeout Note

Phase 7 implemented VisionLab's fixed-checkpoint evaluation and calibration harness. It evaluated the accepted Phase 4B CustomCNN, Phase 6B-2 frozen ResNet-18, and Phase 6C-2 fine-tuned ResNet-18 best checkpoints without training, tuning, or checkpoint mutation. The phase preserved enriched prediction records, class-wise and aggregate metrics, one-vs-rest ROC-AUC/PR-AUC, calibration summaries, reliability diagrams, confidence distributions, confusion-matrix visuals, exact checkpoint identity, and hard validation/test sample alignment.

The clean in-distribution evidence shows that Phase 6C-2 has the strongest accuracy and macro F1 among the fixed references, while the CustomCNN has the lowest measured ECE under the Phase 7 10-bin diagnostic. These are fixed-checkpoint CIFAR-10 observations, not claims of architecture-only superiority, operational reliability, robustness, OOD behavior, or applied-domain readiness.

Recommended next bounded step after builder acceptance and Phase 7 closeout: begin Phase 8 briefing/planning for robustness and OOD evaluation, using Phase 7's metric/calibration helpers while preserving a separate evidence boundary for degraded and out-of-distribution conditions.
