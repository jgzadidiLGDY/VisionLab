# Phase 9A Check - Failure Tables and Error Galleries

Date: 2026-08-28

Status: Ready with small follow-ups.

## 1. Overall Status

**Ready with small follow-ups.**

The Phase 9A phase-check blockers from the prior review are resolved. `model_disagreement_examples.csv` now exposes explicit checkpoint identity fields as top-level columns while preserving the existing fixed checkpoint identities exactly. The generated-artifact schema validation now covers every required Phase 9A selection output, including high-confidence errors, per-class failures, confusion-pair examples, model-disagreement examples, the failure-selection manifest, and the gallery manifest.

Phase 9A remains at the phase-check/review boundary. It has not been automatically closed or accepted.

## 2. Intended-Shape Assessment

Phase 9A matches the approved objective: deterministic failure tables and error galleries only. It uses existing Phase 7 clean CIFAR-10 validation prediction artifacts for the three fixed accepted checkpoints and does not rerun model evaluation or regenerate Phase 7/8 prediction artifacts.

The phase stayed within scope. No Phase 9B spatial diagnostics, Grad-CAM, saliency, embeddings, UMAP, Phase 9C closeout work, training, tuning, checkpoint mutation, new evaluation, model selection, inference-surface work, or applied-domain work was performed.

The repository is coherent for Phase 9A review. Remaining work is review-oriented, not a blocker in the repaired implementation.

## 3. Key Findings

### Vision and ML Correctness

- The required population is explicit: `phase7_clean_cifar10_val`, CIFAR-10 version `phase1b-registered`, split `val`, condition `clean`.
- High-confidence errors use incorrect predictions only, ranked by descending maximum predicted class probability with deterministic sample-ID tie-breaking.
- Per-class failure summaries preserve all `10` CIFAR-10 classes and include support, correct count, false-negative count, false-positive count, and accuracy.
- Confusion-pair examples are selected from deterministic pair rankings and example rankings rather than visual judgment.
- Model disagreements compare only aligned samples with identical sample IDs, labels, context, split, and condition across runs.
- Confidence is treated as maximum predicted class probability and is not presented as correctness or reliability.

### Architecture

- `src/visionlab/evaluation/failures.py` owns deterministic failure selection.
- `src/visionlab/evaluation/galleries.py` owns gallery artifact writing.
- `src/visionlab/experiments/phase9a.py` owns Phase 9A orchestration, contract generation, artifact schema requirements, selection manifest writing, and generated-artifact schema validation.
- The implementation remains evaluation-artifact driven and does not introduce diagnostic architecture prematurely.

### Experimental Evidence

Generated artifacts are isolated under `outputs/phase9a-failure-analysis-galleries/`.

Repaired generated artifact counts inspected during this check:

- `high_confidence_errors.csv`: `72` rows.
- `per_class_failure_summary.csv`: `30` rows.
- `per_class_failure_examples.csv`: `180` rows.
- `confusion_pair_examples.csv`: `90` rows.
- `model_disagreement_examples.csv`: `24` rows.
- `high_confidence_error_gallery_manifest.csv`: `72` rows.

The model-disagreement artifact now includes top-level `checkpoint_tag`, `checkpoint_path`, and `checkpoint_sha256` fields, plus per-run explicit checkpoint identity columns for the three fixed references:

- `phase4b-cifar10-custom-cnn-baseline-001`
- `phase6b2-cifar10-resnet18-frozen-feature-001`
- `phase6c-cifar10-resnet18-layer4-finetune-001`

The fixed checkpoint SHA-256 identities remain unchanged:

- Phase 4B CustomCNN: `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`
- Phase 6B-2 frozen-feature ResNet-18: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`
- Phase 6C-2 layer4 + fc fine-tuned ResNet-18: `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`

Optional Phase 8B degraded-validation and Phase 8C CIFAR-10.1 v6 cross-source populations remain unavailable for Phase 9A example selection because no preserved prediction-level artifacts were found. No evaluation was regenerated to fill that gap.

### Tests and Verification

Focused Phase 9A tests passed after the repair:

```text
.venv\Scripts\python.exe -m unittest tests.test_phase9a_failures
Ran 13 tests in 0.018s
OK
```

Canonical deterministic suite passed after the repair:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
Ran 163 tests in 10.766s
OK (skipped=1)
```

New or strengthened test coverage verifies:

- deterministic high-confidence error ranking and tie-breaking;
- per-class failure extraction;
- confusion-pair extraction;
- model-disagreement extraction;
- hard sample/label alignment failures;
- explicit checkpoint identity fields for disagreement rows;
- required selection-row fields;
- generated-artifact schema validation for every required Phase 9A output;
- schema-validation failure when disagreement checkpoint identity is missing;
- deterministic gallery manifests.

Generated schema validation artifact:

- `outputs/phase9a-failure-analysis-galleries/artifacts/phase9a_artifact_schema_validation.json`
- Status: `passed`

Selection manifest artifact:

- `outputs/phase9a-failure-analysis-galleries/artifacts/failure_selection_manifest.json`

### Documentation and Context

- The Phase 9A report now records generated schema validation status.
- README, phase catalog, builder journal, and closeout docs were not updated as if Phase 9A were accepted, matching the requested boundary.
- The pre-existing root-level deletions of `phase_briefing.md` and `phase_check.md` remain unrelated to this repair and were not modified.

## 4. Builder-Codex Context Check

Phase 9A now establishes deterministic, schema-validated failure-selection artifacts for clean CIFAR-10 validation prediction records from Phase 7. The selected examples are evidence from declared rules and populations, not curated anecdotes.

What remains provisional: visual observations from the generated gallery. The builder still needs to review the gallery before accepting Phase 9A or using its examples as the substrate for Phase 9B diagnostics.

Explicitly deferred: Grad-CAM, saliency, spatial diagnostics, embedding/UMAP exploration, human review tagging, Phase 9C closeout, model changes, intervention planning, inference work, and applied-domain selection.

## 5. Required Follow-Ups

### Blocking

None. The previous blocking issues are resolved.

### Non-blocking

- Builder should visually review `outputs/phase9a-failure-analysis-galleries/artifacts/galleries/high_confidence_error_gallery.html` and record observations separately from hypotheses.
- If the builder wants Phase 9A formally accepted, a separate closeout request should update the closeout trail and current-status documents without adding new implementation scope.

## 6. Next-Phase Readiness

Phase 9A is ready for builder review. Phase 9B should wait until the builder reviews and accepts the repaired Phase 9A artifacts, because the selected examples are intended to become the diagnostic substrate.

Most important carry-forward context: Phase 9B diagnostics must operate on the machine-selected Phase 9A examples and must preserve the distinction between visual observation, attribution output, and causal hypothesis.

## 7. Proposed Phase Closeout Note

Phase 9A implemented deterministic failure tables and a high-confidence-error gallery from existing Phase 7 clean CIFAR-10 validation prediction artifacts for the three fixed accepted checkpoints. It preserved declared selection rules, sample IDs, labels, dataset/split/condition identity, confidence semantics, checkpoint identities, and isolated generated artifacts under `outputs/phase9a-failure-analysis-galleries/`. Optional Phase 8B and Phase 8C contexts were not used because prediction-level artifacts were unavailable and no evaluations were regenerated. The repaired implementation includes explicit checkpoint identity fields for model-disagreement rows and generated-artifact schema validation across required Phase 9A outputs. Phase 9A preserved the boundaries around no training, tuning, checkpoint mutation, new evaluation, model selection, diagnostics, inference, applied-domain work, or closeout. Recommended next step: builder visual review of the generated gallery, then formal Phase 9A closeout if accepted.
