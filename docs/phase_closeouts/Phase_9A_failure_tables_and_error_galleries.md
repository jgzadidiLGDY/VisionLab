# Phase 9A Closeout - Failure Tables and Error Galleries

Date: 2026-08-28

Status: Complete and accepted.

Phase 9A implemented deterministic failure tables and a high-confidence-error gallery from existing Phase 7 clean CIFAR-10 validation prediction artifacts. The phase established a machine-readable selection contract, preserved sample/checkpoint/dataset identity, and stopped before spatial diagnostics, interpretability overlays, hypothesis closeout, inference work, or any model change.

No training, tuning, checkpoint mutation, model selection, new model evaluation, Phase 7/8 artifact regeneration, Grad-CAM, saliency, embeddings, UMAP, inference-surface work, applied-domain work, or Phase 9B/9C work occurred.

## Scope Summary

Phase 9A used only the required population:

- context: `phase7_clean_cifar10_val`;
- dataset: CIFAR-10, version `phase1b-registered`;
- split: `val`;
- condition: `clean`;
- source artifacts: existing Phase 7 validation prediction CSVs;
- compared fixed checkpoints: Phase 4B CustomCNN, Phase 6B-2 frozen-feature ResNet-18, and Phase 6C-2 layer4 + fc fine-tuned ResNet-18.

Optional Phase 8B degraded-validation and Phase 8C CIFAR-10.1 v6 cross-source populations were recorded as unavailable for Phase 9A example selection because no preserved prediction-level artifacts were found. The missing optional prediction-level artifacts were not regenerated.

## Selection Contract

The Phase 9A selection contract is preserved at:

- `outputs/phase9a-failure-analysis-galleries/phase9a_contract.json`

The contract defines:

- high-confidence errors: incorrect predictions only, top `24` per run, ranked by confidence descending and sample ID ascending;
- per-class failures: all `10` CIFAR-10 classes preserved, including support, correct count, false-negative count, false-positive count, and accuracy;
- per-class examples: top `3` false-negative and false-positive examples per class/category/run by confidence descending and sample ID ascending;
- confusion-pair examples: top `10` true-label to predicted-label error pairs by pair count descending, true label ascending, predicted label ascending, with up to `3` examples per pair ranked by confidence descending and sample ID ascending;
- model disagreements: top `24` aligned samples where compared runs do not all predict the same label, ranked by distinct prediction count, incorrect model count, confidence spread, and sample ID.

Selected examples are representative only of the declared selection rule and population. They are not curated anecdotes and are not causal explanations.

## Implementation Summary

Phase 9A added or updated:

- `src/visionlab/evaluation/failures.py` for deterministic failure selection, prediction CSV loading, per-class failure summaries, confusion-pair selection, and hard sample/label/context alignment checks;
- `src/visionlab/evaluation/galleries.py` for deterministic gallery image materialization, gallery manifests, and HTML gallery output;
- `src/visionlab/experiments/phase9a.py` for Phase 9A orchestration, selection-contract generation, selection-manifest writing, generated-artifact schema validation, optional-context availability recording, and report writing;
- `src/visionlab/evaluation/__init__.py` to export the new Phase 9A helpers;
- `scripts/run_phase9a_failure_analysis.py` as the Phase 9A artifact-generation entry point;
- `tests/test_phase9a_failures.py` for focused deterministic tests.

## Fixed Checkpoint Identity

Phase 9A preserved the accepted fixed checkpoint identities:

- Phase 4B CustomCNN: `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`;
- Phase 6B-2 frozen-feature ResNet-18: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`;
- Phase 6C-2 layer4 + fc fine-tuned ResNet-18: `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`.

The model-disagreement artifact includes explicit top-level checkpoint identity fields and per-run checkpoint identity columns. No checkpoint was modified, replaced, reselected, or re-exported.

## Generated Artifacts

Phase 9A generated artifacts under ignored `outputs/phase9a-failure-analysis-galleries/`:

- `phase9a_contract.json`;
- `phase9a_result.json`;
- `phase9a_failure_analysis_report.md`;
- `artifacts/high_confidence_errors.csv` with `72` rows;
- `artifacts/per_class_failure_summary.csv` with `30` rows;
- `artifacts/per_class_failure_examples.csv` with `180` rows;
- `artifacts/confusion_pair_examples.csv` with `90` rows;
- `artifacts/model_disagreement_examples.csv` with `24` rows;
- `artifacts/optional_context_availability.json`;
- `artifacts/failure_selection_manifest.json`;
- `artifacts/phase9a_artifact_schema_validation.json` with status `passed`;
- `artifacts/galleries/high_confidence_error_gallery_manifest.csv` with `72` rows;
- `artifacts/galleries/high_confidence_error_gallery.html`;
- `artifacts/galleries/high_confidence_error_images/` with `66` unique image files.

## Verification

Focused Phase 9A tests passed after blocker repair:

```text
.venv\Scripts\python.exe -m unittest tests.test_phase9a_failures
Ran 13 tests in 0.018s
OK
```

Canonical deterministic suite passed after blocker repair:

```text
powershell -ExecutionPolicy Bypass -File scripts\test.ps1
Ran 163 tests in 10.766s
OK (skipped=1)
```

The formal Phase 9A check is preserved at:

- `docs/phase_checks/Phase_9A_failure_tables_and_error_galleries_check.md`

The final Phase 9A check status is `Ready with small follow-ups`, with no blocking follow-ups remaining.

## Visual and Manual Review

The builder requested Phase 9A closeout after receiving the verification path and repaired phase-check result. The generated gallery remains a qualitative review artifact: observations from it should be recorded separately from hypotheses, and visual inspection does not establish causal explanations.

## Boundaries Preserved

Phase 9A did not perform:

- Phase 9B spatial diagnostics;
- Grad-CAM, saliency, embeddings, or UMAP;
- Phase 9C review-tag or hypothesis-report closeout work;
- training or tuning;
- model selection;
- checkpoint mutation;
- new model evaluation;
- official CIFAR-10 test reruns;
- CIFAR-10.1 v6 reruns;
- Phase 7 or Phase 8 prediction-artifact regeneration;
- inference-surface work;
- applied-domain selection or applied-domain implementation.

## Limitations and Non-Claims

Phase 9A does not establish:

- causal explanations for failures;
- model interpretability;
- Grad-CAM or saliency behavior;
- embedding-space structure;
- robustness failure examples for Phase 8B degraded conditions;
- CIFAR-10.1 v6 cross-source failure examples;
- model superiority beyond already accepted Phase 7/8 evidence;
- any reason to modify, tune, or select a model.

High-confidence selected errors show where models were confidently wrong under declared rules. They do not prove why the models failed.

## Accepted State

Phase 9A - Failure Tables and Error Galleries is complete and accepted.

Overall Phase 9 remains incomplete. The next boundary is a separate Phase 9B concept/implementation plan for spatial diagnostics or other approved interpretability work, using the Phase 9A machine-selected examples as the diagnostic substrate and preserving the distinction between visual observation, attribution output, and causal hypothesis.
