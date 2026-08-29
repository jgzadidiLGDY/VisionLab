# Phase-Check Report - Phase 9B Spatial Diagnostics and Interpretability Artifacts

Date: 2026-08-28

## 1. Overall Status

**Ready with small follow-ups** for builder review. Phase 9B is not closed or accepted.

The prior Phase 9B phase-check blocker for incomplete generated-artifact validation is resolved. The strengthened validator now covers the full required artifact set: `phase9b_contract.json`, `phase9b_result.json`, `diagnostic_selection_manifest.json`, `gradcam_manifest.csv`, `gradcam_schema_validation.json`, every persisted raw heatmap tensor, every overlay PNG, `phase9b_spatial_diagnostics_report.md`, the gallery manifest, and the generated HTML gallery.

Visual QA remains a builder review boundary. Automated validation confirms file readability, browser-resolvable gallery image paths, dimensions, finite/nonempty normalized heatmaps, and required labels/fields, but it does not establish visual alignment, saturation quality, or interpretive usefulness. Phase 9B should not be closed until the builder records actual visual review or explicitly waives it.

## 2. Intended-Shape Assessment

Phase 9B matches the approved objective and boundaries:

- It implements Grad-CAM-style spatial diagnostics only.
- It uses the fixed Phase 4B, Phase 6B-2, and Phase 6C-2 checkpoint references established by Phase 9A/Phase 7.
- It derives high-confidence-error and model-disagreement diagnostics from machine-selected Phase 9A examples.
- It derives correct-control diagnostics deterministically from existing Phase 7 clean CIFAR-10 validation prediction artifacts.
- It preserves the existing Phase 9B diagnostic population, selection criteria, fixed checkpoints, target layers, and preprocessing contracts.
- It does not regenerate Phase 7/8 prediction artifacts.
- It does not run new model evaluation or generate new evaluation metrics.
- It does not train, tune, mutate checkpoints, select models, implement inference, add embeddings/UMAP/t-SNE, add saliency, or start Phase 9C.

The repository is coherent and ready for builder review of the generated diagnostic artifacts.

## 3. Key Findings

### Vision And ML Correctness

- Diagnostic target layers match the approved contract:
  - CustomCNN uses `CustomCNN.feature_blocks[-1]`;
  - Phase 6B-2 and Phase 6C-2 ResNet-18 use `TransferResNet18.model.layer4`.
- The target class for each Grad-CAM pass is the preserved predicted class for that model/sample.
- Live consistency checks found `0` diagnostic-prediction mismatches between preserved prediction context and diagnostic-pass prediction labels.
- Strengthened validation found `0` bad heatmaps: all `72` saved heatmaps are finite, normalized, expected-shaped, and nonempty.
- Heatmap shapes are expected: `24` at `32 x 32` for CustomCNN and `48` at `224 x 224` for ResNet-18 models.
- Overlay PNGs are readable and match expected dimensions for all `72` diagnostic rows. The generated gallery manifest/HTML now use browser-resolvable relative paths such as `../overlays/...`, and validation rejects broken gallery image paths.

### Architecture

- A focused `visionlab.diagnostics` package owns reusable Grad-CAM computation.
- Phase 9B orchestration lives in `src/visionlab/experiments/phase9b.py`, matching the existing phase-runner pattern.
- The implementation reuses existing model restore paths and preprocessing contracts.
- Non-blocking architecture note: Phase 9B imports private Phase 7 loader helpers, which matches existing Phase 8 practice but could later be formalized if diagnostics grow.

### Experimental Evidence

- Generated diagnostic rows remain unchanged from the approved Phase 9B population:
  - `24` high-confidence error rows;
  - `24` model-disagreement rows;
  - `24` correct-control rows.
- Generated diagnostic rows remain balanced by fixed run:
  - `24` for Phase 4B CustomCNN;
  - `24` for Phase 6B-2 frozen-feature ResNet-18;
  - `24` for Phase 6C-2 fine-tuned ResNet-18.
- The model-disagreement population still expands the top `8` Phase 9A disagreement samples to one diagnostic row per fixed model in declared checkpoint order.
- The report uses non-causal language and keeps visual observations as `pending_builder_review`.

### Tests And Verification

Focused Phase 9B tests after repair:

```text
Ran 15 tests in 0.349s
OK
```

Canonical deterministic suite after repair:

```text
Ran 178 tests in 12.336s
OK (skipped=1)
```

Strengthened generated-artifact validation after repair:

```text
status: passed
validated artifacts: diagnostic_selection_manifest, gradcam_gallery_html, gradcam_gallery_manifest, gradcam_manifest, gradcam_schema_validation, phase9b_contract, phase9b_result, spatial_diagnostics_report
heatmaps: 72
overlays: 72
```

Focused tests now cover:

- bounded contract language;
- deterministic selection;
- model-disagreement expansion order;
- missing Phase 9A artifact failure;
- target-layer resolution and failure behavior;
- finite normalized heatmap shape/range for helper output;
- all-empty heatmap hard stop;
- complete generated-artifact validation;
- malformed contract rejection;
- malformed result rejection;
- missing required manifest field rejection;
- malformed heatmap rejection;
- malformed overlay rejection;
- report-without-limitations rejection;
- HTML gallery-without-images rejection;
- broken gallery manifest image-path rejection.

### Documentation And Context

- README, phase catalog, and builder journal were not updated to mark Phase 9B complete or accepted, which is correct for this boundary.
- The formal Phase 9B check now records the repaired validation state and the remaining visual-review boundary.
- The unrelated pre-existing root deletions `phase_briefing.md` and `phase_check.md` remain untouched.

## 4. Builder-Codex Context Check

Phase 9B now establishes:

- a working bounded Grad-CAM pipeline for the three fixed model references;
- deterministic diagnostic populations derived from Phase 9A and Phase 7 artifacts;
- raw heatmaps, overlays, manifests, report, validation artifact, result artifact, and gallery artifacts under `outputs/phase9b-spatial-diagnostics/`;
- complete generated-artifact validation for required Phase 9B artifacts;
- CPU-compatible focused tests for the core diagnostic behavior and artifact validation.

What remains provisional:

- actual visual quality and alignment of overlays until builder visual review is recorded;
- whether heatmap patterns are useful for later interpretation;
- any interpretation or hypothesis attached to the diagnostic outputs.

Explicitly deferred:

- Phase 9C hypothesis/review-tag closeout;
- saliency, embeddings, UMAP/t-SNE, spectra, feature-space exploration;
- model changes, intervention planning, inference, and applied-domain work.

No mismatch was found in checkpoint/sample prediction consistency during live checks.

## 5. Required Follow-Ups

Blocking implementation follow-ups: none found after repair.

Acceptance/review boundary before closeout:

- Builder visual inspection of `outputs/phase9b-spatial-diagnostics/artifacts/galleries/gradcam_gallery.html` for overlay alignment, visible/nonblank heatmaps, saturation behavior, and clear model/context labeling.
- Record visual observations separately from interpretation or hypotheses.

Non-blocking:

- Consider formalizing model restore helpers for future diagnostics if Phase 9C or later phases need them, rather than continuing to import private Phase 7 helpers.

## 6. Next-Phase Readiness

Phase 9C should not begin until Phase 9B visual review is recorded and the builder accepts or explicitly waives the remaining review boundary.

Entry checkpoint for the next phase:

- focused and canonical tests remain passing;
- strengthened generated-artifact validation remains passing;
- builder visual QA is recorded or explicitly waived;
- Phase 9B is accepted or deliberately carried forward with named limitations.

Most important context to carry forward:

- Grad-CAM outputs are diagnostic evidence only;
- they do not prove what the model looked at, prove reasoning, establish causality, or explain why an error occurred;
- correct controls are controls only, not representative explanations;
- any future hypotheses must remain separate from selected examples and visual observations.

## 7. Proposed Phase Closeout Note

Phase 9B added bounded Grad-CAM-style spatial diagnostics over accepted Phase 9A examples and deterministic correct controls, preserving fixed checkpoint identity and avoiding training, tuning, model evaluation, model selection, and later-phase work. The generated artifact set now has complete validation coverage, including contract, result, manifests, schema-validation artifact, report, gallery, heatmaps, and overlays. Before closeout, the builder should inspect the Grad-CAM gallery visually and record observations separately from hypotheses. The next bounded step after acceptance is Phase 9C review/hypothesis closeout, not model intervention.