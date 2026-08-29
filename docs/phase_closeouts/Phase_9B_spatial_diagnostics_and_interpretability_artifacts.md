# Phase 9B Closeout - Spatial Diagnostics and Interpretability Artifacts

Date: 2026-08-28

Status: Complete and accepted.

Phase 9B implemented bounded Grad-CAM-style spatial diagnostics over the accepted Phase 9A diagnostic population. The phase generated raw heatmaps, overlay images, manifests, validation artifacts, an HTML gallery, and a diagnostic report while preserving the fixed checkpoint identities and stopping before Phase 9C interpretation/hypothesis closeout.

No training, tuning, checkpoint mutation, model selection, new model evaluation, Phase 7/8 prediction regeneration, saliency, embeddings, UMAP, t-SNE, inference-surface work, applied-domain work, or Phase 9C work occurred.

## Scope Summary

Phase 9B used only these diagnostic populations:

- high-confidence errors: top `8` per fixed run from accepted Phase 9A `high_confidence_errors.csv`;
- model disagreements: top `8` accepted Phase 9A disagreement samples, expanded to one diagnostic row per fixed model in declared checkpoint order;
- correct controls: top `8` correct predictions per fixed run from existing Phase 7 clean CIFAR-10 validation prediction artifacts, ranked by confidence descending and sample ID ascending.

Correct controls are controls only. They are not representative explanations of model behavior.

## Fixed Checkpoint Identity

Phase 9B preserved the accepted fixed checkpoint references:

- Phase 4B CustomCNN baseline: `phase4b-cifar10-custom-cnn-baseline-001`;
- Phase 6B-2 frozen-feature ResNet-18: `phase6b2-cifar10-resnet18-frozen-feature-001`;
- Phase 6C-2 layer4 + fc fine-tuned ResNet-18: `phase6c-cifar10-resnet18-layer4-finetune-001`.

The implementation verifies checkpoint SHA-256 identity against the accepted Phase 9A manifest and checks checkpoint hashes again after diagnostics. No checkpoint mutation was performed or detected.

## Diagnostic Method

Phase 9B generated Grad-CAM-style weighted activation maps from declared target layers:

- CustomCNN: `CustomCNN.feature_blocks[-1]`;
- Phase 6B-2 ResNet-18: `TransferResNet18.model.layer4`;
- Phase 6C-2 ResNet-18: `TransferResNet18.model.layer4`.

The target class for each diagnostic is the preserved predicted class for that selected model/sample. Diagnostic-pass logits and probabilities are traceability context only, not new evaluation metrics.

## Generated Artifacts

Phase 9B generated artifacts under ignored `outputs/phase9b-spatial-diagnostics/`:

- `phase9b_contract.json`;
- `phase9b_result.json`;
- `phase9b_spatial_diagnostics_report.md`;
- `artifacts/diagnostic_selection_manifest.json`;
- `artifacts/gradcam_manifest.csv` with `72` rows;
- `artifacts/gradcam_schema_validation.json` with status `passed`;
- `artifacts/phase9b_artifact_schema_validation.json` as the legacy validation path with the same passed validation payload;
- `artifacts/heatmaps/` with `72` raw heatmap tensors;
- `artifacts/overlays/` with `72` overlay PNGs;
- `artifacts/galleries/gradcam_gallery_manifest.csv` with `72` rows;
- `artifacts/galleries/gradcam_gallery.html`.

The repaired gallery uses browser-resolvable relative image paths such as `../overlays/...`.

## Verification

Focused Phase 9B tests after repair:

```text
Ran 16 tests
OK
```

Canonical deterministic suite after repair:

```text
Ran 179 tests
OK (skipped=1)
```

Strengthened generated-artifact validation after repair:

```text
status: passed
validated artifacts: diagnostic_selection_manifest, gradcam_gallery_html, gradcam_gallery_manifest, gradcam_manifest, gradcam_schema_validation, phase9b_contract, phase9b_result, spatial_diagnostics_report
heatmaps: 72
overlays: 72
```

## Phase Check

The formal Phase 9B check is preserved at:

- `docs/phase_checks/Phase_9B_spatial_diagnostics_and_interpretability_artifacts_check.md`

The final repaired phase-check status was `Ready with small follow-ups`, with no blocking implementation follow-ups remaining. The remaining review boundary was builder visual inspection of the HTML gallery. The builder subsequently requested Phase 9B closeout; this closeout records that request as acceptance of Phase 9B with the visual-review limitation preserved, not as an automated claim that visual QA proves alignment or interpretive value.

## Interpretation Boundaries

Phase 9B does not establish:

- what the model truly looked at;
- model reasoning;
- causality;
- why an error occurred;
- model superiority;
- a reason to change checkpoints or choose interventions;
- deployment reliability;
- applied-domain readiness.

Grad-CAM outputs are diagnostic evidence only. Observations from the gallery must remain separate from interpretation and hypothesis.

## Deferred Work

Deferred to later approved boundaries:

- Phase 9C review tags, failure hypotheses, and Phase 9 closeout synthesis;
- saliency, embeddings, UMAP/t-SNE, spectra, and feature-space exploration unless separately approved;
- inference-surface work;
- applied-domain selection or intervention planning.

## Accepted State

Phase 9B - Spatial Diagnostics and Interpretability Artifacts is complete and accepted.

Overall Phase 9 remains incomplete. The next boundary is Phase 9C review/hypothesis closeout planning, which should use Phase 9A selections and Phase 9B diagnostics while preserving the distinction between selected examples, visual observations, diagnostic artifacts, interpretations, hypotheses, and causal claims.