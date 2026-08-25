# Phase 7 Closeout - Evaluation Harness and Calibration

Status: Complete; accepted by builder.

## Phase Objective

Phase 7 replaced headline accuracy-only reporting with a structured fixed-checkpoint evaluation and calibration harness. It evaluated the accepted Phase 4B CustomCNN, Phase 6B-2 frozen-feature ResNet-18, and Phase 6C-2 layer4 + fc fine-tuned ResNet-18 best checkpoints without retraining, tuning, regenerating, or overwriting checkpoints.

The phase objective was evaluation and evidence preservation, not model improvement.

## Approved Scope

Phase 7 was approved to add:

- metric engine;
- per-class evaluation;
- confusion matrices;
- calibration metric;
- reliability diagrams;
- confidence distributions;
- prediction-record export;
- compatible model-comparison report;
- metric unit tests.

Approved amendments required:

- hard validation/test sample identity alignment across compared runs;
- exact preserved-checkpoint loading by run ID, checkpoint tag, and checkpoint identity/hash where available;
- separate validation and official test semantics;
- no Phase 7 model selection based on official test performance;
- no training, tuning, checkpoint mutation, robustness/OOD work, diagnostics, inference work, or applied-domain work.

## Implementation Summary

Phase 7 added or updated:

- `src/visionlab/evaluation/classification.py` to enrich prediction records with true/predicted indices, logits, and full class probabilities;
- `src/visionlab/evaluation/metrics.py` for accuracy, balanced accuracy, per-class precision/recall/F1, macro/micro/weighted summaries, one-vs-rest ROC-AUC, and one-vs-rest average precision;
- `src/visionlab/evaluation/calibration.py` for ECE, maximum calibration error, confidence summaries, deterministic bin-boundary behavior, and empty-bin handling;
- `src/visionlab/evaluation/plots.py` for dependency-light SVG/CSV reliability diagrams, confidence histograms, and confusion matrices;
- `src/visionlab/experiments/phase7.py` for fixed-checkpoint orchestration, artifact writing, and hard sample/label alignment checks;
- `scripts/run_phase7_evaluation.py` for rerunning the Phase 7 evaluation pass;
- Phase 7 tests for metric definitions, undefined metric behavior, tied-score AUC handling, ECE bin boundaries, empty bins, invalid confidence rejection, enriched prediction export, and sample/label alignment failures.

No dependency was added for metrics or plotting. The Phase 7 metric definitions remain explicit project code.

## Fixed Checkpoint Identity

Phase 7 evaluated only preserved `best.pt` checkpoints from accepted historical runs.

The Phase 7 contract records:

- Phase 4B CustomCNN best checkpoint: `5904a9a7bf6a40cf98f67c16212cb0bdcd8cede70496b69f94ea3fc928785947`
- Phase 6B-2 frozen-feature ResNet-18 best checkpoint: `5832c71f298ee4d21a18f1e38460a92082a5733af26f108211afcc8a9cdd1af5`
- Phase 6C-2 layer4 + fc fine-tuned ResNet-18 best checkpoint: `0e993ebf64250cba013ee1b688bec5a6b95647679b8012ed31bc662dbf4428d1`

Checkpoint paths, tags, and hashes are preserved in `outputs/phase7-evaluation-harness-and-calibration/phase7_contract.json`.

## Dataset and Sample Alignment

Dataset: Phase 1B registered CIFAR-10.

Class order:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

The Phase 7 sample-alignment report passed:

- validation: `5,000` identical registered sample IDs and true labels across all three compared runs;
- official test: `10,000` identical registered sample IDs and true labels across all three compared runs.

This is a hard comparison invariant. The harness fails rather than silently comparing different samples or labels.

## Evaluation and Calibration Configuration

Calibration uses maximum predicted class probability as confidence.

ECE configuration:

- number of bins: `10`
- bin assignment: `min(int(confidence * num_bins), num_bins - 1)`
- bin interval semantics: `[lower, upper)` except the final bin includes confidence `1.0`
- ECE definition: sum over bins of `bin_fraction * abs(bin_accuracy - bin_average_confidence)`
- maximum calibration error: maximum non-empty-bin absolute calibration gap

The ECE configuration is preserved in `outputs/phase7-evaluation-harness-and-calibration/phase7_contract.json` and in per-run calibration artifacts.

## Results

Validation results from `outputs/phase7-evaluation-harness-and-calibration/artifacts/phase7_comparison_table.csv`:

| Run | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC Macro | PR-AUC Macro | ECE | Avg Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4B CustomCNN | `0.630200` | `0.630200` | `0.618902` | `0.941160` | `0.719183` | `0.018083` | `0.638256` |
| Phase 6B-2 frozen ResNet-18 | `0.864600` | `0.864600` | `0.864046` | `0.989732` | `0.936793` | `0.013265` | `0.851645` |
| Phase 6C-2 fine-tuned ResNet-18 | `0.925800` | `0.925800` | `0.925510` | `0.996397` | `0.975839` | `0.027550` | `0.953350` |

Official test results from `outputs/phase7-evaluation-harness-and-calibration/artifacts/phase7_comparison_table.csv`:

| Run | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC Macro | PR-AUC Macro | ECE | Avg Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4B CustomCNN | `0.635900` | `0.635900` | `0.622400` | `0.940403` | `0.708830` | `0.005757` | `0.640103` |
| Phase 6B-2 frozen ResNet-18 | `0.856100` | `0.856100` | `0.855520` | `0.988789` | `0.932195` | `0.011405` | `0.851709` |
| Phase 6C-2 fine-tuned ResNet-18 | `0.914700` | `0.914700` | `0.913989` | `0.995926` | `0.972716` | `0.037688` | `0.952388` |

## Interpretation

Among the three fixed checkpoints evaluated in Phase 7, Phase 6C-2 achieved the highest clean CIFAR-10 validation and official test accuracy, balanced accuracy, macro F1, ROC-AUC macro, and PR-AUC macro.

Calibration is different from discrimination. Under the Phase 7 10-bin ECE configuration, the CustomCNN exhibited the lowest measured official-test ECE (`0.005757`), while the Phase 6C-2 fine-tuned ResNet-18 exhibited the highest measured official-test ECE (`0.037688`) despite having the strongest discrimination metrics.

This supports the Phase 7 learning point that higher accuracy does not imply better calibration.

## Artifact Inventory

Phase 7 generated artifacts under ignored `outputs/phase7-evaluation-harness-and-calibration/`:

- `phase7_contract.json`
- `phase7_result.json`
- `phase7_model_comparison_report.md`
- `artifacts/phase7_comparison_table.csv`
- `artifacts/sample_alignment.json`
- per-run/per-split summary JSON files
- per-run/per-split enriched prediction CSV files
- per-run/per-split metric JSON files
- per-run/per-split calibration JSON files
- per-run/per-split confidence histogram CSV files
- per-run/per-split reliability diagram SVG files
- per-run/per-split confidence histogram SVG files
- per-run/per-split confusion matrix SVG files

Artifact inspection during phase check confirmed `19` JSON files, `13` CSV files, and `18` SVG files under `outputs/phase7-evaluation-harness-and-calibration/artifacts/`.

## Verification

Final canonical deterministic suite after documentation-only closeout updates:

```text
powershell -ExecutionPolicy Bypass -File scripts	est.ps1
94 tests passed, 1 skipped
```

Phase 7 verification covered:

- metric definitions and averages;
- undefined metric paths;
- tied-score AUC behavior;
- ECE bin boundaries and empty bins;
- invalid confidence rejection;
- enriched prediction export;
- hard sample/label alignment failures;
- full fixed-checkpoint evaluation pass;
- artifact inspection of alignment, comparison table, calibration JSON, per-class metrics JSON, and representative SVG reliability output.

## Boundaries Preserved

Phase 7 did not perform:

- model training;
- checkpoint regeneration or mutation;
- hyperparameter tuning;
- augmentation changes;
- threshold optimization for deployment;
- robustness or degradation sweeps;
- OOD or cross-source evaluation;
- high-confidence failure-gallery selection as a Phase 9 artifact;
- Grad-CAM, saliency, or other diagnostics;
- inference-surface work;
- applied-domain selection or applied-domain implementation.

Validation and official test artifacts remain separate. Official test results are retrospective fixed-checkpoint evaluation evidence only and were not used for model selection.

## Limitations and Non-Claims

Phase 7 does not establish:

- seed or run-to-run variance;
- architecture-only superiority of ResNet-18 over the CustomCNN;
- operational reliability;
- calibrated deployment thresholds;
- robustness to blur, noise, compression, lighting, or other degradations;
- OOD or cross-source generalization;
- failure-cause explanations;
- interpretability or causal model reasoning;
- inference behavior;
- applied-domain readiness.

ECE is a diagnostic under the preserved Phase 7 10-bin configuration, not a universal reliability score. ROC-AUC and PR-AUC are one-vs-rest summaries and should be interpreted alongside per-class metrics and confusion patterns.

## Phase Check and Closeout Decision

The formal Phase 7 phase check found the phase **Ready with small follow-ups** and no blocking implementation issue. The builder then requested formal closeout documentation and status updates.

Phase 7 is closed and accepted. Phase 8 has not started.

## Next Boundary

The next bounded step is a separate Phase 8 concept briefing and implementation plan for robustness and OOD evaluation. Phase 8 should reuse Phase 7 metric/calibration helpers where appropriate, while preserving a separate evidence boundary for degraded and out-of-distribution conditions.
