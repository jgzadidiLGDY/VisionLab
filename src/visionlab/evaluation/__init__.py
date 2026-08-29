"""Evaluation helpers for VisionLab."""

from visionlab.evaluation.calibration import (
    CalibrationBin,
    CalibrationSummary,
    calibration_summary,
    confidence_histogram,
)
from visionlab.evaluation.classification import (
    ClassificationEvaluation,
    ClassificationSummary,
    PredictionRecord,
    evaluate_classification,
    write_evaluation_artifacts,
    write_history_artifacts,
)
from visionlab.evaluation.failures import (
    FailurePrediction,
    align_predictions_by_sample,
    confusion_pair_summary,
    load_prediction_csv,
    per_class_failure_summary,
    select_confusion_pair_examples,
    select_high_confidence_errors,
    select_model_disagreements,
    select_per_class_failure_examples,
    write_csv_rows,
)
from visionlab.evaluation.galleries import (
    materialize_gallery_images,
    write_gallery_html,
    write_gallery_manifest,
)
from visionlab.evaluation.metrics import (
    MetricResult,
    averaged_metrics,
    binary_average_precision,
    binary_roc_auc,
    classification_metrics_from_predictions,
    confusion_matrix,
    multiclass_ovr_auc,
    per_class_metrics,
)
from visionlab.evaluation.plots import (
    write_confidence_histogram_csv,
    write_confidence_histogram_svg,
    write_confusion_matrix_svg,
    write_reliability_diagram_svg,
)

__all__ = [
    "CalibrationBin",
    "CalibrationSummary",
    "ClassificationEvaluation",
    "ClassificationSummary",
    "FailurePrediction",
    "MetricResult",
    "PredictionRecord",
    "align_predictions_by_sample",
    "averaged_metrics",
    "binary_average_precision",
    "binary_roc_auc",
    "calibration_summary",
    "classification_metrics_from_predictions",
    "confusion_pair_summary",
    "confidence_histogram",
    "confusion_matrix",
    "evaluate_classification",
    "load_prediction_csv",
    "materialize_gallery_images",
    "multiclass_ovr_auc",
    "per_class_failure_summary",
    "per_class_metrics",
    "select_confusion_pair_examples",
    "select_high_confidence_errors",
    "select_model_disagreements",
    "select_per_class_failure_examples",
    "write_confidence_histogram_csv",
    "write_confidence_histogram_svg",
    "write_confusion_matrix_svg",
    "write_csv_rows",
    "write_evaluation_artifacts",
    "write_gallery_html",
    "write_gallery_manifest",
    "write_history_artifacts",
    "write_reliability_diagram_svg",
]
