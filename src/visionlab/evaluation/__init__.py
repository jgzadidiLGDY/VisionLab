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
    "MetricResult",
    "PredictionRecord",
    "averaged_metrics",
    "binary_average_precision",
    "binary_roc_auc",
    "calibration_summary",
    "classification_metrics_from_predictions",
    "confidence_histogram",
    "confusion_matrix",
    "evaluate_classification",
    "multiclass_ovr_auc",
    "per_class_metrics",
    "write_confidence_histogram_csv",
    "write_confidence_histogram_svg",
    "write_confusion_matrix_svg",
    "write_evaluation_artifacts",
    "write_history_artifacts",
    "write_reliability_diagram_svg",
]
