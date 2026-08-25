"""Explicit classification metric helpers for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class MetricResult:
    value: float | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "warnings": list(self.warnings)}


def classification_metrics_from_predictions(
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute Phase 7 multiclass metrics from prediction-level evidence."""

    _validate_inputs(true_indices, predicted_indices, probabilities, class_names)
    confusion = confusion_matrix(true_indices, predicted_indices, num_classes=len(class_names))
    per_class = per_class_metrics(confusion, class_names=class_names)
    total = len(true_indices)
    correct = sum(1 for true, pred in zip(true_indices, predicted_indices) if true == pred)
    accuracy = correct / total if total else None
    recalls = [per_class[name]["recall"] for name in class_names]
    balanced_accuracy = _mean_defined(recalls)
    averages = averaged_metrics(per_class, class_names=class_names)
    roc_auc = multiclass_ovr_auc(
        true_indices,
        probabilities,
        class_names=class_names,
        curve="roc",
    )
    pr_auc = multiclass_ovr_auc(
        true_indices,
        probabilities,
        class_names=class_names,
        curve="pr",
    )
    warnings = []
    for family in (roc_auc, pr_auc):
        for class_name, result in family["per_class"].items():
            warnings.extend(f"{class_name}: {warning}" for warning in result["warnings"])
    return {
        "total_examples": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "per_class": per_class,
        "averages": averages,
        "confusion_matrix": confusion,
        "roc_auc_ovr": roc_auc,
        "pr_auc_ovr_average_precision": pr_auc,
        "warnings": warnings,
        "definitions": {
            "balanced_accuracy": "mean recall across classes with actual support",
            "macro_average": "unweighted mean across classes with defined metric values",
            "weighted_average": "mean across classes weighted by actual support",
            "micro_average": "global TP/FP/FN aggregation across one-vs-rest class decisions",
            "roc_auc_ovr": "one-vs-rest trapezoidal ROC AUC; undefined without positive and negative examples",
            "pr_auc_ovr_average_precision": "one-vs-rest average precision step integral; undefined without positive examples",
        },
    }


def confusion_matrix(
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    *,
    num_classes: int,
) -> list[list[int]]:
    if len(true_indices) != len(predicted_indices):
        raise ValueError("true_indices and predicted_indices must have the same length")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, pred in zip(true_indices, predicted_indices):
        if true < 0 or true >= num_classes:
            raise ValueError(f"true label index {true} is outside class range")
        if pred < 0 or pred >= num_classes:
            raise ValueError(f"predicted label index {pred} is outside class range")
        matrix[int(true)][int(pred)] += 1
    return matrix


def per_class_metrics(
    confusion: Sequence[Sequence[int]],
    *,
    class_names: tuple[str, ...],
) -> dict[str, dict[str, float | int | None]]:
    num_classes = len(class_names)
    if len(confusion) != num_classes or any(len(row) != num_classes for row in confusion):
        raise ValueError("confusion matrix shape must match class_names")
    results: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(class_names):
        tp = int(confusion[index][index])
        actual = int(sum(confusion[index]))
        predicted = int(sum(row[index] for row in confusion))
        fp = predicted - tp
        fn = actual - tp
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        f1 = _f1(precision, recall)
        results[name] = {
            "support": actual,
            "predicted": predicted,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return results


def averaged_metrics(
    per_class: dict[str, dict[str, float | int | None]],
    *,
    class_names: tuple[str, ...],
) -> dict[str, dict[str, float | None]]:
    total_support = sum(int(per_class[name]["support"] or 0) for name in class_names)
    output: dict[str, dict[str, float | None]] = {}
    for metric in ("precision", "recall", "f1"):
        values = [per_class[name][metric] for name in class_names]
        output[metric] = {
            "macro": _mean_defined(values),
            "weighted": _weighted_mean_defined(
                values,
                [int(per_class[name]["support"] or 0) for name in class_names],
            ),
        }
    global_tp = sum(int(per_class[name]["true_positive"] or 0) for name in class_names)
    global_predicted = sum(int(per_class[name]["predicted"] or 0) for name in class_names)
    global_support = total_support
    micro_precision = global_tp / global_predicted if global_predicted else None
    micro_recall = global_tp / global_support if global_support else None
    output["micro"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": _f1(micro_precision, micro_recall),
    }
    return output


def multiclass_ovr_auc(
    true_indices: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    *,
    class_names: tuple[str, ...],
    curve: str,
) -> dict[str, Any]:
    if curve not in {"roc", "pr"}:
        raise ValueError("curve must be roc or pr")
    per_class: dict[str, dict[str, Any]] = {}
    values: list[float | None] = []
    supports: list[int] = []
    for class_index, name in enumerate(class_names):
        labels = [1 if true == class_index else 0 for true in true_indices]
        scores = [float(row[class_index]) for row in probabilities]
        result = binary_roc_auc(labels, scores) if curve == "roc" else binary_average_precision(labels, scores)
        per_class[name] = result.to_dict()
        values.append(result.value)
        supports.append(sum(labels))
    return {
        "per_class": per_class,
        "macro": _mean_defined(values),
        "weighted": _weighted_mean_defined(values, supports),
    }


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> MetricResult:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return MetricResult(None, ("ROC-AUC undefined without both positive and negative examples",))
    points = [(0.0, 0.0)]
    tp = 0
    fp = 0
    for group in _score_groups(labels, scores):
        tp += sum(1 for label in group if int(label) == 1)
        fp += sum(1 for label in group if int(label) == 0)
        points.append((fp / negatives, tp / positives))
    return MetricResult(_trapezoid_area(points))


def binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> MetricResult:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(1 for label in labels if int(label) == 1)
    if positives == 0:
        return MetricResult(None, ("PR-AUC undefined without positive examples",))
    tp = 0
    fp = 0
    previous_recall = 0.0
    area = 0.0
    for group in _score_groups(labels, scores):
        group_positive = sum(1 for label in group if int(label) == 1)
        group_negative = len(group) - group_positive
        tp += group_positive
        fp += group_negative
        if group_positive:
            recall = tp / positives
            precision = tp / (tp + fp)
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return MetricResult(area)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def _mean_defined(values: Sequence[float | int | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _weighted_mean_defined(values: Sequence[float | int | None], weights: Sequence[int]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in zip(values, weights):
        if value is None or weight <= 0:
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator == 0:
        return None
    return numerator / denominator


def _trapezoid_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    previous_x, previous_y = points[0]
    for x, y in points[1:]:
        area += (x - previous_x) * (y + previous_y) / 2
        previous_x = x
        previous_y = y
    return area


def _validate_inputs(
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    class_names: tuple[str, ...],
) -> None:
    if not class_names:
        raise ValueError("class_names must not be empty")
    if not true_indices:
        raise ValueError("at least one prediction is required")
    if len(true_indices) != len(predicted_indices) or len(true_indices) != len(probabilities):
        raise ValueError("true indices, predicted indices, and probabilities must align")
    num_classes = len(class_names)
    for row in probabilities:
        if len(row) != num_classes:
            raise ValueError("probability row length must match class_names")



def _score_groups(labels: Sequence[int], scores: Sequence[float]) -> list[list[int]]:
    sorted_pairs = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    groups: list[list[int]] = []
    current_score: float | None = None
    current_labels: list[int] = []
    for label, score in sorted_pairs:
        numeric_score = float(score)
        if current_score is None or numeric_score == current_score:
            current_score = numeric_score
            current_labels.append(int(label))
            continue
        groups.append(current_labels)
        current_score = numeric_score
        current_labels = [int(label)]
    if current_labels:
        groups.append(current_labels)
    return groups
