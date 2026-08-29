"""Deterministic failure selection for VisionLab Phase 9A."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class FailurePrediction:
    run_id: str
    context_id: str
    dataset_id: str
    dataset_version: str
    split: str
    condition_id: str
    sample_id: str
    source_id: str
    true_label: str
    predicted_label: str
    confidence: float
    correct: bool
    true_index: int
    predicted_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "context_id": self.context_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "split": self.split,
            "condition_id": self.condition_id,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "correct": self.correct,
            "true_index": self.true_index,
            "predicted_index": self.predicted_index,
        }


def load_prediction_csv(
    path: Path,
    *,
    run_id: str,
    context_id: str,
    dataset_id: str,
    dataset_version: str,
    split: str,
    condition_id: str,
) -> tuple[FailurePrediction, ...]:
    """Load an existing prediction artifact without evaluating a model."""

    required = {
        "sample_id",
        "split",
        "true_label",
        "predicted_label",
        "confidence",
        "correct",
        "source_id",
        "true_index",
        "predicted_index",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"prediction artifact {path} is missing required fields: {sorted(missing)}")
        rows = []
        for row in reader:
            row_split = row["split"]
            if row_split != split:
                raise ValueError(f"prediction artifact {path} contains split {row_split}, expected {split}")
            rows.append(
                FailurePrediction(
                    run_id=run_id,
                    context_id=context_id,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    split=split,
                    condition_id=condition_id,
                    sample_id=row["sample_id"],
                    source_id=row["source_id"],
                    true_label=row["true_label"],
                    predicted_label=row["predicted_label"],
                    confidence=float(row["confidence"]),
                    correct=_parse_bool(row["correct"]),
                    true_index=int(row["true_index"]),
                    predicted_index=int(row["predicted_index"]),
                )
            )
    return tuple(rows)


def select_high_confidence_errors(
    predictions: Sequence[FailurePrediction],
    *,
    top_n: int,
) -> tuple[FailurePrediction, ...]:
    """Select incorrect predictions by confidence desc, then sample ID."""

    _require_positive(top_n, "top_n")
    errors = [prediction for prediction in predictions if not prediction.correct]
    return tuple(
        sorted(errors, key=lambda item: (-item.confidence, item.sample_id))[:top_n]
    )


def per_class_failure_summary(
    predictions: Sequence[FailurePrediction],
    *,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    """Summarize class support plus false-negative and false-positive counts."""

    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(class_names):
        support = sum(1 for item in predictions if item.true_label == class_name)
        correct = sum(
            1
            for item in predictions
            if item.true_label == class_name and item.predicted_label == class_name
        )
        false_negative = sum(
            1
            for item in predictions
            if item.true_label == class_name and item.predicted_label != class_name
        )
        false_positive = sum(
            1
            for item in predictions
            if item.true_label != class_name and item.predicted_label == class_name
        )
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "support": support,
                "correct": correct,
                "false_negative_count": false_negative,
                "false_positive_count": false_positive,
                "accuracy": correct / support if support else 0.0,
            }
        )
    return tuple(rows)


def select_per_class_failure_examples(
    predictions: Sequence[FailurePrediction],
    *,
    class_names: Sequence[str],
    top_n_per_category: int,
) -> tuple[dict[str, Any], ...]:
    """Select deterministic false-negative and false-positive examples per class."""

    _require_positive(top_n_per_category, "top_n_per_category")
    rows: list[dict[str, Any]] = []
    for class_name in class_names:
        categories = (
            (
                "false_negative",
                [
                    item
                    for item in predictions
                    if item.true_label == class_name and item.predicted_label != class_name
                ],
            ),
            (
                "false_positive",
                [
                    item
                    for item in predictions
                    if item.true_label != class_name and item.predicted_label == class_name
                ],
            ),
        )
        for category, candidates in categories:
            selected = sorted(candidates, key=lambda item: (-item.confidence, item.sample_id))[
                :top_n_per_category
            ]
            for rank, item in enumerate(selected, start=1):
                rows.append(
                    {
                        "run_id": item.run_id,
                        "context_id": item.context_id,
                        "class_name": class_name,
                        "failure_category": category,
                        "rank": rank,
                        **item.to_dict(),
                    }
                )
    return tuple(rows)


def confusion_pair_summary(
    predictions: Sequence[FailurePrediction],
) -> tuple[dict[str, Any], ...]:
    """Rank true-label -> predicted-label error pairs by count."""

    grouped: dict[tuple[str, str], list[FailurePrediction]] = {}
    for item in predictions:
        if item.correct:
            continue
        grouped.setdefault((item.true_label, item.predicted_label), []).append(item)

    rows = []
    for (true_label, predicted_label), items in grouped.items():
        rows.append(
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": len(items),
                "average_confidence": sum(item.confidence for item in items) / len(items),
                "max_confidence": max(item.confidence for item in items),
            }
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -int(row["count"]),
                str(row["true_label"]),
                str(row["predicted_label"]),
            ),
        )
    )


def select_confusion_pair_examples(
    predictions: Sequence[FailurePrediction],
    *,
    top_pairs: int,
    examples_per_pair: int,
) -> tuple[dict[str, Any], ...]:
    """Select examples from the top-ranked confusion pairs."""

    _require_positive(top_pairs, "top_pairs")
    _require_positive(examples_per_pair, "examples_per_pair")
    rows: list[dict[str, Any]] = []
    for pair_rank, pair in enumerate(confusion_pair_summary(predictions)[:top_pairs], start=1):
        candidates = [
            item
            for item in predictions
            if item.true_label == pair["true_label"]
            and item.predicted_label == pair["predicted_label"]
        ]
        selected = sorted(candidates, key=lambda item: (-item.confidence, item.sample_id))[
            :examples_per_pair
        ]
        for example_rank, item in enumerate(selected, start=1):
            rows.append(
                {
                    "pair_rank": pair_rank,
                    "example_rank": example_rank,
                    "pair_count": pair["count"],
                    **item.to_dict(),
                }
            )
    return tuple(rows)


def select_model_disagreements(
    predictions_by_run: dict[str, Sequence[FailurePrediction]],
    *,
    top_n: int,
) -> tuple[dict[str, Any], ...]:
    """Select aligned samples where not all compared models predict the same label."""

    _require_positive(top_n, "top_n")
    aligned = align_predictions_by_sample(predictions_by_run)
    rows: list[dict[str, Any]] = []
    for sample_id, predictions in aligned.items():
        predicted_labels = {item.predicted_label for item in predictions}
        if len(predicted_labels) <= 1:
            continue
        confidences = [item.confidence for item in predictions]
        incorrect_count = sum(1 for item in predictions if not item.correct)
        rows.append(
            {
                "sample_id": sample_id,
                "context_id": predictions[0].context_id,
                "dataset_id": predictions[0].dataset_id,
                "dataset_version": predictions[0].dataset_version,
                "split": predictions[0].split,
                "condition_id": predictions[0].condition_id,
                "source_id": predictions[0].source_id,
                "true_label": predictions[0].true_label,
                "disagreement": True,
                "distinct_prediction_count": len(predicted_labels),
                "incorrect_model_count": incorrect_count,
                "confidence_spread": max(confidences) - min(confidences),
                "run_predictions": json.dumps(
                    [
                        {
                            "run_id": item.run_id,
                            "predicted_label": item.predicted_label,
                            "confidence": item.confidence,
                            "correct": item.correct,
                        }
                        for item in sorted(predictions, key=lambda item: item.run_id)
                    ]
                ),
            }
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -int(row["distinct_prediction_count"]),
                -int(row["incorrect_model_count"]),
                -float(row["confidence_spread"]),
                str(row["sample_id"]),
            ),
        )[:top_n]
    )


def align_predictions_by_sample(
    predictions_by_run: dict[str, Sequence[FailurePrediction]],
) -> dict[str, tuple[FailurePrediction, ...]]:
    """Require identical sample IDs, labels, context, split, and condition across runs."""

    if len(predictions_by_run) < 2:
        raise ValueError("at least two runs are required for model-disagreement analysis")
    ordered_run_ids = sorted(predictions_by_run)
    reference_run = ordered_run_ids[0]
    reference = list(predictions_by_run[reference_run])
    reference_ids = [item.sample_id for item in reference]
    aligned: dict[str, list[FailurePrediction]] = {
        item.sample_id: [item] for item in reference
    }
    for run_id in ordered_run_ids[1:]:
        candidate = list(predictions_by_run[run_id])
        candidate_ids = [item.sample_id for item in candidate]
        if candidate_ids != reference_ids:
            raise ValueError(f"sample identity mismatch between {reference_run} and {run_id}")
        for ref_item, candidate_item in zip(reference, candidate, strict=True):
            if candidate_item.true_label != ref_item.true_label:
                raise ValueError(f"label mismatch for sample {ref_item.sample_id} in {run_id}")
            if candidate_item.context_id != ref_item.context_id:
                raise ValueError(f"context mismatch for sample {ref_item.sample_id} in {run_id}")
            if candidate_item.split != ref_item.split:
                raise ValueError(f"split mismatch for sample {ref_item.sample_id} in {run_id}")
            if candidate_item.condition_id != ref_item.condition_id:
                raise ValueError(f"condition mismatch for sample {ref_item.sample_id} in {run_id}")
            aligned[ref_item.sample_id].append(candidate_item)
    return {sample_id: tuple(items) for sample_id, items in aligned.items()}


def write_csv_rows(rows: Iterable[dict[str, Any]], path: Path) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields or ["empty"]


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
