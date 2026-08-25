"""Calibration and confidence summaries for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CalibrationBin:
    index: int
    lower: float
    upper: float
    count: int
    accuracy: float | None
    average_confidence: float | None
    gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "accuracy": self.accuracy,
            "average_confidence": self.average_confidence,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    total_examples: int
    num_bins: int
    expected_calibration_error: float
    maximum_calibration_error: float
    average_confidence: float
    accuracy: float
    correct_average_confidence: float | None
    incorrect_average_confidence: float | None
    bins: tuple[CalibrationBin, ...]
    definitions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "num_bins": self.num_bins,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "average_confidence": self.average_confidence,
            "accuracy": self.accuracy,
            "correct_average_confidence": self.correct_average_confidence,
            "incorrect_average_confidence": self.incorrect_average_confidence,
            "bins": [item.to_dict() for item in self.bins],
            "definitions": dict(self.definitions),
        }


def calibration_summary(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    num_bins: int = 10,
) -> CalibrationSummary:
    """Compute ECE using [lower, upper) bins, with the last bin inclusive of 1.0."""

    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if not confidences:
        raise ValueError("at least one confidence value is required")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    for confidence in confidences:
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence values must be in [0, 1]")

    total = len(confidences)
    bin_rows: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for confidence, is_correct in zip(confidences, correct):
        index = min(int(confidence * num_bins), num_bins - 1)
        bin_rows[index].append((float(confidence), bool(is_correct)))

    bins: list[CalibrationBin] = []
    ece = 0.0
    mce = 0.0
    for index, rows in enumerate(bin_rows):
        lower = index / num_bins
        upper = (index + 1) / num_bins
        if not rows:
            bins.append(
                CalibrationBin(
                    index=index,
                    lower=lower,
                    upper=upper,
                    count=0,
                    accuracy=None,
                    average_confidence=None,
                    gap=None,
                )
            )
            continue
        count = len(rows)
        accuracy = sum(1 for _confidence, is_correct in rows if is_correct) / count
        average_confidence = sum(confidence for confidence, _is_correct in rows) / count
        gap = abs(accuracy - average_confidence)
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append(
            CalibrationBin(
                index=index,
                lower=lower,
                upper=upper,
                count=count,
                accuracy=accuracy,
                average_confidence=average_confidence,
                gap=gap,
            )
        )

    correct_conf = [confidence for confidence, is_correct in zip(confidences, correct) if is_correct]
    incorrect_conf = [confidence for confidence, is_correct in zip(confidences, correct) if not is_correct]
    return CalibrationSummary(
        total_examples=total,
        num_bins=num_bins,
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        average_confidence=sum(confidences) / total,
        accuracy=sum(1 for item in correct if item) / total,
        correct_average_confidence=_mean_or_none(correct_conf),
        incorrect_average_confidence=_mean_or_none(incorrect_conf),
        bins=tuple(bins),
        definitions={
            "binning": "bin index is min(int(confidence * num_bins), num_bins - 1); bins are [lower, upper) except the final bin includes 1.0",
            "expected_calibration_error": "sum over bins of bin_fraction * abs(bin_accuracy - bin_average_confidence)",
            "maximum_calibration_error": "maximum non-empty-bin absolute calibration gap",
            "confidence": "maximum predicted class probability, not guaranteed correctness",
        },
    )


def confidence_histogram(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    num_bins: int = 10,
) -> list[dict[str, Any]]:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    rows = [
        {
            "bin_index": index,
            "lower": index / num_bins,
            "upper": (index + 1) / num_bins,
            "correct_count": 0,
            "incorrect_count": 0,
        }
        for index in range(num_bins)
    ]
    for confidence, is_correct in zip(confidences, correct):
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence values must be in [0, 1]")
        index = min(int(confidence * num_bins), num_bins - 1)
        key = "correct_count" if is_correct else "incorrect_count"
        rows[index][key] += 1
    return rows


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
