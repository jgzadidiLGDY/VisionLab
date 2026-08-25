"""Classification evaluation artifacts for VisionLab."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from visionlab.training.config import TrainingRunMetadata


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    split: str
    true_label: str
    predicted_label: str
    confidence: float
    correct: bool
    source_id: str = ""
    true_index: int | None = None
    predicted_index: int | None = None
    logits: tuple[float, ...] = ()
    probabilities: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "correct": self.correct,
            "source_id": self.source_id,
            "true_index": self.true_index,
            "predicted_index": self.predicted_index,
            "logits": list(self.logits),
            "probabilities": list(self.probabilities),
        }


@dataclass(frozen=True)
class ClassificationSummary:
    split: str
    total_examples: int
    loss: float
    accuracy: float
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "total_examples": self.total_examples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True)
class ClassificationEvaluation:
    summary: ClassificationSummary
    predictions: tuple[PredictionRecord, ...]


def evaluate_classification(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    class_names: tuple[str, ...],
    split: str,
    loss_fn: nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> ClassificationEvaluation:
    """Evaluate a classification model and preserve prediction-level records."""

    if not class_names:
        raise ValueError("class_names must not be empty")
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.eval()
    device = torch.device(device)
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    num_classes = len(class_names)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    class_totals = {name: 0 for name in class_names}
    class_correct = {name: 0 for name in class_names}
    predictions: list[PredictionRecord] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite evaluation loss")

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((predicted == labels).sum().item())
            total_examples += batch_size

            sample_ids = _as_list(batch["sample_id"])
            source_ids = _as_list(batch.get("source_id", [""] * batch_size))
            splits = _as_list(batch.get("split", [split] * batch_size))
            for row in range(batch_size):
                true_index = int(labels[row].item())
                predicted_index = int(predicted[row].item())
                _validate_label_index(true_index, class_names, "true")
                _validate_label_index(predicted_index, class_names, "predicted")
                true_label = class_names[true_index]
                predicted_label = class_names[predicted_index]
                is_correct = true_index == predicted_index
                class_totals[true_label] += 1
                if is_correct:
                    class_correct[true_label] += 1
                confusion[true_index][predicted_index] += 1
                predictions.append(
                    PredictionRecord(
                        sample_id=str(sample_ids[row]),
                        split=str(splits[row]),
                        true_label=true_label,
                        predicted_label=predicted_label,
                        confidence=float(confidence[row].item()),
                        correct=is_correct,
                        source_id=str(source_ids[row]),
                        true_index=true_index,
                        predicted_index=predicted_index,
                        logits=tuple(float(value) for value in logits[row].detach().cpu().tolist()),
                        probabilities=tuple(
                            float(value) for value in probabilities[row].detach().cpu().tolist()
                        ),
                    )
                )

    if total_examples == 0:
        raise ValueError("evaluation data_loader produced no examples")

    per_class = {}
    for name in class_names:
        total = class_totals[name]
        correct = class_correct[name]
        per_class[name] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
    return ClassificationEvaluation(
        summary=ClassificationSummary(
            split=split,
            total_examples=total_examples,
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
            per_class=per_class,
            confusion_matrix=confusion,
        ),
        predictions=tuple(predictions),
    )


def write_evaluation_artifacts(
    evaluation: ClassificationEvaluation,
    output_dir: Path,
    *,
    prefix: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prefix}_summary.json"
    predictions_path = output_dir / f"{prefix}_predictions.csv"
    summary_path.write_text(
        json.dumps(evaluation.summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "split",
                "true_label",
                "predicted_label",
                "confidence",
                "correct",
                "source_id",
                "true_index",
                "predicted_index",
                "logits",
                "probabilities",
            ],
        )
        writer.writeheader()
        for record in evaluation.predictions:
            row = record.to_dict()
            row["logits"] = json.dumps(row["logits"])
            row["probabilities"] = json.dumps(row["probabilities"])
            writer.writerow(row)
    return {
        "summary": str(summary_path),
        "predictions": str(predictions_path),
    }


def write_history_artifacts(
    metadata: TrainingRunMetadata,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = [metric.to_dict() for metric in metadata.epoch_history]
    history_path = output_dir / "history.json"
    curve_path = output_dir / "curve_data.csv"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
            ],
        )
        writer.writeheader()
        for metric in history:
            writer.writerow(metric)
    return {
        "history": str(history_path),
        "curve_data": str(curve_path),
    }


def _validate_label_index(index: int, class_names: tuple[str, ...], role: str) -> None:
    if index < 0 or index >= len(class_names):
        raise ValueError(f"{role} label index {index} is outside class_names")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)
