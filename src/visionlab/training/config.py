"""Training configuration and run-result contracts for VisionLab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in {"adam", "sgd"}:
            raise ValueError("optimizer name must be adam or sgd")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "step"
    step_size: int = 1
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.name != "step":
            raise ValueError("scheduler name must be step")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step_size": self.step_size,
            "gamma": self.gamma,
        }


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    seed: int = 20260817
    max_epochs: int = 1
    device: str = "cpu"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    checkpoint_best: bool = True
    checkpoint_terminal: bool = True
    selection_metric: str = "val_loss"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.device != "cpu":
            raise ValueError("Phase 3 training verification supports cpu only")
        if self.selection_metric not in {"val_loss", "val_accuracy", "train_loss"}:
            raise ValueError("selection_metric is not supported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "device": self.device,
            "optimizer": self.optimizer.to_dict(),
            "scheduler": self.scheduler.to_dict() if self.scheduler else None,
            "checkpoint_best": self.checkpoint_best,
            "checkpoint_terminal": self.checkpoint_terminal,
            "selection_metric": self.selection_metric,
        }


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None
    learning_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True)
class TrainingRunMetadata:
    run_id: str
    config: dict[str, Any]
    seed: int
    environment: dict[str, Any]
    status: str
    epoch_history: tuple[EpochMetrics, ...]
    checkpoint_references: dict[str, str]
    stop_reason: str
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config,
            "seed": self.seed,
            "environment": self.environment,
            "status": self.status,
            "epoch_history": [metric.to_dict() for metric in self.epoch_history],
            "checkpoint_references": dict(self.checkpoint_references),
            "stop_reason": self.stop_reason,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class TrainingRunResult:
    metadata: TrainingRunMetadata
    best_metric: float | None
    best_epoch: int | None

    @property
    def status(self) -> str:
        return self.metadata.status
