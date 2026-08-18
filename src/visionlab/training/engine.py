"""Compact CPU training engine for VisionLab Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from visionlab.training.checkpoints import save_checkpoint
from visionlab.training.config import (
    EpochMetrics,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingRunMetadata,
    TrainingRunResult,
)
from visionlab.training.reproducibility import apply_reproducibility, environment_summary


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    if config.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {config.name}")


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig | None,
) -> LRScheduler | None:
    if config is None:
        return None
    if config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    raise ValueError(f"unsupported scheduler: {config.name}")


def current_learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("training data_loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")

            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

    if was_training:
        model.train()
    if total_examples == 0:
        raise ValueError("validation data_loader produced no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    config: TrainingConfig,
    val_loader: DataLoader | None = None,
    loss_fn: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    run_dir: Path | None = None,
) -> TrainingRunResult:
    apply_reproducibility(config.seed)
    device = torch.device(config.device)
    model.to(device)
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = optimizer or build_optimizer(model, config.optimizer)
    scheduler = scheduler or build_scheduler(optimizer, config.scheduler)

    history: list[EpochMetrics] = []
    checkpoint_references: dict[str, str] = {}
    best_metric: float | None = None
    best_epoch: int | None = None
    status = "completed"
    stop_reason = "max_epochs_reached"
    failure_reason = ""

    try:
        for epoch in range(1, config.max_epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
                device=device,
            )
            val_metrics = (
                validate(model, val_loader, loss_fn, device=device)
                if val_loader is not None
                else None
            )
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                val_loss=val_metrics["loss"] if val_metrics else None,
                val_accuracy=val_metrics["accuracy"] if val_metrics else None,
                learning_rate=current_learning_rate(optimizer),
            )
            history.append(epoch_metrics)

            selected = _selection_value(epoch_metrics, config.selection_metric)
            if _is_better(selected, best_metric, config.selection_metric):
                best_metric = selected
                best_epoch = epoch
                if run_dir is not None and config.checkpoint_best:
                    path = run_dir / "checkpoints" / "best.pt"
                    save_checkpoint(
                        path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        run_id=config.run_id,
                        seed=config.seed,
                        metrics=[metric.to_dict() for metric in history],
                        tag="best",
                    )
                    checkpoint_references["best"] = str(path)

            if scheduler is not None:
                scheduler.step()
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        status = "failed"
        stop_reason = "failure"
        failure_reason = str(exc)

    if run_dir is not None and history and config.checkpoint_terminal:
        path = run_dir / "checkpoints" / "terminal.pt"
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=history[-1].epoch,
            run_id=config.run_id,
            seed=config.seed,
            metrics=[metric.to_dict() for metric in history],
            tag="terminal",
        )
        checkpoint_references["terminal"] = str(path)

    metadata = TrainingRunMetadata(
        run_id=config.run_id,
        config=config.to_dict(),
        seed=config.seed,
        environment=environment_summary(config.device),
        status=status,
        epoch_history=tuple(history),
        checkpoint_references=checkpoint_references,
        stop_reason=stop_reason,
        failure_reason=failure_reason,
    )
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), indent=2),
            encoding="utf-8",
        )
    return TrainingRunResult(
        metadata=metadata,
        best_metric=best_metric,
        best_epoch=best_epoch,
    )


def _selection_value(metric: EpochMetrics, selection_metric: str) -> float:
    value = getattr(metric, selection_metric)
    if value is None:
        raise ValueError(f"selection metric {selection_metric} is unavailable")
    return float(value)


def _is_better(
    candidate: float,
    current_best: float | None,
    selection_metric: str,
) -> bool:
    if current_best is None:
        return True
    if selection_metric.endswith("loss"):
        return candidate < current_best
    return candidate > current_best
