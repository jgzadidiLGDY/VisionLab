"""Bounded checkpoint helpers for VisionLab Phase 3."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


CHECKPOINT_VERSION = 1


def _config_to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("model_config must be a dataclass, dict, or expose to_dict")


def model_identity(model: nn.Module) -> dict[str, Any]:
    return {
        "class_name": model.__class__.__name__,
        "config": _config_to_dict(getattr(model, "config", None)),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    run_id: str,
    seed: int,
    metrics: list[dict[str, Any]],
    scheduler: LRScheduler | None = None,
    tag: str = "checkpoint",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "tag": tag,
        "run_id": run_id,
        "seed": seed,
        "epoch": epoch,
        "model_identity": model_identity(model),
        "optimizer_class": optimizer.__class__.__name__,
        "scheduler_class": scheduler.__class__.__name__ if scheduler else None,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics": list(metrics),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    if expected_run_id is not None and payload.get("run_id") != expected_run_id:
        raise ValueError("checkpoint run_id does not match expected run")
    if payload.get("model_identity") != model_identity(model):
        raise ValueError("checkpoint model identity is incompatible")
    if optimizer is not None and payload.get("optimizer_class") != optimizer.__class__.__name__:
        raise ValueError("checkpoint optimizer is incompatible")
    if scheduler is not None and payload.get("scheduler_class") != scheduler.__class__.__name__:
        raise ValueError("checkpoint scheduler is incompatible")
    if scheduler is None and payload.get("scheduler_class") is not None:
        raise ValueError("checkpoint requires a compatible scheduler")

    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload
