"""Phase 6A transfer-model contract and tiny smoke paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from visionlab.models import (
    RESNET18_WEIGHT_ENUM,
    TransferModelConfig,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
    verify_phase6a_parameter_counts,
)
from visionlab.training.checkpoints import save_checkpoint
from visionlab.training.reproducibility import apply_reproducibility, environment_summary


PHASE6A_MECHANICS_SMOKE_RUN_ID = "phase6a-resnet18-mechanics-smoke"
PHASE6A_PRETRAINED_SMOKE_RUN_ID = "phase6a-resnet18-pretrained-frozen-smoke"
PHASE6A_SEED = 20260820
PHASE6A_BASELINE_REFERENCE_RUN_ID = "phase4b-cifar10-custom-cnn-baseline-001"


@dataclass(frozen=True)
class Phase6ASmokeResult:
    run_dir: Path
    run_id: str
    status: str
    pretrained_weights_loaded: bool
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "status": self.status,
            "pretrained_weights_loaded": self.pretrained_weights_loaded,
            "artifact_paths": dict(self.artifact_paths),
        }


def run_phase6a_model_mechanics_smoke(run_dir: Path | str) -> Phase6ASmokeResult:
    """Verify transfer-model mechanics without loading pretrained weights."""

    return _run_phase6a_smoke(
        run_dir,
        run_id=PHASE6A_MECHANICS_SMOKE_RUN_ID,
        load_pretrained=False,
        evidence_boundary=(
            "mechanics evidence only; random initialization is not transfer-learning evidence"
        ),
    )


def run_phase6a_pretrained_frozen_smoke(run_dir: Path | str) -> Phase6ASmokeResult:
    """Verify frozen-feature mechanics with cached pretrained weights only."""

    probe = probe_resnet18_weight_cache()
    if not probe.exists:
        raise FileNotFoundError(
            "ResNet18_Weights.IMAGENET1K_V1 is not cached; download approval is required"
        )
    return _run_phase6a_smoke(
        run_dir,
        run_id=PHASE6A_PRETRAINED_SMOKE_RUN_ID,
        load_pretrained=True,
        evidence_boundary=(
            "pretrained frozen-feature smoke only; not material CIFAR-10 training"
        ),
    )


def _run_phase6a_smoke(
    run_dir: Path | str,
    *,
    run_id: str,
    load_pretrained: bool,
    evidence_boundary: str,
) -> Phase6ASmokeResult:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    apply_reproducibility(PHASE6A_SEED)
    probe = probe_resnet18_weight_cache()
    model_config = TransferModelConfig()
    model = build_phase6a_transfer_model(load_pretrained=load_pretrained, config=model_config)
    counts = verify_phase6a_parameter_counts(model)
    before = _snapshot_parameters(model)

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    inputs, labels = _tiny_synthetic_batch()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite Phase 6A smoke loss")
    loss.backward()

    frozen_gradients_blocked = all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    optimizer.step()
    after = _snapshot_parameters(model)
    frozen_unchanged = all(
        torch.equal(before[name], after[name])
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    head_updated = any(
        not torch.equal(before[name], after[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if tuple(logits.shape) != (2, 10):
        raise ValueError(f"expected logits shape (2, 10), got {tuple(logits.shape)}")
    if not frozen_gradients_blocked:
        raise ValueError("frozen parameters received gradients")
    if not frozen_unchanged:
        raise ValueError("frozen backbone parameters changed during smoke step")
    if not head_updated:
        raise ValueError("classifier head did not update during smoke step")

    contract = {
        "phase": "6A",
        "scope": "transfer model contract and tiny non-material frozen-feature smoke",
        "run_id": run_id,
        "baseline_reference": {
            "run_id": PHASE6A_BASELINE_REFERENCE_RUN_ID,
            "preserved_unchanged": True,
            "comparison_asymmetry": (
                "Pretrained comparison is asymmetric because of ImageNet source "
                "pretraining, ResNet-18 model scale, 224x224 preprocessing/input "
                "resolution, and parameter count."
            ),
        },
        "model_identity": model.identity_dict(),
        "weight_cache_probe": probe.to_dict(),
        "pretrained_weights_loaded": load_pretrained,
        "evidence_boundary": evidence_boundary,
        "official_test_evaluation": False,
        "material_cifar10_training": False,
        "fine_tuning": False,
        "seed": PHASE6A_SEED,
        "optimizer": {
            "name": "sgd",
            "learning_rate": 0.01,
            "parameter_scope": "trainable classifier head only",
        },
        "smoke_checks": {
            "logits_shape": list(logits.shape),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "classifier_head_updated": head_updated,
            "frozen_backbone_unchanged": frozen_unchanged,
            "frozen_gradients_blocked": frozen_gradients_blocked,
            "parameter_counts": counts,
        },
        "environment": environment_summary("cpu"),
    }
    contract_path = run_path / "run_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    checkpoint_path = save_checkpoint(
        run_path / "checkpoints" / "mechanics.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        run_id=run_id,
        seed=PHASE6A_SEED,
        metrics=[
            {
                "loss": float(loss.detach().item()),
                "accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
                "evidence_boundary": evidence_boundary,
            }
        ],
        tag="phase6a-mechanics-smoke",
    )
    metadata = {
        "run_id": run_id,
        "status": "completed",
        "pretrained_weights_loaded": load_pretrained,
        "weight_identity": RESNET18_WEIGHT_ENUM,
        "contract": str(contract_path),
        "checkpoint": str(checkpoint_path),
        "evidence_boundary": evidence_boundary,
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifacts = {
        "run_contract": str(contract_path),
        "metadata": str(metadata_path),
        "checkpoint": str(checkpoint_path),
    }
    result = Phase6ASmokeResult(
        run_dir=run_path,
        run_id=run_id,
        status="completed",
        pretrained_weights_loaded=load_pretrained,
        artifact_paths=artifacts,
    )
    result_path = run_path / "phase6a_smoke_result.json"
    artifacts["smoke_result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _tiny_synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.zeros(2, 3, 224, 224)
    inputs[0, 0, :, :] = 1.0
    inputs[1, 1, :, :] = 1.0
    labels = torch.tensor([0, 1], dtype=torch.long)
    return inputs, labels


def _snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
