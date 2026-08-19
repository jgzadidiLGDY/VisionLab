"""Phase 4A bounded baseline-plumbing smoke workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    DataLoaderPolicy,
    build_cifar10_split_datasets,
    build_phase4_dataloaders,
)
from visionlab.evaluation import (
    evaluate_classification,
    write_evaluation_artifacts,
    write_history_artifacts,
)
from visionlab.models import CustomCNN, CustomCNNConfig
from visionlab.training import OptimizerConfig, TrainingConfig, fit
from visionlab.training.checkpoints import load_checkpoint


PHASE4A_SMOKE_RUN_ID = "phase4a-smoke-plumbing"
PHASE4A_SEED = 20260818


@dataclass(frozen=True)
class Phase4ASmokeResult:
    run_dir: Path
    run_id: str
    status: str
    best_epoch: int | None
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "status": self.status,
            "best_epoch": self.best_epoch,
            "artifact_paths": dict(self.artifact_paths),
        }


class TinyCIFARLikeDataset(Dataset):
    """Tiny 10-class CIFAR-shaped dataset for plumbing tests only."""

    classes = CIFAR10_CLASSES

    def __init__(self, repeats_per_class: int, *, offset: float = 0.0) -> None:
        if repeats_per_class <= 0:
            raise ValueError("repeats_per_class must be positive")
        self.targets: list[int] = []
        self.images: list[torch.Tensor] = []
        for label in range(len(self.classes)):
            for repeat in range(repeats_per_class):
                image = torch.zeros(3, 32, 32)
                channel = label % 3
                image[channel, :, :] = min(1.0, 0.15 + label * 0.07 + offset)
                image[:, repeat % 32, :] = min(1.0, 0.25 + offset)
                self.images.append(image)
                self.targets.append(label)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


def run_phase4a_smoke(run_dir: Path | str) -> Phase4ASmokeResult:
    """Run a deliberately tiny end-to-end check of Phase 4 baseline plumbing."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    datasets = build_cifar10_split_datasets(
        upstream_train=TinyCIFARLikeDataset(repeats_per_class=2),
        upstream_test=TinyCIFARLikeDataset(repeats_per_class=1, offset=0.05),
        validation_per_class=1,
    )
    policy = DataLoaderPolicy(
        batch_size=5,
        seed=PHASE4A_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
    )
    loaders = build_phase4_dataloaders(datasets, policy)

    config = TrainingConfig(
        run_id=PHASE4A_SMOKE_RUN_ID,
        seed=PHASE4A_SEED,
        max_epochs=1,
        optimizer=OptimizerConfig(name="adam", learning_rate=0.001),
        selection_metric="val_loss",
    )
    model_config = CustomCNNConfig(
        num_classes=len(CIFAR10_CLASSES),
        feature_channels=(4, 8),
    )
    model = CustomCNN(model_config)
    result = fit(
        model,
        loaders.train,
        config=config,
        val_loader=loaders.val,
        run_dir=run_path,
    )

    artifacts: dict[str, str] = {}
    artifacts.update(write_history_artifacts(result.metadata, run_path / "artifacts"))
    artifacts.update(
        write_selected_checkpoint_evaluation_artifacts(
            checkpoint_path=Path(result.metadata.checkpoint_references["best"]),
            model_config=model_config,
            run_id=config.run_id,
            val_loader=loaders.prediction_val,
            test_loader=loaders.prediction_test,
            output_dir=run_path / "artifacts",
            include_test=False,
            val_prefix="val_smoke",
        )
    )

    contract = {
        "phase": "4A",
        "scope": "pipeline evidence only; not an official baseline result",
        "dataset_contract": datasets.to_contract_dict(),
        "dataloader_policy": policy.to_dict(),
        "training_config": config.to_dict(),
        "test_loader_constructed_count": len(loaders.test.dataset),
        "official_test_evaluation": False,
        "selected_checkpoint_for_evaluation": result.metadata.checkpoint_references["best"],
        "checkpoint_selection_metric": config.selection_metric,
    }
    contract_path = run_path / "run_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    artifacts["run_contract"] = str(contract_path)

    result_path = run_path / "phase4a_smoke_result.json"
    artifacts["smoke_result"] = str(result_path)
    smoke_result = Phase4ASmokeResult(
        run_dir=run_path,
        run_id=config.run_id,
        status=result.status,
        best_epoch=result.best_epoch,
        artifact_paths=artifacts,
    )
    result_path.write_text(json.dumps(smoke_result.to_dict(), indent=2), encoding="utf-8")
    return smoke_result


def write_selected_checkpoint_evaluation_artifacts(
    *,
    checkpoint_path: Path,
    model_config: CustomCNNConfig,
    run_id: str,
    val_loader: Any,
    test_loader: Any,
    output_dir: Path,
    include_test: bool,
    val_prefix: str = "val",
    test_prefix: str = "test",
) -> dict[str, str]:
    """Restore the selected checkpoint before writing validation/test artifacts."""

    restored_model = CustomCNN(model_config)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        expected_run_id=run_id,
    )
    if checkpoint.get("tag") != "best":
        raise ValueError("selected checkpoint for evaluation must have tag 'best'")

    artifacts: dict[str, str] = {
        "selected_checkpoint": str(checkpoint_path),
        "selected_checkpoint_tag": str(checkpoint["tag"]),
    }
    val_evaluation = evaluate_classification(
        restored_model,
        val_loader,
        class_names=CIFAR10_CLASSES,
        split="val",
    )
    artifacts.update(
        {
            f"val_{name}": path
            for name, path in write_evaluation_artifacts(
                val_evaluation,
                output_dir,
                prefix=val_prefix,
            ).items()
        }
    )
    if include_test:
        test_evaluation = evaluate_classification(
            restored_model,
            test_loader,
            class_names=CIFAR10_CLASSES,
            split="test",
        )
        artifacts.update(
            {
                f"test_{name}": path
                for name, path in write_evaluation_artifacts(
                    test_evaluation,
                    output_dir,
                    prefix=test_prefix,
                ).items()
            }
        )
    return artifacts
