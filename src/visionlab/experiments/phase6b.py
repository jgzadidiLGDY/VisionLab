"""Phase 6B-1 pretrained frozen-feature smoke and preprocessing verification."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from visionlab.data import (
    CIFAR10_CLASSES,
    DataLoaderPolicy,
    SplitDatasetBundle,
    build_cifar10_split_datasets,
    build_transfer_dataloaders,
    phase6b_preprocessing_contract_dict,
    preprocess_resnet18_imagenet_tensor,
    verify_material_cifar10_contract,
)
from visionlab.evaluation import (
    evaluate_classification,
    write_evaluation_artifacts,
    write_history_artifacts,
)
from visionlab.models import (
    RESNET18_WEIGHT_ENUM,
    TransferModelConfig,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
    verify_phase6a_parameter_counts,
)
from visionlab.training import OptimizerConfig, TrainingConfig, fit
from visionlab.training.checkpoints import load_checkpoint, save_checkpoint
from visionlab.training.reproducibility import apply_reproducibility, environment_summary


PHASE6B1_PRETRAINED_SMOKE_RUN_ID = "phase6b1-resnet18-pretrained-frozen-smoke"
PHASE6B2_RUN_ID = "phase6b2-cifar10-resnet18-frozen-feature-001"
PHASE6B1_SEED = 20260820
PHASE6B2_SEED = 20260820
PHASE6B1_BASELINE_REFERENCE_RUN_ID = "phase4b-cifar10-custom-cnn-baseline-001"
PHASE6B2_BASELINE_REFERENCE = {
    "run_id": "phase4b-cifar10-custom-cnn-baseline-001",
    "official_test_loss": 1.024515,
    "official_test_accuracy": 0.635900,
}


@dataclass(frozen=True)
class Phase6B2PreparedRun:
    run_dir: Path
    artifact_dir: Path
    datasets: SplitDatasetBundle
    preflight_report: dict[str, Any]
    preflight_path: Path
    preprocessing_contract_path: Path
    run_contract: dict[str, Any]
    run_contract_path: Path
    data_loader_policy: DataLoaderPolicy
    training_config: TrainingConfig
    model_config: TransferModelConfig


@dataclass(frozen=True)
class Phase6B2TimingProbeResult:
    selected_batch_size: int
    selected_device: str
    candidate_results: tuple[dict[str, Any], ...]
    recommendation_reason: str
    practical: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_batch_size": self.selected_batch_size,
            "selected_device": self.selected_device,
            "candidate_results": [dict(result) for result in self.candidate_results],
            "recommendation_reason": self.recommendation_reason,
            "practical": self.practical,
        }


@dataclass(frozen=True)
class Phase6B2Result:
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


@dataclass(frozen=True)
class Phase6B1SmokeResult:
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


def prepare_phase6b2_material_run(
    run_dir: Path | str,
    *,
    batch_size: int,
    device: str = "cpu",
    root: str | Path = "data",
    download: bool = False,
    upstream_train: Any | None = None,
    upstream_test: Any | None = None,
    validation_per_class: int = 500,
    expected_counts: dict[str, int] | None = None,
) -> Phase6B2PreparedRun:
    """Write the exact Phase 6B-2 material run contract and preflight artifacts."""

    if device != "cpu":
        raise ValueError("Phase 6B-2 local preflight supports cpu only")

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    weight_probe = probe_resnet18_weight_cache()
    if not weight_probe.exists:
        raise FileNotFoundError(
            "resnet18-f37072fd.pth is not cached; Phase 6B-2 cannot proceed"
        )
    datasets = build_cifar10_split_datasets(
        root=root,
        download=download,
        upstream_train=upstream_train,
        upstream_test=upstream_test,
        validation_per_class=validation_per_class,
        train_augmentation_profile=None,
    )
    preflight_report = verify_material_cifar10_contract(
        datasets,
        expected_counts=expected_counts,
    )
    preflight_report["weight_cache_probe"] = weight_probe.to_dict()
    preflight_report["preprocessing_probe"] = _preprocessing_probe(datasets)
    preflight_report["status"] = "passed"
    preflight_path = run_path / "preflight_report.json"
    preflight_path.write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")

    preprocessing_contract = phase6b_preprocessing_contract_dict()
    preprocessing_path = artifact_dir / "preprocessing_contract.json"
    preprocessing_path.write_text(
        json.dumps(preprocessing_contract, indent=2),
        encoding="utf-8",
    )

    policy = DataLoaderPolicy(
        batch_size=batch_size,
        seed=PHASE6B2_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )
    config = TrainingConfig(
        run_id=PHASE6B2_RUN_ID,
        seed=PHASE6B2_SEED,
        max_epochs=5,
        device=device,
        optimizer=OptimizerConfig(name="adam", learning_rate=0.001, weight_decay=0.0),
        scheduler=None,
        selection_metric="val_loss",
    )
    model_config = TransferModelConfig()
    model = build_phase6a_transfer_model(load_pretrained=True, config=model_config)
    parameter_counts = verify_phase6a_parameter_counts(model)

    run_contract = {
        "phase": "6B-2",
        "scope": (
            "single material frozen-feature transfer-learning run path; "
            "implementation/preflight only until separate training approval"
        ),
        "run_id": PHASE6B2_RUN_ID,
        "baseline_reference": dict(PHASE6B2_BASELINE_REFERENCE),
        "dataset_contract": datasets.to_contract_dict(),
        "preflight_report": str(preflight_path),
        "weight_cache_probe": weight_probe.to_dict(),
        "preprocessing_contract": preprocessing_contract,
        "preprocessing_contract_path": str(preprocessing_path),
        "dataloader_policy": policy.to_dict(),
        "training_config": config.to_dict(),
        "model_identity": model.identity_dict(),
        "parameter_counts": parameter_counts,
        "experimental_variable": {
            "name": "model_path",
            "phase4b_reference": "CustomCNN trained from scratch on 32x32 CIFAR-10",
            "phase6b2_candidate": (
                "ResNet-18 ImageNet pretrained frozen backbone with CIFAR-10 head"
            ),
        },
        "comparison_asymmetry": {
            "pretrained_source_data": "ImageNet source pretraining is present only for ResNet-18",
            "model_scale": "ResNet-18 has 11,181,642 parameters versus the compact CustomCNN",
            "input_resolution": "ResNet-18 uses 224x224 model inputs; Phase 4B CustomCNN uses 32x32",
            "preprocessing": "ResNet-18 uses ImageNet preprocessing; Phase 4B uses CIFAR-native preprocessing",
            "training_mode": "ResNet-18 freezes the backbone and trains only the head; CustomCNN was trained end to end",
        },
        "augmentation": "none",
        "fine_tuning": False,
        "partial_unfreezing": False,
        "differential_learning_rate_groups": False,
        "seed_sweep": False,
        "hyperparameter_search": False,
        "checkpoint_selection_metric": "val_loss",
        "evaluation_sequence": [
            "train frozen-feature model after separate material-run approval",
            "select best checkpoint by val_loss",
            "restore best checkpoint",
            "generate final validation artifacts",
            "evaluate official test once",
            "generate official test artifacts",
        ],
        "official_test_evaluation": {
            "enabled_for_material_run_only": True,
            "occurs_after_best_checkpoint_restore": True,
            "count": 1,
            "not_performed_during_preflight": True,
        },
    }
    contract_path = run_path / "run_contract.json"
    contract_path.write_text(json.dumps(run_contract, indent=2), encoding="utf-8")

    return Phase6B2PreparedRun(
        run_dir=run_path,
        artifact_dir=artifact_dir,
        datasets=datasets,
        preflight_report=preflight_report,
        preflight_path=preflight_path,
        preprocessing_contract_path=preprocessing_path,
        run_contract=run_contract,
        run_contract_path=contract_path,
        data_loader_policy=policy,
        training_config=config,
        model_config=model_config,
    )


def run_phase6b2_material_frozen_feature(
    run_dir: Path | str,
    *,
    batch_size: int,
    device: str = "cpu",
) -> Phase6B2Result:
    """Run the approved Phase 6B-2 material path.

    This function exists for the later material-run approval boundary. Do not call it
    during Phase 6B-2 implementation/preflight.
    """

    prepared = prepare_phase6b2_material_run(run_dir, batch_size=batch_size, device=device)
    loaders = build_transfer_dataloaders(prepared.datasets, prepared.data_loader_policy)
    model = build_phase6a_transfer_model(load_pretrained=True, config=prepared.model_config)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=prepared.training_config.optimizer.learning_rate,
        weight_decay=prepared.training_config.optimizer.weight_decay,
    )
    result = fit(
        model,
        loaders.train,
        config=prepared.training_config,
        val_loader=loaders.val,
        optimizer=optimizer,
        run_dir=prepared.run_dir,
    )

    artifacts = {
        "preflight_report": str(prepared.preflight_path),
        "preprocessing_contract": str(prepared.preprocessing_contract_path),
        "run_contract": str(prepared.run_contract_path),
    }
    artifacts.update(write_history_artifacts(result.metadata, prepared.artifact_dir))

    if result.status != "completed":
        failure_report = _write_failure_report(prepared.run_dir, result.metadata.to_dict())
        artifacts["failure_report"] = str(failure_report)
        return _write_phase6b2_result(prepared.run_dir, result, artifacts)

    artifacts.update(
        write_selected_transfer_checkpoint_evaluation_artifacts(
            checkpoint_path=Path(result.metadata.checkpoint_references["best"]),
            run_id=prepared.training_config.run_id,
            val_loader=loaders.prediction_val,
            test_loader=loaders.prediction_test,
            output_dir=prepared.artifact_dir,
            include_test=True,
            val_prefix="val",
            test_prefix="test",
        )
    )
    comparison_report = write_phase6b2_comparison_report(
        prepared.run_dir,
        run_id=prepared.training_config.run_id,
        best_epoch=result.best_epoch,
        artifact_paths=artifacts,
    )
    artifacts["comparison_report"] = str(comparison_report)
    return _write_phase6b2_result(prepared.run_dir, result, artifacts)


def run_phase6b1_pretrained_smoke(run_dir: Path | str) -> Phase6B1SmokeResult:
    """Run cached-weight ResNet-18 preprocessing and frozen-feature smoke only."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    apply_reproducibility(PHASE6B1_SEED)
    probe = probe_resnet18_weight_cache()
    if not probe.exists:
        raise FileNotFoundError(
            "resnet18-f37072fd.pth is not cached; explicit download approval is required"
        )

    raw_inputs, labels = _tiny_cifar_like_batch()
    preprocessed_inputs = torch.stack(
        [preprocess_resnet18_imagenet_tensor(raw_inputs[index]) for index in range(raw_inputs.shape[0])]
    )
    if tuple(preprocessed_inputs.shape) != (2, 3, 224, 224):
        raise ValueError(
            "expected preprocessed batch shape (2, 3, 224, 224), "
            f"got {tuple(preprocessed_inputs.shape)}"
        )

    model = build_phase6a_transfer_model(load_pretrained=True)
    counts = verify_phase6a_parameter_counts(model)
    before = _snapshot_parameters(model)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(preprocessed_inputs)
    loss = loss_fn(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite Phase 6B-1 smoke loss")
    loss.backward()
    frozen_gradients_blocked = all(
        parameter.grad is None
        for parameter in model.parameters()
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
        raise ValueError("frozen backbone parameters changed during pretrained smoke")
    if not head_updated:
        raise ValueError("classifier head did not update during pretrained smoke")

    contract = {
        "phase": "6B-1",
        "scope": "cached-weight pretrained frozen-feature smoke and preprocessing verification only",
        "run_id": PHASE6B1_PRETRAINED_SMOKE_RUN_ID,
        "baseline_reference": {
            "run_id": PHASE6B1_BASELINE_REFERENCE_RUN_ID,
            "preserved_unchanged": True,
            "comparison_asymmetry": (
                "Future ResNet-18 comparison is asymmetric because of ImageNet source "
                "pretraining, model scale, 224x224 input resolution, preprocessing, "
                "and parameter count."
            ),
        },
        "model_identity": model.identity_dict(),
        "weight_cache_probe": probe.to_dict(),
        "pretrained_weights_loaded": True,
        "preprocessing_contract": phase6b_preprocessing_contract_dict(),
        "official_test_evaluation": False,
        "material_cifar10_training": False,
        "fine_tuning": False,
        "seed": PHASE6B1_SEED,
        "optimizer": {
            "name": "sgd",
            "learning_rate": 0.01,
            "parameter_scope": "trainable classifier head only",
        },
        "smoke_checks": {
            "raw_input_shape": list(raw_inputs.shape),
            "preprocessed_batch_shape": list(preprocessed_inputs.shape),
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
    preprocessing_path = artifact_dir / "preprocessing_contract.json"
    preprocessing_path.write_text(
        json.dumps(contract["preprocessing_contract"], indent=2),
        encoding="utf-8",
    )
    checkpoint_path = save_checkpoint(
        run_path / "checkpoints" / "pretrained_smoke.pt",
        model=model,
        optimizer=optimizer,
        epoch=0,
        run_id=PHASE6B1_PRETRAINED_SMOKE_RUN_ID,
        seed=PHASE6B1_SEED,
        metrics=[
            {
                "loss": float(loss.detach().item()),
                "accuracy": float((logits.argmax(dim=1) == labels).float().mean().item()),
                "evidence_boundary": "pretrained frozen-feature smoke only; not material training",
            }
        ],
        tag="phase6b1-pretrained-frozen-smoke",
    )
    metadata = {
        "run_id": PHASE6B1_PRETRAINED_SMOKE_RUN_ID,
        "status": "completed",
        "pretrained_weights_loaded": True,
        "weight_identity": RESNET18_WEIGHT_ENUM,
        "contract": str(contract_path),
        "checkpoint": str(checkpoint_path),
        "preprocessing_contract": str(preprocessing_path),
        "evidence_boundary": "pretrained frozen-feature smoke only; not material training",
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    artifacts = {
        "run_contract": str(contract_path),
        "metadata": str(metadata_path),
        "preprocessing_contract": str(preprocessing_path),
        "checkpoint": str(checkpoint_path),
    }
    result = Phase6B1SmokeResult(
        run_dir=run_path,
        run_id=PHASE6B1_PRETRAINED_SMOKE_RUN_ID,
        status="completed",
        pretrained_weights_loaded=True,
        artifact_paths=artifacts,
    )
    result_path = run_path / "phase6b1_smoke_result.json"
    artifacts["smoke_result"] = str(result_path)
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def write_selected_transfer_checkpoint_evaluation_artifacts(
    *,
    checkpoint_path: Path,
    run_id: str,
    val_loader: Any,
    test_loader: Any,
    output_dir: Path,
    include_test: bool,
    val_prefix: str = "val",
    test_prefix: str = "test",
) -> dict[str, str]:
    """Restore a selected transfer checkpoint before writing validation/test artifacts."""

    restored_model = build_phase6a_transfer_model(load_pretrained=True)
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


def run_phase6b2_timing_probe(
    *,
    batch_sizes: tuple[int, ...] = (64, 128),
    root: str | Path = "data",
    max_batches: int = 2,
    output_path: Path | None = None,
) -> Phase6B2TimingProbeResult:
    """Estimate CPU material-run practicality from a few train batches only."""

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    datasets = build_cifar10_split_datasets(root=root, download=False)
    preflight_report = verify_material_cifar10_contract(datasets)
    if preflight_report["status"] != "passed":
        raise ValueError("material CIFAR-10 preflight did not pass")
    if not probe_resnet18_weight_cache().exists:
        raise FileNotFoundError("resnet18-f37072fd.pth is not cached")

    candidate_results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        policy = DataLoaderPolicy(
            batch_size=batch_size,
            seed=PHASE6B2_SEED,
            num_workers=0,
            train_shuffle=True,
            eval_shuffle=False,
            drop_last=False,
        )
        loaders = build_transfer_dataloaders(datasets, policy)
        model = build_phase6a_transfer_model(load_pretrained=True)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.001,
            weight_decay=0.0,
        )
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        start = time.perf_counter()
        examples = 0
        batches = 0
        for inputs, labels in loaders.train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite timing-probe loss")
            loss.backward()
            optimizer.step()
            examples += int(labels.shape[0])
            batches += 1
            if batches >= max_batches:
                break
        elapsed = time.perf_counter() - start
        seconds_per_batch = elapsed / batches
        batches_per_epoch = math.ceil(45_000 / batch_size)
        estimated_epoch_seconds = seconds_per_batch * batches_per_epoch
        candidate_results.append(
            {
                "batch_size": batch_size,
                "device": "cpu",
                "timed_batches": batches,
                "timed_examples": examples,
                "elapsed_seconds": elapsed,
                "seconds_per_batch": seconds_per_batch,
                "estimated_epoch_seconds": estimated_epoch_seconds,
                "estimated_five_epoch_seconds": estimated_epoch_seconds * 5,
            }
        )

    selected = min(
        candidate_results,
        key=lambda result: float(result["estimated_five_epoch_seconds"]),
    )
    practical = float(selected["estimated_five_epoch_seconds"]) <= 4 * 60 * 60
    result = Phase6B2TimingProbeResult(
        selected_batch_size=int(selected["batch_size"]),
        selected_device="cpu",
        candidate_results=tuple(candidate_results),
        recommendation_reason=(
            "Selected the candidate with the lowest estimated 5-epoch CPU runtime "
            "from identical few-batch timing probes."
        ),
        practical=practical,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def write_phase6b2_comparison_report(
    run_dir: Path,
    *,
    run_id: str,
    best_epoch: int | None,
    artifact_paths: dict[str, str],
) -> Path:
    val_summary = json.loads(Path(artifact_paths["val_summary"]).read_text(encoding="utf-8"))
    test_summary = json.loads(Path(artifact_paths["test_summary"]).read_text(encoding="utf-8"))
    history = json.loads(Path(artifact_paths["history"]).read_text(encoding="utf-8"))
    accuracy_delta = test_summary["accuracy"] - PHASE6B2_BASELINE_REFERENCE["official_test_accuracy"]
    loss_delta = test_summary["loss"] - PHASE6B2_BASELINE_REFERENCE["official_test_loss"]
    report_path = run_dir / "phase6b2_comparison_report.md"
    lines = [
        "# Phase 6B-2 Single-Run Frozen-Feature Transfer Report",
        "",
        "This report compares one approved frozen-feature ResNet-18 run against the accepted Phase 4B custom-CNN baseline. The comparison is intentionally asymmetric and is not an architecture-only attribution.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Comparison baseline: `{PHASE6B2_BASELINE_REFERENCE['run_id']}`",
        f"- Selected epoch: `{best_epoch}`",
        "- Model: `torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`",
        "- Trainable parameters: classifier head only",
        "- Checkpoint selection: minimum validation loss",
        "- Test use: evaluated once after restoring the selected best checkpoint",
        "",
        "## Current Run",
        "",
        f"- Final restored-best validation loss: `{val_summary['loss']:.6f}`",
        f"- Final restored-best validation accuracy: `{val_summary['accuracy']:.6f}`",
        f"- Official test loss: `{test_summary['loss']:.6f}`",
        f"- Official test accuracy: `{test_summary['accuracy']:.6f}`",
        "",
        "## Phase 4B Reference",
        "",
        f"- Official test loss: `{PHASE6B2_BASELINE_REFERENCE['official_test_loss']:.6f}`",
        f"- Official test accuracy: `{PHASE6B2_BASELINE_REFERENCE['official_test_accuracy']:.6f}`",
        "",
        "## Deltas Versus Phase 4B",
        "",
        f"- Official test loss delta: `{loss_delta:.6f}`",
        f"- Official test accuracy delta: `{accuracy_delta:.6f}`",
        "",
        "## Comparison Limitations",
        "",
        "- ResNet-18 uses ImageNet source pretraining.",
        "- ResNet-18 uses 224x224 inputs and ImageNet preprocessing; Phase 4B CustomCNN uses 32x32 inputs.",
        "- ResNet-18 has a much larger parameter scale.",
        "- Phase 6B-2 trains only the replacement classifier head while the custom CNN was trained end to end.",
        "- A result difference must not be attributed to architecture alone.",
        "",
        "## Training History",
        "",
        f"- Epochs completed: `{len(history)}`",
    ]
    if history:
        lines.extend(
            [
                f"- Terminal train loss: `{history[-1]['train_loss']:.6f}`",
                f"- Terminal train accuracy: `{history[-1]['train_accuracy']:.6f}`",
                f"- Terminal validation loss: `{history[-1]['val_loss']:.6f}`",
                f"- Terminal validation accuracy: `{history[-1]['val_accuracy']:.6f}`",
            ]
        )
    lines.extend(["", "## Preserved Artifacts", ""])
    for name in sorted(artifact_paths):
        lines.append(f"- `{name}`: `{artifact_paths[name]}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _tiny_cifar_like_batch() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.zeros(2, 3, 32, 32)
    inputs[0, 0, :, :] = 1.0
    inputs[1, 1, :, :] = 1.0
    labels = torch.tensor([0, 1], dtype=torch.long)
    return inputs, labels


def _snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _preprocessing_probe(datasets: SplitDatasetBundle) -> dict[str, Any]:
    sample = datasets.train[0]
    transformed = preprocess_resnet18_imagenet_tensor(sample["raw_input"])
    return {
        "source_sample_id": sample["sample_id"],
        "raw_input_shape": list(sample["raw_input"].shape),
        "preprocessed_shape": list(transformed.shape),
        "preprocessing_contract": phase6b_preprocessing_contract_dict(),
        "status": "passed",
    }


def _write_failure_report(run_dir: Path, metadata: dict[str, Any]) -> Path:
    path = run_dir / "failure_report.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _write_phase6b2_result(
    run_dir: Path,
    result: Any,
    artifact_paths: dict[str, str],
) -> Phase6B2Result:
    phase_result = Phase6B2Result(
        run_dir=run_dir,
        run_id=PHASE6B2_RUN_ID,
        status=result.status,
        best_epoch=result.best_epoch,
        artifact_paths=artifact_paths,
    )
    result_path = run_dir / "phase6b2_result.json"
    phase_result.artifact_paths["phase6b2_result"] = str(result_path)
    result_path.write_text(json.dumps(phase_result.to_dict(), indent=2), encoding="utf-8")
    return phase_result
