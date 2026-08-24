"""Phase 6C fine-tuning contract, material run, and preflight utilities."""

from __future__ import annotations

import hashlib
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
from visionlab.experiments.phase6b import (
    PHASE6B2_BASELINE_REFERENCE,
    PHASE6B2_RUN_ID,
)
from visionlab.evaluation import (
    evaluate_classification,
    write_evaluation_artifacts,
    write_history_artifacts,
)
from visionlab.models import (
    PHASE6A_FREEZE_MODE,
    PHASE6C_FINETUNE_MODE,
    RESNET18_WEIGHT_ENUM,
    TransferModelConfig,
    build_phase6a_transfer_model,
    probe_resnet18_weight_cache,
    transfer_parameter_summary,
)
from visionlab.training import OptimizerConfig, TrainingConfig, fit
from visionlab.training.checkpoints import load_checkpoint, model_identity
from visionlab.training.reproducibility import apply_reproducibility, environment_summary


PHASE6C_RUN_ID = "phase6c-cifar10-resnet18-layer4-finetune-001"
PHASE6C1_SEED = 20260820
PHASE6C_LEARNING_RATE = 0.0001
PHASE6C_WEIGHT_DECAY = 0.0
PHASE6C_EPOCH_BUDGET = 3
PHASE6C_INITIALIZATION_CHECKPOINT = (
    Path("outputs")
    / PHASE6B2_RUN_ID
    / "checkpoints"
    / "best.pt"
)


@dataclass(frozen=True)
class Phase6CInitializationIdentity:
    source_run_id: str
    checkpoint_tag: str
    checkpoint_epoch: int
    checkpoint_path: Path
    checkpoint_sha256: str
    source_weight_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "initialization_source_run_id": self.source_run_id,
            "initialization_checkpoint_tag": self.checkpoint_tag,
            "initialization_checkpoint_epoch": self.checkpoint_epoch,
            "initialization_checkpoint_path": str(self.checkpoint_path),
            "initialization_checkpoint_sha256": self.checkpoint_sha256,
            "source_weight_identity": self.source_weight_identity,
        }


@dataclass(frozen=True)
class Phase6CPreparedRun:
    run_dir: Path
    artifact_dir: Path
    datasets: SplitDatasetBundle
    preflight_report: dict[str, Any]
    preflight_path: Path
    run_contract: dict[str, Any]
    run_contract_path: Path
    mechanics_path: Path
    timing_path: Path | None
    data_loader_policy: DataLoaderPolicy
    training_config: TrainingConfig
    model_config: TransferModelConfig
    initialization_identity: Phase6CInitializationIdentity


@dataclass(frozen=True)
class Phase6CTimingProbeResult:
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
class Phase6CResult:
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


def phase6c_model_config() -> TransferModelConfig:
    return TransferModelConfig(freeze_mode=PHASE6C_FINETUNE_MODE)


def build_phase6c_finetune_model_from_phase6b2(
    checkpoint_path: Path | str = PHASE6C_INITIALIZATION_CHECKPOINT,
) -> tuple[nn.Module, Phase6CInitializationIdentity]:
    """Restore Phase 6B-2 best model state into the Phase 6C fine-tuning contract."""

    source_checkpoint = Path(checkpoint_path)
    identity = phase6c_initialization_identity(source_checkpoint)
    model = build_phase6a_transfer_model(
        load_pretrained=True,
        config=phase6c_model_config(),
    )
    payload = torch.load(source_checkpoint, map_location="cpu")
    frozen_source = build_phase6a_transfer_model(
        load_pretrained=True,
        config=TransferModelConfig(freeze_mode=PHASE6A_FREEZE_MODE),
    )
    if payload.get("model_identity") != model_identity(frozen_source):
        raise ValueError("Phase 6C initialization checkpoint model identity is incompatible")
    model.load_state_dict(payload["model_state"])
    return model, identity


def phase6c_initialization_identity(
    checkpoint_path: Path | str = PHASE6C_INITIALIZATION_CHECKPOINT,
) -> Phase6CInitializationIdentity:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Phase 6B-2 best checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("run_id") != PHASE6B2_RUN_ID:
        raise ValueError("Phase 6C initialization checkpoint must come from Phase 6B-2")
    if payload.get("tag") != "best":
        raise ValueError("Phase 6C initialization checkpoint must have tag 'best'")
    if int(payload.get("epoch", -1)) != 4:
        raise ValueError("Phase 6C initialization checkpoint must be Phase 6B-2 epoch 4")
    return Phase6CInitializationIdentity(
        source_run_id=PHASE6B2_RUN_ID,
        checkpoint_tag=str(payload["tag"]),
        checkpoint_epoch=int(payload["epoch"]),
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        source_weight_identity=RESNET18_WEIGHT_ENUM,
    )


def build_phase6c_optimizer(model: nn.Module) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=PHASE6C_LEARNING_RATE,
        weight_decay=PHASE6C_WEIGHT_DECAY,
    )
    verify_phase6c_optimizer_scope(model, optimizer)
    return optimizer


def verify_phase6c_optimizer_scope(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = {
        name
        for name, _ in model.named_parameters()
        if name.startswith("model.layer4.") or name.startswith("model.fc.")
    }
    if trainable != expected:
        raise ValueError(
            "Phase 6C trainable parameters must be exactly layer4 + fc; "
            f"got {sorted(trainable)}"
        )

    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_names = {
        name_by_id[id(parameter)]
        for group in optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) in name_by_id
    }
    if optimizer_names != trainable:
        raise ValueError(
            "Phase 6C optimizer parameters must exactly match trainable parameters; "
            f"got {sorted(optimizer_names)}"
        )
    frozen_in_optimizer = sorted(
        name for name in optimizer_names if name not in trainable
    )
    return {
        "trainable_parameter_names": sorted(trainable),
        "optimizer_parameter_names": sorted(optimizer_names),
        "frozen_parameters_in_optimizer": frozen_in_optimizer,
        "optimizer_matches_trainable_scope": True,
    }


def run_phase6c1_mechanics_smoke(
    run_dir: Path | str,
    *,
    checkpoint_path: Path | str = PHASE6C_INITIALIZATION_CHECKPOINT,
) -> dict[str, Any]:
    """Run a tiny in-memory fine-tuning mechanics smoke with no official evaluation."""

    apply_reproducibility(PHASE6C1_SEED)
    run_path = Path(run_dir)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model, initialization = build_phase6c_finetune_model_from_phase6b2(checkpoint_path)
    optimizer = build_phase6c_optimizer(model)
    scope = verify_phase6c_optimizer_scope(model, optimizer)
    before = _snapshot_parameters(model)

    raw_inputs, labels = _tiny_cifar_like_batch()
    inputs = torch.stack(
        [preprocess_resnet18_imagenet_tensor(raw_inputs[index]) for index in range(raw_inputs.shape[0])]
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite Phase 6C-1 mechanics loss")
    loss.backward()
    frozen_gradients_blocked = all(
        parameter.grad is None
        for _, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    optimizer.step()
    after = _snapshot_parameters(model)
    frozen_unchanged = all(
        torch.equal(before[name], after[name])
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    trainable_updated = any(
        not torch.equal(before[name], after[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if tuple(logits.shape) != (2, 10):
        raise ValueError(f"expected logits shape (2, 10), got {tuple(logits.shape)}")
    if not frozen_gradients_blocked:
        raise ValueError("Phase 6C frozen parameters received gradients")
    if not frozen_unchanged:
        raise ValueError("Phase 6C frozen parameters changed during mechanics smoke")
    if not trainable_updated:
        raise ValueError("Phase 6C trainable parameters did not update")

    summary = {
        "phase": "6C-1",
        "run_id": PHASE6C_RUN_ID,
        "status": "passed",
        "evidence_boundary": (
            "fine-tuning mechanics and readiness only; no material fine-tuning "
            "and no official test evaluation"
        ),
        "initialization": initialization.to_dict(),
        "model_identity": model.identity_dict(),
        "parameter_counts": transfer_parameter_summary(model)["counts"],
        "optimizer": {
            "name": "adam",
            "learning_rate": PHASE6C_LEARNING_RATE,
            "weight_decay": PHASE6C_WEIGHT_DECAY,
            "parameter_scope": "exactly parameters marked trainable by finetune_layer4_head",
        },
        "optimizer_scope": scope,
        "checks": {
            "raw_input_shape": list(raw_inputs.shape),
            "preprocessed_batch_shape": list(inputs.shape),
            "logits_shape": list(logits.shape),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "frozen_gradients_blocked": frozen_gradients_blocked,
            "frozen_parameters_unchanged": frozen_unchanged,
            "trainable_parameters_updated": trainable_updated,
            "official_test_evaluation": False,
            "material_fine_tuning": False,
        },
        "environment": environment_summary("cpu"),
    }
    mechanics_path = artifact_dir / "mechanics_smoke.json"
    mechanics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_phase6c1_finetuning_preflight(
    run_dir: Path | str,
    *,
    batch_size: int,
    checkpoint_path: Path | str = PHASE6C_INITIALIZATION_CHECKPOINT,
    device: str = "cpu",
    root: str | Path = "data",
    download: bool = False,
    upstream_train: Any | None = None,
    upstream_test: Any | None = None,
    validation_per_class: int = 500,
    expected_counts: dict[str, int] | None = None,
    run_timing_probe: bool = False,
) -> Phase6CPreparedRun:
    """Write Phase 6C-1 fine-tuning preflight artifacts without material training."""

    if device != "cpu":
        raise ValueError("Phase 6C-1 local preflight supports cpu only")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    weight_probe = probe_resnet18_weight_cache()
    if not weight_probe.exists:
        raise FileNotFoundError("resnet18-f37072fd.pth is not cached")
    model, initialization = build_phase6c_finetune_model_from_phase6b2(checkpoint_path)
    optimizer = build_phase6c_optimizer(model)
    optimizer_scope = verify_phase6c_optimizer_scope(model, optimizer)
    mechanics = run_phase6c1_mechanics_smoke(
        run_path,
        checkpoint_path=checkpoint_path,
    )
    mechanics_path = artifact_dir / "mechanics_smoke.json"

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
    preflight_report.update(
        {
            "phase": "6C-1",
            "status": "passed",
            "initialization": initialization.to_dict(),
            "weight_cache_probe": weight_probe.to_dict(),
            "preprocessing_probe": _preprocessing_probe(datasets),
            "official_test_evaluation": False,
            "material_fine_tuning": False,
        }
    )
    preflight_path = run_path / "preflight_report.json"
    preflight_path.write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")

    policy = DataLoaderPolicy(
        batch_size=batch_size,
        seed=PHASE6C1_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )
    config = TrainingConfig(
        run_id=PHASE6C_RUN_ID,
        seed=PHASE6C1_SEED,
        max_epochs=PHASE6C_EPOCH_BUDGET,
        device=device,
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=PHASE6C_LEARNING_RATE,
            weight_decay=PHASE6C_WEIGHT_DECAY,
        ),
        scheduler=None,
        selection_metric="val_loss",
    )
    run_contract = {
        "phase": "6C-1",
        "scope": (
            "fine-tuning contract, Phase 6B-2 checkpoint initialization, "
            "mechanics smoke, and preflight only"
        ),
        "run_id": PHASE6C_RUN_ID,
        "phase6b2_reference": {
            "frozen_feature_run_id": PHASE6B2_RUN_ID,
            "official_test_loss": 0.413686,
            "official_test_accuracy": 0.856100,
            "best_checkpoint_epoch": 4,
            "fixed_reference_not_training_target": True,
        },
        "phase4b_reference": dict(PHASE6B2_BASELINE_REFERENCE),
        "initialization": initialization.to_dict(),
        "experimental_intervention": (
            "Change the training regime from frozen-backbone/head-only training "
            "to layer4 + fc fine-tuning while preserving dataset, initialization, "
            "preprocessing, augmentation, seed, and evaluation protocol."
        ),
        "dataset_contract": datasets.to_contract_dict(),
        "preflight_report": str(preflight_path),
        "weight_cache_probe": weight_probe.to_dict(),
        "preprocessing_contract": phase6b_preprocessing_contract_dict(),
        "dataloader_policy": policy.to_dict(),
        "training_config": config.to_dict(),
        "model_identity": model.identity_dict(),
        "parameter_counts": transfer_parameter_summary(model)["counts"],
        "optimizer_scope": optimizer_scope,
        "mechanics_smoke": mechanics,
        "augmentation": "none",
        "fine_tuning": True,
        "fine_tuning_mode": PHASE6C_FINETUNE_MODE,
        "partial_unfreezing": "layer4_plus_fc_only",
        "differential_learning_rate_groups": False,
        "seed_sweep": False,
        "hyperparameter_search": False,
        "checkpoint_selection_metric": "val_loss",
        "official_test_evaluation": {
            "performed_in_phase6c1": False,
            "allowed_only_after_separate_phase6c2_material_approval": True,
            "must_occur_after_best_checkpoint_restore": True,
            "count_for_future_material_run": 1,
        },
        "non_claims": [
            "no material fine-tuning result",
            "no official test result",
            "no calibration evidence",
            "no robustness or OOD evidence",
            "no seed variance evidence",
            "no architecture-only superiority claim",
        ],
    }
    timing_path = None
    if run_timing_probe:
        timing_path = artifact_dir / "timing_probe.json"
        timing = run_phase6c1_timing_probe(
            batch_sizes=(batch_size,),
            checkpoint_path=checkpoint_path,
            root=root,
            max_batches=2,
            output_path=timing_path,
        )
        run_contract["timing_probe"] = timing.to_dict()
    contract_path = run_path / "run_contract.json"
    contract_path.write_text(json.dumps(run_contract, indent=2), encoding="utf-8")

    return Phase6CPreparedRun(
        run_dir=run_path,
        artifact_dir=artifact_dir,
        datasets=datasets,
        preflight_report=preflight_report,
        preflight_path=preflight_path,
        run_contract=run_contract,
        run_contract_path=contract_path,
        mechanics_path=mechanics_path,
        timing_path=timing_path,
        data_loader_policy=policy,
        training_config=config,
        model_config=phase6c_model_config(),
        initialization_identity=initialization,
    )


def run_phase6c1_timing_probe(
    *,
    batch_sizes: tuple[int, ...] = (64,),
    checkpoint_path: Path | str = PHASE6C_INITIALIZATION_CHECKPOINT,
    root: str | Path = "data",
    max_batches: int = 2,
    output_path: Path | None = None,
) -> Phase6CTimingProbeResult:
    """Estimate CPU fine-tuning practicality from a few train batches only."""

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    datasets = build_cifar10_split_datasets(root=root, download=False)
    preflight_report = verify_material_cifar10_contract(datasets)
    if preflight_report["status"] != "passed":
        raise ValueError("material CIFAR-10 preflight did not pass")

    candidate_results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        policy = DataLoaderPolicy(
            batch_size=batch_size,
            seed=PHASE6C1_SEED,
            num_workers=0,
            train_shuffle=True,
            eval_shuffle=False,
            drop_last=False,
        )
        loaders = build_transfer_dataloaders(datasets, policy)
        model, initialization = build_phase6c_finetune_model_from_phase6b2(checkpoint_path)
        optimizer = build_phase6c_optimizer(model)
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
                raise FloatingPointError("non-finite Phase 6C timing-probe loss")
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
                "estimated_three_epoch_seconds": estimated_epoch_seconds * PHASE6C_EPOCH_BUDGET,
                "initialization_checkpoint_sha256": initialization.checkpoint_sha256,
            }
        )
    selected = min(
        candidate_results,
        key=lambda result: float(result["estimated_three_epoch_seconds"]),
    )
    practical = float(selected["estimated_three_epoch_seconds"]) <= 4 * 60 * 60
    result = Phase6CTimingProbeResult(
        selected_batch_size=int(selected["batch_size"]),
        selected_device="cpu",
        candidate_results=tuple(candidate_results),
        recommendation_reason=(
            "Selected the candidate with the lowest estimated 3-epoch CPU runtime "
            "from identical few-batch Phase 6C fine-tuning timing probes."
        ),
        practical=practical,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def run_phase6c2_material_finetune(
    run_dir: Path | str,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> Phase6CResult:
    """Run the approved Phase 6C-2 material fine-tuning path."""

    prepared = prepare_phase6c1_finetuning_preflight(
        run_dir,
        batch_size=batch_size,
        device=device,
        run_timing_probe=False,
    )
    mark_phase6c2_material_contract_started(prepared.run_contract_path)
    loaders = build_transfer_dataloaders(prepared.datasets, prepared.data_loader_policy)
    model, initialization = build_phase6c_finetune_model_from_phase6b2(
        prepared.initialization_identity.checkpoint_path
    )
    optimizer = build_phase6c_optimizer(model)
    verify_phase6c_optimizer_scope(model, optimizer)
    frozen_before = _snapshot_parameters(model)
    result = fit(
        model,
        loaders.train,
        config=prepared.training_config,
        val_loader=loaders.val,
        optimizer=optimizer,
        run_dir=prepared.run_dir,
    )
    _verify_frozen_parameters_unchanged(model, frozen_before)

    preprocessing_path = prepared.artifact_dir / "preprocessing_contract.json"
    preprocessing_path.write_text(
        json.dumps(phase6b_preprocessing_contract_dict(), indent=2),
        encoding="utf-8",
    )
    artifacts = {
        "preflight_report": str(prepared.preflight_path),
        "preprocessing_contract": str(preprocessing_path),
        "run_contract": str(prepared.run_contract_path),
        "mechanics_smoke": str(prepared.mechanics_path),
    }
    artifacts.update(write_history_artifacts(result.metadata, prepared.artifact_dir))

    if result.status != "completed":
        failure_report = _write_failure_report(prepared.run_dir, result.metadata.to_dict())
        artifacts["failure_report"] = str(failure_report)
        return _write_phase6c_result(prepared.run_dir, result, artifacts)

    artifacts.update(
        write_selected_phase6c_checkpoint_evaluation_artifacts(
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
    finalize_phase6c2_material_contract(
        prepared.run_contract_path,
        result=result,
        artifact_paths=artifacts,
    )
    comparison_report = write_phase6c_comparison_report(
        prepared.run_dir,
        run_id=prepared.training_config.run_id,
        best_epoch=result.best_epoch,
        artifact_paths=artifacts,
        initialization=initialization,
    )
    artifacts["comparison_report"] = str(comparison_report)
    return _write_phase6c_result(prepared.run_dir, result, artifacts)


def mark_phase6c2_material_contract_started(run_contract_path: Path | str) -> dict[str, Any]:
    """Promote a preflight contract to the approved Phase 6C-2 material-run contract."""

    path = Path(run_contract_path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["phase"] = "6C-2"
    contract["scope"] = (
        "approved material layer4 + fc fine-tuning run initialized from the "
        "accepted Phase 6B-2 best checkpoint"
    )
    contract["material_fine_tuning"] = True
    contract["official_test_evaluation"] = {
        "performed": False,
        "count": 0,
        "performed_after_best_checkpoint_restore": None,
        "selection_metric": "val_loss",
        "selected_checkpoint_tag": None,
        "selected_checkpoint_epoch": None,
        "split": "official CIFAR-10 test partition",
        "status": "planned_after_best_checkpoint_restore",
    }
    contract["epoch_budget"] = contract.get("training_config", {}).get(
        "max_epochs",
        PHASE6C_EPOCH_BUDGET,
    )
    contract["batch_size"] = contract.get("dataloader_policy", {}).get("batch_size", 64)
    contract["device"] = contract.get("training_config", {}).get("device", "cpu")
    contract["seed"] = contract.get("training_config", {}).get("seed", PHASE6C1_SEED)
    contract["optimizer"] = contract.get("training_config", {}).get("optimizer", {})
    contract["scheduler"] = contract.get("training_config", {}).get("scheduler")
    contract["metadata_correction"] = {
        "status": "corrected_in_material_runner",
        "issue": (
            "Phase 6C-2 uses Phase 6C-1 preflight artifacts as input evidence, "
            "but the material run contract must identify the approved material phase."
        ),
        "correction": (
            "Promoted top-level contract phase/scope before material training and "
            "finalized official-test/result fields after restored-best evaluation."
        ),
        "training_rerun": False,
        "additional_test_evaluation": False,
        "experimental_evidence_changed": False,
    }
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def finalize_phase6c2_material_contract(
    run_contract_path: Path | str,
    *,
    result: Any,
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    """Write final Phase 6C-2 material result fields into the run contract."""

    path = Path(run_contract_path)
    contract = json.loads(path.read_text(encoding="utf-8"))
    val_summary = json.loads(Path(artifact_paths["val_summary"]).read_text(encoding="utf-8"))
    test_summary = json.loads(Path(artifact_paths["test_summary"]).read_text(encoding="utf-8"))
    history = json.loads(Path(artifact_paths["history"]).read_text(encoding="utf-8"))
    phase6b2_accuracy = contract["phase6b2_reference"]["official_test_accuracy"]
    contract["phase"] = "6C-2"
    contract["material_fine_tuning"] = True
    contract["official_test_evaluation"] = {
        "performed": True,
        "count": 1,
        "performed_after_best_checkpoint_restore": True,
        "selection_metric": "val_loss",
        "selected_checkpoint_tag": "best",
        "selected_checkpoint_epoch": result.best_epoch,
        "split": "official CIFAR-10 test partition",
    }
    contract["material_run_result"] = {
        "status": result.status,
        "best_checkpoint_epoch": result.best_epoch,
        "checkpoint_selection": "minimum validation loss",
        "validation_loss": val_summary["loss"],
        "validation_accuracy": val_summary["accuracy"],
        "official_test_loss": test_summary["loss"],
        "official_test_accuracy": test_summary["accuracy"],
        "phase6b2_reference_test_accuracy": phase6b2_accuracy,
        "test_accuracy_delta_vs_phase6b2": test_summary["accuracy"] - phase6b2_accuracy,
        "single_run_result": True,
    }
    contract["best_checkpoint"] = {
        "tag": "best",
        "epoch": result.best_epoch,
        "path": artifact_paths["selected_checkpoint"],
        "selection_metric": "val_loss",
    }
    contract["terminal_checkpoint"] = {
        "tag": "terminal",
        "epoch": history[-1]["epoch"] if history else None,
        "path": str(Path(path).parent / "checkpoints" / "terminal.pt"),
    }
    contract["non_claims"] = [
        "single-run result only; no seed or run-to-run variance estimate",
        "no optimal unfreezing-depth claim",
        "no optimal hyperparameter claim",
        "no architecture-only superiority claim",
        "no calibration evidence",
        "no robustness or OOD evidence",
        "no generalization claim beyond the evaluated CIFAR-10 experiment",
    ]
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def write_selected_phase6c_checkpoint_evaluation_artifacts(
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
    """Restore a selected Phase 6C fine-tuned checkpoint before evaluation."""

    restored_model, _ = build_phase6c_finetune_model_from_phase6b2()
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        expected_run_id=run_id,
    )
    if checkpoint.get("tag") != "best":
        raise ValueError("selected Phase 6C checkpoint for evaluation must have tag 'best'")

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


def write_phase6c_comparison_report(
    run_dir: Path,
    *,
    run_id: str,
    best_epoch: int | None,
    artifact_paths: dict[str, str],
    initialization: Phase6CInitializationIdentity,
) -> Path:
    val_summary = json.loads(Path(artifact_paths["val_summary"]).read_text(encoding="utf-8"))
    test_summary = json.loads(Path(artifact_paths["test_summary"]).read_text(encoding="utf-8"))
    history = json.loads(Path(artifact_paths["history"]).read_text(encoding="utf-8"))
    phase6b2_loss_delta = test_summary["loss"] - 0.413686
    phase6b2_accuracy_delta = test_summary["accuracy"] - 0.856100
    phase4b_loss_delta = test_summary["loss"] - PHASE6B2_BASELINE_REFERENCE["official_test_loss"]
    phase4b_accuracy_delta = (
        test_summary["accuracy"] - PHASE6B2_BASELINE_REFERENCE["official_test_accuracy"]
    )
    report_path = run_dir / "phase6c_comparison_report.md"
    lines = [
        "# Phase 6C-2 Single-Run Layer4 Fine-Tuning Report",
        "",
        "This report records one approved material fine-tuning run. It compares against the fixed Phase 6B-2 frozen-feature reference and does not represent tuning, a seed sweep, calibration, robustness/OOD evidence, or architecture-only attribution.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Initialization source: `{initialization.source_run_id}`",
        f"- Initialization checkpoint: `{initialization.checkpoint_tag}`, epoch `{initialization.checkpoint_epoch}`",
        f"- Initialization checkpoint SHA-256: `{initialization.checkpoint_sha256}`",
        "- Fine-tuning mode: `finetune_layer4_head`",
        "- Trainable parameters: `layer4 + fc`",
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
        "## Fixed Phase 6B-2 Reference",
        "",
        "- Official test loss: `0.413686`",
        "- Official test accuracy: `0.856100`",
        f"- Official test loss delta: `{phase6b2_loss_delta:.6f}`",
        f"- Official test accuracy delta: `{phase6b2_accuracy_delta:.6f}`",
        "",
        "## Phase 4B Historical Reference",
        "",
        f"- Official test loss: `{PHASE6B2_BASELINE_REFERENCE['official_test_loss']:.6f}`",
        f"- Official test accuracy: `{PHASE6B2_BASELINE_REFERENCE['official_test_accuracy']:.6f}`",
        f"- Official test loss delta: `{phase4b_loss_delta:.6f}`",
        f"- Official test accuracy delta: `{phase4b_accuracy_delta:.6f}`",
        "",
        "## Interpretation Boundaries",
        "",
        "- This is one material fine-tuning run, not a tuned best result.",
        "- The Phase 6B-2 result remains a fixed reference point, not a training target.",
        "- No augmentation, seed sweep, hyperparameter search, differential learning-rate groups, multiple unfreeze strategies, calibration, robustness/OOD, diagnostics, inference, applied-domain work, or Phase 7 work were performed.",
        "- Differences versus Phase 4B remain asymmetric because of ImageNet pretraining, model scale, input resolution, preprocessing, and training regime.",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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



def _verify_frozen_parameters_unchanged(
    model: nn.Module,
    before: dict[str, torch.Tensor],
) -> None:
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and not torch.equal(before[name], parameter.detach())
    ]
    if changed:
        raise ValueError(f"Phase 6C frozen parameters changed: {changed}")


def _write_failure_report(run_dir: Path, metadata: dict[str, Any]) -> Path:
    path = run_dir / "failure_report.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _write_phase6c_result(
    run_dir: Path,
    result: Any,
    artifact_paths: dict[str, str],
) -> Phase6CResult:
    phase_result = Phase6CResult(
        run_dir=run_dir,
        run_id=PHASE6C_RUN_ID,
        status=result.status,
        best_epoch=result.best_epoch,
        artifact_paths=artifact_paths,
    )
    result_path = run_dir / "phase6c_result.json"
    phase_result.artifact_paths["phase6c_result"] = str(result_path)
    result_path.write_text(json.dumps(phase_result.to_dict(), indent=2), encoding="utf-8")
    return phase_result
