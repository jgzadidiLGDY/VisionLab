"""Phase 5B approved augmentation-comparison material baseline path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visionlab.data import profile_registry_dict
from visionlab.data.augmentation import PHASE5A_CANDIDATE_FLIP_CROP_PROFILE
from visionlab.data.cifar10 import (
    DataLoaderPolicy,
    SplitDatasetBundle,
    build_cifar10_split_datasets,
    build_phase4_dataloaders,
    verify_material_cifar10_contract,
)
from visionlab.evaluation import write_history_artifacts
from visionlab.experiments.phase4a import write_selected_checkpoint_evaluation_artifacts
from visionlab.models import CustomCNN, CustomCNNConfig
from visionlab.training import OptimizerConfig, TrainingConfig, fit


PHASE5B_RUN_ID = "phase5b-cifar10-custom-cnn-augmentation-candidate-001"
PHASE5B_SEED = 20260818
PHASE5B_BASELINE_REFERENCE_RUN_ID = "phase4b-cifar10-custom-cnn-baseline-001"
PHASE5B_BASELINE_REFERENCE = {
    "run_id": PHASE5B_BASELINE_REFERENCE_RUN_ID,
    "official_test_loss": 1.024515,
    "official_test_accuracy": 0.635900,
}


@dataclass(frozen=True)
class Phase5BPreparedRun:
    run_dir: Path
    artifact_dir: Path
    datasets: SplitDatasetBundle
    preflight_report: dict[str, Any]
    preflight_path: Path
    profile_registry_snapshot_path: Path
    run_contract: dict[str, Any]
    run_contract_path: Path
    data_loader_policy: DataLoaderPolicy
    training_config: TrainingConfig
    model_config: CustomCNNConfig


@dataclass(frozen=True)
class Phase5BResult:
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


def prepare_phase5b_material_run(
    run_dir: Path | str,
    *,
    root: str | Path = "data",
    download: bool = False,
    upstream_train: Any | None = None,
    upstream_test: Any | None = None,
    expected_counts: dict[str, int] | None = None,
) -> Phase5BPreparedRun:
    """Write the exact Phase 5B run contract and preflight artifacts."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_cifar10_split_datasets(
        root=root,
        download=download,
        upstream_train=upstream_train,
        upstream_test=upstream_test,
        train_augmentation_profile=PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
    )
    preflight_report = verify_material_cifar10_contract(
        datasets,
        expected_counts=expected_counts,
    )
    preflight_path = run_path / "preflight_report.json"
    preflight_path.write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")

    profile_registry_snapshot = profile_registry_dict()
    profile_registry_path = artifact_dir / "augmentation_profile_registry.json"
    profile_registry_path.write_text(
        json.dumps(profile_registry_snapshot, indent=2),
        encoding="utf-8",
    )

    policy = DataLoaderPolicy(
        batch_size=128,
        seed=PHASE5B_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )
    config = TrainingConfig(
        run_id=PHASE5B_RUN_ID,
        seed=PHASE5B_SEED,
        max_epochs=10,
        optimizer=OptimizerConfig(name="adam", learning_rate=0.001, weight_decay=0.0),
        scheduler=None,
        selection_metric="val_loss",
    )
    model_config = CustomCNNConfig(
        input_channels=3,
        image_size=(32, 32),
        num_classes=10,
        feature_channels=(32, 64, 128),
        dropout=0.0,
    )
    model_config_dict = asdict(model_config)
    model_config_dict["image_size"] = list(model_config.image_size)
    model_config_dict["feature_channels"] = list(model_config.feature_channels)
    run_contract = {
        "phase": "5B",
        "scope": "single-run custom CNN augmentation comparison; candidate profile only; not tuned; not variance evidence",
        "run_id": PHASE5B_RUN_ID,
        "baseline_reference": dict(PHASE5B_BASELINE_REFERENCE),
        "dataset_contract": datasets.to_contract_dict(),
        "preflight_report": str(preflight_path),
        "profile_registry_snapshot": str(profile_registry_path),
        "dataloader_policy": policy.to_dict(),
        "training_config": config.to_dict(),
        "model_config": model_config_dict,
        "experimental_variable": {
            "name": "train_augmentation_profile",
            "phase4b_control_profile_id": "phase4-control-no-augmentation",
            "phase4b_control_profile_version": "1.0",
            "phase5b_candidate_profile_id": PHASE5A_CANDIDATE_FLIP_CROP_PROFILE.profile_id,
            "phase5b_candidate_profile_version": PHASE5A_CANDIDATE_FLIP_CROP_PROFILE.version,
        },
        "augmentation_profile": PHASE5A_CANDIDATE_FLIP_CROP_PROFILE.to_dict(),
        "validation_test_preprocessing": "deterministic Phase 4 preprocessing only",
        "checkpoint_selection_metric": "val_loss",
        "evaluation_sequence": [
            "train with candidate train-only augmentation",
            "select best by val_loss",
            "restore best checkpoint",
            "generate final validation artifacts",
            "evaluate test once",
            "generate test artifacts",
        ],
        "official_test_evaluation": {
            "enabled": True,
            "occurs_after_best_checkpoint_restore": True,
            "count": 1,
        },
    }
    contract_path = run_path / "run_contract.json"
    contract_path.write_text(json.dumps(run_contract, indent=2), encoding="utf-8")

    return Phase5BPreparedRun(
        run_dir=run_path,
        artifact_dir=artifact_dir,
        datasets=datasets,
        preflight_report=preflight_report,
        preflight_path=preflight_path,
        profile_registry_snapshot_path=profile_registry_path,
        run_contract=run_contract,
        run_contract_path=contract_path,
        data_loader_policy=policy,
        training_config=config,
        model_config=model_config,
    )


def run_phase5b_material_baseline(run_dir: Path | str) -> Phase5BResult:
    """Run the approved single material custom-CNN augmentation comparison."""

    prepared = prepare_phase5b_material_run(run_dir)
    loaders = build_phase4_dataloaders(prepared.datasets, prepared.data_loader_policy)
    model = CustomCNN(prepared.model_config)
    result = fit(
        model,
        loaders.train,
        config=prepared.training_config,
        val_loader=loaders.val,
        run_dir=prepared.run_dir,
    )

    artifacts = {
        "preflight_report": str(prepared.preflight_path),
        "profile_registry_snapshot": str(prepared.profile_registry_snapshot_path),
        "run_contract": str(prepared.run_contract_path),
    }
    artifacts.update(write_history_artifacts(result.metadata, prepared.artifact_dir))

    if result.status != "completed":
        failure_report = _write_failure_report(prepared.run_dir, result.metadata.to_dict())
        artifacts["failure_report"] = str(failure_report)
        return _write_result(prepared.run_dir, result, artifacts)

    artifacts.update(
        write_selected_checkpoint_evaluation_artifacts(
            checkpoint_path=Path(result.metadata.checkpoint_references["best"]),
            model_config=prepared.model_config,
            run_id=prepared.training_config.run_id,
            val_loader=loaders.prediction_val,
            test_loader=loaders.prediction_test,
            output_dir=prepared.artifact_dir,
            include_test=True,
            val_prefix="val",
            test_prefix="test",
        )
    )
    comparison_report = write_phase5b_comparison_report(
        prepared.run_dir,
        run_id=prepared.training_config.run_id,
        best_epoch=result.best_epoch,
        artifact_paths=artifacts,
    )
    artifacts["comparison_report"] = str(comparison_report)
    return _write_result(prepared.run_dir, result, artifacts)


def write_phase5b_comparison_report(
    run_dir: Path,
    *,
    run_id: str,
    best_epoch: int | None,
    artifact_paths: dict[str, str],
) -> Path:
    val_summary = json.loads(Path(artifact_paths["val_summary"]).read_text(encoding="utf-8"))
    test_summary = json.loads(Path(artifact_paths["test_summary"]).read_text(encoding="utf-8"))
    history = json.loads(Path(artifact_paths["history"]).read_text(encoding="utf-8"))
    report_path = run_dir / "phase5b_comparison_report.md"
    accuracy_delta = test_summary["accuracy"] - PHASE5B_BASELINE_REFERENCE["official_test_accuracy"]
    loss_delta = test_summary["loss"] - PHASE5B_BASELINE_REFERENCE["official_test_loss"]
    lines = [
        "# Phase 5B Single-Run Augmentation Comparison Report",
        "",
        "This report compares one approved candidate augmentation run against the accepted Phase 4B no-augmentation baseline. It is not a tuned best result and is not an estimate of run-to-run variance.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Comparison baseline: `{PHASE5B_BASELINE_REFERENCE_RUN_ID}`",
        f"- Selected epoch: `{best_epoch}`",
        f"- Changed variable: `train_augmentation_profile -> {PHASE5A_CANDIDATE_FLIP_CROP_PROFILE.profile_id}` version `{PHASE5A_CANDIDATE_FLIP_CROP_PROFILE.version}`",
        "- Controlled variables: dataset split, model config, optimizer, learning rate, weight decay, scheduler, batch size, epoch budget, seed, DataLoader policy, and checkpoint-selection rule",
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
        f"- Reference run ID: `{PHASE5B_BASELINE_REFERENCE['run_id']}`",
        f"- Official test loss: `{PHASE5B_BASELINE_REFERENCE['official_test_loss']:.6f}`",
        f"- Official test accuracy: `{PHASE5B_BASELINE_REFERENCE['official_test_accuracy']:.6f}`",
        "",
        "## Deltas Versus Phase 4B",
        "",
        f"- Official test loss delta: `{loss_delta:.6f}`",
        f"- Official test accuracy delta: `{accuracy_delta:.6f}`",
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
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- The candidate profile remains a candidate until builder review interprets this run.",
            "- A single-run result may support adoption, rejection, or an inconclusive outcome; no automatic baseline change is implied here.",
            "",
            "## Preserved Artifacts",
            "",
        ]
    )
    for name in sorted(artifact_paths):
        lines.append(f"- `{name}`: `{artifact_paths[name]}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_failure_report(run_dir: Path, metadata: dict[str, Any]) -> Path:
    path = run_dir / "failure_report.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _write_result(
    run_dir: Path,
    result: Any,
    artifact_paths: dict[str, str],
) -> Phase5BResult:
    phase_result = Phase5BResult(
        run_dir=run_dir,
        run_id=PHASE5B_RUN_ID,
        status=result.status,
        best_epoch=result.best_epoch,
        artifact_paths=artifact_paths,
    )
    result_path = run_dir / "phase5b_result.json"
    phase_result.artifact_paths["phase5b_result"] = str(result_path)
    result_path.write_text(json.dumps(phase_result.to_dict(), indent=2), encoding="utf-8")
    return phase_result
