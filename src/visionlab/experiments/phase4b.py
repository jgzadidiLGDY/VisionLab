"""Phase 4B approved custom-CNN CIFAR-10 material baseline run."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    DataLoaderPolicy,
    build_cifar10_split_datasets,
    build_phase4_dataloaders,
    verify_material_cifar10_contract,
)
from visionlab.evaluation import write_history_artifacts
from visionlab.experiments.phase4a import write_selected_checkpoint_evaluation_artifacts
from visionlab.models import CustomCNN, CustomCNNConfig
from visionlab.training import OptimizerConfig, TrainingConfig, fit


PHASE4B_RUN_ID = "phase4b-cifar10-custom-cnn-baseline-001"
PHASE4B_SEED = 20260818


@dataclass(frozen=True)
class Phase4BResult:
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


def run_phase4b_material_baseline(run_dir: Path | str) -> Phase4BResult:
    """Run the approved single material custom-CNN CIFAR-10 baseline."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_cifar10_split_datasets(root="data", download=False)
    preflight_report = verify_material_cifar10_contract(datasets)
    preflight_path = run_path / "preflight_report.json"
    preflight_path.write_text(json.dumps(preflight_report, indent=2), encoding="utf-8")

    policy = DataLoaderPolicy(
        batch_size=128,
        seed=PHASE4B_SEED,
        num_workers=0,
        train_shuffle=True,
        eval_shuffle=False,
        drop_last=False,
    )
    loaders = build_phase4_dataloaders(datasets, policy)
    config = TrainingConfig(
        run_id=PHASE4B_RUN_ID,
        seed=PHASE4B_SEED,
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
    run_contract = {
        "phase": "4B",
        "scope": "single-run custom CNN material baseline; not tuned; not variance evidence",
        "run_id": PHASE4B_RUN_ID,
        "dataset_contract": datasets.to_contract_dict(),
        "preflight_report": str(preflight_path),
        "dataloader_policy": policy.to_dict(),
        "training_config": config.to_dict(),
        "model_config": asdict(model_config),
        "augmentation": "none",
        "checkpoint_selection_metric": "val_loss",
        "evaluation_sequence": [
            "train",
            "select best by val_loss",
            "restore best checkpoint",
            "generate final validation artifacts",
            "evaluate test once",
            "generate test artifacts",
        ],
    }
    contract_path = run_path / "run_contract.json"
    contract_path.write_text(json.dumps(run_contract, indent=2), encoding="utf-8")

    model = CustomCNN(model_config)
    result = fit(
        model,
        loaders.train,
        config=config,
        val_loader=loaders.val,
        run_dir=run_path,
    )

    artifacts = {
        "preflight_report": str(preflight_path),
        "run_contract": str(contract_path),
    }
    artifacts.update(write_history_artifacts(result.metadata, artifact_dir))

    if result.status != "completed":
        failure_report = _write_failure_report(run_path, result.metadata.to_dict())
        artifacts["failure_report"] = str(failure_report)
        return _write_result(run_path, result, artifacts)

    artifacts.update(
        write_selected_checkpoint_evaluation_artifacts(
            checkpoint_path=Path(result.metadata.checkpoint_references["best"]),
            model_config=model_config,
            run_id=config.run_id,
            val_loader=loaders.prediction_val,
            test_loader=loaders.prediction_test,
            output_dir=artifact_dir,
            include_test=True,
            val_prefix="val",
            test_prefix="test",
        )
    )
    report_path = write_baseline_report(
        run_path,
        run_id=config.run_id,
        best_epoch=result.best_epoch,
        artifact_paths=artifacts,
    )
    artifacts["baseline_report"] = str(report_path)
    return _write_result(run_path, result, artifacts)


def write_baseline_report(
    run_dir: Path,
    *,
    run_id: str,
    best_epoch: int | None,
    artifact_paths: dict[str, str],
) -> Path:
    val_summary = json.loads(Path(artifact_paths["val_summary"]).read_text(encoding="utf-8"))
    test_summary = json.loads(Path(artifact_paths["test_summary"]).read_text(encoding="utf-8"))
    history = json.loads(Path(artifact_paths["history"]).read_text(encoding="utf-8"))
    high_conf_errors = _top_incorrect_predictions(
        Path(artifact_paths["test_predictions"]),
        limit=10,
    )
    report_path = run_dir / "baseline_report.md"
    lines = [
        "# Phase 4B Single-Run Custom CNN Baseline Report",
        "",
        "This report describes one approved custom CNN baseline run. It is not a tuned best result and is not an estimate of run-to-run variance.",
        "",
        f"- Run ID: `{run_id}`",
        f"- Selected epoch: `{best_epoch}`",
        "- Checkpoint selection: minimum validation loss",
        "- Test use: evaluated once after restoring the selected best checkpoint",
        "- Augmentation: none",
        "",
        "## Results",
        "",
        f"- Final restored-best validation loss: `{val_summary['loss']:.6f}`",
        f"- Final restored-best validation accuracy: `{val_summary['accuracy']:.6f}`",
        f"- Official test loss: `{test_summary['loss']:.6f}`",
        f"- Official test accuracy: `{test_summary['accuracy']:.6f}`",
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
            "## Basic Test Error Observations",
            "",
            "Highest-confidence incorrect test predictions are listed as traceability prompts, not a completed failure analysis:",
        ]
    )
    for row in high_conf_errors:
        lines.append(
            f"- `{row['sample_id']}` true `{row['true_label']}` predicted `{row['predicted_label']}` confidence `{float(row['confidence']):.6f}`"
        )
    lines.extend(
        [
            "",
            "## Preserved Artifacts",
            "",
        ]
    )
    for name in sorted(artifact_paths):
        lines.append(f"- `{name}`: `{artifact_paths[name]}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Single run only; no seed sweep or variance estimate.",
            "- No augmentation, calibration, robustness, OOD, transfer-learning, or diagnostic claims.",
            "- Test results were not used for checkpoint selection.",
            "- Phase 4 remains open until a separate phase-check review is completed.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _top_incorrect_predictions(path: Path, *, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["correct"] == "False"]
    rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return rows[:limit]


def _write_failure_report(run_dir: Path, metadata: dict[str, Any]) -> Path:
    path = run_dir / "failure_report.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def _write_result(
    run_dir: Path,
    result: Any,
    artifact_paths: dict[str, str],
) -> Phase4BResult:
    phase_result = Phase4BResult(
        run_dir=run_dir,
        run_id=PHASE4B_RUN_ID,
        status=result.status,
        best_epoch=result.best_epoch,
        artifact_paths=artifact_paths,
    )
    result_path = run_dir / "phase4b_result.json"
    phase_result.artifact_paths["phase4b_result"] = str(result_path)
    result_path.write_text(json.dumps(phase_result.to_dict(), indent=2), encoding="utf-8")
    return phase_result
