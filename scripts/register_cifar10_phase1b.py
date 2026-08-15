"""Register CIFAR-10 for Phase 1B and generate inspection artifacts.

This script is an optional live Phase 1B step. It downloads CIFAR-10 through
torchvision if needed, builds a deterministic train/val/test assignment, writes
manifest and class-count summaries to ignored outputs, and creates raw plus
preprocessed sample grids for builder visual review.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from visionlab.data.splits import stratified_validation_indices


CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

VALIDATION_PER_CLASS = 500
SPLIT_SEED = 20260814
NORMALIZATION_MEAN = (0.5, 0.5, 0.5)
NORMALIZATION_STD = (0.5, 0.5, 0.5)


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    torchvision = _import_torchvision()
    train_source = torchvision.datasets.CIFAR10(
        root=str(data_root), train=True, download=True
    )
    test_source = torchvision.datasets.CIFAR10(
        root=str(data_root), train=False, download=True
    )

    if tuple(train_source.classes) != CLASSES:
        raise RuntimeError(f"unexpected CIFAR-10 classes: {train_source.classes!r}")
    if tuple(test_source.classes) != CLASSES:
        raise RuntimeError(f"unexpected CIFAR-10 classes: {test_source.classes!r}")

    val_indices = stratified_validation_indices(
        train_source.targets,
        validation_per_class=VALIDATION_PER_CLASS,
        seed=SPLIT_SEED,
    )

    records = build_records(train_source.targets, "train", val_indices)
    records.extend(build_records(test_source.targets, "test", set()))
    class_counts = count_classes(records)
    manifest_summary = build_manifest_summary(records, class_counts)

    write_json(output_root / "cifar10_phase1b_manifest_summary.json", manifest_summary)
    write_json(output_root / "cifar10_phase1b_class_counts.json", class_counts)
    write_jsonl(output_root / "cifar10_phase1b_manifest_records.jsonl", records)
    write_grids(train_source, test_source, val_indices, output_root)

    print(json.dumps(manifest_summary["summary"], indent=2, sort_keys=True))
    print(f"Wrote Phase 1B outputs to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/phase1b_cifar10_inspection"),
    )
    return parser.parse_args()


def build_records(
    labels: list[int], upstream_partition: str, val_indices: set[int]
) -> list[dict[str, Any]]:
    records = []
    for upstream_index, label_idx in enumerate(labels):
        visionlab_split = "val" if upstream_index in val_indices else upstream_partition
        label_name = CLASSES[int(label_idx)]
        records.append(
            {
                "sample_id": f"cifar10-{upstream_partition}-{upstream_index:05d}",
                "split": visionlab_split,
                "label": label_name,
                "label_index": int(label_idx),
                "upstream_partition": upstream_partition,
                "upstream_index": upstream_index,
                "source_id": f"cifar10:{upstream_partition}:{upstream_index:05d}",
                "relative_path": f"torchvision/CIFAR10/{upstream_partition}/{upstream_index:05d}",
                "group_id": "",
            }
        )
    return records


def count_classes(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {
        "train": {class_name: 0 for class_name in CLASSES},
        "val": {class_name: 0 for class_name in CLASSES},
        "test": {class_name: 0 for class_name in CLASSES},
    }
    for record in records:
        counts[record["split"]][record["label"]] += 1
    return counts


def build_manifest_summary(
    records: list[dict[str, Any]], class_counts: dict[str, dict[str, int]]
) -> dict[str, Any]:
    split_counts = {split: sum(counts.values()) for split, counts in class_counts.items()}
    return {
        "identity": {
            "dataset_id": "cifar10",
            "version": "python archive c58f30108f718f92721af3b95e74349a",
            "source": "University of Toronto CIFAR-10 python archive via torchvision.datasets.CIFAR10",
            "license_or_usage": (
                "Original CIFAR page requests citation but does not state a license; "
                "UCI repository metadata currently lists CC BY 4.0."
            ),
        },
        "classes": list(CLASSES),
        "split_policy": {
            "upstream_train_count": 50000,
            "upstream_test_count": 10000,
            "visionlab_train_count": split_counts["train"],
            "visionlab_val_count": split_counts["val"],
            "visionlab_test_count": split_counts["test"],
            "validation_per_class": VALIDATION_PER_CLASS,
            "validation_total": split_counts["val"],
            "stratified": True,
            "seed": SPLIT_SEED,
            "stable_sample_id_rule": (
                "cifar10-{upstream_partition}-{upstream_index:05d}; split is stored "
                "separately and does not affect sample identity"
            ),
        },
        "preprocessing": {
            "image_size": [32, 32],
            "color_mode": "RGB",
            "value_range": [0.0, 1.0],
            "normalization_mean": list(NORMALIZATION_MEAN),
            "normalization_std": list(NORMALIZATION_STD),
            "normalization_rationale": (
                "Phase 1B inspection profile uses channel-wise 0.5 mean/std to map "
                "[0, 1] inputs to roughly [-1, 1], matching the PyTorch CIFAR-10 "
                "tutorial convention. It is not computed from test-set statistics."
            ),
            "augmentation": "none",
            "deterministic": True,
        },
        "summary": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": len(records),
            "split_counts": split_counts,
            "class_count_min_max_by_split": {
                split: [min(counts.values()), max(counts.values())]
                for split, counts in class_counts.items()
            },
        },
        "sample_records_preview": records[:5],
    }


def write_grids(train_source: Any, test_source: Any, val_indices: set[int], output_root: Path) -> None:
    train_indices_by_label = first_indices_by_label(train_source.targets)
    val_indices_by_label = first_indices_by_label(train_source.targets, allowed=val_indices)
    test_indices_by_label = first_indices_by_label(test_source.targets)

    grid_specs = (
        ("raw_train_grid.png", train_source, train_indices_by_label, False),
        ("raw_val_grid.png", train_source, val_indices_by_label, False),
        ("raw_test_grid.png", test_source, test_indices_by_label, False),
        ("preprocessed_train_grid.png", train_source, train_indices_by_label, True),
        ("preprocessed_val_grid.png", train_source, val_indices_by_label, True),
        ("preprocessed_test_grid.png", test_source, test_indices_by_label, True),
    )
    for filename, source, indices_by_label, preprocessed in grid_specs:
        grid = make_grid(source, indices_by_label, preprocessed=preprocessed)
        grid.save(output_root / filename)


def first_indices_by_label(
    labels: list[int], allowed: set[int] | None = None, count: int = 8
) -> dict[int, list[int]]:
    found: dict[int, list[int]] = {idx: [] for idx in range(len(CLASSES))}
    allowed_indices = allowed if allowed is not None else set(range(len(labels)))
    for index, label in enumerate(labels):
        if index not in allowed_indices:
            continue
        bucket = found[int(label)]
        if len(bucket) < count:
            bucket.append(index)
        if all(len(values) == count for values in found.values()):
            break
    return found


def make_grid(source: Any, indices_by_label: dict[int, list[int]], preprocessed: bool) -> Any:
    from PIL import Image, ImageDraw

    cell_size = 48
    label_width = 96
    rows = len(CLASSES)
    cols = max(len(indices) for indices in indices_by_label.values())
    canvas = Image.new("RGB", (label_width + cols * cell_size, rows * cell_size), "white")
    draw = ImageDraw.Draw(canvas)
    for label_idx, class_name in enumerate(CLASSES):
        y = label_idx * cell_size
        draw.text((4, y + 16), class_name, fill=(0, 0, 0))
        for col, source_index in enumerate(indices_by_label[label_idx]):
            image, _ = source[source_index]
            if preprocessed:
                image = apply_inspection_preprocessing(image)
            image = image.resize((32, 32))
            x = label_width + col * cell_size + 8
            canvas.paste(image, (x, y + 8))
    return canvas


def apply_inspection_preprocessing(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = np.asarray(image).astype("float32") / 255.0
    mean = np.asarray(NORMALIZATION_MEAN, dtype="float32").reshape(1, 1, 3)
    std = np.asarray(NORMALIZATION_STD, dtype="float32").reshape(1, 1, 3)
    normalized = (array - mean) / std
    display = ((normalized + 1.0) / 2.0).clip(0.0, 1.0)
    return Image.fromarray((display * 255.0).astype("uint8"), mode="RGB")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = (json.dumps(record, sort_keys=True) for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _import_torchvision() -> Any:
    try:
        import torchvision
    except ImportError as exc:
        raise SystemExit(
            "torchvision is required for Phase 1B CIFAR-10 registration. "
            "Use the local .venv verified in T1 or install torchvision first."
        ) from exc
    return torchvision


if __name__ == "__main__":
    main()
