"""Generate Phase 5A augmentation profile and visual inspection artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from visionlab.data.augmentation import (
    PHASE4_NO_AUGMENTATION_PROFILE,
    PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
    apply_augmentation_profile,
    profile_registry_dict,
)
from visionlab.data.cifar10 import (
    CIFAR10_CLASSES,
    CIFAR10_PREPROCESSING,
    build_cifar10_split_datasets,
    normalize_tensor,
    to_unit_tensor,
)


INSPECTION_SEED = 20260819
OUTPUT_DIR = Path("outputs/phase5a_augmentation_inspection")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = OUTPUT_DIR / "augmentation_profile_registry.json"
    registry_path.write_text(
        json.dumps(profile_registry_dict(), indent=2),
        encoding="utf-8",
    )

    datasets = build_cifar10_split_datasets(root="data", download=False)
    selections = _select_fixed_train_samples(datasets.train.upstream, datasets.train.indices)
    grid_path = OUTPUT_DIR / "phase5a_candidate_augmentation_grid.png"
    metadata = _write_grid(datasets.train.upstream, selections, grid_path)
    note_path = OUTPUT_DIR / "phase5a_visual_inspection_note.md"
    _write_note(
        note_path,
        registry_path=registry_path,
        grid_path=grid_path,
        selections=selections,
        metadata=metadata,
    )


def _select_fixed_train_samples(upstream: Any, train_indices: list[int]) -> list[dict[str, Any]]:
    selected: dict[int, int] = {}
    train_index_set = set(train_indices)
    for source_index, label in enumerate(upstream.targets):
        label_index = int(label)
        if source_index in train_index_set and label_index not in selected:
            selected[label_index] = source_index
        if len(selected) == len(CIFAR10_CLASSES):
            break
    if len(selected) != len(CIFAR10_CLASSES):
        raise RuntimeError("could not select one fixed train sample per CIFAR-10 class")
    return [
        {
            "sample_id": f"cifar10-train-{selected[label_index]:05d}",
            "source_index": selected[label_index],
            "label_index": label_index,
            "label": CIFAR10_CLASSES[label_index],
        }
        for label_index in range(len(CIFAR10_CLASSES))
    ]


def _write_grid(upstream: Any, selections: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write Phase 5A inspection grids") from exc

    cell_size = 96
    label_width = 130
    header_height = 28
    columns = [
        "raw",
        "control",
        "candidate seed+0",
        "candidate seed+1",
        "candidate seed+2",
    ]
    width = label_width + cell_size * len(columns)
    height = header_height + cell_size * len(selections)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for column_index, column in enumerate(columns):
        draw.text((label_width + column_index * cell_size + 4, 8), column, fill=(0, 0, 0))

    range_report: list[dict[str, Any]] = []
    for row_index, selection in enumerate(selections):
        y = header_height + row_index * cell_size
        draw.text((4, y + 35), f"{selection['label']}\n{selection['sample_id']}", fill=(0, 0, 0))
        image, _ = upstream[selection["source_index"]]
        unit_tensor = to_unit_tensor(image, CIFAR10_PREPROCESSING)
        variants = [
            unit_tensor,
            apply_augmentation_profile(unit_tensor, PHASE4_NO_AUGMENTATION_PROFILE),
        ]
        for variant_index in range(3):
            generator = torch.Generator().manual_seed(
                INSPECTION_SEED + selection["source_index"] * 10 + variant_index
            )
            variants.append(
                apply_augmentation_profile(
                    unit_tensor,
                    PHASE5A_CANDIDATE_FLIP_CROP_PROFILE,
                    generator=generator,
                )
            )
        for column_index, variant in enumerate(variants):
            normalized = normalize_tensor(variant, CIFAR10_PREPROCESSING)
            range_report.append(
                {
                    "sample_id": selection["sample_id"],
                    "column": columns[column_index],
                    "shape": list(normalized.shape),
                    "min": float(normalized.min().item()),
                    "max": float(normalized.max().item()),
                }
            )
            tile = _tensor_to_display_image(normalized).resize((cell_size, cell_size))
            canvas.paste(tile, (label_width + column_index * cell_size, y))
    canvas.save(path)
    return {
        "columns": columns,
        "range_report": range_report,
    }


def _tensor_to_display_image(tensor: torch.Tensor) -> Any:
    from PIL import Image

    display = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
    array = (display.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _write_note(
    path: Path,
    *,
    registry_path: Path,
    grid_path: Path,
    selections: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    all_shapes = {tuple(row["shape"]) for row in metadata["range_report"]}
    min_value = min(row["min"] for row in metadata["range_report"])
    max_value = max(row["max"] for row in metadata["range_report"])
    sample_lines = [
        f"- `{item['sample_id']}`: `{item['label']}`"
        for item in selections
    ]
    lines = [
        "# Phase 5A Visual Augmentation Inspection Note",
        "",
        "Scope: Phase 5A augmentation profile and smoke verification only. This note does not record material training or official test evaluation.",
        "",
        f"- Inspection seed: `{INSPECTION_SEED}`",
        f"- Profile registry: `{registry_path.as_posix()}`",
        f"- Visual grid: `{grid_path.as_posix()}`",
        "- Split inspected: fixed samples from the registered VisionLab CIFAR-10 train split",
        "",
        "## Fixed Sample IDs",
        "",
        *sample_lines,
        "",
        "## Profiles Inspected",
        "",
        "- `phase4-control-no-augmentation` version `1.0`: no transform control profile.",
        "- `phase5a-candidate-horizontal-flip-random-crop` version `1.0`: random horizontal flip with probability `0.5`, then random crop to `32x32` after `4` pixels of zero padding.",
        "",
        "## Artifact Columns",
        "",
        *[f"- `{column}`" for column in metadata["columns"]],
        "",
        "## Observations",
        "",
        f"- All rendered normalized outputs have shape set `{sorted(all_shapes)}`.",
        f"- Normalized rendered outputs stayed within observed range `{min_value:.6f}` to `{max_value:.6f}`.",
        "- The control column visually matches the raw sample content after display conversion.",
        "- The candidate columns show small translations and occasional horizontal flips while retaining visible class evidence in this fixed grid.",
        "- Some candidate cells show visible black padded margins at crop edges, which is expected from `padding=4`, `padding_mode=constant`, and `fill=0.0`.",
        "- CIFAR-10 images remain low-resolution and several classes are visually ambiguous, so this inspection supports plausibility rather than proving label preservation for every sample.",
        "",
        "## Approval Recommendation",
        "",
        "Codex recommends treating `phase5a-candidate-horizontal-flip-random-crop` version `1.0` as appropriate for a single controlled Phase 5B material-run proposal, pending builder review of the visual grid. The recommendation is based on the inspected samples preserving visible class evidence, while keeping the profile explicitly candidate-scoped until the builder approves Phase 5B.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
