"""Generate Phase 8A degradation visual QA artifacts without model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from visionlab.data.cifar10 import CIFAR10_CLASSES, build_cifar10_split_datasets
from visionlab.data.degradations import (
    PHASE8A_DEFAULT_SEED,
    PHASE8A_DEGRADATION_PROFILES,
    apply_degradation,
    degradation_registry_dict,
)


PHASE8A_RUN_ID = "phase8a-degradation-registry-visual-qa-tiny-smoke"
PHASE8A_OUTPUT_DIR = Path("outputs") / PHASE8A_RUN_ID
PHASE8A_VISUAL_SAMPLE_IDS = (
    "cifar10-test-00000",
    "cifar10-test-00001",
    "cifar10-test-00002",
    "cifar10-test-00003",
    "cifar10-test-00004",
)


def generate_phase8a_visual_qa(
    output_dir: Path | str = PHASE8A_OUTPUT_DIR,
    *,
    data_root: Path | str = "data",
    seed: int = PHASE8A_DEFAULT_SEED,
) -> dict[str, str]:
    """Write registry, fixed-sample manifest, PNG grid, and inspection note."""

    output_path = Path(output_dir)
    artifact_dir = output_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    registry = degradation_registry_dict()
    registry_path = output_path / "phase8a_degradation_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    datasets = build_cifar10_split_datasets(root=data_root, download=False)
    selected_samples = _select_samples_by_id(datasets.test, PHASE8A_VISUAL_SAMPLE_IDS)
    sample_manifest = [
        {
            "sample_id": sample["sample_id"],
            "source_id": sample["source_id"],
            "split": sample["split"],
            "label_index": int(sample["label"]),
            "label": CIFAR10_CLASSES[int(sample["label"])],
        }
        for sample in selected_samples
    ]
    manifest_path = artifact_dir / "phase8a_visual_sample_manifest.json"
    manifest_path.write_text(json.dumps(sample_manifest, indent=2), encoding="utf-8")

    grid_paths: dict[str, str] = {}
    for profile in PHASE8A_DEGRADATION_PROFILES:
        grid_path = artifact_dir / f"{profile.profile_id}_visual_grid.png"
        _write_profile_grid(selected_samples, profile.profile_id, grid_path, seed=seed)
        grid_paths[profile.profile_id] = str(grid_path)

    note_path = output_path / "phase8a_visual_inspection_note.md"
    _write_inspection_note(
        note_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        grid_paths=grid_paths,
        sample_manifest=sample_manifest,
        seed=seed,
    )
    result = {
        "registry": str(registry_path),
        "sample_manifest": str(manifest_path),
        "inspection_note": str(note_path),
        **{f"{profile_id}_grid": path for profile_id, path in grid_paths.items()},
    }
    result_path = output_path / "phase8a_visual_qa_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["result"] = str(result_path)
    return result


def _select_samples_by_id(dataset: Any, sample_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    wanted = set(sample_ids)
    samples: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = str(sample["sample_id"])
        if sample_id in wanted:
            samples[sample_id] = sample
        if len(samples) == len(wanted):
            break
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples]
    if missing:
        raise ValueError(f"missing registered CIFAR-10 samples for visual QA: {missing}")
    return [samples[sample_id] for sample_id in sample_ids]


def _write_profile_grid(
    samples: list[dict[str, Any]],
    profile_id: str,
    path: Path,
    *,
    seed: int,
) -> Path:
    from PIL import Image, ImageDraw

    cell = 64
    label_height = 18
    columns = 6
    rows = len(samples)
    image = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "white")
    draw = ImageDraw.Draw(image)
    headers = ["clean", "S1", "S2", "S3", "S4", "S5"]
    for col, header in enumerate(headers):
        draw.text((col * cell + 4, 2), header, fill=(0, 0, 0))
    for row, sample in enumerate(samples):
        y = row * (cell + label_height) + label_height
        tensors = [sample["raw_input"]]
        for severity_id in headers[1:]:
            tensors.append(
                apply_degradation(
                    sample["raw_input"],
                    profile_id=profile_id,
                    severity_id=severity_id,
                    seed=seed,
                    sample_id=str(sample["sample_id"]),
                    source_id=str(sample["source_id"]),
                )
            )
        for col, tensor in enumerate(tensors):
            tile = _tensor_to_image(tensor).resize((cell, cell), Image.Resampling.NEAREST)
            image.paste(tile, (col * cell, y))
        draw.text((4, y + cell - 12), str(sample["sample_id"]), fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _tensor_to_image(tensor: torch.Tensor):
    from PIL import Image

    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _write_inspection_note(
    path: Path,
    *,
    registry_path: Path,
    manifest_path: Path,
    grid_paths: dict[str, str],
    sample_manifest: list[dict[str, Any]],
    seed: int,
) -> Path:
    registry = degradation_registry_dict()
    lines = [
        "# Phase 8A Visual Inspection Note",
        "",
        "Status: generated for builder visual review; Phase 8A is not closed.",
        "",
        "## Scope",
        "",
        "These artifacts inspect degradation mechanics and visual plausibility only. They do not evaluate any model checkpoint and must not be interpreted as evidence of robustness.",
        "",
        "Do not interpret visual QA as evidence that a degradation is semantically label-preserving; record it only as qualitative inspection of plausibility, and defer any robustness conclusion to Phase 8B.",
        "",
        "## Seed Policy",
        "",
        f"- Base seed: `{seed}`",
        "- Deterministic transforms ignore seed.",
        "- Gaussian noise derives an order-independent effective seed from profile/version/severity/base seed/sample_id/source_id.",
        "",
        "## Fixed Visual Samples",
        "",
    ]
    for sample in sample_manifest:
        lines.append(
            f"- `{sample['sample_id']}`: label `{sample['label']}`, source `{sample['source_id']}`"
        )
    lines.extend(["", "## Profile Parameters", ""])
    for profile in registry["profiles"]:
        lines.append(
            f"### `{profile['profile_id']}` version `{profile['version']}`"
        )
        lines.append("")
        lines.append(f"- Type: `{profile['degradation_type']}`")
        lines.append(f"- Deterministic: `{profile['deterministic']}`")
        lines.append(f"- Rationale: {profile['rationale']}")
        for severity in profile["severity_levels"]:
            lines.append(
                f"- `{severity['severity_id']}`: `{json.dumps(severity['parameters'], sort_keys=True)}`"
            )
        lines.append("")
    lines.extend(["## Artifact Paths", ""])
    lines.append(f"- Registry: `{registry_path}`")
    lines.append(f"- Fixed sample manifest: `{manifest_path}`")
    for profile_id, grid_path in grid_paths.items():
        lines.append(f"- `{profile_id}` grid: `{grid_path}`")
    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- Pending builder visual review.",
            "- Record only what is visibly present in the generated grids.",
            "- Do not treat visual plausibility as proof of label preservation or robustness.",
            "",
            "## Approval Recommendation",
            "",
            "- Pending builder review.",
            "- If accepted, freeze these profile/version/severity parameters for Phase 8B unless a later requirement change is approved.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    paths = generate_phase8a_visual_qa()
    print(json.dumps(paths, indent=2))
