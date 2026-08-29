"""Gallery artifact writers for deterministic failure selections."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image, ImageDraw


ImageResolver = Callable[[str], Image.Image]


def write_gallery_manifest(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write a deterministic CSV-like manifest for gallery rows."""

    from visionlab.evaluation.failures import write_csv_rows

    return write_csv_rows(rows, path)


def write_gallery_html(
    rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    title: str,
    image_key: str = "image_path",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;color:#111827}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}",
        ".item{border:1px solid #d1d5db;padding:8px;border-radius:6px}",
        "img{width:100%;image-rendering:pixelated;border:1px solid #e5e7eb}",
        ".meta{font-size:12px;line-height:1.35;margin-top:6px;word-break:break-word}",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        '<div class="grid">',
    ]
    for row in rows:
        image_path = str(row.get(image_key, ""))
        label = f"true {row.get('true_label', '')}; predicted {row.get('predicted_label', '')}"
        lines.extend(
            [
                '<div class="item">',
                f'<img src="{html.escape(image_path)}" alt="{html.escape(label)}">',
                '<div class="meta">',
                f"<div><strong>{html.escape(str(row.get('sample_id', '')))}</strong></div>",
                f"<div>{html.escape(label)}</div>",
                f"<div>confidence {float(row.get('confidence', 0.0)):.6f}</div>",
                f"<div>run {html.escape(str(row.get('run_id', '')))}</div>",
                "</div>",
                "</div>",
            ]
        )
    lines.extend(["</div>", "</body>", "</html>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def materialize_gallery_images(
    rows: Sequence[dict[str, Any]],
    image_dir: Path,
    *,
    resolve_image: ImageResolver,
) -> tuple[dict[str, Any], ...]:
    """Save one source image per unique selected sample and return rows with paths."""

    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths: dict[str, Path] = {}
    updated: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in image_paths:
            image = resolve_image(sample_id).convert("RGB")
            path = image_dir / f"{_safe_name(sample_id)}.png"
            image.save(path)
            image_paths[sample_id] = path
        updated_row = dict(row)
        updated_row["image_path"] = str(image_paths[sample_id].as_posix())
        updated.append(updated_row)
    return tuple(updated)


def write_placeholder_gallery_images(rows: Sequence[dict[str, Any]], image_dir: Path) -> tuple[dict[str, Any], ...]:
    """Create deterministic placeholder images for tests and missing visual contexts."""

    image_dir.mkdir(parents=True, exist_ok=True)
    updated: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        path = image_dir / f"{_safe_name(sample_id)}.png"
        image = Image.new("RGB", (96, 96), color=(245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), sample_id[:10], fill=(17, 24, 39))
        image.save(path)
        updated_row = dict(row)
        updated_row["image_path"] = str(path.as_posix())
        updated.append(updated_row)
    return tuple(updated)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
