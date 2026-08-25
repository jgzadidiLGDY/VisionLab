"""Dependency-light visual artifact writers for Phase 7."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any, Sequence

from visionlab.evaluation.calibration import CalibrationSummary


def write_reliability_diagram_svg(summary: CalibrationSummary, path: Path, *, title: str) -> Path:
    width = 720
    height = 460
    margin_left = 70
    margin_bottom = 70
    plot_size = 320
    origin_x = margin_left
    origin_y = height - margin_bottom
    bar_width = plot_size / summary.num_bins
    lines = [
        _svg_header(width, height),
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x + plot_size}" y2="{origin_y}" stroke="black"/>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{origin_y - plot_size}" stroke="black"/>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x + plot_size}" y2="{origin_y - plot_size}" stroke="#666" stroke-dasharray="4 4"/>',
        f'<text x="{origin_x + plot_size / 2}" y="{height - 25}" text-anchor="middle" font-size="13">confidence</text>',
        f'<text x="20" y="{origin_y - plot_size / 2}" text-anchor="middle" transform="rotate(-90 20 {origin_y - plot_size / 2})" font-size="13">accuracy</text>',
    ]
    for item in summary.bins:
        x = origin_x + item.index * bar_width + 2
        if item.accuracy is not None:
            bar_height = item.accuracy * plot_size
            y = origin_y - bar_height
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 4:.2f}" height="{bar_height:.2f}" fill="#3b82f6" opacity="0.75"/>'
            )
        if item.average_confidence is not None:
            cx = origin_x + item.average_confidence * plot_size
            cy = origin_y - (item.accuracy or 0.0) * plot_size
            lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" fill="#111827"/>')
    lines.extend(
        [
            f'<text x="430" y="95" font-size="14">ECE: {summary.expected_calibration_error:.6f}</text>',
            f'<text x="430" y="120" font-size="14">MCE: {summary.maximum_calibration_error:.6f}</text>',
            f'<text x="430" y="145" font-size="14">Accuracy: {summary.accuracy:.6f}</text>',
            f'<text x="430" y="170" font-size="14">Avg confidence: {summary.average_confidence:.6f}</text>',
            '<text x="430" y="215" font-size="12">Bars show bin accuracy.</text>',
            '<text x="430" y="235" font-size="12">Dashed diagonal is perfect calibration.</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_confidence_histogram_csv(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bin_index", "lower", "upper", "correct_count", "incorrect_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_confidence_histogram_svg(rows: Sequence[dict[str, Any]], path: Path, *, title: str) -> Path:
    width = 760
    height = 420
    origin_x = 70
    origin_y = 330
    plot_width = 500
    plot_height = 250
    max_count = max(
        [int(row["correct_count"]) + int(row["incorrect_count"]) for row in rows] or [1]
    )
    if max_count == 0:
        max_count = 1
    group_width = plot_width / max(1, len(rows))
    lines = [
        _svg_header(width, height),
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x + plot_width}" y2="{origin_y}" stroke="black"/>',
        f'<line x1="{origin_x}" y1="{origin_y}" x2="{origin_x}" y2="{origin_y - plot_height}" stroke="black"/>',
        f'<text x="{origin_x + plot_width / 2}" y="385" text-anchor="middle" font-size="13">confidence bin</text>',
        f'<text x="20" y="{origin_y - plot_height / 2}" text-anchor="middle" transform="rotate(-90 20 {origin_y - plot_height / 2})" font-size="13">count</text>',
    ]
    for row in rows:
        index = int(row["bin_index"])
        correct = int(row["correct_count"])
        incorrect = int(row["incorrect_count"])
        x = origin_x + index * group_width + 4
        correct_height = correct / max_count * plot_height
        incorrect_height = incorrect / max_count * plot_height
        lines.append(
            f'<rect x="{x:.2f}" y="{origin_y - correct_height:.2f}" width="{(group_width - 8) / 2:.2f}" height="{correct_height:.2f}" fill="#16a34a" opacity="0.8"/>'
        )
        lines.append(
            f'<rect x="{x + (group_width - 8) / 2:.2f}" y="{origin_y - incorrect_height:.2f}" width="{(group_width - 8) / 2:.2f}" height="{incorrect_height:.2f}" fill="#dc2626" opacity="0.8"/>'
        )
    lines.extend(
        [
            '<rect x="610" y="92" width="14" height="14" fill="#16a34a" opacity="0.8"/>',
            '<text x="632" y="104" font-size="13">correct</text>',
            '<rect x="610" y="122" width="14" height="14" fill="#dc2626" opacity="0.8"/>',
            '<text x="632" y="134" font-size="13">incorrect</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_confusion_matrix_svg(
    matrix: Sequence[Sequence[int]],
    class_names: tuple[str, ...],
    path: Path,
    *,
    title: str,
) -> Path:
    cell = 34
    label_width = 110
    top = 72
    width = label_width + cell * len(class_names) + 40
    height = top + cell * len(class_names) + 70
    max_value = max([int(value) for row in matrix for value in row] or [1])
    if max_value == 0:
        max_value = 1
    lines = [
        _svg_header(width, height),
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="16">{html.escape(title)}</text>',
    ]
    for col, name in enumerate(class_names):
        x = label_width + col * cell + cell / 2
        lines.append(
            f'<text x="{x:.2f}" y="58" text-anchor="end" transform="rotate(-45 {x:.2f} 58)" font-size="10">{html.escape(name)}</text>'
        )
    for row_index, name in enumerate(class_names):
        y = top + row_index * cell + cell / 2 + 4
        lines.append(f'<text x="{label_width - 8}" y="{y:.2f}" text-anchor="end" font-size="10">{html.escape(name)}</text>')
        for col_index, value in enumerate(matrix[row_index]):
            intensity = int(245 - (int(value) / max_value) * 175)
            fill = f'rgb({intensity},{intensity},255)'
            x = label_width + col_index * cell
            y_top = top + row_index * cell
            lines.append(f'<rect x="{x}" y="{y_top}" width="{cell}" height="{cell}" fill="{fill}" stroke="#e5e7eb"/>')
            lines.append(f'<text x="{x + cell / 2}" y="{y_top + cell / 2 + 4}" text-anchor="middle" font-size="9">{int(value)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="white"/>'
    )
