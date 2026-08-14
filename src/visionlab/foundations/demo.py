"""Generate tiny T1 visual artifacts from a synthetic image and edge kernel."""

from __future__ import annotations

from pathlib import Path

from visionlab.foundations.convolution import convolve2d
from visionlab.foundations.tiny_image import write_pgm


def _scale_to_pixels(values: list[list[int | float]]) -> list[list[int]]:
    flat = [value for row in values for value in row]
    low = min(flat)
    high = max(flat)
    if low == high:
        return [[0 for _ in row] for row in values]
    return [
        [int(round(((value - low) / (high - low)) * 255)) for value in row]
        for row in values
    ]


def synthetic_step_image(size: int = 16) -> list[list[int]]:
    """Create a simple dark-to-light vertical edge image."""
    if size < 4:
        raise ValueError("size must be at least 4")
    return [[32 if x < size // 2 else 224 for x in range(size)] for _ in range(size)]


def generate_artifacts(output_dir: str | Path = "outputs/t1_foundations") -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source = synthetic_step_image()
    vertical_edge_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    feature_map = convolve2d(source, vertical_edge_kernel, padding=1)
    scaled_feature_map = _scale_to_pixels(feature_map)

    source_path = output / "synthetic_step_image.pgm"
    feature_path = output / "vertical_edge_feature_map.pgm"
    write_pgm(source_path, source)
    write_pgm(feature_path, scaled_feature_map)

    summary_path = output / "README.md"
    summary_path.write_text(
        "\n".join(
            [
                "# T1 Foundation Visual Artifacts",
                "",
                "Generated from a synthetic grayscale step image and a vertical edge kernel.",
                "These files demonstrate tensor/image shape and convolution behavior only.",
                "They are not dataset samples, model outputs, or diagnostic evidence.",
                "",
                "- `synthetic_step_image.pgm`: dependency-light source image.",
                "- `vertical_edge_feature_map.pgm`: scaled convolution response.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source_image": str(source_path),
        "feature_map": str(feature_path),
        "summary": str(summary_path),
    }


def main() -> int:
    for name, path in generate_artifacts().items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
