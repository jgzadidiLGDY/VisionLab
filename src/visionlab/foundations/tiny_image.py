"""Minimal PGM image helpers for T1 image-loading exercises."""

from __future__ import annotations

from pathlib import Path


PixelGrid = list[list[int]]


def _validate_pixels(pixels: PixelGrid) -> tuple[int, int]:
    if not pixels or not pixels[0]:
        raise ValueError("pixels must be a non-empty 2D grid")
    width = len(pixels[0])
    if any(len(row) != width for row in pixels):
        raise ValueError("pixels must be rectangular")
    for row in pixels:
        for value in row:
            if not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError("PGM pixels must be integers in [0, 255]")
    return len(pixels), width


def write_pgm(path: str | Path, pixels: PixelGrid) -> None:
    """Write an ASCII PGM image for dependency-light visual checks."""
    height, width = _validate_pixels(pixels)
    lines = ["P2", f"{width} {height}", "255"]
    lines.extend(" ".join(str(value) for value in row) for row in pixels)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def read_pgm(path: str | Path) -> PixelGrid:
    """Read the small ASCII PGM subset emitted by write_pgm."""
    tokens: list[str] = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        body = line.split("#", 1)[0].strip()
        if body:
            tokens.extend(body.split())
    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError("expected an ASCII PGM file starting with P2")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    if width <= 0 or height <= 0 or max_value != 255:
        raise ValueError("expected positive dimensions and max value 255")
    values = [int(token) for token in tokens[4:]]
    if len(values) != width * height:
        raise ValueError("PGM pixel count does not match dimensions")
    pixels = [values[index : index + width] for index in range(0, len(values), width)]
    _validate_pixels(pixels)
    return pixels
