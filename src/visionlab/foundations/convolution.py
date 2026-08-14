"""Tiny convolution and pooling helpers for T1 concept exercises."""

from __future__ import annotations

Number = int | float
Matrix = list[list[Number]]


def _validate_rectangular(matrix: Matrix, name: str) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return len(matrix), width


def convolve2d(image: Matrix, kernel: Matrix, stride: int = 1, padding: int = 0) -> Matrix:
    """Apply cross-correlation-style 2D convolution to a single-channel image."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")

    image_h, image_w = _validate_rectangular(image, "image")
    kernel_h, kernel_w = _validate_rectangular(kernel, "kernel")
    if kernel_h > image_h + 2 * padding or kernel_w > image_w + 2 * padding:
        raise ValueError("kernel cannot be larger than the padded image")

    padded: Matrix = [[0 for _ in range(image_w + 2 * padding)] for _ in range(padding)]
    for row in image:
        padded.append([0 for _ in range(padding)] + list(row) + [0 for _ in range(padding)])
    padded.extend([[0 for _ in range(image_w + 2 * padding)] for _ in range(padding)])

    padded_h, padded_w = _validate_rectangular(padded, "padded image")
    out_h = ((padded_h - kernel_h) // stride) + 1
    out_w = ((padded_w - kernel_w) // stride) + 1
    output: Matrix = []
    for y in range(out_h):
        row: list[Number] = []
        for x in range(out_w):
            top = y * stride
            left = x * stride
            total: Number = 0
            for ky in range(kernel_h):
                for kx in range(kernel_w):
                    total += padded[top + ky][left + kx] * kernel[ky][kx]
            row.append(total)
        output.append(row)
    return output


def max_pool2d(image: Matrix, pool_size: int = 2, stride: int | None = None) -> Matrix:
    """Apply non-overlapping or strided max pooling to a single-channel image."""
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    step = pool_size if stride is None else stride
    if step <= 0:
        raise ValueError("stride must be positive")

    image_h, image_w = _validate_rectangular(image, "image")
    if pool_size > image_h or pool_size > image_w:
        raise ValueError("pool_size cannot be larger than the image")

    out_h = ((image_h - pool_size) // step) + 1
    out_w = ((image_w - pool_size) // step) + 1
    output: Matrix = []
    for y in range(out_h):
        row: list[Number] = []
        for x in range(out_w):
            top = y * step
            left = x * step
            values = [
                image[top + dy][left + dx]
                for dy in range(pool_size)
                for dx in range(pool_size)
            ]
            row.append(max(values))
        output.append(row)
    return output
