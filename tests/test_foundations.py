import tempfile
import unittest
from pathlib import Path

from visionlab.foundations.convolution import convolve2d, max_pool2d
from visionlab.foundations.tiny_image import read_pgm, write_pgm


class FoundationsTest(unittest.TestCase):
    def test_convolve2d_detects_vertical_edge(self):
        image = [
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ]
        kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

        result = convolve2d(image, kernel)

        self.assertEqual(result, [[40, 40], [40, 40]])

    def test_convolve2d_rejects_non_rectangular_image(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            convolve2d([[1, 2], [3]], [[1]])

    def test_max_pool2d_reduces_spatial_size(self):
        result = max_pool2d([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7], [2, 4, 6, 8]])

        self.assertEqual(result, [[6, 8], [4, 8]])

    def test_pgm_round_trip_preserves_shape_and_pixels(self):
        pixels = [[0, 128], [200, 255]]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tiny.pgm"
            write_pgm(path, pixels)

            self.assertEqual(read_pgm(path), pixels)

    def test_pgm_rejects_out_of_range_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pgm"

            with self.assertRaisesRegex(ValueError, r"\[0, 255\]"):
                write_pgm(path, [[256]])
