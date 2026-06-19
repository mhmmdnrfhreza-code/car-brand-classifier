import os
import sys
import unittest

import numpy as np
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import (  # noqa: E402
    InvalidImageError,
    make_batch,
    preprocess_bytes,
    preprocess_pil,
)
from config import IMG_SIZE  # noqa: E402


def _png_bytes(color, size=(64, 48)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestPreprocessing(unittest.TestCase):
    def test_shape_and_range(self):
        arr = preprocess_bytes(_png_bytes((128, 128, 128)))
        self.assertEqual(arr.shape, (IMG_SIZE[0], IMG_SIZE[1], 3))
        self.assertEqual(arr.dtype, np.float32)
        self.assertGreaterEqual(arr.min(), -1.0001)
        self.assertLessEqual(arr.max(), 1.0001)

    def test_white_maps_to_one(self):
        arr = preprocess_bytes(_png_bytes((255, 255, 255)))
        self.assertAlmostEqual(float(arr.max()), 1.0, places=3)
        self.assertAlmostEqual(float(arr.min()), 1.0, places=3)

    def test_black_maps_to_minus_one(self):
        arr = preprocess_bytes(_png_bytes((0, 0, 0)))
        self.assertAlmostEqual(float(arr.min()), -1.0, places=3)
        self.assertAlmostEqual(float(arr.max()), -1.0, places=3)

    def test_resize_to_img_size(self):
        img = Image.new("RGB", (300, 200), (10, 20, 30))
        arr = preprocess_pil(img)
        self.assertEqual(arr.shape, (IMG_SIZE[0], IMG_SIZE[1], 3))

    def test_make_batch(self):
        arr = preprocess_bytes(_png_bytes((100, 100, 100)))
        batch = make_batch(arr)
        self.assertEqual(batch.shape, (1, IMG_SIZE[0], IMG_SIZE[1], 3))

    def test_invalid_image_raises(self):
        with self.assertRaises(InvalidImageError):
            preprocess_bytes(b"not an image")

    def test_empty_bytes_raises(self):
        with self.assertRaises(InvalidImageError):
            preprocess_bytes(b"")


if __name__ == "__main__":
    unittest.main()
