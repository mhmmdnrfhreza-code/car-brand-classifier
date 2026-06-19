import io
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predictor  # noqa: E402
from predictor import ModelNotReadyError, predict  # noqa: E402
from preprocessing import InvalidImageError  # noqa: E402


class FakeModel:
    """Model palsu: mengembalikan probabilitas tetap untuk menguji logika Top-K."""

    def __init__(self, probs):
        self._probs = np.array([probs], dtype="float32")

    def predict(self, batch, verbose=0):
        return self._probs


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


INDEX_TO_LABEL = {0: "hyundai", 1: "lexus", 2: "mazda", 3: "toyota"}


class TestPredictor(unittest.TestCase):
    def test_predict_topk_order(self):
        model = FakeModel([0.1, 0.6, 0.05, 0.25])
        out = predict(_png_bytes(), top_k=3, model=model, index_to_label=INDEX_TO_LABEL)
        self.assertEqual(out["label"], "lexus")
        self.assertAlmostEqual(out["confidence"], 0.6, places=5)
        self.assertEqual([t["label"] for t in out["top_k"]], ["lexus", "toyota", "hyundai"])

    def test_topk_clamped_to_num_classes(self):
        model = FakeModel([0.7, 0.3])
        out = predict(
            _png_bytes(), top_k=10, model=model, index_to_label={0: "a", 1: "b"}
        )
        self.assertEqual(len(out["top_k"]), 2)

    def test_model_not_ready(self):
        predictor._model = None
        predictor._index_to_label = None
        with self.assertRaises(ModelNotReadyError):
            predict(_png_bytes())

    def test_invalid_image(self):
        model = FakeModel([0.5, 0.5])
        with self.assertRaises(InvalidImageError):
            predict(b"not-an-image", model=model, index_to_label={0: "a", 1: "b"})


if __name__ == "__main__":
    unittest.main()
