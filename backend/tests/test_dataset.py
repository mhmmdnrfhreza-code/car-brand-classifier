import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (  # noqa: E402
    build_label_index,
    class_distribution,
    discover_dataset,
    filter_min_images,
    load_samples_from_dir,
    resolve_images_dir,
    save_class_indices,
    load_class_indices,
    split_train_val,
    stratified_split,
)

BRANDS = ["hyundai", "lexus", "mazda", "mercedes"]


def _touch_images(directory, brand, n):
    d = Path(directory) / brand
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"{brand}_{i}.jpg").write_bytes(b"x")


class TestDataset(unittest.TestCase):
    def test_load_samples_from_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            for b in BRANDS:
                _touch_images(tmp, b, 5)
            samples = load_samples_from_dir(tmp)
            self.assertEqual(len(samples), 20)
            self.assertEqual(set(l for _, l in samples), set(BRANDS))

    def test_discover_split_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            for b in BRANDS:
                _touch_images(Path(tmp) / "Train", b, 8)
                _touch_images(Path(tmp) / "Test", b, 2)
            mode, train_dir, test_dir, root = discover_dataset(tmp)
            self.assertEqual(mode, "split")
            self.assertEqual(len(load_samples_from_dir(train_dir)), 32)
            self.assertEqual(len(load_samples_from_dir(test_dir)), 8)

    def test_discover_flat_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            for b in BRANDS:
                _touch_images(tmp, b, 4)
            mode, train_dir, test_dir, root = discover_dataset(tmp)
            self.assertEqual(mode, "flat")
            self.assertIsNone(train_dir)

    def test_resolve_descends_single_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "Car_Brand_Logos"
            for b in BRANDS:
                _touch_images(wrapper / "Train", b, 3)
                _touch_images(wrapper / "Test", b, 1)
            resolved = resolve_images_dir(tmp)
            self.assertEqual(resolved, wrapper)

    def test_build_label_index_sorted(self):
        samples = [("a.jpg", "toyota"), ("b.jpg", "opel"), ("c.jpg", "opel")]
        idx = build_label_index(samples)
        self.assertEqual(idx, {"opel": 0, "toyota": 1})

    def test_filter_min_images(self):
        samples = (
            [("a%d.jpg" % i, "toyota") for i in range(5)]
            + [("b%d.jpg" % i, "opel") for i in range(2)]
        )
        filtered = filter_min_images(samples, 3)
        self.assertEqual(set(l for _, l in filtered), {"toyota"})
        # min_count <= 1 => tanpa filter
        self.assertEqual(len(filter_min_images(samples, 1)), len(samples))

    def test_split_train_val_per_label(self):
        samples = [("%s_%d.jpg" % (b, i), b) for b in BRANDS for i in range(10)]
        train, val = split_train_val(samples, val_split=0.2)
        # Tiap kelas punya val (10*0.2=2) dan train (8)
        self.assertEqual(len(val), 8)
        self.assertEqual(len(train), 32)
        for b in BRANDS:
            self.assertTrue(any(l == b for _, l in val))
            self.assertTrue(any(l == b for _, l in train))

    def test_stratified_split_covers_classes(self):
        samples = [("%s_%d.jpg" % (b, i), b) for b in BRANDS for i in range(10)]
        train, val, test = stratified_split(samples)
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, 40)
        for b in BRANDS:
            self.assertTrue(any(l == b for _, l in train))

    def test_class_indices_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "class_indices.json"
            idx = {"hyundai": 0, "toyota": 1}
            save_class_indices(idx, path)
            self.assertEqual(load_class_indices(path), idx)

    def test_class_distribution(self):
        samples = [("a.jpg", "x"), ("b.jpg", "x"), ("c.jpg", "y")]
        self.assertEqual(class_distribution(samples), {"x": 2, "y": 1})


if __name__ == "__main__":
    unittest.main()
