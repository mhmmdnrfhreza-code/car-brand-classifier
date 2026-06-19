import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (  # noqa: E402
    build_evaluation,
    confusion_matrix,
    macro_f1,
    overall_accuracy,
    per_class_metrics,
)

CLASSES = ["a", "b"]
Y_TRUE = [0, 0, 1, 1]
Y_PRED = [0, 1, 1, 1]
# cm = [[1, 1], [0, 2]]


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix(self):
        cm = confusion_matrix(Y_TRUE, Y_PRED, 2)
        self.assertEqual(cm, [[1, 1], [0, 2]])

    def test_overall_accuracy(self):
        cm = confusion_matrix(Y_TRUE, Y_PRED, 2)
        self.assertAlmostEqual(overall_accuracy(cm), 0.75)

    def test_per_class_metrics(self):
        cm = confusion_matrix(Y_TRUE, Y_PRED, 2)
        pcm = per_class_metrics(cm, CLASSES)
        a, b = pcm[0], pcm[1]
        self.assertEqual(a["support"], 2)
        self.assertAlmostEqual(a["precision"], 1.0)
        self.assertAlmostEqual(a["recall"], 0.5)
        self.assertAlmostEqual(a["f1"], 0.6667, places=4)
        self.assertAlmostEqual(b["precision"], 0.6667, places=4)
        self.assertAlmostEqual(b["recall"], 1.0)

    def test_macro_f1(self):
        cm = confusion_matrix(Y_TRUE, Y_PRED, 2)
        pcm = per_class_metrics(cm, CLASSES)
        self.assertAlmostEqual(macro_f1(pcm), 0.7333, places=4)

    def test_build_evaluation_structure(self):
        ev = build_evaluation(Y_TRUE, Y_PRED, CLASSES, loss=1.05, fine_tuned=True)
        self.assertEqual(ev["num_classes"], 2)
        self.assertEqual(ev["num_samples"], 4)
        self.assertEqual(ev["classes"], CLASSES)
        self.assertEqual(len(ev["confusion_matrix"]), 2)
        self.assertEqual(len(ev["per_class"]), 2)
        self.assertTrue(ev["fine_tuned"])
        self.assertAlmostEqual(ev["accuracy"], 0.75)
        self.assertAlmostEqual(ev["loss"], 1.05)


if __name__ == "__main__":
    unittest.main()
