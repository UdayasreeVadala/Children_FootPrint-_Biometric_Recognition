"""Unit tests for TensorFlow-independent training metrics and helpers."""

import unittest

import numpy as np

from app.inference.architectures import (
    ARCHITECTURE_LABELS,
    normalize_architecture_name,
)
from app.inference.training import _binary_auc, _choose_best, _class_weights, _macro_f1


class TestArchitectureNames(unittest.TestCase):
    def test_aliases_normalize_to_canonical_keys(self):
        self.assertEqual(normalize_architecture_name("MobileNet-V2"), "mobilenetv2")
        self.assertEqual(normalize_architecture_name("efficientnet"), "efficientnetb0")
        self.assertEqual(normalize_architecture_name("custom_cnn"), "custom_cnn")
        self.assertEqual(normalize_architecture_name("CNN"), "custom_cnn")

    def test_every_supported_key_has_a_label(self):
        for key in ("mobilenetv2", "efficientnetb0", "custom_cnn"):
            self.assertIn(key, ARCHITECTURE_LABELS)

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            normalize_architecture_name("resnet50")


class TestTrainingMetrics(unittest.TestCase):
    def test_macro_f1_is_one_for_perfect_predictions(self):
        true_labels = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(_macro_f1(true_labels, true_labels, 3), 1.0)

    def test_macro_f1_penalises_misclassification(self):
        true_labels = np.array([0, 0, 1, 1])
        predicted = np.array([0, 1, 1, 1])
        score = _macro_f1(true_labels, predicted, 2)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_binary_auc_for_perfect_and_reversed_rankings(self):
        true_labels = np.array([0, 0, 1, 1])
        perfect = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        reversed_scores = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
        self.assertAlmostEqual(_binary_auc(true_labels, perfect), 1.0)
        self.assertAlmostEqual(_binary_auc(true_labels, reversed_scores), 0.0)

    def test_binary_auc_is_undefined_for_single_class(self):
        self.assertIsNone(
            _binary_auc(np.array([1, 1]), np.array([[0.1, 0.9], [0.2, 0.8]]))
        )

    def test_class_weights_balance_rare_classes(self):
        weights = _class_weights(np.array([0, 0, 0, 1]), 2)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(weights[0], 4 / (2 * 3))

    def test_choose_best_prefers_higher_accuracy(self):
        metrics = {
            "mobilenetv2": {"test_accuracy": 0.80, "test_top3_accuracy": 0.9},
            "efficientnetb0": {"test_accuracy": 0.91, "test_top3_accuracy": 0.95},
            "custom_cnn": {"test_accuracy": 0.85, "test_top3_accuracy": 0.90},
        }
        self.assertEqual(_choose_best(metrics), "efficientnetb0")


if __name__ == "__main__":
    unittest.main()
