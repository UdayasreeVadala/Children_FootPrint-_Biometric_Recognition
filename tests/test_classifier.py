"""Tests for confidence scoring without constructing a TensorFlow model."""

import unittest

import numpy as np

from app.inference.classifier import FootprintClassifier


class TestClassifier(unittest.TestCase):
    def make_demo_classifier(self, threshold=0.01):
        classifier = FootprintClassifier.__new__(FootprintClassifier)
        classifier.confidence_threshold = threshold
        classifier.class_records = {
            0: {"child_id": "C001", "child_name": "Asha"},
            1: {"child_id": "C002", "child_name": "Ravi"},
        }
        classifier.is_demo = True
        classifier.model = None
        classifier.last_conv_layer_name = ""
        return classifier

    def test_demo_prediction_is_repeatable_and_includes_child_id(self):
        classifier = self.make_demo_classifier()
        image = np.full((224, 224), 128, dtype=np.uint8)
        first = classifier.predict(image)
        second = classifier.predict(image)
        self.assertEqual(first["all_probabilities"], second["all_probabilities"])
        self.assertEqual(first["top_3"][0]["child_id"], first["predicted_id"])

    def test_threshold_rejects_low_confidence_prediction(self):
        classifier = self.make_demo_classifier(threshold=1.0)
        result = classifier.predict(np.zeros((224, 224), dtype=np.uint8))
        self.assertFalse(result["is_recognized"])
        self.assertEqual(result["predicted_name"], "Unknown Child")

    def test_softmax_sums_to_one(self):
        probabilities = FootprintClassifier._softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
