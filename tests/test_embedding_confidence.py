"""Tests for embedding-mode verification and confidence calibration."""

import unittest
from unittest.mock import patch

import numpy as np

from app.inference.classifier import FootprintClassifier


def make_embedding_classifier(threshold: float = 0.70):
    classifier = FootprintClassifier.__new__(FootprintClassifier)
    classifier.confidence_threshold = threshold
    classifier.calibration = {"center": 0.72, "scale": 0.08}
    classifier.is_demo = False
    classifier.match_mode = "embedding"
    classifier.task = "side"
    classifier.architecture = "custom_cnn"
    classifier.architecture_label = "Custom CNN"
    classifier.model = None
    classifier.embedding_model = None
    classifier.last_conv_layer_name = "custom_last_conv"
    classifier.class_records = {}
    classifier.class_names = ["Left", "Right"]
    classifier._child_embeddings = {
        "B003": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    }
    classifier._child_records = {
        "B003": {"child_id": "B003", "child_name": "Child B003"},
    }
    return classifier


class TestEmbeddingConfidence(unittest.TestCase):
    def test_similarity_calibration_is_monotonic(self):
        classifier = make_embedding_classifier()
        low = classifier._similarity_confidence(0.60)
        middle = classifier._similarity_confidence(0.72)
        high = classifier._similarity_confidence(0.88)

        self.assertLess(low, middle)
        self.assertLess(middle, high)
        self.assertAlmostEqual(middle, 0.5, places=6)
        self.assertLess(low, 0.20)
        self.assertGreater(high, 0.85)

    def test_confidence_reports_similarity_and_top_three(self):
        classifier = make_embedding_classifier(threshold=0.70)
        image = np.full((224, 224), 120, dtype=np.uint8)

        with patch.object(FootprintClassifier, "refresh_enrollment_index", return_value=None):
            with patch.object(
                FootprintClassifier,
                "_query_embedding",
                return_value=np.array([1.0, 0.0, 0.0]),
            ):
                result = classifier.predict(image)

        self.assertTrue(result["is_recognized"])
        self.assertEqual(result["match_mode"], "embedding")
        self.assertEqual(result["predicted_id"], "B003")
        self.assertAlmostEqual(result["similarity"], 1.0, places=6)
        self.assertEqual(result["top_3"][0]["child_id"], "B003")
        self.assertIsNotNone(result["top_3"][0]["similarity"])
        self.assertEqual(result["calibration"], {"center": 0.72, "scale": 0.08})

    def test_low_similarity_is_rejected_as_unknown(self):
        classifier = make_embedding_classifier(threshold=0.70)
        image = np.full((224, 224), 10, dtype=np.uint8)

        with patch.object(FootprintClassifier, "refresh_enrollment_index", return_value=None):
            with patch.object(
                FootprintClassifier,
                "_query_embedding",
                return_value=np.array([0.0, 1.0, 0.0]),
            ):
                result = classifier.predict(image)

        self.assertFalse(result["is_recognized"])
        self.assertEqual(result["predicted_name"], "Unknown Child")
        self.assertIsNone(result["predicted_id"])
        self.assertGreaterEqual(result["threshold"], 0.70)

    def test_no_enrollments_returns_actionable_message(self):
        classifier = make_embedding_classifier()
        classifier._child_embeddings = {}
        classifier._child_records = {}

        with patch.object(FootprintClassifier, "refresh_enrollment_index", return_value=None):
            result = classifier.predict(np.zeros((224, 224), dtype=np.uint8))

        self.assertFalse(result["is_recognized"])
        self.assertEqual(result["top_3"], [])
        self.assertIn("Enroll", result["message"])


if __name__ == "__main__":
    unittest.main()
