"""Tests for TensorFlow-independent explainability and dataset validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.inference.gradcam import apply_colormap, gradcam_to_base64, overlay_heatmap
from app.inference.training import load_enrollment_dataset


class TestExplainabilityAndTraining(unittest.TestCase):
    def test_overlay_and_png_encoding_are_displayable(self):
        heatmap = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        original = np.full((32, 32), 120, dtype=np.uint8)
        overlay = overlay_heatmap(original, heatmap, alpha=0.4, target_size=(32, 32))
        self.assertEqual(overlay.shape, (32, 32, 3))
        self.assertTrue(gradcam_to_base64(overlay))
        self.assertEqual(apply_colormap(heatmap).shape, (8, 8, 3))

    def test_training_dataset_requires_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "No enrolled children"):
                load_enrollment_dataset(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
