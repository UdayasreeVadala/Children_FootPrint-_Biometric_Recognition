"""Regression tests for Component 2: ROI detection and Gabor enhancement."""

import unittest

import cv2
import numpy as np

from app.preprocessing.enhancement import (
    apply_gabor_filter_bank,
    create_gabor_kernel,
    enhance_footprint,
)
from app.preprocessing.roi_detection import process_footprint


def synthetic_footprint() -> np.ndarray:
    """Create a dark footprint-like shape on a light scan background."""
    image = np.full((480, 320, 3), 245, dtype=np.uint8)
    cv2.ellipse(image, (160, 250), (75, 180), 0, 0, 360, (35, 35, 35), -1)
    cv2.ellipse(image, (125, 270), (45, 65), 0, 0, 360, (245, 245, 245), -1)
    for y in range(180, 330, 24):
        cv2.line(image, (100, y), (220, y + 12), (90, 90, 90), 2)
    return image


class TestPreprocessing(unittest.TestCase):
    def test_roi_pipeline_returns_target_size_and_visualization(self):
        result = process_footprint(synthetic_footprint(), return_visualization=True)

        self.assertEqual(result["roi"].shape, (224, 224, 3))
        self.assertEqual(result["mask"].shape, (480, 320))
        self.assertEqual(result["visualization"].shape, (480, 320, 3))
        self.assertLess(result["region_info"]["forefoot_y"], result["region_info"]["midfoot_y"])

    def test_gabor_bank_has_six_responses_and_zero_mean_kernels(self):
        image = cv2.cvtColor(synthetic_footprint(), cv2.COLOR_BGR2GRAY)
        fused, responses = apply_gabor_filter_bank(image)

        self.assertEqual(fused.shape, image.shape)
        self.assertEqual(fused.dtype, np.uint8)
        self.assertEqual(len(responses), 6)
        self.assertAlmostEqual(float(create_gabor_kernel(30).mean()), 0.0, places=10)

    def test_full_enhancement_returns_displayable_images(self):
        roi = process_footprint(synthetic_footprint())["roi"]
        result = enhance_footprint(roi)

        self.assertEqual(result["enhanced"].shape, (224, 224))
        self.assertEqual(result["enhanced"].dtype, np.uint8)
        self.assertEqual(result["clahe"].shape, (224, 224))


if __name__ == "__main__":
    unittest.main()
