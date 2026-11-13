"""Tests for the embedding-similarity calibration logic."""

import unittest
from dataclasses import dataclass

import numpy as np

from calibrate import calibrate_samples


@dataclass
class FakeSample:
    side: str
    group: str
    cache_path: str = ""


class TestCalibration(unittest.TestCase):
    def test_separated_distributions_produce_sensible_calibration(self):
        samples = []
        embeddings = []
        for side, direction in (("Left", [1.0, 0.08]), ("Right", [0.08, 1.0])):
            for group_index in range(4):
                jitter = np.array([0.0, 0.01 * ((-1) ** group_index)])
                vector = np.array(direction) + jitter
                vector = vector / np.linalg.norm(vector)
                samples.append(FakeSample(side=side, group=f"{side}-{group_index}"))
                embeddings.append(vector)

        result = calibrate_samples(samples, np.asarray(embeddings))

        self.assertGreater(result["genuine_mean"], result["impostor_mean"])
        self.assertGreater(result["center"], result["impostor_mean"])
        self.assertLess(result["center"], result["genuine_mean"])
        self.assertGreater(result["threshold_similarity"], result["impostor_mean"])
        self.assertEqual(result["false_accept_rate"], 0.0)
        self.assertEqual(result["false_reject_rate"], 0.0)
        self.assertGreater(result["genuine_pairs"], 0)
        self.assertGreater(result["impostor_pairs"], 0)

    def test_overlapping_distributions_fall_back_to_defaults(self):
        samples = [
            FakeSample("Left", "L0"),
            FakeSample("Left", "L1"),
            FakeSample("Right", "R0"),
            FakeSample("Right", "R1"),
        ]
        embeddings = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        result = calibrate_samples(samples, embeddings)
        self.assertIn("warning", result)
        self.assertEqual((result["center"], result["scale"]), (0.72, 0.08))


if __name__ == "__main__":
    unittest.main()
