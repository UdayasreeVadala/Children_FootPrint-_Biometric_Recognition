"""Tests for dataset discovery, task resolution, and grouped splitting."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.inference.dataset import (
    PreprocessedSample,
    class_distribution,
    discover_dataset,
    grouped_split,
    resolve_task,
    summarize_samples,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (24, 24), (30, 30, 30), -1)
    assert cv2.imwrite(str(path), image)


class TestDatasetDiscovery(unittest.TestCase):
    def test_parses_layout_and_filters_phone_sensor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_image(root / "B003" / "Visit 1" / "Paper" / "Left" / "a.tif")
            write_image(root / "B003" / "Visit 1" / "Paper" / "Right" / "b.tif")
            write_image(root / "B003" / "Visit 5" / "Left" / "phone.jpg")

            clean = discover_dataset(root, include_sensors=("Paper", "Scanner", "Camera"))
            all_images = discover_dataset(root)

            self.assertEqual(len(clean), 2)
            self.assertEqual(len(all_images), 3)
            self.assertEqual({sample.side for sample in clean}, {"Left", "Right"})
            self.assertEqual({sample.sensor for sample in clean}, {"Paper"})
            self.assertEqual({sample.visit for sample in clean}, {"Visit 1"})
            filtered_phone = [sample for sample in all_images if sample.sensor == "Phone"]
            self.assertEqual(len(filtered_phone), 1)

    def test_single_subject_resolves_to_side_task(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_image(root / "B003" / "Visit 1" / "Scanner" / "Left" / "l.bmp")
            write_image(root / "B003" / "Visit 1" / "Scanner" / "Right" / "r.bmp")
            samples = discover_dataset(root)
            self.assertEqual(resolve_task(samples, "auto"), "side")
            self.assertEqual(resolve_task(samples, "side"), "side")

    def test_two_subjects_resolve_to_identity_task(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_image(root / "B003" / "Visit 1" / "Paper" / "Left" / "l.tif")
            write_image(root / "B004" / "Visit 1" / "Paper" / "Left" / "l.tif")
            samples = discover_dataset(root)
            self.assertEqual({sample.subject for sample in samples}, {"B003", "B004"})
            self.assertEqual(resolve_task(samples, "auto"), "identity")

    def test_parses_side_from_filename_and_keeps_capture_together(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_image(root / "Adult01" / "Scanner" / "Visit 1" / "1-left.png")
            write_image(root / "Adult01" / "Scanner" / "Visit 1" / "1-right.png")
            samples = discover_dataset(root)

            self.assertEqual({sample.side for sample in samples}, {"Left", "Right"})
            self.assertEqual(len({sample.group for sample in samples}), 1)

    def test_summary_reports_counts_and_visit_breakdown(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_image(root / "B003" / "Visit 1" / "Paper" / "Left" / "a.tif")
            write_image(root / "B003" / "Visit 1" / "Paper" / "Right" / "b.tif")
            write_image(root / "B003" / "Visit 2" / "Scanner" / "Left" / "c.bmp")

            summary = summarize_samples(discover_dataset(root))

            self.assertEqual(summary["total_images"], 3)
            self.assertEqual(summary["subject_count"], 1)
            self.assertEqual(summary["subjects"], ["B003"])
            self.assertEqual(summary["by_visit"]["Visit 1"], {"total": 2, "left": 1, "right": 1})
            self.assertEqual(summary["by_visit"]["Visit 2"]["total"], 1)


def make_sample(label: str, group: str, index: int) -> PreprocessedSample:
    return PreprocessedSample(
        cache_path=f"{group}-{index}.png",
        source_path=f"{group}-{index}.jpg",
        relative_path=f"{group}-{index}.jpg",
        subject="B003",
        visit="Visit 1",
        side=label,
        sensor="Paper",
        group=group,
        roi_ok=True,
        roi_area_ratio=0.5,
        width=224,
        height=224,
    )


class TestGroupedSplit(unittest.TestCase):
    def test_groups_never_leak_across_splits(self):
        samples = []
        for label in ("Left", "Right"):
            for group_index in range(10):
                for image_index in range(2):
                    samples.append(
                        make_sample(label, f"{label}-group-{group_index}", image_index)
                    )

        train, validation, test = grouped_split(samples, "side", seed=42)

        self.assertTrue(train and validation and test)
        group_sets = [
            {sample.group for sample in split} for split in (train, validation, test)
        ]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])
        self.assertEqual(
            sum(len(split) for split in (train, validation, test)), len(samples)
        )

    def test_class_distribution_covers_all_labels(self):
        samples = [
            make_sample("Left", "a", 0),
            make_sample("Right", "b", 0),
            make_sample("Right", "c", 0),
        ]
        self.assertEqual(class_distribution(samples, "side"), {"Left": 1, "Right": 2})


if __name__ == "__main__":
    unittest.main()
