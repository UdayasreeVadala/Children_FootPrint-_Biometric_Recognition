"""Regression tests for safe enrollment storage helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.main import (
    DATA_DIR,
    _activate_enrollment_model,
    _child_directory,
    _rename_enrolled_child,
    _validate_child_details,
)


class TestEnrollmentHelpers(unittest.TestCase):
    def test_valid_child_details_are_trimmed(self):
        self.assertEqual(
            _validate_child_details("  Asha Patel  ", " CHILD_001 "),
            ("Asha Patel", "CHILD_001"),
        )

    def test_rejects_path_traversal_and_invalid_characters(self):
        for child_id in ("../outside", "child/name", "child name", ""):
            with self.assertRaises(HTTPException):
                _validate_child_details("Asha", child_id)

    def test_child_directory_remains_under_enrollment_directory(self):
        child_directory = _child_directory("CHILD-001")
        self.assertEqual(child_directory.parent, DATA_DIR.resolve())


class TestRenameEnrolledChild(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.data_dir = Path(self.temporary_directory.name)
        self.metadata_path = self.data_dir / "metadata.json"
        self.metadata_path.write_text(
            json.dumps(
                {
                    "children": [
                        {"child_id": "CHILD_001", "child_name": "B001", "image_count": 3},
                        {"child_id": "CHILD_002", "child_name": "B002", "image_count": 2},
                    ]
                }
            ),
            encoding="utf-8",
        )

    def test_rename_changes_only_the_display_name(self):
        with patch("app.main.DATA_DIR", self.data_dir):
            result = _rename_enrolled_child("CHILD_001", "  Aarav  ")

        self.assertEqual(result["previous_name"], "B001")
        self.assertEqual(result["child_name"], "Aarav")

        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["children"][0]["child_name"], "Aarav")
        self.assertEqual(metadata["children"][0]["image_count"], 3)
        self.assertEqual(metadata["children"][1]["child_name"], "B002")

    def test_unknown_child_is_rejected(self):
        with patch("app.main.DATA_DIR", self.data_dir):
            with self.assertRaises(HTTPException) as context:
                _rename_enrolled_child("CHILD_999", "Nobody")

        self.assertEqual(context.exception.status_code, 404)

    def test_blank_name_is_rejected(self):
        with patch("app.main.DATA_DIR", self.data_dir):
            with self.assertRaises(HTTPException) as context:
                _rename_enrolled_child("CHILD_001", "   ")

        self.assertEqual(context.exception.status_code, 400)


class TestActivateEnrollmentModel(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.models_dir = Path(self.temporary_directory.name)
        self.classes_path = self.models_dir / "footprint_model_classes.json"
        self.classes_path.write_text(
            json.dumps(
                {
                    "0": {"child_id": "CHILD_001", "child_name": "Aarav"},
                    "1": {"child_id": "CHILD_002", "child_name": "Diya"},
                }
            ),
            encoding="utf-8",
        )
        (self.models_dir / "footprint_model_meta.json").write_text(
            json.dumps({"last_conv_layer": "Conv_1"}), encoding="utf-8"
        )
        self.registry_path = self.models_dir / "registry.json"
        self.registry_path.write_text(
            json.dumps(
                {
                    "task": "identity",
                    "class_names": ["B001", "B002", "B003"],
                    "architectures": {
                        "efficientnetb0": {
                            "architecture": "efficientnetb0",
                            "model_path": str(self.models_dir / "efficientnetb0_identity.keras"),
                            "source": "dataset-benchmark",
                            "selectable": False,
                        },
                        "custom_cnn": {
                            "architecture": "custom_cnn",
                            "model_path": str(self.models_dir / "footprint_model.keras"),
                            "source": "enrollments",
                            "selectable": True,
                        },
                    },
                    "model_paths": {
                        "efficientnetb0": str(self.models_dir / "efficientnetb0_identity.keras"),
                        "custom_cnn": str(self.models_dir / "footprint_model.keras"),
                    },
                    "calibrations": {"custom_cnn": {"center": 0.72, "scale": 0.08}},
                }
            ),
            encoding="utf-8",
        )

    def test_retrain_preserves_other_architectures(self):
        result = {
            "classes_path": str(self.classes_path),
            "model_path": str(self.models_dir / "footprint_model.keras"),
            "validation_accuracy": 42.5,
            "epochs_completed": 30,
            "fine_tune_epochs_completed": 12,
            "val_images": 10,
            "images_used": 60,
        }
        with patch("app.main.MODELS_DIR", self.models_dir):
            _activate_enrollment_model(result, "mobilenetv2")

        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        architectures = registry["architectures"]

        # The freshly trained model becomes active.
        self.assertEqual(registry["active_model"], "mobilenetv2")
        self.assertEqual(registry["best_model"], "mobilenetv2")
        self.assertTrue(architectures["mobilenetv2"]["selectable"])
        self.assertEqual(architectures["mobilenetv2"]["accuracy_label"], "Validation accuracy")
        self.assertAlmostEqual(architectures["mobilenetv2"]["test_accuracy"], 0.425)
        self.assertEqual(architectures["mobilenetv2"]["epochs_completed"], 42)
        self.assertEqual(registry["class_names"], ["CHILD_001", "CHILD_002"])

        # The benchmark architecture survives, still display-only.
        self.assertIn("efficientnetb0", architectures)
        self.assertFalse(architectures["efficientnetb0"]["selectable"])

        # A stale enrollment entry would point at the new weights, so it is dropped.
        self.assertNotIn("custom_cnn", architectures)
        self.assertNotIn("custom_cnn", registry["model_paths"])


if __name__ == "__main__":
    unittest.main()
