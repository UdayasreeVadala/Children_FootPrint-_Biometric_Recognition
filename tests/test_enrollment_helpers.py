"""Regression tests for safe enrollment storage helpers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.main import (
    DATA_DIR,
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


if __name__ == "__main__":
    unittest.main()
