"""Regression tests for safe enrollment storage helpers."""

import unittest

from fastapi import HTTPException

from app.main import DATA_DIR, _child_directory, _validate_child_details


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


if __name__ == "__main__":
    unittest.main()
