"""Route and configuration contract tests for the final application workflow."""

import os
import unittest
from unittest.mock import patch

from app.main import _configured_threshold, app


class TestApiContracts(unittest.TestCase):
    def test_required_routes_are_registered(self):
        routes = {
            (route.path, tuple(route.methods))
            for route in app.routes
            if getattr(route, "methods", None)
        }
        expected = {
            ("/", ("GET",)),
            ("/api/health", ("GET",)),
            ("/health", ("GET",)),
            ("/api/enroll", ("POST",)),
            ("/api/identify", ("POST",)),
            ("/api/enrolled", ("GET",)),
            ("/api/enrolled/{child_id}", ("DELETE",)),
            ("/api/enrolled/{child_id}/name", ("PATCH",)),
            ("/api/retrain", ("POST",)),
            ("/api/preprocess-preview", ("POST",)),
            ("/api/dataset-info", ("GET",)),
            ("/api/models", ("GET",)),
            ("/api/models/artifact", ("GET",)),
            ("/api/models/select", ("POST",)),
            ("/api/train", ("POST",)),
            ("/api/train/status", ("GET",)),
            ("/api/enroll-sample", ("POST",)),
            ("/api/model-info", ("GET",)),
        }
        self.assertTrue(expected.issubset(routes))

    def test_default_confidence_threshold_is_seventy_percent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_configured_threshold(), 0.70)

    def test_environment_can_configure_confidence_threshold(self):
        with patch.dict(os.environ, {"FOOTPRINT_CONFIDENCE_THRESHOLD": "0.82"}):
            self.assertEqual(_configured_threshold(), 0.82)


if __name__ == "__main__":
    unittest.main()
