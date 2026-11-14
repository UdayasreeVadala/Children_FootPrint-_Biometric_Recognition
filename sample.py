#!/usr/bin/env python
"""Smoke test for the FootprintID server.

Checks the live API at http://127.0.0.1:8000:
  * /api/health
  * /api/model-info
  * /api/identify with a synthetic (unknown) footprint
  * /api/identify with a real image from Training_Data (if present)

Start the server first (in its own window):

    python launch_server.py

Then run:

    python sample.py

Requires: requests, numpy, opencv-python
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("The 'requests' package is required: pip install requests")

BASE_URL = os.environ.get("FOOTPRINT_SERVER", "http://127.0.0.1:8000")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _ascii_safe_stdout() -> None:
    """Avoid UnicodeEncodeError on the Windows cp1252 console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def get_json(path: str, timeout: float = 30.0) -> dict:
    response = requests.get(f"{BASE_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def identify(image_bytes: bytes, filename: str, content_type: str, timeout: float = 180.0) -> dict:
    response = requests.post(
        f"{BASE_URL}/api/identify",
        files={"file": (filename, image_bytes, content_type)},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def synthetic_footprint() -> bytes:
    """A flat grey image - guaranteed not to match any enrolled child."""
    image = np.full((224, 224), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode the synthetic image")
    return buffer.tobytes()


def first_training_image() -> tuple[str, bytes] | tuple[None, None]:
    pattern = os.path.join(PROJECT_ROOT, "Training_Data", "**", "*.JPG")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        return None, None
    path = paths[0]
    with open(path, "rb") as handle:
        return path, handle.read()


def report(label: str, result: dict, threshold: float) -> None:
    prediction = result.get("prediction", {}) or {}
    confidence_percent = prediction.get("confidence_percent")
    recognized = prediction.get("is_recognized")

    print(f"\n--- {label} ---")
    print(f"  predicted name : {prediction.get('predicted_name')}")
    print(f"  predicted id   : {prediction.get('predicted_id')}")
    print(f"  confidence     : {confidence_percent}%")
    print(f"  is_recognized  : {recognized}")
    for entry in prediction.get("top_3", []) or []:
        print(f"    top3: {entry.get('child_id')} -> {entry.get('confidence')}%")

    gradcam = result.get("gradcam", {}) or {}
    if gradcam.get("error"):
        print(f"  gradcam        : ERROR - {gradcam['error']}")
    elif gradcam.get("overlay"):
        print(f"  gradcam        : ok ({gradcam.get('method')})")

    if confidence_percent is None:
        return
    # confidence_percent is 0-100, the threshold is 0-1.
    expected = confidence_percent >= threshold * 100.0
    verdict = "PASS" if expected == bool(recognized) else "FAIL"
    print(
        f"  threshold check: {verdict} "
        f"({confidence_percent}% vs {threshold * 100:.0f}% -> expected is_recognized={expected})"
    )


def main() -> int:
    _ascii_safe_stdout()
    print("=" * 64)
    print("FootprintID smoke test")
    print("=" * 64)

    try:
        health = get_json("/api/health", timeout=10)
    except Exception as error:
        print(f"\nCannot reach {BASE_URL}")
        print(f"  {error}")
        print("\nStart the server in its own window first:")
        print("  python launch_server.py")
        return 1

    print("\n1. health")
    print(f"   {json.dumps(health)}")
    if not health.get("model_loaded"):
        print("   model still warming up - retrying in a moment...")
        for _ in range(30):
            try:
                info = get_json("/api/model-info", timeout=30)
                break
            except Exception:
                continue

    print("\n2. model info")
    info = get_json("/api/model-info", timeout=120)
    threshold = float(info.get("threshold", 0.7))
    print(f"   task              : {info.get('task')}")
    print(f"   match mode        : {info.get('match_mode')}")
    print(f"   architecture      : {info.get('architecture')}")
    print(f"   enrolled children : {info.get('enrolled_children')}")
    print(f"   threshold         : {threshold:.2f} ({threshold * 100:.0f}%)")
    print(f"   grad-cam layer    : {info.get('last_conv_layer')}")

    print("\n3. unknown footprint (must be rejected)")
    report("synthetic grey image", identify(synthetic_footprint(), "grey.png", "image/png"), threshold)

    path, image_bytes = first_training_image()
    if image_bytes is not None:
        relative = os.path.relpath(path, PROJECT_ROOT)
        print(f"\n4. real image from the dataset: {relative}")
        parts = relative.replace("\\", "/").split("/")
        actual_id = parts[1] if parts[0] == "Training_Data" and len(parts) > 1 else "unknown"
        result = identify(image_bytes, os.path.basename(path), "image/jpeg")
        prediction = result.get("prediction", {}) or {}
        predicted_id = prediction.get("predicted_id") or (prediction.get("top_3") or [{}])[0].get("child_id")
        report("real image", result, threshold)
        print(f"  expected child : {actual_id}")
        print(f"  top-1 child    : {predicted_id}")
        print(f"  top-1 correct  : {predicted_id == actual_id}")
    else:
        print("\n4. no images found under Training_Data/ - skipping real-image test")

    print("\n" + "=" * 64)
    print("Done.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
