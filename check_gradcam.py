#!/usr/bin/env python
# Check Grad-CAM layer expectations
from app.inference import gradcam
import inspect

# Get the source of generate_gradcam and generate_similarity_gradcam
try:
    source = inspect.getsource(gradcam.generate_gradcam)
    print("=== generate_gradcam ===")
    print(source[:2000])
except Exception as e:
    print(f"Error getting generate_gradcam: {e}")

print("---")

try:
    source = inspect.getsource(gradcam.generate_similarity_gradcam)
    print("=== generate_similarity_gradcam ===")
    print(source[:2000])
except Exception as e:
    print(f"Error getting generate_similarity_gradcam: {e}")
PYEOF