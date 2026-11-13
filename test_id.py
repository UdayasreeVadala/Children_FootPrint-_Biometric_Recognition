#!/usr/bin/env python
import urllib.request, json, time, numpy as np, cv2
from io import BytesIO

# Wait for model to load
time.sleep(10)

# Create a test footprint image
img = np.full((224, 224), 120, dtype=np.uint8)
buf = BytesIO()
cv2.imencode('.png', img)[1].tobytes()

# Test identification
try:
    files = {'file': ('test.png', buf.getvalue(), 'image/png')}
    r = urllib.request.urlopen('http://127.0.0.1:8000/api/identify', files=files, timeout=20)
    result = json.loads(r.read().decode())
    
    print('=== IDENTIFICATION RESULT ===')
    p = result.get('prediction', {})
    print(f'Predicted ID: {p.get("predicted_id")}')
    print(f'Predicted Name: {p.get("predicted_name")}')
    print(f'Confidence: {p.get("confidence_percent")}%')
    print(f'Is Recognized: {p.get("is_recognized")}')
    print(f'Match Mode: {p.get("match_mode")}')
    print(f'Threshold: {p.get("threshold")}')
    print(f'Top 3: {p.get("top_3")}')
    
except Exception as e:
    print(f'Identify error: {e}')
    import traceback
    traceback.print_exc()
PYEOF