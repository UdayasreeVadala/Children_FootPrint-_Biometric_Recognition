import urllib.request, json

# 1. Check health
r = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)
print('Health:', r.read()[:100].decode())

# 2. Check model state
r = urllib.request.urlopen('http://127.0.0.1:8000/api/model-info', timeout=15)
info = json.loads(r.read().decode())
print(f'''=== MODEL STATE ===
Task: {info.get('task')}
Match mode: {info.get('match_mode')}
Architecture: {info.get('architecture')}
Enrolled children: {info.get('enrolled_children')}
Calibration: {info.get('calibration')}
Threshold: {info.get('threshold')}
Last_conv_layer: {info.get('last_conv_layer')}
''')