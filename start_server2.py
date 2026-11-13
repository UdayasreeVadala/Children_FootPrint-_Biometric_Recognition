#!/usr/bin/env python
import subprocess, sys, time, os

# Kill any existing python processes on port 8000
os.system('taskkill /F /IM python.exe 2>nul')

# Start uvicorn
proc = subprocess.Popen(
    [sys.executable, '-c',
     'import uvicorn; uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="error")'],
    cwd=r'D:\Children_FootPrint _Biometric_Recognition',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for startup
time.sleep(8)

# Test health
import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)
    print(f'Server started: {r.status}')
    
    # Test model-info
    r2 = urllib.request.urlopen('http://127.0.0.1:8000/api/model-info', timeout=15)
    info = json.loads(r2.read().decode())
    print('Model info keys:', list(info.keys()))
    
    # Test train/status
    r3 = urllib.request.urlopen('http://127.0.0.1:8000/api/train/status', timeout=10)
    print('Train status:', r3.read()[:200].decode())
    
except Exception as e:
    print(f'Error: {e}')
    proc.terminate()