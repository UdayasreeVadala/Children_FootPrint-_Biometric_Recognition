#!/usr/bin/env python
import subprocess, sys, os, time

# Start uvicorn server
proc = subprocess.Popen(
    [sys.executable, '-c',
     'import uvicorn; uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="error")'],
    cwd=r'D:\Children_FootPrint _Biometric_Recognition',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for startup and verify
time.sleep(8)

import urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)
    print(f'Server started: {r.status}')
except Exception as e:
    print(f'Server failed: {e}')
    proc.terminate()