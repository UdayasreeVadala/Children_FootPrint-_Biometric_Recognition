#!/usr/bin/env python
import subprocess, sys, os, atexit

# Start uvicorn in a subprocess
proc = subprocess.Popen(
    [sys.executable, '-u',
     'app/main.py'] if os.name != 'nt' 
     else [sys.executable, '-c',
     'import uvicorn; uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="error")'],
    cwd='D:\\Children_FootPrint _Biometric_Recognition',
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Ensure server dies when this script exits
atexit.register(lambda: proc.terminate() if proc.poll() is None else None)

import time
# Wait for startup
for i in range(20):
    time.sleep(1)
    try:
        import urllib.request
        r = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)
        print(f'Server started after {i+1}s (health={r.status})')
        break
    except Exception as e:
        if i == 19:
            print(f'Server failed to start after 20s: {e}')
            proc.terminate()
        pass