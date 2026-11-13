#!/usr/bin/env python
import subprocess, sys, time, os

# Kill any existing python processes on port 8000
ret = os.system('taskkill /F /IM python.exe 2>nul')

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

# Check if process is still running
if proc.poll() is None:
    print(f'Server started with PID {proc.pid}')
    
    # Test health
    import urllib.request
    try:
        r = urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)
        print(f'Health: {r.status}')
        
        # Test model-info
        r2 = urllib.request.urlopen('http://127.0.0.1:8000/api/model-info', timeout=15)
        info = json.loads(r2.read().decode())
        print(f'Model info: task={info.get("task")}, mode={info.get("match_mode")}, enrolled={info.get("enrolled_children")}')
    except Exception as e:
        print(f'Error accessing endpoints: {e}')
else:
    print(f'Server exited with code {proc.poll()}')
    stderr = proc.stderr.read().decode( errors='replace')
    if stderr:
        print(f'Stderr: {stderr[:300]}')