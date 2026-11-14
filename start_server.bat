@echo off
REM Start the FootprintID server. Double-click this file, or run it in cmd.
REM The server runs in this window. Close the window (or press Ctrl+C) to stop.

cd /d "%~dp0"
python launch_server.py

echo.
echo Server stopped.
pause
