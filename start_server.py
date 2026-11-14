#!/usr/bin/env python
"""Deprecated alias kept for compatibility.

This script used to run ``taskkill /F /IM python.exe`` (killing every Python
process on the machine) and then start uvicorn as a child that died with the
script. Use ``launch_server.py`` instead.
"""

import os
import runpy
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("start_server.py is deprecated - starting launch_server.py instead.\n")
    sys.argv = [os.path.join(here, "launch_server.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")
