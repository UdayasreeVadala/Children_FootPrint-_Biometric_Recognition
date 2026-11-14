#!/usr/bin/env python
"""Deprecated alias kept for compatibility.

Use ``launch_server.py``: it runs the server in a single long-lived process
instead of a child process that terminated with this script.
"""

import os
import runpy
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("start_server2.py is deprecated - starting launch_server.py instead.\n")
    sys.argv = [os.path.join(here, "launch_server.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")
