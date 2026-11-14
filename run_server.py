#!/usr/bin/env python
"""Deprecated alias kept for compatibility.

The server used to be started here as a detached child process, which caused it
to be killed as soon as this script exited. Use ``launch_server.py`` instead;
it owns the server in a single long-lived process.
"""

import os
import runpy
import sys

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("run_server.py is deprecated - starting launch_server.py instead.\n")
    sys.argv = [os.path.join(here, "launch_server.py")]
    runpy.run_path(sys.argv[0], run_name="__main__")
