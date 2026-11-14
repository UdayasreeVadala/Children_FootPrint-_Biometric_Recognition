#!/usr/bin/env python
"""Start the FootprintID API server.

The server runs in THIS process, in the foreground, so it stays alive for as
long as this terminal window is open. There is no child process and no
``atexit`` cleanup, which is what previously caused the server to die one
second after "Server started".

Press Ctrl+C to stop.

Environment overrides:
    FOOTPRINT_HOST   bind address  (default 127.0.0.1)
    FOOTPRINT_PORT   bind port     (default 8000)
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    # Always run from the project root so "app.main:app", models/ and
    # data/enrolled/ resolve to the right paths.
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    host = os.environ.get("FOOTPRINT_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("FOOTPRINT_PORT", "8000"))
    except ValueError:
        print(f"Invalid FOOTPRINT_PORT: {os.environ.get('FOOTPRINT_PORT')!r}")
        return 2

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run: pip install -r requirements.txt")
        return 1

    print("=" * 64)
    print(" FootprintID server")
    print(f" Listening : http://{host}:{port}")
    print(f" Health    : http://{host}:{port}/api/health")
    print(f" Model info: http://{host}:{port}/api/model-info")
    print(f" Web UI    : http://{host}:{port}/")
    print()
    print(" Leave this window open. The server stops when you press Ctrl+C.")
    print("=" * 64)
    sys.stdout.flush()

    # reload=False on purpose: the reloader adds a supervising subprocess and
    # is unnecessary (and less reliable on Windows) for normal use.
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
