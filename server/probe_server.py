"""Compatibility shim — the implementation lives in the `lenslapse` package.

Kept so the documented `python server/probe_server.py` commands keep working from a checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenslapse.server import main  # noqa: E402

if __name__ == "__main__":
    main()
