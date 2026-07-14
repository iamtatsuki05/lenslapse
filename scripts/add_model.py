"""Compatibility shim — the implementation lives in the `lenslapse` package.

Kept so the documented `python scripts/add_model.py` commands keep working from a checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenslapse.add_model import main  # noqa: E402

if __name__ == "__main__":
    main()
