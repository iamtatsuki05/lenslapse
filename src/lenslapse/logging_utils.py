"""Shared logging setup for this package's CLI entry points.

Every module in this package logs via its own `logging.getLogger(__name__)` (a child of the
`lenslapse` logger) and never configures handlers or levels itself — that would be a library
reconfiguring a caller's own logging setup as a side effect of being imported or called, which a
well-behaved library must not do. `configure_cli_logging()` is the one place that actually does
this, and it must only be called from a true entry point: each module's own `if __name__ ==
"__main__":` guard, and `cli.py`'s `main()` (the installed `lenslapse` command's actual entry
point, which every other file's `__main__` guard bypasses).
"""

import logging


def configure_cli_logging() -> None:
    """Root logger stays at WARNING (third-party libraries like onnx/onnxscript/transformers
    default to INFO-or-noisier internal logging that nobody asked to see); only this package's
    own loggers are raised to INFO, matching what used to be plain `print()` calls.

    Both the `lenslapse` namespace and `__main__` are raised: whichever of this package's own
    files is the actual entry point (`python -m lenslapse.export_checkpoints`, a subprocess
    add_model.py shells out to, ...) is loaded with `__name__ == "__main__"`, not its real dotted
    name — a `logging.getLogger(__name__)` there is a logger literally named "__main__", not a
    child of "lenslapse", and raising only the latter would silently leave that file's own
    messages at the default WARNING threshold."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logging.getLogger("lenslapse").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
