"""`lenslapse` command-line entry point.

Designed for people who just want to point the hosted web app at their own models:

    pip install lenslapse
    lenslapse server        # starts the probe server and opens the app in the browser

Subcommands delegate to the underlying modules, so every flag they document works here too
(`lenslapse server --port 9000`, `lenslapse add-model --model gpt2 ...`).
"""

import sys

from lenslapse.logging_utils import configure_cli_logging

USAGE = """\
usage: lenslapse <command> [options]

commands:
  server       start the local probe server and open the web app (--no-open to skip)
  probe        logit-lens one checkpoint from the terminal (the UI's "Live probe")
  trace        probe every checkpoint (the UI's "trace across training")
  models       list/add/remove/convert models (the UI's "models" dialog)
  add-model    convert a model to ONNX and register it for in-browser use
"""


def main() -> None:
    # the single true entry point for the installed `lenslapse` command (pyproject.toml's
    # [project.scripts] calls this directly, bypassing every other file's __main__ guard below
    # it), so logging must be configured here rather than relying on any of them.
    configure_cli_logging()
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        raise SystemExit(0)
    if not argv:
        print(USAGE, end="", file=sys.stderr)
        raise SystemExit(2)
    command, rest = argv[0], argv[1:]
    if command == "server":
        import fire

        from lenslapse.server import main as server_main

        # the CLI is the friendly path: open the app unless explicitly declined
        if "--no-open" in rest:
            rest = [a for a in rest if a != "--no-open"]
        elif "--open" not in rest:
            rest = [*rest, "--open"]
        fire.Fire(server_main, command=rest)
    elif command in ("probe", "trace", "models"):
        import fire

        from lenslapse.client import COMMANDS

        fire.Fire(COMMANDS[command], command=rest)
    elif command == "add-model":
        import fire

        from lenslapse.add_model import main as add_model_main

        fire.Fire(add_model_main, command=rest)
    else:
        print(f"unknown command {command!r}\n{USAGE}", end="", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
