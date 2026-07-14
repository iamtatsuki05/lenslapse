"""`lenslapse` command-line entry point.

Designed for people who just want to point the hosted web app at their own models:

    pip install "git+https://github.com/iamtatsuki05/lenslapse.git"
    lenslapse server        # starts the probe server and opens the app in the browser

Subcommands delegate to the underlying modules, so every flag they document works here too
(`lenslapse server --port 9000`, `lenslapse add-model --model gpt2 ...`).
"""

import sys

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
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        raise SystemExit(0)
    if not argv:
        print(USAGE, end="", file=sys.stderr)
        raise SystemExit(2)
    command, rest = argv[0], argv[1:]
    if command == "server":
        from .server import main as server_main

        # the CLI is the friendly path: open the app unless explicitly declined
        if "--no-open" in rest:
            rest = [a for a in rest if a != "--no-open"]
        elif "--open" not in rest:
            rest = [*rest, "--open"]
        sys.argv = ["lenslapse server", *rest]
        server_main()
    elif command in ("probe", "trace", "models"):
        from .client import main as client_main

        sys.argv = ["lenslapse", command, *rest]
        client_main()
    elif command == "add-model":
        from .add_model import main as add_model_main

        sys.argv = ["lenslapse add-model", *rest]
        add_model_main()
    else:
        print(f"unknown command {command!r}\n{USAGE}", end="", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
