"""Terminal access to everything the web app's UI can do.

Every interactive feature has a scriptable counterpart:

  lenslapse models add --ref /path/to/trainer_output --id my-run   # the "models" dialog
  lenslapse probe --model my-run --text "The capital of France is" # the "Live probe" button
  lenslapse trace --model my-run --text "The capital of France is" # the "trace across training" button
  lenslapse models convert my-run                                  # the "convert to ONNX" button

Commands drive a running `lenslapse server` when one is reachable (default port 8017), sharing
its registry, loaded weights, and probe cache; otherwise the same code runs in-process, writing
to the same on-disk state. Either way a result computed here replays instantly in the app's
"live - server" mode, and vice versa. `--json` prints the exact payload the web app receives.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

DEFAULT_SERVER = "http://localhost:8017"
_T = TypeVar("_T")


def _eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class Backend(Protocol):
    """What a command needs from either a running server or an in-process fallback."""

    where: str

    def probe(self, model: str, step: int, text: str, targets: list[int] | None = None) -> dict[str, Any]: ...

    def models(self) -> list[dict[str, Any]]: ...

    def add(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def remove(self, model_id: str) -> dict[str, Any]: ...

    def convert_start(self, model_id: str) -> dict[str, Any]: ...

    def convert_status(self, model_id: str) -> dict[str, Any]: ...


def server_alive(base: str) -> bool:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=2):
            return True
    except (OSError, ValueError):  # ValueError: malformed --server URL
        return False


class HttpBackend:
    """Drives a running probe server over HTTP — exactly what the web app does."""

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.where = self.base

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float | None = 30
    ) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read())["detail"]
            except Exception:  # noqa: BLE001 — any unparseable error body falls back to the status line
                detail = f"HTTP {e.code} {e.reason}"
            raise SystemExit(f"error: {detail}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"cannot reach the probe server at {self.base} ({e.reason})") from e

    def probe(self, model: str, step: int, text: str, targets: list[int] | None = None) -> dict[str, Any]:
        body = {"model": model, "step": step, "text": text, "targets": targets}
        # no read timeout: the first probe of a checkpoint may download its weights
        return dict(self._request("POST", "/probe", body, timeout=None))

    def models(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/models"))

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._request("POST", "/models", payload))

    def remove(self, model_id: str) -> dict[str, Any]:
        return dict(self._request("DELETE", f"/models/{urllib.parse.quote(model_id, safe='')}"))

    def convert_start(self, model_id: str) -> dict[str, Any]:
        return dict(self._request("POST", f"/models/{urllib.parse.quote(model_id, safe='')}/convert"))

    def convert_status(self, model_id: str) -> dict[str, Any]:
        return dict(self._request("GET", f"/models/{urllib.parse.quote(model_id, safe='')}/convert"))


class LocalBackend:
    """Runs the same code paths as the probe server, in this process (no HTTP, no second port).

    Shares the server's on-disk state at the default locations — registry file and probe
    cache — so a model registered here appears in the web dialog after the next server start,
    and a probe computed here replays instantly in the browser (and vice versa). Registry
    writes are last-writer-wins against a concurrently running server: prefer driving the
    server over HTTP (the default) for add/remove while one is up."""

    where = "in-process"

    def __init__(self, device_map: bool = False) -> None:
        from . import server  # deferred: imports torch, which HTTP mode must not pay for

        self._server = server
        server.STATE["registry_file"] = server._STATE_HOME / "registry.json"
        server.STATE["registry"] = server.build_registry(
            server.default_models_json(), server.STATE["registry_file"], []
        )
        server.STATE["cache_dir"] = server._STATE_HOME / "probe-cache"
        server.STATE["cache_dir"].mkdir(parents=True, exist_ok=True)
        server.STATE["max_loaded"] = 1
        server.STATE["device_map"] = device_map

    def _call(self, fn: Callable[..., _T], *args: Any) -> _T:
        from fastapi import HTTPException

        try:
            return fn(*args)
        except HTTPException as e:
            raise SystemExit(f"error: {e.detail}") from e

    def probe(self, model: str, step: int, text: str, targets: list[int] | None = None) -> dict[str, Any]:
        req = self._server.ProbeRequest(model=model, step=step, text=text, targets=targets)
        return self._call(self._server.probe, req)

    def models(self) -> list[dict[str, Any]]:
        return self._server.list_models()

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._call(self._server.register_model, self._server.RegisterRequest(**payload))

    def remove(self, model_id: str) -> dict[str, Any]:
        return self._call(self._server.unregister_model, model_id)

    def convert_start(self, model_id: str) -> dict[str, Any]:
        entry = self._server.STATE["registry"].get(model_id)
        ref = entry.ref if entry else "<hf-id-or-path>"
        raise SystemExit(
            "ONNX conversion runs as a background job of `lenslapse server` — start one and retry, or run\n"
            f"  lenslapse add-model --model {ref} --id {model_id}"
        )

    def convert_status(self, model_id: str) -> dict[str, Any]:
        return self.convert_start(model_id)  # same guidance either way


def choose_backend(args: argparse.Namespace) -> Backend:
    if args.server:
        base = str(args.server).rstrip("/")
        if "://" not in base:
            base = f"http://{base}"  # `--server localhost:9000` must not read as an unreachable server
        if not server_alive(base):
            raise SystemExit(f"no probe server responding at {base}")
        return HttpBackend(base)
    if not args.local and server_alive(DEFAULT_SERVER):
        _eprint(f"using the probe server at {DEFAULT_SERVER} (pass --local to run in-process instead)")
        return HttpBackend(DEFAULT_SERVER)
    if not args.local:
        _eprint(f"no probe server at {DEFAULT_SERVER} — running in-process")
    return LocalBackend(device_map=args.device_map)


def parse_targets(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    try:
        ids = [int(part) for part in spec.split(",") if part.strip()]
    except ValueError as e:
        raise SystemExit(f"--targets expects comma-separated token ids (see probe output), got {spec!r}") from e
    if not ids:
        raise SystemExit("--targets got an empty list")
    deduped: list[int] = []
    for tid in ids:
        if tid not in deduped:
            deduped.append(tid)
    return deduped


def infer_mode(ref: str, steps: str | None) -> str:
    """Mirror the registration dialog's radio buttons without making the user pick one."""
    if Path(ref).expanduser().is_dir():
        return "local"
    return "suite" if steps else "final"


def model_entry(backend: Backend, model_id: str) -> dict[str, Any]:
    for m in backend.models():
        if m["id"] == model_id:
            return m
    raise SystemExit(f"unknown model id {model_id!r} — see `lenslapse models`")


def _checked_pos(pos_arg: int | None, positions: int) -> int:
    pos = positions - 1 if pos_arg is None else pos_arg
    if not 0 <= pos < positions:
        raise SystemExit(f"--pos must be 0..{positions - 1} for this prompt, got {pos}")
    return pos


def cmd_probe(backend: Backend, args: argparse.Namespace) -> None:
    entry = model_entry(backend, args.model)
    steps: list[int] = entry["steps"]
    step = args.step
    if step is None:
        if not steps:
            raise SystemExit(f"{args.model!r} has no checkpoints (local directory moved?)")
        step = steps[-1]
        _eprint(f"probing the last checkpoint (step {step:,}); pass --step to pick another")
    elif step not in steps:
        if entry["mode"] == "suite":
            # the registered grid is a subset of the hub's step{N} revisions — let the server try
            _eprint(f"note: step {step:,} is outside {args.model!r}'s registered grid; trying that hub revision")
        else:
            raise SystemExit(f"{args.model!r} has no checkpoint for step {step}; available: {_fmt_steps(steps)}")
    res = backend.probe(args.model, step, args.text, parse_targets(args.targets))
    if args.json:
        print(json.dumps(res, ensure_ascii=False))
        return
    grid = res["grid"]
    pos = _checked_pos(args.pos, grid["positions"])
    tokens = res["tokens"]
    head = f"{res['model']} · step {res['step']:,} · {res['device']} · forward {res['timing']['forward']} ms"
    print(head + (" · replayed from cache" if res.get("cached") else ""))
    print(f"tokens ({len(tokens)}):", " ".join(repr(t) for t in tokens))
    print(f"top-1 by layer at position {pos} ({tokens[pos]!r}):")
    for li, row in enumerate(grid["cells"]):
        cell = row[pos]
        print(f"  {'emb' if li == 0 else f'L{li}':>4}  {cell['prob']:6.4f}  {cell['token']!r}")
    top5 = grid["cells"][-1][pos]["top"][:5]
    print("final-layer top-5:", " · ".join(f"{t!r} {p:.4f} (id {i})" for t, p, i in top5))
    for tid, t in (res.get("tgt") or {}).items():
        print(f"target {t['token']!r} (id {tid}): p={t['p'][-1][pos]:.4f} rank={t['r'][-1][pos]} at the final layer")


def cmd_trace(backend: Backend, args: argparse.Namespace) -> None:
    steps: list[int] = model_entry(backend, args.model)["steps"]
    if len(steps) < 2:
        raise SystemExit(
            f"{args.model!r} has {len(steps)} checkpoint(s); tracing needs a suite — use `lenslapse probe`"
        )
    if not 1 <= args.top <= 10:
        raise SystemExit(f"--top must be 1..10 (the server returns 10 candidates per cell), got {args.top}")
    ids = parse_targets(args.targets)
    if ids is None:
        # same convention as the UI (and the precomputed shards): fix the tracked tokens from
        # the FINAL checkpoint — final-layer top-k at the traced position
        _eprint(f"fixing targets from the final checkpoint (step {steps[-1]:,})…")
        final = backend.probe(args.model, steps[-1], args.text)
        pos = _checked_pos(args.pos, final["grid"]["positions"])
        ids = [i for _, _, i in final["grid"]["cells"][-1][pos]["top"][: args.top]]

    labels: list[str] = []
    widths: list[int] = []
    collected: list[dict[str, Any]] = []
    layer = args.layer
    pos = args.pos  # re-checked against the first result below (grid shape is constant across steps)
    for n, st in enumerate(steps, start=1):
        _eprint(f"step {st:>9,}  ({n}/{len(steps)})")
        res = backend.probe(args.model, st, args.text, ids)
        tgt = res["tgt"]
        if not labels:
            grid = res["grid"]
            pos = _checked_pos(pos, grid["positions"])
            layer = grid["layers"] - 1 if layer is None else layer
            if not 0 <= layer < grid["layers"]:
                raise SystemExit(f"--layer must be 0..{grid['layers'] - 1} for this model, got {layer}")
            labels = [tgt[str(i)]["token"] for i in ids]
            widths = [max(len(repr(lab)), 16) for lab in labels]
            if not args.json:
                where = f"layer {layer}, position {pos} ({res['tokens'][pos]!r})"
                print(f"{res['model']} · “{args.text}” · {len(steps)} checkpoints · {where}")
                print(f"{'step':>10}  " + "  ".join(repr(lab).ljust(w) for lab, w in zip(labels, widths)))
        if not args.json:
            cells = [f"{tgt[str(i)]['p'][layer][pos]:.4f} (r{tgt[str(i)]['r'][layer][pos]})" for i in ids]
            print(f"{st:>10,}  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        collected.append({"step": st, "tgt": tgt, "cached": res.get("cached", False)})
    if args.json:
        payload = {
            "model": args.model,
            "text": args.text,
            "layer": layer,
            "pos": pos,
            "targets": [{"id": i, "token": lab} for i, lab in zip(ids, labels)],
            "steps": collected,
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        _eprint(f"traced {len(steps)} checkpoints — the app replays these instantly from the shared probe cache")


def _fmt_steps(steps: list[int]) -> str:
    if not steps:
        return "none"
    if len(steps) <= 6:
        return ", ".join(f"{s:,}" for s in steps)
    return f"{steps[0]:,} … {steps[-1]:,} ({len(steps)} checkpoints)"


def cmd_models(backend: Backend, args: argparse.Namespace) -> None:
    action = getattr(args, "action", None) or "list"
    if action == "list":
        entries = backend.models()
        if not entries:
            print("no models registered")
            return
        for m in entries:
            steps = m["steps"]
            span = f"{len(steps)} step" + ("s" if len(steps) != 1 else "")
            print(f"{m['id']:<24} {m['mode']:<6} {span:>9}  {m['origin']:<8} {m['ref']}")
    elif action == "add":
        mode = args.mode or infer_mode(args.ref, args.steps)
        payload: dict[str, Any] = {"id": args.id, "ref": args.ref, "mode": mode}
        if args.label:
            payload["label"] = args.label
        if args.steps:
            try:
                payload["steps"] = [int(s) for s in args.steps.split(",") if s.strip()]
            except ValueError as e:
                raise SystemExit(f"--steps expects comma-separated step numbers, got {args.steps!r}") from e
        created = backend.add(payload)
        print(f"registered {created['id']} ({created['mode']}) — steps: {_fmt_steps(created['steps'])}")
    elif action == "remove":
        backend.remove(args.id)
        print(f"removed {args.id}")
    elif action == "convert":
        backend.convert_start(args.id)
        _eprint(f"converting {args.id} on the server (this exports ONNX per checkpoint — minutes, not seconds)")
        last = ""
        while True:
            time.sleep(3)
            status = backend.convert_status(args.id)
            log = list(status.get("log") or [])
            if log and log[-1] != last:
                last = log[-1]
                _eprint(f"  {last}")
            if status["status"] != "running":
                break
        if status["status"] != "done":
            raise SystemExit("conversion failed:\n  " + "\n  ".join(log))
        print(f"converted {args.id}" + (f" — {status['note']}" if status.get("note") else ""))


def main() -> None:
    conn = argparse.ArgumentParser(add_help=False)
    conn.add_argument(
        "--server",
        metavar="URL",
        default=None,
        help=f"probe server to drive (default: auto-detect {DEFAULT_SERVER}, else run in-process)",
    )
    conn.add_argument("--local", action="store_true", help="run in-process even if a probe server is running")
    conn.add_argument("--device-map", action="store_true", help="in-process only: load with device_map='auto'")

    ap = argparse.ArgumentParser(
        prog="lenslapse", description="Terminal access to everything the web app's UI can do."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe", parents=[conn], help='one logit-lens pass — the UI\'s "Live probe" button')
    pp.add_argument("--model", required=True, help="registered model id (see `lenslapse models`)")
    pp.add_argument("--text", required=True)
    pp.add_argument("--step", type=int, default=None, help="checkpoint step (default: the model's last)")
    pp.add_argument("--pos", type=int, default=None, help="token position to print (default: last)")
    pp.add_argument("--targets", default=None, help="comma-separated token ids to track exactly")
    pp.add_argument("--json", action="store_true", help="print the full probe payload (what the web app receives)")

    pt = sub.add_parser(
        "trace", parents=[conn], help='probe every checkpoint — the UI\'s "trace across training" button'
    )
    pt.add_argument("--model", required=True, help="registered model id (see `lenslapse models`)")
    pt.add_argument("--text", required=True)
    pt.add_argument("--pos", type=int, default=None, help="token position to trace (default: last)")
    pt.add_argument("--layer", type=int, default=None, help="layer whose probability the table shows (default: final)")
    pt.add_argument("--top", type=int, default=3, help="how many final-checkpoint top tokens to track (default: 3)")
    pt.add_argument("--targets", default=None, help="track these token ids instead of the final-checkpoint top-k")
    pt.add_argument("--json", action="store_true", help="print the whole trajectory as JSON (table goes away)")

    pm = sub.add_parser("models", help='list/add/remove/convert models — the UI\'s "models" dialog')
    pm.set_defaults(action=None, server=None, local=False, device_map=False)
    msub = pm.add_subparsers(dest="action")
    msub.add_parser("list", parents=[conn], help="show every registered model")
    ma = msub.add_parser("add", parents=[conn], help="register a model (HF id or a folder on the server machine)")
    ma.add_argument("--ref", required=True, help="HF id or local checkpoint directory")
    ma.add_argument("--id", required=True, help="id the app shows in its model picker")
    ma.add_argument("--label", default=None)
    ma.add_argument(
        "--mode",
        choices=["suite", "final", "local"],
        default=None,
        help="default: local if --ref is a directory, suite if --steps is given, else final",
    )
    ma.add_argument("--steps", default=None, help="suite only: comma-separated step numbers")
    mr = msub.add_parser("remove", parents=[conn], help="unregister a model")
    mr.add_argument("id")
    mc = msub.add_parser("convert", parents=[conn], help="ONNX-convert a registered model for in-browser use")
    mc.add_argument("id")

    args = ap.parse_args()
    backend = choose_backend(args)
    if args.cmd == "probe":
        cmd_probe(backend, args)
    elif args.cmd == "trace":
        cmd_trace(backend, args)
    else:
        cmd_models(backend, args)


if __name__ == "__main__":
    main()
