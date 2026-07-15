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

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import fire
from pydantic import BaseModel, field_validator

# sources.py has no torch/fastapi import — cheap to import eagerly even in pure-HTTP mode,
# unlike `server`, which LocalBackend only imports lazily (see its __init__).
from lenslapse.logging_utils import configure_cli_logging
from lenslapse.sources import DType, Mode, coerce_fire_csv_arg

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "http://localhost:8017"
_T = TypeVar("_T")


def _eprint(message: str) -> None:
    """Progress/status message, not a command's result — logged (stderr by default), so piping
    `--json` output to another tool (e.g. `| jq`) never sees anything but the actual payload."""
    logger.info(message)


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

    def __init__(self, device_map: bool = False, dtype: DType = "float32") -> None:
        from lenslapse import server  # deferred: imports torch, which HTTP mode must not pay for

        self._server = server
        server.STATE["registry_file"] = server._STATE_HOME / "registry.json"
        server.STATE["registry"] = server.build_registry(
            server.default_models_json(), server.STATE["registry_file"], []
        )
        server.STATE["cache_dir"] = server._STATE_HOME / "probe-cache"
        server.STATE["cache_dir"].mkdir(parents=True, exist_ok=True)
        server.STATE["max_loaded"] = 1
        server.STATE["dtype"] = dtype
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
        # server.py types this response richly (ModelListEntry) for its own callers; this CLI
        # boundary deliberately treats it as a generic JSON payload, same as HttpBackend
        return [dict(m) for m in self._server.list_models()]

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        return dict(self._call(self._server.register_model, self._server.RegisterRequest(**payload)))

    def remove(self, model_id: str) -> dict[str, Any]:
        return dict(self._call(self._server.unregister_model, model_id))

    def convert_start(self, model_id: str) -> dict[str, Any]:
        entry = self._server.STATE["registry"].get(model_id)
        ref = entry.ref if entry else "<hf-id-or-path>"
        raise SystemExit(
            "ONNX conversion runs as a background job of `lenslapse server` — start one and retry, or run\n"
            f"  lenslapse add-model --model {ref} --id {model_id}"
        )

    def convert_status(self, model_id: str) -> dict[str, Any]:
        return self.convert_start(model_id)  # same guidance either way


class ConnectionConfig(BaseModel):
    """Which backend to drive — shared by every command; see `probe`'s docstring for each field."""

    server: str | None = None
    local: bool = False
    device_map: bool = False
    dtype: DType = "float32"


def choose_backend(conn: ConnectionConfig) -> Backend:
    if conn.server:
        base = str(conn.server).rstrip("/")
        if "://" not in base:
            base = f"http://{base}"  # `--server localhost:9000` must not read as an unreachable server
        if not server_alive(base):
            raise SystemExit(f"no probe server responding at {base}")
        return HttpBackend(base)
    if not conn.local and server_alive(DEFAULT_SERVER):
        _eprint(f"using the probe server at {DEFAULT_SERVER} (pass --local to run in-process instead)")
        return HttpBackend(DEFAULT_SERVER)
    if not conn.local:
        _eprint(f"no probe server at {DEFAULT_SERVER} — running in-process")
    return LocalBackend(device_map=conn.device_map, dtype=conn.dtype)


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


def infer_mode(ref: str, steps: str | None) -> Mode:
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


class ProbeCliConfig(ConnectionConfig):
    """Validated arguments for `probe`; see `probe`'s docstring for what each means."""

    model: str
    text: str
    step: int | None = None
    pos: int | None = None
    targets: str | None = None
    # named json_output, not json: pydantic.BaseModel already defines a (deprecated) .json() method,
    # and mypy rejects a field that overrides a method with an incompatible type.
    json_output: bool = False

    _coerce_targets = field_validator("targets", mode="before")(coerce_fire_csv_arg)


def cmd_probe(backend: Backend, cfg: ProbeCliConfig) -> None:
    entry = model_entry(backend, cfg.model)
    steps: list[int] = entry["steps"]
    step = cfg.step
    if step is None:
        if not steps:
            raise SystemExit(f"{cfg.model!r} has no checkpoints (local directory moved?)")
        step = steps[-1]
        _eprint(f"probing the last checkpoint (step {step:,}); pass --step to pick another")
    elif step not in steps:
        if entry["mode"] == "suite":
            # the registered grid is a subset of the hub's step{N} revisions — let the server try
            _eprint(f"note: step {step:,} is outside {cfg.model!r}'s registered grid; trying that hub revision")
        else:
            raise SystemExit(f"{cfg.model!r} has no checkpoint for step {step}; available: {_fmt_steps(steps)}")
    res = backend.probe(cfg.model, step, cfg.text, parse_targets(cfg.targets))
    if cfg.json_output:
        print(json.dumps(res, ensure_ascii=False))
        return
    grid = res["grid"]
    pos = _checked_pos(cfg.pos, grid["positions"])
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


class TraceCliConfig(ConnectionConfig):
    """Validated arguments for `trace`; see `trace`'s docstring for what each means."""

    model: str
    text: str
    pos: int | None = None
    layer: int | None = None
    top: int = 3
    targets: str | None = None
    # named json_output, not json: pydantic.BaseModel already defines a (deprecated) .json() method,
    # and mypy rejects a field that overrides a method with an incompatible type.
    json_output: bool = False

    _coerce_targets = field_validator("targets", mode="before")(coerce_fire_csv_arg)


def cmd_trace(backend: Backend, cfg: TraceCliConfig) -> None:
    steps: list[int] = model_entry(backend, cfg.model)["steps"]
    if len(steps) < 2:
        raise SystemExit(
            f"{cfg.model!r} has {len(steps)} checkpoint(s); tracing needs a suite — use `lenslapse probe`"
        )
    if not 1 <= cfg.top <= 10:
        raise SystemExit(f"--top must be 1..10 (the server returns 10 candidates per cell), got {cfg.top}")
    ids = parse_targets(cfg.targets)
    if ids is None:
        # same convention as the UI (and the precomputed shards): fix the tracked tokens from
        # the FINAL checkpoint — final-layer top-k at the traced position
        _eprint(f"fixing targets from the final checkpoint (step {steps[-1]:,})…")
        final = backend.probe(cfg.model, steps[-1], cfg.text)
        final_pos = _checked_pos(cfg.pos, final["grid"]["positions"])
        ids = [i for _, _, i in final["grid"]["cells"][-1][final_pos]["top"][: cfg.top]]

    labels: list[str] = []
    widths: list[int] = []
    collected: list[dict[str, Any]] = []
    layer = cfg.layer
    pos = cfg.pos  # re-checked against the first result below (grid shape is constant across steps)
    for n, st in enumerate(steps, start=1):
        _eprint(f"step {st:>9,}  ({n}/{len(steps)})")
        res = backend.probe(cfg.model, st, cfg.text, ids)
        tgt = res["tgt"]
        if not labels:
            grid = res["grid"]
            pos = _checked_pos(pos, grid["positions"])
            layer = grid["layers"] - 1 if layer is None else layer
            if not 0 <= layer < grid["layers"]:
                raise SystemExit(f"--layer must be 0..{grid['layers'] - 1} for this model, got {layer}")
            labels = [tgt[str(i)]["token"] for i in ids]
            widths = [max(len(repr(lab)), 16) for lab in labels]
            if not cfg.json_output:
                where = f"layer {layer}, position {pos} ({res['tokens'][pos]!r})"
                print(f"{res['model']} · “{cfg.text}” · {len(steps)} checkpoints · {where}")
                print(f"{'step':>10}  " + "  ".join(repr(lab).ljust(w) for lab, w in zip(labels, widths)))
        if not cfg.json_output:
            cells = [f"{tgt[str(i)]['p'][layer][pos]:.4f} (r{tgt[str(i)]['r'][layer][pos]})" for i in ids]
            print(f"{st:>10,}  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))
        collected.append({"step": st, "tgt": tgt, "cached": res.get("cached", False)})
    if cfg.json_output:
        payload = {
            "model": cfg.model,
            "text": cfg.text,
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


def probe(
    model: str,
    text: str,
    step: int | None = None,
    pos: int | None = None,
    targets: str | None = None,
    # named `json`, not `json_output`: must match the `--json` flag fire derives from it. Shadows
    # the `json` module only inside this function's scope; it never calls json.dumps/loads
    # itself (that lives in cmd_probe, via cfg.json_output) — keep it that way.
    json: bool = False,
    server: str | None = None,
    local: bool = False,
    device_map: bool = False,
    dtype: DType = "float32",
) -> None:
    """One logit-lens pass — the UI's "Live probe" button.

    Args:
        model: registered model id (see `lenslapse models`).
        text: prompt text.
        step: checkpoint step (default: the model's last).
        pos: token position to print (default: last).
        targets: comma-separated token ids to track exactly.
        json: print the full probe payload (what the web app receives).
        server: probe server to drive (default: auto-detect http://localhost:8017, else run in-process).
        local: run in-process even if a probe server is running.
        device_map: in-process only: load with device_map='auto'.
        dtype: in-process only: compute dtype (float32 matches the shards; lower halves memory).
    """
    cfg = ProbeCliConfig(
        model=model,
        text=text,
        step=step,
        pos=pos,
        targets=targets,
        json_output=json,
        server=server,
        local=local,
        device_map=device_map,
        dtype=dtype,
    )
    backend = choose_backend(cfg)
    cmd_probe(backend, cfg)


def trace(
    model: str,
    text: str,
    pos: int | None = None,
    layer: int | None = None,
    top: int = 3,
    targets: str | None = None,
    # named `json`, not `json_output`: must match the `--json` flag fire derives from it. Shadows
    # the `json` module only inside this function's scope; it never calls json.dumps/loads
    # itself (that lives in cmd_trace, via cfg.json_output) — keep it that way.
    json: bool = False,
    server: str | None = None,
    local: bool = False,
    device_map: bool = False,
    dtype: DType = "float32",
) -> None:
    """Probe every checkpoint — the UI's "trace across training" button.

    Args:
        model: registered model id (see `lenslapse models`).
        text: prompt text.
        pos: token position to trace (default: last).
        layer: layer whose probability the table shows (default: final).
        top: how many final-checkpoint top tokens to track (default: 3).
        targets: track these token ids instead of the final-checkpoint top-k.
        json: print the whole trajectory as JSON (table goes away).
        server: probe server to drive (default: auto-detect http://localhost:8017, else run in-process).
        local: run in-process even if a probe server is running.
        device_map: in-process only: load with device_map='auto'.
        dtype: in-process only: compute dtype (float32 matches the shards; lower halves memory).
    """
    cfg = TraceCliConfig(
        model=model,
        text=text,
        pos=pos,
        layer=layer,
        top=top,
        targets=targets,
        json_output=json,
        server=server,
        local=local,
        device_map=device_map,
        dtype=dtype,
    )
    backend = choose_backend(cfg)
    cmd_trace(backend, cfg)


class ModelsAddCliConfig(ConnectionConfig):
    """Validated arguments for `models add`; see `ModelsCommands.add`'s docstring for what each means."""

    ref: str
    id: str
    label: str | None = None
    mode: Mode | None = None
    steps: str | None = None

    _coerce_steps = field_validator("steps", mode="before")(coerce_fire_csv_arg)


class ModelsIdCliConfig(ConnectionConfig):
    """Validated arguments for `models remove`/`models convert`: a model id + connection flags."""

    id: str


class ModelsCommands:
    """`lenslapse models [list|add|remove|convert]` — the UI's "models" dialog."""

    def __call__(
        self,
        server: str | None = None,
        local: bool = False,
        device_map: bool = False,
        dtype: DType = "float32",
    ) -> None:
        """Bare `lenslapse models` (no subcommand) shows every registered model, same as `list`."""
        self.list(server=server, local=local, device_map=device_map, dtype=dtype)

    def list(
        self,
        server: str | None = None,
        local: bool = False,
        device_map: bool = False,
        dtype: DType = "float32",
    ) -> None:
        """Show every registered model.

        Args:
            server: probe server to drive (default: auto-detect http://localhost:8017, else run in-process).
            local: run in-process even if a probe server is running.
            device_map: in-process only: load with device_map='auto'.
            dtype: in-process only: compute dtype (float32 matches the shards; lower halves memory).
        """
        cfg = ConnectionConfig(server=server, local=local, device_map=device_map, dtype=dtype)
        backend = choose_backend(cfg)
        entries = backend.models()
        if not entries:
            print("no models registered")
            return
        for m in entries:
            steps = m["steps"]
            span = f"{len(steps)} step" + ("s" if len(steps) != 1 else "")
            print(f"{m['id']:<24} {m['mode']:<6} {span:>9}  {m['origin']:<8} {m['ref']}")

    def add(
        self,
        ref: str,
        id: str,
        label: str | None = None,
        mode: Mode | None = None,
        steps: str | None = None,
        server: str | None = None,
        local: bool = False,
        device_map: bool = False,
        dtype: DType = "float32",
    ) -> None:
        """Register a model (HF id or a folder on the server machine).

        Args:
            ref: HF id or local checkpoint directory.
            id: id the app shows in its model picker.
            label: display label; defaults to `id`.
            mode: default: local if `ref` is a directory, suite if `steps` is given, else final.
            steps: suite only: comma-separated step numbers.
            server: probe server to drive (default: auto-detect http://localhost:8017, else run in-process).
            local: run in-process even if a probe server is running.
            device_map: in-process only: load with device_map='auto'.
            dtype: in-process only: compute dtype (float32 matches the shards; lower halves memory).
        """
        cfg = ModelsAddCliConfig(
            ref=ref,
            id=id,
            label=label,
            mode=mode,
            steps=steps,
            server=server,
            local=local,
            device_map=device_map,
            dtype=dtype,
        )
        backend = choose_backend(cfg)
        resolved_mode = cfg.mode or infer_mode(cfg.ref, cfg.steps)
        payload: dict[str, Any] = {"id": cfg.id, "ref": cfg.ref, "mode": resolved_mode}
        if cfg.label:
            payload["label"] = cfg.label
        if cfg.steps:
            try:
                payload["steps"] = [int(s) for s in cfg.steps.split(",") if s.strip()]
            except ValueError as e:
                raise SystemExit(f"--steps expects comma-separated step numbers, got {cfg.steps!r}") from e
        created = backend.add(payload)
        print(f"registered {created['id']} ({created['mode']}) — steps: {_fmt_steps(created['steps'])}")

    def remove(
        self,
        id: str,
        server: str | None = None,
        local: bool = False,
        device_map: bool = False,
        dtype: DType = "float32",
    ) -> None:
        """Unregister a model.

        Args:
            id: registered model id.
            server: probe server to drive (default: auto-detect http://localhost:8017, else run in-process).
            local: run in-process even if a probe server is running.
            device_map: in-process only: load with device_map='auto'.
            dtype: in-process only: compute dtype (float32 matches the shards; lower halves memory).
        """
        cfg = ModelsIdCliConfig(id=id, server=server, local=local, device_map=device_map, dtype=dtype)
        backend = choose_backend(cfg)
        backend.remove(cfg.id)
        print(f"removed {cfg.id}")

    def convert(
        self,
        id: str,
        server: str | None = None,
        local: bool = False,
        device_map: bool = False,
        dtype: DType = "float32",
    ) -> None:
        """ONNX-convert a registered model for in-browser use.

        Args:
            id: registered model id.
            server: probe server to drive (default: auto-detect http://localhost:8017, else run in-process).
            local: run in-process even if a probe server is running.
            device_map: in-process only: load with device_map='auto'.
            dtype: in-process only: compute dtype (float32 matches the shards; lower halves memory).
        """
        cfg = ModelsIdCliConfig(id=id, server=server, local=local, device_map=device_map, dtype=dtype)
        backend = choose_backend(cfg)
        backend.convert_start(cfg.id)
        _eprint(f"converting {cfg.id} on the server (this exports ONNX per checkpoint — minutes, not seconds)")
        last = ""
        while True:
            time.sleep(3)
            status = backend.convert_status(cfg.id)
            log = list(status.get("log") or [])
            if log and log[-1] != last:
                last = log[-1]
                _eprint(f"  {last}")
            if status["status"] != "running":
                break
        if status["status"] != "done":
            raise SystemExit("conversion failed:\n  " + "\n  ".join(log))
        print(f"converted {cfg.id}" + (f" — {status['note']}" if status.get("note") else ""))


# lenslapse probe/trace/models — dispatched individually by cli.py (`fire.Fire(COMMANDS[cmd], ...)`)
# and, when this module runs standalone, all together via `fire.Fire(COMMANDS)` below.
COMMANDS: dict[str, Any] = {"probe": probe, "trace": trace, "models": ModelsCommands()}


if __name__ == "__main__":
    configure_cli_logging()
    fire.Fire(COMMANDS)
