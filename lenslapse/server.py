"""Optional local probe server for models too heavy for in-browser inference.

The logit lens needs the *entire per-layer residual stream* of one teacher-forced forward
pass — a batch-1 workload with no decoding loop, which generation-oriented serving engines
neither expose natively nor accelerate. This server therefore runs plain transformers on
CUDA/MPS/CPU, reusing the exact hooked-forward + lens implementation that generates the
precomputed shards — the numbers agree by construction.

Every probe result is persisted to --cache-dir keyed by (model ref, revision, prompt), and
identical requests are replayed from disk, byte-for-byte: results are reproducible across
sessions and auditable as plain JSON files. Note the key is the *reference*, not the weights:
for mutable refs (revision "main", local run directories) a re-trained model under the same
path keeps replaying the old cached results — clear the cache directory after retraining.

Usage:
  lenslapse server                  (pip install; serves the bundled app + API on one port)
  uv run lenslapse server           (repo checkout; serves the fresh web/dist build)

The registry defaults to the app's models.json (hub suites with step{N} revisions); extend it
with --extra for local runs or heavy hub models, e.g.:
  --extra my-run=/path/to/trainer_output --extra llama=meta-llama/Llama-3.2-1B:final

Models can also be registered at runtime from the web app's "models" dialog (GET/POST/DELETE
/models); user-registered entries persist to --registry-file and survive restarts. This is a
management API for a localhost tool: anyone who can reach the server can register any Hub model
or any directory readable by the server process, so do not expose the port beyond your machine.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from huggingface_hub import model_info
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from .precompute_lens import lens_all
from .sources import CheckpointSource, resolve_sources

TOPK = 10
MAX_TOKENS = 64
DEFAULT_SUITE_STEPS = [
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    64000,
    128000,
    143000,
]
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

app = FastAPI(title="LensLapse probe server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def allow_private_network(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    # Chrome's Private Network Access: an HTTPS page (e.g. the GitHub Pages deployment) probing
    # this localhost server sends a preflight with Access-Control-Request-Private-Network; without
    # this response header the browser refuses (or stalls) the request.
    response = await call_next(request)
    if request.method == "OPTIONS" and request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


STATE: dict = {"registry": {}, "cache_dir": None, "max_loaded": 1, "device_map": False, "registry_file": None}
LOADED: OrderedDict = OrderedDict()  # (ref, revision) -> (model, tokenizer)
# FastAPI runs sync endpoints in a threadpool; lens_all installs forward hooks on the shared
# model, so concurrent probes would capture each other's layer outputs (and persist the garbage
# to the cache). One global lock serializes load+forward+write — fine for a batch-1 local server.
PROBE_LOCK = threading.Lock()
REG_LOCK = threading.Lock()  # registry mutations + registry-file writes
JOBS: dict = {}  # model id -> {"status": "running|done|failed", "log": deque}; conversions run one at a time
JOBS_LOCK = threading.Lock()  # guards JOBS check-and-set (endpoints run concurrently in the threadpool)
PICK_LOCK = threading.Lock()  # at most one native folder dialog at a time


class ProbeRequest(BaseModel):
    model: str
    step: int = 0
    text: str
    # optional token ids to track exactly (probability + rank at every layer/position); lets the
    # app draw training trajectories for live prompts the same way precomputed shards do
    targets: list[int] | None = None


class RegisterRequest(BaseModel):
    id: str
    ref: str  # HF id or a directory on the server machine
    mode: str  # "suite" | "final" | "local" (checked with a 400, not a 422, for a friendlier message)
    label: str | None = None
    steps: list[int] | None = None  # suite only; defaults to the Pythia step grid


class RegistryEntry(BaseModel):
    """One probeable model. Serialized (minus origin) to the registry file."""

    ref: str
    mode: Literal["suite", "final", "local"]
    label: str | None = None
    steps: list[int] | None = None  # suite only
    origin: Literal["catalog", "user"] = "user"


def build_registry(
    models_json: Path | None, registry_file: Path | None, extras: list[str]
) -> dict[str, RegistryEntry]:
    registry: dict[str, RegistryEntry] = {}
    if models_json and models_json.exists():
        for m in json.loads(models_json.read_text())["models"]:
            if m.get("mode", "suite") == "local":
                continue  # local paths are machine-specific; register via --extra id=/path
            # "source" is the true HF ref; "hf" doubles as the app's local tokenizer directory
            # name, which for add_model.py onboarded models is the app id, not a Hub id.
            registry[m["id"]] = RegistryEntry(
                ref=m.get("source", m["hf"]),
                mode=m.get("mode", "suite"),
                label=m.get("label"),
                steps=m.get("steps"),
                origin="catalog",
            )
    if registry_file and registry_file.exists():
        try:
            for mid, entry in json.loads(registry_file.read_text()).items():
                registry[mid] = RegistryEntry(**{**entry, "origin": "user"})
        except (json.JSONDecodeError, ValidationError) as err:
            raise SystemExit(f"{registry_file} is corrupt ({err}); fix or delete it") from err
    for e in extras:
        mid, sep, ref = e.partition("=")
        if not sep or not mid or not ref:
            raise SystemExit(f"--extra expects id=hf_ref[:final] or id=/local/dir, got {e!r}")
        mode: Literal["suite", "final", "local"] = "suite"
        if ref.endswith(":final"):
            ref, mode = ref[: -len(":final")], "final"
        elif Path(ref).is_dir():
            mode = "local"
        registry[mid] = RegistryEntry(ref=ref, mode=mode, origin="user")
    return registry


def save_user_registry() -> None:
    """Persist user-registered entries (dialog/API/--extra) so they survive restarts."""
    path: Path = STATE["registry_file"]
    entries = {
        mid: e.model_dump(exclude={"origin"}, exclude_none=True)
        for mid, e in STATE["registry"].items()
        if e.origin == "user"
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp, path)


def model_steps(entry: RegistryEntry) -> list[int]:
    if entry.mode == "final":
        return [0]
    if entry.mode == "local":
        if not Path(entry.ref).is_dir():
            return []  # directory gone since registration; probing it 404s with an explicit message
        return [src.step for src in resolve_sources(entry.ref, "0", final_only=False)]
    return entry.steps or DEFAULT_SUITE_STEPS


def source_for(model_id: str, step: int) -> CheckpointSource:
    entry = STATE["registry"].get(model_id)
    if entry is None:
        raise HTTPException(404, f"unknown model id {model_id!r}; register it via --extra")
    mode, ref = entry.mode, entry.ref
    if mode == "final":
        return resolve_sources(ref, "0", final_only=True)[0]
    if mode == "local":
        if not Path(ref).is_dir():
            # never fall through to hub-suite resolution for a vanished local path
            raise HTTPException(404, f"{model_id!r}: local directory {ref} no longer exists")
        for src in resolve_sources(ref, "0", final_only=False):
            if src.step == step:
                return src
        raise HTTPException(404, f"{model_id!r} has no local checkpoint for step {step}")
    return resolve_sources(ref, str(step), final_only=False)[0]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(src: CheckpointSource) -> tuple[Any, Any]:
    key = (src.load_ref, src.revision)
    if key in LOADED:
        LOADED.move_to_end(key)
        return LOADED[key]
    # evict before loading so peak memory stays at max_loaded models, not max_loaded+1
    while LOADED and len(LOADED) >= STATE["max_loaded"]:
        old_key, (old_model, _) = LOADED.popitem(last=False)
        del old_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"evicted {old_key}", flush=True)
    tok = AutoTokenizer.from_pretrained(src.load_ref, revision=src.revision)
    if STATE["device_map"]:
        # opt-in: shard very large models across devices via accelerate
        model = AutoModelForCausalLM.from_pretrained(
            src.load_ref, revision=src.revision, dtype="auto", device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(src.load_ref, revision=src.revision, dtype="auto")
        model.to(pick_device())
    model.eval()
    LOADED[key] = (model, tok)
    return LOADED[key]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "models": sorted(STATE["registry"]), "device": pick_device()}


def _folder_dialog_cmd() -> list[str] | None:
    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "LensLapse: select a model / checkpoint folder")'
        return ["osascript", "-e", script]
    if sys.platform.startswith("linux"):
        return ["zenity", "--file-selection", "--directory", "--title=LensLapse: select a model folder"]
    if sys.platform == "win32":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }"
        )
        return ["powershell", "-NoProfile", "-Command", script]
    return None


@app.get("/pick-folder")
def pick_folder() -> dict[str, str]:
    """Open a native folder dialog on the server machine and return the chosen absolute path.

    Lets the dialog's "browse" button fill in local checkpoint paths for people who do not
    want to type them. The dialog opens on the machine running this server (they are the same
    machine in the intended localhost setup)."""
    cmd = _folder_dialog_cmd()
    if cmd is None:
        raise HTTPException(501, "no native folder dialog on this platform — type the path instead")
    if not PICK_LOCK.acquire(blocking=False):
        raise HTTPException(409, "a folder dialog is already open on the server machine")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=300)
    except FileNotFoundError as e:
        raise HTTPException(501, f"folder dialog helper not available ({cmd[0]}) — type the path instead") from e
    except subprocess.TimeoutExpired as e:
        raise HTTPException(408, "folder dialog timed out on the server machine") from e
    finally:
        PICK_LOCK.release()
    path = proc.stdout.strip()
    # distinguish "the user cancelled" from "no dialog could be shown" (headless/SSH sessions):
    # osascript reports cancel as error -128; zenity exits 1; the powershell script prints nothing
    cancelled = (
        "-128" in proc.stderr or (cmd[0] == "zenity" and proc.returncode == 1) or (proc.returncode == 0 and not path)
    )
    if cancelled:
        raise HTTPException(400, "folder selection was cancelled")
    if proc.returncode != 0:
        raise HTTPException(500, f"folder dialog failed on the server machine: {proc.stderr.strip()[-200:]}")
    return {"path": path.rstrip("/") or "/"}


@app.get("/models")
def list_models() -> list[dict[str, Any]]:
    return [
        {
            "id": mid,
            "ref": e.ref,
            "mode": e.mode,
            "label": e.label or mid,
            "steps": model_steps(e),
            "origin": e.origin,
        }
        for mid, e in sorted(STATE["registry"].items())
    ]


def validate_source(req: RegisterRequest) -> None:
    """Fail fast with an actionable message before the entry is persisted."""
    if req.mode == "local":
        path = Path(req.ref)
        if not path.is_dir():
            raise HTTPException(400, f"{req.ref} is not a directory on the server machine")
        if not (path / "config.json").exists() and not list(path.glob("checkpoint-*")):
            raise HTTPException(400, f"{req.ref} has neither config.json nor checkpoint-* subdirectories")
        return
    try:
        if req.mode == "suite":
            steps = req.steps or DEFAULT_SUITE_STEPS
            model_info(req.ref, revision=f"step{steps[0]}")  # revision convention check, one cheap metadata call
        else:
            model_info(req.ref)
    except Exception as e:
        raise HTTPException(400, f"cannot resolve {req.ref!r} on the Hugging Face Hub: {e}") from e


@app.post("/models", status_code=201)
def register_model(req: RegisterRequest) -> dict[str, Any]:
    if not MODEL_ID_RE.fullmatch(req.id):
        raise HTTPException(400, "id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    if req.mode not in ("suite", "final", "local"):
        raise HTTPException(400, f"unknown mode {req.mode!r} (expected suite, final, or local)")
    if req.steps is not None and req.mode != "suite":
        raise HTTPException(400, f"steps only applies to suite mode, not {req.mode!r}")
    if req.mode == "suite" and req.steps is not None and (not req.steps or any(s < 0 for s in req.steps)):
        raise HTTPException(400, "steps must be a non-empty list of non-negative ints")
    if req.mode == "local":
        # store an absolute path: a CWD-relative one would silently stop resolving after a
        # restart from another directory (and ~ would never resolve at all)
        req.ref = str(Path(req.ref).expanduser().resolve())
    validate_source(req)
    with REG_LOCK:
        if req.id in STATE["registry"]:
            raise HTTPException(409, f"model id {req.id!r} is already registered; remove it first")
        entry = RegistryEntry(
            ref=req.ref,
            mode=req.mode,  # type: ignore[arg-type]  # validated to the Literal above
            label=req.label,
            steps=sorted(set(req.steps)) if req.mode == "suite" and req.steps else None,
        )
        STATE["registry"][req.id] = entry
        try:
            save_user_registry()
        except OSError as e:
            del STATE["registry"][req.id]  # keep memory and registry.json consistent
            raise HTTPException(500, f"could not persist registry: {e}") from e
    print(f"registered {req.id} -> {req.ref} ({req.mode})", flush=True)
    return {"id": req.id, **entry.model_dump(exclude={"origin"}, exclude_none=True), "steps": model_steps(entry)}


@app.delete("/models/{model_id}")
def unregister_model(model_id: str) -> dict[str, str]:
    with JOBS_LOCK:
        job = JOBS.get(model_id)
        if job and job["status"] == "running":
            raise HTTPException(409, f"{model_id!r} is being converted; wait for the job to finish")
        JOBS.pop(model_id, None)  # a later re-registration must not inherit this id's old job state
    with REG_LOCK:
        entry = STATE["registry"].get(model_id)
        if entry is None:
            raise HTTPException(404, f"unknown model id {model_id!r}")
        if entry.origin == "catalog":
            raise HTTPException(400, f"{model_id!r} comes from models.json; edit that file instead")
        del STATE["registry"][model_id]
        try:
            save_user_registry()
        except OSError as e:
            STATE["registry"][model_id] = entry  # keep memory and registry.json consistent
            raise HTTPException(500, f"could not persist registry: {e}") from e
    # loaded weights (if any) stay resident until the LRU evicts them; only the registry changes
    print(f"unregistered {model_id}", flush=True)
    return {"removed": model_id}


@app.post("/models/{model_id}/convert", status_code=202)
def convert_model(model_id: str) -> dict[str, str]:
    """Run the full ONNX onboarding pipeline (add_model.py) for a registered model, in the
    background. On success the model graduates into web/public + models.json (browser-runnable
    after the app is rebuilt or served from source), and its dialog registration is retired."""
    entry = STATE["registry"].get(model_id)
    if entry is None:
        raise HTTPException(404, f"unknown model id {model_id!r}")
    if entry.origin == "catalog":
        raise HTTPException(400, f"{model_id!r} is already in models.json (already converted or shipped)")
    cmd = [
        sys.executable,
        "-m", "lenslapse.add_model",
        "--model", entry.ref,
        "--id", model_id,
        f"--label={entry.label or model_id}",  # = form: a label starting with '-' must not read as an option
        "--models-root", str(STATE["models_root"]),
        "--force",  # explicit user action: re-converting the same id overwrites its own artifacts
    ]  # fmt: skip
    if entry.mode == "final":
        cmd.append("--final-only")
    elif entry.mode == "suite":
        cmd += ["--steps", ",".join(str(s) for s in model_steps(entry))]
    with JOBS_LOCK:
        # check-and-set under one lock: concurrent POSTs must not start two export subprocesses
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(409, "another conversion is already running; wait for it to finish")
        job = {"status": "running", "log": deque(maxlen=50)}
        JOBS[model_id] = job
    threading.Thread(target=_run_convert, args=(model_id, job, cmd), daemon=True).start()
    print(f"converting {model_id}: {' '.join(cmd)}", flush=True)
    return {"id": model_id, "status": "running"}


def _run_convert(model_id: str, job: dict, cmd: list[str]) -> None:
    proc = None
    try:
        # the child runs `-m lenslapse.add_model`; make the package importable when running
        # from a checkout (pip installs need nothing, but PYTHONPATH is harmless there)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", env=env
        )
        assert proc.stdout is not None  # PIPE was requested
        for line in proc.stdout:
            job["log"].append(line.rstrip())
        proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "failed"
    except Exception as e:  # noqa: BLE001 — a crashed job must surface as failed, not hang as running
        job["log"].append(f"conversion crashed: {e}")
        job["status"] = "failed"
        if proc is not None:
            proc.kill()  # never leave an orphan exporter writing to models_root/web/public
            proc.wait()
        return
    if job["status"] == "done":
        # Hub models now live in models.json, which a fresh server start reads — retire the
        # dialog registration so the two never diverge. Local-directory models are *skipped* by
        # build_registry (machine-specific paths), so their user registration must stay.
        with REG_LOCK:
            entry = STATE["registry"].get(model_id)
            # pip installs have no models.json, so add_model exports only — the registration
            # must stay a user entry there, or it would vanish from the registry on restart
            if entry and entry.origin == "user" and entry.mode != "local" and _IN_REPO:
                STATE["registry"][model_id] = entry.model_copy(update={"origin": "catalog"})
                try:
                    save_user_registry()
                except OSError as e:
                    job["log"].append(f"warning: could not update registry file: {e}")
    print(f"conversion of {model_id}: {job['status']}", flush=True)


@app.get("/models/{model_id}/convert")
def convert_status(model_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(model_id)
        if job is None:
            raise HTTPException(404, f"no conversion job for {model_id!r}")
        result = {"id": model_id, "status": job["status"], "log": list(job["log"])[-8:]}
        if job["status"] == "done" and not _IN_REPO:
            result["note"] = (
                f"exported — upload {STATE['models_root'] / model_id} to your model host to use it in-browser"
            )
        return result


def probe_cache_key(src: CheckpointSource, req: ProbeRequest) -> str:
    """Content key for one probe. JSON-encoded fields, not string concatenation: free-form
    prompt text must never be able to collide with the key of a different (text, targets)."""
    payload = [src.load_ref, src.revision, req.text, sorted(set(req.targets or []))]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def read_cache(cache_file: Path) -> dict | None:
    if not cache_file.exists():
        return None
    try:
        result = json.loads(cache_file.read_text())
    except json.JSONDecodeError:
        print(f"corrupt cache file {cache_file}; recomputing", file=sys.stderr, flush=True)
        cache_file.unlink(missing_ok=True)
        return None
    result["cached"] = True
    return result


@app.post("/probe")
def probe(req: ProbeRequest) -> dict[str, Any]:
    src = source_for(req.model, req.step)
    cache_file = STATE["cache_dir"] / f"{probe_cache_key(src, req)}.json"
    if (result := read_cache(cache_file)) is not None:
        return result

    with PROBE_LOCK:
        # another request may have computed the same key while we waited on the lock
        if (result := read_cache(cache_file)) is not None:
            return result
        return _probe_locked(req, src, cache_file)


def _probe_locked(req: ProbeRequest, src: CheckpointSource, cache_file: Path) -> dict[str, Any]:
    t0 = time.time()
    model, tok = load(src)
    ids = tok(req.text)["input_ids"]
    if not 0 < len(ids) <= MAX_TOKENS:
        raise HTTPException(400, f"prompt must be 1..{MAX_TOKENS} tokens (got {len(ids)})")
    device = next(model.parameters()).device
    t1 = time.time()
    lp = lens_all(model, torch.tensor([ids], device=device)).float().cpu()
    probs = lp.exp()
    top = torch.topk(probs, TOPK, dim=-1)
    t2 = time.time()

    cells = [
        [
            {
                "token": tok.convert_ids_to_tokens([int(top.indices[li, t, 0])])[0],
                "prob": float(top.values[li, t, 0]),
                "top": [
                    [
                        tok.convert_ids_to_tokens([int(top.indices[li, t, k])])[0],
                        float(top.values[li, t, k]),
                        int(top.indices[li, t, k]),
                    ]
                    for k in range(TOPK)
                ],
            }
            for t in range(lp.shape[1])
        ]
        for li in range(lp.shape[0])
    ]
    tgt: dict[str, Any] = {}
    for tid in sorted(set(req.targets or [])):
        if not 0 <= tid < lp.shape[-1]:
            raise HTTPException(400, f"target id {tid} is outside the vocabulary (0..{lp.shape[-1] - 1})")
        ranks = (lp > lp[:, :, tid : tid + 1]).sum(dim=-1) + 1  # [L+1, T]
        tgt[str(tid)] = {
            "token": tok.convert_ids_to_tokens([tid])[0],
            "p": [[round(float(x), 6) for x in row] for row in probs[:, :, tid].tolist()],
            "r": [[int(x) for x in row] for row in ranks.tolist()],
        }

    result = {
        "model": req.model,
        "ref": src.load_ref,
        "revision": src.revision,
        "step": src.step,
        "text": req.text,
        "tokens": tok.convert_ids_to_tokens(ids),
        "grid": {"layers": lp.shape[0], "positions": lp.shape[1], "cells": cells},
        **({"tgt": tgt} if tgt else {}),
        "timing": {"total": round((t2 - t0) * 1000), "forward": round((t2 - t1) * 1000)},
        "device": str(device),
        "cached": False,
    }
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, cache_file)  # atomic: a crash mid-write must not leave a corrupt cache entry
    return result


# In a repo checkout, server state stays under src/server/ where it has always lived; from a
# pip install (no web/ checkout next to the package) it goes to ~/.lenslapse/ instead of
# writing into site-packages.
_PKG_PARENT = Path(__file__).resolve().parent.parent
_IN_REPO = (_PKG_PARENT / "web" / "public" / "data").is_dir()
_STATE_HOME = _PKG_PARENT / "server" if _IN_REPO else Path.home() / ".lenslapse"


def _webapp_root() -> Path | None:
    """Directory holding a complete build of the web app, if one is available.

    Serving the app from this server makes everything same-origin: no CORS, no
    private-network permission prompt, and converted models resolve via /models/ directly.
    Preference: a fresh `npm run build` in a checkout, then the bundle shipped inside the
    wheel (lenslapse/webapp: app shell committed to git, data/tokenizer force-included at
    build time). A shell-only directory (repo tree without the wheel's data) is skipped."""
    for root in _webapp_candidates():
        if (root / "index.html").is_file() and (root / "data" / "models.json").is_file():
            return root
    return None


def _webapp_candidates() -> list[Path]:
    candidates = []
    if _IN_REPO:
        candidates.append(_PKG_PARENT / "web" / "dist")
    candidates.append(Path(__file__).resolve().parent / "webapp")
    return candidates


def default_models_json() -> Path:
    """The catalog the app itself uses: the checkout's copy, or the one bundled in the wheel."""
    if _IN_REPO:
        return _PKG_PARENT / "web/public/data/models.json"
    return Path(__file__).resolve().parent / "webapp" / "data" / "models.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--models-json", default=str(default_models_json()))
    ap.add_argument("--extra", action="append", default=[], help="id=hf_ref[:final] or id=/local/dir (repeatable)")
    ap.add_argument(
        "--registry-file",
        default=str(_STATE_HOME / "registry.json"),
        help="where models registered via the web dialog / POST /models persist",
    )
    ap.add_argument("--cache-dir", default=str(_STATE_HOME / "probe-cache"))
    ap.add_argument(
        "--models-root",
        default=str(_STATE_HOME / "exported-models"),
        help="where dialog-triggered ONNX conversions write <id>/step*/...; served back at /models/",
    )
    ap.add_argument("--max-loaded", type=int, default=1, help="models kept in memory (heavy models: keep 1)")
    ap.add_argument(
        "--device-map",
        action="store_true",
        help="load with device_map='auto' (requires accelerate; multi-device sharding is untested)",
    )
    ap.add_argument("--open", action="store_true", help="open the hosted web app pointed at this server")
    args = ap.parse_args()

    STATE["registry_file"] = Path(args.registry_file)
    STATE["registry"] = build_registry(Path(args.models_json), STATE["registry_file"], args.extra)
    STATE["cache_dir"] = Path(args.cache_dir)
    STATE["cache_dir"].mkdir(parents=True, exist_ok=True)
    STATE["max_loaded"] = args.max_loaded
    STATE["device_map"] = args.device_map
    STATE["models_root"] = Path(args.models_root)
    STATE["models_root"].mkdir(parents=True, exist_ok=True)
    # serve converted ONNX pairs back to the app (mounted after the API routes, which win)
    app.mount("/models", StaticFiles(directory=STATE["models_root"]), name="models")
    if args.extra:
        save_user_registry()  # --extra entries persist like dialog-registered ones
    print(f"registry: {sorted(STATE['registry'])}; cache: {STATE['cache_dir']}", flush=True)
    display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    webapp = _webapp_root()
    if webapp is not None:
        # serve the web app itself: one local port for UI + API, fully offline-capable
        app.mount("/", StaticFiles(directory=webapp, html=True), name="webapp")
        app_url = f"http://{display_host}:{args.port}/"
        print(f"probe server + web app ready — open {app_url}", flush=True)
    else:
        app_url = f"https://iamtatsuki05.github.io/lenslapse/?probe=http://{display_host}:{args.port}"
        print(f"probe server ready (no local web bundle) — open {app_url}", flush=True)
    if args.open:
        threading.Timer(1.5, webbrowser.open, [app_url]).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
