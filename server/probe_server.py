"""Optional local probe server for models too heavy for in-browser inference.

Why not vLLM/TGI: the logit lens needs the *entire per-layer residual stream* of one
teacher-forced forward pass. Generation engines do not natively expose per-layer hidden
states (tracing layers like NNsight can reach them, but require enforce_eager and are
CUDA-only), and this workload (no decoding loop, batch size 1) gains nothing from a serving
engine. So this server runs plain transformers on CUDA/MPS/CPU, reusing the exact
hooked-forward + lens implementation that generates the precomputed shards — the numbers
agree by construction.

Every probe result is persisted to --cache-dir keyed by (model ref, revision, prompt), and
identical requests are replayed from disk, byte-for-byte: results are reproducible across
sessions and auditable as plain JSON files. Note the key is the *reference*, not the weights:
for mutable refs (revision "main", local run directories) a re-trained model under the same
path keeps replaying the old cached results — clear the cache directory after retraining.

Usage:
  pip install -e ".[server]"        (or: pip install fastapi uvicorn)
  python server/probe_server.py --port 8017
  # then open the app with ?probe=http://localhost:8017

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
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import model_info
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from precompute_lens import lens_all  # noqa: E402
from sources import resolve_sources  # noqa: E402

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
async def allow_private_network(request, call_next):
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


class ProbeRequest(BaseModel):
    model: str
    step: int = 0
    text: str


class RegisterRequest(BaseModel):
    id: str
    ref: str  # HF id or a directory on the server machine
    mode: str  # "suite" | "final" | "local"
    label: str | None = None
    steps: list[int] | None = None  # suite only; defaults to the Pythia step grid


def build_registry(models_json: Path | None, registry_file: Path | None, extras: list[str]) -> dict:
    registry: dict = {}
    if models_json and models_json.exists():
        for m in json.loads(models_json.read_text())["models"]:
            mode = m.get("mode", "suite")
            if mode == "local":
                continue  # local paths are machine-specific; register via --extra id=/path
            # "source" is the true HF ref; "hf" doubles as the app's local tokenizer directory
            # name, which for add_model.py onboarded models is the app id, not a Hub id.
            registry[m["id"]] = {
                "ref": m.get("source", m["hf"]),
                "mode": mode,
                "label": m.get("label"),
                "origin": "catalog",
            }
    if registry_file and registry_file.exists():
        for mid, entry in json.loads(registry_file.read_text()).items():
            registry[mid] = {**entry, "origin": "user"}
    for e in extras:
        mid, sep, ref = e.partition("=")
        if not sep or not mid or not ref:
            raise SystemExit(f"--extra expects id=hf_ref[:final] or id=/local/dir, got {e!r}")
        mode = "suite"
        if ref.endswith(":final"):
            ref, mode = ref[: -len(":final")], "final"
        elif Path(ref).is_dir():
            mode = "local"
        registry[mid] = {"ref": ref, "mode": mode, "label": None, "origin": "user"}
    return registry


def save_user_registry() -> None:
    """Persist user-registered entries (dialog/API/--extra) so they survive restarts."""
    path = STATE["registry_file"]
    entries = {
        mid: {k: v for k, v in e.items() if k != "origin"}
        for mid, e in STATE["registry"].items()
        if e.get("origin") == "user"
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2) + "\n")
    os.replace(tmp, path)


def model_steps(entry: dict) -> list[int]:
    if entry["mode"] == "final":
        return [0]
    if entry["mode"] == "local":
        if not Path(entry["ref"]).is_dir():
            return []  # directory gone since registration; probing it 404s with an explicit message
        return [src.step for src in resolve_sources(entry["ref"], "0", final_only=False)]
    return entry.get("steps") or DEFAULT_SUITE_STEPS


def source_for(model_id: str, step: int):
    entry = STATE["registry"].get(model_id)
    if entry is None:
        raise HTTPException(404, f"unknown model id {model_id!r}; register it via --extra")
    mode, ref = entry["mode"], entry["ref"]
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


def load(src):
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
def health():
    return {"ok": True, "models": sorted(STATE["registry"]), "device": pick_device()}


@app.get("/models")
def list_models():
    return [
        {
            "id": mid,
            "ref": e["ref"],
            "mode": e["mode"],
            "label": e.get("label") or mid,
            "steps": model_steps(e),
            "origin": e.get("origin", "user"),
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
def register_model(req: RegisterRequest):
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
        entry = {"ref": req.ref, "mode": req.mode, "label": req.label, "origin": "user"}
        if req.mode == "suite" and req.steps:
            entry["steps"] = sorted(set(req.steps))
        STATE["registry"][req.id] = entry
        try:
            save_user_registry()
        except OSError as e:
            del STATE["registry"][req.id]  # keep memory and registry.json consistent
            raise HTTPException(500, f"could not persist registry: {e}") from e
    print(f"registered {req.id} -> {req.ref} ({req.mode})", flush=True)
    return {"id": req.id, **{k: v for k, v in entry.items() if k != "origin"}, "steps": model_steps(entry)}


@app.delete("/models/{model_id}")
def unregister_model(model_id: str):
    with REG_LOCK:
        entry = STATE["registry"].get(model_id)
        if entry is None:
            raise HTTPException(404, f"unknown model id {model_id!r}")
        if entry.get("origin") == "catalog":
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
def probe(req: ProbeRequest):
    src = source_for(req.model, req.step)
    cache_key = hashlib.sha256(f"{src.load_ref}@{src.revision}::{req.text}".encode()).hexdigest()
    cache_file = STATE["cache_dir"] / f"{cache_key}.json"
    if (result := read_cache(cache_file)) is not None:
        return result

    with PROBE_LOCK:
        # another request may have computed the same key while we waited on the lock
        if (result := read_cache(cache_file)) is not None:
            return result
        return _probe_locked(req, src, cache_file)


def _probe_locked(req: ProbeRequest, src, cache_file: Path):
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
    result = {
        "model": req.model,
        "ref": src.load_ref,
        "revision": src.revision,
        "step": src.step,
        "text": req.text,
        "tokens": tok.convert_ids_to_tokens(ids),
        "grid": {"layers": lp.shape[0], "positions": lp.shape[1], "cells": cells},
        "timing": {"total": round((t2 - t0) * 1000), "forward": round((t2 - t1) * 1000)},
        "device": str(device),
        "cached": False,
    }
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, cache_file)  # atomic: a crash mid-write must not leave a corrupt cache entry
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--models-json", default=str(Path(__file__).resolve().parent.parent / "web/public/data/models.json")
    )
    ap.add_argument("--extra", action="append", default=[], help="id=hf_ref[:final] or id=/local/dir (repeatable)")
    ap.add_argument(
        "--registry-file",
        default=str(Path(__file__).resolve().parent / "registry.json"),
        help="where models registered via the web dialog / POST /models persist",
    )
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "probe-cache"))
    ap.add_argument("--max-loaded", type=int, default=1, help="models kept in memory (heavy models: keep 1)")
    ap.add_argument(
        "--device-map",
        action="store_true",
        help="load with device_map='auto' (requires accelerate; multi-device sharding is untested)",
    )
    args = ap.parse_args()

    STATE["registry_file"] = Path(args.registry_file)
    STATE["registry"] = build_registry(Path(args.models_json), STATE["registry_file"], args.extra)
    STATE["cache_dir"] = Path(args.cache_dir)
    STATE["cache_dir"].mkdir(parents=True, exist_ok=True)
    STATE["max_loaded"] = args.max_loaded
    STATE["device_map"] = args.device_map
    if args.extra:
        save_user_registry()  # --extra entries persist like dialog-registered ones
    print(f"registry: {sorted(STATE['registry'])}; cache: {STATE['cache_dir']}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
