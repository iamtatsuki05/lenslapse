"""Optional local probe server for models too heavy for in-browser inference.

Why not vLLM/TGI: the logit lens needs the *entire per-layer residual stream* of one
teacher-forced forward pass. Generation engines neither expose per-layer hidden states nor
benefit this workload (no decoding loop, batch size 1), so this server runs plain
transformers with `device_map="auto"` (CUDA/MPS/CPU), reusing the exact hooked-forward +
lens implementation that generates the precomputed shards — the numbers agree by construction.

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
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from precompute_lens import lens_all  # noqa: E402
from sources import resolve_sources  # noqa: E402

TOPK = 10
MAX_TOKENS = 64

app = FastAPI(title="LensLapse probe server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"registry": {}, "cache_dir": None, "max_loaded": 1, "device_map": False}
LOADED: OrderedDict = OrderedDict()  # (ref, revision) -> (model, tokenizer)
# FastAPI runs sync endpoints in a threadpool; lens_all installs forward hooks on the shared
# model, so concurrent probes would capture each other's layer outputs (and persist the garbage
# to the cache). One global lock serializes load+forward+write — fine for a batch-1 local server.
PROBE_LOCK = threading.Lock()


class ProbeRequest(BaseModel):
    model: str
    step: int = 0
    text: str


def build_registry(models_json: Path | None, extras: list[str]) -> dict:
    registry: dict = {}
    if models_json and models_json.exists():
        for m in json.loads(models_json.read_text())["models"]:
            mode = m.get("mode", "suite")
            if mode == "local":
                continue  # local paths are machine-specific; register via --extra id=/path
            # "source" is the true HF ref; "hf" doubles as the app's local tokenizer directory
            # name, which for add_model.py onboarded models is the app id, not a Hub id.
            registry[m["id"]] = {"ref": m.get("source", m["hf"]), "mode": mode}
    for e in extras:
        mid, sep, ref = e.partition("=")
        if not sep or not mid or not ref:
            raise SystemExit(f"--extra expects id=hf_ref[:final] or id=/local/dir, got {e!r}")
        mode = "suite"
        if ref.endswith(":final"):
            ref, mode = ref[: -len(":final")], "final"
        elif Path(ref).is_dir():
            mode = "local"
        registry[mid] = {"ref": ref, "mode": mode}
    return registry


def source_for(model_id: str, step: int):
    entry = STATE["registry"].get(model_id)
    if entry is None:
        raise HTTPException(404, f"unknown model id {model_id!r}; register it via --extra")
    mode, ref = entry["mode"], entry["ref"]
    if mode == "final":
        return resolve_sources(ref, "0", final_only=True)[0]
    if mode == "local":
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
    ap.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / "probe-cache"))
    ap.add_argument("--max-loaded", type=int, default=1, help="models kept in memory (heavy models: keep 1)")
    ap.add_argument(
        "--device-map",
        action="store_true",
        help="load with device_map='auto' (requires accelerate; multi-device sharding is untested)",
    )
    args = ap.parse_args()

    STATE["registry"] = build_registry(Path(args.models_json), args.extra)
    STATE["cache_dir"] = Path(args.cache_dir)
    STATE["cache_dir"].mkdir(parents=True, exist_ok=True)
    STATE["max_loaded"] = args.max_loaded
    STATE["device_map"] = args.device_map
    print(f"registry: {sorted(STATE['registry'])}; cache: {STATE['cache_dir']}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
