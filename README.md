# LensLapse ⧖

**A fully in-browser time-lapse for the logit lens: scrub across Pythia's public training checkpoints and watch next-token predictions crystallize from noise into knowledge — layer by layer, with zero backend.**

- **Live demo:** https://iamtatsuki05.github.io/lenslapse/ (works in any modern browser; WebGPU used when available, WASM otherwise)
- **Three model sizes** (Pythia 14M / 70M / 160M) switchable in the header; the recipe itself is architecture-generic (GPT-NeoX, GPT-2, and Llama-style RMSNorm models all pass the parity check — see `scripts/check_arch_parity.py`).
- **One-click figure export**: the current view (grid + trajectory + metadata) downloads as a publication-ready PNG (3×) or PDF.
- Curated prompts are **instant**: logit-lens grids across training checkpoints are precomputed (fp32) and served as static JSON.
- Free-text prompts run **live in your browser**: per-checkpoint ONNX pairs (fp16 weights, fp32 compute) are fetched once, cached, and probed with a single forward pass — your prompt never leaves your device.

## Why

- No public, hosted tool lets you interactively inspect a real LLM's internals *across training time* (Pythia ships 154 checkpoints, but existing views are loss curves and static galleries).
- No logit-lens tool of any kind runs fully client-side; hosted server-side demos rot when their backends die.
- LensLapse makes training time a first-class axis of token-level interpretability, and its zero-backend design means unlimited concurrent users at zero hosting cost — the demo cannot rot.

## Architecture

```
Pythia checkpoint (HF Hub, revision step{N})
   └─ scripts/export_checkpoints.py
        ├─ backbone.f16.onnx   input_ids → hidden states [L+1, T, H]   (pre-ln, uniform; via forward hooks)
        └─ lens.f16.onnx       hidden [N, H] → logits [N, V]           (final_layer_norm + unembedding)
   └─ scripts/precompute_lens.py → static JSON shards (top-10 per cell + exact target trajectories)

web/ (Vite, vanilla JS)
   ├─ precomputed mode: fetch JSON shard → canvas grid + SVG trajectories (no model download)
   └─ live mode: onnxruntime-web (WebGPU→WASM fallback) + @huggingface/transformers tokenizer
```

Key property: `lens(hidden[-1]) == model logits` **exactly** (validated per checkpoint at export). Weights are stored fp16 and cast to fp32 at session load; dynamic int8 was rejected because its final-layer top-1 agreement with fp32 drops to 52% (per-tensor; 71% per-channel) at late checkpoints (see `scripts/fidelity_eval.py`).

## Develop

```bash
cd web
npm install
LENSLAPSE_MODELS_DIR=/path/to/converted/models npm run dev   # models dir optional (precomputed mode works without)
```

## Convert checkpoints & precompute

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch transformers onnx onnxscript onnxruntime
# per model id in web/public/data/models.json (NOTE: the default --steps list is the 20-step live
# set; the shipped 14m/70m precomputed data uses a denser 38-step list — pass it explicitly to
# reproduce, or you will overwrite the shipped shards with a coarser grid):
python scripts/export_checkpoints.py --model EleutherAI/pythia-70m --out /path/to/models/pythia-70m
python scripts/precompute_lens.py    --model EleutherAI/pythia-70m \
  --steps 0,1,2,4,8,16,32,64,128,256,512,1000,2000,3000,4000,6000,8000,12000,16000,20000,24000,28000,32000,36000,40000,48000,56000,64000,72000,80000,88000,96000,104000,112000,120000,128000,136000,143000 \
  --out web/public/data/pythia-70m
python scripts/fidelity_eval.py --out /tmp/fidelity_report.json     # weight-format fidelity table
python scripts/check_arch_parity.py --model gpt2                    # lens-identity check on any HF decoder
```

## Add your own model (Hub or local)

```bash
# any HF checkpoint suite (step{N} revisions), a single HF model, or a local HF-Trainer run dir:
python scripts/add_model.py --model EleutherAI/pythia-31m --id pythia-31m --label "Pythia 31M"     --steps 0,512,8000,143000 --models-root /path/to/models
python scripts/add_model.py --model gpt2 --id gpt2 --label "GPT-2 124M" --final-only --models-root /path/to/models
python scripts/add_model.py --model /path/to/trainer_output --id my-run --label "My run" --models-root /path/to/models
```

One command exports the ONNX pairs (parity-checked), precomputes the lens shards, installs the
tokenizer, and registers the model in `models.json` — adding a model is a data change, not a code
change. Architectures are resolved generically (GPT-NeoX / GPT-2 / Llama-style RMSNorm verified).

## Heavy models: the local probe server

For models too large to download into a browser, run the optional probe server and point the app
at it with `?probe=`:

```bash
pip install -e ".[server]"
python server/probe_server.py --port 8017 --extra my-big-model=meta-llama/Llama-3.2-1B:final
# open http://localhost:5199/?probe=http://localhost:8017 — the badge switches to "live · server"
```

When a probe server is connected, a **⚙ models** button appears in the header: register a Hub
model (single checkpoint or a `step{N}` suite) or a Trainer directory on the server machine from
the dialog — no ONNX export, no CLI. Registered models show up in the picker as “(server)”,
are live-only (no precomputed prompts), and persist to `server/registry.json` across restarts.
The management API (`GET/POST/DELETE /models`) is unauthenticated by design — keep the server
bound to localhost.

It runs plain `transformers` on CUDA/MPS/CPU (add `--device-map` to shard very large models via
accelerate). We deliberately did **not** use vLLM/TGI: the logit lens needs the entire per-layer
residual stream of one teacher-forced forward pass, which generation engines neither expose nor
accelerate. The server reuses the exact hooked-forward + lens code that builds the precomputed
shards, so its numbers agree with them by construction (verified 56/56 top-1 on Pythia-70M).

## Reproducibility of probes

Every live probe result is persisted — in the browser (IndexedDB) and on the probe server
(`server/probe-cache/*.json`, keyed by model ref x revision x prompt). Identical requests replay
the stored result byte-for-byte, so a shared permalink keeps showing the same numbers across
sessions, backends, and model-host outages.

The key is the model *reference*, not the weights: if you re-export a model under the same id
(or retrain a local run in place), saved probes are stale. Open the app with `?fresh` to bypass
replay and recompute (the new result overwrites the saved one), and clear `server/probe-cache/`
after retraining.

## Benchmark

```bash
npx playwright install chromium firefox webkit
LENSLAPSE_MODELS_DIR=/path/to/models npm run preview -- --port 5199 &
node bench/bench.mjs --base http://localhost:5199 --out bench.json
```

Measures checkpoint load and probe latency across browser engines (Chromium/Firefox/WebKit),
execution providers (WebGPU/WASM), emulated CPU throttling, model sizes, and prompt lengths.

## Deploy (zero cost)

1. Upload the converted models to a public Hugging Face repo and set `HF_DEFAULT` in `web/src/live.js`.
2. `npm run build`, publish `web/dist/` to GitHub Pages (workflow in `.github/workflows/deploy-pages.yml`).
3. There is no step 3 — no server, no keys, no bills. See `docs/deployment.md`.

## License

MIT. Pythia checkpoints are © EleutherAI, Apache-2.0.
