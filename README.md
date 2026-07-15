# LensLapse ⧖

**A fully in-browser time-lapse for the logit lens: scrub across Pythia's public training checkpoints and watch next-token predictions crystallize from noise into knowledge — layer by layer, with zero backend.**

- **Live demo:** https://iamtatsuki05.github.io/lenslapse/ (works in any modern browser; WebGPU used when available, WASM otherwise)
- **Seven shipped models** — Pythia 14M / 70M / 160M across 20 training checkpoints each, GPT-2 124M (final checkpoint), and three multilingual suites unchanged by the same recipe: MAP-Neo-250M and BAAI Aquila-135M (Chinese/English, Hub-subfolder checkpoints), and BLOOM-560M (46 languages, `global_step{N}` revisions) — switchable in the header. The recipe itself is architecture-generic (GPT-NeoX, GPT-2, Llama-style RMSNorm, and Mistral-style RMSNorm models all pass the parity check — see `lenslapse/check_arch_parity.py`).
- **One-click figure export**: the current view (grid + trajectory + metadata) downloads as a publication-ready PNG (3× pixel density) or PDF.
- Curated prompts are **instant**: logit-lens grids across training checkpoints are precomputed (fp32) and served as static JSON.
- Free-text prompts run **live in your browser**: per-checkpoint ONNX pairs (fp16 weights, fp32 compute) are fetched once, cached, and probed with a single forward pass — your prompt never leaves your device.

## Quick start: probe your own models (no checkout needed)

```bash
pip install "git+https://github.com/iamtatsuki05/lenslapse.git"
lenslapse server        # serves the web app AND the probe API on one local port, then opens it
```

Everything runs on `http://localhost:8017/` — the UI is bundled into the package, so this works
fully offline once models are downloaded, with no CORS or browser permission prompts.
Click **⚙ models** in the header, pick a Hugging Face id or press **📁 Browse…** to choose a
checkpoint folder with your OS's file dialog, and probe it live — no ONNX conversion, no config
files. (In a checkout, `uv run lenslapse server` serves your own `web/dist` build instead;
run `scripts/bundle_webapp.sh` after changing `web/` to refresh the packaged shell.)

## Why

- No public, hosted tool lets you interactively inspect a real LLM's internals *across training time* (Pythia ships 154 checkpoints, but existing views are loss curves and static galleries).
- No logit-lens tool of any kind runs fully client-side; hosted server-side demos rot when their backends die.
- LensLapse makes training time a first-class axis of token-level interpretability, and its zero-backend design means unlimited concurrent users at zero hosting cost — the demo cannot rot.

## Architecture

```
Pythia checkpoint (HF Hub, revision step{N})
   └─ lenslapse/export_checkpoints.py
        ├─ backbone.f16.onnx   input_ids → hidden states [L+1, T, H]   (pre-ln, uniform; via forward hooks)
        └─ lens.f16.onnx       hidden [N, H] → logits [N, V]           (final_layer_norm + unembedding)
   └─ lenslapse/precompute_lens.py → static JSON shards (top-10 per cell + exact target trajectories)

web/ (Vite, TypeScript)
   ├─ precomputed mode: fetch JSON shard → canvas grid + SVG trajectories (no model download)
   └─ live mode: onnxruntime-web (WebGPU→WASM fallback) + @huggingface/transformers tokenizer
```

Key property: `lens(hidden[-1]) == model logits` **exactly** (validated per checkpoint at export). Weights are stored fp16 and cast to fp32 at session load; dynamic int8 was rejected because its final-layer top-1 agreement with fp32 drops to 52% (per-tensor; 71% per-channel) at late checkpoints (see `lenslapse/fidelity_eval.py`).

## Develop

```bash
cd web
npm install
LENSLAPSE_MODELS_DIR=/path/to/converted/models npm run dev   # models dir optional (precomputed mode works without)
npm run typecheck && npm run test                            # TypeScript + vitest
```

Python side (pipeline + probe server), from the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync        # creates .venv from uv.lock (dev tools included)
uv run tox     # pytest (-n auto) + ruff + mypy
```

Both suites run in CI (`.github/workflows/ci.yml`) on every push.

## Convert checkpoints & precompute

```bash
# uv sync first (see Develop); per model id in web/public/data/models.json
# (NOTE: the default --steps list is the 20-step live
# set; the shipped 14m/70m precomputed data uses a denser 38-step list — pass it explicitly to
# reproduce, or you will overwrite the shipped shards with a coarser grid):
uv run python -m lenslapse.export_checkpoints --model EleutherAI/pythia-70m --out /path/to/models/pythia-70m
uv run python -m lenslapse.precompute_lens  --model EleutherAI/pythia-70m \
  --steps 0,1,2,4,8,16,32,64,128,256,512,1000,2000,3000,4000,6000,8000,12000,16000,20000,24000,28000,32000,36000,40000,48000,56000,64000,72000,80000,88000,96000,104000,112000,120000,128000,136000,143000 \
  --out web/public/data/pythia-70m
uv run python -m lenslapse.fidelity_eval --out /tmp/fidelity_report.json     # weight-format fidelity table
uv run python -m lenslapse.check_arch_parity --model gpt2      # lens-identity check on any HF decoder
```

## Add your own model (Hub or local)

```bash
# any HF checkpoint suite (step{N} revisions), a single HF model, or a local HF-Trainer run dir:
uv run lenslapse add-model --model EleutherAI/pythia-31m --id pythia-31m --label "Pythia 31M"     --steps 0,512,8000,143000 --models-root /path/to/models
uv run lenslapse add-model --model gpt2 --id gpt2 --label "GPT-2 124M" --final-only --models-root /path/to/models
uv run lenslapse add-model --model /path/to/trainer_output --id my-run --label "My run" --models-root /path/to/models

# repos that nest each checkpoint as a Hub subfolder within one revision instead of a git revision
# per checkpoint (e.g. MAP-Neo, BAAI Aquila), a non-default revision naming (BLOOM's global_step{N}),
# and a tokenizer loaded from a different ref than the checkpoint weights:
uv run lenslapse add-model --model m-a-p/neo_scalinglaw_250M --id mapneo-250m --label "MAP-Neo 250M" \
  --subfolder-map "16780:hf_ckpt/16.78B,33550:hf_ckpt/33.55B" --models-root /path/to/models
uv run lenslapse add-model --model bigscience/bloom-560m-intermediate --id bloom-560m --label "BLOOM 560M" \
  --steps 1000,10000,100000 --revision-template "global_step{}" --tokenizer-ref bigscience/bloom-560m \
  --models-root /path/to/models
```

One command exports the ONNX pairs (parity-checked), precomputes the lens shards, installs the
tokenizer, and registers the model in `models.json` — adding a model is a data change, not a code
change. Architectures are resolved generically (GPT-NeoX / GPT-2 / Llama-style RMSNorm / Mistral-style
RMSNorm verified).

## Heavy models: the local probe server

For models too large to download into a browser, run the optional probe server:

```bash
# the server ships with the package — install via the Quick start (pip) or `uv sync` (checkout)
lenslapse server --extra my-big-model=meta-llama/Llama-3.2-1B:final   # or: uv run lenslapse server
# a locally served app (npm run dev / preview) finds the default port by itself;
# probe any suite step — the badge switches to "live · server"
```

A locally served app auto-detects the server on the default port `8017`. Anywhere else (for
example the public Pages deployment), opt in once with `?probe=http://localhost:8017` — the app
remembers the server across visits, so later visits need no parameter; `?probe=off` forgets it.
On an HTTPS deployment your browser may ask permission to reach the local network the first time.

When a probe server is connected, a **⚙ models** button appears in the header: register a Hub
model (single checkpoint or a `step{N}` suite) or a local checkpoint folder — pick it with
**📁 Browse…**, which opens the server machine's native folder dialog — no ONNX export, no CLI.
Registered models show up in the picker as “(server)”, are live-only (no precomputed prompts),
and persist across restarts (`server/registry.json` in a checkout, `~/.lenslapse/` for pip
installs). The management API (`GET/POST/DELETE /models`, `/pick-folder`) is unauthenticated by
design — keep the server bound to localhost.

Each registered model also has a **convert to ONNX** button that runs the full onboarding
pipeline (`add_model.py`) on the server machine in the background (roughly a minute per sub-200M
checkpoint). When it finishes, the model is in `web/public` + `models.json` like a shipped one:
an app served from source (`npm run dev` with `LENSLAPSE_MODELS_DIR=server/exported-models`)
picks it up on reload and runs it **fully in-browser**; for the deployed site, upload the ONNX
pairs from `server/exported-models/<id>/` to your model host and rebuild.

It runs plain `transformers` on CUDA/MPS/CPU (add `--device-map` to shard very large models via
accelerate): the logit lens needs the entire per-layer residual stream of one teacher-forced
forward pass — a batch-1 workload with no decoding loop, which generation-oriented serving
engines neither expose natively nor accelerate. The server reuses the exact hooked-forward +
lens code that builds the precomputed shards and computes in fp32 by default, so its numbers
agree with them (verified 56/56 top-1 on Pythia-70M). Half-precision *compute* is not free:
an fp16 forward already flips 5/56 late-checkpoint top-1s, so `--dtype float16/bfloat16`
(halving memory for heavy models) is an explicit opt-in, and results at different dtypes
never replay for one another in the probe cache.

## Script it: the CLI mirrors the UI

Every interactive feature also runs headless from the terminal (`probe --json` prints the
exact payload the web app receives; `trace --json` a step-by-step trajectory bundle):

```bash
lenslapse models add --ref /path/to/trainer_output --id my-run    # the “⚙ models” dialog
lenslapse probe --model my-run --text "The capital of France is"  # the “Live probe” button
lenslapse trace --model my-run --text "The capital of France is"  # “▶ trace across training”
lenslapse models convert my-run                                   # the “convert to ONNX” button
```

Commands drive a running `lenslapse server` when one is reachable (default port 8017), sharing
its registry, loaded weights, and probe cache; otherwise the same code runs in-process
(`--local` forces this) against the same default on-disk state. Either way the results land in
the shared probe cache, so a trajectory traced in the terminal replays instantly in the
browser — and vice versa. `probe` prints the token ids you can feed back into `--targets`;
`trace` fixes its tracked tokens from the final checkpoint exactly like the UI and the
precomputed shards do. One caveat: while a server is running, do `models add`/`remove` through
it (the default), not with `--local` — both sides rewrite the same registry file on change,
and the last writer wins.

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
cd web
npx playwright install chromium firefox webkit
LENSLAPSE_MODELS_DIR=/path/to/models npm run preview -- --port 5199 &
node bench/bench.mjs --base http://localhost:5199 --out bench.json
```

Measures checkpoint load and probe latency across browser engines (Chromium/Firefox/WebKit),
execution providers (WebGPU/WASM), emulated CPU throttling, model sizes, and prompt lengths.

## Deploy (zero cost)

1. Upload the converted models to a public Hugging Face model repo and set `HF_DEFAULT` in `web/src/live.ts`.
2. `npm run build`, publish `web/dist/` to GitHub Pages (workflow in `.github/workflows/deploy-pages.yml`).
3. There is no step 3 — no server, no keys, no bills. See `docs/deployment.md`.

## License

MIT. Pythia checkpoints are © EleutherAI, Apache-2.0; GPT-2 weights are © OpenAI, MIT (Modified);
MAP-Neo and BAAI Aquila checkpoints are Apache-2.0. **BLOOM checkpoints are © BigScience Workshop,
licensed under the BigScience RAIL License v1.0** — not a plain permissive license like the others
here; it attaches use-based behavioral restrictions to downstream recipients. See
`docs/model-card.md` for the full attribution and license text for every suite.
