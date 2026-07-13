# LensLapse ⧖

**A fully in-browser time-lapse for the logit lens: scrub across Pythia's public training checkpoints and watch next-token predictions crystallize from noise into knowledge — layer by layer, with zero backend.**

- **Live demo:** https://iamtatsuki05.github.io/lenslapse/ (works in any modern browser; WebGPU used when available, WASM otherwise)
- **Three model sizes** (Pythia 14M / 70M / 160M) switchable in the header; the recipe itself is architecture-generic (GPT-NeoX, GPT-2, and Llama-style RMSNorm models all pass the parity check — see `scripts/check_arch_parity.py`).
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
# per model id in web/public/data/models.json:
python scripts/export_checkpoints.py --model EleutherAI/pythia-70m --out /path/to/models/pythia-70m
python scripts/precompute_lens.py    --model EleutherAI/pythia-70m --out web/public/data/pythia-70m
python scripts/fidelity_eval.py --out fidelity_report.json          # weight-format fidelity table
python scripts/check_arch_parity.py --model gpt2                    # lens-identity check on any HF decoder
```

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
