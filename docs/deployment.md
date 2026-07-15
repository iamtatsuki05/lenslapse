# Zero-cost deployment

The live site is a static bundle (GitHub Pages) plus model weights on the Hugging Face Hub. Nothing
runs server-side, so hosting is free, permanent, and scales with the number of visitors' devices.

## 1. Upload converted checkpoints to Hugging Face Hub

```bash
pip install huggingface_hub
hf auth login
hf upload-large-folder <user>/lenslapse-onnx --repo-type model /path/to/lenslapse-models-multi
```

Notes:
- A public **model** repo is used (free storage for public repos, CORS-enabled `resolve/` URLs;
  the files are model weights, just converted to ONNX).
- Layout is one directory per model id:
  `<model-id>/manifest.json` + `<model-id>/step{N}/backbone.f16.onnx` + `<model-id>/step{N}/lens.f16.onnx`,
  with model ids matching `web/public/data/models.json` (e.g. `pythia-14m`, `pythia-70m`,
  `pythia-160m`, `gpt2`, `mapneo-250m`, `aquila-135m`, `bloom-560m`).
- Add `docs/model-card.md` as the repo README — re-upload it (`hf upload iamtatsuki05/lenslapse-onnx
  docs/model-card.md README.md`) whenever it changes; the file living in the checkout does not sync
  itself.

## 2. Point the app at the repo

Edit `HF_DEFAULT` in `web/src/live.ts`:

```ts
const HF_DEFAULT = 'https://huggingface.co/<user>/lenslapse-onnx/resolve/main/'
```

The app resolves models in this order: `?models=` URL parameter → same-origin `models/` → `HF_DEFAULT`.
If none is reachable it degrades to precomputed-only mode with a status badge (no crash).

## 3. Publish the site on GitHub Pages

1. Push this repository to GitHub (public, e.g. `lenslapse`).
2. Repository Settings → Pages → Source: **GitHub Actions**.
3. The included workflow (`.github/workflows/deploy-pages.yml`) builds `web/` and deploys `web/dist`
   on every push to `main`.

The built site is ~160MB for the seven shipped models (two ONNX Runtime WASM binaries ~44MB +
precomputed shards ~58MB + tokenizer files ~28MB + app assets ~46MB), well under the 1GB Pages
limit. Model *weights* are **not** part of the site — only the small per-model tokenizer directories
that the live-probe path and the free-text tokenizer feature need locally.

## Verification checklist after deploy

- [ ] Page loads and the lens grid renders for the default prompt (precomputed, no download).
- [ ] Slider scrubbing updates the grid; ticks show live-capable diamonds.
- [ ] The model picker switches between all eleven shipped models — Pythia 14M / 70M / 160M, GPT-2,
      the multilingual MAP-Neo-250M / Aquila-135M / BLOOM-560M suites, and the four architecture-
      coverage examples SmolLM2-135M / Qwen3-0.6B / OPT-125M / Gemma-3-270M (steps and grid depth
      change per model; every model shows live-capable diamonds for every one of its checkpoints).
- [ ] Clicking a cell pins it and draws trajectories; the permalink button reproduces the exact view
      (including the selected model).
- [ ] Live probe on a free-text prompt downloads one checkpoint (~29MB for 14M, ~142MB for 70M,
      ~326MB for 160M, ~327MB for GPT-2, once per checkpoint) and completes; badge shows WebGPU
      or WASM.
- [ ] `?ep=wasm` forces the WASM path (for browsers without WebGPU).
- [ ] DevTools Network tab shows requests only to the site origin and `huggingface.co`/its CDN.
