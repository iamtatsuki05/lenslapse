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
  `pythia-160m`, `gpt2`).
- Add `docs/model-card.md` as the repo README.

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

The built site is ~65MB (two ONNX Runtime WASM binaries ~44MB + precomputed shards ~15MB + app
assets), well under the 1GB Pages limit. Model weights are **not** part of the site.

## Verification checklist after deploy

- [ ] Page loads and the lens grid renders for the default prompt (precomputed, no download).
- [ ] Slider scrubbing updates the grid; ticks show live-capable diamonds.
- [ ] The model picker switches between the shipped models — Pythia 14M / 70M / 160M and GPT-2
      (steps and grid depth change).
- [ ] Clicking a cell pins it and draws trajectories; the permalink button reproduces the exact view
      (including the selected model).
- [ ] Live probe on a free-text prompt downloads one checkpoint (~29MB for 14M, ~142MB for 70M,
      ~326MB for 160M, ~327MB for GPT-2, once per checkpoint) and completes; badge shows WebGPU
      or WASM.
- [ ] `?ep=wasm` forces the WASM path (for browsers without WebGPU).
- [ ] DevTools Network tab shows requests only to the site origin and `huggingface.co`/its CDN.
