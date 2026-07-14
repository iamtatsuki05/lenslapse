---
license: apache-2.0
base_model:
  - EleutherAI/pythia-70m
  - openai-community/gpt2
tags:
  - onnx
  - interpretability
  - logit-lens
  - training-dynamics
  - lenslapse
---

# LensLapse ONNX checkpoints — Pythia suite across training, plus GPT-2

Browser-runnable ONNX conversions of 20 log-spaced **training checkpoints** each of
[EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m),
[pythia-70m](https://huggingface.co/EleutherAI/pythia-70m), and
[pythia-160m](https://huggingface.co/EleutherAI/pythia-160m), plus the final checkpoint of
[GPT-2 124M](https://huggingface.co/openai-community/gpt2), packaged for the
[LensLapse](https://iamtatsuki05.github.io/lenslapse/) in-browser logit-lens demo.

## Contents

One directory per model (`pythia-14m/`, `pythia-70m/`, `pythia-160m/`, `gpt2/`), each with a
`manifest.json` (steps, per-file sizes, export-time parity metrics). The three Pythia
directories hold one `step{N}/` pair for each training step
`N` ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
128000, 143000}:

| file | interface | size (14m / 70m / 160m) |
|---|---|---|
| `step{N}/backbone.f16.onnx` | `input_ids, attention_mask → hidden_states [L+1, B, T, H]` | ~16 / ~91 / ~249 MB |
| `step{N}/lens.f16.onnx` | `hidden [N, H] → logits [N, V]` (final norm + unembedding) | ~13 / ~52 / ~77 MB |

`gpt2/` ships a single `step0/` pair — the released final weights (~250 MB backbone + ~77 MB lens);
GPT-2 has no public intermediate checkpoints.

The backbone outputs the **pre-final-layer-norm residual stream** after the embedding and after each
transformer block, so applying the lens head to any layer implements the logit lens uniformly, and
applying it to the last layer reproduces the model's actual output distribution exactly.

## Fidelity

Weights are stored fp16 and cast to fp32 at session load (compute is fp32). Against fp32 PyTorch,
top-1 lens predictions agree on 100.0% of (layer, position) cells over a 16-prompt suite at steps
8000/64000/143000; the export-time probe over all 20 checkpoints shows a max logit diff of 0.0087.
Dynamic int8 quantization was evaluated and rejected (final-layer top-1 agreement drops to 52%
per-tensor / 71% per-channel at step 143000). See the LensLapse repository for the evaluation
script (`lenslapse/fidelity_eval.py`).

## Reproduce

```bash
uv run python -m lenslapse.export_checkpoints --model EleutherAI/pythia-70m --out ./models
uv run lenslapse add-model --model gpt2 --id gpt2 --label "GPT-2 124M" --final-only --models-root ./models
```

## License and attribution

Conversion scripts MIT. The `pythia-*/` directories are derived from Pythia weights,
Apache-2.0 © EleutherAI (Biderman et al., 2023, "Pythia: A Suite for Analyzing Large Language
Models Across Training and Scaling"); the `gpt2/` directory is derived from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2), MIT (Modified) © OpenAI
(Radford et al., 2019). The `license` metadata field above names the Pythia license; the GPT-2
files keep their own. This repo redistributes derived weights unchanged in ONNX form.
