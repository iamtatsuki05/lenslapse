---
license: apache-2.0
base_model: EleutherAI/pythia-70m
tags:
  - onnx
  - interpretability
  - logit-lens
  - training-dynamics
  - lenslapse
---

# LensLapse ONNX checkpoints — Pythia suite across training

Browser-runnable ONNX conversions of 20 log-spaced **training checkpoints** each of
[EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m),
[pythia-70m](https://huggingface.co/EleutherAI/pythia-70m), and
[pythia-160m](https://huggingface.co/EleutherAI/pythia-160m), packaged for the
[LensLapse](https://iamtatsuki05.github.io/lenslapse/) in-browser logit-lens demo
(AACL-IJCNLP 2026 System Demonstrations submission).

## Contents

One directory per model (`pythia-14m/`, `pythia-70m/`, `pythia-160m/`), each with a
`manifest.json` (steps, per-file sizes, export-time parity metrics) and, for each training step
`N` ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
128000, 143000}:

| file | interface | size (14m / 70m / 160m) |
|---|---|---|
| `step{N}/backbone.f16.onnx` | `input_ids, attention_mask → hidden_states [L+1, B, T, H]` | ~16 / ~91 / ~249 MB |
| `step{N}/lens.f16.onnx` | `hidden [N, H] → logits [N, V]` (final norm + unembedding) | ~13 / ~52 / ~77 MB |

The backbone outputs the **pre-final-layer-norm residual stream** after the embedding and after each
transformer block, so applying the lens head to any layer implements the logit lens uniformly, and
applying it to the last layer reproduces the model's actual output distribution exactly.

## Fidelity

Weights are stored fp16 and cast to fp32 at session load (compute is fp32). Against fp32 PyTorch,
top-1 lens predictions agree on 100.0% of (layer, position) cells over a 16-prompt suite at steps
8000/64000/143000; the export-time probe over all 20 checkpoints shows a max logit diff of 0.0087.
Dynamic int8 quantization was evaluated and rejected (final-layer top-1 agreement drops to 52%
per-tensor / 71% per-channel at step 143000). See the LensLapse repository for the evaluation
script (`scripts/fidelity_eval.py`).

## Reproduce

```bash
python scripts/export_checkpoints.py --model EleutherAI/pythia-70m --out ./models
```

## License and attribution

Conversion scripts MIT; the underlying Pythia weights are Apache-2.0 © EleutherAI
(Biderman et al., 2023, "Pythia: A Suite for Analyzing Large Language Models Across Training and
Scaling"). This repo redistributes derived weights unchanged in ONNX form.
