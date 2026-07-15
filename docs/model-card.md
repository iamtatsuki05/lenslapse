---
license: apache-2.0
base_model:
  - EleutherAI/pythia-70m
  - openai-community/gpt2
  - m-a-p/neo_scalinglaw_250M
  - BAAI/Aquila-135M-Intermediate
  - bigscience/bloom-560m
tags:
  - onnx
  - interpretability
  - logit-lens
  - training-dynamics
  - lenslapse
  - multilingual
---

# LensLapse ONNX checkpoints — Pythia suite across training, plus GPT-2 and three multilingual suites

Browser-runnable ONNX conversions of 20 log-spaced **training checkpoints** each of
[EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m),
[pythia-70m](https://huggingface.co/EleutherAI/pythia-70m), and
[pythia-160m](https://huggingface.co/EleutherAI/pythia-160m), plus the final checkpoint of
[GPT-2 124M](https://huggingface.co/openai-community/gpt2), packaged for the
[LensLapse](https://iamtatsuki05.github.io/lenslapse/) in-browser logit-lens demo. Three
multilingual checkpoint suites are also included, demonstrating that the conversion recipe
generalizes beyond English-only, GPT-NeoX-only training runs:
[m-a-p/neo_scalinglaw_250M](https://huggingface.co/m-a-p/neo_scalinglaw_250M) (Chinese/English,
Llama-style), [BAAI/Aquila-135M-Intermediate](https://huggingface.co/BAAI/Aquila-135M-Intermediate)
(Chinese/English, Mistral-style), and
[bigscience/bloom-560m-intermediate](https://huggingface.co/bigscience/bloom-560m-intermediate)
(46 natural languages including Chinese, LayerNorm+ALiBi — **not** Japanese; see License below for
its non-Apache terms).

## Contents

One directory per model (`pythia-14m/`, `pythia-70m/`, `pythia-160m/`, `gpt2/`, `mapneo-250m/`,
`aquila-135m/`, `bloom-560m/`), each with a `manifest.json` (steps, per-file sizes, export-time
parity metrics).

The three Pythia directories hold one `step{N}/` pair for each training step
`N` ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000,
128000, 143000}:

| file | interface | size (14m / 70m / 160m) |
|---|---|---|
| `step{N}/backbone.f16.onnx` | `input_ids, attention_mask → hidden_states [L+1, B, T, H]` | ~16 / ~91 / ~249 MB |
| `step{N}/lens.f16.onnx` | `hidden [N, H] → logits [N, V]` (final norm + unembedding) | ~13 / ~52 / ~77 MB |

`gpt2/` ships a single `step0/` pair — the released final weights (~250 MB backbone + ~77 MB lens);
GPT-2 has no public intermediate checkpoints.

`mapneo-250m/` holds 8 `step{N}/` pairs, `N` ∈ {16780, 33550, 67110, 100660, 201330, 402650, 704640,
1002650} — **millions of training tokens**, not optimizer steps (MAP-Neo's own published checkpoints
are labeled by cumulative tokens, e.g. `16.78B`; the LensLapse slider needs a single monotonic
integer axis, so the token count in millions is used directly as the step value). Backbone ~372 MB,
lens ~131 MB per step (hidden=1024, 8 layers, vocab=64128).

`aquila-135m/` holds 6 `step{N}/` pairs at the model's own published training iterations,
`N` ∈ {250000, 500000, 750000, 1000000, 1300000, 1410844} (the "pretrain" phase only — Aquila also
publishes a separate, non-contiguous 3-checkpoint "annealing" phase at lower iteration numbers,
excluded here to keep the step axis monotonic). Backbone ~392 MB, lens ~175 MB per step (hidden=576,
30 layers, vocab=151851).

`bloom-560m/` holds 8 `step{N}/` pairs at the model's own published training steps,
`N` ∈ {1000, 10000, 100000, 200000, 300000, 400000, 500000, 600000} (BLOOM's own checkpoint
revisions are named `global_step{N}`, matching Pythia's convention closely enough that no synthetic
step relabeling was needed). Backbone ~1120 MB, lens ~514 MB per step (hidden=1024, 24 layers,
vocab=250880).

The backbone outputs the **pre-final-layer-norm residual stream** after the embedding and after each
transformer block, so applying the lens head to any layer implements the logit lens uniformly, and
applying it to the last layer reproduces the model's actual output distribution exactly. This
identity — and the export pipeline's per-checkpoint assertion of it — holds unmodified for MAP-Neo's
Llama-style RMSNorm architecture, Aquila's Mistral-style RMSNorm architecture, and BLOOM's
GPT-2-style LayerNorm-with-ALiBi architecture: the conversion recipe introspects the decoder stack,
final norm, and unembedding generically (`lenslapse/arch.py`) rather than hard-coding Pythia's
GPT-NeoX layout.

## Fidelity

Weights are stored fp16 and cast to fp32 at session load (compute is fp32). Against fp32 PyTorch,
top-1 lens predictions agree on 100.0% of (layer, position) cells over a 16-prompt suite at steps
8000/64000/143000; the export-time probe over all 20 checkpoints shows a max logit diff of 0.0087.
Dynamic int8 quantization was evaluated and rejected (final-layer top-1 agreement drops to 52%
per-tensor / 71% per-channel at step 143000). See the LensLapse repository for the evaluation
script (`lenslapse/fidelity_eval.py`).

MAP-Neo, Aquila, and BLOOM were validated with the same per-checkpoint export-time assertion (not
the separate 16-prompt/int8 sweep above, which is Pythia-specific): top-1 agreement between the fp32
PyTorch reference and both the fp32-ONNX and fp16-ONNX exports is 100.0% at every (layer, position)
cell of the probe prompt, for every shipped checkpoint of all three models. Max logit deviation from
fp32 PyTorch: MAP-Neo 2.6e-5 to 1.6e-4 across its 8 checkpoints; Aquila 1.1e-4 to 1.8e-4 across its
6; BLOOM 8.6e-6 to 7.6e-4 across its 8.

## Reproduce

```bash
uv run python -m lenslapse.export_checkpoints --model EleutherAI/pythia-70m --out ./models
uv run lenslapse add-model --model gpt2 --id gpt2 --label "GPT-2 124M" --final-only --models-root ./models
```

MAP-Neo and Aquila publish checkpoints as subfolders within a single revision rather than one git
revision per checkpoint, Aquila's tokenizer requires `trust_remote_code`, and BLOOM's per-revision
tokenizer files fail to load under current `transformers` (a `transformers`-side regression, not a
BLOOM error — its tokenizer is loaded from the final `bigscience/bloom-560m` repo instead, identical
across checkpoints of the same run); see `lenslapse/sources.py`'s `resolve_subfolder_sources` /
`--subfolder-map`, `--revision-template`, `--tokenizer-ref`, and `lenslapse/prompts_zh.json` (the
Chinese curated-prompt set used in place of the English default) for the exact commands.

## License and attribution

Conversion scripts MIT. The `pythia-*/` directories are derived from Pythia weights,
Apache-2.0 © EleutherAI (Biderman et al., 2023, "Pythia: A Suite for Analyzing Large Language
Models Across Training and Scaling"); the `gpt2/` directory is derived from
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2), MIT (Modified) © OpenAI
(Radford et al., 2019). The `mapneo-250m/` directory is derived from
[m-a-p/neo_scalinglaw_250M](https://huggingface.co/m-a-p/neo_scalinglaw_250M), Apache-2.0 © the
M-A-P community (Zhang et al., 2024, "MAP-Neo: Highly Capable and Transparent Bilingual Large
Language Model Series"); its tokenizer is redistributed from the separate
[m-a-p/neo_7b](https://huggingface.co/m-a-p/neo_7b) repository (same license, same authors — the
scaling-law repos do not bundle a tokenizer). The `aquila-135m/` directory is derived from
[BAAI/Aquila-135M-Intermediate](https://huggingface.co/BAAI/Aquila-135M-Intermediate), Apache-2.0 ©
Beijing Academy of Artificial Intelligence. **The `bloom-560m/` directory is derived from
[bigscience/bloom-560m-intermediate](https://huggingface.co/bigscience/bloom-560m-intermediate),
© BigScience Workshop, licensed under the [BigScience RAIL License
v1.0](https://huggingface.co/spaces/bigscience/license) — *not* a plain permissive license: RAIL
grants broad use and redistribution rights but attaches use-based behavioral restrictions (Attachment
A) that bind downstream recipients of these weights, unlike the Apache-2.0/MIT terms on every other
directory in this repo.** The `license` metadata field above names the Pythia license; the other
directories keep their own, and downstream users of `bloom-560m/` specifically should read the RAIL
license's Attachment A before deploying it. This repo redistributes derived weights unchanged in
ONNX form.
