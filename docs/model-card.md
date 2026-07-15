---
license: apache-2.0
base_model:
  - EleutherAI/pythia-70m
  - openai-community/gpt2
  - m-a-p/neo_scalinglaw_250M
  - BAAI/Aquila-135M-Intermediate
  - bigscience/bloom-560m
  - HuggingFaceTB/SmolLM2-135M
  - Qwen/Qwen3-0.6B
  - facebook/opt-125m
  - google/gemma-3-270m
tags:
  - onnx
  - interpretability
  - logit-lens
  - training-dynamics
  - lenslapse
  - multilingual
---

# LensLapse ONNX checkpoints — Pythia suite across training, plus single-checkpoint architecture coverage

Browser-runnable ONNX conversions of 20 log-spaced **training checkpoints** each of
[EleutherAI/pythia-14m](https://huggingface.co/EleutherAI/pythia-14m),
[pythia-70m](https://huggingface.co/EleutherAI/pythia-70m), and
[pythia-160m](https://huggingface.co/EleutherAI/pythia-160m), packaged for the
[LensLapse](https://iamtatsuki05.github.io/lenslapse/) in-browser logit-lens demo. The remaining
directories are single-checkpoint (final-weights-only) examples that demonstrate the same
architecture-generic conversion recipe on models the flagship Pythia suite says nothing about —
three multilingual suites and, separately, four more decoder architecture families that don't
otherwise change the recipe's own claim (see Contents below for the full list and what's proved by
each one).

## Contents

One directory per model (`pythia-14m/`, `pythia-70m/`, `pythia-160m/`, `gpt2/`, `mapneo-250m/`,
`aquila-135m/`, `bloom-560m/`, `smollm2-135m/`, `qwen3-0.6b/`, `opt-125m/`, `gemma3-270m/`), each
with a `manifest.json` (steps, per-file sizes, export-time parity metrics).

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

`smollm2-135m/`, `qwen3-0.6b/`, `opt-125m/`, and `gemma3-270m/` each ship a single `step0/` pair —
the released final weights, final-only like `gpt2/` (no public intermediate checkpoints). Backbone
~273 / ~1197 / ~252 / ~540 MB, lens ~57 / ~311 / ~77 / ~336 MB (SmolLM2: hidden=576, 30 layers,
vocab=49152, Llama-style RMSNorm; Qwen3: hidden=1024, 28 layers, vocab=151936, Llama-style RMSNorm;
OPT: hidden=768, 12 layers, vocab=50272, GPT-2-style LayerNorm; Gemma 3: hidden=640, 18 layers,
vocab=262144, Gemma-style RMSNorm). Unlike the suites above, these four are included specifically as
**architecture-coverage examples** rather than for their training-dynamics story, and two of them
exposed real gaps in the conversion recipe rather than just confirming it:

- **SmolLM2 and Qwen3** already resolved correctly through the existing GPT-NeoX/Llama-style/GPT-2
  attribute heuristic with no code change.
- **OPT did not**: its decoder stack sits at `model.model.decoder`, two attribute hops below
  `model`, which the previous one-hop-only base search could not reach (`no block list found on
  OPTModel`); `lenslapse/arch.py` now resolves dotted multi-hop base paths, verified against OPT and
  regression-checked against every other shipped architecture. **`opt-125m/` carries a
  non-permissive, non-commercial license — see License below before using it.**
- **Gemma 3 did not either**: its RMSNorm computes `normalized(x) * (1 + weight)`, not the
  `normalized(x) * weight` every other shipped model uses (Gemma initializes the weight near zero
  rather than near one, for training stability, and adds the 1 back at call time). The export
  pipeline's ONNX reconstruction of the norm previously assumed the plain formula unconditionally,
  which for a real Gemma 3 checkpoint produced a 23.2 max logit deviation against the fp32 PyTorch
  reference — caught by the pipeline's own `assert fp32_diff < 0.05` rather than shipped silently.
  `export_checkpoints.py` now probes the real norm module against both candidate formulas on a
  fixed random vector and reconstructs whichever one it actually matches; fixed, `gemma3-270m/`
  passes with a 5.3e-5 max deviation. **`gemma3-270m/` carries Google's Gemma Terms of Use — see
  License below (commercial use is allowed, unlike `opt-125m/`, but with its own redistribution and
  prohibited-use obligations).**

The backbone outputs the **pre-final-layer-norm residual stream** after the embedding and after each
transformer block, so applying the lens head to any layer implements the logit lens uniformly, and
applying it to the last layer reproduces the model's actual output distribution exactly. This
identity — and the export pipeline's per-checkpoint assertion of it — holds unmodified for MAP-Neo's,
Aquila's, SmolLM2's, and Qwen3's Llama/Mistral-style RMSNorm architectures, Gemma 3's
plus-one-weight RMSNorm variant, and BLOOM's and OPT's GPT-2-style LayerNorm architectures: the
conversion recipe introspects the decoder stack, final norm, and unembedding generically
(`lenslapse/arch.py`) rather than hard-coding Pythia's GPT-NeoX layout.

Two further caveats the recipe does *not* paper over, both caught rather than silently shipped:
Gemma-2-style models (unlike Gemma 3) apply an additional `tanh`-based final-logit softcapping the
plain `final_norm → lm_head` path does not reproduce (raw logit deviation of ~51 on a real
checkpoint, though top-1 still agrees since `tanh` is monotonic) — no Gemma-2-family model is
included here. OLMo-2's smallest public checkpoint (1B parameters) was also attempted and rejected
for an unrelated reason: its backbone (embedding + transformer stack, ~1.28B parameters) serializes
to ~2.6 GB in fp16, over the 2 GB hard limit of a single protobuf message that plain `onnx.save()`
enforces; `export_checkpoints.py` does not currently use ONNX's external-data format, which would
lift this ceiling. No public OLMo-2 checkpoint small enough to fit is available at the time of
writing.

## Fidelity

Weights are stored fp16 and cast to fp32 at session load (compute is fp32). Against fp32 PyTorch,
top-1 lens predictions agree on 100.0% of (layer, position) cells over a 16-prompt suite at steps
8000/64000/143000; the export-time probe over all 20 checkpoints shows a max logit diff of 0.0087.
Dynamic int8 quantization was evaluated and rejected (final-layer top-1 agreement drops to 52%
per-tensor / 71% per-channel at step 143000). See the LensLapse repository for the evaluation
script (`lenslapse/fidelity_eval.py`).

MAP-Neo, Aquila, BLOOM, SmolLM2, Qwen3, OPT, and Gemma 3 were validated with the same per-checkpoint
export-time assertion (not the separate 16-prompt/int8 sweep above, which is Pythia-specific):
top-1 agreement between the fp32 PyTorch reference and both the fp32-ONNX and fp16-ONNX exports is
100.0% at every (layer, position) cell of the probe prompt, for every shipped checkpoint of all
seven models. Max logit deviation from fp32 PyTorch: MAP-Neo 2.6e-5 to 1.6e-4 across its 8
checkpoints; Aquila 1.1e-4 to 1.8e-4 across its 6; BLOOM 8.6e-6 to 7.6e-4 across its 8; SmolLM2
3.03e-4; Qwen3 1.02e-4; OPT 3.19e-5; Gemma 3 270M 5.34e-5.

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
`--subfolder-map`, `--revision-template`, `--tokenizer-ref`, and `lenslapse/prompts_zh_en.json` (the
combined English + Chinese curated-prompt set used for these three multilingual models, in place of
the English-only default) for the exact commands.

## License and attribution

**⚠️ Three directories here are not plain Apache-2.0/MIT — read this section before redistributing
or deploying `bloom-560m/`, `opt-125m/`, or `gemma3-270m/` specifically.**

- **`opt-125m/` is the one directory in this repo that is *not* usable commercially.** It is
  derived from [facebook/opt-125m](https://huggingface.co/facebook/opt-125m), © Meta Platforms,
  under the [OPT-175B License Agreement](https://huggingface.co/facebook/opt-125m/blob/main/LICENSE.md)
  — a custom, non-permissive license granting rights "solely for your non-commercial research
  purposes." No commercial use is permitted at all.
- **`gemma3-270m/` is derived from [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m),
  © Google, under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).** Commercial use
  *is* permitted (unlike `opt-125m/`), but redistribution — which this repo does, in ONNX form —
  requires passing the Agreement and its referenced
  [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) on to downstream
  recipients as an enforceable term, carrying prominent notice of any modification, and remains
  subject to Google's unilateral right to restrict usage or terminate the Agreement for a violation
  — obligations no other license in this repo imposes on downstream users.
- **`bloom-560m/` carries the BigScience RAIL License v1.0** (see below) — broad, including
  commercial, use subject to behavioral restrictions (Attachment A), with no flow-down or
  termination clause like Gemma's.

Every other directory in this repo is Apache-2.0 or MIT with no comparable restrictions. Read each
license in full before using the corresponding directory beyond casual/research inspection.

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
A) that bind downstream recipients of these weights.** The `smollm2-135m/` directory is derived from
[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), Apache-2.0 © Hugging
Face. The `qwen3-0.6b/` directory is derived from
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Apache-2.0 © Alibaba Cloud / the Qwen
team. The `opt-125m/` and `gemma3-270m/` directories are derived from
[facebook/opt-125m](https://huggingface.co/facebook/opt-125m) (© Meta Platforms) and
[google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) (© Google) respectively — see the
warning at the top of this section for both. Apart from `bloom-560m/`, `opt-125m/`, and
`gemma3-270m/`, every directory in this repo is Apache-2.0 or MIT. The `license` metadata field
above names the Pythia license; the other directories keep their own, and downstream users of
`bloom-560m/`, `opt-125m/`, and `gemma3-270m/` specifically should read those licenses in full
before deploying them. This repo redistributes derived weights unchanged in ONNX form.
