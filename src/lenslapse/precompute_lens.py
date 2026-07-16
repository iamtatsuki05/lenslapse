"""Precompute logit-lens data for curated prompts across Pythia training checkpoints.

Output layout (static JSON consumed by the web app):
  out_dir/index.json   model metadata, step list, prompt catalog (tokens, gold continuation, targets)
  out_dir/p{i}.json    per-prompt shard:
                         vocab: {token_id: token_string} for every id referenced
                         steps: {step: {"top": [layer][pos][k] -> [id, prob],
                                        "tgt": {target_id: {"p": [layer][pos], "r": [layer][pos]}}}}

Targets per position = top-3 final-layer predictions of the LAST step in --steps (+ gold next token
for the last position), so trajectories are exact for the tokens the model eventually converges to.
The last step is processed first to fix targets; every step then stores exact prob/rank for them.

Lens convention: hidden states are the raw (pre-final_layer_norm) residual stream after each block,
plus embeddings; lens(h) = embed_out(final_layer_norm(h)). lens at the last layer equals the model's
actual output distribution (validated in export_checkpoints.py).
"""

import json
import logging
from pathlib import Path
from typing import Any

import fire
import torch
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM

from lenslapse.arch import resolve
from lenslapse.logging_utils import configure_cli_logging
from lenslapse.sources import (
    DEFAULT_STEPS_CSV,
    coerce_fire_csv_arg,
    load_tokenizer,
    resolve_all_sources,
    resolve_tokenizer_ref,
    token_display_text,
)

logger = logging.getLogger(__name__)

PROMPTS = [
    {"text": "The capital of Japan is the city of", "gold": " Tokyo", "story": "fact"},
    {"text": "The Eiffel Tower is located in the city of", "gold": " Paris", "story": "fact"},
    {"text": "Water is made of hydrogen and", "gold": " oxygen", "story": "fact"},
    {"text": "The first president of the United States was George", "gold": " Washington", "story": "fact"},
    {"text": "The opposite of hot is", "gold": " cold", "story": "relation"},
    {"text": "Paris is to France as Tokyo is to", "gold": " Japan", "story": "relation"},
    {"text": "Two plus two equals", "gold": " four", "story": "math"},
    {"text": "3 + 4 =", "gold": " 7", "story": "math"},
    {"text": "The quick brown fox jumps over the lazy", "gold": " dog", "story": "idiom"},
    {"text": "Once upon a", "gold": " time", "story": "idiom"},
    {"text": "Thank you very", "gold": " much", "story": "idiom"},
    {"text": "The keys to the cabinet", "gold": " are", "story": "syntax"},
    {"text": "def add(a, b):\n    return a +", "gold": " b", "story": "code"},
    {"text": "import numpy as", "gold": " np", "story": "code"},
    {"text": "The DNA molecule has the shape of a double", "gold": " helix", "story": "fact"},
    {
        "text": "Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly",
        "gold": " normal",
        "story": "copy",
    },
]

TOPK = 10


@torch.no_grad()
def lens_all(model: "torch.nn.Module", input_ids: "torch.Tensor") -> "torch.Tensor":
    """Returns per-layer pre-final-norm hidden states passed through the lens head. [L+1, T, V] log-probs."""
    handles = resolve(model)
    captured: list[torch.Tensor] = []

    def cap(_m: object, _i: object, o: object) -> None:
        captured.append(o[0] if isinstance(o, tuple) else o)

    hooks = [layer.register_forward_hook(cap) for layer in handles.layers]
    try:
        out = handles.base(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    # under device_map sharding the per-layer outputs live on different devices; normalize to
    # the embedding's device for the stack, then to the lens head's device (no-ops otherwise)
    dev = out.hidden_states[0].device
    hs = torch.stack([out.hidden_states[0], *[c.to(dev) for c in captured]], dim=0)[:, 0]  # [L+1, T, H]
    lens_dev = next(handles.final_norm.parameters(), next(handles.lm_head.parameters())).device
    logits = handles.lm_head(handles.final_norm(hs.to(lens_dev)))  # [L+1, T, V]
    return torch.log_softmax(logits.float(), dim=-1)


def target_stats(lp: "torch.Tensor", tid: int) -> dict[str, list[list[Any]]]:
    """Exact probability (rounded to 6 decimals) and strictly-greater rank of one target token per
    (layer, position), from `lp` [L+1, T, V] log-probs. The precomputed shards and the probe
    server's /probe both serialize targets through this one function: the app overlays live and
    precomputed trajectories, so the two paths must agree to the digit."""
    ranks = (lp > lp[:, :, tid : tid + 1]).sum(dim=-1) + 1  # [L+1, T]
    return {
        "p": [[round(float(x), 6) for x in row] for row in lp[:, :, tid].exp().tolist()],
        "r": [[int(x) for x in row] for row in ranks.tolist()],
    }


class PrecomputeConfig(BaseModel):
    """Validated arguments for `precompute_lens`; see `main`'s docstring for what each means."""

    model: str = "EleutherAI/pythia-70m"
    steps: str = DEFAULT_STEPS_CSV
    out: Path
    final_only: bool = False
    subfolder_map: str | None = None
    prompts_file: str | None = None
    revision_template: str = "step{}"
    tokenizer_ref: str | None = None

    _coerce_steps = field_validator("steps", mode="before")(coerce_fire_csv_arg)


def main(
    out: str,
    model: str = "EleutherAI/pythia-70m",
    steps: str = DEFAULT_STEPS_CSV,
    final_only: bool = False,
    subfolder_map: str | None = None,
    prompts_file: str | None = None,
    revision_template: str = "step{}",
    tokenizer_ref: str | None = None,
) -> None:
    """Precompute per-prompt logit-lens shards across a model's training checkpoints.

    Args:
        out: output directory for `index.json` and per-prompt `p{i}.json` shards.
        model: HF id or local directory.
        steps: comma-separated training steps for a hub suite (ignored if `subfolder_map` is set).
        final_only: single checkpoint (revision "main") instead of a step suite.
        subfolder_map: "step:path,step:path,..." for repos that nest checkpoints as subfolders of
            one revision instead of using git revisions per checkpoint; overrides `steps`.
        prompts_file: JSON file with the same shape as PROMPTS (list of {text, gold, story});
            defaults to the built-in English curated set.
        revision_template: revision naming for hub suites, e.g. "global_step{}" for
            ``bigscience/bloom-*-intermediate``.
        tokenizer_ref: load the tokenizer from a different ref than the checkpoint weights, as
            "repo_id" or "repo_id@revision"; see export_checkpoints.py.
    """
    cfg = PrecomputeConfig(
        model=model,
        steps=steps,
        out=out,  # type: ignore[arg-type]  # pydantic coerces str -> Path
        final_only=final_only,
        subfolder_map=subfolder_map,
        prompts_file=prompts_file,
        revision_template=revision_template,
        tokenizer_ref=tokenizer_ref,
    )

    cfg.out.mkdir(parents=True, exist_ok=True)
    sources = resolve_all_sources(cfg.model, cfg.steps, cfg.final_only, cfg.subfolder_map, cfg.revision_template)
    by_step = {src.step: src for src in sources}
    steps_sorted = sorted(by_step)
    final_step = steps_sorted[-1]
    tok_load_ref, tok_rev, tok_subfolder = resolve_tokenizer_ref(cfg.tokenizer_ref, sources[0])
    tok = load_tokenizer(tok_load_ref, tok_rev, tok_subfolder)

    prompt_defs = json.loads(Path(cfg.prompts_file).read_text()) if cfg.prompts_file else PROMPTS
    prompts: list[dict[str, Any]] = []
    for i, p in enumerate(prompt_defs):
        # the model's own default (special tokens included — gemma3's shards really start with
        # <bos>): this is the GRID tokenization convention, shared with the probe server's /probe
        # and the browser's in-ONNX probe, so live and precomputed grids line up position for
        # position. Token-PICKING paths (gold below, /tokenize, the app's track feature) instead
        # disable special tokens — they must never select a BOS.
        ids = tok(p["text"])["input_ids"]
        gold_id = tok(p["gold"], add_special_tokens=False)["input_ids"][0]
        prompts.append({"id": i, **p, "ids": ids, "gold_id": gold_id})

    # Pass 1: final step first to fix per-position targets.
    order = [final_step] + [s for s in steps_sorted if s != final_step]
    targets: dict[Any, list[list[int]]] = {}  # prompt_id -> [pos] -> target ids
    shards: dict[Any, dict[str, Any]] = {i: {"vocab": {}, "steps": {}} for i in range(len(prompts))}

    for step in order:
        src = by_step[step]
        step_model = AutoModelForCausalLM.from_pretrained(
            src.load_ref, revision=src.revision, subfolder=src.subfolder or "", dtype=torch.float32
        )
        step_model.eval()
        for p in prompts:
            ids = torch.tensor([p["ids"]])
            lp = lens_all(step_model, ids)  # [L+1, T, V] log-probs
            # top-k on log-probs directly (exp is monotone → identical indices), avoiding a full
            # [L+1, T, V] probability tensor. tolist() the top-k once and index Python lists in the
            # comprehension rather than scalar-indexing the tensor L*T*K times. Verified identical
            # to the old exp-then-topk path on real checkpoints; indices could differ only when
            # top-k tail probabilities underflow to 0.0 in float32 (ties among prob-0 tokens).
            top = torch.topk(lp, TOPK, dim=-1)  # values(log-probs)/indices [L+1, T, K]
            top_idx = top.indices.tolist()
            top_prob = top.values.exp().tolist()

            if step == final_step and p["id"] not in targets:
                fin = [row[:3] for row in top_idx[-1]]  # final layer top-3 per position
                tg = [list(dict.fromkeys(row)) for row in fin]
                tg[-1] = list(dict.fromkeys(tg[-1] + [p["gold_id"]]))
                targets[p["id"]] = tg

            sh = shards[p["id"]]
            tg = targets[p["id"]]
            entry: dict[str, Any] = {
                "top": [
                    [
                        [[top_idx[li][t][k], round(top_prob[li][t][k], 5)] for k in range(TOPK)]
                        for t in range(lp.shape[1])
                    ]
                    for li in range(lp.shape[0])
                ],
                "tgt": {},
            }
            all_tgt_ids = sorted({i for row in tg for i in row})
            for tid in all_tgt_ids:
                entry["tgt"][str(tid)] = target_stats(lp, tid)
            sh["steps"][str(step)] = entry
            ref_ids = {int(i) for layer in entry["top"] for pos in layer for i, _ in pos} | set(all_tgt_ids)
            for tid in ref_ids:
                sh["vocab"].setdefault(str(tid), token_display_text(tok, tok.convert_ids_to_tokens([tid])[0]))
        del step_model
        logger.info("[step%d] done", step)

        # persist incrementally so partial runs are usable
        for p in prompts:
            (cfg.out / f"p{p['id']}.json").write_text(json.dumps(shards[p["id"]], separators=(",", ":")))
        index = {
            "model": cfg.model,
            "steps": [s for s in steps_sorted if str(s) in shards[0]["steps"]],
            "prompts": [
                {
                    "id": p["id"],
                    "text": p["text"],
                    "gold": p["gold"],
                    "gold_id": p["gold_id"],
                    "story": p["story"],
                    "ids": p["ids"],
                    "tokens": [token_display_text(tok, t) for t in tok.convert_ids_to_tokens(p["ids"])],
                    "targets": targets.get(p["id"], []),
                }
                for p in prompts
            ],
        }
        (cfg.out / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    # the completion signal, not a diagnostic message: scripts pipe stdout and grep for this,
    # so it must stay on print() rather than move to the logger (which defaults to stderr).
    print("ALL_DONE")


if __name__ == "__main__":
    configure_cli_logging()
    fire.Fire(main)
