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

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .arch import resolve
from .sources import resolve_sources, resolve_subfolder_sources, resolve_tokenizer_ref, token_display_text

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument(
        "--steps", default="0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--final-only", action="store_true", help="single checkpoint (revision main) instead of a step suite"
    )
    ap.add_argument(
        "--subfolder-map",
        default=None,
        help='hub-subfolder suite: "step:path,step:path,..." — see export_checkpoints.py; overrides --steps',
    )
    ap.add_argument(
        "--prompts-file",
        default=None,
        help="JSON file with the same shape as PROMPTS (list of {text, gold, story}); "
        "defaults to the built-in English curated set",
    )
    ap.add_argument(
        "--revision-template",
        default="step{}",
        help='revision naming for hub suites, e.g. "global_step{}" for bigscience/bloom-*-intermediate',
    )
    ap.add_argument(
        "--tokenizer-ref",
        default=None,
        help="load the tokenizer from a different ref than the checkpoint weights; see export_checkpoints.py",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = (
        resolve_subfolder_sources(args.model, args.subfolder_map)
        if args.subfolder_map
        else resolve_sources(args.model, args.steps, args.final_only, args.revision_template)
    )
    by_step = {src.step: src for src in sources}
    steps = sorted(by_step)
    final_step = steps[-1]
    tok_load_ref, tok_rev, tok_subfolder = resolve_tokenizer_ref(args.tokenizer_ref, sources[0])
    tok = AutoTokenizer.from_pretrained(
        tok_load_ref, revision=tok_rev, subfolder=tok_subfolder, trust_remote_code=True
    )

    prompt_defs = json.loads(Path(args.prompts_file).read_text()) if args.prompts_file else PROMPTS
    prompts: list[dict[str, Any]] = []
    for i, p in enumerate(prompt_defs):
        ids = tok(p["text"])["input_ids"]
        # no special tokens: a BOS-prepending tokenizer (Llama-style) would make gold_id the BOS id
        gold_id = tok(p["gold"], add_special_tokens=False)["input_ids"][0]
        prompts.append({"id": i, **p, "ids": ids, "gold_id": gold_id})

    # Pass 1: final step first to fix per-position targets.
    order = [final_step] + [s for s in steps if s != final_step]
    targets: dict[Any, list[list[int]]] = {}  # prompt_id -> [pos] -> target ids
    shards: dict[Any, dict[str, Any]] = {i: {"vocab": {}, "steps": {}} for i in range(len(prompts))}

    for step in order:
        src = by_step[step]
        model = AutoModelForCausalLM.from_pretrained(
            src.load_ref, revision=src.revision, subfolder=src.subfolder or "", dtype=torch.float32
        )
        model.eval()
        for p in prompts:
            ids = torch.tensor([p["ids"]])
            lp = lens_all(model, ids)  # [L+1, T, V] log-probs
            probs = lp.exp()
            top = torch.topk(probs, TOPK, dim=-1)  # values/indices [L+1, T, K]

            if step == final_step and p["id"] not in targets:
                fin = top.indices[-1, :, :3].tolist()  # final layer top-3 per position
                tg = [list(dict.fromkeys(row)) for row in fin]
                tg[-1] = list(dict.fromkeys(tg[-1] + [p["gold_id"]]))
                targets[p["id"]] = tg

            sh = shards[p["id"]]
            tg = targets[p["id"]]
            entry: dict[str, Any] = {
                "top": [
                    [
                        [[int(top.indices[li, t, k]), round(float(top.values[li, t, k]), 5)] for k in range(TOPK)]
                        for t in range(lp.shape[1])
                    ]
                    for li in range(lp.shape[0])
                ],
                "tgt": {},
            }
            all_tgt_ids = sorted({i for row in tg for i in row})
            for tid in all_tgt_ids:
                pvals = probs[:, :, tid]  # [L+1, T]
                ranks = (lp > lp[:, :, tid : tid + 1]).sum(dim=-1) + 1  # [L+1, T]
                entry["tgt"][str(tid)] = {
                    "p": [[round(float(x), 6) for x in row] for row in pvals.tolist()],
                    "r": [[int(x) for x in row] for row in ranks.tolist()],
                }
            sh["steps"][str(step)] = entry
            ref_ids = {int(i) for layer in entry["top"] for pos in layer for i, _ in pos} | set(all_tgt_ids)
            for tid in ref_ids:
                sh["vocab"].setdefault(str(tid), token_display_text(tok, tok.convert_ids_to_tokens([tid])[0]))
        del model
        print(f"[step{step}] done", flush=True)

        # persist incrementally so partial runs are usable
        for p in prompts:
            (out_dir / f"p{p['id']}.json").write_text(json.dumps(shards[p["id"]], separators=(",", ":")))
        index = {
            "model": args.model,
            "steps": [s for s in steps if str(s) in shards[0]["steps"]],
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
        (out_dir / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
