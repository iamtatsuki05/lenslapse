"""Export Pythia training checkpoints as browser-runnable ONNX pairs (backbone + lens head).

For each requested training step, produces:
  out_dir/step{N}/backbone.f16.onnx  input_ids, attention_mask -> hidden_states [L+1, B, T, H] (pre-ln_f, uniform)
  out_dir/step{N}/lens.f16.onnx      hidden [N, H] -> logits [N, V]  (final_layer_norm + unembedding)
and a top-level manifest.json.

Design notes (validated 2026-07-13, see experiments/feasibility-note.md):
- HF GPTNeoX applies final_layer_norm to the last entry of output_hidden_states, so a uniform lens
  head would double-normalize it. Forward hooks capture each block's raw (pre-ln) output instead;
  lens(hidden[-1]) then equals the model's logits exactly.
- The lens head is built by hand with onnx.helper (LayerNormalization + MatMul): torch.onnx.export
  emits a graph that fails ORT shape inference during quantization.
- opset >= 18 is required by the torch dynamo exporter (Split num_outputs).
- Weights are stored as fp16 + Cast (see onnx_f16.py); compute stays fp32. Dynamic int8 was
  rejected by fidelity_eval.py: final-layer top-1 agreement vs fp32 drops to 52-76% at late
  checkpoints, while fp16 weight storage keeps 100.0% agreement at half the fp32 size.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from transformers import AutoModelForCausalLM, AutoTokenizer

from arch import resolve
from onnx_f16 import save_f16
from sources import resolve_sources

PROBE = "The capital of Japan is the city of"


class Backbone(torch.nn.Module):
    def __init__(self, m: AutoModelForCausalLM):
        super().__init__()
        handles = resolve(m)
        self.base = handles.base
        self.blocks = handles.layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        captured: list[torch.Tensor] = []

        def cap(_m, _i, o):
            captured.append(o[0] if isinstance(o, tuple) else o)

        hooks = [layer.register_forward_hook(cap) for layer in self.blocks]
        try:
            out = self.base(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
            )
        finally:
            for h in hooks:
                h.remove()
        return torch.stack([out.hidden_states[0], *captured], dim=0)


def build_lens_onnx(model: AutoModelForCausalLM, path: Path) -> None:
    """Lens head = final norm (LayerNorm or RMSNorm, composed from primitive ops) + unembedding."""
    handles = resolve(model)
    h = model.config.hidden_size
    norm = handles.final_norm
    w_unembed = handles.lm_head.weight.detach().numpy().T.copy()  # [H, V]
    inits = [numpy_helper.from_array(norm.weight.detach().numpy(), "norm_w")]
    if handles.norm_type == "layernorm":
        norm_bias = norm.bias.detach().numpy() if getattr(norm, "bias", None) is not None else np.zeros(h, np.float32)
        inits.append(numpy_helper.from_array(norm_bias, "norm_b"))
        nodes = [
            helper.make_node(
                "LayerNormalization", ["hidden", "norm_w", "norm_b"], ["normed"], axis=-1, epsilon=handles.eps
            ),
        ]
    else:  # rmsnorm: x * rsqrt(mean(x^2) + eps) * w
        inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes"))
        inits.append(numpy_helper.from_array(np.array(handles.eps, dtype=np.float32), "eps"))
        nodes = [
            helper.make_node("Mul", ["hidden", "hidden"], ["sq"]),
            helper.make_node("ReduceMean", ["sq", "axes"], ["ms"], keepdims=1),
            helper.make_node("Add", ["ms", "eps"], ["ms_eps"]),
            helper.make_node("Sqrt", ["ms_eps"], ["rms"]),
            helper.make_node("Div", ["hidden", "rms"], ["xn"]),
            helper.make_node("Mul", ["xn", "norm_w"], ["normed"]),
        ]
    head_bias = getattr(handles.lm_head, "bias", None)
    if head_bias is not None:
        inits.append(numpy_helper.from_array(head_bias.detach().numpy(), "head_b"))
        nodes += [
            helper.make_node("MatMul", ["normed", "w_unembed"], ["logits_nobias"]),
            helper.make_node("Add", ["logits_nobias", "head_b"], ["logits"]),
        ]
    else:
        nodes.append(helper.make_node("MatMul", ["normed", "w_unembed"], ["logits"]))
    inits.append(numpy_helper.from_array(w_unembed, "w_unembed"))
    graph = helper.make_graph(
        nodes,
        "lens_head",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["n", h])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", w_unembed.shape[1]])],
        initializer=inits,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def export_source(src, out_dir: Path, tok) -> dict:
    rev = src.name
    model = AutoModelForCausalLM.from_pretrained(src.load_ref, revision=src.revision, dtype=torch.float32)
    model.eval()
    bb = Backbone(model)

    enc = tok(PROBE, return_tensors="pt")
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        ref_logits = model(input_ids=input_ids, attention_mask=attn).logits[0].numpy()

    step_dir = out_dir / rev
    step_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        bb_fp32 = Path(td) / "backbone.onnx"
        lens_fp32 = Path(td) / "lens.onnx"
        torch.onnx.export(
            bb,
            (input_ids, attn),
            str(bb_fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "b", 1: "t"},
                "attention_mask": {0: "b", 1: "t"},
                "hidden_states": {1: "b", 2: "t"},
            },
            opset_version=18,
        )
        build_lens_onnx(model, lens_fp32)

        # fp32 parity: lens(hidden[-1]) must match torch logits. Late checkpoints have large logit
        # magnitudes, so the absolute bound is a loose sanity check; the meaningful gate is top-1
        # agreement (recorded below) plus the systematic fidelity_eval.py evidence.
        sb = ort.InferenceSession(str(bb_fp32))
        sl = ort.InferenceSession(str(lens_fp32))
        hs = sb.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attn.numpy()})[0]
        lo = sl.run(None, {"hidden": hs[-1, 0]})[0]
        fp32_diff = float(np.abs(ref_logits - lo).max())
        assert fp32_diff < 0.05, f"{rev}: fp32 parity failed ({fp32_diff})"
        fp32_top1 = bool((lo.argmax(-1) == ref_logits.argmax(-1)).all())
        assert fp32_top1, f"{rev}: fp32 ONNX top-1 disagrees with torch on the probe"

        bb_f16 = step_dir / "backbone.f16.onnx"
        lens_f16 = step_dir / "lens.f16.onnx"
        save_f16(str(bb_fp32), str(bb_f16))
        save_f16(str(lens_fp32), str(lens_f16))

    # f16 sanity: top-1 must match torch fp32 at every position of the probe
    sbf = ort.InferenceSession(str(bb_f16))
    slf = ort.InferenceSession(str(lens_f16))
    hf = sbf.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attn.numpy()})[0]
    L1, _, T, H = hf.shape
    lf = slf.run(None, {"hidden": hf[-1, 0]})[0]
    f16_diff = float(np.abs(ref_logits - lf).max())
    top1_match = bool((lf.argmax(-1) == ref_logits.argmax(-1)).all())
    assert top1_match, f"{rev}: f16 top-1 disagrees with torch on the probe — do not ship this checkpoint"

    info = {
        "step": src.step,
        "fp32_max_diff": fp32_diff,
        "fp32_top1_match_all_pos": fp32_top1,
        "f16_max_diff": f16_diff,
        "f16_top1_match_all_pos": top1_match,
        "top1_token": tok.convert_ids_to_tokens([int(ref_logits[-1].argmax())])[0],
        "backbone_bytes": bb_f16.stat().st_size,
        "lens_bytes": lens_f16.stat().st_size,
    }
    print(
        f"[{rev}] fp32_diff={fp32_diff:.2e} f16_diff={f16_diff:.2e} top1_match={top1_match} "
        f"(top1={info['top1_token']!r}) "
        f"bb={info['backbone_bytes'] / 1e6:.1f}MB lens={info['lens_bytes'] / 1e6:.1f}MB",
        flush=True,
    )
    del model
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument(
        "--steps", default="0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument(
        "--final-only", action="store_true", help="single checkpoint (revision main) instead of a step suite"
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = resolve_sources(args.model, args.steps, args.final_only)
    tok = AutoTokenizer.from_pretrained(sources[0].load_ref, revision=sources[0].revision)

    cfg_model = AutoModelForCausalLM.from_pretrained(
        sources[0].load_ref, revision=sources[0].revision, dtype=torch.float32
    )
    meta = {
        "model": args.model,
        "format": "f16",
        "files": ["backbone.f16.onnx", "lens.f16.onnx"],
        "hidden_size": cfg_model.config.hidden_size,
        "num_layers": cfg_model.config.num_hidden_layers,
        "vocab_size": cfg_model.config.vocab_size,
        "tokenizer": args.model,
        "probe": PROBE,
        "steps": [],
    }
    del cfg_model

    # merge with a previous run's manifest so incremental exports never drop existing checkpoints
    manifest_path = out_dir / "manifest.json"
    results: dict[int, dict] = {}
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
            if old.get("model") == args.model:
                results = {s["step"]: s for s in old.get("steps", [])}
            else:
                raise SystemExit(
                    f"{out_dir} already holds a manifest for {old.get('model')!r}; refusing to mix models"
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise SystemExit(f"{manifest_path} is unreadable ({e}); fix or delete it before re-exporting") from e

    for src in sources:
        step = src.step
        step_dir = out_dir / src.name
        complete = all((step_dir / f).exists() for f in meta["files"])
        if args.skip_existing and complete and step in results:
            print(f"[{src.name}] exists, skipping", flush=True)
            continue
        results[step] = export_source(src, out_dir, tok)
        meta["steps"] = [results[k] for k in sorted(results)]
        manifest_path.write_text(json.dumps(meta, indent=1))

    meta["steps"] = [results[k] for k in sorted(results)]
    manifest_path.write_text(json.dumps(meta, indent=1))
    print("ALL_DONE steps:", len(meta["steps"]), flush=True)


if __name__ == "__main__":
    main()
