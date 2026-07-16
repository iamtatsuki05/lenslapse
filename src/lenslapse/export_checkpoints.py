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

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import fire
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM

from lenslapse.arch import resolve
from lenslapse.logging_utils import configure_cli_logging
from lenslapse.onnx_f16 import save_f16
from lenslapse.sources import (
    DEFAULT_STEPS_CSV,
    CheckpointSource,
    coerce_fire_csv_arg,
    load_tokenizer,
    resolve_all_sources,
    resolve_tokenizer_ref,
    token_display_text,
)

logger = logging.getLogger(__name__)

PROBE = "The capital of Japan is the city of"


class Backbone(torch.nn.Module):
    def __init__(self, m: AutoModelForCausalLM):
        super().__init__()
        handles = resolve(m)
        self.base = handles.base
        self.blocks = handles.layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        captured: list[torch.Tensor] = []

        def cap(_m: object, _i: object, o: object) -> None:
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


def _rmsnorm_weight_has_implicit_plus_one(norm: torch.nn.Module, hidden_size: int) -> bool:
    """Most RMSNorm variants compute normalized(x) * weight, but some (Gemma 2/3) compute
    normalized(x) * (1 + weight) instead -- Gemma initializes the weight near zero rather than
    near one for training stability and adds the 1 back at call time (see Gemma3RMSNorm.forward
    and https://github.com/huggingface/transformers/pull/29402). The class name alone does not
    signal this, so probe the real module directly: run it on a fixed random vector and check
    which reconstruction (with or without +1) it actually matches, using the same eps convention
    read elsewhere in this module."""
    with torch.no_grad():
        probe = torch.randn(1, 1, hidden_size, generator=torch.Generator().manual_seed(0))
        real = norm(probe).double()
        eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon", 1e-5)
        normed = (probe.double() * torch.rsqrt(probe.double().pow(2).mean(-1, keepdim=True) + eps)).float().double()
        w = norm.weight.detach().double()
        vanilla_diff = (real - normed * w).abs().max()
        plus_one_diff = (real - normed * (1 + w)).abs().max()
        return bool(plus_one_diff < vanilla_diff)


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
    else:  # rmsnorm: x * rsqrt(mean(x^2) + eps) * w  (or * (1 + w) -- see the helper above)
        inits.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), "axes"))
        inits.append(numpy_helper.from_array(np.array(handles.eps, dtype=np.float32), "eps"))
        scale_nodes = (
            [
                helper.make_node("Constant", [], ["one"], value=numpy_helper.from_array(np.array(1.0, np.float32))),
                helper.make_node("Add", ["one", "norm_w"], ["norm_w_scaled"]),
            ]
            if _rmsnorm_weight_has_implicit_plus_one(norm, h)
            else []
        )
        nodes = [
            helper.make_node("Mul", ["hidden", "hidden"], ["sq"]),
            helper.make_node("ReduceMean", ["sq", "axes"], ["ms"], keepdims=1),
            helper.make_node("Add", ["ms", "eps"], ["ms_eps"]),
            helper.make_node("Sqrt", ["ms_eps"], ["rms"]),
            helper.make_node("Div", ["hidden", "rms"], ["xn"]),
            *scale_nodes,
            helper.make_node("Mul", ["xn", "norm_w_scaled" if scale_nodes else "norm_w"], ["normed"]),
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


def export_backbone_onnx(bb: "Backbone", input_ids: "torch.Tensor", attn: "torch.Tensor", path: Path) -> None:
    """The one torch.onnx.export call for a backbone — fidelity_eval.py and check_arch_parity.py
    must export with exactly these axes/opset or their diagnostics stop matching production."""
    torch.onnx.export(
        bb,
        (input_ids, attn),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "b", 1: "t"},
            "attention_mask": {0: "b", 1: "t"},
            "hidden_states": {1: "b", 2: "t"},
        },
        opset_version=18,
    )


def export_source(src: "CheckpointSource", out_dir: Path, tok: Any) -> dict[str, Any]:
    rev = src.name
    model = AutoModelForCausalLM.from_pretrained(
        src.load_ref, revision=src.revision, subfolder=src.subfolder or "", dtype=torch.float32
    )
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
        export_backbone_onnx(bb, input_ids, attn, bb_fp32)
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
        # write under pending names and promote only after the f16 gate below passes: writing
        # the final names first meant a failed assert left unvalidated files in the ship
        # directory under their shippable names — which a later --skip-existing run (the old
        # manifest still listing this step as passing) would then skip right over
        bb_f16_pending = step_dir / "backbone.f16.onnx.pending"
        lens_f16_pending = step_dir / "lens.f16.onnx.pending"
        save_f16(str(bb_fp32), str(bb_f16_pending))
        save_f16(str(lens_fp32), str(lens_f16_pending))

    # f16 sanity: top-1 must match torch fp32 at every position of the probe
    try:
        sbf = ort.InferenceSession(str(bb_f16_pending))
        slf = ort.InferenceSession(str(lens_f16_pending))
        hf = sbf.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attn.numpy()})[0]
        L1, _, T, H = hf.shape
        lf = slf.run(None, {"hidden": hf[-1, 0]})[0]
        f16_diff = float(np.abs(ref_logits - lf).max())
        top1_match = bool((lf.argmax(-1) == ref_logits.argmax(-1)).all())
        assert top1_match, f"{rev}: f16 top-1 disagrees with torch on the probe — do not ship this checkpoint"
    except BaseException:
        bb_f16_pending.unlink(missing_ok=True)
        lens_f16_pending.unlink(missing_ok=True)
        raise
    bb_f16_pending.replace(bb_f16)
    lens_f16_pending.replace(lens_f16)

    # this is metadata for humans reading the manifest, not something anything parses back, so a
    # lossy decode (bytes, out-of-vocab ids) is fine — see token_display_text's docstring.
    top1_token = token_display_text(tok, tok.convert_ids_to_tokens([int(ref_logits[-1].argmax())])[0])
    info: dict[str, Any] = {
        "step": src.step,
        "fp32_max_diff": fp32_diff,
        "fp32_top1_match_all_pos": fp32_top1,
        "f16_max_diff": f16_diff,
        "f16_top1_match_all_pos": top1_match,
        "top1_token": top1_token,
        "backbone_bytes": bb_f16.stat().st_size,
        "lens_bytes": lens_f16.stat().st_size,
    }
    logger.info(
        "[%s] fp32_diff=%.2e f16_diff=%.2e top1_match=%s (top1=%r) bb=%.1fMB lens=%.1fMB",
        rev,
        fp32_diff,
        f16_diff,
        top1_match,
        info["top1_token"],
        info["backbone_bytes"] / 1e6,
        info["lens_bytes"] / 1e6,
    )
    del model
    return info


class ExportConfig(BaseModel):
    """Validated arguments for `export_checkpoints`; see `main`'s docstring for what each means."""

    model: str = "EleutherAI/pythia-70m"
    steps: str = DEFAULT_STEPS_CSV
    out: Path
    skip_existing: bool = False
    final_only: bool = False
    subfolder_map: str | None = None
    revision_template: str = "step{}"
    tokenizer_ref: str | None = None

    _coerce_steps = field_validator("steps", mode="before")(coerce_fire_csv_arg)


def main(
    out: str,
    model: str = "EleutherAI/pythia-70m",
    steps: str = DEFAULT_STEPS_CSV,
    skip_existing: bool = False,
    final_only: bool = False,
    subfolder_map: str | None = None,
    revision_template: str = "step{}",
    tokenizer_ref: str | None = None,
) -> None:
    """Export a model's training checkpoints as browser-runnable ONNX pairs.

    Args:
        out: output directory for the ONNX pairs and `manifest.json`.
        model: HF id or local directory.
        steps: comma-separated training steps for a hub suite (ignored if `subfolder_map` is set).
        skip_existing: skip steps whose ONNX files already exist in `out`.
        final_only: single checkpoint (revision "main") instead of a step suite.
        subfolder_map: "step:path,step:path,..." for repos that nest checkpoints as subfolders of
            one revision instead of using git revisions per checkpoint; overrides `steps`.
        revision_template: revision naming for hub suites, e.g. "global_step{}" for
            bigscience/bloom-*-intermediate.
        tokenizer_ref: load the tokenizer from a different ref than the checkpoint weights, as
            "repo_id" or "repo_id@revision" — for repos where the per-checkpoint tokenizer files
            do not load cleanly (bigscience/bloom-*-intermediate) or live in a separate repo
            entirely (m-a-p/neo_scalinglaw_*, whose tokenizer is only published under
            m-a-p/neo_7b). The tokenizer is identical across checkpoints of the same pretraining
            run, so this is always safe when it applies.
    """
    cfg = ExportConfig(
        model=model,
        steps=steps,
        out=out,  # type: ignore[arg-type]  # pydantic coerces str -> Path
        skip_existing=skip_existing,
        final_only=final_only,
        subfolder_map=subfolder_map,
        revision_template=revision_template,
        tokenizer_ref=tokenizer_ref,
    )

    cfg.out.mkdir(parents=True, exist_ok=True)
    sources = resolve_all_sources(cfg.model, cfg.steps, cfg.final_only, cfg.subfolder_map, cfg.revision_template)
    tok_load_ref, tok_rev, tok_subfolder = resolve_tokenizer_ref(cfg.tokenizer_ref, sources[0])
    tok = load_tokenizer(tok_load_ref, tok_rev, tok_subfolder)

    cfg_model = AutoModelForCausalLM.from_pretrained(
        sources[0].load_ref, revision=sources[0].revision, subfolder=sources[0].subfolder or "", dtype=torch.float32
    )
    meta = {
        "model": cfg.model,
        "format": "f16",
        "files": ["backbone.f16.onnx", "lens.f16.onnx"],
        "hidden_size": cfg_model.config.hidden_size,
        "num_layers": cfg_model.config.num_hidden_layers,
        "vocab_size": cfg_model.config.vocab_size,
        "tokenizer": cfg.model,
        "probe": PROBE,
        "steps": [],
    }
    del cfg_model

    # merge with a previous run's manifest so incremental exports never drop existing checkpoints
    manifest_path = cfg.out / "manifest.json"
    results: dict[int, dict] = {}
    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text())
            if old.get("model") == cfg.model:
                results = {s["step"]: s for s in old.get("steps", [])}
            else:
                raise SystemExit(
                    f"{cfg.out} already holds a manifest for {old.get('model')!r}; refusing to mix models"
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise SystemExit(f"{manifest_path} is unreadable ({e}); fix or delete it before re-exporting") from e

    for src in sources:
        step = src.step
        step_dir = cfg.out / src.name
        complete = all((step_dir / f).exists() for f in meta["files"])
        if cfg.skip_existing and complete and step in results:
            logger.info("[%s] exists, skipping", src.name)
            continue
        results[step] = export_source(src, cfg.out, tok)
        meta["steps"] = [results[k] for k in sorted(results)]
        manifest_path.write_text(json.dumps(meta, indent=1))

    meta["steps"] = [results[k] for k in sorted(results)]
    manifest_path.write_text(json.dumps(meta, indent=1))
    # the completion signal, not a diagnostic message: scripts pipe stdout and grep for this,
    # so it must stay on print() rather than move to the logger (which defaults to stderr).
    print(f"ALL_DONE steps: {len(meta['steps'])}")


if __name__ == "__main__":
    configure_cli_logging()
    fire.Fire(main)
