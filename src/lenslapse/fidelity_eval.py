"""Fidelity evaluation: which ONNX weight format keeps the browser lens faithful to fp32 torch?

Compares weight formats against fp32 torch ground truth on the curated prompt set:
  q8pt         : dynamic int8, per-tensor
  q8pc         : dynamic int8, per-channel
  q8pt+f16lens : per-tensor int8 backbone + fp16-stored lens head
  q8pc+f16lens : per-channel int8 backbone + fp16-stored lens head
  f16w         : fp16-stored weights for both graphs (the format the demo ships)

Metrics per (config, step): top-1 agreement with fp32 torch over all (layer, position) cells,
final-layer-only top-1 agreement, and mean KL(fp32 || onnx) at the final layer.
Writes a JSON report for the paper's fidelity table.
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
from onnxruntime.quantization import QuantType, quantize_dynamic
from pydantic import BaseModel, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from lenslapse.export_checkpoints import Backbone, build_lens_onnx, export_backbone_onnx
from lenslapse.logging_utils import configure_cli_logging
from lenslapse.precompute_lens import PROMPTS, lens_all
from lenslapse.sources import coerce_fire_csv_arg

logger = logging.getLogger(__name__)


def build_lens_onnx_f16(model: Any, path: Path) -> None:
    """Lens head with fp16-stored weights cast to fp32 at run time (half the size, fp32 math).

    LayerNorm-only variant used for the quantization comparison (Pythia); the shipped f16w format
    (onnx_f16.save_f16 over build_lens_onnx) covers RMSNorm architectures as well.
    """
    from lenslapse.arch import resolve

    handles = resolve(model)
    h = model.config.hidden_size
    ln = handles.final_norm
    w_unembed = handles.lm_head.weight.detach().numpy().T.astype(np.float16)
    nodes = [
        helper.make_node("Cast", ["w_unembed_f16"], ["w_unembed"], to=TensorProto.FLOAT),
        helper.make_node("LayerNormalization", ["hidden", "ln_w", "ln_b"], ["normed"], axis=-1, epsilon=float(ln.eps)),
        helper.make_node("MatMul", ["normed", "w_unembed"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "lens_head_f16",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, ["n", h])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", w_unembed.shape[1]])],
        initializer=[
            numpy_helper.from_array(ln.weight.detach().numpy(), "ln_w"),
            numpy_helper.from_array(ln.bias.detach().numpy(), "ln_b"),
            numpy_helper.from_array(w_unembed, "w_unembed_f16"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), str(path))


def export_fp32(model: Any, td: Path) -> tuple[Path, Path]:
    tok_ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(tok_ids)
    bb_path = Path(td) / "bb.onnx"
    lens_path = Path(td) / "lens.onnx"
    export_backbone_onnx(Backbone(model), tok_ids, mask, bb_path)
    build_lens_onnx(model, lens_path)
    return bb_path, lens_path


def make_variants(model: Any, bb_path: Path, lens_path: Path, td: Path) -> dict[str, tuple[Path, Path]]:
    variants = {}
    q8pt_bb, q8pt_lens = Path(td) / "bb.q8pt.onnx", Path(td) / "lens.q8pt.onnx"
    quantize_dynamic(str(bb_path), str(q8pt_bb), weight_type=QuantType.QInt8)
    quantize_dynamic(str(lens_path), str(q8pt_lens), weight_type=QuantType.QInt8)
    variants["q8pt"] = (q8pt_bb, q8pt_lens)

    q8pc_bb, q8pc_lens = Path(td) / "bb.q8pc.onnx", Path(td) / "lens.q8pc.onnx"
    quantize_dynamic(str(bb_path), str(q8pc_bb), weight_type=QuantType.QInt8, per_channel=True)
    quantize_dynamic(str(lens_path), str(q8pc_lens), weight_type=QuantType.QInt8, per_channel=True)
    variants["q8pc"] = (q8pc_bb, q8pc_lens)

    f16_lens = Path(td) / "lens.f16.onnx"
    build_lens_onnx_f16(model, f16_lens)
    variants["q8pc+f16lens"] = (q8pc_bb, f16_lens)
    variants["q8pt+f16lens"] = (q8pt_bb, f16_lens)

    from lenslapse.onnx_f16 import save_f16

    f16w_bb, f16w_lens = Path(td) / "bb.f16w.onnx", Path(td) / "lens.f16w.onnx"
    save_f16(str(bb_path), str(f16w_bb))
    save_f16(str(lens_path), str(f16w_lens))
    variants["f16w"] = (f16w_bb, f16w_lens)
    return variants


def eval_variant(bb_path: Path, lens_path: Path, prompts_enc: Any, ref: Any) -> dict[str, Any]:
    sb = ort.InferenceSession(str(bb_path))
    sl = ort.InferenceSession(str(lens_path))
    agree_all, agree_final, kl_final, n_all, n_final = 0, 0, 0.0, 0, 0
    for pid, ids in prompts_enc.items():
        hs = sb.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]  # [L+1,1,T,H]
        L1, _, T, H = hs.shape
        lo = sl.run(None, {"hidden": hs[:, 0].reshape(L1 * T, H).astype(np.float32)})[0].reshape(L1, T, -1)
        ref_lp = ref[pid]  # [L+1, T, V] log-probs (torch fp32)
        onnx_top1 = lo.argmax(-1)
        ref_top1 = ref_lp.argmax(-1)
        agree_all += (onnx_top1 == ref_top1).sum()
        n_all += onnx_top1.size
        agree_final += (onnx_top1[-1] == ref_top1[-1]).sum()
        n_final += T
        x = lo[-1] - lo[-1].max(-1, keepdims=True)
        onnx_lp = x - np.log(np.exp(x).sum(-1, keepdims=True))
        kl_final += float((np.exp(ref_lp[-1]) * (ref_lp[-1] - onnx_lp)).sum(-1).mean())
    return {
        "top1_agree_all_layers": round(float(agree_all / n_all), 4),
        "top1_agree_final_layer": round(float(agree_final / n_final), 4),
        "mean_kl_final_layer": round(kl_final / len(prompts_enc), 5),
        "backbone_mb": round(Path(bb_path).stat().st_size / 1e6, 1),
        "lens_mb": round(Path(lens_path).stat().st_size / 1e6, 1),
    }


class FidelityEvalConfig(BaseModel):
    """Validated arguments for `fidelity_eval`; see `main`'s docstring for what each means."""

    model: str = "EleutherAI/pythia-70m"
    steps: str = "8000,64000,143000"
    out: Path

    _coerce_steps = field_validator("steps", mode="before")(coerce_fire_csv_arg)


def main(out: str, model: str = "EleutherAI/pythia-70m", steps: str = "8000,64000,143000") -> None:
    """Evaluate ONNX weight-format fidelity against fp32 torch ground truth on the curated prompts.

    Args:
        out: output path for the JSON fidelity report.
        model: HF id or local directory.
        steps: comma-separated training steps to evaluate.
    """
    cfg = FidelityEvalConfig(
        out=out,  # type: ignore[arg-type]  # pydantic coerces str -> Path
        model=model,
        steps=steps,
    )

    tok = AutoTokenizer.from_pretrained(cfg.model)
    prompts_enc = {i: np.array([tok(p["text"])["input_ids"]], dtype=np.int64) for i, p in enumerate(PROMPTS)}

    report: dict[str, Any] = {"model": cfg.model, "prompts": len(PROMPTS), "steps": {}}
    for step in [int(s) for s in cfg.steps.split(",")]:
        step_model = AutoModelForCausalLM.from_pretrained(cfg.model, revision=f"step{step}", dtype=torch.float32)
        step_model.eval()
        ref = {}
        for i, p in enumerate(PROMPTS):
            ref[i] = lens_all(step_model, torch.tensor([tok(p["text"])["input_ids"]])).numpy()

        with tempfile.TemporaryDirectory() as td:
            bb_path, lens_path = export_fp32(step_model, Path(td))
            variants = {"fp32": (bb_path, lens_path), **make_variants(step_model, bb_path, lens_path, Path(td))}
            report["steps"][step] = {name: eval_variant(b, le, prompts_enc, ref) for name, (b, le) in variants.items()}
        del step_model
        # per-step result data, not a diagnostic message: keep on print() like ALL_DONE below,
        # so piping stdout to a file captures the actual report instead of just progress noise.
        print(f"[step{step}] {json.dumps(report['steps'][step], indent=1)}")

    cfg.out.write_text(json.dumps(report, indent=1))
    print("ALL_DONE")


if __name__ == "__main__":
    configure_cli_logging()
    fire.Fire(main)
