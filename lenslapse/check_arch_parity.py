"""Cross-architecture parity check: does the lens recipe hold for a given model?

Exports the (backbone, lens) pair to a temp dir and asserts:
  1. torch: lens(last block output) == model logits (the identity the whole demo rests on)
  2. ONNX fp32 matches torch
  3. f16-stored weights keep top-1 at every position of the probe

Usage:
  python check_arch_parity.py --model gpt2
  python check_arch_parity.py --model HuggingFaceTB/SmolLM2-135M
  python check_arch_parity.py --model EleutherAI/pythia-70m --revision step1000
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .arch import resolve
from .export_checkpoints import PROBE, Backbone, build_lens_onnx
from .onnx_f16 import save_f16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float32)
    model.eval()
    handles = resolve(model)
    print(f"{args.model}@{args.revision}: norm={handles.norm_type} layers={len(handles.layers)}")

    enc = tok(PROBE, return_tensors="pt")
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attn).logits[0]
        hs = Backbone(model)(input_ids, attn)
        lens_final = handles.lm_head(handles.final_norm(hs[-1, 0]))
    torch_diff = float((ref - lens_final).abs().max())
    print(f"torch lens(last)==logits max diff: {torch_diff:.2e}")
    assert torch_diff < 1e-3, "lens identity violated — unsupported (e.g. post-norm) architecture"

    with tempfile.TemporaryDirectory() as td:
        bb, lens = Path(td) / "bb.onnx", Path(td) / "lens.onnx"
        torch.onnx.export(
            Backbone(model),
            (input_ids, attn),
            str(bb),
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "b", 1: "t"},
                "attention_mask": {0: "b", 1: "t"},
                "hidden_states": {1: "b", 2: "t"},
            },
            opset_version=18,
        )
        build_lens_onnx(model, lens)
        sb, sl = ort.InferenceSession(str(bb)), ort.InferenceSession(str(lens))
        h = sb.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attn.numpy()})[0]
        lo = sl.run(None, {"hidden": h[-1, 0]})[0]
        onnx_diff = float(np.abs(ref.numpy() - lo).max())
        onnx_top1 = bool((lo.argmax(-1) == ref.numpy().argmax(-1)).all())
        print(f"onnx fp32 max diff: {onnx_diff:.2e} top-1 match: {onnx_top1}")
        # late checkpoints have large logit magnitudes; match export_checkpoints.py's loose bound
        assert onnx_diff < 0.05 and onnx_top1

        bb16, lens16 = Path(td) / "bb16.onnx", Path(td) / "lens16.onnx"
        save_f16(str(bb), str(bb16))
        save_f16(str(lens), str(lens16))
        sb16, sl16 = ort.InferenceSession(str(bb16)), ort.InferenceSession(str(lens16))
        h16 = sb16.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attn.numpy()})[0]
        lo16 = sl16.run(None, {"hidden": h16[-1, 0]})[0]
        top1 = bool((lo16.argmax(-1) == ref.numpy().argmax(-1)).all())
        print(f"f16 top-1 match at all positions: {top1}")
        assert top1

    print("PARITY_OK")


if __name__ == "__main__":
    main()
