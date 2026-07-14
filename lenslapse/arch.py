"""Architecture introspection: locate the pieces the lens recipe needs on any HF decoder LM.

Supports the common decoder layouts:
  GPT-NeoX / Pythia : model.gpt_neox.layers,  final_layer_norm (LayerNorm),  embed_out
  Llama / OLMo-2 /
  SmolLM / Qwen     : model.model.layers,     model.model.norm (RMSNorm),    lm_head
  GPT-2             : model.transformer.h,    transformer.ln_f (LayerNorm),  lm_head (tied)

The lens identity — lens(last block output) == model logits — holds for pre-LN decoders where the
LM applies `final_norm` then `lm_head` to the last block's output. export_checkpoints.py asserts it
per checkpoint, so an unsupported layout fails loudly instead of shipping a wrong lens.
"""

from dataclasses import dataclass

import torch


@dataclass
class ArchHandles:
    base: torch.nn.Module  # decoder stack: (input_ids, attention_mask, output_hidden_states=True)
    layers: torch.nn.ModuleList
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    norm_type: str  # 'layernorm' | 'rmsnorm'
    eps: float


_BASE_PATHS = ("gpt_neox", "model", "transformer")
_LAYER_ATTRS = ("layers", "h")
_NORM_ATTRS = ("final_layer_norm", "norm", "ln_f")


def resolve(model: torch.nn.Module) -> ArchHandles:
    base = None
    for name in _BASE_PATHS:
        cand = getattr(model, name, None)
        if isinstance(cand, torch.nn.Module):
            base = cand
            break
    if base is None:
        raise ValueError(f"unsupported architecture: no decoder stack found on {type(model).__name__}")

    layers = None
    for name in _LAYER_ATTRS:
        cand = getattr(base, name, None)
        if isinstance(cand, torch.nn.ModuleList):
            layers = cand
            break
    if layers is None:
        raise ValueError(f"unsupported architecture: no block list found on {type(base).__name__}")

    final_norm = None
    for name in _NORM_ATTRS:
        cand = getattr(base, name, None)
        if isinstance(cand, torch.nn.Module):
            final_norm = cand
            break
    if final_norm is None:
        raise ValueError(f"unsupported architecture: no final norm found on {type(base).__name__}")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise ValueError("unsupported architecture: no output embeddings / lm_head")

    # Classify by class name: RMSNorm skips mean subtraction, so bias absence alone is not a safe
    # signal (LayerNorm(bias=False) still centers). A misclassification is caught by the fp32
    # parity assert at export time.
    is_rms = "rms" in type(final_norm).__name__.lower()
    eps = getattr(final_norm, "eps", None)
    if eps is None:
        eps = getattr(final_norm, "variance_epsilon", 1e-5)

    return ArchHandles(
        base=base,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        norm_type="rmsnorm" if is_rms else "layernorm",
        eps=float(eps),
    )
