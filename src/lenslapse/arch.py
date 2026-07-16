"""Architecture introspection: locate the pieces the lens recipe needs on any HF decoder LM.

A generic attribute-name heuristic resolves most decoder layouts with no configuration at all:

  GPT-NeoX / Pythia : model.gpt_neox.layers,         final_layer_norm (LayerNorm),  embed_out
  Llama / OLMo-2 /
  SmolLM / Qwen     : model.model.layers,            model.model.norm (RMSNorm),    lm_head
  GPT-2             : model.transformer.h,           transformer.ln_f (LayerNorm),  lm_head (tied)

For a layout the heuristic can't (or shouldn't have to) guess — e.g. a decoder stack nested more
than one hop deep, or an attribute name outside the common set below — register an explicit
`ArchSpec` keyed by the model's own `config.model_type` (a stable identifier every Transformers
architecture defines) via `register_architecture()`. `resolve()` checks the registry first, so
adding a new architecture never requires touching the generic heuristic or its ordering:

    from lenslapse.arch import register_architecture
    register_architecture("my_new_model", base_path="model.decoder")

The lens identity — lens(last block output) == model logits — holds for pre-LN decoders where the
LM applies `final_norm` then `lm_head` to the last block's output. export_checkpoints.py asserts it
per checkpoint, so an unsupported layout fails loudly instead of shipping a wrong lens.
"""

import torch
from pydantic import BaseModel, ConfigDict


class ArchHandles(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: torch.nn.Module  # decoder stack: (input_ids, attention_mask, output_hidden_states=True)
    layers: torch.nn.ModuleList
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module
    norm_type: str  # 'layernorm' | 'rmsnorm'
    eps: float


_LAYER_ATTRS = ("layers", "h")
_NORM_ATTRS = ("final_layer_norm", "norm", "ln_f")
_BASE_PATHS = ("gpt_neox", "model", "transformer")


class ArchSpec(BaseModel):
    """Explicit attribute paths for one `config.model_type`, used instead of (not blended with)
    the generic heuristic below. `base_path` is dotted for a stack nested more than one hop deep
    (e.g. "model.decoder"); `layer_attrs`/`norm_attrs` default to the same names the generic
    heuristic already tries, since only the base path usually needs to be architecture-specific."""

    base_path: str
    layer_attrs: tuple[str, ...] = _LAYER_ATTRS
    norm_attrs: tuple[str, ...] = _NORM_ATTRS


_ARCH_OVERRIDES: dict[str, ArchSpec] = {}


def register_architecture(
    model_type: str,
    base_path: str,
    layer_attrs: tuple[str, ...] | None = None,
    norm_attrs: tuple[str, ...] | None = None,
) -> None:
    """Register explicit attribute paths for `model_type` (from the model's own `config.model_type`
    — e.g. "opt", "gemma3_text"; check `AutoConfig.from_pretrained(ref).model_type` if unsure).

    Call this — from anywhere, before `resolve()` runs — to add support for a new Transformers
    decoder architecture without editing this module. Once registered, `resolve()` uses only
    these paths for that model_type; it does not also try the generic heuristic, so a wrong
    override fails loudly (an unresolved path is a bug in the registration, not something to
    silently paper over by guessing).

    Raises ValueError if `model_type` is already registered — silently overwriting a previous
    registration would let a copy-paste mistake elsewhere in the process quietly change which
    architecture an already-working model_type resolves as."""
    if model_type in _ARCH_OVERRIDES:
        raise ValueError(f"{model_type!r} is already registered (base_path={_ARCH_OVERRIDES[model_type].base_path!r})")
    _ARCH_OVERRIDES[model_type] = ArchSpec(
        base_path=base_path,
        layer_attrs=_LAYER_ATTRS if layer_attrs is None else layer_attrs,
        norm_attrs=_NORM_ATTRS if norm_attrs is None else norm_attrs,
    )


# OPT nests its decoder stack two hops down (model.model.decoder), so the generic single-hop
# "model" base path (-> OPTModel, which has neither `layers` nor a final norm of its own) would
# resolve to the wrong object before the correct one was ever tried.
register_architecture("opt", base_path="model.decoder")


def _resolve_path(obj: torch.nn.Module, dotted_path: str) -> torch.nn.Module | None:
    for name in dotted_path.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj if isinstance(obj, torch.nn.Module) else None


def resolve(model: torch.nn.Module) -> ArchHandles:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    spec = _ARCH_OVERRIDES.get(model_type) if model_type else None
    base_paths = (spec.base_path,) if spec else _BASE_PATHS
    layer_attrs = spec.layer_attrs if spec else _LAYER_ATTRS
    norm_attrs = spec.norm_attrs if spec else _NORM_ATTRS

    # named once so every failure below can say whether a registered override was in play, not
    # just the base_path lookup — a bad override should always be identifiable as the cause.
    registered = f" (model_type {model_type!r} is registered with {spec!r})" if spec else ""

    base = None
    for path in base_paths:
        cand = _resolve_path(model, path)
        if cand is not None:
            base = cand
            break
    if base is None:
        raise ValueError(f"unsupported architecture: no decoder stack found on {type(model).__name__}{registered}")

    layers = None
    for name in layer_attrs:
        cand = getattr(base, name, None)
        if isinstance(cand, torch.nn.ModuleList):
            layers = cand
            break
    if layers is None:
        raise ValueError(f"unsupported architecture: no block list found on {type(base).__name__}{registered}")

    final_norm = None
    for name in norm_attrs:
        cand = getattr(base, name, None)
        if isinstance(cand, torch.nn.Module):
            final_norm = cand
            break
    if final_norm is None:
        raise ValueError(f"unsupported architecture: no final norm found on {type(base).__name__}{registered}")

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
        eps=float(eps if eps is not None else 1e-5),
    )
