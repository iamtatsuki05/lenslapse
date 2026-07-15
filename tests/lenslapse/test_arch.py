"""Architecture resolution: GPT-NeoX / Llama / GPT-2 / OPT decoder layouts, via synthetic modules
(no real model download — the shapes are what matters, not real weights)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lenslapse.arch import _ARCH_OVERRIDES, register_architecture, resolve


class _Block(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make(norm: nn.Module, n_layers: int = 2) -> nn.ModuleList:
    return nn.ModuleList([_Block() for _ in range(n_layers)])


class _WithLMHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, 4)

    def get_output_embeddings(self) -> nn.Module:
        return self.embed


class _GPTNeoXStyle(_WithLMHead):
    def __init__(self) -> None:
        super().__init__()
        self.gpt_neox = nn.Module()
        self.gpt_neox.layers = _make(nn.LayerNorm(4))
        self.gpt_neox.final_layer_norm = nn.LayerNorm(4)


class _LlamaStyle(_WithLMHead):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = _make(nn.LayerNorm(4))
        self.model.norm = nn.RMSNorm(4)


class _GPT2Style(_WithLMHead):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = _make(nn.LayerNorm(4))
        self.transformer.ln_f = nn.LayerNorm(4)


class _OPTStyle(_WithLMHead):
    """model.model.decoder.{layers,final_layer_norm} — two hops below `model`, unlike every
    other supported layout where the decoder stack sits directly on the first hop. Real OPT
    checkpoints report config.model_type == "opt" (verified against facebook/opt-125m), which is
    what `resolve()` actually keys its registered override on — not the attribute shape alone."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="opt")
        self.model = nn.Module()
        self.model.decoder = nn.Module()
        self.model.decoder.layers = _make(nn.LayerNorm(4))
        self.model.decoder.final_layer_norm = nn.LayerNorm(4)


def test_gpt_neox_style_resolves() -> None:
    handles = resolve(_GPTNeoXStyle())
    assert len(handles.layers) == 2
    assert handles.norm_type == "layernorm"


def test_llama_style_resolves_with_rmsnorm() -> None:
    handles = resolve(_LlamaStyle())
    assert len(handles.layers) == 2
    assert handles.norm_type == "rmsnorm"


def test_gpt2_style_resolves() -> None:
    handles = resolve(_GPT2Style())
    assert len(handles.layers) == 2
    assert handles.norm_type == "layernorm"


def test_opt_style_nested_decoder_resolves() -> None:
    """Regression test: model.model has no layers/norm of its own for OPT, only model.model.decoder
    does. A flat one-hop base search finds `model` first and stops there, raising "no block list
    found" even though the real stack is one hop further down."""
    model = _OPTStyle()
    handles = resolve(model)
    assert handles.base is model.model.decoder
    assert len(handles.layers) == 2
    assert handles.norm_type == "layernorm"


def test_plain_model_attr_resolves_without_a_registered_override() -> None:
    """A Llama-style model has no config.model_type override registered (only "opt" does), so the
    generic heuristic's own base-path search must still find `model` directly."""
    handles = resolve(_LlamaStyle())
    assert isinstance(handles.base, nn.Module)


def test_unsupported_architecture_raises_with_class_name() -> None:
    class _Unsupported(_WithLMHead):
        pass

    with pytest.raises(ValueError, match="_Unsupported"):
        resolve(_Unsupported())


class _CustomNestedStyle(_WithLMHead):
    """A hypothetical future architecture nested two hops down, like OPT, but under a different
    model_type and attribute names — support added purely via register_architecture(), with no
    change to arch.py's own generic heuristic or its ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="my_new_arch")
        self.backbone = nn.Module()
        self.backbone.stack = nn.Module()
        self.backbone.stack.blocks = _make(nn.LayerNorm(4))
        self.backbone.stack.final_norm = nn.RMSNorm(4)


def test_register_architecture_extends_support_without_editing_arch_py() -> None:
    """The whole point of the registry: a new architecture's attribute paths are registered from
    outside the module (simulating a downstream caller), not added to arch.py's own tuples."""
    register_architecture(
        "my_new_arch", base_path="backbone.stack", layer_attrs=("blocks",), norm_attrs=("final_norm",)
    )
    try:
        handles = resolve(_CustomNestedStyle())
        assert len(handles.layers) == 2
        assert handles.norm_type == "rmsnorm"
    finally:
        del _ARCH_OVERRIDES["my_new_arch"]  # tests must not leak global registry state


def test_register_architecture_defaults_layer_and_norm_attrs_to_the_generic_ones() -> None:
    """Only base_path is usually architecture-specific (see ArchSpec's docstring); layer_attrs and
    norm_attrs should default to the same names the generic heuristic already tries."""
    register_architecture("__test_only_base_path__", base_path="model.decoder")
    try:
        spec = _ARCH_OVERRIDES["__test_only_base_path__"]
        assert spec.layer_attrs == ("layers", "h")
        assert spec.norm_attrs == ("final_layer_norm", "norm", "ln_f")
    finally:
        del _ARCH_OVERRIDES["__test_only_base_path__"]


def test_registered_override_that_fails_to_resolve_names_the_bad_path() -> None:
    """An override is trusted, not blended with the generic heuristic — if its base_path doesn't
    resolve, that's a bug in the registration to fix, not something to silently paper over by
    falling back to guessing. The error should name the path that failed, for a quick diagnosis."""
    register_architecture("__test_bad_path__", base_path="nonexistent.path")
    try:

        class _Model(_WithLMHead):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(model_type="__test_bad_path__")

        with pytest.raises(ValueError, match="nonexistent.path"):
            resolve(_Model())
    finally:
        del _ARCH_OVERRIDES["__test_bad_path__"]


def test_registered_override_that_fails_on_layers_also_names_the_override() -> None:
    """The base_path itself may resolve while layer_attrs doesn't (a typo'd block-list name) —
    that failure must identify the override too, not just the base_path failure case above."""
    register_architecture("__test_bad_layers__", base_path="model", layer_attrs=("nonexistent_blocks",))
    try:

        class _Model(_WithLMHead):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(model_type="__test_bad_layers__")
                self.model = nn.Module()

        with pytest.raises(ValueError, match="__test_bad_layers__"):
            resolve(_Model())
    finally:
        del _ARCH_OVERRIDES["__test_bad_layers__"]


def test_register_architecture_rejects_duplicate_model_type() -> None:
    """Silently overwriting a previous registration would let a copy-paste mistake elsewhere in
    the process quietly change which architecture an already-working model_type resolves as —
    so re-registering the same model_type must fail loudly instead."""
    register_architecture("__test_dup__", base_path="a")
    try:
        with pytest.raises(ValueError, match="__test_dup__"):
            register_architecture("__test_dup__", base_path="b")
        assert _ARCH_OVERRIDES["__test_dup__"].base_path == "a"  # first registration must survive
    finally:
        del _ARCH_OVERRIDES["__test_dup__"]


def test_register_architecture_preserves_explicit_empty_attrs() -> None:
    """layer_attrs=() / norm_attrs=() is a deliberate choice (e.g. to avoid ever matching a
    generic name that collides with something else on that model), not the same as omitting the
    argument — `x or default` would conflate the two, since an empty tuple is falsy."""
    register_architecture("__test_empty_attrs__", base_path="model", layer_attrs=(), norm_attrs=())
    try:
        spec = _ARCH_OVERRIDES["__test_empty_attrs__"]
        assert spec.layer_attrs == ()
        assert spec.norm_attrs == ()
    finally:
        del _ARCH_OVERRIDES["__test_empty_attrs__"]
