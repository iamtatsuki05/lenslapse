"""Architecture resolution: GPT-NeoX / Llama / GPT-2 / OPT decoder layouts, via synthetic modules
(no real model download — the shapes are what matters, not real weights)."""

import pytest
import torch
from torch import nn

from lenslapse.arch import resolve


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
    other supported layout where the decoder stack sits directly on the first hop."""

    def __init__(self) -> None:
        super().__init__()
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


def test_plain_model_attr_still_wins_when_no_decoder_present() -> None:
    """model.decoder is tried before model, but a Llama-style model has no `.decoder` attribute at
    all, so the search must fall through to `model` rather than raising early."""
    handles = resolve(_LlamaStyle())
    assert isinstance(handles.base, nn.Module)


def test_unsupported_architecture_raises_with_class_name() -> None:
    class _Unsupported(_WithLMHead):
        pass

    with pytest.raises(ValueError, match="_Unsupported"):
        resolve(_Unsupported())
