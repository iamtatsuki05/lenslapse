"""RMSNorm variant detection: most models compute normalized(x) * weight, but Gemma 2/3 compute
normalized(x) * (1 + weight) instead (weight is initialized near zero, not near one). Synthetic
modules only — no real model download, matching test_arch.py's approach."""

import torch
from torch import nn

from lenslapse.export_checkpoints import _rmsnorm_weight_has_implicit_plus_one

_HIDDEN = 8
_EPS = 1e-6


class _VanillaRMSNorm(nn.Module):
    """normalized(x) * weight -- e.g. Llama, Qwen3, SmolLM2."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(_HIDDEN))  # trained weights center near 1, not 0
        self.eps = _EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


class _GemmaStyleRMSNorm(nn.Module):
    """normalized(x) * (1 + weight) -- matches transformers' Gemma3RMSNorm.forward exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(_HIDDEN) * 0.1)  # trained weights center near 0
        self.eps = _EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * (1.0 + self.weight)


def test_vanilla_rmsnorm_is_not_flagged_as_plus_one() -> None:
    assert _rmsnorm_weight_has_implicit_plus_one(_VanillaRMSNorm(), _HIDDEN) is False


def test_gemma_style_rmsnorm_is_flagged_as_plus_one() -> None:
    assert _rmsnorm_weight_has_implicit_plus_one(_GemmaStyleRMSNorm(), _HIDDEN) is True


def test_vanilla_with_near_zero_weight_is_still_correctly_classified() -> None:
    """A vanilla RMSNorm whose trained weight happens to be small should not be misclassified just
    because 1+w and w are numerically closer when w is near zero -- the probe compares against the
    real module's actual output, not the weight's magnitude."""
    norm = _VanillaRMSNorm()
    with torch.no_grad():
        norm.weight.mul_(0.01)  # small, but the module still computes normed * weight, not (1+weight)
    assert _rmsnorm_weight_has_implicit_plus_one(norm, _HIDDEN) is False
