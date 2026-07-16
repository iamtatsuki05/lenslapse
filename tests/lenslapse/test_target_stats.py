"""target_stats is the single serialization of target probability/rank shared by the precomputed
shards and the probe server's /probe — the app overlays live and precomputed trajectories, so the
numeric recipe (exp of log-probs, round to 6, strictly-greater rank) is pinned here."""

import math

import torch

from lenslapse.precompute_lens import target_stats


def _lp(logits: list[list[list[float]]]) -> torch.Tensor:
    return torch.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)


def test_probability_is_softmax_rounded_to_6_decimals() -> None:
    logits = [[[1.0, 3.0, 2.0, -1.0]]]  # [L+1=1, T=1, V=4]
    stats = target_stats(_lp(logits), tid=1)
    z = sum(math.exp(v - 3.0) for v in logits[0][0])
    assert stats["p"] == [[round(math.exp(0.0) / z, 6)]]


def test_rank_is_strictly_greater_plus_one() -> None:
    # ties do not inflate the rank: both 3.0 logits rank 1, the 2.0 ranks 3, the 1.0 ranks 4
    lp = _lp([[[1.0, 3.0, 2.0, 3.0]]])
    assert target_stats(lp, tid=1)["r"] == [[1]]
    assert target_stats(lp, tid=3)["r"] == [[1]]
    assert target_stats(lp, tid=2)["r"] == [[3]]
    assert target_stats(lp, tid=0)["r"] == [[4]]


def test_shapes_follow_layers_and_positions() -> None:
    lp = _lp([[[0.0, 1.0], [2.0, 0.0]], [[1.0, 1.0], [0.0, 3.0]]])  # [2, 2, 2]
    stats = target_stats(lp, tid=0)
    assert len(stats["p"]) == 2 and all(len(row) == 2 for row in stats["p"])
    assert stats["r"][0][1] == 1  # position where tid=0 has the larger logit
    assert stats["r"][1][1] == 2
