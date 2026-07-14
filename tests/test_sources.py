"""Checkpoint-source resolution: hub suites, single hub models, and local Trainer directories."""

from pathlib import Path

import pytest

from lenslapse.sources import resolve_sources


def test_hub_suite_maps_steps_to_revisions() -> None:
    sources = resolve_sources("EleutherAI/pythia-70m", "8000,0", final_only=False)
    assert [(s.load_ref, s.revision, s.step) for s in sources] == [
        ("EleutherAI/pythia-70m", "step0", 0),
        ("EleutherAI/pythia-70m", "step8000", 8000),
    ]


def test_final_only_uses_main_revision_at_step_zero() -> None:
    (src,) = resolve_sources("gpt2", "0,8000", final_only=True)
    assert (src.load_ref, src.revision, src.step) == ("gpt2", "main", 0)


def test_trainer_directory_scans_checkpoints_in_step_order(tmp_path: Path) -> None:
    for name in ("checkpoint-800", "checkpoint-0", "checkpoint-12", "not-a-checkpoint"):
        (tmp_path / name).mkdir()
    sources = resolve_sources(str(tmp_path), "0", final_only=False)
    assert [s.step for s in sources] == [0, 12, 800]
    assert all(s.revision is None for s in sources)
    assert sources[-1].load_ref.endswith("checkpoint-800")


def test_trainer_directory_rejects_final_only(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-0").mkdir()
    with pytest.raises(SystemExit, match="final-only is ambiguous"):
        resolve_sources(str(tmp_path), "0", final_only=True)


def test_plain_local_directory_is_a_single_step_zero_checkpoint(tmp_path: Path) -> None:
    (src,) = resolve_sources(str(tmp_path), "0", final_only=False)
    assert (src.load_ref, src.revision, src.step) == (str(tmp_path), None, 0)
