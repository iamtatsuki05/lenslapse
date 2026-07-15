"""Checkpoint-source resolution: hub suites, single hub models, and local Trainer directories."""

from pathlib import Path

import huggingface_hub
import pytest
import transformers

from lenslapse.sources import (
    CheckpointSource,
    coerce_fire_csv_arg,
    load_tokenizer,
    resolve_sources,
    resolve_subfolder_sources,
    resolve_tokenizer_ref,
)


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


def test_subfolder_suite_maps_steps_to_subfolders_sorted_by_step() -> None:
    sources = resolve_subfolder_sources("m-a-p/neo_scalinglaw_250M", "33550:hf_ckpt/33.55B,16780:hf_ckpt/16.78B")
    assert [(s.load_ref, s.revision, s.step, s.subfolder) for s in sources] == [
        ("m-a-p/neo_scalinglaw_250M", None, 16780, "hf_ckpt/16.78B"),
        ("m-a-p/neo_scalinglaw_250M", None, 33550, "hf_ckpt/33.55B"),
    ]


def test_hub_suite_sources_have_no_subfolder() -> None:
    (src,) = resolve_sources("EleutherAI/pythia-70m", "0", final_only=False)
    assert src.subfolder is None


def test_tokenizer_ref_falls_back_to_the_checkpoint_source_when_unset() -> None:
    fallback = CheckpointSource("m-a-p/neo_scalinglaw_250M", None, 16780, subfolder="hf_ckpt/16.78B")
    assert resolve_tokenizer_ref(None, fallback) == ("m-a-p/neo_scalinglaw_250M", None, "hf_ckpt/16.78B")


def test_tokenizer_ref_override_without_revision() -> None:
    fallback = CheckpointSource("bigscience/bloom-560m-intermediate", "global_step1000", 1000)
    assert resolve_tokenizer_ref("bigscience/bloom-560m", fallback) == ("bigscience/bloom-560m", None, "")


def test_tokenizer_ref_override_with_revision() -> None:
    fallback = CheckpointSource("some/repo", "step0", 0)
    assert resolve_tokenizer_ref("some/other-repo@main", fallback) == ("some/other-repo", "main", "")


def test_coerce_fire_csv_arg_rejoins_a_fire_parsed_tuple() -> None:
    """fire parses an unquoted CLI value like `0,512,8000` as the Python literal tuple
    (0, 512, 8000), not the string this pipeline's --steps/--targets parameters expect."""
    assert coerce_fire_csv_arg((0, 512, 8000)) == "0,512,8000"


def test_coerce_fire_csv_arg_rejoins_a_fire_parsed_single_int() -> None:
    """A single bare number (e.g. `--steps 8000`) arrives from fire as a plain int."""
    assert coerce_fire_csv_arg(8000) == "8000"


def test_coerce_fire_csv_arg_passes_through_an_already_correct_string() -> None:
    """A quoted value, a default, or a programmatic caller already passes a str; leave it alone."""
    assert coerce_fire_csv_arg("0,512,8000") == "0,512,8000"


def test_load_tokenizer_returns_the_direct_result_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    calls: list[tuple[str, dict]] = []

    def fake_from_pretrained(load_ref: str, **kwargs: object) -> object:
        calls.append((load_ref, kwargs))
        return sentinel

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    result = load_tokenizer("org/model", "main", "sub/path")
    assert result is sentinel
    assert calls == [("org/model", {"revision": "main", "subfolder": "sub/path", "trust_remote_code": True})]


def test_load_tokenizer_reraises_os_error_when_there_is_no_subfolder(monkeypatch: pytest.MonkeyPatch) -> None:
    """subfolder="" means this is not a hub-subfolder source, so the OSError can't be the known
    subfolder/dynamic-module bug this fallback exists for — there is nothing useful to retry."""

    def raise_os_error(load_ref: str, **kwargs: object) -> object:
        raise OSError("network down")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", raise_os_error)
    with pytest.raises(OSError, match="network down"):
        load_tokenizer("org/model", "main", "")


def test_load_tokenizer_falls_back_to_a_local_subfolder_snapshot_on_os_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Simulates the confirmed transformers 4.57.6 bug: the direct subfolder= call 404s, so the
    fallback downloads just that subfolder locally and retries from there (no subfolder= needed)."""
    sentinel = object()
    calls: list[tuple] = []
    (tmp_path / "sub" / "path").mkdir(parents=True)

    def fake_from_pretrained(load_ref: str, **kwargs: object) -> object:
        calls.append(("from_pretrained", load_ref, kwargs))
        if "subfolder" in kwargs:
            raise OSError("custom tokenizer code file not found (subfolder dropped)")
        return sentinel

    def fake_snapshot_download(load_ref: str, **kwargs: object) -> str:
        calls.append(("snapshot_download", load_ref, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    result = load_tokenizer("org/model", None, "sub/path")

    assert result is sentinel
    assert calls == [
        ("from_pretrained", "org/model", {"revision": None, "subfolder": "sub/path", "trust_remote_code": True}),
        ("snapshot_download", "org/model", {"revision": None, "allow_patterns": ["sub/path/*"]}),
        ("from_pretrained", str(tmp_path / "sub" / "path"), {"trust_remote_code": True}),
    ]


def test_load_tokenizer_reraises_when_the_fallback_also_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """If the local-snapshot retry fails too, the caller must still see an OSError (not some
    unrelated exception type), so a genuinely broken repo/network still surfaces as expected."""

    def always_os_error(load_ref: str, **kwargs: object) -> object:
        raise OSError(f"failed for {load_ref}")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", always_os_error)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda load_ref, **kwargs: str(tmp_path))
    with pytest.raises(OSError, match="failed for"):
        load_tokenizer("org/model", None, "sub/path")
