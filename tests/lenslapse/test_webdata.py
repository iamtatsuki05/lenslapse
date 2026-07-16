"""Lazy download + cache for shipped models' data/tokenizer files — no real network calls;
`_fetch` is monkeypatched throughout so these stay hermetic and fast."""

import json
from pathlib import Path

import pytest

from lenslapse import webdata

INDEX = json.dumps({"prompts": [{"id": 0, "text": "a"}, {"id": 3, "text": "b"}]}).encode()


def test_ensure_data_file_downloads_index_and_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_fetch(url: str) -> bytes | None:
        calls.append(url)
        if url.endswith("index.json"):
            return INDEX
        if url.endswith("p0.json"):
            return b'{"cell": 1}'
        return None

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    path = webdata.ensure_data_file(tmp_path, "pythia-70m", "p0.json")
    assert path is not None
    assert path.read_bytes() == b'{"cell": 1}'
    assert calls == [
        f"{webdata.RAW_BASE}/data/pythia-70m/index.json",
        f"{webdata.RAW_BASE}/data/pythia-70m/p0.json",
    ]


def test_ensure_data_file_caches_after_first_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_fetch(url: str) -> bytes | None:
        calls.append(url)
        return INDEX if url.endswith("index.json") else b"{}"

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    webdata.ensure_data_file(tmp_path, "pythia-70m", "index.json")
    webdata.ensure_data_file(tmp_path, "pythia-70m", "index.json")
    assert calls == [f"{webdata.RAW_BASE}/data/pythia-70m/index.json"]  # second call: cache hit, no re-fetch


def test_ensure_data_file_rejects_prompt_id_not_in_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_fetch(url: str) -> bytes | None:
        calls.append(url)
        return INDEX

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    assert webdata.ensure_data_file(tmp_path, "pythia-70m", "p999.json") is None
    assert calls == [f"{webdata.RAW_BASE}/data/pythia-70m/index.json"]  # rejected before a shard fetch


@pytest.mark.parametrize(
    ("model_id", "filename"),
    [
        ("../../etc", "index.json"),
        ("pythia-70m", "../../../etc/passwd"),
        ("pythia-70m", "not-a-shard.txt"),
        ("..", "index.json"),
        (".", "index.json"),
    ],
)
def test_ensure_data_file_rejects_unsafe_or_invalid_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_id: str, filename: str
) -> None:
    monkeypatch.setattr(webdata, "_fetch", lambda url: pytest.fail(f"must not fetch: {url}"))
    assert webdata.ensure_data_file(tmp_path, model_id, filename) is None


def test_ensure_data_file_returns_none_for_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webdata, "_fetch", lambda url: None)
    assert webdata.ensure_data_file(tmp_path, "no-such-model", "index.json") is None


def test_ensure_tokenizer_file_downloads_only_existing_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = {"tokenizer.json", "tokenizer_config.json"}
    fetched = []

    def fake_fetch(url: str) -> bytes | None:
        fetched.append(url)
        name = url.rsplit("/", 1)[-1]
        return b"content" if name in present else None

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    path = webdata.ensure_tokenizer_file(tmp_path, "gemma3-270m", "tokenizer_config.json")
    assert path is not None and path.read_bytes() == b"content"
    assert len(fetched) == len(webdata._TOKENIZER_CANDIDATES)  # tried every candidate, once
    for name in present:
        assert (tmp_path / "tokenizer" / "gemma3-270m" / name).is_file()
    for name in set(webdata._TOKENIZER_CANDIDATES) - present:
        assert not (tmp_path / "tokenizer" / "gemma3-270m" / name).is_file()


def test_ensure_tokenizer_file_fetches_candidates_once_across_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []

    def fake_fetch(url: str) -> bytes | None:
        fetched.append(url)
        return b"x"

    monkeypatch.setattr(webdata, "_fetch", fake_fetch)
    webdata.ensure_tokenizer_file(tmp_path, "gemma3-270m", "tokenizer.json")
    first_round = len(fetched)
    webdata.ensure_tokenizer_file(tmp_path, "gemma3-270m", "tokenizer_config.json")  # different file, same model
    assert len(fetched) == first_round  # marker from the first call short-circuits this one entirely


def test_ensure_tokenizer_file_rejects_unknown_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webdata, "_fetch", lambda url: pytest.fail(f"must not fetch: {url}"))
    assert webdata.ensure_tokenizer_file(tmp_path, "gemma3-270m", "not_a_real_file.json") is None


def test_ensure_tokenizer_file_no_tokenizer_upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webdata, "_fetch", lambda url: None)
    assert webdata.ensure_tokenizer_file(tmp_path, "pythia-70m", "tokenizer.json") is None
    assert not (tmp_path / "tokenizer" / "pythia-70m").exists()  # no marker left behind for a later retry
