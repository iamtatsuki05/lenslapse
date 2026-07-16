"""Probe-server management API: registry, registration validation, persistence, folder picker.

Hermetic: Hub lookups are stubbed, everything else runs against tmp_path. The /probe endpoint
itself is exercised end-to-end elsewhere (it loads real model weights), not here.
"""

import json
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lenslapse import server
from lenslapse.server import RegistryEntry, build_registry, model_steps, read_cache


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setitem(server.STATE, "registry", {})
    monkeypatch.setitem(server.STATE, "registry_file", tmp_path / "registry.json")
    monkeypatch.setitem(server.STATE, "cache_dir", tmp_path / "probe-cache")
    monkeypatch.setitem(server.STATE, "models_root", tmp_path / "exported-models")
    (tmp_path / "probe-cache").mkdir()
    monkeypatch.setattr(server, "JOBS", {})
    with TestClient(server.app) as c:
        yield c


@pytest.fixture()
def trainer_dir(tmp_path: Path) -> Path:
    run = tmp_path / "trainer-run"
    for name in ("checkpoint-0", "checkpoint-800"):
        (run / name).mkdir(parents=True)
    return run


def register_local(client: TestClient, trainer_dir: Path, model_id: str = "my-run") -> Any:
    return client.post("/models", json={"id": model_id, "ref": str(trainer_dir), "mode": "local", "label": "My run"})


def test_health_lists_registered_models(client: TestClient, trainer_dir: Path) -> None:
    register_local(client, trainer_dir)
    assert client.get("/health").json()["models"] == ["my-run"]


def test_register_local_scans_steps_and_persists(client: TestClient, trainer_dir: Path) -> None:
    res = register_local(client, trainer_dir)
    assert res.status_code == 201
    assert res.json()["steps"] == [0, 800]
    saved = json.loads(server.STATE["registry_file"].read_text())
    assert saved["my-run"]["mode"] == "local"
    assert "origin" not in saved["my-run"]


def test_register_response_omits_label_when_unset(client: TestClient, trainer_dir: Path) -> None:
    """RegisterResponse.label is a pydantic Optional field (unlike the old TypedDict's
    NotRequired), so the route must set response_model_exclude_none to keep an unset label
    genuinely absent from the JSON rather than serialized as `"label": null`."""
    res = client.post("/models", json={"id": "no-label", "ref": str(trainer_dir), "mode": "local"})
    assert res.status_code == 201
    assert "label" not in res.json()

    labeled = register_local(client, trainer_dir, model_id="has-label")
    assert labeled.json()["label"] == "My run"


def test_register_rejects_bad_input(client: TestClient, trainer_dir: Path) -> None:
    bad_id = client.post("/models", json={"id": "../evil", "ref": "gpt2", "mode": "final"})
    assert bad_id.status_code == 400
    bad_mode = client.post("/models", json={"id": "x", "ref": "gpt2", "mode": "banana"})
    assert bad_mode.status_code == 400
    steps_on_final = client.post("/models", json={"id": "x", "ref": "gpt2", "mode": "final", "steps": [0]})
    assert steps_on_final.status_code == 400
    missing_dir = client.post("/models", json={"id": "x", "ref": "/no/such/dir", "mode": "local"})
    assert missing_dir.status_code == 400
    register_local(client, trainer_dir)
    duplicate = register_local(client, trainer_dir)
    assert duplicate.status_code == 409


def test_register_hub_failure_is_a_400_not_a_500(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("no such repo")

    monkeypatch.setattr(server, "model_info", boom)
    res = client.post("/models", json={"id": "nope", "ref": "no/such-model", "mode": "final"})
    assert res.status_code == 400
    assert "no such repo" in res.json()["detail"]


def test_unregister_guards(client: TestClient, trainer_dir: Path) -> None:
    assert client.delete("/models/ghost").status_code == 404
    server.STATE["registry"]["shipped"] = RegistryEntry(ref="org/shipped", mode="suite", origin="catalog")
    assert client.delete("/models/shipped").status_code == 400
    register_local(client, trainer_dir)
    server.JOBS["my-run"] = {"status": "running", "log": deque()}
    assert client.delete("/models/my-run").status_code == 409
    server.JOBS["my-run"]["status"] = "done"
    assert client.delete("/models/my-run").status_code == 200
    assert json.loads(server.STATE["registry_file"].read_text()) == {}
    assert "my-run" not in server.JOBS  # a re-registration must not inherit the old job


def test_registry_precedence_and_roundtrip(tmp_path: Path) -> None:
    models_json = tmp_path / "models.json"
    models_json.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "a", "hf": "a", "label": "A", "mode": "final", "source": "org/a"},
                    {"id": "b", "hf": "org/b", "label": "B"},
                    {"id": "skip-me", "hf": "skip-me", "mode": "local", "source": "/somewhere"},
                ]
            }
        )
    )
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps({"b": {"ref": "org/b-override", "mode": "final"}}))
    registry = build_registry(models_json, registry_file, ["c=org/c:final"])
    assert registry["a"].ref == "org/a" and registry["a"].origin == "catalog"
    assert registry["b"].ref == "org/b-override" and registry["b"].origin == "user"
    assert registry["c"].mode == "final" and registry["c"].origin == "user"
    assert "skip-me" not in registry  # machine-specific local paths never come from models.json
    # entries without a custom revision naming / tokenizer source get the hub-suite defaults
    assert registry["a"].revision_template == "step{}" and registry["a"].tokenizer_ref is None


def test_build_registry_reads_revision_template_and_tokenizer_ref(tmp_path: Path) -> None:
    """A models.json catalog entry can override the hub suite's revision naming (e.g. BLOOM's
    global_step{N}) and tokenizer source (e.g. a checkpoint whose own tokenizer fails to load) —
    the live-probe registry must carry these through, not just the offline export/precompute CLIs."""
    models_json = tmp_path / "models.json"
    models_json.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "bloom-560m",
                        "hf": "bloom-560m",
                        "mode": "suite",
                        "source": "bigscience/bloom-560m-intermediate",
                        "steps": [1000, 10000],
                        "revision_template": "global_step{}",
                        "tokenizer_ref": "bigscience/bloom-560m",
                    }
                ]
            }
        )
    )
    registry = build_registry(models_json, None, [])
    assert registry["bloom-560m"].revision_template == "global_step{}"
    assert registry["bloom-560m"].tokenizer_ref == "bigscience/bloom-560m"


def test_source_for_uses_the_registered_revision_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """source_for() (the live /probe and /tokenize path) must resolve suite revisions the same
    way the offline export/precompute pipeline does, not always assume the step{N} default."""
    monkeypatch.setitem(
        server.STATE,
        "registry",
        {
            "bloom-560m": RegistryEntry(
                ref="bigscience/bloom-560m-intermediate",
                mode="suite",
                steps=[1000, 10000],
                revision_template="global_step{}",
            )
        },
    )
    src = server.source_for("bloom-560m", 1000)
    assert src.revision == "global_step1000"


def test_source_for_uses_the_registered_subfolder_map(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAP-Neo/Aquila-style repos nest checkpoints as subfolders of one revision instead of using
    git revisions per checkpoint; source_for() must resolve those the same way the offline
    export/precompute pipeline's --subfolder-map does, not assume a step{N} git revision."""
    monkeypatch.setitem(
        server.STATE,
        "registry",
        {
            "mapneo-250m": RegistryEntry(
                ref="m-a-p/neo_scalinglaw_250M",
                mode="suite",
                steps=[16780, 33550],
                subfolder_map="16780:hf_ckpt/16.78B,33550:hf_ckpt/33.55B",
                tokenizer_ref="m-a-p/neo_7b",
            )
        },
    )
    src = server.source_for("mapneo-250m", 33550)
    assert src.subfolder == "hf_ckpt/33.55B"
    assert src.revision is None  # subfolder sources have no git revision of their own
    with pytest.raises(HTTPException):
        server.source_for("mapneo-250m", 99999)


def test_model_steps_per_mode(trainer_dir: Path) -> None:
    assert model_steps(RegistryEntry(ref="gpt2", mode="final")) == [0]
    assert model_steps(RegistryEntry(ref=str(trainer_dir), mode="local")) == [0, 800]
    assert model_steps(RegistryEntry(ref="/vanished", mode="local")) == []
    assert model_steps(RegistryEntry(ref="org/x", mode="suite", steps=[0, 5])) == [0, 5]
    assert model_steps(RegistryEntry(ref="org/x", mode="suite")) == server.DEFAULT_SUITE_STEPS


def test_read_cache_replays_and_heals_corruption(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"tokens": ["a"]}))
    assert read_cache(good) == {"tokens": ["a"], "cached": True}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert read_cache(corrupt) is None
    assert not corrupt.exists()  # deleted so the next probe recomputes instead of erroring forever


def test_tokenize_uses_the_models_own_tokenizer(
    client: TestClient, trainer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubTok:
        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            assert add_special_tokens is False  # a BOS must never become the tracked token
            return {"input_ids": list(range(len(text.split())))}

        def convert_ids_to_tokens(self, ids: list) -> list:
            return [f"tok{i}" for i in ids]

        def convert_tokens_to_string(self, tokens: list) -> str:
            return "".join(tokens)  # plain stub vocab strings carry no marker to reverse

    monkeypatch.setattr(server, "load_tokenizer", lambda load_ref, revision, subfolder: StubTok())
    monkeypatch.setattr(server, "TOKENIZERS", server.OrderedDict())
    assert client.post("/tokenize", json={"model": "ghost", "text": "hi"}).status_code == 404
    register_local(client, trainer_dir)
    res = client.post("/tokenize", json={"model": "my-run", "text": "one two three"})
    assert res.status_code == 200
    assert res.json() == {"ids": [0, 1, 2], "tokens": ["tok0", "tok1", "tok2"]}


def test_convert_guards(client: TestClient, trainer_dir: Path) -> None:
    assert client.post("/models/ghost/convert").status_code == 404
    server.STATE["registry"]["shipped"] = RegistryEntry(ref="org/shipped", mode="suite", origin="catalog")
    assert client.post("/models/shipped/convert").status_code == 400
    register_local(client, trainer_dir)
    server.JOBS["other"] = {"status": "running", "log": deque()}
    assert client.post("/models/my-run/convert").status_code == 409  # one conversion at a time
    assert client.get("/models/ghost/convert").status_code == 404


def test_convert_status_omits_note_and_log_when_unset(
    client: TestClient, trainer_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ConvertStatusResponse.log/note are pydantic Optional fields; the routes must keep them
    genuinely absent from the JSON (not `null`) exactly where the old TypedDict-based dict
    literal never set the key at all."""
    # the real conversion job is a background subprocess (add_model.py); stub it so the POST
    # below only exercises response construction, not a real export
    monkeypatch.setattr(server, "_run_convert", lambda model_id, job, cmd: None)
    register_local(client, trainer_dir)
    started = client.post("/models/my-run/convert")
    assert started.status_code == 202
    assert "log" not in started.json() and "note" not in started.json()

    server.JOBS["my-run"] = {"status": "running", "log": deque(["line1"])}
    running = client.get("/models/my-run/convert")
    assert running.json()["log"] == ["line1"]
    assert "note" not in running.json()  # note is only set once the job is done


def test_pick_folder_paths(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_folder_dialog_cmd", lambda: None)
    assert client.get("/pick-folder").status_code == 501

    import sys

    monkeypatch.setattr(server, "_folder_dialog_cmd", lambda: [sys.executable, "-c", "print('/tmp/picked/')"])
    res = client.get("/pick-folder")
    assert res.status_code == 200 and res.json() == {"path": "/tmp/picked"}

    monkeypatch.setattr(server, "_folder_dialog_cmd", lambda: [sys.executable, "-c", "raise SystemExit(0)"])
    assert client.get("/pick-folder").status_code == 400  # no output + rc 0 = cancelled

    monkeypatch.setattr(
        server,
        "_folder_dialog_cmd",
        lambda: [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); raise SystemExit(2)"],
    )
    res = client.get("/pick-folder")
    assert res.status_code == 500 and "boom" in res.json()["detail"]


def test_webapp_root_prefers_fresh_dist_and_requires_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dist = tmp_path / "web" / "dist"
    packaged = tmp_path / "pkg" / "webapp"
    monkeypatch.setattr(server, "_webapp_candidates", lambda: [dist, packaged])

    # neither exists -> hosted-app fallback
    assert server._webapp_root() is None

    # a shell without data (repo tree without the wheel's force-included halves) is skipped
    packaged.mkdir(parents=True)
    (packaged / "index.html").write_text("<html></html>")
    assert server._webapp_root() is None

    # the packaged bundle works once complete
    (packaged / "data").mkdir()
    (packaged / "data" / "models.json").write_text("{}")
    assert server._webapp_root() == packaged

    # ...but a fresh checkout build wins over it
    (dist / "data").mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")
    (dist / "data" / "models.json").write_text("{}")
    assert server._webapp_root() == dist


def test_shipped_data_routes_reject_traversal_segments(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single '..' segment routes fine and previously joined straight into the webapp path —
    GET /data/%2e%2e/index.html served webapp/index.html. Both routes must apply the same
    segment guard the download fallback applies to itself, and never reach the network."""
    webapp = tmp_path / "webapp"
    (webapp / "data").mkdir(parents=True)
    (webapp / "index.html").write_text("<html>shell</html>")
    (webapp / "data" / "models.json").write_text("{}")
    monkeypatch.setattr(server, "_webapp_root", lambda: webapp)
    monkeypatch.setattr(server, "ensure_data_file", lambda *a: pytest.fail("must not fall through to a download"))
    monkeypatch.setattr(server, "ensure_tokenizer_file", lambda *a: pytest.fail("must not fall through to a download"))

    assert client.get("/data/%2e%2e/index.html").status_code == 404
    assert client.get("/tokenizer/%2e%2e/index.html").status_code == 404
    assert client.get("/data/pythia-70m/%2e%2e").status_code == 404
    # a well-formed request still serves the bundled file, without touching the stubbed fallback
    (webapp / "data" / "pythia-70m").mkdir()
    (webapp / "data" / "pythia-70m" / "index.json").write_text("{}")
    assert client.get("/data/pythia-70m/index.json").status_code == 200


def test_probe_cache_key_cannot_collide_across_text_and_targets() -> None:
    from lenslapse.server import ProbeRequest, probe_cache_key
    from lenslapse.sources import CheckpointSource

    src = CheckpointSource("org/m", "step0", 0)
    plain = probe_cache_key(src, ProbeRequest(model="m", step=0, text='foo", "targets": [5]'))
    targeted = probe_cache_key(src, ProbeRequest(model="m", step=0, text="foo", targets=[5]))
    assert plain != targeted
    # the old concatenation scheme collided exactly here
    concat_a = probe_cache_key(src, ProbeRequest(model="m", step=0, text="foo::targets=5"))
    concat_b = probe_cache_key(src, ProbeRequest(model="m", step=0, text="foo", targets=[5]))
    assert concat_a != concat_b
    # target order and duplicates do not change the key
    assert probe_cache_key(src, ProbeRequest(model="m", step=0, text="foo", targets=[5, 3, 5])) == probe_cache_key(
        src, ProbeRequest(model="m", step=0, text="foo", targets=[3, 5])
    )


def test_probe_cache_key_includes_compute_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """fp16 and fp32 forwards genuinely disagree at late checkpoints — one must not replay as the other."""
    from lenslapse.server import ProbeRequest, probe_cache_key
    from lenslapse.sources import CheckpointSource

    src = CheckpointSource("org/m", "step0", 0)
    req = ProbeRequest(model="m", step=0, text="foo")
    fp32_key = probe_cache_key(src, req)
    monkeypatch.setitem(server.STATE, "dtype", "float16")
    assert probe_cache_key(src, req) != fp32_key


def test_probe_cache_key_distinguishes_subfolder_checkpoints() -> None:
    """Hub-subfolder suites (MAP-Neo, Aquila) have no per-checkpoint git revision — every step
    shares the same (load_ref, revision) — so subfolder must be part of the key, or two different
    checkpoints of the same suite would replay each other's cached probe results."""
    from lenslapse.server import ProbeRequest, probe_cache_key
    from lenslapse.sources import CheckpointSource

    req = ProbeRequest(model="m", step=0, text="foo")
    a = CheckpointSource("org/m", None, 16780, subfolder="hf_ckpt/16.78B")
    b = CheckpointSource("org/m", None, 33550, subfolder="hf_ckpt/33.55B")
    assert probe_cache_key(a, req) != probe_cache_key(b, req)


class StubModel:
    """A weightless stand-in for the two load() tests below — enough surface for load()'s
    .to(device).eval() chain."""

    def to(self, device: str) -> "StubModel":
        return self

    def eval(self) -> "StubModel":
        return self


def test_load_defaults_to_float32_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    from lenslapse.sources import CheckpointSource

    seen: dict[str, Any] = {}

    class StubAuto:
        @staticmethod
        def from_pretrained(ref: str, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return StubModel()

    monkeypatch.setattr(server, "AutoModelForCausalLM", StubAuto)
    monkeypatch.setattr(server, "load_tokenizer", lambda load_ref, revision, subfolder: object())
    monkeypatch.setattr(server, "LOADED", server.OrderedDict())
    monkeypatch.setitem(server.STATE["registry"], "m", RegistryEntry(ref="org/m", mode="suite"))
    server.load("m", CheckpointSource("org/m", "step0", 0))
    assert seen["dtype"] == "float32"


def test_load_distinguishes_subfolder_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same regression as probe_cache_key: hub-subfolder suites share (load_ref, revision) across
    every step, so the in-memory model cache must key on subfolder too, or probing a second
    checkpoint of the same suite would silently reuse the first one's weights."""
    from lenslapse.sources import CheckpointSource

    loaded_subfolders: list[str] = []

    class StubAuto:
        @staticmethod
        def from_pretrained(ref: str, subfolder: str = "", **kwargs: Any) -> Any:
            loaded_subfolders.append(subfolder)
            return StubModel()

    monkeypatch.setattr(server, "AutoModelForCausalLM", StubAuto)
    monkeypatch.setattr(server, "load_tokenizer", lambda load_ref, revision, subfolder: object())
    monkeypatch.setattr(server, "LOADED", server.OrderedDict())
    monkeypatch.setitem(server.STATE["registry"], "mapneo", RegistryEntry(ref="org/mapneo", mode="suite"))
    server.load("mapneo", CheckpointSource("org/mapneo", None, 16780, subfolder="hf_ckpt/16.78B"))
    server.load("mapneo", CheckpointSource("org/mapneo", None, 33550, subfolder="hf_ckpt/33.55B"))
    # if (load_ref, revision) alone were the cache key, the second load() would have hit the
    # first's cache entry and from_pretrained would show only one call, not two
    assert loaded_subfolders == ["hf_ckpt/16.78B", "hf_ckpt/33.55B"]
