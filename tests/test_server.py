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


def test_convert_guards(client: TestClient, trainer_dir: Path) -> None:
    assert client.post("/models/ghost/convert").status_code == 404
    server.STATE["registry"]["shipped"] = RegistryEntry(ref="org/shipped", mode="suite", origin="catalog")
    assert client.post("/models/shipped/convert").status_code == 400
    register_local(client, trainer_dir)
    server.JOBS["other"] = {"status": "running", "log": deque()}
    assert client.post("/models/my-run/convert").status_code == 409  # one conversion at a time
    assert client.get("/models/ghost/convert").status_code == 404


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
