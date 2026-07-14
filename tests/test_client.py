"""CLI client: backend selection, UI-parity semantics (target fixing), and error mapping.

Hermetic — no HTTP sockets, no model weights. The trace/probe commands run against a fake
backend; the local backend runs against a tmp_path registry with Hub lookups stubbed.
"""

import argparse
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from lenslapse import client
from lenslapse.client import HttpBackend, cmd_probe, cmd_trace, infer_mode, model_entry, parse_targets


def ns(**kw: Any) -> argparse.Namespace:
    base = dict(model="m", text="t", step=None, pos=None, layer=None, top=3, targets=None, json=False)
    return argparse.Namespace(**{**base, **kw})


class FakeBackend:
    """Canned probe payloads in the exact shape the server returns."""

    where = "fake"

    def __init__(self, steps: list[int], layers: int = 3, positions: int = 2, mode: str = "suite") -> None:
        self.steps = steps
        self.layers, self.positions = layers, positions
        self.mode = mode
        self.top = [[" a", 0.5, 11], [" b", 0.3, 22], [" c", 0.1, 33]]
        self.calls: list[tuple[int, tuple[int, ...] | None]] = []

    def models(self) -> list[dict[str, Any]]:
        return [{"id": "m", "ref": "org/m", "mode": self.mode, "label": "M", "steps": self.steps, "origin": "user"}]

    def probe(self, model: str, step: int, text: str, targets: list[int] | None = None) -> dict[str, Any]:
        self.calls.append((step, tuple(targets) if targets else None))
        cell = {"token": " a", "prob": 0.5, "top": self.top}
        res: dict[str, Any] = {
            "model": model,
            "step": step,
            "text": text,
            "tokens": ["x"] * self.positions,
            "grid": {
                "layers": self.layers,
                "positions": self.positions,
                "cells": [[dict(cell) for _ in range(self.positions)] for _ in range(self.layers)],
            },
            "timing": {"total": 1, "forward": 1},
            "device": "cpu",
            "cached": False,
        }
        if targets:
            res["tgt"] = {
                str(t): {
                    "token": f"tok{t}",
                    "p": [[0.1 * (li + 1)] * self.positions for li in range(self.layers)],
                    "r": [[li + 1] * self.positions for li in range(self.layers)],
                }
                for t in targets
            }
        return res

    # unused Backend-protocol members (mypy checks commands against the full protocol)
    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def remove(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def convert_start(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def convert_status(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError


def test_parse_targets() -> None:
    assert parse_targets(None) is None
    assert parse_targets("3,1,3") == [3, 1]  # deduped, order kept
    with pytest.raises(SystemExit):
        parse_targets("abc")
    with pytest.raises(SystemExit):
        parse_targets(",")


def test_infer_mode(tmp_path: Path) -> None:
    assert infer_mode(str(tmp_path), None) == "local"
    assert infer_mode("org/m", "0,800") == "suite"
    assert infer_mode("org/m", None) == "final"


def test_model_entry_unknown_id_exits() -> None:
    with pytest.raises(SystemExit, match="unknown model id"):
        model_entry(FakeBackend([0]), "ghost")


def test_probe_defaults_to_the_last_checkpoint(capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend([0, 10, 20])
    cmd_probe(backend, ns())
    assert backend.calls == [(20, None)]
    out = capsys.readouterr().out
    assert "top-1 by layer at position 1" in out
    assert "(id 11)" in out  # the ids users feed back into --targets are printed


def test_probe_step_validation_depends_on_mode(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="no checkpoint for step 5"):  # local/final steps are exhaustive
        cmd_probe(FakeBackend([0, 100], mode="local"), ns(step=5))
    suite = FakeBackend([0, 100], mode="suite")  # a suite grid is a subset of the hub's revisions
    cmd_probe(suite, ns(step=5))
    assert suite.calls == [(5, None)]
    assert "outside" in capsys.readouterr().err


def test_probe_json_prints_the_exact_payload(capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend([0, 10])
    cmd_probe(backend, ns(step=10, targets="22", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["step"] == 10 and payload["tgt"]["22"]["token"] == "tok22"


def test_trace_fixes_targets_from_final_step_and_probes_every_checkpoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = FakeBackend([0, 10, 20])
    cmd_trace(backend, ns())
    # UI convention: one untargeted probe of the FINAL checkpoint fixes the top-3 target ids,
    # then every checkpoint (final included) is probed with exactly those ids
    assert backend.calls[0] == (20, None)
    assert backend.calls[1:] == [(0, (11, 22, 33)), (10, (11, 22, 33)), (20, (11, 22, 33))]
    out = capsys.readouterr().out
    assert "layer 2, position 1" in out  # defaults: final layer, last position
    assert "'tok11'" in out and "0.3000 (r3)" in out  # p[layer=2] = 0.3, r[2] = 3


def test_trace_with_explicit_targets_skips_the_fixing_probe() -> None:
    backend = FakeBackend([0, 10])
    cmd_trace(backend, ns(targets="7"))
    assert backend.calls == [(0, (7,)), (10, (7,))]


def test_trace_json_payload(capsys: pytest.CaptureFixture[str]) -> None:
    backend = FakeBackend([0, 10])
    cmd_trace(backend, ns(targets="7", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["targets"] == [{"id": 7, "token": "tok7"}]
    assert [s["step"] for s in payload["steps"]] == [0, 10]
    assert payload["steps"][0]["tgt"]["7"]["p"][2][1] == pytest.approx(0.3)


def test_trace_needs_a_suite() -> None:
    with pytest.raises(SystemExit, match="needs a suite"):
        cmd_trace(FakeBackend([0]), ns())


def test_trace_rejects_out_of_range_pos() -> None:
    with pytest.raises(SystemExit, match="--pos must be"):
        cmd_trace(FakeBackend([0, 10]), ns(pos=5))


def test_trace_rejects_bad_top() -> None:
    with pytest.raises(SystemExit, match="--top must be"):
        cmd_trace(FakeBackend([0, 10]), ns(top=0))


def test_main_routes_bare_models_to_list(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    picked: list[argparse.Namespace] = []

    def fake_choose(args: argparse.Namespace) -> FakeBackend:
        picked.append(args)
        return FakeBackend([0, 10])

    monkeypatch.setattr(client, "choose_backend", fake_choose)
    monkeypatch.setattr(client.sys, "argv", ["lenslapse", "models"])
    client.main()
    assert "org/m" in capsys.readouterr().out
    # bare `lenslapse models` must still carry the connection defaults choose_backend reads
    assert picked[0].server is None and picked[0].local is False and picked[0].device_map is False


def test_choose_backend_auto_detects_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    class StubLocal:
        where = "in-process"

        def __init__(self, device_map: bool = False) -> None:
            created.append("local")

    monkeypatch.setattr(client, "LocalBackend", StubLocal)
    monkeypatch.setattr(client, "server_alive", lambda base: True)
    args = argparse.Namespace(server=None, local=False, device_map=False)
    assert isinstance(client.choose_backend(args), HttpBackend)

    monkeypatch.setattr(client, "server_alive", lambda base: False)
    assert client.choose_backend(args).where == "in-process"

    args_local = argparse.Namespace(server=None, local=True, device_map=False)
    monkeypatch.setattr(client, "server_alive", lambda base: pytest.fail("--local must not even look for a server"))
    assert client.choose_backend(args_local).where == "in-process"

    with pytest.raises(SystemExit, match="no probe server responding"):
        monkeypatch.setattr(client, "server_alive", lambda base: False)
        client.choose_backend(argparse.Namespace(server="http://localhost:9", local=False, device_map=False))


def test_http_backend_surfaces_server_error_details(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_404(req: Any, timeout: float | None = None) -> Any:
        body = io.BytesIO(json.dumps({"detail": "unknown model id 'ghost'"}).encode())
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, body)  # type: ignore[arg-type]

    monkeypatch.setattr(client.urllib.request, "urlopen", raise_404)
    with pytest.raises(SystemExit, match="unknown model id 'ghost'"):
        HttpBackend("http://localhost:1").models()


def test_http_backend_builds_the_same_requests_as_the_web_app(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def capture(req: Any, timeout: float | None = None) -> FakeResponse:
        seen.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr(client.urllib.request, "urlopen", capture)
    backend = HttpBackend("http://localhost:1/")
    backend.probe("m", 8, "hi", [5])
    backend.remove("we ird/id")
    assert seen[0][0].full_url == "http://localhost:1/probe"
    assert json.loads(seen[0][0].data) == {"model": "m", "step": 8, "text": "hi", "targets": [5]}
    assert seen[0][1] is None  # a probe may download weights first — it must never time out
    assert seen[1][0].get_method() == "DELETE"
    assert seen[1][0].full_url == "http://localhost:1/models/we%20ird%2Fid"  # ids never break the URL path
    assert seen[1][1] == 30  # management calls do time out instead of hanging on a wedged server


def test_local_backend_registry_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lenslapse import server

    monkeypatch.setattr(server, "_STATE_HOME", tmp_path)
    monkeypatch.setattr(server, "default_models_json", lambda: tmp_path / "missing-models.json")
    backend = client.LocalBackend()
    assert backend.models() == []

    run = tmp_path / "trainer-run"
    (run / "checkpoint-0").mkdir(parents=True)
    (run / "checkpoint-800").mkdir()
    created = backend.add({"id": "my-run", "ref": str(run), "mode": "local"})
    assert created["steps"] == [0, 800]
    assert json.loads((tmp_path / "registry.json").read_text())["my-run"]["mode"] == "local"

    with pytest.raises(SystemExit, match="already registered"):  # HTTPException mapped to a clean exit
        backend.add({"id": "my-run", "ref": str(run), "mode": "local"})
    with pytest.raises(SystemExit, match="lenslapse add-model"):  # conversion is a server job
        backend.convert_start("my-run")
    assert backend.remove("my-run") == {"removed": "my-run"}
    assert backend.models() == []
