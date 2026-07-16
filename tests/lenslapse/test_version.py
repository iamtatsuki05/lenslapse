"""__version__ and pyproject.toml are two statements of the same fact; a release bump that
touches only one of them shipped once (0.1.1 left __init__ at 0.1.0) — this pins them together."""

import tomllib
from pathlib import Path

import lenslapse


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert lenslapse.__version__ == tomllib.load(f)["project"]["version"]
