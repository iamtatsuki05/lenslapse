#!/bin/sh
# Build the Sphinx documentation site into docs/_build/html.
# sphinx-apidoc regenerates the API reference stubs (docs/api/, gitignored) from src/lenslapse
# on every build so they can never go stale against the code; sphinx-build then renders the
# stubs plus the hand-written Markdown guides already in docs/. Deployed under /docs/ on the
# GitHub Pages site by deploy-pages.yml.
set -eu
cd "$(dirname "$0")/.."
uv run --group docs sphinx-apidoc -f -o docs/api src/lenslapse
# -b (direct mode), NOT -M: make-mode swallows --fail-on-warning's non-zero exit on Sphinx 9.1,
# printing "warnings treated as errors" while still exiting 0 — the gate would be decorative
uv run --group docs sphinx-build -b html --fail-on-warning docs docs/_build/html
echo "docs built at docs/_build/html/index.html"
