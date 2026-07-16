#!/bin/sh
# Build the Sphinx documentation site into docs/_build/html.
# sphinx-apidoc regenerates the API reference stubs (docs/api/, gitignored) from src/lenslapse
# on every build so they can never go stale against the code; sphinx-build then renders the
# stubs plus the hand-written Markdown guides already in docs/. Deployed under /docs/ on the
# GitHub Pages site by deploy-pages.yml.
set -eu
cd "$(dirname "$0")/.."
uv run --group docs sphinx-apidoc -f -o docs/api src/lenslapse
uv run --group docs sphinx-build -M html docs docs/_build --fail-on-warning
echo "docs built at docs/_build/html/index.html"
