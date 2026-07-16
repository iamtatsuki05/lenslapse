#!/bin/sh
# Refresh the app shell shipped inside the Python package (src/lenslapse/webapp).
# Run after changing web/ source; models.json is force-included from web/public at wheel-build
# time (so it's excluded here too), and per-model data/tokenizer files are fetched on demand at
# runtime instead of being bundled at all (see lenslapse/webdata.py) -- excluded here so a repo
# checkout's own web/public copies (used directly via web/dist) are never shadowed by stale ones.
set -eu
cd "$(dirname "$0")/.."
(cd web && npm run build)
rm -rf src/lenslapse/webapp
cp -R web/dist src/lenslapse/webapp
rm -rf src/lenslapse/webapp/data src/lenslapse/webapp/tokenizer src/lenslapse/webapp/models
echo "src/lenslapse/webapp refreshed from web/dist"
