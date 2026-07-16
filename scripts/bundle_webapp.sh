#!/bin/sh
# Refresh the app shell shipped inside the Python package (src/lenslapse/webapp).
# Run after changing web/ source; the data/tokenizer halves are force-included from
# web/public at wheel-build time, so they are excluded from the committed shell.
set -eu
cd "$(dirname "$0")/.."
(cd web && npm run build)
rm -rf src/lenslapse/webapp
cp -R web/dist src/lenslapse/webapp
rm -rf src/lenslapse/webapp/data src/lenslapse/webapp/tokenizer src/lenslapse/webapp/models
echo "src/lenslapse/webapp refreshed from web/dist"
