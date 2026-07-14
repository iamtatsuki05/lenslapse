#!/bin/sh
# Refresh the app shell shipped inside the Python package (lenslapse/webapp).
# Run after changing web/ source; the data/tokenizer halves are force-included from
# web/public at wheel-build time, so they are excluded from the committed shell.
set -eu
cd "$(dirname "$0")/.."
(cd web && npm run build)
rm -rf lenslapse/webapp
cp -R web/dist lenslapse/webapp
rm -rf lenslapse/webapp/data lenslapse/webapp/tokenizer lenslapse/webapp/models
echo "lenslapse/webapp refreshed from web/dist"
