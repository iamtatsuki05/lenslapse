"""Lazy download + local cache for shipped models' precomputed data and tokenizer files.

The wheel only bundles the app shell and the models.json catalog (kept small enough for PyPI's
100 MB per-file limit); each shipped model's precomputed shards and tokenizer files are instead
fetched from this repo's own `main` branch via raw.githubusercontent.com on first use and cached
locally, so a `pip install lenslapse` still ends up fully offline-capable for the shipped models
after each one has been opened once. A repo checkout is unaffected: `web/public/data`/`tokenizer`
are already present on disk there, so nothing here is ever reached (see `_webapp_root` in
server.py, which is checked first).
"""

import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RAW_BASE = "https://raw.githubusercontent.com/iamtatsuki05/lenslapse/main/web/public"

# Every filename any shipped tokenizer currently uses (plain HF format, custom tiktoken/code
# tokenizers, or SentencePiece); unknown ones simply 404 upstream and are skipped.
_TOKENIZER_CANDIDATES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "qwen.tiktoken",
    "tokenization_qwen.py",
    "tokenization_neo.py",
    "chat_template.jinja",
]

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _safe(segment: str) -> bool:
    """Reject anything but a plain filename/id segment — no `/`, `..`, or other path escapes,
    since both come from request paths and end up in local filesystem paths and upstream URLs."""
    return bool(_SAFE_SEGMENT.match(segment)) and segment != ".." and segment != "."


def _fetch(url: str) -> bytes | None:
    """GET url, returning its body, or None on a 404. Any other failure (network down, DNS,
    5xx) raises, so callers surface a real error instead of silently treating it as "missing"."""
    try:
        with urlopen(Request(url, headers={"User-Agent": "lenslapse"}), timeout=15) as resp:
            return resp.read()
    except HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"fetching {url}: HTTP {e.code}") from e
    except URLError as e:
        raise RuntimeError(f"fetching {url}: {e.reason}") from e


def _write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(body)
    tmp.replace(path)  # atomic: a crash mid-download must not leave a corrupt cached file


def ensure_data_file(cache_root: Path, model_id: str, filename: str) -> Path | None:
    """Local path to data/<model_id>/<filename> (index.json or one prompt shard p{id}.json),
    downloading and caching it on first request. None if it doesn't exist upstream either."""
    if not (_safe(model_id) and _safe(filename)):
        return None
    # only p{id}.json (for an id actually listed in this model's own index) or index.json
    # itself is fetchable — rejects typos and any attempt to use this route to pull arbitrary
    # repo paths before touching the network at all, not just before the final download
    is_index = filename == "index.json"
    shard_match = re.fullmatch(r"p(\d+)\.json", filename)
    if not is_index and not shard_match:
        return None
    model_dir = cache_root / "data" / model_id
    index_path = model_dir / "index.json"
    if not index_path.is_file():
        body = _fetch(f"{RAW_BASE}/data/{model_id}/index.json")
        if body is None:
            return None
        _write(index_path, body)
    if is_index:
        return index_path
    assert shard_match is not None  # the guard above guarantees one of is_index/shard_match
    valid_ids = {p["id"] for p in json.loads(index_path.read_text()).get("prompts", [])}
    if int(shard_match.group(1)) not in valid_ids:
        return None
    target = model_dir / filename
    if target.is_file():
        return target
    body = _fetch(f"{RAW_BASE}/data/{model_id}/{filename}")
    if body is None:
        return None
    _write(target, body)
    return target


def ensure_tokenizer_file(cache_root: Path, model_id: str, filename: str) -> Path | None:
    """Local path to tokenizer/<model_id>/<filename>, downloading the model's full tokenizer
    (every file in _TOKENIZER_CANDIDATES that exists upstream) on first request for it. None if
    the model has no tokenizer directory upstream at all, or filename isn't a real member of it."""
    if not (_safe(model_id) and _safe(filename)) or filename not in _TOKENIZER_CANDIDATES:
        return None
    tok_dir = cache_root / "tokenizer" / model_id
    marker = tok_dir / ".complete"
    if not marker.is_file():
        found_any = False
        for name in _TOKENIZER_CANDIDATES:
            body = _fetch(f"{RAW_BASE}/tokenizer/{model_id}/{name}")
            if body is not None:
                _write(tok_dir / name, body)
                found_any = True
        if not found_any:
            return None
        tok_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    target = tok_dir / filename
    return target if target.is_file() else None
