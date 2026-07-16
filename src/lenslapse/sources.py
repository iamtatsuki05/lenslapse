"""Resolve a model source into (load_ref, revision, step) tuples for export/precompute.

Four source shapes are supported:
  1. Hub suite     : --model EleutherAI/pythia-70m --steps 0,1000,...   -> revisions step{N}
  2. Hub single    : --model gpt2 --final-only                          -> revision main, step 0
  3. Local dir     : --model /path/to/run --local-checkpoints           -> checkpoint-*/ subdirs
                     (Hugging Face Trainer layout; the number suffix is the training step), or a
                     plain single-model directory when no checkpoint-* subdirs exist.
  4. Hub subfolder : --model m-a-p/neo_scalinglaw_250M --subfolder-map "16780:hf_ckpt/16.78B,..."
                     for repos that nest each checkpoint as a subfolder within a single revision
                     rather than using git revisions per checkpoint (seen on Megatron-derived HF
                     exports that keep the raw training state and converted HF-format weights in
                     the same repo, e.g. MAP-Neo, BAAI Aquila).
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Shared across server.py (registry/fire CLI) and client.py (fire CLI) — defined here, not in
# server.py, because this module has no torch/fastapi import: client.py's HTTP-only path must
# stay able to import these without paying for the heavy deps server.py pulls in.
Mode = Literal["suite", "final", "local"]
MODE_CHOICES: tuple[Mode, ...] = ("suite", "final", "local")

DType = Literal["float32", "float16", "bfloat16", "auto"]
DTYPE_CHOICES: tuple[DType, ...] = ("float32", "float16", "bfloat16", "auto")

# The Pythia suite's public checkpoint grid — the default for every CLI's --steps and the
# server's suite mode. One definition: the CLIs, their Config models, and DEFAULT_SUITE_STEPS
# in server.py must all agree or exported/precomputed/served checkpoint sets silently diverge.
DEFAULT_STEPS_CSV = "0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000"


def coerce_fire_csv_arg(v: object) -> object:
    """fire always tries to parse a bare CLI value as a Python literal before falling back to
    str, so an unquoted comma-separated list of numbers (e.g. `--steps 0,512,8000`) arrives as
    the tuple (0, 512, 8000) instead of a string, and a single bare number (e.g. `--steps 8000`)
    arrives as a plain int — for every CLI parameter in this package that expects a
    comma-separated string of ids/steps (`steps`, `targets`). Use as a pydantic
    `field_validator(mode="before")` to normalize back to the comma-joined string the rest of the
    pipeline expects; a value already shaped as a str (the default, or a programmatic caller)
    passes through unchanged."""
    if isinstance(v, (tuple, list)):
        return ",".join(str(x) for x in v)
    if isinstance(v, int):
        return str(v)
    return v


@dataclass
class CheckpointSource:
    """A pydantic dataclass (not BaseModel): callers throughout this package construct it
    positionally (CheckpointSource(model, revision, step)), which BaseModel's keyword-only
    __init__ does not support — the dataclass shape keeps that calling convention while still
    validating field types."""

    load_ref: str  # HF id or local path passed to from_pretrained
    revision: str | None  # HF revision, None for local paths and subfolder sources
    step: int  # slider position in the app
    subfolder: str | None = None  # path within load_ref, for hub-subfolder sources (shape 4 above)

    @property
    def name(self) -> str:
        return f"step{self.step}"


def resolve_sources(
    model: str, steps_arg: str, final_only: bool, revision_template: str = "step{}"
) -> list[CheckpointSource]:
    path = Path(model)
    if path.exists() and path.is_dir():
        ckpts = sorted((step, c) for c in path.glob("checkpoint-*") if (step := _ckpt_step(c)) is not None)
        if ckpts:
            if final_only:
                raise SystemExit(
                    f"{model} is a Trainer directory with checkpoint-* subdirs; --final-only is ambiguous — "
                    "pass the specific checkpoint directory instead"
                )
            return [CheckpointSource(str(c), None, step) for step, c in ckpts]
        # plain local model directory: single checkpoint at step 0
        return [CheckpointSource(str(path), None, 0)]
    if final_only:
        return [CheckpointSource(model, "main", 0)]
    steps = sorted({int(s) for s in steps_arg.split(",")})
    return [CheckpointSource(model, revision_template.format(s), s) for s in steps]


def _ckpt_step(p: Path) -> int | None:
    m = re.fullmatch(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else None


def resolve_subfolder_sources(model: str, subfolder_map: str) -> list[CheckpointSource]:
    """Hub subfolder suite (shape 4): subfolder_map is "step:path,step:path,...", e.g.
    "16780:hf_ckpt/16.78B,33550:hf_ckpt/33.55B". step is whatever synthetic slider value the
    caller has already decided on (e.g. tokens in millions when the repo labels checkpoints by
    tokens rather than optimizer steps) — this function does no unit conversion of its own."""
    by_step: dict[int, str] = {}  # last one wins on a duplicate step, matching resolve_sources' dedup
    for pair in subfolder_map.split(","):
        step_s, sub = pair.split(":", 1)
        by_step[int(step_s)] = sub
    return [CheckpointSource(model, None, step, subfolder=sub) for step, sub in sorted(by_step.items())]


def resolve_all_sources(
    model: str,
    steps: str,
    final_only: bool = False,
    subfolder_map: str | None = None,
    revision_template: str = "step{}",
) -> list[CheckpointSource]:
    """The one precedence rule every CLI shares: a --subfolder-map fully replaces the
    steps/final-only/revision-template resolution (shape 4 repos label checkpoints by
    subfolder, not by git revision)."""
    if subfolder_map:
        return resolve_subfolder_sources(model, subfolder_map)
    return resolve_sources(model, steps, final_only, revision_template)


def resolve_tokenizer_ref(tokenizer_ref: str | None, fallback: CheckpointSource) -> tuple[str, str | None, str]:
    """Where to load the tokenizer from: `tokenizer_ref` ("repo_id" or "repo_id@revision") when
    given, else the same ref as `fallback` (typically sources[0]). Returns
    (load_ref, revision, subfolder) ready to splat into AutoTokenizer.from_pretrained(...).

    A --tokenizer-ref override is for repos where the per-checkpoint tokenizer files don't load
    cleanly (bigscience/bloom-*-intermediate, a transformers-version incompatibility) or live in a
    separate repo entirely (m-a-p/neo_scalinglaw_*, whose tokenizer is only published under
    m-a-p/neo_7b) — always safe when it applies, since the tokenizer is identical across
    checkpoints of the same pretraining run.
    """
    if not tokenizer_ref:
        return fallback.load_ref, fallback.revision, fallback.subfolder or ""
    ref, _, rev = tokenizer_ref.partition("@")
    return ref, (rev or None), ""


def load_tokenizer(
    load_ref: str, revision: str | None, subfolder: str, *, trust_remote_code: bool = True
) -> "PreTrainedTokenizerBase":
    """AutoTokenizer.from_pretrained, with a fallback for a confirmed transformers limitation
    (checked against transformers 4.57.6): a repo whose tokenizer needs custom code
    (trust_remote_code=True, e.g. Qwen-family tokenizers) does not pass `subfolder` through to
    that code file's own download — only the tokenizer_config.json/vocab file resolution honors
    it — so a subfolder-nested custom tokenizer 404s on the code file even though the exact same
    weights load fine via AutoModelForCausalLM.from_pretrained(subfolder=...). This is generic
    (not specific to one model): any future subfolder-suite architecture with a custom-code
    tokenizer can hit it. When it does, download just that subfolder's files locally (not the
    whole repo) and retry from there, where `subfolder` is no longer needed."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            load_ref, revision=revision, subfolder=subfolder, trust_remote_code=trust_remote_code
        )
    except OSError:
        if not subfolder:
            raise
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(load_ref, revision=revision, allow_patterns=[f"{subfolder}/*"])
        return AutoTokenizer.from_pretrained(str(Path(local_dir) / subfolder), trust_remote_code=trust_remote_code)


_WORD_START_MARKERS = ("Ġ", "▁")  # GPT-2-style byte-level BPE and SentencePiece, respectively


def token_display_text(tok: "PreTrainedTokenizerBase", t: str | bytes | None) -> str:
    """convert_ids_to_tokens returns each token in the tokenizer's internal vocab representation,
    not display text: raw bytes for tiktoken-based tokenizers (e.g. QWenTokenizer, not always
    valid standalone UTF-8 and never JSON-serializable as-is), and for byte-level BPE tokenizers
    (GPT-2/Pythia/BLOOM) a per-byte visible-codepoint encoding that renders non-ASCII text (e.g.
    Chinese) as mojibake unless reversed. convert_tokens_to_string() undoes both, but called on a
    single token it also treats that token as the start of a decoded string and drops (rather than
    converts to a real space) a leading word-start marker (confirmed on SentencePiece; GPT-2-style
    byte-level BPE already returns a leading space unchanged) — reattach one when the raw token
    asked for it and the conversion ate it, so every layer's word-initial predictions still read as
    "starts a new word" instead of silently looking like mid-word continuations. This is
    tooltip/grid display text, not round-tripped byte-exactly, so a lossy decode is fine.

    None means the id has no vocab entry at all — some checkpoints (e.g. BLOOM) pad the
    embedding/lm_head matrix past the tokenizer's real vocab size for hardware alignment, and an
    undertrained layer can assign a padding id nonzero top-k probability. '?' matches the
    frontend's own fallback (web/src/data.ts) for a vocab id it can't find."""
    if t is None:
        return "?"
    s = t.decode("utf-8", errors="replace") if isinstance(t, bytes) else t
    converted = tok.convert_tokens_to_string([s])
    if s[:1] in _WORD_START_MARKERS and not converted.startswith(" "):
        converted = " " + converted
    return converted
