"""Resolve a model source into (load_ref, revision, step) tuples for export/precompute.

Three source shapes are supported:
  1. Hub suite   : --model EleutherAI/pythia-70m --steps 0,1000,...   -> revisions step{N}
  2. Hub single  : --model gpt2 --final-only                          -> revision main, step 0
  3. Local dir   : --model /path/to/run --local-checkpoints           -> checkpoint-*/ subdirs
                   (Hugging Face Trainer layout; the number suffix is the training step), or a
                   plain single-model directory when no checkpoint-* subdirs exist.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointSource:
    load_ref: str  # HF id or local path passed to from_pretrained
    revision: str | None  # HF revision, None for local paths
    step: int  # slider position in the app

    @property
    def name(self) -> str:
        return f"step{self.step}"


def resolve_sources(
    model: str, steps_arg: str, final_only: bool, revision_template: str = "step{}"
) -> list[CheckpointSource]:
    path = Path(model)
    if path.exists() and path.is_dir():
        ckpts = [c for c in path.glob("checkpoint-*") if _ckpt_step(c) is not None]
        ckpts.sort(key=_ckpt_step)
        if ckpts:
            if final_only:
                raise SystemExit(
                    f"{model} is a Trainer directory with checkpoint-* subdirs; --final-only is ambiguous — "
                    "pass the specific checkpoint directory instead"
                )
            return [CheckpointSource(str(c), None, _ckpt_step(c)) for c in ckpts]
        # plain local model directory: single checkpoint at step 0
        return [CheckpointSource(str(path), None, 0)]
    if final_only:
        return [CheckpointSource(model, "main", 0)]
    steps = sorted({int(s) for s in steps_arg.split(",")})
    return [CheckpointSource(model, revision_template.format(s), s) for s in steps]


def _ckpt_step(p: Path) -> int | None:
    m = re.fullmatch(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else None
