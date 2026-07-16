"""One-command model onboarding for LensLapse.

Runs the full pipeline for a new model — export ONNX pairs, precompute lens shards, install the
tokenizer for the web app, and register the model in web/public/data/models.json — so that adding
a model is a data change, not a code change.

Supported sources (see sources.py):
  Hub suite    : python add_model.py --model EleutherAI/pythia-31m --id pythia-31m --label "Pythia 31M" \
                     --steps 0,512,8000,143000 --models-root /path/to/models
  Hub single   : python add_model.py --model gpt2 --id gpt2 --label "GPT-2 124M" --final-only ...
  Local run dir: python add_model.py --model /path/to/trainer_output --id my-run --label "My run" ...
                 (Hugging Face Trainer layout: checkpoint-<step>/ subdirectories)
  Hub subfolder: python add_model.py --model m-a-p/neo_scalinglaw_250M --id mapneo-250m --label "MAP-Neo" \
                     --subfolder-map "16780:hf_ckpt/16.78B,33550:hf_ckpt/33.55B" --models-root /path/to/models
                 (checkpoints nested as subfolders within one revision; overrides --steps)

After it finishes: rebuild the web app, and upload <models-root>/<id>/ to your static model host.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
from pydantic import BaseModel, field_validator

from lenslapse.logging_utils import configure_cli_logging
from lenslapse.sources import (
    DEFAULT_STEPS_CSV,
    Mode,
    coerce_fire_csv_arg,
    load_tokenizer,
    resolve_all_sources,
    resolve_tokenizer_ref,
)

logger = logging.getLogger(__name__)

# present in a repo checkout; absent when the package was pip-installed
WEB = Path(__file__).resolve().parent.parent.parent / "web"


def run(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", f"lenslapse.{module}", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


class AddModelConfig(BaseModel):
    """Validated arguments for `add_model`; see `main`'s docstring for what each means."""

    model: str
    id: str
    label: str
    models_root: Path
    steps: str = DEFAULT_STEPS_CSV
    final_only: bool = False
    skip_export: bool = False
    force: bool = False
    subfolder_map: str | None = None
    prompts_file: str | None = None
    revision_template: str = "step{}"
    tokenizer_ref: str | None = None

    _coerce_steps = field_validator("steps", mode="before")(coerce_fire_csv_arg)


def main(
    model: str,
    id: str,
    label: str,
    models_root: str,
    steps: str = DEFAULT_STEPS_CSV,
    final_only: bool = False,
    skip_export: bool = False,
    force: bool = False,
    subfolder_map: str | None = None,
    prompts_file: str | None = None,
    revision_template: str = "step{}",
    tokenizer_ref: str | None = None,
) -> None:
    """Onboard a new model: export ONNX pairs, precompute lens shards, install the tokenizer, and
    register the model in web/public/data/models.json.

    Args:
        model: HF id or local directory.
        id: model id used by the app (directory name).
        label: display name in the model picker.
        models_root: directory that holds <id>/step*/... for the model host.
        steps: hub suites only; local Trainer dirs always use every checkpoint-* they contain.
        final_only: single checkpoint (revision "main") instead of a step suite.
        skip_export: only precompute + register.
        force: overwrite an id that is already registered.
        subfolder_map: "step:path,step:path,..." for repos that nest checkpoints as subfolders of
            one revision instead of using git revisions per checkpoint; overrides `steps`.
        prompts_file: JSON curated-prompt list; see precompute_lens.py.
        revision_template: revision naming for hub suites, e.g. "global_step{}" for
            bigscience/bloom-*-intermediate.
        tokenizer_ref: load the tokenizer from a different ref than the checkpoint weights; see
            export_checkpoints.py.
    """
    cfg = AddModelConfig(
        model=model,
        id=id,
        label=label,
        models_root=models_root,  # type: ignore[arg-type]  # pydantic coerces str -> Path
        steps=steps,
        final_only=final_only,
        skip_export=skip_export,
        force=force,
        subfolder_map=subfolder_map,
        prompts_file=prompts_file,
        revision_template=revision_template,
        tokenizer_ref=tokenizer_ref,
    )

    # Without a web/ checkout (pip install), only the ONNX export is possible; the precomputed
    # shards, tokenizer, and models.json registration all target the app's source tree.
    in_repo = WEB.is_dir()
    if not in_repo:
        logger.info("no web/ checkout next to the package — exporting ONNX only (no precompute/registration)")

    catalog = None
    if in_repo:
        # refuse to clobber an existing model's shards/tokenizer/registry entry before any work runs
        models_json = WEB / "public" / "data" / "models.json"
        catalog = json.loads(models_json.read_text())
        if any(m["id"] == cfg.id for m in catalog["models"]) and not cfg.force:
            sys.exit(f"model id {cfg.id!r} is already registered in {models_json}; rerun with --force to overwrite")

    sources = resolve_all_sources(cfg.model, cfg.steps, cfg.final_only, cfg.subfolder_map, cfg.revision_template)
    logger.info("%d checkpoint(s): steps %s", len(sources), [s.step for s in sources])

    common = ["--model", cfg.model, "--steps", cfg.steps] + (["--final-only"] if cfg.final_only else [])
    if cfg.subfolder_map:
        common += ["--subfolder-map", cfg.subfolder_map]
    if cfg.revision_template != "step{}":
        common += ["--revision-template", cfg.revision_template]
    if cfg.tokenizer_ref:
        common += ["--tokenizer-ref", cfg.tokenizer_ref]
    if not cfg.skip_export:
        run("export_checkpoints", [*common, "--out", str(cfg.models_root / cfg.id), "--skip-existing"])
    if not in_repo:
        logger.info("exported %s; upload it to your model host to use it in the app", cfg.models_root / cfg.id)
        return
    precompute_args = [*common, "--out", str(WEB / "public" / "data" / cfg.id)]
    if cfg.prompts_file:
        precompute_args += ["--prompts-file", cfg.prompts_file]
    run("precompute_lens", precompute_args)

    # tokenizer for the app (served locally; the app never fetches tokenizers remotely)
    tok_load_ref, tok_rev, tok_subfolder = resolve_tokenizer_ref(cfg.tokenizer_ref, sources[0])
    tok = load_tokenizer(tok_load_ref, tok_rev, tok_subfolder)
    tok_dir = WEB / "public" / "tokenizer" / cfg.id
    tok.save_pretrained(tok_dir)

    # register in models.json. `hf` is the name the app's AutoTokenizer resolves against its
    # local tokenizer/ directory, so it must match the directory we just wrote; `source` keeps
    # the true origin ref for tools that need real weights (the probe server).
    mode: Mode
    if cfg.final_only:
        mode = "final"
    elif Path(cfg.model).is_dir():
        mode = "local"
    else:
        mode = "suite"
    assert catalog is not None  # in_repo is True on this path, so the catalog was loaded above
    entry: dict[str, Any] = {"id": cfg.id, "hf": cfg.id, "label": cfg.label, "mode": mode, "source": cfg.model}
    if mode == "suite":
        # record the step grid: consumers with no shard data (e.g. the probe server) need it.
        # Derived from the resolved sources, not cfg.steps directly — a --subfolder-map suite's
        # real step values live only on `sources`, not in the (irrelevant, default-valued) steps.
        entry["steps"] = sorted({s.step for s in sources})
        # the live-probe server (server.py's source_for()) needs these too, to resolve the same
        # hub revisions/subfolders this export used (e.g. BLOOM's global_step{N}, MAP-Neo's
        # hf_ckpt/*), not just the offline pipeline here — final/local sources have neither
        # concept.
        if cfg.revision_template != "step{}":
            entry["revision_template"] = cfg.revision_template
        if cfg.subfolder_map:
            entry["subfolder_map"] = cfg.subfolder_map
    if cfg.tokenizer_ref:
        # likewise: the live-probe server's tokenizer loading (server.py's load()/tokenize())
        # needs this whenever the checkpoint's own tokenizer isn't the one to serve.
        entry["tokenizer_ref"] = cfg.tokenizer_ref
    catalog["models"] = [m for m in catalog["models"] if m["id"] != cfg.id] + [entry]
    tmp = models_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2) + "\n")
    tmp.replace(models_json)

    logger.info("registered '%s' in models.json; tokenizer at %s", cfg.id, tok_dir)
    logger.info("next: rebuild the web app and upload %s to your model host", cfg.models_root / cfg.id)


if __name__ == "__main__":
    configure_cli_logging()
    fire.Fire(main)
