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

After it finishes: rebuild the web app, and upload <models-root>/<id>/ to your static model host.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from .sources import resolve_sources

# present in a repo checkout; absent when the package was pip-installed
WEB = Path(__file__).resolve().parent.parent / "web"


def run(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", f"lenslapse.{module}", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local directory")
    ap.add_argument("--id", required=True, help="model id used by the app (directory name)")
    ap.add_argument("--label", required=True, help="display name in the model picker")
    ap.add_argument(
        "--steps",
        default="0,1,2,4,8,16,32,64,128,256,512,1000,2000,4000,8000,16000,32000,64000,128000,143000",
        help="hub suites only; local Trainer dirs always use every checkpoint-* they contain",
    )
    ap.add_argument("--final-only", action="store_true")
    ap.add_argument("--models-root", required=True, help="directory that holds <id>/step*/... for the model host")
    ap.add_argument("--skip-export", action="store_true", help="only precompute + register")
    ap.add_argument("--force", action="store_true", help="overwrite an id that is already registered")
    args = ap.parse_args()

    # Without a web/ checkout (pip install), only the ONNX export is possible; the precomputed
    # shards, tokenizer, and models.json registration all target the app's source tree.
    in_repo = WEB.is_dir()
    if not in_repo:
        print("note: no web/ checkout next to the package — exporting ONNX only (no precompute/registration)")

    catalog = None
    if in_repo:
        # refuse to clobber an existing model's shards/tokenizer/registry entry before any work runs
        models_json = WEB / "public" / "data" / "models.json"
        catalog = json.loads(models_json.read_text())
        if any(m["id"] == args.id for m in catalog["models"]) and not args.force:
            sys.exit(f"model id {args.id!r} is already registered in {models_json}; rerun with --force to overwrite")

    sources = resolve_sources(args.model, args.steps, args.final_only)
    print(f"{len(sources)} checkpoint(s): steps {[s.step for s in sources]}")

    common = ["--model", args.model, "--steps", args.steps] + (["--final-only"] if args.final_only else [])
    if not args.skip_export:
        run("export_checkpoints", [*common, "--out", str(Path(args.models_root) / args.id), "--skip-existing"])
    if not in_repo:
        print(f"exported {Path(args.models_root) / args.id}; upload it to your model host to use it in the app")
        return
    run("precompute_lens", [*common, "--out", str(WEB / "public" / "data" / args.id)])

    # tokenizer for the app (served locally; the app never fetches tokenizers remotely)
    tok = AutoTokenizer.from_pretrained(sources[0].load_ref, revision=sources[0].revision)
    tok_dir = WEB / "public" / "tokenizer" / args.id
    tok.save_pretrained(tok_dir)

    # register in models.json. `hf` is the name the app's AutoTokenizer resolves against its
    # local tokenizer/ directory, so it must match the directory we just wrote; `source` keeps
    # the true origin ref for tools that need real weights (the probe server).
    if args.final_only:
        mode = "final"
    elif Path(args.model).is_dir():
        mode = "local"
    else:
        mode = "suite"
    entry = {"id": args.id, "hf": args.id, "label": args.label, "mode": mode, "source": args.model}
    if mode == "suite":
        # record the step grid: consumers with no shard data (e.g. the probe server) need it
        entry["steps"] = sorted({int(s) for s in args.steps.split(",")})
    catalog["models"] = [m for m in catalog["models"] if m["id"] != args.id] + [entry]
    tmp = models_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(catalog, indent=2) + "\n")
    os.replace(tmp, models_json)

    print(f"registered '{args.id}' in models.json; tokenizer at {tok_dir}")
    print(f"next: rebuild the web app and upload {Path(args.models_root) / args.id} to your model host")


if __name__ == "__main__":
    main()
