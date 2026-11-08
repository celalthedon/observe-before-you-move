"""Run the archived OBYM V3 verification and opt-in reproduction workflows."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from .common import Bundle, REPO_ROOT, require, verify_dataset_layout


def paths(command):
    command.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "data/v3",
                         help="Bundle directory containing results/ and pinned/")
    command.add_argument("--source-root", type=Path, help="Pinned src/ and tools/ (default: ARTIFACT_ROOT/pinned)")
    command.add_argument("--output", type=Path, required=True, help="Separate directory for generated outputs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stats = commands.add_parser("stats", help="Regenerate and verify statistics from archived branches and predictions")
    predictions = commands.add_parser("predictions", help="Check frozen CPU predictions and decisions on validation and held-out inputs")
    preflight = commands.add_parser("preflight", help="Reconstruct the test manifest and check validation probes")
    collect_final = commands.add_parser("collect-final", help="Collect deterministic final action branches in bounded batches")
    collect_dataset = commands.add_parser("collect-dataset", help="Recollect the original train/validation first-eligible dataset")
    train_spatial = commands.add_parser("train-spatial", help="Rerun source-grouped training-only selection and ridge fitting")
    train_cnn = commands.add_parser("train-cnn", help="Rerun the original two three-member CNN ensembles")
    for command in (stats, predictions, preflight, collect_final, collect_dataset, train_spatial, train_cnn):
        paths(command)
    stats.add_argument("--deep", action="store_true", help="Also hash every episode trace after extracting the full final trace archive")
    preflight.add_argument("--historical-root", type=Path, action="append", default=[],
                           help="Scan another directory for prior *manifest*.json files (repeatable)")
    preflight.add_argument("--cuda", action="store_true", help="Also compare CUDA and CPU validation predictions")
    for command in (collect_final, collect_dataset, train_spatial, train_cnn):
        command.add_argument("--execute", action="store_true", required=True,
                             help="Explicitly opt in to running the collection or training worker")
    for command in (collect_final, collect_dataset):
        command.add_argument("--batch-limit", type=int, default=1, help="Maximum new cases in this batch (default: 1)")
    collect_final.add_argument("--minutes", type=float, default=5,
                               help="Stop starting episodes after this many minutes (default: 5)")
    for command in (collect_final, train_spatial, train_cnn):
        command.add_argument("--dataset", type=Path, help="Train/validation input directory (default: archived dataset)")
    train_cnn.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def run_worker(name, *args):
    environment = os.environ.copy()
    for key in ("PYTHONOPTIMIZE", "KMP_DUPLICATE_LIB_OK"):
        environment.pop(key, None)
    environment.update(
        MKL_THREADING_LAYER="SEQUENTIAL", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    worker = Path(__file__).with_name(f"{name}.py")
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(worker), *map(str, args)],
                   check=True, env=environment)


def resumable(output, identity):
    require(
        not output.exists() or output.is_dir() and (not any(output.iterdir()) or (output / identity).is_file()),
        f"Use an empty directory or resume an existing {identity} run: {output}",
    )


def main():
    sys.dont_write_bytecode = True
    args = parse_args()
    root = args.artifact_root.resolve()
    source = (args.source_root or root / "pinned").resolve()
    bundle = Bundle(root, source)
    output = bundle.output_path(args.output)
    if args.command == "stats":
        from .report import regenerate
        regenerate(bundle, output, deep=args.deep)
        return
    if args.command == "predictions":
        from .predict import replay
        replay(bundle, output)
        return
    if args.command == "preflight":
        from .preflight import verify_preflight
        default_history = [REPO_ROOT / "data" / name for name in ("dataset", "heldout")]
        roots = [path for path in default_history if path.is_dir()] + args.historical_root
        verify_preflight(bundle, output, roots, cuda=args.cuda)
        return

    bundle.protocol(verify_source=args.command != "train-spatial")
    dataset = (getattr(args, "dataset", None) or bundle.dataset).resolve()
    require(
        not output.is_relative_to(dataset) and not dataset.is_relative_to(output),
        f"Use an output directory separate from the dataset: {output}",
    )
    manifest = bundle.frozen / "dataset_manifest.json"
    if args.command != "collect-dataset":
        verify_dataset_layout(dataset, bundle.frozen)
    if args.command == "collect-final":
        require(1 <= args.batch_limit <= 80 and 0 < args.minutes < float("inf"), "Use a batch of 1–80 and a positive finite time limit")
        resumable(output, "collection_manifest.json")
        run_worker("collect_final", source, bundle.frozen, dataset, output, args.batch_limit, args.minutes)
    elif args.command == "collect-dataset":
        require(1 <= args.batch_limit <= 120, "Use a batch of 1–120")
        resumable(output, "protocol.json")
        run_worker("collect_dataset", source, manifest, output, args.batch_limit)
    elif args.command == "train-spatial":
        run_worker("train_spatial", dataset, bundle.training, output)
    else:
        run_worker("train_cnn", source, manifest, dataset, output, args.device)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
