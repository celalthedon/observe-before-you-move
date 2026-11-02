"""Launch the archived V2 training worker in a fresh Windows-safe process."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "results/trained_v2")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env.update(MKL_THREADING_LAYER="SEQUENTIAL", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "-u", str(ROOT / "tools/train_worker_v2.py"), str(args.dataset.resolve()), str(output)], env=env, check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
