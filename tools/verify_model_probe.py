"""Verify extracted CPU inference against the archived V2 probe."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obym.inference import verify_probe

if __name__ == "__main__":
    error = verify_probe(ROOT / "data/model/ensemble.pt", ROOT / "data/model/inference_probe.npz")
    print(f"PASS | V2 CPU probe | max absolute error {error:.8f} s")
