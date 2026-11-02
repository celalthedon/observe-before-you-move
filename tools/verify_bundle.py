"""Cheap integrity, provenance, relative-link, and compact-evidence checks."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    checksum_path = ROOT / "data/artifact_checksums.json"
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        path = ROOT / name
        assert path.stat().st_size == expected["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name
    manifest = json.loads((ROOT / "data/heldout/manifest.json").read_text(encoding="utf-8"))
    dataset = json.loads((ROOT / "data/dataset/manifest.json").read_text(encoding="utf-8"))
    tree = ast.parse((ROOT / "src/obym/simulation.py").read_text(encoding="utf-8"))
    runner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_postturn_audit")
    assert hashlib.sha256(ast.dump(runner).encode()).hexdigest() == manifest["runner_sha256"] == dataset["runner_sha256"]
    provenance = json.loads((ROOT / "data/workflow_provenance.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / "tools/train_worker_v2.py").read_bytes()).hexdigest() == provenance["training_worker_sha256"]
    for name in ("README.md", "VERIFICATION.md"):
        for target in re.findall(r'\]\(([^)]+)\)', (ROOT / name).read_text(encoding="utf-8")):
            if target.startswith(("http:", "https:", "#")):
                continue
            assert (ROOT / target.split("#")[0]).exists(), (name, target)
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tools").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"))
    assert manifest["minimum_value_s"] == .15
    rows = json.loads((ROOT / "data/heldout/results.json").read_text(encoding="utf-8"))
    assert len(rows) == 560
    assert sum(r["result_origin"] == "new_navigation_run" for r in rows) == 480
    assert sum(r["result_origin"] == "derived_identical_policy" for r in rows) == 80
    assert all("directory" not in r for r in rows)
    print(f"PASS | {len(checksums)} artifact checksums | frozen runner | relative links | Python parse | 480 executed + 80 derived")


if __name__ == "__main__":
    main()
