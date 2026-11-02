"""Results-only reproduction of the held-out OBYM table and main figure.

No simulator, GPU, model server, or notebook kernel is used.  It recomputes
the public summary from the 560 portable result entries and writes a figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_artifacts import METHODS, build_main_figure, make_route_figure, capped
from tools.analyze_decisions import analyze
from tools.build_replay import build as build_replay


def main(root: Path, output: Path) -> None:
    heldout = root / "data/heldout"
    manifest = json.loads((heldout / "manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((heldout / "results.json").read_text(encoding="utf-8"))
    comparisons = json.loads((heldout / "paired_comparisons.json").read_text(encoding="utf-8"))
    cases = {c["name"]: c for c in manifest["cases"]}
    index = {(r["scene"], r["method"]): r for r in rows}
    assert len(cases) == 80 and len(rows) == 560 and len(index) == 560
    assert len({c["source_spec"]["name"] for c in cases.values()}) == 20
    assert set(manifest["methods"]) == set(METHODS)
    summary = {"test_sha": "fa34bcbcb4a4", "checkpoint_sha256": manifest["checkpoint_sha256"], "source_scenes": 20, "variants": 80, "methods": {}, "cnn_paired_comparisons": comparisons}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        safe = sum(r["outcome"] == "GOAL" and r["collision_attempts"] == 0 for r in subset)
        assert len(subset) == 80 and safe == 80
        summary["methods"][method] = {"episodes": 80, "safe_success": safe, "mean_capped_time_s": float(np.mean([capped(r, manifest["failure_cost_s"]) for r in subset])), "completed_views": int(sum(r["completed_views"] for r in subset)), "mean_prediction_wall_s": float(np.mean([r.get("prediction_wall_s", 0.0) for r in subset]))}
    sources = [s["name"] for s in manifest["test_specs"]]
    draws = np.random.default_rng(manifest["bootstrap_seed"]).integers(0, len(sources), size=(10000, len(sources)))
    for other, report in comparisons.items():
        gain = float(np.mean([capped(index[(scene, other)], manifest["failure_cost_s"]) - capped(index[(scene, "ValueCNN")], manifest["failure_cost_s"]) for scene in cases]))
        assert abs(gain - report["mean_gain_s"]) < 1e-8, other
        group_gains = np.asarray([np.mean([capped(index[(c["name"], other)], manifest["failure_cost_s"]) - capped(index[(c["name"], "ValueCNN")], manifest["failure_cost_s"]) for c in cases.values() if c["source_spec"]["name"] == source]) for source in sources])
        interval = np.quantile(group_gains[draws].mean(axis=1), [.025, .975])
        assert np.allclose(interval, report["ci95_s"], atol=1e-12, rtol=0), other
    # Strong guards against unintentionally switching to a different evidence bundle.
    assert abs(summary["methods"]["NoView"]["mean_capped_time_s"] - 41.51469649732793) < 1e-9
    assert abs(summary["methods"]["FixedMinus45"]["mean_capped_time_s"] - 39.63088629668157) < 1e-9
    assert abs(summary["methods"]["ValueCNN"]["mean_capped_time_s"] - 39.73426517439013) < 1e-9
    assert summary["methods"]["ValueCNN"]["completed_views"] == 67
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    build_main_figure(summary, comparisons, output / "heldout_comparison.png")
    with (output / "policy_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["policy", "mean_simulated_time_s", "optional_looks", "safe_success"])
        for method, item in summary["methods"].items():
            writer.writerow([method, item["mean_capped_time_s"], item["completed_views"], item["safe_success"]])
    analyze(root, output)
    demos = json.loads((root / "data/traces/selected_examples.json").read_text(encoding="utf-8"))["demos"]
    for demo in demos:
        for key in ("cnn_motion", "fixed_motion"):
            demo[key] = str(root / demo[key])
    make_route_figure(cases, demos, output / "paired_route_examples.png")
    build_replay(root, output / "replay.html")
    print("PASS | 80 variants | 20 source groups | 560 entries (480 executed + 80 derived HeadingRule)")
    for method in METHODS:
        item = summary["methods"][method]
        print(f"{method:15s} {item['mean_capped_time_s']:.6f} s | success={item['safe_success']}/80 | looks={item['completed_views']}")
    print("wrote", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--output", type=Path, default=Path("results/generated"))
    args = parser.parse_args(); main(args.root.resolve(), args.output.resolve())
