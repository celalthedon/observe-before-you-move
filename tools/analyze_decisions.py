"""Recompute the 80 decision pairs from bundled archived decision logs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np

TIE_TOLERANCE_S = 1e-6


def analyze(root: Path, output: Path):
    heldout = root / "data/heldout"
    with gzip.open(heldout / "heldout_decisions.json.gz", "rt", encoding="utf-8") as f:
        traces = json.load(f)
    manifest = json.loads((heldout / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((heldout / "results.json").read_text(encoding="utf-8"))
    index = {(r["scene"], r["method"]): r for r in results}
    rows = []
    for case in sorted(manifest["cases"], key=lambda c: c["name"]):
        scene = case["name"]
        choices = {}
        for method in ("ValueCNN", "FixedMinus45"):
            trace = traces[scene][method]
            assert [r["decision_index"] for r in trace] == list(range(len(trace)))
            looks = [r for r in trace if r["action"] == "OBSERVE"]
            assert len(looks) <= 1
            choices[method] = looks[0] if looks else None
        cc, fc = choices["ValueCNN"], choices["FixedMinus45"]
        same = (cc is None and fc is None) or (cc is not None and fc is not None and (cc["decision_index"], cc["selected_offset_deg"]) == (fc["decision_index"], fc["selected_offset_deg"]))
        cnn, fixed = index[scene, "ValueCNN"], index[scene, "FixedMinus45"]
        def cost(r):
            return r["sim_time_s"] if r["outcome"] == "GOAL" and r["collision_attempts"] == 0 else manifest["failure_cost_s"]
        gain = cost(fixed) - cost(cnn)
        rows.append({"scene": scene, "source_group": case["source_spec"]["name"], "decision_relation": "matching" if same else "different",
            "cnn_action": "OBSERVE" if cc else "SKIP", "cnn_decision_index": "" if cc is None else cc["decision_index"], "cnn_angle_deg": "" if cc is None else cc["selected_offset_deg"],
            "fixed_action": "OBSERVE" if fc else "SKIP", "fixed_decision_index": "" if fc is None else fc["decision_index"], "fixed_angle_deg": "" if fc is None else fc["selected_offset_deg"],
            "cnn_time_s": cost(cnn), "fixed_time_s": cost(fixed), "fixed_minus_cnn_s": gain,
            "outcome": "tie" if abs(gain) <= TIE_TOLERANCE_S else "cnn_win" if gain > 0 else "cnn_loss"})
    differing = [r for r in rows if r["decision_relation"] == "different"]
    mean = float(np.mean([r["fixed_minus_cnn_s"] for r in rows]))
    comparisons = json.loads((heldout / "paired_comparisons.json").read_text(encoding="utf-8"))
    assert len(rows) == 80 and abs(mean - comparisons["FixedMinus45"]["mean_gain_s"]) < 1e-12
    summary = {"pairs": len(rows), "matching_decisions": len(rows) - len(differing), "different_decision_count": len(differing),
        "tie_tolerance_s": TIE_TOLERANCE_S, "gain_sign": "fixed_minus_cnn_s; positive favors ValueCNN",
        "different_decisions": {"count": len(differing), "cnn_wins": sum(r["outcome"] == "cnn_win" for r in differing), "ties": sum(r["outcome"] == "tie" for r in differing), "cnn_losses": sum(r["outcome"] == "cnn_loss" for r in differing), "mean_paired_gain_s": float(np.mean([r["fixed_minus_cnn_s"] for r in differing]))},
        "all_80_mean_paired_gain_s": mean, "published_all_80_mean_paired_gain_s": comparisons["FixedMinus45"]["mean_gain_s"]}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cnn_vs_fixedminus45_decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (output / "decision_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("PASS | archived decisions |", json.dumps(summary))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/generated"))
    args = parser.parse_args(); analyze(args.root.resolve(), args.output.resolve())
