"""Create the public, relative-path evidence bundle from an archived OBYM run.

This is deliberately an export tool: it reads the original experiments and
never mutates them.  The committed bundle was made from the paths in the
project README.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


METHODS = ("NoView", "FixedMinus45", "FixedPlus45", "HeadingRule", "PoseOnly", "ValueLinear", "ValueCNN")
TIE_TOLERANCE_S = 1e-6


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def capped(row: dict, failure_cost: float) -> float:
    return float(row["sim_time_s"]) if row["outcome"] == "GOAL" and row["collision_attempts"] == 0 else failure_cost


def observation(trace: list[dict]) -> dict | None:
    choices = [row for row in trace if row.get("action") == "OBSERVE"]
    if len(choices) > 1:
        raise ValueError("the held-out protocol allows no more than one optional look")
    return choices[0] if choices else None


def build_main_figure(summary: dict, comparisons: dict, output: Path) -> None:
    methods = list(summary["methods"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    means = [summary["methods"][method]["mean_capped_time_s"] for method in methods]
    positions = np.arange(len(methods))
    axes[0].scatter(means, positions, c=["#2563eb" if x == "ValueCNN" else "#64748b" for x in methods], s=65, zorder=3)
    for pos, value in zip(positions, means):
        axes[0].annotate(f"{value:.3f}", (value, pos), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    axes[0].set(yticks=positions, yticklabels=methods, xlabel="Mean simulated time (s); failures capped at 180 s", title="Held-out navigation: 80 variants")
    axes[0].invert_yaxis(); axes[0].grid(axis="x", alpha=.2)
    axes[0].set_xlim(min(means) - .2, max(means) + .6)
    others = [m for m in methods if m != "ValueCNN"]
    for pos, other in enumerate(others):
        value = comparisons[other]["mean_gain_s"]; low, high = comparisons[other]["ci95_s"]
        color = "#2563eb" if low > 0 else "#64748b"
        axes[1].plot([low, high], [pos, pos], color=color, linewidth=2)
        axes[1].scatter(value, pos, color=color, s=45, zorder=3)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set(yticks=np.arange(len(others)), yticklabels=others, xlabel="Time saved by CNN (s); positive favors CNN", title="Paired gain: 95% source bootstrap intervals")
    axes[1].invert_yaxis(); axes[1].grid(axis="x", alpha=.2)
    fig.suptitle("OBYM | 20 held-out source scenes", fontsize=15)
    fig.tight_layout(); fig.savefig(output, dpi=200, bbox_inches="tight"); plt.close(fig)


def route_panel(ax, case: dict, cnn_motion: list[dict], fixed_motion: list[dict], cnn_choice: dict | None, fixed_choice: dict | None, cnn_time: float, fixed_time: float) -> None:
    for size, position in case["geometry"]:
        ax.add_patch(Rectangle((position[0] - size[0] / 2, position[1] - size[1] / 2), size[0], size[1], color="#475569", alpha=.8))
    for motion, color, label, choice, time in ((cnn_motion, "#2563eb", "ValueCNN", cnn_choice, cnn_time), (fixed_motion, "#d97706", "FixedMinus45", fixed_choice, fixed_time)):
        xy = np.asarray([(r["x"], r["y"]) for r in motion])
        action = "skip optional look" if choice is None else f"look {choice['selected_offset_deg']:+d}°"
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2, linestyle="--" if label == "FixedMinus45" else "-", label=f"{label}: {time:.3f} s ({action})")
        if choice:
            x, y = choice["xy"]
            ax.scatter(x, y, s=70, marker="*", color=color, edgecolor="white", linewidth=.7, zorder=4)
            ax.annotate(f"{label} {choice['selected_offset_deg']:+d}°", (x, y), xytext=(8, 22 if label == "ValueCNN" else -24), textcoords="offset points", color=color, fontsize=8, arrowprops={"arrowstyle": "-", "color": color})
    start, goal = case["start_xy"], case["goal_xy"]
    ax.scatter(*start, marker="o", color="#16a34a", s=40, label="start")
    ax.scatter(*goal, marker="X", color="#dc2626", s=55, label="goal")
    ax.set_aspect("equal"); ax.set(xlim=(-4.5, 4.5), ylim=(-3.5, 3.5), xlabel="x (m)", ylabel="y (m)")
    ax.grid(alpha=.18); ax.legend(fontsize=7, loc="best")


def make_route_figure(cases: dict, demos: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, layout="constrained")
    for ax, demo in zip(axes, demos):
        route_panel(ax, cases[demo["scene"]], read_json(Path(demo["cnn_motion"])), read_json(Path(demo["fixed_motion"])), demo["cnn_choice"], demo["fixed_choice"], demo["cnn_time_s"], demo["fixed_time_s"])
        title = "CNN gain" if demo["kind"] == "cnn_gain" else "CNN loss"
        ax.set_title(f"{title}: {demo['scene']}\nFixed − CNN = {demo['gain_s']:+.3f} s", fontsize=11)
    fig.suptitle("Actual held-out trajectories | median-within-sign examples", fontsize=14)
    fig.savefig(output, dpi=200, bbox_inches="tight"); plt.close(fig)


def main(source: Path, destination: Path) -> None:
    source = source.resolve(); destination = destination.resolve()
    heldout = source / "results/ml/heldout_v2_navigation/fa34bcbcb4a4"
    dataset = source / "results/ml/task_value_v2/97e8ab583a96"
    training = source / "results/ml/task_value_v2_training/20261101_231250_751344"
    data = destination / "data"; figures = destination / "figures"; reports = data / "heldout/reports"; traces = data / "traces"
    for folder in (reports, figures, traces, data / "dataset", data / "model"): folder.mkdir(parents=True, exist_ok=True)
    manifest = read_json(heldout / "manifest.json"); rows = read_json(heldout / "results.json"); comparisons = read_json(heldout / "paired_comparisons.json")
    index = {(r["scene"], r["method"]): r for r in rows}; cases = {c["name"]: c for c in manifest["cases"]}
    if len(cases) != 80 or len(rows) != 560 or set(manifest["methods"]) != set(METHODS): raise ValueError("unexpected held-out bundle")
    # Export all aggregate evidence with the machine-specific episode paths removed.
    portable_rows = [{k: v for k, v in row.items() if k != "directory"} for row in rows]
    for path, value in ((data / "heldout/manifest.json", manifest), (data / "heldout/results.json", portable_rows), (data / "heldout/paired_comparisons.json", comparisons)):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    for name in ("train.npz", "validation.npz", "manifest.json"):
        shutil.copy2(dataset / name, data / "dataset" / name)
    for name in ("ensemble.pt", "inference_probe.npz", "training_config.json", "validation_metrics.json", "validation_predictions.npz"):
        shutil.copy2(training / name, data / "model" / name)
    metrics_columns = ("scene", "source_group", "method", "outcome", "sim_time_s", "distance_m", "completed_views", "collision_attempts", "prediction_wall_s", "result_origin")
    with (reports / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_columns); writer.writeheader(); writer.writerows({k: r.get(k) for k in metrics_columns} for r in rows)
    summary = {"test_sha": heldout.name, "checkpoint_sha256": manifest["checkpoint_sha256"], "source_scenes": 20, "variants": 80, "methods": {}, "cnn_paired_comparisons": comparisons}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {"episodes": len(subset), "safe_success": sum(r["outcome"] == "GOAL" and r["collision_attempts"] == 0 for r in subset), "mean_capped_time_s": float(np.mean([capped(r, manifest["failure_cost_s"]) for r in subset])), "completed_views": sum(r["completed_views"] for r in subset), "mean_prediction_wall_s": float(np.mean([r.get("prediction_wall_s", 0) for r in subset]))}
    (reports / "test_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    build_main_figure(summary, comparisons, figures / "heldout_comparison.png")
    decision_rows = []; differences = []; archived_decisions = {}
    for scene, case in sorted(cases.items()):
        cnn, fixed = index[scene, "ValueCNN"], index[scene, "FixedMinus45"]
        cnn_trace = read_json(Path(cnn["directory"]) / "heldout_decisions.json"); fixed_trace = read_json(Path(fixed["directory"]) / "heldout_decisions.json")
        archived_decisions[scene] = {"ValueCNN": cnn_trace, "FixedMinus45": fixed_trace}
        cc, fc = observation(cnn_trace), observation(fixed_trace)
        same = (cc is None and fc is None) or (cc is not None and fc is not None and cc["decision_index"] == fc["decision_index"] and cc.get("selected_offset_deg") == fc.get("selected_offset_deg"))
        gain = capped(fixed, manifest["failure_cost_s"]) - capped(cnn, manifest["failure_cost_s"])
        row = {"scene": scene, "source_group": case["source_spec"]["name"], "decision_relation": "matching" if same else "different", "cnn_action": "OBSERVE" if cc else "SKIP", "cnn_decision_index": "" if not cc else cc["decision_index"], "cnn_angle_deg": "" if not cc else cc.get("selected_offset_deg", ""), "fixed_action": "OBSERVE" if fc else "SKIP", "fixed_decision_index": "" if not fc else fc["decision_index"], "fixed_angle_deg": "" if not fc else fc.get("selected_offset_deg", ""), "cnn_time_s": capped(cnn, manifest["failure_cost_s"]), "fixed_time_s": capped(fixed, manifest["failure_cost_s"]), "fixed_minus_cnn_s": gain, "outcome": "tie" if abs(gain) <= TIE_TOLERANCE_S else ("cnn_win" if gain > 0 else "cnn_loss")}
        decision_rows.append(row)
        if not same: differences.append((row, cnn, fixed, cc, fc))
    with gzip.GzipFile(filename=str(data / "heldout/heldout_decisions.json.gz"), mode="wb", mtime=0) as f:
        f.write(json.dumps(archived_decisions, separators=(",", ":")).encode("utf-8"))
    columns = tuple(decision_rows[0])
    with (reports / "cnn_vs_fixedminus45_decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(decision_rows)
    wins = sum(x[0]["outcome"] == "cnn_win" for x in differences); ties = sum(x[0]["outcome"] == "tie" for x in differences); losses = sum(x[0]["outcome"] == "cnn_loss" for x in differences)
    decision_summary = {
        "pairs": len(decision_rows),
        "matching_decisions": len(decision_rows) - len(differences),
        "different_decision_count": len(differences),
        "tie_tolerance_s": TIE_TOLERANCE_S,
        "gain_sign": "fixed_minus_cnn_s; positive favors ValueCNN",
        "different_decisions": {
            "count": len(differences), "cnn_wins": wins, "ties": ties,
            "cnn_losses": losses,
            "mean_paired_gain_s": float(np.mean([x[0]["fixed_minus_cnn_s"] for x in differences])),
        },
        "all_80_mean_paired_gain_s": float(np.mean([r["fixed_minus_cnn_s"] for r in decision_rows])),
        "published_all_80_mean_paired_gain_s": comparisons["FixedMinus45"]["mean_gain_s"],
    }
    (reports / "decision_analysis_summary.json").write_text(json.dumps(decision_summary, indent=2), encoding="utf-8")
    # Median-within-sign selection rule, scene-name tie break.
    demos = []
    for kind, signed in (("cnn_gain", [x for x in differences if x[0]["fixed_minus_cnn_s"] > TIE_TOLERANCE_S]), ("cnn_loss", [x for x in differences if x[0]["fixed_minus_cnn_s"] < -TIE_TOLERANCE_S])):
        median = float(np.median([x[0]["fixed_minus_cnn_s"] for x in signed])); chosen = sorted(signed, key=lambda x: (abs(x[0]["fixed_minus_cnn_s"] - median), x[0]["scene"]))[0]
        row, cnn, fixed, cc, fc = chosen; stem = kind
        cnn_motion = traces / f"{stem}_valuecnn_motion.json"; fixed_motion = traces / f"{stem}_fixedminus45_motion.json"
        shutil.copy2(Path(cnn["directory"]) / "motion.json", cnn_motion); shutil.copy2(Path(fixed["directory"]) / "motion.json", fixed_motion)
        demos.append({"kind": kind, "scene": row["scene"], "gain_s": row["fixed_minus_cnn_s"], "group_median_gain_s": median, "cnn_time_s": row["cnn_time_s"], "fixed_time_s": row["fixed_time_s"], "cnn_choice": cc, "fixed_choice": fc, "cnn_motion": str(cnn_motion), "fixed_motion": str(fixed_motion)})
    make_route_figure(cases, demos, figures / "paired_route_examples.png")
    for demo in demos: demo["cnn_motion"] = Path(demo["cnn_motion"]).relative_to(destination).as_posix(); demo["fixed_motion"] = Path(demo["fixed_motion"]).relative_to(destination).as_posix()
    (traces / "selected_examples.json").write_text(json.dumps({"selection_rule": "among differing decisions, closest to median gain within each strict sign group; scene-name tie break", "demos": demos}, indent=2), encoding="utf-8")
    print(f"Exported {len(rows)} results, {len(decision_rows)} pairs, {len(demos)} route demos to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); main(args.source, args.destination)
