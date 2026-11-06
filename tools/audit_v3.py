"""Audit V2 input ambiguity and compare controllers on development scenes."""

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ("NoView", "FixedMinus45", "FixedPlus45")


def audit_inputs(dataset, cases):
    candidates = defaultdict(list)
    decisions = defaultdict(list)
    positive_labels = {}
    for split in ("train", "validation"):
        samples = defaultdict(dict)
        with np.load(dataset / f"{split}.npz", allow_pickle=False) as data:
            images, poses = data["image"], data["pose"]
            scene_names, states, offsets = data["scene"], data["state_index"], data["offset_deg"]
            labels = data["target_time_saving_s"]
            positive = labels > 1e-6
            initial = states == 0
            positive_labels[split] = {
                "initial": int(np.sum(positive & initial)),
                "later": int(np.sum(positive & ~initial)),
            }
            for i in range(len(initial)):
                scene = str(scene_names[i])
                state = int(states[i])
                offset = int(offsets[i])
                digest = hashlib.sha256(
                    images[i].tobytes() + poses[i].tobytes()
                ).hexdigest()
                gain = float(labels[i])
                samples[scene, state][offset] = (digest, gain)
                candidates[digest].append({
                    "scene": scene, "state": state, "offset_deg": offset, "gain_s": gain,
                })
        for (scene, state), angles in samples.items():
            minus, plus = angles[-45], angles[45]
            gains = [0.0, minus[1], plus[1]]
            case = cases[scene]
            goal_delta = np.subtract(case["goal_xy"], case["start_xy"])
            decisions[minus[0], plus[0]].append({
                "scene": scene,
                "state": state,
                "initial_goal_delta": goal_delta.tolist() if state == 0 else None,
                "gains_s": gains,
                "best_action": ACTIONS[int(np.argmax(gains))],
            })
    return {
        "positive_labels": positive_labels,
        "candidate_conflicts": [
            rows for rows in candidates.values() if np.ptp([row["gain_s"] for row in rows]) > 1e-4
        ],
        "decision_conflicts": [
            rows for rows in decisions.values()
            if np.ptp([row["gains_s"] for row in rows], axis=0).max() > 1e-4
        ],
    }


def verify_initial_states(manifest, cases, conflicts):
    from evaluate import run_case, selectors, sim

    class InitialStateCaptured(Exception):
        pass

    def capture_initial_state(mode, snapshots, trace, capture):
        selector = original_selector(mode, snapshots, trace, capture)

        def capture_and_stop(*args):
            selector(*args)
            raise InitialStateCaptured

        return capture_and_stop

    selectors.ns, selectors.test_config = vars(sim), manifest
    original_selector = selectors.test_selector
    selectors.test_selector = capture_initial_state
    comparisons = []
    sim.p.connect(sim.p.DIRECT)
    try:
        for rows in conflicts:
            if any(row["state"] != 0 for row in rows):
                continue
            snapshots = []
            for row in rows:
                try:
                    run_case(cases[row["scene"]], "NoView", snapshots)
                except InitialStateCaptured:
                    pass
                else:
                    raise RuntimeError(f"No initial decision in {row['scene']}")
            reference = snapshots[0]
            for row, state in zip(rows[1:], snapshots[1:]):
                reference_case, case = cases[rows[0]["scene"]], cases[row["scene"]]
                comparisons.append({
                    "reference_scene": reference_case["name"],
                    "scene": case["name"],
                    "different_grid_cells": int(np.count_nonzero(reference["grid"] != state["grid"])),
                    "same_path": reference["path"] == state["path"],
                    "same_goal": reference_case["goal_xy"] == case["goal_xy"],
                    "same_geometry": reference_case["geometry"] == case["geometry"],
                })
    finally:
        selectors.test_selector = original_selector
        sim.p.disconnect()
    return comparisons


def audit_controller(manifest, case, output):
    from evaluate import run_case, selectors, sim
    from obym.initial_centering_audit import make_initial_centering_audit

    source = Path(sim.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    runner = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_postturn_audit"
    )
    assert hashlib.sha256(ast.dump(runner).encode()).hexdigest() == manifest["runner_sha256"]
    centered = make_initial_centering_audit(runner)
    centered_tree = ast.fix_missing_locations(ast.Module(body=[centered], type_ignores=[]))
    exec(compile(centered_tree, sim.__file__, "exec"), vars(sim))
    selectors.ns, selectors.test_config = vars(sim), manifest

    previous_cwd = Path.cwd()
    original = sim.run_postturn_audit
    controller_results = {}
    os.chdir(output)
    sim.p.connect(sim.p.DIRECT)
    try:
        for name, controller in (
            ("postturn", original), ("skip_initial_centering", sim.run_initial_centering_audit)
        ):
            sim.run_postturn_audit = controller
            snapshots = []
            results = {}
            for method in ACTIONS:
                row = run_case(case, method, snapshots)
                log = Path(row["directory"]) / "heldout_decisions.json"
                decisions = json.loads(log.read_text(encoding="utf-8"))
                selected = next((d for d in decisions if "selected_offset_deg" in d), None)
                print(name, method, row["outcome"], row["sim_time_s"], flush=True)
                results[method] = {
                    "outcome": row["outcome"],
                    "collision_attempts": row["collision_attempts"],
                    "sim_time_s": row["sim_time_s"],
                    "completed_views": row["completed_views"],
                    "decision_index": None if selected is None else selected["decision_index"],
                    "episode_directory": row["directory"],
                }
            valid = all(
                row["outcome"] == "GOAL"
                and row["collision_attempts"] == 0
                and row["completed_views"] == int(action != "NoView")
                for action, row in results.items()
            )
            controller_results[name] = {
                "episodes": results,
                "gains_s": {
                    action: results["NoView"]["sim_time_s"] - results[action]["sim_time_s"]
                    for action in ACTIONS
                } if valid else None,
            }
    finally:
        sim.run_postturn_audit = original
        sim.p.disconnect()
        os.chdir(previous_cwd)
    return controller_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", help="Replay a single training scene under both controllers")
    parser.add_argument("--verify-initial", action="store_true", help="Compare full maps at ambiguous initial decisions")
    args = parser.parse_args()
    dataset = ROOT / "data" / "dataset"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in manifest["cases"]}
    if args.scene and (args.scene not in cases or cases[args.scene]["split"] != "train"):
        parser.error("--scene must name a training scene in the dataset manifest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = audit_inputs(dataset, cases)
    if args.verify_initial:
        report["initial_state_comparisons"] = verify_initial_states(
            manifest, cases, report["decision_conflicts"]
        )
    if args.scene:
        report["scene"] = args.scene
        report["controller_results"] = audit_controller(manifest, cases[args.scene], output)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output / "audit.json")


if __name__ == "__main__":
    main()
