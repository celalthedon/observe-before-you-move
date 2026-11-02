"""Run a selected frozen-protocol case; the default never launches the full test."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obym import simulation as sim
from obym import heldout_selector as selectors
from obym.isolated_inference import IsolatedPredictor


def run_case(case, method, snapshots):
    modes = {"NoView": "none", "FixedMinus45": "minus", "FixedPlus45": "plus", "PoseOnly": "pose", "ValueLinear": "linear", "ValueCNN": "cnn"}
    if method == "HeadingRule":
        method_mode = "minus" if np.cos(case["initial_yaw_rad"]) < 0 else "none"
    else:
        method_mode = modes[method]
    trace = []
    sim.start_xy, sim.goal_xy = list(case["start_xy"]), list(case["goal_xy"])
    sim.ml_scene_geometry = lambda spec: [(list(size), list(pos)) for size, pos in case["geometry"]]
    sim.choose_obym_view = selectors.test_selector(method_mode, snapshots, trace, method_mode == "none")
    sim.AUDIT_TRACE = {"postturn_checks": 0, "changed_next_action": 0}

    def reset():
        sim.reset_ml_scene(case["source_spec"])
        pos, _ = sim.p.getBasePositionAndOrientation(sim.robot_id)
        sim.p.resetBasePositionAndOrientation(sim.robot_id, pos, sim.p.getQuaternionFromEuler([0, 0, case["initial_yaw_rad"]]))
        assert not any(sim.p.getClosestPoints(sim.robot_id, body, distance=0) for body in sim.solid_ids)

    result = sim.run_postturn_audit(method="OBYM", reset_fn=reset, scene_name=case["name"], animate=False,
                                   max_sim_time=selectors.test_config["max_sim_time"], max_decisions=selectors.test_config["max_decisions"])
    assert result["view_attempts"] <= 1 and result["completed_views"] <= 1
    result.update(method=method, postturn_trace=dict(sim.AUDIT_TRACE),
                  prediction_wall_s=sum(r.get("prediction_wall_s", 0) for r in trace))
    path = Path(result["directory"])
    (path / "heldout_decisions.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(args):
    root, output = args.root.resolve(), args.output.resolve()
    config = json.loads((root / "data/heldout/manifest.json").read_text(encoding="utf-8"))
    dataset = json.loads((root / "data/dataset/manifest.json").read_text(encoding="utf-8"))
    tree = ast.parse((root / "src/obym/simulation.py").read_text(encoding="utf-8"))
    runner = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_postturn_audit")
    assert hashlib.sha256(ast.dump(runner).encode()).hexdigest() == config["runner_sha256"] == dataset["runner_sha256"]
    sim.ML_CHANNELS, sim.ML_PATCH_SIZE = config["base_config"]["channels"], config["base_config"]["patch_size"]
    selectors.ns, selectors.test_config = vars(sim), config
    cases = config["cases"] if args.all_cases else [next(c for c in config["cases"] if c["name"] == args.scene)]
    output.mkdir(parents=True, exist_ok=False)
    old_cwd = Path.cwd()
    os.chdir(output)  # the frozen runner writes relative results/day3 paths
    sim.p.connect(sim.p.DIRECT)
    rows = []
    try:
        with IsolatedPredictor(root / "data/model") as predictor:
            assert predictor.health["checkpoint_sha256"] == config["checkpoint_sha256"]
            assert predictor.health["minimum_value_s"] == config["minimum_value_s"]
            selectors.test_predict = predictor.predict
            print("PASS | isolated CNN/linear/pose probes |", predictor.health["probe_max_difference_s"])
            for case in cases:
                snapshots = []
                baseline = run_case(case, "NoView", snapshots)
                if args.method == "all":
                    methods = [m for m in config["methods"] if m not in ("NoView", "HeadingRule")]
                else:
                    methods = [] if args.method == "NoView" else [args.method]
                rows.append(baseline)
                for method in methods:
                    rows.append(run_case(case, method, snapshots))
                if args.method == "all":
                    reference = "FixedMinus45" if np.cos(case["initial_yaw_rad"]) < 0 else "NoView"
                    heading = next(r for r in rows if r["scene"] == case["name"] and r["method"] == reference).copy()
                    heading.update(method="HeadingRule", result_origin="derived_identical_policy", reference_method=reference)
                    rows.append(heading)
    finally:
        sim.p.disconnect()
        os.chdir(old_cwd)
    (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.parity:
        expected = json.loads((root / "data/heldout/results.json").read_text(encoding="utf-8"))
        index = {(r["scene"], r["method"]): r for r in expected}
        for row in rows:
            ref = index[row["scene"], row["method"]]
            for key in ("outcome", "completed_views", "collision_attempts", "view_attempts", "observations", "planning_cycles"):
                assert row[key] == ref[key], (key, row[key], ref[key])
            for key in ("sim_time_s", "distance_m", "goal_error_m"):
                assert abs(row[key] - ref[key]) <= 1e-6, (key, row[key], ref[key])
        print(f"PASS | archived episode parity | {len(rows)} policy entries")
    print("wrote", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results/episode_parity")
    parser.add_argument("--scene", default="test_layout_3001_myP_LR")
    parser.add_argument("--method", choices=["all", "NoView", "FixedMinus45", "FixedPlus45", "HeadingRule", "PoseOnly", "ValueLinear", "ValueCNN"], default="ValueCNN")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--all-cases", action="store_true", help="Explicitly request all 80 cases (expensive)")
    main(parser.parse_args())
