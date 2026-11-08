"""Reconstruct the frozen test layout and check archived validation probes."""
import ast
import hashlib
import json
import sys

import numpy as np

from .common import FREEZE_ID, digest, read_json, require, write_json
from .predict import Predictors, validation


def function_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def signature(boxes):
    rows = sorted(tuple(round(float(value), 9) for value in [*size, *position]) for size, position in boxes)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def historical_layouts(paths, geometry):
    seeds, signatures = set(), set()

    def inspect(value):
        if isinstance(value, dict):
            if value.get("kind") in ("barrier", "gate_open", "gate_closed") and "seed" in value:
                seeds.add(int(value["seed"]))
                boxes = geometry(value)
                for mirror in (1, -1):
                    signatures.add(signature([
                        (size, [position[0], mirror * position[1], position[2]]) for size, position in boxes
                    ]))
            for child in value.values():
                inspect(child)
        elif isinstance(value, list) and value:
            if all(
                isinstance(box, list) and len(box) == 2
                and all(isinstance(vector, list) and len(vector) == 3
                        and all(isinstance(component, (int, float)) for component in vector)
                        for vector in box)
                for box in value
            ):
                signatures.add(signature(value))
            else:
                for child in value:
                    inspect(child)

    for path in sorted(paths):
        inspect(json.loads(path.read_text(encoding="utf-8-sig")))
    return seeds, signatures


def reconstruct(sources, geometry, old_seeds, old_signatures):
    seeds = {int(source["seed"]) for source in sources}
    require(len(sources) == len(seeds) == 20 and seeds.isdisjoint(old_seeds), "Fresh seed separation failed")
    cases, new_geometry = [], {}
    for source in sources:
        boxes = geometry(source)
        for mirror in (1, -1):
            mirrored = [
                [list(size), [float(pos[0]), float(mirror * pos[1]), float(pos[2])]]
                for size, pos in boxes
            ]
            key = signature(mirrored)
            require(key not in old_signatures, "Test geometry overlaps a historical layout")
            require(new_geometry.get(key, source["name"]) == source["name"], "Test source geometries overlap")
            new_geometry[key] = source["name"]
            for direction in (1, -1):
                suffix = f"my{'P' if mirror == 1 else 'M'}_{'LR' if direction == 1 else 'RL'}"
                cases.append({
                    "name": source["name"] + "_" + suffix,
                    "split": "test",
                    "source_group": source["name"],
                    "source_spec": dict(source),
                    "mirror_y": mirror,
                    "travel_direction": direction,
                    "start_xy": [-3.5 * direction, 0.0],
                    "goal_xy": [3.5 * direction, 0.0],
                    "initial_yaw_rad": 0.0 if direction == 1 else float(np.pi),
                    "geometry": mirrored,
                })
    require(len(cases) == len({case["name"] for case in cases}) == 80, "Expected 80 distinct test variants")
    return cases


def historical_manifests(bundle, roots):
    paths = {bundle.frozen / "dataset_manifest.json"}
    for root in roots:
        require(root.is_dir(), f"Historical manifest root does not exist: {root}")
        paths.update(path.resolve() for path in root.rglob("*manifest*.json"))
    return {
        path for path in paths
        if bundle.final not in path.parents
        and bundle.manifest_path.parent not in path.parents
    }


def verify_preflight(bundle, output, roots, *, cuda=False):
    protocol = bundle.protocol(verify_source=True)
    archived = bundle.manifest(protocol)
    simulation = bundle.source / "src/obym/simulation.py"
    node = function_node(simulation, "ml_scene_geometry")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<geometry>", "exec"), namespace)
    geometry = namespace["ml_scene_geometry"]

    sys.path.insert(0, str(bundle.source / "src"))
    from obym.initial_centering_audit import make_initial_centering_audit

    runner = function_node(simulation, "run_postturn_audit")
    controller = make_initial_centering_audit(runner)
    require(
        hashlib.sha256(ast.dump(controller).encode()).hexdigest() == protocol["controller_sha256"],
        "Controller AST hash mismatch. Use Python 3.12 for preflight.",
    )
    paths = historical_manifests(bundle, roots)
    old_seeds, old_signatures = historical_layouts(paths, geometry)
    expected_seeds = {*range(1000, 1024), *range(2000, 2006), *range(3000, 3020)}
    require(expected_seeds <= old_seeds, "Missing historical seeds from train, validation, or previous held-out scenes")
    cases = reconstruct(protocol["test_sources"], geometry, old_seeds, old_signatures)
    require(cases == archived["cases"], "Reconstructed cases differ from the archived test manifest")
    require(
        archived["base_config"] == read_json(bundle.frozen / "dataset_manifest.json")["base_config"],
        "Simulator base configuration differs",
    )

    models = Predictors(bundle)
    checks = validation(bundle, models, cuda_models=Predictors(bundle, device="cuda") if cuda else None)
    for method, check in checks.items():
        require(check["minimum_gain_s"] == archived["model_checks"][method]["minimum_gain_s"], f"Threshold mismatch: {method}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preflight_check.json", {
        "freeze_id": FREEZE_ID,
        "archived_preflight_sha256": digest(bundle.manifest_path),
        "historical_manifests_checked": len(paths),
        "historical_seeds_checked": len(old_seeds),
        "fresh_sources": len(protocol["test_sources"]),
        "test_cases_matched": len(cases),
        "torch_version": str(models.torch.__version__),
        "navigation_episodes_executed": 0,
        "model_checks": checks,
    })
    print(f"Preflight passed: {len(old_seeds)} historical seeds, {len(cases)}/80 test cases, 24/24 validation choices per model")
    print(f"Navigation episodes executed: 0; wrote {output}")
