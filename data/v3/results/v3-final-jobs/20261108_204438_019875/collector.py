
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1]).resolve()
batch_limit = int(sys.argv[2])
minutes = float(sys.argv[3])
assert batch_limit > 0 and minutes > 0

frozen = repo / "results/v3-final-protocol/94fd47e48b4f"
root = repo / "results/v3-final-test" / frozen.name
manifest_path = frozen / "preflight/test_manifest.json"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)

def save_npz(path, **values):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temp.replace(path)

protocol = read(frozen / "protocol.json")
manifest = read(manifest_path)
assert manifest["freeze_id"] == frozen.name
assert manifest["protocol_sha256"] == digest(frozen / "protocol.json")
assert manifest["fp32_precision"] == "ieee"

for name, expected in protocol["artifact_hashes"].items():
    assert digest(frozen / name) == expected, name
for name, expected in protocol["source_hashes"].items():
    assert digest(repo / name) == expected, name

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(repo / "tools"))

from evaluate import run_case, selectors, sim
from obym.initial_centering_audit import make_initial_centering_audit

assert "torch" not in sys.modules

tree = ast.parse(Path(sim.__file__).read_text(encoding="utf-8"))
runner = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "run_postturn_audit"
)
patched = make_initial_centering_audit(runner)
assert hashlib.sha256(
    ast.dump(patched).encode()
).hexdigest() == protocol["controller_sha256"]

exec(
    compile(
        ast.fix_missing_locations(
            ast.Module(body=[patched], type_ignores=[])
        ),
        sim.__file__,
        "exec",
    ),
    vars(sim),
)
sim.run_postturn_audit = sim.run_initial_centering_audit

base = manifest["base_config"]
sim.ML_CHANNELS = base["channels"]
sim.ML_PATCH_SIZE = base["patch_size"]

checks = {
    "grid_res": sim.GRID_RES,
    "camera": dict(sim.CAMERA),
    "channels": list(sim.ML_CHANNELS),
    "patch_size": sim.ML_PATCH_SIZE,
    "view_config": dict(sim.OBYM_VIEW_CONFIG),
    "nav_speed": sim.NAV_SPEED,
    "turn_speed": sim.TURN_SPEED,
    "sensor_period": sim.SENSOR_PERIOD,
    "robot_radius": sim.ROBOT_RADIUS,
    "clearance": sim.CLEARANCE,
    "goal_tolerance": sim.GOAL_TOLERANCE,
    "voxel": [sim.Z_MIN, sim.Z_MAX, sim.NZ],
}
for key, value in checks.items():
    assert value == base[key], key

selectors.ns = vars(sim)
selectors.test_config = {
    **manifest,
    "max_sim_time": protocol["max_sim_time_s"],
    "max_decisions": protocol["max_decisions"],
}

root.mkdir(parents=True, exist_ok=True)
identity = {
    "freeze_id": frozen.name,
    "preflight_sha256": digest(manifest_path),
    "collector_sha256": digest(Path(__file__)),
    "branches": ["NoView", "FixedMinus45", "FixedPlus45"],
    "policy_results": (
        "derive_frozen_first_eligible_choices_from_executed_branches"
    ),
}
identity_path = root / "collection_manifest.json"

if identity_path.exists():
    assert read(identity_path) == identity, (
        "Collection code or manifest changed."
    )
else:
    write(identity_path, identity)

os.chdir(root)
started = time.perf_counter()
new_runs = 0

class Pause(Exception):
    pass

def episode(case, method, snapshots):
    global new_runs
    if time.perf_counter() - started >= minutes * 60:
        raise Pause()

    print(f"RUN | {case['name']} | {method}", flush=True)
    summary = run_case(case, method, snapshots)
    new_runs += 1
    summary["directory"] = str(Path(summary["directory"]).resolve())
    summary["freeze_id"] = frozen.name

    directory = Path(summary["directory"])
    write(directory / "summary.json", summary)
    hashes = {
        path.name: digest(path)
        for path in directory.iterdir()
        if path.is_file()
    }
    return {
        "summary": summary,
        "episode_hashes": hashes,
        "origin": "executed",
    }

def verify(record):
    directory = Path(record["summary"]["directory"])
    for name, expected in record["episode_hashes"].items():
        assert digest(directory / name) == expected, str(directory / name)

def collect(case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    case_path = directory / "case.json"

    if case_path.exists():
        assert read(case_path) == case
    else:
        write(case_path, case)

    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = read(complete_path)
        for name, expected in complete["file_hashes"].items():
            assert digest(directory / name) == expected
        for method in identity["branches"]:
            verify(read(directory / f"{method}.json"))
        return

    baseline_path = directory / "baseline.npz"

    if baseline_path.exists():
        with np.load(baseline_path, allow_pickle=False) as saved:
            record = json.loads(saved["metadata"].item())
            paths = json.loads(saved["paths"].item())
            snapshots = [
                {
                    "grid": saved["grid"][i].copy(),
                    "xy": saved["xy"][i].copy(),
                    "yaw": float(saved["yaw"][i]),
                    "time_s": float(saved["time_s"][i]),
                    "path": path,
                }
                for i, path in enumerate(paths)
            ]
        verify(record)
    else:
        snapshots = []
        record = episode(case, "NoView", snapshots)
        save_npz(
            baseline_path,
            grid=(
                np.stack([state["grid"] for state in snapshots])
                if snapshots else
                np.empty((0, sim.NY, sim.NX), dtype=np.int8)
            ),
            xy=np.asarray(
                [state["xy"] for state in snapshots],
                dtype=np.float64,
            ).reshape(-1, 2),
            yaw=np.asarray([state["yaw"] for state in snapshots]),
            time_s=np.asarray([state["time_s"] for state in snapshots]),
            paths=json.dumps([state["path"] for state in snapshots]),
            metadata=json.dumps(record, allow_nan=False),
        )

    write(directory / "NoView.json", record)

    eligible = None
    for index, state in enumerate(snapshots):
        path = (
            None if state["path"] is None
            else [tuple(cell) for cell in state["path"]]
        )
        region = selectors.test_region(
            state["grid"], state["xy"], path
        )
        if region is not None:
            eligible = index
            samples = [
                sim.encode_view_sample(
                    state["grid"],
                    state["xy"],
                    state["yaw"],
                    state["xy"],
                    sim.wrap_angle(
                        state["yaw"] + np.deg2rad(angle)
                    ),
                    region,
                )
                for angle in (-45, 45)
            ]
            images = np.stack([sample[0] for sample in samples])
            poses = np.stack([sample[1] for sample in samples])

            assert images.shape == (2, 7, 80, 80)
            assert poses.shape == (2, 6)
            assert np.isin(images, [0, 1]).all()
            assert np.isfinite(poses).all()

            input_path = directory / "input.npz"
            if input_path.exists():
                with np.load(input_path, allow_pickle=False) as saved:
                    assert np.array_equal(saved["image"], images)
                    assert np.array_equal(saved["pose"], poses)
                    assert int(saved["state_index"].item()) == index
            else:
                save_npz(
                    input_path,
                    image=images,
                    pose=poses,
                    offset_deg=np.array([-45, 45]),
                    state_index=index,
                )
            break

    for angle, method in (
        (-45, "FixedMinus45"), (45, "FixedPlus45")
    ):
        path = directory / f"{method}.json"
        if path.exists():
            verify(read(path))
            continue

        if eligible is None:
            branch = {
                **record,
                "origin": "identical_no_eligible_state",
            }
        else:
            branch = episode(case, method, snapshots)
            trace = read(
                Path(branch["summary"]["directory"])
                / "heldout_decisions.json"
            )
            selected = [
                (
                    row["decision_index"],
                    row["selected_offset_deg"],
                    row["prefix_verified"],
                )
                for row in trace
                if "selected_offset_deg" in row
            ]
            assert selected == [(eligible, angle, True)], (
                case["name"], selected
            )
            assert branch["summary"]["view_attempts"] == 1

        write(path, branch)

    names = [
        "case.json", "baseline.npz", "NoView.json",
        "FixedMinus45.json", "FixedPlus45.json",
    ]
    if eligible is not None:
        names.append("input.npz")

    write(complete_path, {
        "freeze_id": frozen.name,
        "eligible": eligible is not None,
        "state_index": eligible,
        "file_hashes": {
            name: digest(directory / name) for name in names
        },
    })

completed_normally = False
sim.p.connect(sim.p.DIRECT)

try:
    original = read(frozen / "dataset_manifest.json")
    parity_case = next(
        case for case in original["cases"]
        if case["split"] == "train"
    )
    parity_dir = root / "train_parity"
    collect(parity_case, parity_dir)

    reference_path = (
        repo / "results/v3-initial-dataset/train"
        / f"{parity_case['name']}.npz"
    )
    with np.load(reference_path, allow_pickle=False) as saved:
        reference = json.loads(saved["metadata"].item())
        with np.load(
            parity_dir / "input.npz", allow_pickle=False
        ) as current:
            assert np.array_equal(saved["image"], current["image"])
            assert np.array_equal(saved["pose"], current["pose"])

    expected = {
        "NoView": reference["baseline"],
        **reference["views"],
    }

    for method in identity["branches"]:
        actual = read(parity_dir / f"{method}.json")["summary"]
        for key in (
            "outcome", "collision_attempts",
            "view_attempts", "completed_views",
        ):
            assert actual[key] == expected[method][key], (method, key)
        for key in ("sim_time_s", "distance_m"):
            assert abs(
                actual[key] - expected[method][key]
            ) <= 1e-6, (method, key)

    print(
        "TRAIN PARITY PASS | 3 archived branches and encoded inputs",
        flush=True,
    )

    cases = manifest["cases"]
    assert len(cases) == 80

    pending = [
        case for case in cases
        if not (
            root / "cases" / case["name"] / "complete.json"
        ).exists()
    ]

    print(
        f"FINAL TEST | complete={80-len(pending)}/80 | "
        f"batch={min(batch_limit, len(pending))} | "
        f"limit={minutes:g} min",
        flush=True,
    )

    selected_names = {
        case["name"] for case in pending[:batch_limit]
    }

    for case in cases:
        directory = root / "cases" / case["name"]
        if (
            (directory / "complete.json").exists()
            or case["name"] in selected_names
        ):
            collect(case, directory)
        else:
            continue

        done = sum(
            (root / "cases" / item["name"] / "complete.json").exists()
            for item in cases
        )
        print(
            f"SAVED | {case['name']} | completed={done}/80",
            flush=True,
        )

    completed_normally = True

except Pause:
    print(
        "BATCH TIME LIMIT | saved branches will be reused",
        flush=True,
    )

finally:
    sim.p.disconnect()
    done = sum(
        (root / "cases" / case["name"] / "complete.json").exists()
        for case in manifest["cases"]
    )
    progress = {
        "freeze_id": frozen.name,
        "complete": done,
        "total": 80,
        "new_episodes_this_batch": new_runs,
        "elapsed_min": (time.perf_counter() - started) / 60,
    }
    write(root / "progress.json", progress)

    print(
        f"PROGRESS | complete={done}/80 | "
        f"new episodes={new_runs} | "
        f"minutes={progress['elapsed_min']:.1f}",
        flush=True,
    )

    if completed_normally and done == 80:
        print("BRANCH COLLECTION READY", flush=True)
    elif done < 80:
        print(
            "RESUME | run the same cell again after any reported error is resolved",
            flush=True,
        )
    else:
        print(
            "STOPPED | inspect the reported error before continuing",
            flush=True,
        )

    print("OUTPUT |", root, flush=True)
