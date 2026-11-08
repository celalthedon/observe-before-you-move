import subprocess
import sys
from pathlib import Path

repo = Path.home() / "obym-github"
output = repo / "results" / "v3-initial-dataset"
batch_limit = 108

collector = r'''
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

repo, output = map(Path, sys.argv[1:3])
limit = int(sys.argv[3])
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(repo / "tools"))

from evaluate import run_case, selectors, sim
from obym.initial_centering_audit import make_initial_centering_audit

manifest_bytes = (repo / "data/dataset/manifest.json").read_bytes()
manifest = json.loads(manifest_bytes)
cases = [case for case in manifest["cases"] if case["split"] in ("train", "validation")]

source = Path(sim.__file__).read_text(encoding="utf-8")
runner = next(
    node for node in ast.parse(source).body
    if isinstance(node, ast.FunctionDef) and node.name == "run_postturn_audit"
)
if hashlib.sha256(ast.dump(runner).encode()).hexdigest() != manifest["runner_sha256"]:
    raise RuntimeError("Controller hash manifestlə uyğun gəlmir.")

patched = make_initial_centering_audit(runner)
module = ast.fix_missing_locations(ast.Module(body=[patched], type_ignores=[]))
exec(compile(module, sim.__file__, "exec"), vars(sim))
sim.run_postturn_audit = sim.run_initial_centering_audit
sim.ML_CHANNELS = manifest["base_config"]["channels"]
sim.ML_PATCH_SIZE = manifest["base_config"]["patch_size"]
selectors.ns, selectors.test_config = vars(sim), manifest

protocol = {
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "controller_sha256": hashlib.sha256(ast.dump(patched).encode()).hexdigest(),
    "decision": "first_eligible",
    "angles_deg": [-45, 45],
    "target": "NoView_sim_time_minus_single_view_sim_time",
}
output.mkdir(parents=True, exist_ok=True)
protocol_path = output / "protocol.json"
if protocol_path.exists():
    if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Mövcud dataset fərqli protokolla toplanıb.")
protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

def valid(result, views):
    return (
        result["outcome"] == "GOAL"
        and result["collision_attempts"] == 0
        and result["completed_views"] == views
    )

def collect(case):
    snapshots = []
    baseline = run_case(case, "NoView", snapshots)
    meta = {
        "scene": case["name"],
        "split": case["split"],
        "source_group": case["source_spec"]["name"],
        "baseline": baseline,
        "status": "invalid_baseline",
    }
    if not valid(baseline, 0):
        return meta, {}

    for index, state in enumerate(snapshots):
        path = state["path"]
        grid, xy, yaw = state["grid"], state["xy"], state["yaw"]
        path = None if path is None else [tuple(cell) for cell in path]
        corridor = selectors.test_region(grid, xy, path)
        if corridor is not None:
            break
    else:
        meta["status"] = "no_eligible_state"
        return meta, {}

    meta["state_index"] = index
    meta["views"] = {}
    images, poses, gains, flags = [], [], [], []

    for angle, method in [(-45, "FixedMinus45"), (45, "FixedPlus45")]:
        candidate_yaw = sim.wrap_angle(yaw + np.deg2rad(angle))
        image, pose = sim.encode_view_sample(grid, xy, yaw, xy, candidate_yaw, corridor)
        result = run_case(case, method, snapshots)
        trace_path = Path(result["directory"]) / "heldout_decisions.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        selected = [
            (row["decision_index"], row["selected_offset_deg"], row["prefix_verified"])
            for row in trace if "selected_offset_deg" in row
        ]
        if selected != [(index, angle, True)]:
            raise RuntimeError(f"Baxış uyğun gəlmədi: {case['name']} {angle:+d}")

        ok = valid(result, 1)
        images.append(image)
        poses.append(pose)
        flags.append(ok)
        gains.append(baseline["sim_time_s"] - result["sim_time_s"] if ok else np.nan)
        meta["views"][method] = result

    meta["status"] = "valid" if all(flags) else "invalid_view"
    return meta, {
        "image": np.stack(images),
        "pose": np.stack(poses),
        "offset_deg": np.array([-45, 45], dtype=np.int16),
        "target_time_saving_s": np.array(gains, dtype=np.float32),
        "valid": np.array(flags, dtype=bool),
    }

os.chdir(output)
for split in ("train", "validation"):
    Path(split).mkdir(exist_ok=True)

pending = []
for case in cases:
    destination = Path(case["split"]) / f"{case['name']}.npz"
    if not destination.is_file():
        pending.append((case, destination))

print(f"V3 | hazır={len(cases) - len(pending)}/{len(cases)} | bu batch={min(limit, len(pending))}", flush=True)

sim.p.connect(sim.p.DIRECT)
try:
    for case, destination in pending[:limit]:
        meta, arrays = collect(case)
        temp = destination.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, metadata=json.dumps(meta), **arrays)
        temp.replace(destination)
        gains = arrays.get("target_time_saving_s", [])
        summary = " / ".join(f"{gain:+.3f} s" for gain in gains)
        print(f"DATA | {case['name']} | {meta['status']} | {summary}", flush=True)
finally:
    sim.p.disconnect()

for split in ("train", "validation"):
    done = len(list(Path(split).glob("*.npz")))
    total = sum(case["split"] == split for case in cases)
    print(f"PROGRESS | {split} | {done}/{total}", flush=True)
print(output, flush=True)
'''

with subprocess.Popen(
    [sys.executable, "-X", "utf8", "-u", "-c", collector, str(repo), str(output), str(batch_limit)],
    cwd=repo,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
) as process:
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError(f"Data yığımı xəta ilə dayandı: {process.returncode}")