
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1])
frozen = repo / "results/v3-final-protocol/94fd47e48b4f"
out = frozen / "preflight"
protocol = json.loads(
    (frozen / "protocol.json").read_text(encoding="utf-8")
)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert hashlib.sha256(
    json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
).hexdigest()[:12] == frozen.name

for name, expected in protocol["artifact_hashes"].items():
    assert digest(frozen / name) == expected, f"Frozen artifact changed: {name}"

for name, expected in protocol["source_hashes"].items():
    assert digest(repo / name) == expected, f"Source changed: {name}"

tree = ast.parse(
    (repo / "src/obym/simulation.py").read_text(encoding="utf-8")
)
node = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "ml_scene_geometry"
)
namespace = {"np": np}
exec(
    compile(
        ast.Module(body=[node], type_ignores=[]),
        "<geometry>",
        "exec",
    ),
    namespace,
)
geometry = namespace["ml_scene_geometry"]

def signature(boxes):
    rows = sorted(
        tuple(round(float(v), 9) for v in list(size) + list(pos))
        for size, pos in boxes
    )
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

old_seeds, old_geometry = set(), set()

def remember_geometry(value):
    if not isinstance(value, list) or not value:
        return
    if all(
        isinstance(box, list) and len(box) == 2
        and all(
            isinstance(vector, list) and len(vector) == 3
            and all(isinstance(v, (int, float)) for v in vector)
            for vector in box
        )
        for box in value
    ):
        old_geometry.add(signature(value))
    else:
        for child in value:
            remember_geometry(child)

def inspect(value):
    if isinstance(value, dict):
        if (
            value.get("kind") in ("barrier", "gate_open", "gate_closed")
            and "seed" in value
        ):
            old_seeds.add(int(value["seed"]))
            boxes = geometry(value)
            for mirror in (1, -1):
                old_geometry.add(signature([
                    (size, [pos[0], mirror * pos[1], pos[2]])
                    for size, pos in boxes
                ]))
        remember_geometry(value.get("geometry"))
        for child in value.values():
            inspect(child)
    elif isinstance(value, list):
        for child in value:
            inspect(child)

manifest_paths = set()
for root in (
    repo / "data",
    repo / "results",
    Path.home() / "observe-before-you-move/results",
):
    if root.is_dir():
        manifest_paths.update(root.rglob("*manifest*.json"))

own_test = repo / "results/v3-final-test" / frozen.name

for path in sorted(manifest_paths):
    if (
        out in path.parents
        or own_test in path.parents
        or path == frozen / "dataset_manifest.json"
    ):
        continue
    inspect(json.loads(path.read_text(encoding="utf-8-sig")))

assert set(range(1000, 1024)) <= old_seeds
assert set(range(2000, 2006)) <= old_seeds
assert set(range(3000, 3020)) <= old_seeds

fresh_seeds = [
    int(source["seed"]) for source in protocol["test_sources"]
]
assert len(fresh_seeds) == len(set(fresh_seeds)) == 20
assert not set(fresh_seeds) & old_seeds, (
    "Test seed əvvəlki manifestdə var: "
    + str(sorted(set(fresh_seeds) & old_seeds))
)

cases, new_geometry = [], {}

for source in protocol["test_sources"]:
    boxes = geometry(source)
    for mirror in (1, -1):
        mirrored = [
            [
                list(size),
                [float(pos[0]), float(mirror * pos[1]), float(pos[2])],
            ]
            for size, pos in boxes
        ]
        key = signature(mirrored)
        assert key not in old_geometry, "Test geometry overlaps an archived layout."
        assert key not in new_geometry or new_geometry[key] == source["name"]
        new_geometry[key] = source["name"]

        for direction in (1, -1):
            suffix = (
                f"my{'P' if mirror == 1 else 'M'}_"
                f"{'LR' if direction == 1 else 'RL'}"
            )
            cases.append({
                "name": source["name"] + "_" + suffix,
                "split": "test",
                "source_group": source["name"],
                "source_spec": dict(source),
                "mirror_y": mirror,
                "travel_direction": direction,
                "start_xy": [-3.5 * direction, 0.0],
                "goal_xy": [3.5 * direction, 0.0],
                "initial_yaw_rad": (
                    0.0 if direction == 1 else float(np.pi)
                ),
                "geometry": mirrored,
            })

assert len(cases) == len({case["name"] for case in cases}) == 80

print(
    f"SEPARATION PASS | archived seeds={len(old_seeds)} | "
    "fresh sources=20 | variants=80",
    flush=True,
)

data = repo / "results/v3-initial-dataset/validation"
images, poses, targets, names = [], [], [], []

for path in sorted(data.glob("*.npz")):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(saved["metadata"].item())
        assert meta["status"] == "valid"
        assert meta["split"] == "validation"
        assert saved["image"].shape == (2, 7, 80, 80)
        assert saved["pose"].shape == (2, 6)
        assert np.array_equal(saved["offset_deg"], [-45, 45])
        assert saved["valid"].all()
        images.append(saved["image"].copy())
        poses.append(saved["pose"].copy())
        targets.append(saved["target_time_saving_s"].copy())
        names.append(meta["scene"])

images, poses, targets = map(np.stack, (images, poses, targets))
assert len(names) == len(set(names)) == 24
assert all(
    np.isfinite(array).all()
    for array in (images, poses, targets)
)

with (frozen / "validation_decisions.csv").open(
    encoding="utf-8", newline=""
) as handle:
    rows = list(csv.DictReader(handle))

archive = {(row["policy"], row["scene"]): row for row in rows}
assert len(archive) == len(rows)

for method, index in (("FixedMinus45", 0), ("FixedPlus45", 1)):
    assert np.allclose(
        targets[:, index],
        [float(archive[method, name]["gain_s"]) for name in names],
        atol=1e-4,
        rtol=0,
    )

def choose(values, threshold):
    best = values.argmax(axis=1)
    return np.where(
        values[np.arange(len(values)), best] > threshold,
        np.array([-45, 45])[best],
        0,
    )

def check_choices(method, values, threshold):
    expected = np.array([
        int(archive[method, name]["offset_deg"]) for name in names
    ])
    assert np.array_equal(
        choose(values, threshold), expected
    ), f"Archived decisions differ: {method}"

checks = {}

for method in ("SpatialRidgeCV", "PoseRidgeCV", "MeanMapRidgeCV"):
    with np.load(
        frozen / f"{method}.npz", allow_pickle=False
    ) as saved:
        level = int(saved["level"].item())
        threshold = float(saved["threshold_s"].item())
        parts = [poses.astype(np.float64)]

        for size in (1, 2, 4, 8):
            if size <= level:
                pooled = images.reshape(
                    24, 2, 7, size, 80 // size, size, 80 // size
                ).mean(axis=(4, 6))
                parts.append(pooled.reshape(24, 2, -1))

        x = np.concatenate(parts, axis=2)
        values = (
            ((x - saved["mean"]) / saved["std"])
            @ saved["coefficient"]
            + saved["target_center_s"]
        )
        assert np.array_equal(saved["validation_scene"], names)
        difference = float(np.max(np.abs(
            values - saved["validation_prediction_s"]
        )))
        assert difference <= 1e-6, (method, difference)

    check_choices(method, values, threshold)
    checks[method] = {
        "probe_max_difference_s": difference,
        "minimum_gain_s": threshold,
    }
    print(
        f"MODEL PASS | {method} | "
        f"prediction difference={difference:.8f} s | decisions=24/24",
        flush=True,
    )

sys.path.insert(0, str(repo / "src"))

import torch
from obym.inference import _ModelFactory

torch.set_num_threads(1)
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "ieee"
print("PRECISION | IEEE FP32 | TF32 disabled", flush=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

checkpoint = torch.load(
    frozen / "cnn_models.pt",
    map_location="cpu",
    weights_only=True,
)
assert "config" in checkpoint and "cnn_members" in checkpoint, list(checkpoint)

scale = float(checkpoint["config"]["target_scale_s"])
threshold = float(checkpoint["config"]["minimum_gain_s"])
assert np.isfinite(scale) and scale > 0

model_type = _ModelFactory.make()
x = torch.from_numpy(images.astype(np.float32)).flatten(0, 1)
z = torch.from_numpy(poses.astype(np.float32)).flatten(0, 1)

for method in ("CNN_MSE", "CNN_MSE_Rank"):
    members = checkpoint["cnn_members"][method]
    assert sorted(member["seed"] for member in members) == [11, 23, 47]
    predictions, cuda_predictions = [], []

    for member in members:
        model = model_type()
        model.load_state_dict(member["state_dict"], strict=True)
        model.eval()

        with torch.inference_mode():
            predictions.append(
                model(x, z).numpy().reshape(24, 2).astype(np.float64)
                * scale
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
                cuda_predictions.append(
                    model(x.to("cuda"), z.to("cuda"))
                    .cpu().numpy().reshape(24, 2).astype(np.float64)
                    * scale
                )
        del model

    values = np.mean(predictions, axis=0)
    check_choices(method, values, threshold)
    difference = None

    if cuda_predictions:
        cuda_values = np.mean(cuda_predictions, axis=0)
        difference = float(np.max(np.abs(values - cuda_values)))
        assert difference <= 1e-4, (method, difference)
        check_choices(method, cuda_values, threshold)

    checks[method] = {
        "archived_decisions_matched": 24,
        "cpu_cuda_max_difference_s": difference,
        "minimum_gain_s": threshold,
        "target_scale_s": scale,
    }
    print(
        f"MODEL PASS | {method} | archived decisions=24/24 | "
        f"CPU/CUDA difference={difference}",
        flush=True,
    )

base = json.loads(
    (frozen / "dataset_manifest.json").read_text(encoding="utf-8")
)
test_manifest = {
    "freeze_id": frozen.name,
    "protocol_sha256": digest(frozen / "protocol.json"),
    "controller_sha256": protocol["controller_sha256"],
    "base_config": base["base_config"],
    "cases": cases,
    "policies": protocol["policies"],
    "model_checks": checks,
    "fp32_precision": "ieee",
}

out.mkdir(parents=True, exist_ok=True)
manifest_path = out / "test_manifest.json"

if manifest_path.exists():
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert all(
        previous[key] == test_manifest[key]
        for key in (
            "freeze_id", "protocol_sha256", "controller_sha256",
            "base_config", "cases", "policies",
        )
    )
else:
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(test_manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)

print("TEST PREFLIGHT READY |", manifest_path, flush=True)
print("Navigation episodes executed: 0", flush=True)
