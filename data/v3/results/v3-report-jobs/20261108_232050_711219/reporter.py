
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1]).resolve()
frozen = repo / "results/v3-final-protocol/94fd47e48b4f"
root = repo / "results/v3-final-test" / frozen.name
out = root / "reports"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)

def csv_write(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

protocol = read(frozen / "protocol.json")
manifest_path = frozen / "preflight/test_manifest.json"
manifest = read(manifest_path)

assert manifest["protocol_sha256"] == digest(frozen / "protocol.json")
assert manifest["fp32_precision"] == "ieee"
assert manifest["freeze_id"] == frozen.name
assert hashlib.sha256(
    json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
).hexdigest()[:12] == frozen.name

identity = read(root / "collection_manifest.json")
assert identity["freeze_id"] == frozen.name
assert identity["preflight_sha256"] == digest(manifest_path)

for name, expected in protocol["artifact_hashes"].items():
    assert digest(frozen / name) == expected, name
for name, expected in protocol["source_hashes"].items():
    assert digest(repo / name) == expected, name

cases = manifest["cases"]
policies = protocol["policies"]
assert len(cases) == 80
assert len(policies) == len(set(policies)) == 9

groups = np.array([case["source_group"] for case in cases])
sources = sorted(set(groups))
assert len(sources) == 20
assert all(np.count_nonzero(groups == group) == 4 for group in sources)

case_hashes = {}
for case in cases:
    directory = root / "cases" / case["name"]
    complete = read(directory / "complete.json")
    assert complete["freeze_id"] == frozen.name
    assert read(directory / "case.json") == case

    for name, expected in complete["file_hashes"].items():
        assert digest(directory / name) == expected, str(directory / name)

    for method in identity["branches"]:
        record = read(directory / f"{method}.json")
        episode_dir = Path(record["summary"]["directory"])
        for name, expected in record["episode_hashes"].items():
            assert digest(episode_dir / name) == expected, str(episode_dir / name)

    case_hashes[case["name"]] = digest(directory / "complete.json")

print("INTEGRITY PASS | 80/80 cases and episode artifacts", flush=True)

def show(summary, paired):
    for policy in policies:
        row = summary["policies"][policy]
        print(
            f"TEST | {policy:18s} | safe={row['safe_success']}/80 | "
            f"mean={row['mean_penalized_time_s']:.3f} s | "
            f"gain={row['mean_gain_vs_no_view_s']:+.3f} s | "
            f"looks={row['selected_looks']} | harmful={row['harmful_looks']}",
            flush=True,
        )
    print(
        "PAIRED | baseline time minus SpatialRidgeCV time; "
        "positive means time saved",
        flush=True,
    )
    for row in paired:
        print(
            f"Spatial vs {row['baseline']:18s} | "
            f"gain={row['mean_saved_s']:+.3f} s | "
            f"95% CI=[{row['ci_low_s']:+.3f}, {row['ci_high_s']:+.3f}] s",
            flush=True,
        )

marker = out / "report_complete.json"
if marker.exists():
    previous = read(marker)
    assert previous["case_hashes"] == case_hashes
    assert previous["protocol_sha256"] == digest(frozen / "protocol.json")
    for name, expected in previous["report_hashes"].items():
        assert digest(out / name) == expected, name
    show(
        read(out / "test_summary.json"),
        read(out / "paired_comparisons.json"),
    )
    print("EXISTING FINAL REPORT VERIFIED |", out, flush=True)
    sys.exit(0)

sys.path.insert(0, str(repo / "src"))

import torch
from obym.inference import _ModelFactory

torch.set_num_threads(1)
torch.backends.fp32_precision = "ieee"
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"
torch.backends.cudnn.conv.fp32_precision = "ieee"

checkpoint = torch.load(
    frozen / "cnn_models.pt",
    map_location="cpu",
    weights_only=True,
)
scale = float(checkpoint["config"]["target_scale_s"])
thresholds, ridge, cnns = {}, {}, {}

for method in ("SpatialRidgeCV", "PoseRidgeCV", "MeanMapRidgeCV"):
    with np.load(frozen / f"{method}.npz", allow_pickle=False) as saved:
        ridge[method] = {
            key: saved[key].copy() for key in saved.files
        }
    thresholds[method] = float(ridge[method]["threshold_s"].item())

model_type = _ModelFactory.make()

for method in ("CNN_MSE", "CNN_MSE_Rank"):
    cnns[method] = []
    for member in checkpoint["cnn_members"][method]:
        model = model_type()
        model.load_state_dict(member["state_dict"], strict=True)
        model.eval()
        cnns[method].append(model)
    thresholds[method] = float(checkpoint["config"]["minimum_gain_s"])

learned = list(ridge) + list(cnns)

def predict(method, image, pose):
    if method in ridge:
        saved = ridge[method]
        parts = [pose.astype(np.float64)]
        for size in (1, 2, 4, 8):
            if size <= int(saved["level"].item()):
                pooled = image.reshape(
                    len(image), 2, 7,
                    size, 80 // size, size, 80 // size,
                ).mean(axis=(4, 6))
                parts.append(pooled.reshape(len(image), 2, -1))
        x = np.concatenate(parts, axis=2)
        return (
            ((x - saved["mean"]) / saved["std"])
            @ saved["coefficient"]
            + saved["target_center_s"]
        )

    x = torch.from_numpy(image.astype(np.float32)).flatten(0, 1)
    z = torch.from_numpy(pose.astype(np.float32)).flatten(0, 1)
    with torch.inference_mode():
        values = [
            model(x, z).numpy().reshape(-1, 2).astype(np.float64) * scale
            for model in cnns[method]
        ]
    return np.mean(values, axis=0)

def choose(values, threshold):
    best = values.argmax(axis=1)
    return np.where(
        values[np.arange(len(values)), best] > threshold,
        np.array([-45, 45])[best],
        0,
    )

with (frozen / "validation_decisions.csv").open(
    encoding="utf-8", newline=""
) as handle:
    archived = {
        (row["policy"], row["scene"]): row
        for row in csv.DictReader(handle)
    }

probe_images, probe_poses, probe_names = [], [], []
for path in sorted(
    (repo / "results/v3-initial-dataset/validation").glob("*.npz")
):
    with np.load(path, allow_pickle=False) as saved:
        probe_images.append(saved["image"].copy())
        probe_poses.append(saved["pose"].copy())
        probe_names.append(
            json.loads(saved["metadata"].item())["scene"]
        )

assert len(probe_names) == 24
probe_images = np.stack(probe_images)
probe_poses = np.stack(probe_poses)

for method in learned:
    values = np.concatenate([
        predict(method, image[None], pose[None])
        for image, pose in zip(probe_images, probe_poses)
    ])
    expected = np.array([
        int(archived[method, name]["offset_deg"])
        for name in probe_names
    ])
    assert np.array_equal(
        choose(values, thresholds[method]), expected
    ), method

    if method in ridge:
        assert np.array_equal(
            ridge[method]["validation_scene"], probe_names
        )
        assert np.max(np.abs(
            values - ridge[method]["validation_prediction_s"]
        )) <= 1e-6

print(
    "PREDICTOR PASS | archived validation choices=24/24 "
    "for all five models | CPU FP32",
    flush=True,
)

choices = {
    method: np.zeros(80, dtype=int) for method in policies
}
values_by_model = {
    method: np.full((80, 2), np.nan) for method in learned
}
latencies = {
    method: np.zeros(80) for method in policies
}
eligible = np.zeros(80, dtype=bool)

for index, case in enumerate(cases):
    directory = root / "cases" / case["name"]
    eligible[index] = read(directory / "complete.json")["eligible"]
    if not eligible[index]:
        continue

    with np.load(directory / "input.npz", allow_pickle=False) as saved:
        image = saved["image"][None]
        pose = saved["pose"][None]
        assert image.shape == (1, 2, 7, 80, 80)
        assert pose.shape == (1, 2, 6)
        assert np.array_equal(saved["offset_deg"], [-45, 45])

    choices["FixedMinus45"][index] = -45
    choices["FixedPlus45"][index] = 45
    choices["HeadingRule"][index] = (
        -45 if np.cos(case["initial_yaw_rad"]) < 0 else 0
    )

    for method in learned:
        started = time.perf_counter()
        prediction = predict(method, image, pose)
        latencies[method][index] = time.perf_counter() - started

        assert prediction.shape == (1, 2)
        assert np.isfinite(prediction).all()
        values_by_model[method][index] = prediction[0]
        choices[method][index] = int(
            choose(prediction, thresholds[method])[0]
        )

print(
    "DECISIONS READY | 80 cases | frozen parameters only",
    flush=True,
)

penalty = float(protocol["max_sim_time_s"])
rows, executed, parity_episodes = [], set(), set()

for method in identity["branches"]:
    parity_episodes.add(
        read(root / "train_parity" / f"{method}.json")
        ["summary"]["directory"]
    )

for index, case in enumerate(cases):
    directory = root / "cases" / case["name"]
    branches = {
        offset: read(directory / f"{method}.json")
        for offset, method in (
            (0, "NoView"),
            (-45, "FixedMinus45"),
            (45, "FixedPlus45"),
        )
    }
    costs = {}

    for offset, record in branches.items():
        summary = record["summary"]
        assert summary["scene"] == case["name"]
        assert summary["freeze_id"] == frozen.name
        assert 0 <= summary["view_attempts"] <= 1
        assert 0 <= summary["completed_views"] <= 1

        safe = (
            summary["outcome"] == "GOAL"
            and summary["collision_attempts"] == 0
        )
        costs[offset] = (
            float(summary["sim_time_s"]) if safe else penalty
        )
        if record["origin"] == "executed":
            executed.add(summary["directory"])

    oracle = min(costs.values())

    for method in policies:
        offset = int(choices[method][index])
        record = branches[offset]
        summary = record["summary"]
        safe = (
            summary["outcome"] == "GOAL"
            and summary["collision_attempts"] == 0
        )
        gain = costs[0] - costs[offset]
        prediction = (
            values_by_model[method][index]
            if method in learned and eligible[index]
            else [None, None]
        )
        rows.append({
            "scene": case["name"],
            "source_group": case["source_group"],
            "kind": case["source_spec"]["kind"],
            "policy": method,
            "selected_offset_deg": offset,
            "eligible": bool(eligible[index]),
            "outcome": summary["outcome"],
            "safe_success": bool(safe),
            "sim_time_s": float(summary["sim_time_s"]),
            "penalized_time_s": costs[offset],
            "distance_m": float(summary["distance_m"]),
            "collision_attempts": int(summary["collision_attempts"]),
            "view_attempts": int(summary["view_attempts"]),
            "completed_views": int(summary["completed_views"]),
            "gain_vs_no_view_s": gain,
            "regret_s": costs[offset] - oracle,
            "harmful_look": bool(offset != 0 and gain < -1e-6),
            "predicted_minus45_s": (
                None if prediction[0] is None
                else float(prediction[0])
            ),
            "predicted_plus45_s": (
                None if prediction[1] is None
                else float(prediction[1])
            ),
            "threshold_s": thresholds.get(method),
            "prediction_wall_s": float(latencies[method][index]),
            "result_origin": (
                "executed_branch"
                if method in identity["branches"]
                and record["origin"] == "executed"
                else "derived_from_executed_branch"
            ),
            "reference_directory": summary["directory"],
        })

assert len(rows) == len({
    (row["scene"], row["policy"]) for row in rows
}) == 720

by_policy = {
    method: [row for row in rows if row["policy"] == method]
    for method in policies
}
policy_summary, source_rows = {}, []

for method, members in by_policy.items():
    assert [row["scene"] for row in members] == [
        case["name"] for case in cases
    ]
    safe = [row for row in members if row["safe_success"]]

    policy_summary[method] = {
        "policy": method,
        "safe_success": len(safe),
        "mean_penalized_time_s": float(np.mean([
            row["penalized_time_s"] for row in members
        ])),
        "mean_gain_vs_no_view_s": float(np.mean([
            row["gain_vs_no_view_s"] for row in members
        ])),
        "mean_regret_s": float(np.mean([
            row["regret_s"] for row in members
        ])),
        "selected_looks": sum(
            row["selected_offset_deg"] != 0 for row in members
        ),
        "completed_views": sum(
            row["completed_views"] for row in members
        ),
        "harmful_looks": sum(
            row["harmful_look"] for row in members
        ),
        "mean_distance_success_m": (
            float(np.mean([row["distance_m"] for row in safe]))
            if safe else None
        ),
        "mean_prediction_wall_ms_when_eligible": (
            float(latencies[method][eligible].mean() * 1000)
            if eligible.any() else None
        ),
    }

    for group in sources:
        group_rows = [
            row for row in members
            if row["source_group"] == group
        ]
        assert len(group_rows) == 4
        source_rows.append({
            "source_group": group,
            "policy": method,
            "mean_penalized_time_s": float(np.mean([
                row["penalized_time_s"] for row in group_rows
            ])),
            "safe_success": sum(
                row["safe_success"] for row in group_rows
            ),
        })

source_costs = {
    (row["policy"], row["source_group"]):
    row["mean_penalized_time_s"]
    for row in source_rows
}
rng = np.random.default_rng(
    protocol["statistics"]["bootstrap_seed"]
)
indices = rng.integers(
    0, 20,
    size=(protocol["statistics"]["bootstrap_samples"], 20),
)
paired = []

for method in policies:
    if method == "SpatialRidgeCV":
        continue

    delta = np.array([
        source_costs[method, group]
        - source_costs["SpatialRidgeCV", group]
        for group in sources
    ])
    low, high = np.quantile(
        delta[indices].mean(axis=1), [0.025, 0.975]
    )
    paired.append({
        "baseline": method,
        "mean_saved_s": float(delta.mean()),
        "ci_low_s": float(low),
        "ci_high_s": float(high),
        "source_wins": int((delta > 1e-6).sum()),
        "source_ties": int((np.abs(delta) <= 1e-6).sum()),
        "source_losses": int((delta < -1e-6).sum()),
    })

summary = {
    "freeze_id": frozen.name,
    "source_scenes": 20,
    "variants": 80,
    "policy_outcomes": 720,
    "collection_manifest_sha256": digest(
        root / "collection_manifest.json"
    ),
    "executed_test_episodes": len(executed),
    "train_parity_episodes": len(parity_episodes),
    "evaluation_mode": (
        "first-eligible frozen choices derived from "
        "deterministic executed action branches"
    ),
    "scope": protocol["test_scope"],
    "inference_device": "cpu",
    "torch_version": str(torch.__version__),
    "inference_timing": (
        "single-state two-candidate warm CPU prediction including "
        "feature/tensor preparation; excludes encoding and model loading"
    ),
    "inference_wall_time_in_primary_metric": False,
    "failure_penalty_s": penalty,
    "bootstrap_unit": "source_scene_with_four_variants",
    "bootstrap_seed": protocol["statistics"]["bootstrap_seed"],
    "bootstrap_samples": protocol["statistics"]["bootstrap_samples"],
    "intervals": (
        "paired percentile 95%; no multiple-comparison adjustment"
    ),
    "oracle": (
        "retrospective best of three executed action branches, "
        "not a deployable policy"
    ),
    "policies": policy_summary,
}

out.mkdir(parents=True, exist_ok=True)
csv_write(out / "episode_metrics.csv", rows)
csv_write(out / "source_metrics.csv", source_rows)
csv_write(
    out / "policy_summary.csv",
    list(policy_summary.values()),
)
csv_write(out / "paired_comparisons.csv", paired)
write(out / "paired_comparisons.json", paired)
write(out / "test_summary.json", summary)

np.savez_compressed(
    out / "predictions.npz",
    scene=np.array([case["name"] for case in cases]),
    eligible=eligible,
    metadata=json.dumps({
        "thresholds": thresholds,
        "device": "cpu",
        "precision": "ieee_fp32",
    }),
    **values_by_model,
)

os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    1, 2, figsize=(14, 6.5), layout="constrained"
)

for index, method in enumerate(policies):
    value = policy_summary[method]["mean_penalized_time_s"]
    axes[0].scatter(
        value, index,
        color="#1769e8" if method == "SpatialRidgeCV" else "#64748b",
        s=45,
    )
    axes[0].annotate(
        f"{value:.3f}", (value, index),
        xytext=(7, 0), textcoords="offset points", va="center",
    )

axes[0].set_yticks(range(len(policies)), policies)
axes[0].invert_yaxis()
axes[0].margins(x=0.22)
axes[0].set_xlabel(
    "Mean simulated time (s); failures penalized at 180 s"
)
axes[0].set_title("Frozen first-eligible policies")

for index, row in enumerate(paired):
    color = (
        "#1769e8" if row["ci_low_s"] > 0
        else "#c45a20" if row["ci_high_s"] < 0
        else "#64748b"
    )
    axes[1].plot(
        [row["ci_low_s"], row["ci_high_s"]],
        [index, index], color=color, lw=2,
    )
    axes[1].scatter(
        row["mean_saved_s"], index, color=color, s=35
    )

axes[1].set_yticks(
    range(len(paired)),
    [row["baseline"] for row in paired],
)
axes[1].invert_yaxis()
axes[1].axvline(0, color="black", linestyle="--", lw=1)
axes[1].set_xlabel(
    "Time saved by SpatialRidgeCV (s); positive favors spatial model"
)
axes[1].set_title("Paired 95% source bootstrap intervals")

for axis in axes:
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)

fig.suptitle("OBYM V3 | 20 fresh source scenes, 80 variants")
fig.savefig(out / "heldout_comparison.png", dpi=180)
fig.savefig(out / "heldout_comparison.pdf")
plt.close(fig)

files = [
    path for path in out.iterdir()
    if path.is_file() and path.name != "report_complete.json"
]
write(marker, {
    "protocol_sha256": digest(frozen / "protocol.json"),
    "case_hashes": case_hashes,
    "reporter_sha256": digest(Path(__file__)),
    "report_hashes": {
        path.name: digest(path) for path in files
    },
})

show(summary, paired)
print(
    f"FINAL REPORT READY | {len(executed)} executed test episodes | "
    "720 policy outcomes",
    flush=True,
)
print("OUTPUT |", out, flush=True)
