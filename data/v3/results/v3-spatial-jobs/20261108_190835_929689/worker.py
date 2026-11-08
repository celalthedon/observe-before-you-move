
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1])
data = repo / "results/v3-initial-dataset"
previous = repo / "results/v3-training/20261107_014522_557567"
output = repo / "results/v3-spatial-ridge" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def load(split, expected, expected_groups):
    images, poses, targets, names, groups = [], [], [], [], []
    paths = sorted((data / split).glob("*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(saved["metadata"].item())
            assert meta["status"] == "valid" and meta["split"] == split
            assert saved["image"].shape == (2, 7, 80, 80)
            assert saved["pose"].shape == (2, 6)
            assert np.array_equal(saved["offset_deg"], [-45, 45])
            assert saved["valid"].shape == (2,) and saved["valid"].all()
            assert saved["target_time_saving_s"].shape == (2,)
            assert np.isin(saved["image"], [0, 1]).all()
            images.append(saved["image"].copy())
            poses.append(saved["pose"].copy())
            targets.append(saved["target_time_saving_s"].copy())
            names.append(meta["scene"])
            groups.append(meta["source_group"])
    assert len(paths) == expected and len(set(groups)) == expected_groups
    assert len(set(names)) == len(names)
    result = tuple(map(np.stack, (images, poses, targets))) + (np.array(names), np.array(groups))
    assert all(np.isfinite(array).all() for array in result[:3])
    assert all(np.count_nonzero(result[4] == group) == 4 for group in set(groups))
    return result

def features(images, poses, level):
    parts = [poses.astype(np.float64)]
    for size in (1, 2, 4, 8):
        if size <= level:
            pooled = images.reshape(
                len(images), 2, 7, size, 80 // size, size, 80 // size
            ).mean(axis=(4, 6))
            parts.append(pooled.reshape(len(images), 2, -1))
    return np.concatenate(parts, axis=2)

def decompose(x, y):
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1)
    center, spread = x.mean(axis=0), x.std(axis=0)
    spread = np.where(spread > 1e-8, spread, 1.0)
    u, singular, vt = np.linalg.svd(
        (x - center) / spread, full_matrices=False
    )
    return center, spread, float(y.mean()), singular, vt, u.T @ (y - y.mean())

def model(parts, alpha):
    center, spread, target_center, singular, vt, projected = parts
    coefficient = vt.T @ (singular / (singular ** 2 + alpha) * projected)
    return center, spread, target_center, coefficient

def predict(fitted, x):
    center, spread, target_center, coefficient = fitted
    return ((x - center) / spread) @ coefficient + target_center

def decide(predictions, threshold):
    return np.where(
        predictions.max(axis=1) > threshold,
        predictions.argmax(axis=1) + 1,
        0,
    )

def metrics(y, choice, groups):
    values = np.column_stack([np.zeros(len(y)), y])
    gains = values[np.arange(len(y)), choice]
    regret = values.max(axis=1) - gains
    grouped = lambda array: float(np.mean([
        array[groups == group].mean() for group in np.unique(groups)
    ]))
    return {
        "gain_s": grouped(gains),
        "regret_s": grouped(regret),
        "harmful": int((gains < -1e-6).sum()),
        "looks": int((choice != 0).sum()),
    }

train_images, train_poses, train_y, train_names, train_groups = load("train", 96, 24)
protocol_path = data / "protocol.json"
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
assert protocol["decision"] == "first_eligible"
assert protocol["angles_deg"] == [-45, 45]

source_groups = np.unique(train_groups)
np.random.default_rng(20261108).shuffle(source_groups)
folds = [
    np.isin(train_groups, block)
    for block in np.array_split(source_groups, 6)
]

levels = {
    "PoseRidgeCV": [0],
    "MeanMapRidgeCV": [1],
    "SpatialRidgeCV": [2, 4, 8],
}
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
thresholds = [0.0, 0.15, 0.5]
search, selected = [], {}

print("TRAIN CV | 24 source groups | 6 folds", flush=True)
print("Validation parametr seçimində istifadə edilmir.", flush=True)

for family, options in levels.items():
    candidates = []
    for level in options:
        x = features(train_images, train_poses, level)
        oof = {
            alpha: np.full(train_y.shape, np.nan)
            for alpha in alphas
        }
        for held in folds:
            assert not set(train_groups[held]) & set(train_groups[~held])
            parts = decompose(x[~held], train_y[~held])
            for alpha in alphas:
                oof[alpha][held] = predict(model(parts, alpha), x[held])
        for alpha in alphas:
            assert np.isfinite(oof[alpha]).all()
            for threshold in thresholds:
                row = {
                    "family": family,
                    "level": level,
                    "alpha": alpha,
                    "threshold_s": threshold,
                    **metrics(
                        train_y,
                        decide(oof[alpha], threshold),
                        train_groups,
                    ),
                }
                candidates.append(row)
                search.append(row)

    selected[family] = min(
        candidates,
        key=lambda row: (
            row["regret_s"], row["harmful"], row["level"],
            -row["alpha"], -row["threshold_s"],
        ),
    )
    row = selected[family]
    print(
        f"OOF | {family} | gain={row['gain_s']:+.3f} s | "
        f"regret={row['regret_s']:.3f} s | level={row['level']} | "
        f"alpha={row['alpha']} | threshold={row['threshold_s']}",
        flush=True,
    )

output.mkdir(parents=True, exist_ok=False)
selection = {
    "seed": 20261108,
    "fold_groups": [
        source_groups[i:i + 4].tolist() for i in range(0, 24, 4)
    ],
    "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    "train_sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((data / "train").glob("*.npz"))
    },
    "search": search,
    "selected": selected,
}
(output / "train_selection.json").write_text(
    json.dumps(selection, indent=2, allow_nan=False),
    encoding="utf-8",
)

val_images, val_poses, val_y, val_names, val_groups = load("validation", 24, 6)
assert not set(train_groups) & set(val_groups)

predictions = {}
for family, row in selected.items():
    x = features(train_images, train_poses, row["level"])
    fitted = model(decompose(x, train_y), row["alpha"])
    predictions[family] = predict(
        fitted, features(val_images, val_poses, row["level"])
    )
    np.savez_compressed(
        output / f"{family}.npz",
        mean=fitted[0],
        std=fitted[1],
        target_center_s=fitted[2],
        coefficient=fitted[3],
        level=row["level"],
        threshold_s=row["threshold_s"],
        validation_prediction_s=predictions[family],
        validation_scene=val_names,
    )

choices = {
    "NoView": np.zeros(len(val_y), dtype=int),
    "FixedMinus45": np.ones(len(val_y), dtype=int),
    "FixedPlus45": np.full(len(val_y), 2, dtype=int),
}
for family, prediction in predictions.items():
    choices[family] = decide(prediction, selected[family]["threshold_s"])

old_csv = previous / "validation.csv"
if old_csv.is_file():
    with old_csv.open(encoding="utf-8", newline="") as handle:
        archived = list(csv.DictReader(handle))
    values = np.column_stack([np.zeros(len(val_y)), val_y])

    for policy in ("CNN_MSE", "CNN_MSE_Rank", "PoseRidge", "MapRidge"):
        matches = [row for row in archived if row["policy"] == policy]
        assert len(matches) == len(val_names)
        mapping = {row["scene"]: row for row in matches}
        assert set(mapping) == set(val_names)
        choice = np.array([
            {0: 0, -45: 1, 45: 2}[int(mapping[name]["offset_deg"])]
            for name in val_names
        ])
        assert all(
            mapping[name]["source_group"] == group
            for name, group in zip(val_names, val_groups)
        )
        assert np.allclose(
            values[np.arange(len(val_y)), choice],
            [float(mapping[name]["gain_s"]) for name in val_names],
            atol=1e-4,
            rtol=0,
        )
        choices[policy] = choice
else:
    print("Əvvəlki validation.csv tapılmadı; CNN müqayisəsi ötürüldü.", flush=True)

report, rows = {}, []
for policy, choice in choices.items():
    report[policy] = metrics(val_y, choice, val_groups)
    stat = report[policy]
    print(
        f"VAL | {policy:18s} | gain={stat['gain_s']:+.3f} s | "
        f"regret={stat['regret_s']:.3f} s | "
        f"harmful={stat['harmful']} | looks={stat['looks']}",
        flush=True,
    )
    for index, action in enumerate(choice):
        gain = float(np.r_[0.0, val_y[index]][action])
        rows.append({
            "scene": str(val_names[index]),
            "source_group": str(val_groups[index]),
            "policy": policy,
            "offset_deg": [0, -45, 45][action],
            "gain_s": gain,
        })

with (output / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

(output / "summary.json").write_text(
    json.dumps(report, indent=2, allow_nan=False),
    encoding="utf-8",
)
print("SPATIAL COMPARISON READY |", output, flush=True)
