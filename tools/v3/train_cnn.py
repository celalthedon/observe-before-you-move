
"""Portable worker adapted from notebook cell 79. Use the train-cnn CLI."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

source_root, manifest_path, root, out = map(Path, sys.argv[1:5])
requested_device = sys.argv[5]
sys.path.insert(0, str(source_root / "src"))

from obym.inference import _ModelFactory

seeds = [11, 23, 47]
epochs = 60
minimum_gain = 0.15
rank_weight = 0.5
ridge_alpha = 10.0
device = torch.device(
    ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "auto" else requested_device
)
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but not available")
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def load(split):
    images, poses, gains, names, groups = [], [], [], [], []
    for path in sorted((root / split).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            if meta["status"] != "valid" or data["offset_deg"].tolist() != [-45, 45]:
                raise ValueError(path.name)
            images.append(data["image"])
            poses.append(data["pose"])
            gains.append(data["target_time_saving_s"])
            names.append(meta["scene"])
            groups.append(meta["source_group"])
    return np.stack(images), np.stack(poses), np.stack(gains), names, groups

x, p, y, _, train_groups = load("train")
xv, pv, yv, names, groups = load("validation")
assert (len(y), len(yv)) == (96, 24)
assert set(train_groups).isdisjoint(groups)

def features(images, poses, use_map):
    if not use_map:
        return poses
    return np.concatenate([images.mean(axis=(-2, -1)), poses], axis=-1)

def ridge(use_map):
    train = features(x, p, use_map).reshape(y.size, -1).astype(np.float64)
    val = features(xv, pv, use_map).reshape(yv.size, -1).astype(np.float64)
    mean = train.mean(axis=0)
    std = train.std(axis=0).clip(1e-6)
    train = (train - mean) / std
    val = (val - mean) / std
    center = float(y.mean())
    beta = np.linalg.solve(
        train.T @ train + ridge_alpha * np.eye(train.shape[1]),
        train.T @ (y.ravel() - center),
    )
    return (center + val @ beta).reshape(-1, 2), {
        "mean": mean.tolist(), "std": std.tolist(),
        "coefficient": beta.tolist(), "center_s": center,
    }

scale = max(float(y.std()), 1.0)
images, poses, target = (
    torch.as_tensor(a, dtype=torch.float32, device=device) for a in (x, p, y / scale)
)
val_images, val_poses = (
    torch.as_tensor(a, dtype=torch.float32, device=device) for a in (xv, pv)
)
model_type = _ModelFactory.make()

def score(model, images, poses):
    return model(images.flatten(0, 1), poses.flatten(0, 1)).reshape(-1, 2)

def ranking_loss(pred, truth):
    pred = F.pad(pred, (1, 0))
    truth = F.pad(truth, (1, 0))
    pairs_a, pairs_b = [0, 0, 1], [1, 2, 2]
    gap = truth[:, pairs_a] - truth[:, pairs_b]
    predicted_gap = pred[:, pairs_a] - pred[:, pairs_b]
    return (gap.abs() * F.softplus(-gap.sign() * predicted_gap)).mean()

print(f"DEVICE | {device} | train={len(y)} | validation={len(yv)}", flush=True)
out.mkdir(parents=True, exist_ok=False)
predictions, models = {}, {}

for name, weight in (("CNN_MSE", 0.0), ("CNN_MSE_Rank", rank_weight)):
    members, values = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        model = model_type().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)

        for _ in range(epochs):
            for idx in torch.randperm(len(target), device=device).split(16):
                pred = score(model, images[idx], poses[idx])
                loss = F.mse_loss(pred, target[idx])
                if weight:
                    loss = loss + weight * ranking_loss(pred, target[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()

        model.eval()
        with torch.inference_mode():
            values.append(score(model, val_images, val_poses).cpu().numpy() * scale)
        members.append({"seed": seed, "state_dict": model.cpu().state_dict()})
        print(f"FIT | {name} | seed={seed} | epochs={epochs}", flush=True)

    predictions[name] = np.mean(values, axis=0)
    models[name] = members

ridge_models = {}
for name, use_map in (("PoseRidge", False), ("MapRidge", True)):
    predictions[name], ridge_models[name] = ridge(use_map)

actual = np.column_stack([np.zeros(len(yv)), yv])
oracle = actual.max(axis=1)
actions = {
    name: np.full(len(yv), i)
    for i, name in enumerate(("NoView", "FixedMinus45", "FixedPlus45"))
}

manifest = json.loads(manifest_path.read_bytes())
headings = {case["name"]: case["initial_yaw_rad"] for case in manifest["cases"]}
actions["HeadingRule"] = np.array([int(np.cos(headings[name]) < 0) for name in names])

for name, pred in predictions.items():
    assert np.isfinite(pred).all(), name
    actions[name] = np.where(pred.max(axis=1) > minimum_gain, pred.argmax(axis=1) + 1, 0)

print(f"ORACLE | gain={oracle.mean():+.3f} s", flush=True)
stats = {}
for name, selected in actions.items():
    gains = actual[np.arange(len(yv)), selected]
    stats[name] = {
        "gain_s": float(gains.mean()),
        "regret_s": float((oracle - gains).mean()),
        "harmful": int((gains < -1e-6).sum()),
        "looks": int((selected != 0).sum()),
    }
    row = stats[name]
    print(
        f"VAL | {name:14} | gain={row['gain_s']:+.3f} s"
        f" | regret={row['regret_s']:.3f} s"
        f" | harmful={row['harmful']} | looks={row['looks']}",
        flush=True,
    )

config = {
    "version": "v3_initial_pilot",
    "epochs": epochs,
    "seeds": seeds,
    "minimum_gain_s": minimum_gain,
    "rank_weight": rank_weight,
    "ridge_alpha": ridge_alpha,
    "target_scale_s": scale,
    "protocol": json.loads((root / "protocol.json").read_text(encoding="utf-8")),
}
torch.save({"config": config, "cnn_members": models, "ridge": ridge_models}, out / "models.pt")
(out / "summary.json").write_text(
    json.dumps({"config": config, "validation": stats}, indent=2), encoding="utf-8"
)

with (out / "validation.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["scene", "source_group", "policy", "offset_deg", "gain_s", "regret_s"])
    for name, selected in actions.items():
        for i, action in enumerate(selected):
            gain = actual[i, action]
            writer.writerow([names[i], groups[i], name, (0, -45, 45)[action], gain, oracle[i] - gain])

print(f"V3 TRAIN READY | {out}", flush=True)
