
import sys
import json
import time
import hashlib
from pathlib import Path
import torch
from torch import nn
import numpy as np

torch.set_num_threads(1)

root, output = map(Path, sys.argv[1:3])
manifest = json.loads(
    (root / "manifest.json").read_text(encoding="utf-8")
)
config = {
    "seeds": [11, 23, 47],
    "epochs": 180,
    "patience": 30,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "minimum_value_s": 0.15,
    "loss": "MSE_standardized_signed_seconds",
    "ridge_alpha": 10.0,
}

arrays = {}
for split, count in [("train", 576), ("validation", 144)]:
    with np.load(root / f"{split}.npz", allow_pickle=False) as saved:
        arrays[split] = {
            key: saved[key].copy()
            for key in [
                "image", "pose", "target_time_saving_s", "valid",
                "scene", "source_group", "state_index", "offset_deg",
            ]
        }
        assert json.loads(saved["metadata"].item()) == manifest

    data = arrays[split]
    assert data["image"].shape == (count, 7, 80, 80)
    assert data["pose"].shape == (count, 6)
    assert data["valid"].all()
    assert np.isfinite(data["target_time_saving_s"]).all()

assert not (
    set(arrays["train"]["source_group"])
    & set(arrays["validation"]["source_group"])
)

y_train = arrays["train"]["target_time_saving_s"].astype(np.float64)
y_val = arrays["validation"]["target_time_saving_s"].astype(np.float64)
center = float(y_train.mean())
scale = max(float(y_train.std()), 1.0)
train_target = torch.tensor(
    (y_train - center) / scale, dtype=torch.float32
)

val_pairs = {}
for index, key in enumerate(zip(
    arrays["validation"]["scene"].tolist(),
    arrays["validation"]["state_index"].tolist(),
)):
    val_pairs.setdefault(key, []).append(index)

assert all(len(indices) == 2 for indices in val_pairs.values())


def metrics(prediction):
    chosen, regrets = [], []

    for indices in val_pairs.values():
        best = indices[int(np.argmax(prediction[indices]))]
        gain = (
            float(y_val[best])
            if prediction[best] > config["minimum_value_s"]
            else 0.0
        )
        chosen.append(gain)
        regrets.append(
            max(0.0, float(y_val[indices].max())) - gain
        )

    chosen = np.asarray(chosen)
    positive = y_val > 1e-6

    return {
        "mae_s": float(np.abs(prediction - y_val).mean()),
        "positive_mae_s": (
            float(np.abs(prediction[positive] - y_val[positive]).mean())
            if positive.any() else None
        ),
        "regret_s": float(np.mean(regrets)),
        "chosen_gain_s": float(chosen.mean()),
        "beneficial": int((chosen > 1e-6).sum()),
        "harmful": int((chosen < -1e-6).sum()),
        "states": len(chosen),
    }


def features(data, pose_only=False):
    pose = data["pose"].astype(np.float64)
    if pose_only:
        return pose
    return np.concatenate([
        data["image"].mean(axis=(2, 3)), pose,
    ], axis=1)


def ridge(pose_only=False):
    x = features(arrays["train"], pose_only)
    v = features(arrays["validation"], pose_only)

    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    x = np.column_stack([np.ones(len(x)), (x - mean) / std])
    v = np.column_stack([np.ones(len(v)), (v - mean) / std])

    penalty = np.eye(x.shape[1]) * config["ridge_alpha"]
    penalty[0, 0] = 0.0

    coefficient = np.linalg.solve(
        x.T @ x + penalty,
        x.T @ ((y_train - center) / scale),
    )
    prediction = center + scale * (v @ coefficient)

    return prediction, {
        "features": (
            "pose_only_6" if pose_only
            else "map_means_7_plus_pose_6"
        ),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coefficient": coefficient.tolist(),
        "alpha": config["ridge_alpha"],
    }


collisions = {}
for split, data in arrays.items():
    for index in range(len(data["image"])):
        digest = hashlib.sha256(
            data["image"][index].tobytes()
            + data["pose"][index].astype(np.float32).tobytes()
        ).hexdigest()

        collisions.setdefault(digest, []).append({
            "split": split,
            "scene": str(data["scene"][index]),
            "state_index": int(data["state_index"][index]),
            "offset_deg": int(data["offset_deg"][index]),
            "value_s": float(data["target_time_saving_s"][index]),
        })

collision_report = [
    {"input_sha256": digest, "members": members}
    for digest, members in collisions.items()
    if len(members) > 1
    and np.ptp([item["value_s"] for item in members]) > 1e-4
]
(output / "input_collision_report.json").write_text(
    json.dumps(collision_report, indent=2), encoding="utf-8"
)
print(
    "Eyni giriş / fərqli nəticə qrupları:",
    len(collision_report), flush=True,
)

linear_prediction, linear_model = ridge()
pose_prediction, pose_model = ridge(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cihaz:", device, "| PyTorch:", torch.__version__, flush=True)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("train:", len(y_train), "| validation:", len(y_val), flush=True)
print("Hədəf normallaşdırması:", center, scale, flush=True)
torch.backends.cudnn.benchmark = False


class TaskValueCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(7, 16, 3, 2, 1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1),
            nn.GroupNorm(4, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(6, 16), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(784, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image, pose):
        encoded = torch.cat([
            self.map_encoder(image),
            self.pose_encoder(pose),
        ], dim=1)
        return self.head(encoded).squeeze(1)


train_image = torch.tensor(
    arrays["train"]["image"], dtype=torch.float32
)
train_pose = torch.tensor(
    arrays["train"]["pose"], dtype=torch.float32
)
val_image = torch.tensor(
    arrays["validation"]["image"], dtype=torch.float32
)
val_pose = torch.tensor(
    arrays["validation"]["pose"], dtype=torch.float32
)


def predict(model):
    model.eval()
    parts = []

    with torch.no_grad():
        for start in range(0, len(val_image), 64):
            z = model(
                val_image[start:start + 64].to(device),
                val_pose[start:start + 64].to(device),
            )
            parts.append(z.cpu().numpy())

    result = (
        center + scale * np.concatenate(parts).astype(np.float64)
    )
    assert np.isfinite(result).all(), "Etibarsız proqnoz."
    return result


members, histories, member_predictions = [], {}, []
started = time.perf_counter()

for seed in config["seeds"]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = TaskValueCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_key, best_state, best_epoch, stale = None, None, 0, 0
    history = []

    for epoch in range(config["epochs"] + 1):
        losses = []

        if epoch:
            model.train()
            order = torch.randperm(len(train_image))

            for index in order.split(config["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                z = model(
                    train_image[index].to(device),
                    train_pose[index].to(device),
                )
                loss = torch.nn.functional.mse_loss(
                    z, train_target[index].to(device)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Təlim itkisi sonlu deyil.")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                losses.append(float(loss.item()))

        prediction = predict(model)
        report = metrics(prediction)
        history.append({
            "epoch": epoch,
            "train_mse": float(np.mean(losses)) if losses else None,
            **report,
        })

        key = (report["regret_s"], report["mae_s"])
        if best_key is None or key < best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1

        if epoch % 10 == 0:
            print(
                f"Seed {seed} | epoch={epoch:03d} | "
                f"MAE={report['mae_s']:.3f} s | "
                f"seçim itkisi={report['regret_s']:.3f} s | "
                f"faydalı/zərərli="
                f"{report['beneficial']}/{report['harmful']}",
                flush=True,
            )

        if stale >= config["patience"]:
            break

    model.load_state_dict(best_state)
    prediction = predict(model)
    member_predictions.append(prediction)

    member = {
        "seed": seed,
        "epoch": best_epoch,
        "model_state_dict": best_state,
        "validation_metrics": metrics(prediction),
    }
    members.append(member)
    histories[str(seed)] = history
    torch.save(member, output / f"seed_{seed}.pt")

    print(
        f"Seed {seed} tamamlandı | seçilmiş epoch={best_epoch}",
        flush=True,
    )
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

ensemble_prediction = np.mean(member_predictions, axis=0)

comparison = {
    name: metrics(prediction)
    for name, prediction in [
        ("NoView", np.zeros(len(y_val))),
        ("PoseOnly", pose_prediction),
        ("ValueLinear", linear_prediction),
        ("ValueCNN", ensemble_prediction),
    ]
}

checkpoint = {
    "version": "task_value_v2_ensemble_mse_v1",
    "architecture": "TaskValueCNN_v1",
    "members": members,
    "config": config,
    "target_center_s": center,
    "target_scale_s": scale,
    "ensemble_rule": "mean_decoded_seconds",
    "manifest": manifest,
    "dataset_sha256": {
        split: hashlib.sha256(
            (root / f"{split}.npz").read_bytes()
        ).hexdigest()
        for split in arrays
    },
    "linear_baseline": linear_model,
    "pose_baseline": pose_model,
    "validation_comparison": comparison,
    "torch_version": str(torch.__version__),
}

torch.save(checkpoint, output / "ensemble.pt")

np.savez_compressed(
    output / "inference_probe.npz",
    image=arrays["validation"]["image"][:8],
    pose=arrays["validation"]["pose"][:8],
    expected_value_s=ensemble_prediction[:8],
)
np.savez_compressed(
    output / "validation_predictions.npz",
    target_s=y_val,
    cnn_s=ensemble_prediction,
    linear_s=linear_prediction,
    pose_only_s=pose_prediction,
    members_s=np.stack(member_predictions),
    scene=arrays["validation"]["scene"],
    state_index=arrays["validation"]["state_index"],
    offset_deg=arrays["validation"]["offset_deg"],
)

for name, payload in [
    ("history.json", histories),
    ("validation_metrics.json", comparison),
    ("training_config.json", config),
]:
    (output / name).write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

print("\nVALIDATION — OFFLINE QƏRAR MÜQAYİSƏSİ", flush=True)
for name, report in comparison.items():
    print(
        f"{name} | MAE={report['mae_s']:.3f} s | "
        f"seçim itkisi={report['regret_s']:.3f} s | "
        f"orta seçilmiş qazanc={report['chosen_gain_s']:+.3f} s | "
        f"faydalı/zərərli={report['beneficial']}/{report['harmful']}",
        flush=True,
    )

print(
    f"Müddət: {(time.perf_counter() - started) / 60:.1f} dəqiqə",
    flush=True,
)
print("Model saxlanıldı:", output / "ensemble.pt", flush=True)
