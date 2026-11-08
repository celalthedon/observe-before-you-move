"""CPU inference for the verified V2 ensemble, independent of notebook state."""
from __future__ import annotations

from pathlib import Path
import hashlib
import os
import numpy as np

from .encoding import validate_encoded_candidates


def _torch():
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from torch import nn
    torch.set_num_threads(1)  # preserves the Windows process-isolation setup
    return torch, nn


class _ModelFactory:
    @staticmethod
    def make():
        torch, nn = _torch()

        class TaskValueCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.map_encoder = nn.Sequential(
                    nn.Conv2d(7, 16, 3, 2, 1), nn.GroupNorm(4, 16), nn.SiLU(),
                    nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
                    nn.Conv2d(32, 48, 3, 2, 1), nn.GroupNorm(4, 48), nn.SiLU(),
                    nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                )
                self.pose_encoder = nn.Sequential(nn.Linear(6, 16), nn.SiLU())
                self.head = nn.Sequential(nn.Linear(784, 64), nn.SiLU(), nn.Dropout(0.15), nn.Linear(64, 1))
                nn.init.zeros_(self.head[-1].weight)
                nn.init.zeros_(self.head[-1].bias)

            def forward(self, image, pose):
                return self.head(torch.cat([self.map_encoder(image), self.pose_encoder(pose)], dim=1)).squeeze(1)
        return TaskValueCNN


class Ensemble:
    """Decoded-seconds V2 ensemble. Default device is CPU for portability."""
    def __init__(self, checkpoint_path: str | Path):
        torch, _ = _torch()
        self.path = Path(checkpoint_path)
        self.checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        if self.checkpoint["version"] != "task_value_v2_ensemble_mse_v1":
            raise ValueError("unexpected checkpoint version")
        model_type = _ModelFactory.make()
        self.models = []
        for member in self.checkpoint["members"]:
            model = model_type()
            model.load_state_dict(member["model_state_dict"], strict=True)
            model.eval()
            self.models.append(model)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def predict(self, images: np.ndarray, poses: np.ndarray, mode: str = "cnn") -> np.ndarray:
        validate_encoded_candidates(images, poses)
        center, scale = self.checkpoint["target_center_s"], self.checkpoint["target_scale_s"]
        if mode in ("linear", "pose"):
            baseline = self.checkpoint["linear_baseline" if mode == "linear" else "pose_baseline"]
            features = poses.astype(np.float64)
            if mode == "linear":
                features = np.concatenate([images.mean(axis=(2, 3)), features], axis=1)
            features = (features - np.asarray(baseline["mean"])) / np.asarray(baseline["std"])
            return center + scale * (np.column_stack([np.ones(len(images)), features]) @ np.asarray(baseline["coefficient"]))
        if mode != "cnn":
            raise ValueError(f"unknown prediction mode: {mode}")
        torch, _ = _torch()
        image = torch.tensor(images, dtype=torch.float32)
        pose = torch.tensor(poses, dtype=torch.float32)
        with torch.no_grad():
            values = [center + scale * model(image, pose).numpy().astype(np.float64) for model in self.models]
        return np.mean(values, axis=0)


def verify_probe(checkpoint: str | Path, probe: str | Path, atol: float = 1e-4) -> float:
    """Return maximum prediction error against the archived eight-sample probe."""
    with np.load(probe, allow_pickle=False) as saved:
        actual = Ensemble(checkpoint).predict(saved["image"], saved["pose"])
        expected = saved["expected_value_s"]
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, atol=atol, rtol=1e-5):
        raise AssertionError(f"inference probe mismatch: {error:.7g} s")
    return error
