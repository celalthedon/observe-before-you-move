"""Input contract between the depth/occupancy mapper and V2 inference."""
from __future__ import annotations

import numpy as np

CHANNELS = (
    "free", "occupied", "unknown", "route_unknown", "candidate_position",
    "predicted_new_view", "valid_world",
)
PATCH_SHAPE = (7, 80, 80)
POSE_SHAPE = (6,)


def validate_encoded_candidates(images: np.ndarray, poses: np.ndarray) -> None:
    """Validate the exact V2 encoded-observation contract before inference."""
    if images.ndim != 4 or images.shape[1:] != PATCH_SHAPE:
        raise ValueError(f"expected N×{PATCH_SHAPE} image tensor; got {images.shape}")
    if poses.shape != (len(images), *POSE_SHAPE):
        raise ValueError(f"expected N×{POSE_SHAPE} pose tensor; got {poses.shape}")
    if not np.isin(images, (0, 1)).all() or not np.isfinite(poses).all():
        raise ValueError("V2 inputs must be finite binary map channels and finite poses")
