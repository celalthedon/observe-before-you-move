"""The one-look decision rule used by the held-out ValueCNN policy.

The simulator supplies two encoded candidates at -45 and +45 degrees.  The
model predicts the net simulated seconds saved by each candidate.  A look is
made only when the best decoded prediction clears the recorded threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    selected_offset_deg: int | None = None
    predicted_gain_s: float | None = None


def choose_one_look(
    offsets_deg: Sequence[int], predicted_gain_s: Sequence[float], *,
    minimum_value_s: float = 0.15, optional_look_used: bool = False,
) -> Decision:
    """Decode predictions into the recorded one-optional-look action.

    Positive predicted gain means the optional turn/sense action is predicted
    to reduce *simulated navigation time*, including its movement and sensing
    costs. This function deliberately does not account for inference wall
    time: that was logged separately in the experiment.
    """
    if optional_look_used:
        return Decision("NAVIGATE", "VIEW_BUDGET")
    if len(offsets_deg) != len(predicted_gain_s) or not offsets_deg:
        raise ValueError("offsets and predictions must be non-empty and aligned")
    best = max(range(len(offsets_deg)), key=lambda i: predicted_gain_s[i])
    value = float(predicted_gain_s[best])
    if value > minimum_value_s:
        return Decision("OBSERVE", "PREDICTED_NET_TIME_GAIN", int(offsets_deg[best]), value)
    return Decision("NAVIGATE", "LOW_VALUE", None, value)
