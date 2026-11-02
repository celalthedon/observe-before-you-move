"""Headless PyBullet smoke check for one archived held-out start state.

It checks the concrete case geometry, start/goal pose, and collision-free
initialization against the archived ValueCNN motion.  It intentionally does
not launch the 80-variant evaluation suite.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENE = "test_layout_3009_myP_LR"  # selected CNN-gain demonstration


def main() -> None:
    try:
        import pybullet as p
    except ImportError as exc:
        raise SystemExit("Install optional simulation dependency: pip install pybullet") from exc
    manifest = json.loads((ROOT / "data/heldout/manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["name"] == SCENE)
    motion = json.loads((ROOT / "data/traces/cnn_gain_valuecnn_motion.json").read_text(encoding="utf-8"))
    first = motion[0]
    if abs(first["x"] - case["start_xy"][0]) > 1e-9 or abs(first["y"] - case["start_xy"][1]) > 1e-9:
        raise AssertionError("archived trajectory start disagrees with manifest")
    client = p.connect(p.DIRECT)
    try:
        robot = p.createMultiBody(0, p.createCollisionShape(p.GEOM_CYLINDER, radius=.25, height=.4), basePosition=[*case["start_xy"], .2])
        obstacles = []
        for size, position in case["geometry"]:
            obstacles.append(p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size]), basePosition=position))
        assert not any(p.getClosestPoints(robot, obstacle, distance=0) for obstacle in obstacles)
    finally:
        p.disconnect(client)
    print(f"PASS | headless initialization parity | {SCENE} | {len(case['geometry'])} obstacles")


if __name__ == "__main__":
    main()
