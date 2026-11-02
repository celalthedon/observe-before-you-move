"""Private pipe worker: isolated CPU inference, no network service required."""
import base64
import io
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from obym.inference import Ensemble


def main():
    folder = Path(sys.argv[1])
    model = Ensemble(folder / "ensemble.pt")
    with np.load(folder / "inference_probe.npz", allow_pickle=False) as saved:
        images, poses, expected_cnn = saved["image"], saved["pose"], saved["expected_value_s"]
    with np.load(folder / "validation_predictions.npz", allow_pickle=False) as saved:
        references = {"cnn": expected_cnn, "linear": saved["linear_s"][:len(images)], "pose": saved["pose_only_s"][:len(images)]}
    differences = {}
    for mode, expected in references.items():
        actual = model.predict(images, poses, mode)
        if not np.allclose(actual, expected, atol=1e-4, rtol=1e-5):
            raise AssertionError(f"{mode} probe mismatch")
        differences[mode] = float(np.max(np.abs(actual - expected)))
    print(json.dumps({"checkpoint_sha256": model.sha256, "minimum_value_s": model.checkpoint["config"]["minimum_value_s"], "probe_max_difference_s": differences}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with np.load(io.BytesIO(base64.b64decode(request["npz"])), allow_pickle=False) as saved:
                values = model.predict(saved["image"], saved["pose"], request["mode"])
            print(json.dumps({"value_s": values.tolist()}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        raise
