"""Keep PyTorch out of the simulator process, as in the Windows notebooks."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


class IsolatedPredictor:
    def __init__(self, model_folder: Path):
        env = os.environ.copy()
        env.pop("KMP_DUPLICATE_LIB_OK", None)
        env.update(MKL_THREADING_LAYER="SEQUENTIAL", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
        worker = Path(__file__).resolve().parents[2] / "tools/inference_worker.py"
        self.process = subprocess.Popen([sys.executable, "-u", str(worker), str(model_folder.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            self.health = json.loads(self.process.stdout.readline())
            if "error" in self.health:
                raise RuntimeError(self.health["error"])
        except Exception:
            self.close()
            raise

    def predict(self, images, poses, mode="cnn"):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, image=images, pose=poses)
        request = {"mode": mode, "npz": base64.b64encode(buffer.getvalue()).decode("ascii")}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return np.asarray(response["value_s"], dtype=np.float64)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
