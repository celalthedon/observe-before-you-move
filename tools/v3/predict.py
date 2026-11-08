"""Replay frozen prediction probes against the archived validation and test cases."""
import json
import sys

import numpy as np

from .common import (
    ACTIONS, CNNS, MODELS, RIDGES, choose, features, load_validation, read_csv,
    read_json, require, verify_hashes, write_json,
)
from .report import archive_predictions

RIDGE_ATOL = 1e-6
CNN_ATOL = 1e-4


class Predictors:
    def __init__(self, bundle, *, device="cpu"):
        import torch

        sys.path.insert(0, str(bundle.source / "src"))
        from obym.inference import _ModelFactory

        self.torch = torch
        self.device = torch.device(device)
        require(device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
        torch.set_num_threads(1)
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = "ieee"
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        else:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        checkpoint = torch.load(bundle.frozen / "cnn_models.pt", map_location="cpu", weights_only=True)
        self.scale = float(checkpoint["config"]["target_scale_s"])
        require(np.isfinite(self.scale) and self.scale > 0, "Invalid target scale")
        self.ridges, self.cnns, self.thresholds = {}, {}, {}
        for method in RIDGES:
            with np.load(bundle.frozen / f"{method}.npz", allow_pickle=False) as saved:
                self.ridges[method] = dict(saved)
                self.thresholds[method] = float(saved["threshold_s"].item())

        model_type = _ModelFactory.make()
        for method in CNNS:
            members = checkpoint["cnn_members"][method]
            require(sorted(member["seed"] for member in members) == [11, 23, 47], f"Unexpected seeds: {method}")
            self.cnns[method] = []
            for member in members:
                model = model_type()
                model.load_state_dict(member["state_dict"], strict=True)
                self.cnns[method].append(model.to(self.device).eval())
            self.thresholds[method] = float(checkpoint["config"]["minimum_gain_s"])

    def predict(self, method, images, poses):
        if method in RIDGES:
            saved = self.ridges[method]
            x = features(images, poses, int(saved["level"].item()))
            return ((x - saved["mean"]) / saved["std"]) @ saved["coefficient"] + saved["target_center_s"]

        torch = self.torch
        x = torch.as_tensor(images, dtype=torch.float32, device=self.device).flatten(0, 1)
        z = torch.as_tensor(poses, dtype=torch.float32, device=self.device).flatten(0, 1)
        with torch.inference_mode():
            values = [
                model(x, z).cpu().numpy().reshape(-1, 2).astype(np.float64) * self.scale
                for model in self.cnns[method]
            ]
        return np.mean(values, axis=0)


def check_offsets(values, threshold, expected, context):
    require(
        np.isfinite(values).all() and np.array_equal(choose(values, threshold), expected),
        f"Archived decisions differ: {context}",
    )


def max_difference(values, expected, tolerance, context):
    require(values.shape == expected.shape, f"Shape mismatch: {context}")
    difference = float(np.max(np.abs(values - expected))) if values.size else 0.0
    require(difference <= tolerance, f"Prediction difference {difference:.8g} s exceeds {tolerance:g} s: {context}")
    return difference


def validation(bundle, models, *, cuda_models=None):
    dataset_manifest = read_json(bundle.frozen / "dataset_manifest.json")
    expected_names = [case["name"] for case in dataset_manifest["cases"] if case["split"] == "validation"]
    images, poses, targets, names = load_validation(bundle.dataset / "validation", expected_names)
    rows = read_csv(bundle.frozen / "validation_decisions.csv")
    decisions = {(row["policy"], row["scene"]): row for row in rows}
    require(len(decisions) == len(rows), "Duplicate archived validation decisions")
    for method, column in (("FixedMinus45", 0), ("FixedPlus45", 1)):
        expected = np.array([float(decisions[method, name]["gain_s"]) for name in names])
        max_difference(targets[:, column], expected, 1e-4, f"{method} validation targets")

    checks = {}
    for method in MODELS:
        # The final reporter evaluated one state at a time, with two candidate views.
        values = np.concatenate([
            models.predict(method, image[None], pose[None])
            for image, pose in zip(images, poses, strict=True)
        ])
        threshold = models.thresholds[method]
        expected = np.array([int(decisions[method, name]["offset_deg"]) for name in names])
        check_offsets(values, threshold, expected, f"{method} validation")
        result = {"minimum_gain_s": threshold, "archived_decisions_matched": len(names)}
        if method in RIDGES:
            saved = models.ridges[method]
            require(np.array_equal(saved["validation_scene"], names), f"Validation scene order: {method}")
            result["probe_max_difference_s"] = max_difference(
                values, saved["validation_prediction_s"], RIDGE_ATOL, f"{method} validation"
            )
        else:
            result["target_scale_s"] = models.scale
            if cuda_models:
                cuda_values = cuda_models.predict(method, images, poses)
                cpu_values = models.predict(method, images, poses)
                check_offsets(cuda_values, threshold, expected, f"{method} CUDA validation")
                result["cpu_cuda_max_difference_s"] = max_difference(
                    cuda_values, cpu_values, CNN_ATOL, f"{method} CPU/CUDA validation"
                )
        checks[method] = result
    return checks


def replay(bundle, output):
    protocol = bundle.protocol(verify_source=True)
    manifest = bundle.manifest(protocol)
    cases, _, marker = bundle.checked_cases(manifest)
    reports = bundle.final / "reports"
    names = ("predictions.npz", "episode_metrics.csv")
    verify_hashes(reports, {name: marker["report_hashes"][name] for name in names})
    eligible, thresholds, archived = archive_predictions(reports, cases)
    models = Predictors(bundle)
    require(models.thresholds == thresholds, "Frozen and archived thresholds differ")
    checks = validation(bundle, models)

    values = {method: np.full((len(cases), 2), np.nan) for method in MODELS}
    for index, (_, directory, complete, _) in enumerate(cases):
        if not eligible[index]:
            continue
        with np.load(directory / "input.npz", allow_pickle=False) as saved:
            image, pose = saved["image"], saved["pose"]
            require(image.shape == (2, 7, 80, 80) and pose.shape == (2, 6), f"Input shape: {directory}")
            require(np.array_equal(saved["offset_deg"], ACTIONS), f"Action order: {directory}")
            require(saved["state_index"].item() == complete["state_index"], f"Decision state: {directory}")
        for method in MODELS:
            values[method][index] = models.predict(method, image[None], pose[None])[0]

    rows = read_csv(reports / "episode_metrics.csv")
    decisions = {(row["policy"], row["scene"]): int(row["selected_offset_deg"]) for row in rows}
    for method in MODELS:
        difference = max_difference(
            values[method][eligible], archived[method][eligible],
            RIDGE_ATOL if method in RIDGES else CNN_ATOL,
            f"{method} held-out",
        )
        offsets = np.zeros(len(cases), dtype=int)
        offsets[eligible] = choose(values[method][eligible], thresholds[method])
        expected = [decisions[method, case["name"]] for case, *_ in cases]
        require(np.array_equal(offsets, expected), f"Archived held-out decisions differ: {method}")
        checks[method]["heldout_decisions_matched"] = len(cases)
        checks[method]["heldout_max_difference_s"] = difference
        print(f"{method}: 24/24 validation and 80/80 held-out decisions; max held-out difference {difference:.8g} s")

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "predictions.npz",
        scene=np.array([case["name"] for case, *_ in cases]),
        eligible=eligible,
        metadata=json.dumps({"thresholds": thresholds, "device": "cpu", "precision": "ieee_fp32"}),
        **values,
    )
    write_json(output / "prediction_parity.json", {
        "torch_version": str(models.torch.__version__),
        "ridge_atol_s": RIDGE_ATOL,
        "cnn_atol_s": CNN_ATOL,
        "models": checks,
    })
    print(f"CPU prediction parity passed; wrote {output}")
