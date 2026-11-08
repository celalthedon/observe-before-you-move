"""Paths and integrity checks for the frozen V3 experiment."""
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

FREEZE_ID = "94fd47e48b4f"
REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCHES = {0: "NoView", -45: "FixedMinus45", 45: "FixedPlus45"}
RIDGES = ("SpatialRidgeCV", "PoseRidgeCV", "MeanMapRidgeCV")
CNNS = ("CNN_MSE", "CNN_MSE_Rank")
MODELS = RIDGES + CNNS
ACTIONS = np.array([-45, 45])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Invalid archive member: {name}")
        path = directory / relative
        require(digest(path) == expected, f"Hash mismatch: {path}")


def episode_path(reference):
    """Translate a recorded Windows or POSIX path to a bundle-relative episode."""
    parts = PurePosixPath(reference.replace("\\", "/")).parts
    require(
        len(parts) >= 3 and parts[-3:-1] == ("results", "day3") and parts[-1] != "..",
        f"Invalid episode directory: {reference}",
    )
    return Path(*parts[-3:])


def features(images, poses, level):
    parts = [poses.astype(np.float64)]
    for size in (1, 2, 4, 8):
        if size <= level:
            pooled = images.reshape(len(images), 2, 7, size, 80 // size, size, 80 // size)
            parts.append(pooled.mean(axis=(4, 6)).reshape(len(images), 2, -1))
    return np.concatenate(parts, axis=2)


def choose(values, threshold):
    best = values.argmax(axis=1)
    return np.where(values[np.arange(len(values)), best] > threshold, ACTIONS[best], 0)


def load_validation(directory, names_expected):
    images, poses, targets, names = [], [], [], []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(saved["metadata"].item())
            require(meta["status"] == "valid" and meta["split"] == "validation", str(path))
            require(np.array_equal(saved["offset_deg"], ACTIONS), f"Action order: {path}")
            require(saved["valid"].shape == (2,) and saved["valid"].all(), str(path))
            require(saved["image"].shape == (2, 7, 80, 80), f"Image shape: {path}")
            require(saved["pose"].shape == (2, 6), f"Pose shape: {path}")
            require(saved["target_time_saving_s"].shape == (2,), f"Target shape: {path}")
            images.append(saved["image"])
            poses.append(saved["pose"])
            targets.append(saved["target_time_saving_s"])
            names.append(meta["scene"])
    require(len(names) == len(set(names)) == 24, "Expected 24 distinct validation scenes")
    require(set(names) == set(names_expected), "Unexpected validation scene names")
    arrays = tuple(np.stack(values) for values in (images, poses, targets))
    require(all(np.isfinite(array).all() for array in arrays), "Non-finite validation data")
    return *arrays, names


def verify_dataset_layout(dataset, frozen):
    require(
        read_json(dataset / "protocol.json") == read_json(frozen / "collection_protocol.json"),
        "Dataset collection protocol differs from the frozen experiment",
    )
    cases = read_json(frozen / "dataset_manifest.json")["cases"]
    for split in ("train", "validation"):
        expected = {f"{case['name']}.npz" for case in cases if case["split"] == split}
        actual = {path.name for path in (dataset / split).glob("*.npz")}
        require(actual == expected, f"Unexpected {split} NPZ inventory in {dataset}")


@dataclass(frozen=True)
class Bundle:
    root: Path
    source: Path

    @property
    def frozen(self):
        return self.root / "results/v3-final-protocol" / FREEZE_ID

    @property
    def final(self):
        return self.root / "results/v3-final-test" / FREEZE_ID

    @property
    def dataset(self):
        return self.root / "results/v3-initial-dataset"

    @property
    def training(self):
        return self.root / "results/v3-training/20261107_014522_557567"

    @property
    def manifest_path(self):
        return self.frozen / "preflight/test_manifest.json"

    def protocol(self, *, verify_source=False):
        protocol = read_json(self.frozen / "protocol.json")
        encoded = json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
        require(hashlib.sha256(encoded).hexdigest()[:12] == FREEZE_ID, "Protocol freeze ID mismatch")
        verify_hashes(self.frozen, protocol["artifact_hashes"])
        if verify_source:
            verify_hashes(self.source, protocol["source_hashes"])
        return protocol

    def manifest(self, protocol):
        manifest = read_json(self.manifest_path)
        require(manifest["freeze_id"] == FREEZE_ID, "Preflight freeze ID mismatch")
        require(manifest["protocol_sha256"] == digest(self.frozen / "protocol.json"), "Preflight protocol mismatch")
        require(manifest["controller_sha256"] == protocol["controller_sha256"], "Preflight controller mismatch")
        require(manifest["policies"] == protocol["policies"], "Preflight policies mismatch")
        require(manifest["fp32_precision"] == "ieee", "Preflight precision mismatch")
        return manifest

    def checked_cases(self, manifest, *, deep=False):
        """Verify the compact archive, with optional hashes of the full traces."""
        collection_path = self.final / "collection_manifest.json"
        identity = read_json(collection_path)
        require(identity["freeze_id"] == FREEZE_ID, "Collection freeze ID mismatch")
        require(identity["preflight_sha256"] == digest(self.manifest_path), "Collection preflight mismatch")
        require(identity["branches"] == list(BRANCHES.values()), "Unexpected executed branches")
        marker = read_json(self.final / "reports/report_complete.json")
        require(marker["protocol_sha256"] == digest(self.frozen / "protocol.json"), "Report protocol mismatch")

        cases = manifest["cases"]
        names = [case["name"] for case in cases]
        require(len(names) == len(set(names)) == 80, "Expected 80 distinct test cases")
        require(set(names) == set(marker["case_hashes"]), "Report case inventory mismatch")

        def checked_directory(case, directory):
            marker_path = directory / "complete.json"
            completed = read_json(marker_path)
            require(completed["freeze_id"] == FREEZE_ID, f"Case freeze ID mismatch: {directory}")
            require(read_json(directory / "case.json") == case, f"Case identity mismatch: {directory}")
            filenames = {"case.json", "baseline.npz"} | {f"{method}.json" for method in BRANCHES.values()}
            if completed["eligible"]:
                filenames.add("input.npz")
            require(set(completed["file_hashes"]) == filenames, f"Case inventory mismatch: {directory}")
            verify_hashes(directory, completed["file_hashes"])
            records = {action: read_json(directory / f"{method}.json") for action, method in BRANCHES.items()}
            if deep:
                for record in records.values():
                    verify_hashes(self.final / episode_path(record["summary"]["directory"]), record["episode_hashes"])
            return completed, records

        checks = []
        for case in cases:
            directory = self.final / "cases" / case["name"]
            require(digest(directory / "complete.json") == marker["case_hashes"][case["name"]], f"Report case hash mismatch: {directory}")
            completed, records = checked_directory(case, directory)
            checks.append((case, directory, completed, records))
        parity_case = next(case for case in read_json(self.frozen / "dataset_manifest.json")["cases"] if case["split"] == "train")
        _, parity = checked_directory(parity_case, self.final / "train_parity")
        return checks, parity, marker

    def output_path(self, output):
        path = output.resolve()
        protected = (self.root / "results", self.source, self.root / "pinned")
        # The original checkout can also be passed as the pinned-source root.
        # Protect its source subtrees without forbidding all output under it.
        if self.source == self.root or self.source == REPO_ROOT:
            protected = (self.root / "results", self.root / "pinned", self.source / "src", self.source / "tools")
        require(
            all(path != item and not path.is_relative_to(item) and not item.is_relative_to(path) for item in protected),
            f"Use an output directory separate from archived inputs and source: {path}",
        )
        return path
