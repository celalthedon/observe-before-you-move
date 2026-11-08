"""Verify the V3 release bundle and, optionally, its original simulator traces."""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
from zipfile import ZipFile


FREEZE_ID = "94fd47e48b4f"
PROTOCOL_SHA256 = "2042f839866917fa3292c225049877b3f4d4ffd3a0155d936a5a543844b42d00"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "v3"
BRANCHES = ["NoView", "FixedMinus45", "FixedPlus45"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(path, expected):
    require(digest(path) == expected, f"SHA-256 mismatch: {path}")


def verify_core(root):
    index = read(root / "release_manifest.json")
    require(index["freeze_id"] == FREEZE_ID, "Release freeze ID mismatch")
    for name, expected in index["core_sha256"].items():
        check(root / name, expected)

    results = root / "results"
    frozen = results / "v3-final-protocol" / FREEZE_ID
    final = results / "v3-final-test" / FREEZE_ID
    dataset = results / "v3-initial-dataset"
    pinned = root / "pinned"
    check(frozen / "protocol.json", PROTOCOL_SHA256)
    protocol = read(frozen / "protocol.json")
    canonical = json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest()[:12] == FREEZE_ID, "Canonical freeze ID mismatch")
    for name, expected in protocol["artifact_hashes"].items():
        check(frozen / name, expected)
    for name, expected in protocol["source_hashes"].items():
        check(pinned / name, expected)

    tree = ast.parse((pinned / "src/obym/simulation.py").read_text(encoding="utf-8"))
    runner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "run_postturn_audit")
    patch = runpy.run_path(str(pinned / "src/obym/initial_centering_audit.py"))
    controller = patch["make_initial_centering_audit"](runner)
    controller_sha = hashlib.sha256(ast.dump(controller).encode()).hexdigest()
    require(controller_sha == protocol["controller_sha256"],
            "Controller AST hash mismatch; use the archived Python 3.12 AST format")

    selection = read(frozen / "train_selection.json")
    check(dataset / "protocol.json", selection["protocol_sha256"])
    for name, expected in selection["train_sha256"].items():
        check(dataset / "train" / name, expected)
    require(len(selection["train_sha256"]) == 96, "Expected 96 recorded training checksums")

    collection = read(final / "collection_manifest.json")
    manifest_path = frozen / "preflight/test_manifest.json"
    check(manifest_path, collection["preflight_sha256"])
    manifest = read(manifest_path)
    report = read(final / "reports/report_complete.json")
    require(all(value == PROTOCOL_SHA256 for value in
                [manifest["protocol_sha256"], report["protocol_sha256"]]),
            "Report or preflight protocol identity mismatch")
    require(collection["freeze_id"] == manifest["freeze_id"] == FREEZE_ID
            and collection["branches"] == BRANCHES, "Collection identity mismatch")
    check(results / "v3-final-jobs/20261108_215534_579066/collector.py",
          collection["collector_sha256"])
    check(results / "v3-report-jobs/20261108_232050_711219/reporter.py",
          report["reporter_sha256"])
    for name, expected in report["report_hashes"].items():
        check(final / "reports" / name, expected)
    check(final / "collection_manifest.json",
          read(final / "reports/test_summary.json")["collection_manifest_sha256"])

    names = {case["name"] for case in manifest["cases"]}
    require(names == set(report["case_hashes"]) and len(names) == 80,
            "Expected identical preflight and report inventories of 80 test variants")
    episodes = {}
    for path, marker_hash in [
        *((final / "cases" / name, report["case_hashes"][name]) for name in sorted(names)),
        (final / "train_parity", None),
    ]:
        marker = path / "complete.json"
        if marker_hash is not None:
            check(marker, marker_hash)
        complete = read(marker)
        require(complete["freeze_id"] == FREEZE_ID, f"Case freeze ID mismatch: {path}")
        for name, expected in complete["file_hashes"].items():
            check(path / name, expected)
        for branch in BRANCHES:
            record = read(path / f"{branch}.json")
            run_id = record["summary"]["run_id"]
            require(record["origin"] == "executed" and run_id not in episodes,
                    f"Expected a distinct executed branch: {path / branch}")
            episodes[run_id] = record["episode_hashes"]

    require(len(episodes) == 243, "Expected 240 test and 3 train-parity executed episodes")
    print(f"Core integrity passed ({len(index['core_sha256'])} files, "
          "80 test cases, 240 executed test branches, 3 train-parity branches)")
    return index, episodes


def verify_traces(archive_root, index, episodes):
    for name, archive in index["trace_archives"].items():
        path = archive_root / name
        check(path, archive["sha256"])
        hashes = archive["member_sha256"]
        if archive["kind"] == "final_test_and_train_parity":
            prefix = PurePosixPath("results/v3-final-test") / FREEZE_ID / "results/day3"
            expected = {str(prefix / run_id / file): sha for run_id, files in episodes.items()
                        for file, sha in files.items()}
            require(expected == hashes, "Final trace inventory differs from archived episode hashes")
        with ZipFile(path) as zip_file:
            members = zip_file.namelist()
            require(len(members) == len(hashes) and set(members) == set(hashes),
                    f"ZIP member inventory mismatch: {path}")
            for member, expected in hashes.items():
                with zip_file.open(member) as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                require(actual == expected, f"ZIP member SHA-256 mismatch: {member}")
        print(f"Trace archive integrity passed ({name}, {len(hashes)} files)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trace-archives", type=Path, metavar="DIRECTORY",
                        help="also verify both optional trace ZIPs in DIRECTORY")
    args = parser.parse_args()
    try:
        index, episodes = verify_core(args.artifact_root)
        if args.trace_archives is not None:
            verify_traces(args.trace_archives, index, episodes)
    except (OSError, ValueError) as error:
        parser.exit(1, f"V3 integrity check failed: {error}\n")


if __name__ == "__main__":
    main()
