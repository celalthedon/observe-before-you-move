import ast
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.v3.common import Bundle, choose, digest, episode_path, read_json, verify_hashes
from tools.v3.report import paired_statistics, safe


class WorkflowTests(unittest.TestCase):
    def test_strict_threshold_and_minus45_tie_break(self):
        values = np.array([[0.15, 0.15], [0.16, 0.16], [0.16, 0.17], [-1, -2]])
        np.testing.assert_array_equal(choose(values, 0.15), [0, -45, 45, 0])

    def test_archived_episode_references_are_relocatable(self):
        expected = Path("results/day3/obym_12345678")
        self.assertEqual(episode_path(r"C:\original\checkout\results\day3\obym_12345678"), expected)
        self.assertEqual(episode_path("/tmp/moved/results/day3/obym_12345678"), expected)
        self.assertEqual(episode_path(expected.as_posix()), expected)
        with self.assertRaises(ValueError):
            episode_path("../outside")

    def test_failed_or_colliding_branches_are_unsafe(self):
        self.assertTrue(safe({"outcome": "GOAL", "collision_attempts": 0}))
        self.assertFalse(safe({"outcome": "GOAL", "collision_attempts": 1}))
        self.assertFalse(safe({"outcome": "TIMEOUT", "collision_attempts": 0}))

    def test_paired_bootstrap_uses_source_means_and_baseline_minus_spatial(self):
        groups = ["a", "b"]
        policies = ["NoView", "SpatialRidgeCV"]
        rows = [
            {"source_group": group, "policy": policy, "mean_penalized_time_s": time}
            for group, policy, time in [("a", "NoView", 5), ("b", "NoView", 2),
                                        ("a", "SpatialRidgeCV", 1), ("b", "SpatialRidgeCV", 4)]
        ]
        self.assertEqual(paired_statistics(groups, policies, rows, 1, 200), [{
            "baseline": "NoView", "mean_saved_s": 1.0, "ci_low_s": -2.0, "ci_high_s": 4.0,
            "source_wins": 1, "source_ties": 0, "source_losses": 1,
        }])

    def test_frozen_inputs_cannot_be_output_and_hashes_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            bundle = Bundle(root / "bundle", root / "bundle/pinned")
            with self.assertRaises(ValueError):
                bundle.output_path(bundle.final / "reports")
            self.assertEqual(bundle.output_path(root / "generated"), root / "generated")
            member = root / "episode.json"
            member.write_text("original", encoding="utf-8")
            hashes = {member.name: digest(member)}
            verify_hashes(root, hashes)
            member.write_text("modified", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_hashes(root, hashes)
            with self.assertRaises(ValueError):
                verify_hashes(root, {"../episode.json": hashes[member.name]})

    def test_recovered_notebook_cell_hashes(self):
        provenance = Path(__file__).resolve().parents[1] / "provenance"
        evidence = read_json(provenance / "notebook_evidence.json")
        payload_hashes = {}
        for cell in evidence["cells"]:
            source = provenance / cell["source_file"]
            self.assertEqual(digest(source), cell["source_sha256"])
            body = ast.parse(source.read_bytes()).body
            variable = cell["embedded_variable"]
            assignment = next(node for node in body if isinstance(node, ast.Assign) and node.targets[0].id == variable)
            checksum = hashlib.sha256(ast.literal_eval(assignment.value).encode()).hexdigest()
            self.assertEqual(checksum, cell["embedded_source_sha256"])
            self.assertEqual(checksum, payload_hashes.setdefault(variable, checksum))


if __name__ == "__main__":
    unittest.main()
