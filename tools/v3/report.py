"""Recalculate archived outcome statistics without simulator or PyTorch."""
import json

import numpy as np

from .common import (
    BRANCHES, FREEZE_ID, MODELS, choose, digest, episode_path, read_csv,
    read_json, require, verify_hashes, write_csv, write_json,
)


def safe(summary):
    return summary["outcome"] == "GOAL" and summary["collision_attempts"] == 0


def mean(rows, field):
    return float(np.mean([row[field] for row in rows]))


def paired_statistics(sources, policies, source_rows, seed, samples):
    costs = {
        (row["policy"], row["source_group"]): row["mean_penalized_time_s"]
        for row in source_rows
    }
    indices = np.random.default_rng(seed).integers(0, len(sources), size=(samples, len(sources)))
    paired = []
    for policy in policies:
        if policy == "SpatialRidgeCV":
            continue
        delta = np.array([
            costs[policy, source] - costs["SpatialRidgeCV", source]
            for source in sources
        ])
        low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        paired.append({
            "baseline": policy,
            "mean_saved_s": float(delta.mean()),
            "ci_low_s": float(low),
            "ci_high_s": float(high),
            "source_wins": int((delta > 1e-6).sum()),
            "source_ties": int((np.abs(delta) <= 1e-6).sum()),
            "source_losses": int((delta < -1e-6).sum()),
        })
    return paired


def archive_predictions(reports, cases):
    with np.load(reports / "predictions.npz", allow_pickle=False) as data:
        eligible = np.array([complete["eligible"] for _, _, complete, _ in cases])
        require(np.array_equal(data["scene"], [case["name"] for case, *_ in cases]), "Prediction scene order mismatch")
        require(np.array_equal(data["eligible"], eligible), "Prediction eligibility mismatch")
        metadata = json.loads(data["metadata"].item())
        require(metadata["device"] == "cpu" and metadata["precision"] == "ieee_fp32", "Prediction precision mismatch")
        thresholds = metadata["thresholds"]
        require(set(thresholds) == set(MODELS), "Prediction threshold inventory mismatch")
        values = {method: data[method] for method in MODELS}
        for method, predictions in values.items():
            require(predictions.shape == (len(cases), 2), f"Prediction shape mismatch: {method}")
            require(np.isfinite(predictions[eligible]).all(), f"Non-finite eligible predictions: {method}")
            require(np.isnan(predictions[~eligible]).all(), f"Unexpected ineligible predictions: {method}")
    return eligible, thresholds, values


def assert_values(actual, expected, context):
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, float):
            matches = np.isclose(value, float(reference), atol=1e-10, rtol=0)
        elif value is None:
            matches = reference in (None, "")
        elif isinstance(reference, str) and not isinstance(value, str):
            matches = str(value) == reference
        else:
            matches = value == reference
        require(matches, f"Archived {context} differs at {key}: {value!r} vs {reference!r}")


def check_rows(actual, expected, context, *, portable_references=False):
    require(len(actual) == len(expected), f"Archived row count differs: {context}")
    for new, old in zip(actual, expected, strict=True):
        if portable_references:
            old = {**old, "reference_directory": episode_path(old["reference_directory"]).as_posix()}
        assert_values(new, old, context)


def regenerate(bundle, output, *, deep=False):
    protocol = bundle.protocol()
    manifest = bundle.manifest(protocol)
    cases, parity, marker = bundle.checked_cases(manifest, deep=deep)
    reports = bundle.final / "reports"
    checked_names = ("predictions.npz", "test_summary.json", "episode_metrics.csv",
                     "source_metrics.csv", "policy_summary.csv", "paired_comparisons.json",
                     "paired_comparisons.csv")
    verify_hashes(reports, {name: marker["report_hashes"][name] for name in checked_names})
    eligible, thresholds, predictions = archive_predictions(reports, cases)
    policies = protocol["policies"]
    require(set(policies) == {*BRANCHES.values(), "HeadingRule", *MODELS} and len(policies) == 9, "Unexpected policies")

    choices = {policy: np.zeros(len(cases), dtype=int) for policy in policies}
    choices["FixedMinus45"][eligible] = -45
    choices["FixedPlus45"][eligible] = 45
    yaw = np.array([case["initial_yaw_rad"] for case, *_ in cases])
    choices["HeadingRule"][eligible & (np.cos(yaw) < 0)] = -45
    for model in MODELS:
        choices[model][eligible] = choose(predictions[model][eligible], thresholds[model])

    penalty = float(protocol["max_sim_time_s"])
    rows, executed = [], set()
    for index, (case, _, complete, records) in enumerate(cases):
        costs = {}
        for offset, record in records.items():
            summary = record["summary"]
            require(summary["scene"] == case["name"] and summary["freeze_id"] == FREEZE_ID, f"Branch identity mismatch: {case['name']}")
            require(0 <= summary["completed_views"] <= summary["view_attempts"] <= 1, f"Invalid view count: {case['name']}")
            costs[offset] = float(summary["sim_time_s"]) if safe(summary) else penalty
            if record["origin"] == "executed":
                executed.add(episode_path(summary["directory"]))

        oracle = min(costs.values())
        for policy in policies:
            offset = int(choices[policy][index])
            record = records[offset]
            summary = record["summary"]
            gain = costs[0] - costs[offset]
            prediction = predictions[policy][index] if policy in MODELS and eligible[index] else (None, None)
            rows.append({
                "scene": case["name"],
                "source_group": case["source_group"],
                "kind": case["source_spec"]["kind"],
                "policy": policy,
                "selected_offset_deg": offset,
                "eligible": complete["eligible"],
                "outcome": summary["outcome"],
                "safe_success": safe(summary),
                "sim_time_s": float(summary["sim_time_s"]),
                "penalized_time_s": costs[offset],
                "distance_m": float(summary["distance_m"]),
                "collision_attempts": int(summary["collision_attempts"]),
                "view_attempts": int(summary["view_attempts"]),
                "completed_views": int(summary["completed_views"]),
                "gain_vs_no_view_s": gain,
                "regret_s": costs[offset] - oracle,
                "harmful_look": bool(offset != 0 and gain < -1e-6),
                "predicted_minus45_s": None if prediction[0] is None else float(prediction[0]),
                "predicted_plus45_s": None if prediction[1] is None else float(prediction[1]),
                "threshold_s": thresholds.get(policy),
                "result_origin": (
                    "executed_branch" if policy in BRANCHES.values() and record["origin"] == "executed"
                    else "derived_from_executed_branch"
                ),
                "reference_directory": episode_path(summary["directory"]).as_posix(),
            })

    sources = sorted({row["source_group"] for row in rows})
    require(len(sources) == 20, "Expected 20 independent source groups")
    policy_summary, source_rows = {}, []
    for policy in policies:
        members = [row for row in rows if row["policy"] == policy]
        successes = [row for row in members if row["safe_success"]]
        policy_summary[policy] = {
            "policy": policy,
            "safe_success": len(successes),
            "mean_penalized_time_s": mean(members, "penalized_time_s"),
            "mean_gain_vs_no_view_s": mean(members, "gain_vs_no_view_s"),
            "mean_regret_s": mean(members, "regret_s"),
            "selected_looks": sum(row["selected_offset_deg"] != 0 for row in members),
            "completed_views": sum(row["completed_views"] for row in members),
            "harmful_looks": sum(row["harmful_look"] for row in members),
            "mean_distance_success_m": mean(successes, "distance_m") if successes else None,
        }
        for source in sources:
            variants = [row for row in members if row["source_group"] == source]
            require(len(variants) == 4, f"Expected four variants: {source}")
            source_rows.append({
                "source_group": source,
                "policy": policy,
                "mean_penalized_time_s": mean(variants, "penalized_time_s"),
                "safe_success": sum(row["safe_success"] for row in variants),
            })

    statistics = protocol["statistics"]
    paired = paired_statistics(sources, policies, source_rows, statistics["bootstrap_seed"], statistics["bootstrap_samples"])
    summary = {
        "freeze_id": FREEZE_ID,
        "source_scenes": len(sources),
        "variants": len(cases),
        "policy_outcomes": len(rows),
        "collection_manifest_sha256": digest(bundle.final / "collection_manifest.json"),
        "executed_test_episodes": len(executed),
        "train_parity_episodes": len({episode_path(record["summary"]["directory"]) for record in parity.values()}),
        "evaluation_mode": "first-eligible frozen choices derived from deterministic executed action branches",
        "scope": protocol["test_scope"],
        "inference_device": "cpu",
        "inference_wall_time_in_primary_metric": False,
        "failure_penalty_s": penalty,
        "bootstrap_unit": "source_scene_with_four_variants",
        "bootstrap_seed": statistics["bootstrap_seed"],
        "bootstrap_samples": statistics["bootstrap_samples"],
        "intervals": "paired percentile 95%; no multiple-comparison adjustment",
        "oracle": "retrospective best of three executed action branches, not a deployable policy",
    }

    archived_summary = read_json(reports / "test_summary.json")
    assert_values(summary, archived_summary, "test summary")
    for policy in policies:
        assert_values(policy_summary[policy], archived_summary["policies"][policy], f"summary for {policy}")
    check_rows(rows, read_csv(reports / "episode_metrics.csv"), "episode metrics", portable_references=True)
    check_rows(source_rows, read_csv(reports / "source_metrics.csv"), "source metrics")
    check_rows(list(policy_summary.values()), read_csv(reports / "policy_summary.csv"), "policy summary")
    check_rows(paired, read_csv(reports / "paired_comparisons.csv"), "paired comparisons CSV")
    check_rows(paired, read_json(reports / "paired_comparisons.json"), "paired comparisons JSON")

    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "episode_metrics.csv", rows)
    write_csv(output / "source_metrics.csv", source_rows)
    write_csv(output / "policy_summary.csv", list(policy_summary.values()))
    write_csv(output / "paired_comparisons.csv", paired)
    write_json(output / "paired_comparisons.json", paired)
    write_json(output / "test_summary.json", {**summary, "policies": policy_summary})
    print(f"Archive parity passed: {len(rows)} policy outcomes, {len(source_rows)} source-policy rows, {len(paired)} paired intervals")
    print(f"Verified {'full episode traces and ' if deep else ''}compact cases; wrote {output}")
