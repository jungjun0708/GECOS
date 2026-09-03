from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_upc_training_policy import (
    EXPECTED_GROUP_ASSIGNMENT_FIELDS,
    FORBIDDEN_PROTOCOL,
    PRIMARY_PROTOCOL,
    SENSITIVITY_PROTOCOL,
    UpcTrainingPolicyError,
    analyze_order_changes,
    evaluate_training_policy,
    load_group_assignment_rows,
    load_policy_config,
    require_training_allowed,
    run_training_policy,
)

TRAIN_COUNTS = [
    38,
    10,
    0,
    1,
    0,
    5,
    1,
    14,
    1050,
    235,
    196,
    339,
    428,
    1309,
    326,
    441,
    472,
    606,
    2043,
    614,
    897,
    773,
    173,
    29,
]
FULL_COUNTS = [
    50,
    8,
    0,
    2,
    0,
    5,
    1,
    8,
    1048,
    234,
    191,
    274,
    348,
    1369,
    310,
    362,
    520,
    562,
    2137,
    714,
    866,
    820,
    150,
    21,
]
CHANGED_GROUPS = {7, 8, 9, 10, 12, 13, 14, 15, 16, 17}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_summary(*, swapped_full_labels: bool = False) -> dict:
    full_mapping = {"0": 1, "1": 0} if swapped_full_labels else {"0": 0, "1": 1}
    return {
        "schema_version": 1,
        "status": "complete_with_order_sensitivity_review",
        "protocol_roles": {
            "primary_model_protocol": PRIMARY_PROTOCOL,
            "sensitivity_model_protocol": SENSITIVITY_PROTOCOL,
            "forbidden_model_protocol": FORBIDDEN_PROTOCOL,
        },
        "algorithm_contract": {
            "primary_remaining_order": "ascending_group_id",
            "sensitivity_remaining_order": "descending_group_id",
        },
        "protocols": {
            PRIMARY_PROTOCOL: {
                "remaining_order_sensitivity": {
                    "all_cells_label_invariant_agreement": {
                        "item_count": 10_000,
                        "matched_item_count": 10_000,
                        "agreement_ratio": 1.0,
                        "candidate_label_mapping_to_reference": {"0": 0, "1": 1},
                    }
                }
            },
            SENSITIVITY_PROTOCOL: {
                "remaining_order_sensitivity": {
                    "all_cells_label_invariant_agreement": {
                        "item_count": 10_000,
                        "matched_item_count": 5_048,
                        "agreement_ratio": 0.5048,
                        "candidate_label_mapping_to_reference": full_mapping,
                    }
                }
            },
        },
        "engineering_gate": {
            "minimum_order_sensitivity_agreement": 0.95,
            "order_sensitivity_review_required": True,
            "ready_for_expensive_model_training": False,
        },
    }


def make_group_rows(*, swapped_full_labels: bool = False) -> list[dict]:
    rows: list[dict] = []
    for protocol, counts in (
        (PRIMARY_PROTOCOL, TRAIN_COUNTS),
        (SENSITIVITY_PROTOCOL, FULL_COUNTS),
    ):
        for group_id, count in enumerate(counts):
            valid = count > 0
            primary_cluster = 0 if valid else None
            changed = protocol == SENSITIVITY_PROTOCOL and group_id in CHANGED_GROUPS
            descending_cluster = (1 if changed else 0) if valid else None
            if swapped_full_labels and protocol == SENSITIVITY_PROTOCOL and valid:
                descending_cluster = 1 - descending_cluster
            rows.append(
                {
                    "protocol": protocol,
                    "group_id": group_id,
                    "cell_count": count,
                    "profile_valid": valid,
                    "primary_cluster": primary_cluster,
                    "descending_cluster": descending_cluster,
                }
            )
    return rows


def write_group_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(EXPECTED_GROUP_ASSIGNMENT_FIELDS)
        for row in rows:
            valid = row["profile_valid"]
            writer.writerow(
                [
                    row["protocol"],
                    row["group_id"],
                    row["cell_count"],
                    int(valid),
                    0,
                    0,
                    row["primary_cluster"] if valid else "",
                    row["descending_cluster"] if valid else "",
                    0,
                ]
            )


class PolicyDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_policy_config(
            Path("configs/upc_training_policy_milan_nov2013.json")
        )

    def test_repository_config_marks_decision_as_post_clustering(self) -> None:
        self.assertEqual(self.config.decision_stage, "post_clustering_pre_model")
        self.assertEqual(self.config.allowed_protocols, (PRIMARY_PROTOCOL,))
        self.assertEqual(
            self.config.blocked_protocols,
            (SENSITIVITY_PROTOCOL, FORBIDDEN_PROTOCOL),
        )
        self.assertFalse(self.config.model_performance_used_for_decision)

    def test_policy_allows_only_stable_train_only_protocol(self) -> None:
        evaluated = evaluate_training_policy(
            self.config, make_summary(), make_group_rows()
        )

        self.assertTrue(evaluated["aggregate_gate"]["ready_for_primary_model_training"])
        self.assertFalse(
            evaluated["aggregate_gate"]["ready_for_all_preregistered_protocol_training"]
        )
        self.assertTrue(evaluated["gates"][PRIMARY_PROTOCOL]["model_training_allowed"])
        self.assertFalse(
            evaluated["gates"][SENSITIVITY_PROTOCOL]["model_training_allowed"]
        )
        self.assertFalse(
            evaluated["gates"][FORBIDDEN_PROTOCOL]["model_training_allowed"]
        )

    def test_protocol_guard_rejects_blocked_and_unknown_protocols(self) -> None:
        evaluated = evaluate_training_policy(
            self.config, make_summary(), make_group_rows()
        )
        policy = {"training_gates": evaluated["gates"]}

        require_training_allowed(policy, PRIMARY_PROTOCOL)
        with self.assertRaisesRegex(UpcTrainingPolicyError, "차단"):
            require_training_allowed(policy, SENSITIVITY_PROTOCOL)
        with self.assertRaisesRegex(UpcTrainingPolicyError, "차단"):
            require_training_allowed(policy, FORBIDDEN_PROTOCOL)
        with self.assertRaisesRegex(UpcTrainingPolicyError, "없는 protocol"):
            require_training_allowed(policy, "unknown")

    def test_label_swap_does_not_change_detected_unstable_groups(self) -> None:
        direct = analyze_order_changes(
            make_summary(), make_group_rows(), SENSITIVITY_PROTOCOL
        )
        swapped = analyze_order_changes(
            make_summary(swapped_full_labels=True),
            make_group_rows(swapped_full_labels=True),
            SENSITIVITY_PROTOCOL,
        )

        self.assertEqual(direct["changed_group_ids"], sorted(CHANGED_GROUPS))
        self.assertEqual(swapped["changed_group_ids"], sorted(CHANGED_GROUPS))
        self.assertEqual(direct["changed_cell_count"], 4_952)
        self.assertEqual(swapped["changed_cell_count"], 4_952)

    def test_changed_cell_count_mismatch_is_rejected(self) -> None:
        rows = make_group_rows()
        row = next(
            value
            for value in rows
            if value["protocol"] == SENSITIVITY_PROTOCOL and value["group_id"] == 7
        )
        row["cell_count"] += 1

        with self.assertRaisesRegex(UpcTrainingPolicyError, "cell 합"):
            evaluate_training_policy(self.config, make_summary(), rows)


class PolicyPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.pcc_config_path = self.root / "pcc_config.json"
        self.pcc_config_path.write_text('{"fixture": true}\n', encoding="utf-8")
        self.summary_path = self.root / "summary.json"
        self.summary = make_summary()
        self.summary_path.write_text(
            json.dumps(self.summary, indent=2) + "\n", encoding="utf-8"
        )
        self.group_path = self.root / "group_assignments.csv"
        write_group_csv(self.group_path, make_group_rows())
        self.all_memberships_path = self.root / "all_cell_memberships.csv"
        self.all_memberships_path.write_text(
            "all membership fixture\n", encoding="utf-8"
        )
        self.central_memberships_path = self.root / "central_900_memberships.csv"
        self.central_memberships_path.write_text(
            "central membership fixture\n", encoding="utf-8"
        )
        paths = {
            "group_assignments_csv": self.group_path,
            "all_cell_memberships_csv": self.all_memberships_path,
            "central_900_memberships_csv": self.central_memberships_path,
            "summary_json": self.summary_path,
        }
        outputs = {
            key: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in paths.items()
        }
        self.pcc_manifest_path = self.root / "pcc_manifest.json"
        self.pcc_manifest_path.write_text(
            json.dumps(
                {
                    "status": (
                        "complete_with_upstream_diagnostic_mismatch_and_"
                        "order_sensitivity_review"
                    ),
                    "config": {"sha256": sha256(self.pcc_config_path)},
                    "outputs": outputs,
                    "summary": self.summary,
                    "upstream_disclosure": {
                        "status": "diagnostic_mismatch",
                        "figure4_exact_match": False,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raw_config = json.loads(
            Path("configs/upc_training_policy_milan_nov2013.json").read_text(
                encoding="utf-8"
            )
        )
        raw_config["inputs"]["pcc_config"] = str(self.pcc_config_path)
        raw_config["inputs"]["pcc_config_sha256"] = sha256(self.pcc_config_path)
        raw_config["inputs"]["pcc_manifest"] = str(self.pcc_manifest_path)
        raw_config["inputs"]["protected_output_sha256"] = {
            key: metadata["sha256"] for key, metadata in outputs.items()
        }
        raw_config["outputs"] = {
            "policy": str(self.root / "training_policy.json"),
            "manifest": str(self.root / "training_policy_manifest.json"),
        }
        self.config_path = self.root / "policy_config.json"
        self.config_path.write_text(
            json.dumps(raw_config, indent=2) + "\n", encoding="utf-8"
        )
        self.config = load_policy_config(self.config_path, base_directory=self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_pipeline_is_deterministic_and_preserves_global_gate(self) -> None:
        first = run_training_policy(self.config)
        first_bytes = self.config.outputs.policy.read_bytes()
        second = run_training_policy(self.config)
        second_bytes = self.config.outputs.policy.read_bytes()

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(first["aggregate_gate"]["ready_for_primary_model_training"])
        self.assertFalse(
            first["aggregate_gate"]["ready_for_all_preregistered_protocol_training"]
        )
        self.assertFalse(
            first["source_disclosure"][
                "clustering_global_ready_for_expensive_model_training"
            ]
        )
        self.assertTrue(
            first["source_disclosure"]["clustering_global_gate_value_preserved"]
        )

    def test_checksum_tampering_does_not_overwrite_existing_policy(self) -> None:
        run_training_policy(self.config)
        original_policy = self.config.outputs.policy.read_bytes()
        with self.all_memberships_path.open("ab") as handle:
            handle.write(b"tamper")

        with self.assertRaisesRegex(UpcTrainingPolicyError, "크기|checksum"):
            run_training_policy(self.config)

        self.assertEqual(self.config.outputs.policy.read_bytes(), original_policy)

    def test_group_csv_requires_all_48_unique_rows(self) -> None:
        rows = make_group_rows()[:-1]
        write_group_csv(self.group_path, rows)

        with self.assertRaisesRegex(UpcTrainingPolicyError, "48행"):
            load_group_assignment_rows(self.group_path)


if __name__ == "__main__":
    unittest.main()
