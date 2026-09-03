from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_upc_final_clusters import (
    PRIMARY_ORDER,
    SENSITIVITY_ORDER,
    FinalClusterConfig,
    UpcFinalClusterError,
    assign_groups,
    compute_group_profiles,
    eligible_seed_groups,
    label_invariant_agreement,
    load_final_cluster_config,
    pearson_correlation_matrix,
    select_seed_pair,
    verify_upstream_initial_groups,
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class UpcPccPrimitiveTests(unittest.TestCase):
    def test_group_profile_is_cell_and_weekday_mean_after_cell_scaling(self) -> None:
        day_count = 2
        observations_per_hour = 2
        step_count = day_count * 24 * observations_per_hour
        indices = np.arange(step_count, dtype=np.int64).reshape(day_count, 24, 2)
        traffic = np.zeros((4, step_count), dtype=np.float32)
        shaped = traffic.reshape(4, day_count, 24, observations_per_hour)
        increasing = np.arange(24, dtype=np.float32)
        decreasing = increasing[::-1]
        shaped[0, :, :, :] = increasing[None, :, None]
        shaped[1, :, :, :] = increasing[None, :, None] * 2.0
        shaped[2, :, :, :] = decreasing[None, :, None]
        shaped[3, :, :, :] = decreasing[None, :, None] * 3.0
        groups = np.asarray([0, 0, 1, 1], dtype=np.int8)

        one_cell_chunks = compute_group_profiles(
            np.asfortranarray(traffic), indices, groups, cell_chunk_size=1
        )
        all_cells_chunk = compute_group_profiles(
            np.ascontiguousarray(traffic), indices, groups, cell_chunk_size=4
        )

        expected_increasing = 2.0 * increasing / 23.0
        expected_decreasing = 2.0 * decreasing / 23.0
        np.testing.assert_allclose(one_cell_chunks.profiles[0], expected_increasing)
        np.testing.assert_allclose(one_cell_chunks.profiles[1], expected_decreasing)
        np.testing.assert_array_equal(one_cell_chunks.counts[:2], [2, 2])
        np.testing.assert_array_equal(one_cell_chunks.valid[:2], [True, True])
        np.testing.assert_array_equal(
            one_cell_chunks.profiles, all_cells_chunk.profiles
        )
        np.testing.assert_array_equal(one_cell_chunks.valid, all_cells_chunk.valid)

    def test_pcc_matrix_has_expected_extremes_and_invariants(self) -> None:
        profiles = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 2.0, 4.0, 6.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )

        pcc = pearson_correlation_matrix(profiles, np.asarray([8, 9, 20]))

        np.testing.assert_allclose(np.diag(pcc), 1.0)
        np.testing.assert_allclose(pcc, pcc.T)
        self.assertAlmostEqual(float(pcc[0, 1]), 1.0)
        self.assertAlmostEqual(float(pcc[0, 2]), -1.0)
        self.assertGreaterEqual(float(pcc.min()), -1.0)
        self.assertLessEqual(float(pcc.max()), 1.0)

    def test_pcc_rejects_zero_variance_profile(self) -> None:
        with self.assertRaisesRegex(UpcFinalClusterError, "분산이 0"):
            pearson_correlation_matrix(
                np.asarray([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]),
                np.asarray([3, 7]),
            )

    def test_theta_is_strictly_greater_than_ten(self) -> None:
        counts = np.zeros(24, dtype=np.int64)
        counts[3] = 10
        counts[4] = 11

        eligible = eligible_seed_groups(counts, theta=10)

        np.testing.assert_array_equal(eligible, np.asarray([4], dtype=np.int8))

    def test_seed_pair_uses_minimum_pcc_and_lexicographic_tie(self) -> None:
        group_ids = np.asarray([2, 4, 7], dtype=np.int8)
        pcc = np.asarray(
            [
                [1.0, -0.5, -0.5],
                [-0.5, 1.0, 0.2],
                [-0.5, 0.2, 1.0],
            ]
        )

        pair, score = select_seed_pair(pcc, group_ids, group_ids, tolerance=1e-12)

        self.assertEqual(pair, (2, 4))
        self.assertEqual(score, -0.5)

    def test_small_nonempty_group_is_assigned_empty_group_is_not(self) -> None:
        group_ids = np.asarray([0, 1, 3], dtype=np.int8)
        pcc = np.asarray(
            [
                [1.0, -0.8, 0.5],
                [-0.8, 1.0, 0.5],
                [0.5, 0.5, 1.0],
            ]
        )

        result = assign_groups(
            pcc,
            group_ids,
            (0, 1),
            order=PRIMARY_ORDER,
            tolerance=1e-12,
        )

        self.assertEqual(int(result.group_to_cluster[3]), 0)
        self.assertEqual(int(result.group_to_cluster[2]), -1)
        self.assertEqual(result.cluster_members, ((0, 3), (1,)))
        self.assertTrue(result.trace[0]["tie"])

    def test_remaining_order_is_explicit_and_can_change_assignment(self) -> None:
        group_ids = np.asarray([0, 1, 2, 3], dtype=np.int8)
        pcc = np.asarray(
            [
                [1.0, -0.9, 0.6, 0.1],
                [-0.9, 1.0, 0.5, 0.6],
                [0.6, 0.5, 1.0, 0.9],
                [0.1, 0.6, 0.9, 1.0],
            ]
        )

        ascending = assign_groups(
            pcc, group_ids, (0, 1), order=PRIMARY_ORDER, tolerance=1e-12
        )
        descending = assign_groups(
            pcc, group_ids, (0, 1), order=SENSITIVITY_ORDER, tolerance=1e-12
        )

        self.assertEqual([item["group_id"] for item in ascending.trace], [2, 3])
        self.assertEqual([item["group_id"] for item in descending.trace], [3, 2])
        self.assertFalse(
            np.array_equal(
                ascending.group_to_cluster[group_ids],
                descending.group_to_cluster[group_ids],
            )
        )

    def test_label_invariant_agreement_recovers_swapped_labels(self) -> None:
        comparison = label_invariant_agreement(
            np.asarray([0, 0, 1, 1], dtype=np.int8),
            np.asarray([1, 1, 0, 0], dtype=np.int8),
        )

        self.assertEqual(comparison["agreement_ratio"], 1.0)
        self.assertEqual(
            comparison["candidate_label_mapping_to_reference"],
            {"0": 1, "1": 0},
        )


class UpcPccContractTests(unittest.TestCase):
    def test_repository_config_loads_with_preregistered_roles(self) -> None:
        config = load_final_cluster_config(Path("configs/upc_pcc_milan_nov2013.json"))

        self.assertEqual(config.primary_protocol, "train_only")
        self.assertEqual(config.sensitivity_protocol, "algorithm1_full_month")
        self.assertEqual(config.cluster_count, 2)
        self.assertEqual(config.theta, 10)
        self.assertEqual(config.primary_order, PRIMARY_ORDER)
        self.assertEqual(config.sensitivity_order, SENSITIVITY_ORDER)

    def test_upstream_membership_checksum_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = json.loads(
                Path("configs/upc_milan_nov2013.json").read_text(encoding="utf-8")
            )
            for field in source["inputs"]:
                source["inputs"][field] = str(root / "unused" / field)
            output_directory = root / "upstream"
            output_directory.mkdir()
            for field in source["outputs"]:
                suffix = ".npy" if field.endswith("peak_hours") else ".json"
                if field.endswith("csv"):
                    suffix = ".csv"
                source["outputs"][field] = str(output_directory / f"{field}{suffix}")
            upstream_config_path = root / "upstream_config.json"
            upstream_config_path.write_text(
                json.dumps(source, indent=2) + "\n", encoding="utf-8"
            )
            membership_paths = {
                name: Path(source["outputs"][f"{name}_peak_hours"])
                for name in ("train_only", "algorithm1_full_month")
            }
            arrays = {
                "train_only": np.zeros(10_000, dtype=np.int8),
                "algorithm1_full_month": np.full(10_000, 1, dtype=np.int8),
            }
            for name, path in membership_paths.items():
                with path.open("wb") as handle:
                    np.save(handle, arrays[name], allow_pickle=False)
            upstream_config_sha = sha256(upstream_config_path)
            manifest = {
                "status": "diagnostic_mismatch",
                "config": {"sha256": upstream_config_sha},
                "protocol_roles": {
                    "primary_model_protocol": "train_only",
                    "sensitivity_model_protocol": "algorithm1_full_month",
                    "diagnostic_only": "figure4_probe_complete_weeks_mean_profile",
                },
                "continuation_decision": {
                    "exact_figure4_match_required_for_independent_baselines": False
                },
                "protocols": {
                    name: {
                        "group_counts_hour_0_to_23": np.bincount(
                            values, minlength=24
                        ).tolist()
                    }
                    for name, values in arrays.items()
                },
                "outputs": {
                    f"{name}_peak_hours": {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for name, path in membership_paths.items()
                },
            }
            upstream_manifest_path = root / "upstream_manifest.json"
            upstream_manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            config = FinalClusterConfig(
                path=root / "pcc_config.json",
                name="test",
                upstream_config_path=upstream_config_path,
                upstream_config_sha256=upstream_config_sha,
                upstream_manifest_path=upstream_manifest_path,
                accepted_upstream_status="diagnostic_mismatch",
                primary_protocol="train_only",
                sensitivity_protocol="algorithm1_full_month",
                forbidden_protocol="figure4_probe_complete_weeks_mean_profile",
                cluster_count=2,
                theta=10,
                pcc_tie_tolerance=1e-12,
                primary_order=PRIMARY_ORDER,
                sensitivity_order=SENSITIVITY_ORDER,
                paper_group_start=8,
                paper_group_end=18,
                minimum_order_agreement=0.95,
                require_all_cells_assigned_once=True,
                require_all_central_cells_assigned_once=True,
                cell_chunk_size=4,
                output_directory=root / "outputs",
            )
            with membership_paths["train_only"].open("ab") as handle:
                handle.write(b"tamper")

            with self.assertRaisesRegex(UpcFinalClusterError, "크기|checksum"):
                verify_upstream_initial_groups(config)


if __name__ == "__main__":
    unittest.main()
