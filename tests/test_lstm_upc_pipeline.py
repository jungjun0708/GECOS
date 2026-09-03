from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.lstm_contract import LstmSmokeContractError
from scripts.prepare_lstm_upc_smoke import (
    build_lstm_smoke_arrays,
    evenly_spaced_positions,
    load_central_cluster_memberships,
)
from scripts.run_lstm_upc_smoke import (
    recombine_cluster_predictions,
    verified_source_git,
)


class LstmSmokeArrayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell_count = 4
        self.step_count = 80
        self.traffic = np.stack(
            [
                np.arange(self.step_count, dtype=np.float32) + 1000 * cell
                for cell in range(self.cell_count)
            ]
        )
        self.missing = np.zeros_like(self.traffic, dtype=bool)
        self.internet_null = np.zeros_like(self.traffic, dtype=bool)
        self.cell_ids = np.array([10, 20, 30, 40], dtype=np.int32)
        self.timestamps = np.arange(self.step_count, dtype=np.int64) * 600_000
        self.central_positions = np.arange(self.cell_count, dtype=np.int64)
        self.memberships = np.array([0, 1, 0, 1], dtype=np.int8)
        self.targets = {
            "train": np.arange(8, 32, dtype=np.int64),
            "validation": np.arange(32, 56, dtype=np.int64),
            "test": np.arange(56, 80, dtype=np.int64),
        }

    def test_evenly_spaced_selection_includes_split_ends(self) -> None:
        np.testing.assert_array_equal(
            evenly_spaced_positions(24, 4), np.array([0, 8, 15, 23])
        )

    def test_windows_targets_masks_and_split_boundaries(self) -> None:
        self.missing[0, 32] = True
        self.internet_null[1, 55] = True
        arrays, metadata = build_lstm_smoke_arrays(
            traffic=self.traffic,
            missing_mask=self.missing,
            internet_null_mask=self.internet_null,
            cell_ids=self.cell_ids,
            timestamps_ms=self.timestamps,
            central_positions=self.central_positions,
            memberships=self.memberships,
            target_indices_by_split=self.targets,
            input_length=8,
            targets_per_split=4,
            split_order=("train", "validation", "test"),
        )

        self.assertEqual(arrays["x_train"].shape, (4, 4, 8, 1))
        self.assertEqual(arrays["y_test"].shape, (4, 4, 1))
        np.testing.assert_array_equal(arrays["target_indices_train"], [8, 16, 23, 31])
        np.testing.assert_array_equal(
            arrays["target_indices_validation"], [32, 40, 47, 55]
        )
        np.testing.assert_array_equal(arrays["target_indices_test"], [56, 64, 71, 79])
        np.testing.assert_array_equal(arrays["x_train"][0, 0, :, 0], np.arange(8))
        self.assertEqual(arrays["y_train"][0, 0, 0], 8)
        self.assertEqual(arrays["persistence_train"][0, 0, 0], 7)
        self.assertEqual(metadata["sample_count_per_split"], 16)
        self.assertEqual(metadata["cluster_counts"], {"0": 2, "1": 2})
        self.assertEqual(metadata["splits"]["validation"]["target_missing_count"], 1)
        self.assertEqual(
            metadata["splits"]["validation"]["target_internet_null_count"], 1
        )
        for split in ("train", "validation", "test"):
            targets = arrays[f"target_indices_{split}"]
            last_inputs = targets - 1
            self.assertTrue(np.all(last_inputs < targets))

    def test_npz_payload_is_byte_deterministic(self) -> None:
        arrays, _ = build_lstm_smoke_arrays(
            traffic=self.traffic,
            missing_mask=self.missing,
            internet_null_mask=self.internet_null,
            cell_ids=self.cell_ids,
            timestamps_ms=self.timestamps,
            central_positions=self.central_positions,
            memberships=self.memberships,
            target_indices_by_split=self.targets,
            input_length=8,
            targets_per_split=4,
            split_order=("train", "validation", "test"),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"bundle-{index}.npz" for index in range(2)]
            for path in paths:
                with path.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        self.assertEqual(hashes[0], hashes[1])


class LstmMembershipTests(unittest.TestCase):
    def test_csv_order_and_cluster_counts_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memberships.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["cell_id", "train_only_cluster"])
                writer.writerows([[10, 0], [20, 1], [30, 0], [40, 1]])
            values, metadata = load_central_cluster_memberships(
                path,
                expected_cell_ids=np.array([10, 20, 30, 40], dtype=np.int32),
                protocol="train_only",
                expected_cluster_counts=((0, 2), (1, 2)),
            )

            np.testing.assert_array_equal(values, [0, 1, 0, 1])
            self.assertEqual(metadata["cluster_counts"], {"0": 2, "1": 2})

    def test_csv_reordered_cells_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memberships.csv"
            path.write_text(
                "cell_id,train_only_cluster\n20,1\n10,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(LstmSmokeContractError, "순서"):
                load_central_cluster_memberships(
                    path,
                    expected_cell_ids=np.array([10, 20], dtype=np.int32),
                    protocol="train_only",
                    expected_cluster_counts=((0, 1), (1, 1)),
                )


class LstmPredictionRecombinationTests(unittest.TestCase):
    def test_interleaved_clusters_return_to_original_cell_order(self) -> None:
        memberships = np.array([0, 1, 0, 1], dtype=np.int8)
        cluster_zero = np.array([[[10.0], [11.0]], [[30.0], [31.0]]], dtype=np.float32)
        cluster_one = np.array([[[20.0], [21.0]], [[40.0], [41.0]]], dtype=np.float32)
        combined, report = recombine_cluster_predictions(
            memberships=memberships,
            predictions_by_cluster={0: cluster_zero, 1: cluster_one},
            expected_cluster_counts=((0, 2), (1, 2)),
        )

        np.testing.assert_array_equal(
            combined[:, :, 0],
            np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32),
        )
        self.assertTrue(report["exact"])
        self.assertEqual(report["filled_cell_count"], 4)

    def test_missing_cluster_prediction_is_rejected(self) -> None:
        with self.assertRaisesRegex(LstmSmokeContractError, "key"):
            recombine_cluster_predictions(
                memberships=np.array([0, 1], dtype=np.int8),
                predictions_by_cluster={0: np.zeros((1, 2, 1), dtype=np.float32)},
                expected_cluster_counts=((0, 1), (1, 1)),
            )


class LstmSourceGitTests(unittest.TestCase):
    def test_clean_full_commit_is_preserved_as_colab_provenance(self) -> None:
        commit = "a" * 40

        result = verified_source_git({"git": {"commit": commit, "dirty": False}})

        self.assertEqual(result["commit"], commit)
        self.assertFalse(result["dirty"])
        self.assertEqual(result["provenance"], "locally_prepared_input_manifest")

    def test_dirty_or_malformed_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(LstmSmokeContractError, "clean Git commit"):
            verified_source_git({"git": {"commit": "a" * 40, "dirty": True}})
        with self.assertRaisesRegex(LstmSmokeContractError, "commit 형식"):
            verified_source_git({"git": {"commit": "not-a-commit", "dirty": False}})


if __name__ == "__main__":
    unittest.main()
