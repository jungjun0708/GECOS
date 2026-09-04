#!/usr/bin/env python3
"""LSTM 전체 Train·Validation 준비·window·집계 순수 함수 테스트."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from scripts.aggregate_lstm_full_validation import (
    MODEL_LABELS,
    evaluate_seed_predictions,
    summarize_seed_metrics,
)
from scripts.lstm_full_contract import (
    DEFAULT_CONFIG,
    FullJobSpec,
    FullSplitSpec,
    load_lstm_full_config,
)
from scripts.prepare_lstm_full_training import (
    BUNDLE_ARRAY_NAMES,
    build_compact_train_validation_arrays,
    job_descriptor,
)
from scripts.run_lstm_full_training_job import build_job_arrays


class LstmFullSyntheticFixture(unittest.TestCase):
    def setUp(self) -> None:
        base = load_lstm_full_config(DEFAULT_CONFIG)
        splits = (
            FullSplitSpec(
                name="train",
                target_start_index_inclusive=2,
                target_end_index_exclusive=4,
                targets_per_cell=2,
                samples=4,
            ),
            FullSplitSpec(
                name="validation",
                target_start_index_inclusive=4,
                target_end_index_exclusive=6,
                targets_per_cell=2,
                samples=4,
            ),
        )
        data = replace(
            base.data,
            expected_cell_count=2,
            input_length=2,
            bundle_global_end_index_exclusive=6,
            splits=splits,
            test_target_start_index_inclusive=6,
        )
        scaling = replace(
            base.scaling,
            fit_end_index_exclusive=4,
            roundtrip_max_absolute_error=1e-5,
        )
        upc = replace(base.upc, expected_cluster_counts=((0, 1), (1, 1)))
        self.config = replace(base, data=data, scaling=scaling, upc=upc)
        self.traffic = np.array(
            [
                [0, 1, 2, 3, 4, 5, 600, 700],
                [10, 12, 14, 16, 18, 20, 800, 900],
            ],
            dtype=np.float32,
        )
        self.missing = np.zeros_like(self.traffic, dtype=bool)
        self.internet_null = np.zeros_like(self.traffic, dtype=bool)
        self.missing[0, 4] = True
        self.internet_null[1, 5] = True
        self.timestamps = np.arange(8, dtype=np.int64) * 600_000
        self.cell_ids = np.array([101, 202], dtype=np.int32)
        self.memberships = np.array([0, 1], dtype=np.int8)
        self.arrays, self.metadata = build_compact_train_validation_arrays(
            config=self.config,
            traffic=self.traffic,
            missing_mask=self.missing,
            internet_null_mask=self.internet_null,
            timestamps_ms=self.timestamps,
            cell_ids=self.cell_ids,
            memberships=self.memberships,
        )

    def test_compact_bundle_stops_exactly_before_test(self) -> None:
        self.assertEqual(tuple(self.arrays), BUNDLE_ARRAY_NAMES)
        self.assertEqual(self.arrays["traffic_train_validation"].shape, (2, 6))
        self.assertNotIn(600, self.arrays["traffic_train_validation"])
        self.assertEqual(self.arrays["target_indices_validation"].tolist(), [4, 5])
        self.assertLess(int(self.arrays["target_indices_validation"].max()), 6)
        self.assertFalse(self.metadata["test_arrays_present"])
        self.assertFalse(self.metadata["scaling"]["fit_used_validation"])
        self.assertGreater(
            self.metadata["scaling"]["validation_scaled_above_one_count"], 0
        )

    def test_job_windows_are_causal_scaled_and_cell_major(self) -> None:
        job = FullJobSpec(
            job_id="synthetic_upc_off",
            seed=42,
            condition="upc_off",
            cluster_id=None,
            expected_cell_count=2,
        )
        values = build_job_arrays(self.config, self.arrays, job)

        self.assertEqual(values["x_train"].shape, (2, 2, 2, 1))
        self.assertEqual(values["x_validation"].shape, (2, 2, 2, 1))
        np.testing.assert_allclose(values["x_train"][0, 0, :, 0], [0, 1 / 3])
        np.testing.assert_allclose(values["y_train"][0, :, 0], [2 / 3, 1])
        np.testing.assert_array_equal(
            values["raw_y_validation"][:, :, 0], [[4, 5], [18, 20]]
        )
        self.assertTrue(values["target_missing_mask_validation"][0, 0])
        self.assertTrue(values["target_internet_null_mask_validation"][1, 1])
        self.assertFalse(any("test" in name.lower() for name in values))

    def test_cluster_job_selects_manifest_positions_without_reordering(self) -> None:
        job = FullJobSpec(
            job_id="synthetic_cluster_1",
            seed=42,
            condition="upc_on_cluster_1",
            cluster_id=1,
            expected_cell_count=1,
        )
        values = build_job_arrays(self.config, self.arrays, job)

        np.testing.assert_array_equal(values["central_positions"], [1])
        np.testing.assert_array_equal(values["cell_ids"], [202])
        np.testing.assert_array_equal(values["raw_y_validation"][0, :, 0], [18, 20])

    def test_descriptor_explicitly_forbids_test(self) -> None:
        job = FullJobSpec(
            job_id="seed_42_upc_off",
            seed=42,
            condition="upc_off",
            cluster_id=None,
            expected_cell_count=900,
        )
        descriptor = job_descriptor(
            config=self.config,
            job=job,
            config_sha256="a" * 64,
            input_npz_sha256="b" * 64,
            source_git={"commit": "c" * 40, "dirty": False},
        )

        self.assertFalse(descriptor["test_allowed"])
        self.assertNotIn("test", descriptor["output_relative_directory"].lower())


class LstmFullMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_lstm_full_config(DEFAULT_CONFIG)

    def test_seed_summary_uses_sample_standard_deviation(self) -> None:
        results = []
        for model_index, model in enumerate(MODEL_LABELS):
            for target_policy in ("all_targets", "observed_targets_only"):
                for seed, value in zip((42, 43, 44), (1.0, 2.0, 3.0), strict=True):
                    base = value + model_index * 10
                    results.append(
                        {
                            "seed": seed,
                            "baseline": model,
                            "target_policy": target_policy,
                            "micro": {
                                "mae": base,
                                "mape_ratio": base,
                                "mape_percent": base,
                                "wape": base,
                            },
                            "cell_macro": {
                                "mae": base,
                                "mape_ratio": base,
                                "mape_percent": base,
                                "wape": base,
                            },
                        }
                    )

        summary = summarize_seed_metrics(results, (42, 43, 44))

        row = next(
            item
            for item in summary
            if item["model"] == "lstm_full_upc_off"
            and item["target_policy"] == "all_targets"
            and item["aggregation"] == "micro"
            and item["metric"] == "mae"
        )
        self.assertEqual(row["mean"], 2.0)
        self.assertEqual(row["sample_standard_deviation_ddof_1"], 1.0)
        self.assertEqual(row["values_by_seed"], {"42": 1.0, "43": 2.0, "44": 3.0})

    def test_validation_metrics_accept_unclipped_negative_lstm_predictions(
        self,
    ) -> None:
        tiny_data = replace(
            self.config.data,
            expected_cell_count=2,
            splits=(
                self.config.data.split("train"),
                replace(
                    self.config.data.split("validation"),
                    target_start_index_inclusive=2,
                    target_end_index_exclusive=4,
                    targets_per_cell=2,
                    samples=4,
                ),
            ),
        )
        config = replace(self.config, data=tiny_data)
        traffic = np.array([[0, 1, 2, 3], [0, 2, 4, 6]], dtype=np.float32)
        source = {
            "cell_ids": np.array([10, 20], dtype=np.int32),
            "target_indices_validation": np.array([2, 3], dtype=np.int64),
            "traffic_train_validation": traffic,
            "missing_mask_train_validation": np.zeros_like(traffic, dtype=bool),
            "internet_null_mask_train_validation": np.zeros_like(traffic, dtype=bool),
        }
        negative = np.full((2, 2, 1), -1.0, dtype=np.float32)
        predictions = {
            seed: {
                "lstm_full_upc_off": negative,
                "lstm_full_upc_on": negative + 0.5,
            }
            for seed in config.training.seeds
        }

        results, per_cell = evaluate_seed_predictions(config, source, predictions)

        self.assertEqual(len(results), 12)
        self.assertEqual(len(per_cell), 24)
        self.assertTrue(all(row["micro"]["mae"] > 0 for row in results))


if __name__ == "__main__":
    unittest.main()
