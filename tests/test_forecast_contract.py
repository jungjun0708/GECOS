from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.forecast_contract import (
    BaselineOutputPaths,
    BaselineSpec,
    ForecastConfig,
    ForecastContractError,
    TimeRangeSpec,
    build_forecast_index_contract,
    load_forecast_config,
)

INTERVAL_MS = 600_000
ROME = ZoneInfo("Europe/Rome")


def local_timestamps(day_count: int) -> np.ndarray:
    start = datetime(2013, 11, 1, tzinfo=ROME)
    start_ms = int(start.timestamp() * 1000)
    return start_ms + np.arange(day_count * 144, dtype=np.int64) * INTERVAL_MS


def time_range(name: str, start: str, end: str, count: int) -> TimeRangeSpec:
    return TimeRangeSpec(
        name=name,
        start_local=datetime.fromisoformat(start),
        end_exclusive_local=datetime.fromisoformat(end),
        expected_targets_per_cell=count,
    )


def synthetic_config(root: Path) -> ForecastConfig:
    return ForecastConfig(
        path=root / "config.json",
        name="synthetic-forecast",
        upc_config_path=root / "upc.json",
        input_length=8,
        horizon=1,
        evaluation_mode="rolling_one_step_with_observed_history",
        partitions={
            "train": time_range(
                "train",
                "2013-11-01T00:00:00",
                "2013-11-03T00:00:00",
                280,
            ),
            "validation": time_range(
                "validation",
                "2013-11-03T00:00:00",
                "2013-11-04T00:00:00",
                144,
            ),
            "test": time_range(
                "test",
                "2013-11-04T00:00:00",
                "2013-11-05T00:00:00",
                144,
            ),
        },
        auxiliary_splits={
            "paper_holdout_10d": time_range(
                "paper_holdout_10d",
                "2013-11-03T00:00:00",
                "2013-11-05T00:00:00",
                288,
            )
        },
        evaluated_splits=("validation", "test", "paper_holdout_10d"),
        primary_split="test",
        paper_comparison_split="paper_holdout_10d",
        expected_total_targets_per_cell=568,
        baselines=(
            BaselineSpec("persistence", 1, "previous"),
            BaselineSpec("daily_seasonal_naive", 144, "previous day"),
        ),
        scopes=("central_900", "all_10000"),
        primary_target_policy="all_targets",
        target_policies=("all_targets", "observed_targets_only"),
        mape_positive_targets_only=True,
        report_mape_ratio_and_percent=True,
        report_micro_and_cell_macro=True,
        per_cell_output_splits=("test",),
        cell_chunk_size=2,
        outputs=BaselineOutputPaths(
            summary_json=root / "summary.json",
            per_cell_metrics_csv=root / "cells.csv",
            manifest=root / "manifest.json",
        ),
    )


class ForecastIndexContractTests(unittest.TestCase):
    def test_target_timestamp_split_counts_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            result = build_forecast_index_contract(
                local_timestamps(4),
                config,
                timezone_name="Europe/Rome",
                interval_ms=INTERVAL_MS,
            )

        np.testing.assert_array_equal(result.target_indices["train"], np.arange(8, 288))
        np.testing.assert_array_equal(
            result.target_indices["validation"], np.arange(288, 432)
        )
        np.testing.assert_array_equal(
            result.target_indices["test"], np.arange(432, 576)
        )
        np.testing.assert_array_equal(
            result.target_indices["paper_holdout_10d"], np.arange(288, 576)
        )
        self.assertEqual(result.total_targets_per_cell, 568)
        self.assertEqual(
            result.split_metadata["train"]["first_target_local"],
            "2013-11-01T01:20:00+01:00",
        )
        self.assertEqual(
            result.split_metadata["validation"]["first_target_local"],
            "2013-11-03T00:00:00+01:00",
        )
        self.assertEqual(result.split_metadata["test"]["first_input_index"], 424)
        self.assertLess(
            result.split_metadata["test"]["last_input_index_for_first_target"],
            result.split_metadata["test"]["first_target_index"],
        )

    def test_timestamp_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            timestamps = local_timestamps(4)
            timestamps[100] += 1

            with self.assertRaisesRegex(ForecastContractError, "간격"):
                build_forecast_index_contract(
                    timestamps,
                    config,
                    timezone_name="Europe/Rome",
                    interval_ms=INTERVAL_MS,
                )

    def test_incorrect_expected_split_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            config.partitions["validation"] = time_range(
                "validation",
                "2013-11-03T00:00:00",
                "2013-11-04T00:00:00",
                143,
            )

            with self.assertRaisesRegex(ForecastContractError, "target 수"):
                build_forecast_index_contract(
                    local_timestamps(4),
                    config,
                    timezone_name="Europe/Rome",
                    interval_ms=INTERVAL_MS,
                )

    def test_config_rejects_changed_baseline_lag(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (
                repository_root / "configs" / "naive_baselines_milan_nov2013.json"
            ).read_text(encoding="utf-8")
        )
        payload["baselines"][1]["lag_steps"] = 143
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ForecastContractError, "daily=144"):
                load_forecast_config(path, base_directory=repository_root)

    def test_config_rejects_changed_input_length(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (
                repository_root / "configs" / "naive_baselines_milan_nov2013.json"
            ).read_text(encoding="utf-8")
        )
        payload["forecast"]["input_length"] = 7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ForecastContractError, "input_length는 8"):
                load_forecast_config(path, base_directory=repository_root)


if __name__ == "__main__":
    unittest.main()
