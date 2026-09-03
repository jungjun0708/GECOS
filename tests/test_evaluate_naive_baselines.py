from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.build_upc_initial_groups import UpcInitialGroupError
from scripts.evaluate_naive_baselines import (
    EvaluationScope,
    PerCellMetricParts,
    compute_per_cell_metric_parts,
    finalize_metric_parts,
    run_naive_baselines,
)
from scripts.forecast_contract import load_forecast_config

INTERVAL_MS = 600_000
OBSERVATIONS_PER_HOUR = 6
ROME = ZoneInfo("Europe/Rome")


def local_timestamps(day_count: int) -> np.ndarray:
    start = datetime(2013, 11, 1, tzinfo=ROME)
    start_ms = int(start.timestamp() * 1000)
    return start_ms + np.arange(day_count * 144, dtype=np.int64) * INTERVAL_MS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class NaiveMetricTests(unittest.TestCase):
    def test_micro_cell_macro_missing_and_zero_target_contract(self) -> None:
        targets = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.float32)
        predictions = np.array([[1, 1, 6], [1, 5, 0]], dtype=np.float32)
        eligible = np.array([[True, True, False], [True, True, True]])
        missing = np.array([[False, False, True], [False, False, False]])
        internet_null = np.zeros_like(missing)
        lag_missing = np.array([[True, False, False], [False, False, False]])
        lag_internet_null = np.zeros_like(missing)
        parts = PerCellMetricParts()
        parts.append(
            compute_per_cell_metric_parts(
                targets,
                predictions,
                eligible,
                missing,
                internet_null,
                lag_missing,
                lag_internet_null,
            )
        )
        scope = EvaluationScope(
            name="synthetic",
            cell_ids=np.array([10, 20], dtype=np.int32),
            positions=np.array([0, 1], dtype=np.int64),
            protocol="test",
        )

        summary, rows = finalize_metric_parts(
            parts,
            scope=scope,
            split="test",
            baseline="persistence",
            target_policy="observed_targets_only",
            target_count_per_cell=3,
        )

        self.assertEqual(summary["candidate_target_count"], 6)
        self.assertEqual(summary["eligible_target_count"], 5)
        self.assertEqual(summary["excluded_missing_target_count"], 1)
        self.assertEqual(summary["positive_target_count_for_mape"], 4)
        self.assertEqual(summary["zero_target_count_excluded_from_mape"], 1)
        self.assertEqual(summary["lag_source_missing_count"], 1)
        self.assertAlmostEqual(summary["micro"]["mae"], 9 / 5)
        self.assertAlmostEqual(summary["micro"]["mape_ratio"], 13 / 24)
        self.assertAlmostEqual(summary["micro"]["mape_percent"], 100 * 13 / 24)
        self.assertAlmostEqual(summary["micro"]["wape"], 9 / 11)
        self.assertAlmostEqual(summary["cell_macro"]["mae"], 5 / 3)
        self.assertAlmostEqual(summary["cell_macro"]["mape_ratio"], 19 / 36)
        self.assertAlmostEqual(summary["cell_macro"]["wape"], 8 / 9)
        self.assertEqual([row["cell_id"] for row in rows], [10, 20])


class NaiveBaselinePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.processed = self.root / "processed"
        self.results = self.processed / "baselines"
        self.processed.mkdir()
        self.upc_config_path = self.root / "upc.json"
        self.forecast_config_path = self.root / "forecast.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _save(self, name: str, array: np.ndarray) -> Path:
        path = self.processed / name
        with path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        return path

    def _metadata(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    def write_case(self, *, chunk_size: int) -> None:
        timestamps = local_timestamps(4)
        daily = np.arange(1, 145, dtype=np.float32)
        traffic = np.stack(
            [
                np.tile(daily, 4),
                np.tile(2 * daily, 4),
                np.tile(np.full(144, 7, dtype=np.float32), 4),
            ]
        )
        missing = np.zeros_like(traffic, dtype=bool)
        internet_null = np.zeros_like(traffic, dtype=bool)
        missing[0, 432] = True
        internet_null[1, 433] = True
        traffic[0, 432] = 0
        traffic[1, 433] = 0
        cell_ids = np.array([1, 2, 3], dtype=np.int32)
        paths = {
            "traffic": self._save("traffic.npy", traffic),
            "cell_ids": self._save("cell_ids.npy", cell_ids),
            "timestamps_ms": self._save("timestamps_ms.npy", timestamps),
            "missing_mask": self._save("missing_mask.npy", missing),
            "internet_null_mask": self._save("internet_null_mask.npy", internet_null),
        }
        processed_manifest = self.processed / "manifest.json"
        processed_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "contract": {
                        "shape": [3, len(timestamps)],
                        "timezone": "Europe/Rome",
                        "interval_ms": INTERVAL_MS,
                    },
                    "outputs": {
                        name: self._metadata(path) for name, path in paths.items()
                    },
                }
            ),
            encoding="utf-8",
        )

        central_csv = self.processed / "central.csv"
        with central_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "cell_id",
                    "grid_row",
                    "grid_column",
                    "centroid_lon",
                    "centroid_lat",
                ]
            )
            writer.writerow([1, 0, 0, "9.0", "45.0"])
            writer.writerow([3, 0, 1, "9.1", "45.0"])
        central_ids = np.array([1, 3], dtype="<i4")
        central_manifest = self.processed / "central_manifest.json"
        central_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "selection": {
                        "cell_count": 2,
                        "cell_ids_int32_sha256": hashlib.sha256(
                            central_ids.tobytes()
                        ).hexdigest(),
                    },
                    "outputs": {"central_cells_csv": self._metadata(central_csv)},
                }
            ),
            encoding="utf-8",
        )

        output_names = [
            "algorithm1_full_month_peak_hours",
            "train_only_peak_hours",
            "figure4_probe_peak_hours",
            "all_cell_memberships_csv",
            "central_900_memberships_csv",
            "group_counts_json",
            "manifest",
        ]
        upc_payload = {
            "schema_version": 1,
            "name": "synthetic-upc",
            "inputs": {
                "processed_manifest": str(processed_manifest),
                "traffic": str(paths["traffic"]),
                "cell_ids": str(paths["cell_ids"]),
                "timestamps_ms": str(paths["timestamps_ms"]),
                "missing_mask": str(paths["missing_mask"]),
                "internet_null_mask": str(paths["internet_null_mask"]),
                "central_manifest": str(central_manifest),
                "central_cells_csv": str(central_csv),
            },
            "grid": {
                "expected_cell_count": 3,
                "expected_central_cell_count": 2,
            },
            "time": {
                "timezone": "Europe/Rome",
                "interval_ms": INTERVAL_MS,
                "observations_per_hour": OBSERVATIONS_PER_HOUR,
                "expected_step_count": len(timestamps),
            },
            "protocols": {
                "algorithm1_full_month": {
                    "start_local": "2013-11-01T00:00:00",
                    "end_exclusive_local": "2013-11-05T00:00:00",
                    "weekdays_only": True,
                },
                "train_only": {
                    "start_local": "2013-11-01T00:00:00",
                    "end_exclusive_local": "2013-11-03T00:00:00",
                    "weekdays_only": True,
                },
            },
            "diagnostics": {
                "figure4_probe_complete_weeks_mean_profile": {
                    "method": "mean_hourly_profile_then_argmax",
                    "start_local": "2013-11-01T00:00:00",
                    "end_exclusive_local": "2013-11-05T00:00:00",
                    "weekdays_only": True,
                }
            },
            "validation": {
                "paper_group_counts": [3] + [0] * 23,
                "require_exact_paper_fingerprint": False,
            },
            "execution": {"cell_chunk_size": 2},
            "outputs": {
                name: str(self.processed / "unused" / f"{name}.out")
                for name in output_names
            },
        }
        self.upc_config_path.write_text(json.dumps(upc_payload), encoding="utf-8")

        forecast_payload = {
            "schema_version": 1,
            "name": "synthetic-naive-baselines",
            "upc_config": str(self.upc_config_path),
            "forecast": {
                "input_length": 8,
                "horizon": 1,
                "evaluation_mode": "rolling_one_step_with_observed_history",
                "partitions": {
                    "train": {
                        "start_local": "2013-11-01T00:00:00",
                        "end_exclusive_local": "2013-11-03T00:00:00",
                        "expected_targets_per_cell": 280,
                    },
                    "validation": {
                        "start_local": "2013-11-03T00:00:00",
                        "end_exclusive_local": "2013-11-04T00:00:00",
                        "expected_targets_per_cell": 144,
                    },
                    "test": {
                        "start_local": "2013-11-04T00:00:00",
                        "end_exclusive_local": "2013-11-05T00:00:00",
                        "expected_targets_per_cell": 144,
                    },
                },
                "auxiliary_splits": {
                    "paper_holdout_10d": {
                        "start_local": "2013-11-03T00:00:00",
                        "end_exclusive_local": "2013-11-05T00:00:00",
                        "expected_targets_per_cell": 288,
                    }
                },
                "evaluated_splits": [
                    "validation",
                    "test",
                    "paper_holdout_10d",
                ],
                "primary_split": "test",
                "paper_comparison_split": "paper_holdout_10d",
                "expected_total_targets_per_cell": 568,
            },
            "baselines": [
                {
                    "name": "persistence",
                    "lag_steps": 1,
                    "description": "previous step",
                },
                {
                    "name": "daily_seasonal_naive",
                    "lag_steps": 144,
                    "description": "previous day",
                },
            ],
            "scopes": ["central_900", "all_10000"],
            "metrics": {
                "primary_target_policy": "all_targets",
                "target_policies": [
                    "all_targets",
                    "observed_targets_only",
                ],
                "mape_positive_targets_only": True,
                "report_mape_ratio_and_percent": True,
                "report_micro_and_cell_macro": True,
                "per_cell_output_splits": ["test"],
            },
            "execution": {"cell_chunk_size": chunk_size},
            "outputs": {
                "summary_json": str(self.results / "summary.json"),
                "per_cell_metrics_csv": str(self.results / "per_cell.csv"),
                "manifest": str(self.results / "manifest.json"),
            },
        }
        self.forecast_config_path.write_text(
            json.dumps(forecast_payload), encoding="utf-8"
        )

    @staticmethod
    def _result(
        summary: dict[str, object],
        *,
        scope: str,
        split: str,
        baseline: str,
        target_policy: str,
    ) -> dict[str, object]:
        return next(
            row
            for row in summary["results"]
            if row["scope"] == scope
            and row["split"] == split
            and row["baseline"] == baseline
            and row["target_policy"] == target_policy
        )

    def test_pipeline_metrics_missing_policy_and_chunk_determinism(self) -> None:
        self.write_case(chunk_size=1)
        config = load_forecast_config(
            self.forecast_config_path, base_directory=self.root
        )
        first_manifest = run_naive_baselines(config)
        first_hashes = {
            name: metadata["sha256"]
            for name, metadata in first_manifest["outputs"].items()
        }
        summary = json.loads(config.outputs.summary_json.read_text(encoding="utf-8"))
        central_daily = self._result(
            summary,
            scope="central_900",
            split="test",
            baseline="daily_seasonal_naive",
            target_policy="observed_targets_only",
        )
        all_persistence = self._result(
            summary,
            scope="all_10000",
            split="test",
            baseline="persistence",
            target_policy="observed_targets_only",
        )

        self.assertEqual(first_manifest["result_row_count"], 24)
        self.assertEqual(first_manifest["per_cell_result_row_count"], 20)
        self.assertEqual(central_daily["eligible_target_count"], 287)
        self.assertEqual(central_daily["micro"]["mae"], 0)
        self.assertEqual(all_persistence["eligible_target_count"], 430)
        self.assertEqual(all_persistence["excluded_missing_target_count"], 2)
        self.assertEqual(all_persistence["lag_source_missing_count"], 1)
        self.assertEqual(all_persistence["lag_source_internet_all_null_count"], 1)

        self.write_case(chunk_size=3)
        second_config = load_forecast_config(
            self.forecast_config_path, base_directory=self.root
        )
        second_manifest = run_naive_baselines(second_config)
        second_hashes = {
            name: metadata["sha256"]
            for name, metadata in second_manifest["outputs"].items()
        }
        self.assertEqual(first_hashes, second_hashes)

    def test_modified_input_is_rejected_before_outputs_are_published(self) -> None:
        self.write_case(chunk_size=2)
        with (self.processed / "traffic.npy").open("ab") as handle:
            handle.write(b"changed")
        config = load_forecast_config(
            self.forecast_config_path, base_directory=self.root
        )

        with self.assertRaisesRegex(UpcInitialGroupError, "크기"):
            run_naive_baselines(config)

        self.assertFalse(config.outputs.summary_json.exists())
        self.assertFalse(config.outputs.per_cell_metrics_csv.exists())
        self.assertFalse(config.outputs.manifest.exists())


if __name__ == "__main__":
    unittest.main()
