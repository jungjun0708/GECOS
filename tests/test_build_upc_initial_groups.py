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

from scripts.build_upc_initial_groups import (
    ProtocolSpec,
    UpcInitialGroupError,
    aggregate_hourly_traffic,
    build_local_time_axis,
    compute_mean_profile_peak_hours,
    compute_protocol_peak_hours,
    load_upc_config,
    run_upc_initial_groups,
    select_protocol_hours,
)

INTERVAL_MS = 600_000
OBSERVATIONS_PER_HOUR = 6
ROME = ZoneInfo("Europe/Rome")


def local_timestamps(start_local: datetime, day_count: int) -> np.ndarray:
    zone = ZoneInfo("Europe/Rome")
    start_ms = int(start_local.replace(tzinfo=zone).timestamp() * 1000)
    step_count = day_count * 24 * OBSERVATIONS_PER_HOUR
    return start_ms + np.arange(step_count, dtype=np.int64) * INTERVAL_MS


def protocol(
    name: str,
    start: str,
    end_exclusive: str,
) -> ProtocolSpec:
    return ProtocolSpec(
        name=name,
        start_local=datetime.fromisoformat(start),
        end_exclusive_local=datetime.fromisoformat(end_exclusive),
        weekdays_only=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class UpcTimeAndPeakTests(unittest.TestCase):
    def test_timezone_weekdays_and_train_cutoff(self) -> None:
        timestamps = local_timestamps(datetime(2013, 11, 1, tzinfo=ROME), 30)
        axis = build_local_time_axis(
            timestamps,
            timezone_name="Europe/Rome",
            interval_ms=INTERVAL_MS,
            observations_per_hour=OBSERVATIONS_PER_HOUR,
        )

        self.assertEqual(axis.first_local_timestamp, "2013-11-01T00:00:00+01:00")
        self.assertEqual(axis.last_local_timestamp, "2013-11-30T23:50:00+01:00")
        self.assertEqual(len(axis.dates), 30)
        paper = select_protocol_hours(
            axis,
            protocol(
                "algorithm1_full_month",
                "2013-11-01T00:00:00",
                "2013-12-01T00:00:00",
            ),
        )
        train = select_protocol_hours(
            axis,
            protocol(
                "train_only",
                "2013-11-01T00:00:00",
                "2013-11-21T00:00:00",
            ),
        )

        self.assertEqual(len(paper.dates), 21)
        self.assertEqual(len(train.dates), 14)
        self.assertEqual(train.dates[-1].isoformat(), "2013-11-20")
        self.assertTrue(all(value.weekday() < 5 for value in paper.dates))
        self.assertEqual(paper.indices.shape, (21, 24, 6))
        self.assertEqual(train.indices.shape, (14, 24, 6))

    def test_six_ten_minute_values_are_summed_into_one_hour(self) -> None:
        timestamps = local_timestamps(datetime(2013, 11, 1, tzinfo=ROME), 1)
        axis = build_local_time_axis(
            timestamps,
            timezone_name="Europe/Rome",
            interval_ms=INTERVAL_MS,
            observations_per_hour=OBSERVATIONS_PER_HOUR,
        )
        selected = select_protocol_hours(
            axis,
            protocol(
                "algorithm1_full_month",
                "2013-11-01T00:00:00",
                "2013-11-02T00:00:00",
            ),
        )
        traffic = np.zeros((1, 144), dtype=np.float32)
        traffic[0, :6] = np.arange(1, 7, dtype=np.float32)

        hourly = aggregate_hourly_traffic(traffic, selected.indices)

        self.assertEqual(hourly.shape, (1, 1, 24))
        self.assertEqual(hourly[0, 0, 0], 21.0)
        self.assertEqual(hourly[0, 0, 1], 0.0)

    def test_daily_and_mode_ties_choose_earliest_hour(self) -> None:
        timestamps = local_timestamps(datetime(2013, 11, 4, tzinfo=ROME), 2)
        axis = build_local_time_axis(
            timestamps,
            timezone_name="Europe/Rome",
            interval_ms=INTERVAL_MS,
            observations_per_hour=OBSERVATIONS_PER_HOUR,
        )
        selected = select_protocol_hours(
            axis,
            protocol(
                "algorithm1_full_month",
                "2013-11-04T00:00:00",
                "2013-11-06T00:00:00",
            ),
        )
        traffic = np.zeros((3, 288), dtype=np.float32)
        shaped = traffic.reshape(3, 2, 24, 6)
        shaped[0, 0, 8, :] = 10.0
        shaped[0, 1, 9, :] = 10.0
        shaped[1, 0, 5, :] = 8.0
        shaped[1, 0, 7, :] = 8.0
        shaped[1, 1, 5, :] = 9.0
        shaped[2, :, :, :] = 3.0
        mask = np.zeros_like(traffic, dtype=bool)

        result = compute_protocol_peak_hours(
            traffic,
            mask,
            mask,
            selected,
            cell_chunk_size=2,
        )

        np.testing.assert_array_equal(result.peak_hours, np.array([8, 5, 0]))
        self.assertEqual(result.diagnostics["constant_cell_count"], 1)
        self.assertGreaterEqual(result.diagnostics["daily_peak_tie_cell_day_count"], 3)
        self.assertGreaterEqual(
            result.diagnostics["representative_mode_tie_cell_count"], 1
        )
        self.assertEqual(
            result.diagnostics["scaling_invariance_mismatch_cell_day_count"], 0
        )

    def test_chunk_size_and_memory_layout_do_not_change_membership(self) -> None:
        timestamps = local_timestamps(datetime(2013, 11, 4, tzinfo=ROME), 2)
        axis = build_local_time_axis(
            timestamps,
            timezone_name="Europe/Rome",
            interval_ms=INTERVAL_MS,
            observations_per_hour=OBSERVATIONS_PER_HOUR,
        )
        selected = select_protocol_hours(
            axis,
            protocol(
                "algorithm1_full_month",
                "2013-11-04T00:00:00",
                "2013-11-06T00:00:00",
            ),
        )
        generator = np.random.default_rng(7)
        traffic = generator.random((7, 288), dtype=np.float32)
        mask = np.zeros_like(traffic, dtype=bool)

        first = compute_protocol_peak_hours(
            np.ascontiguousarray(traffic),
            mask,
            mask,
            selected,
            cell_chunk_size=1,
        )
        second = compute_protocol_peak_hours(
            np.asfortranarray(traffic),
            np.asfortranarray(mask),
            np.asfortranarray(mask),
            selected,
            cell_chunk_size=7,
        )

        np.testing.assert_array_equal(first.peak_hours, second.peak_hours)
        self.assertEqual(
            first.diagnostics["group_counts_hour_0_to_23"],
            second.diagnostics["group_counts_hour_0_to_23"],
        )

    def test_mean_profile_diagnostic_is_distinct_from_algorithm_one(self) -> None:
        timestamps = local_timestamps(datetime(2013, 11, 4, tzinfo=ROME), 3)
        axis = build_local_time_axis(
            timestamps,
            timezone_name="Europe/Rome",
            interval_ms=INTERVAL_MS,
            observations_per_hour=OBSERVATIONS_PER_HOUR,
        )
        selected = select_protocol_hours(
            axis,
            protocol(
                "algorithm1_full_month",
                "2013-11-04T00:00:00",
                "2013-11-07T00:00:00",
            ),
        )
        traffic = np.zeros((1, 432), dtype=np.float32)
        shaped = traffic.reshape(1, 3, 24, 6)
        shaped[0, 0, 8, :] = 10.0
        shaped[0, 1, 8, :] = 10.0
        shaped[0, 2, 21, :] = 100.0
        mask = np.zeros_like(traffic, dtype=bool)

        algorithm_one = compute_protocol_peak_hours(
            traffic, mask, mask, selected, cell_chunk_size=1
        )
        diagnostic = compute_mean_profile_peak_hours(
            traffic, mask, mask, selected, cell_chunk_size=1
        )

        self.assertEqual(int(algorithm_one.peak_hours[0]), 8)
        self.assertEqual(int(diagnostic.peak_hours[0]), 21)

    def test_timestamp_axis_rejects_non_ten_minute_alignment(self) -> None:
        timestamps = local_timestamps(datetime(2013, 11, 1, tzinfo=ROME), 1)
        timestamps[1] += 1

        with self.assertRaisesRegex(UpcInitialGroupError, "간격"):
            build_local_time_axis(
                timestamps,
                timezone_name="Europe/Rome",
                interval_ms=INTERVAL_MS,
                observations_per_hour=OBSERVATIONS_PER_HOUR,
            )


class UpcPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.processed = self.root / "processed"
        self.outputs = self.processed / "upc"
        self.processed.mkdir()
        self.config_path = self.root / "config.json"

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

    def write_case(self, *, expected_fingerprint: list[int] | None = None) -> None:
        cell_count = 4
        timestamps = local_timestamps(datetime(2013, 11, 1, tzinfo=ROME), 2)
        traffic = np.zeros((cell_count, len(timestamps)), dtype=np.float32)
        shaped = traffic.reshape(cell_count, 2, 24, 6)
        shaped[0, 0, 0, :] = 3.0
        shaped[1, 0, 1, :] = 4.0
        shaped[2, 0, 1, :] = 5.0
        shaped[3, 0, 23, :] = 6.0
        shaped[:, 1, :, :] = 100.0
        cell_ids = np.arange(1, cell_count + 1, dtype=np.int32)
        missing = np.zeros_like(traffic, dtype=bool)
        null = np.zeros_like(traffic, dtype=bool)
        missing[0, 0] = True
        null[1, 1] = True
        paths = {
            "traffic": self._save("traffic.npy", traffic),
            "cell_ids": self._save("cell_ids.npy", cell_ids),
            "timestamps_ms": self._save("timestamps_ms.npy", timestamps),
            "missing_mask": self._save("missing_mask.npy", missing),
            "internet_null_mask": self._save("internet_null_mask.npy", null),
        }
        processed_manifest = self.processed / "manifest.json"
        processed_manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "contract": {
                        "shape": [cell_count, len(timestamps)],
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
            writer.writerow([4, 0, 1, "9.1", "45.0"])
        central_ids = np.array([1, 4], dtype="<i4")
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

        fingerprint = [0] * 24
        fingerprint[0] = 1
        fingerprint[1] = 2
        fingerprint[23] = 1
        config = {
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
                "expected_cell_count": cell_count,
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
                    "end_exclusive_local": "2013-11-03T00:00:00",
                    "weekdays_only": True,
                },
                "train_only": {
                    "start_local": "2013-11-01T00:00:00",
                    "end_exclusive_local": "2013-11-02T00:00:00",
                    "weekdays_only": True,
                },
            },
            "diagnostics": {
                "figure4_probe_complete_weeks_mean_profile": {
                    "method": "mean_hourly_profile_then_argmax",
                    "start_local": "2013-11-01T00:00:00",
                    "end_exclusive_local": "2013-11-03T00:00:00",
                    "weekdays_only": True,
                }
            },
            "validation": {
                "paper_group_counts": expected_fingerprint or fingerprint,
                "require_exact_paper_fingerprint": True,
            },
            "execution": {"cell_chunk_size": 2},
            "outputs": {
                "algorithm1_full_month_peak_hours": str(
                    self.outputs / "algorithm1_peak.npy"
                ),
                "train_only_peak_hours": str(self.outputs / "train_peak.npy"),
                "figure4_probe_peak_hours": str(self.outputs / "figure4_peak.npy"),
                "all_cell_memberships_csv": str(self.outputs / "all.csv"),
                "central_900_memberships_csv": str(self.outputs / "central.csv"),
                "group_counts_json": str(self.outputs / "counts.json"),
                "manifest": str(self.outputs / "manifest.json"),
            },
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def test_pipeline_outputs_fingerprint_central_mapping_and_determinism(self) -> None:
        self.write_case()
        config = load_upc_config(self.config_path, base_directory=self.root)

        first = run_upc_initial_groups(config)
        first_hashes = {
            name: metadata["sha256"] for name, metadata in first["outputs"].items()
        }
        second = run_upc_initial_groups(config)
        second_hashes = {
            name: metadata["sha256"] for name, metadata in second["outputs"].items()
        }

        self.assertTrue(first["paper_fingerprint"]["exact_match"])
        self.assertEqual(first["paper_fingerprint"]["l1_difference"], 0)
        self.assertTrue(first["figure4_probe"]["exact_match"])
        self.assertFalse(first["figure4_probe"]["eligible_for_model_input"])
        self.assertEqual(
            first["protocol_roles"]["primary_model_protocol"], "train_only"
        )
        self.assertEqual(first_hashes, second_hashes)
        np.testing.assert_array_equal(
            np.load(config.outputs.algorithm1_full_month_peak_hours),
            np.array([0, 1, 1, 23], dtype=np.int8),
        )
        with config.outputs.central_900_memberships_csv.open(
            encoding="utf-8", newline=""
        ) as handle:
            central_rows = list(csv.DictReader(handle))
        self.assertEqual([row["cell_id"] for row in central_rows], ["1", "4"])
        self.assertEqual(
            [row["algorithm1_full_month_peak_hour"] for row in central_rows],
            ["0", "23"],
        )
        self.assertEqual(
            first["protocols"]["algorithm1_full_month"]["weekday_count"], 1
        )
        self.assertEqual(first["protocols"]["train_only"]["weekday_count"], 1)
        self.assertEqual(
            first["protocols"]["algorithm1_full_month"]["missing_pair_count"],
            1,
        )
        self.assertEqual(
            first["protocols"]["algorithm1_full_month"]["internet_all_null_pair_count"],
            1,
        )

    def test_modified_input_is_rejected_by_manifest_checksum(self) -> None:
        self.write_case()
        with (self.processed / "traffic.npy").open("ab") as handle:
            handle.write(b"changed")
        config = load_upc_config(self.config_path, base_directory=self.root)

        with self.assertRaisesRegex(UpcInitialGroupError, "크기"):
            run_upc_initial_groups(config)

    def test_config_rejects_fingerprint_with_wrong_sum(self) -> None:
        wrong = [0] * 24
        self.write_case(expected_fingerprint=wrong)

        with self.assertRaisesRegex(UpcInitialGroupError, "fingerprint의 합"):
            load_upc_config(self.config_path, base_directory=self.root)


if __name__ == "__main__":
    unittest.main()
