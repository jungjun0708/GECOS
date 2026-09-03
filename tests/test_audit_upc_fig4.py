from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from scripts.audit_upc_fig4 import (
    AUDIT_VARIANT_NAMES,
    DIAGNOSTIC_VARIANT,
    AuditVariant,
    Fig4AuditError,
    compute_audit_variant,
    load_fig4_audit_config,
    run_fig4_audit,
)
from scripts.build_upc_initial_groups import (
    ProtocolSpec,
    build_local_time_axis,
    compute_protocol_peak_hours,
    select_protocol_hours,
)

INTERVAL_MS = 600_000
OBSERVATIONS_PER_HOUR = 6
ROME = ZoneInfo("Europe/Rome")


def local_timestamps(start_local: datetime, day_count: int) -> np.ndarray:
    start_ms = int(start_local.replace(tzinfo=ROME).timestamp() * 1000)
    step_count = day_count * 24 * OBSERVATIONS_PER_HOUR
    return start_ms + np.arange(step_count, dtype=np.int64) * INTERVAL_MS


def selected_hours(start: str, end_exclusive: str, day_count: int):
    timestamps = local_timestamps(datetime.fromisoformat(start), day_count)
    axis = build_local_time_axis(
        timestamps,
        timezone_name="Europe/Rome",
        interval_ms=INTERVAL_MS,
        observations_per_hour=OBSERVATIONS_PER_HOUR,
    )
    return select_protocol_hours(
        axis,
        ProtocolSpec(
            name="test",
            start_local=datetime.fromisoformat(start),
            end_exclusive_local=datetime.fromisoformat(end_exclusive),
            weekdays_only=True,
        ),
    )


def variant(
    name: str,
    *,
    reducer: str = "sum",
    representative: str = "mean_hourly_profile",
    missing_policy: str = "zero_filled",
    dtype: str = "float64",
) -> AuditVariant:
    return AuditVariant(
        name=name,
        start_local=datetime.fromisoformat("2013-11-04T00:00:00"),
        end_exclusive_local=datetime.fromisoformat("2013-11-07T00:00:00"),
        weekdays_only=True,
        hourly_reducer=reducer,
        representative_method=representative,
        missing_policy=missing_policy,
        calculation_dtype=dtype,
    )


class AuditVariantTests(unittest.TestCase):
    def test_registered_factors_produce_distinct_interpretations(self) -> None:
        hours = selected_hours("2013-11-04T00:00:00", "2013-11-07T00:00:00", 3)
        traffic = np.zeros((2, 3 * 144), dtype=np.float32)
        shaped = traffic.reshape(2, 3, 24, 6)

        shaped[0, 0, 8, :] = 10
        shaped[0, 1, 8, :] = 10
        shaped[0, 2, 21, :] = 100

        shaped[1, :, 8, :] = 5
        shaped[1, :, 9, 0] = 10
        mask = np.zeros_like(traffic, dtype=bool)

        daily_mode = compute_audit_variant(
            traffic,
            mask,
            mask,
            hours,
            variant("daily", representative="daily_peak_mode"),
            cell_chunk_size=1,
        )
        algorithm_one = compute_protocol_peak_hours(
            traffic,
            mask,
            mask,
            hours,
            cell_chunk_size=1,
        )
        mean_profile = compute_audit_variant(
            traffic,
            mask,
            mask,
            hours,
            variant("profile"),
            cell_chunk_size=2,
        )
        hourly_max = compute_audit_variant(
            traffic,
            mask,
            mask,
            hours,
            variant("max", reducer="max"),
            cell_chunk_size=2,
        )

        self.assertEqual(int(daily_mode.peak_hours[0]), 8)
        np.testing.assert_array_equal(daily_mode.peak_hours, algorithm_one.peak_hours)
        self.assertEqual(int(mean_profile.peak_hours[0]), 21)
        self.assertEqual(int(mean_profile.peak_hours[1]), 8)
        self.assertEqual(int(hourly_max.peak_hours[1]), 9)

    def test_missing_exclusion_is_explicit_and_renormalized(self) -> None:
        hours = selected_hours("2013-11-04T00:00:00", "2013-11-05T00:00:00", 1)
        traffic = np.zeros((1, 144), dtype=np.float32)
        shaped = traffic.reshape(1, 1, 24, 6)
        shaped[0, 0, 8, 0] = 10
        shaped[0, 0, 9, :] = 2
        missing = np.zeros_like(traffic, dtype=bool).reshape(1, 1, 24, 6)
        missing[0, 0, 8, 1:] = True
        missing = missing.reshape(1, 144)
        internet_null = np.zeros_like(missing)

        zero_filled = compute_audit_variant(
            traffic,
            missing,
            internet_null,
            hours,
            variant("zero"),
            cell_chunk_size=1,
        )
        excluded = compute_audit_variant(
            traffic,
            missing,
            internet_null,
            hours,
            variant("exclude", missing_policy="exclude_flagged_and_renormalize"),
            cell_chunk_size=1,
        )

        self.assertEqual(int(zero_filled.peak_hours[0]), 9)
        self.assertEqual(int(excluded.peak_hours[0]), 8)
        self.assertEqual(excluded.diagnostics["missing_pair_count"], 5)


class AuditPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.processed = self.root / "processed"
        self.processed.mkdir()
        self.upc_config_path = self.root / "upc.json"
        self.audit_config_path = self.root / "audit.json"
        self._write_case()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _save(self, name: str, value: np.ndarray) -> Path:
        path = self.processed / name
        with path.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        return path

    def _metadata(self, path: Path) -> dict[str, object]:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": self._sha256(path),
        }

    def _write_case(self) -> None:
        cell_count = 4
        timestamps = local_timestamps(datetime(2013, 11, 1, tzinfo=ROME), 30)
        traffic = np.zeros((cell_count, len(timestamps)), dtype=np.float32)
        cell_ids = np.arange(1, cell_count + 1, dtype=np.int32)
        missing = np.zeros_like(traffic, dtype=bool)
        internet_null = np.zeros_like(traffic, dtype=bool)
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
        fingerprint = [0] * 24
        fingerprint[0] = cell_count
        output_directory = self.processed / "upc"
        upc_config = {
            "schema_version": 1,
            "name": "synthetic-upc",
            "inputs": {
                "processed_manifest": str(processed_manifest),
                **{name: str(path) for name, path in paths.items()},
                "central_manifest": str(self.processed / "unused-central.json"),
                "central_cells_csv": str(self.processed / "unused-central.csv"),
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
                    "end_exclusive_local": "2013-12-01T00:00:00",
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
                    "start_local": "2013-11-04T00:00:00",
                    "end_exclusive_local": "2013-11-05T00:00:00",
                    "weekdays_only": True,
                }
            },
            "validation": {
                "paper_group_counts": fingerprint,
                "require_exact_paper_fingerprint": False,
            },
            "execution": {"cell_chunk_size": 2},
            "outputs": {
                "algorithm1_full_month_peak_hours": str(
                    output_directory / "algorithm1.npy"
                ),
                "train_only_peak_hours": str(output_directory / "train.npy"),
                "figure4_probe_peak_hours": str(output_directory / "probe.npy"),
                "all_cell_memberships_csv": str(output_directory / "all.csv"),
                "central_900_memberships_csv": str(output_directory / "central.csv"),
                "group_counts_json": str(output_directory / "counts.json"),
                "manifest": str(output_directory / "manifest.json"),
            },
        }
        self.upc_config_path.write_text(json.dumps(upc_config), encoding="utf-8")

        factor_specs = [
            ("sum", "daily_peak_mode", "float64", "zero_filled"),
            ("sum", "mean_hourly_profile", "float64", "zero_filled"),
            ("max", "daily_peak_mode", "float64", "zero_filled"),
            ("max", "mean_hourly_profile", "float64", "zero_filled"),
            ("sum", "daily_peak_mode", "float64", "zero_filled"),
            ("sum", "mean_hourly_profile", "float64", "zero_filled"),
            ("max", "daily_peak_mode", "float64", "zero_filled"),
            ("max", "mean_hourly_profile", "float64", "zero_filled"),
            ("sum", "mean_hourly_profile", "float32", "zero_filled"),
            (
                "sum",
                "mean_hourly_profile",
                "float64",
                "exclude_flagged_and_renormalize",
            ),
        ]
        variants = []
        for index, (name, factors) in enumerate(
            zip(AUDIT_VARIANT_NAMES, factor_specs, strict=True)
        ):
            complete_weeks = index >= 4
            reducer, representative, dtype, missing_policy = factors
            variants.append(
                {
                    "name": name,
                    "start_local": (
                        "2013-11-04T00:00:00"
                        if complete_weeks
                        else "2013-11-01T00:00:00"
                    ),
                    "end_exclusive_local": (
                        "2013-11-30T00:00:00"
                        if complete_weeks
                        else "2013-12-01T00:00:00"
                    ),
                    "weekdays_only": True,
                    "hourly_reducer": reducer,
                    "representative_method": representative,
                    "missing_policy": missing_policy,
                    "calculation_dtype": dtype,
                }
            )
        audit_config = {
            "schema_version": 1,
            "name": "synthetic-audit",
            "upc_config": str(self.upc_config_path),
            "scope_contract": {
                "pre_registered_before_execution": True,
                "allow_post_result_variant_expansion": False,
                "stop_after_declared_variants": True,
                "rationale": "test preregistration",
                "analytical_equivalence": "sum and mean have equal argmax",
            },
            "variants": variants,
            "decision": {
                "primary_model_protocol": "train_only",
                "sensitivity_model_protocol": "algorithm1_full_month",
                "diagnostic_variant": DIAGNOSTIC_VARIANT,
                "diagnostic_eligible_for_model_input": False,
                "exact_figure4_match_required_for_baselines": False,
                "exact_figure4_match_required_for_reproduction_claim": True,
            },
            "outputs": {
                "report_json": str(output_directory / "audit.json"),
                "comparison_csv": str(output_directory / "audit.csv"),
                "manifest": str(output_directory / "audit-manifest.json"),
            },
        }
        self.audit_config_path.write_text(json.dumps(audit_config), encoding="utf-8")

    def test_bounded_audit_is_deterministic_and_emits_no_membership_file(self) -> None:
        config = load_fig4_audit_config(
            self.audit_config_path, base_directory=self.root
        )
        first = run_fig4_audit(config)
        first_hashes = {
            name: metadata["sha256"] for name, metadata in first["outputs"].items()
        }
        second = run_fig4_audit(config)
        second_hashes = {
            name: metadata["sha256"] for name, metadata in second["outputs"].items()
        }

        self.assertEqual(first_hashes, second_hashes)
        self.assertEqual(first["summary"]["executed_variant_count"], 10)
        self.assertTrue(first["summary"]["audit_stopped_after_declared_variants"])
        self.assertEqual(set(first["outputs"]), {"report_json", "comparison_csv"})
        report = json.loads(config.outputs.report_json.read_text(encoding="utf-8"))
        self.assertTrue(
            all(not row["eligible_for_model_input"] for row in report["variants"])
        )

    def test_config_rejects_post_result_variant_expansion(self) -> None:
        raw = json.loads(self.audit_config_path.read_text(encoding="utf-8"))
        raw["variants"].append(dict(raw["variants"][-1], name="extra_variant"))
        self.audit_config_path.write_text(json.dumps(raw), encoding="utf-8")

        with self.assertRaisesRegex(Fig4AuditError, "사전 등록 목록"):
            load_fig4_audit_config(self.audit_config_path, base_directory=self.root)

    def test_config_rejects_registered_factor_mutation(self) -> None:
        raw = json.loads(self.audit_config_path.read_text(encoding="utf-8"))
        raw["variants"][0]["hourly_reducer"] = "max"
        self.audit_config_path.write_text(json.dumps(raw), encoding="utf-8")

        with self.assertRaisesRegex(Fig4AuditError, "사전 등록 계약"):
            load_fig4_audit_config(self.audit_config_path, base_directory=self.root)


if __name__ == "__main__":
    unittest.main()
