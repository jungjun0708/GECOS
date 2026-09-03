from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scripts.preprocess_internet import (
    PreprocessingError,
    load_preprocess_config,
    preprocess_dataset,
)
from scripts.verify_raw_data import load_reference_manifest


def md5_bytes(payload: bytes) -> str:
    try:
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(payload).hexdigest()


def row(
    cell_id: int,
    timestamp_ms: int,
    country_code: int,
    internet: str,
) -> str:
    fields = [
        str(cell_id),
        str(timestamp_ms),
        str(country_code),
        "",
        "",
        "",
        "",
        internet,
    ]
    return "\t".join(fields) + "\n"


class PreprocessInternetTests(unittest.TestCase):
    start_ms = 1_577_836_800_000
    interval_ms = 600_000

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_directory = self.root / "raw"
        self.data_directory.mkdir()
        self.reference_path = self.root / "reference.json"
        self.config_path = self.root / "config.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_case(
        self,
        payload: bytes,
        *,
        expected_rows: int,
        output_directory: str = "processed",
    ) -> tuple[object, object]:
        filename = "day-01.txt"
        (self.data_directory / filename).write_bytes(payload)
        reference = {
            "schema_version": 1,
            "source": {
                "title": "synthetic dataset",
                "persistent_id": "doi:test/preprocessing",
                "doi_url": "https://example.test/preprocessing",
                "dataset_version": "1.0",
                "license": "test-only",
            },
            "selection": {
                "file_glob": "day-*.txt",
                "expected_file_count": 1,
                "expected_total_bytes": len(payload),
            },
            "integrity": {"algorithm": "md5"},
            "files": [
                {
                    "name": filename,
                    "size_bytes": len(payload),
                    "checksum": md5_bytes(payload),
                }
            ],
        }
        self.reference_path.write_text(json.dumps(reference), encoding="utf-8")

        outputs = self.root / output_directory
        config = {
            "schema_version": 1,
            "name": "synthetic",
            "raw_reference_manifest": str(self.reference_path),
            "expected_total_rows": expected_rows,
            "grid": {
                "cell_id_min": 1,
                "cell_id_max": 2,
                "expected_cell_count": 2,
            },
            "time": {
                "timezone": "UTC",
                "start_local": "2020-01-01T00:00:00",
                "end_exclusive_local": "2020-01-01T00:20:00",
                "interval_ms": self.interval_ms,
                "steps_per_file": 2,
                "expected_steps": 2,
            },
            "parser": {
                "block_size_bytes": 1024,
                "column_names": [
                    "cell_id",
                    "timestamp_ms",
                    "country_code",
                    "sms_in",
                    "sms_out",
                    "call_in",
                    "call_out",
                    "internet",
                ],
            },
            "aggregation": {
                "target": "internet",
                "null_value": 0.0,
                "accumulator_dtype": "float64",
                "output_dtype": "float32",
            },
            "parquet": {
                "compression": "zstd",
                "compression_level": 3,
                "row_group_rows": 2,
            },
            "outputs": {
                "interim_parquet": str(outputs / "internet.parquet"),
                "traffic": str(outputs / "traffic.npy"),
                "cell_ids": str(outputs / "cell_ids.npy"),
                "timestamps_ms": str(outputs / "timestamps_ms.npy"),
                "missing_mask": str(outputs / "missing_mask.npy"),
                "internet_null_mask": str(outputs / "internet_null_mask.npy"),
                "manifest": str(outputs / "manifest.json"),
            },
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        loaded_config = load_preprocess_config(
            self.config_path, base_directory=self.root
        )
        loaded_reference = load_reference_manifest(self.reference_path)
        return loaded_config, loaded_reference

    def valid_rows(self) -> list[str]:
        return [
            row(1, self.start_ms, 39, "1.5"),
            row(1, self.start_ms, 40, "2.5"),
            row(2, self.start_ms, 39, ""),
            row(2, self.start_ms + self.interval_ms, 39, "3.0"),
        ]

    def test_aggregation_masks_order_and_manifest(self) -> None:
        payload = "".join(self.valid_rows()).encode()
        config, reference = self.write_case(payload, expected_rows=4)

        manifest = preprocess_dataset(config, reference, self.data_directory)

        np.testing.assert_array_equal(
            np.load(config.outputs.traffic),
            np.array([[4.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.load(config.outputs.missing_mask),
            np.array([[False, True], [False, False]], dtype=bool),
        )
        np.testing.assert_array_equal(
            np.load(config.outputs.internet_null_mask),
            np.array([[False, False], [True, False]], dtype=bool),
        )
        np.testing.assert_array_equal(
            np.load(config.outputs.cell_ids), np.array([1, 2], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            np.load(config.outputs.timestamps_ms),
            np.array([self.start_ms, self.start_ms + self.interval_ms], dtype=np.int64),
        )

        table = pq.read_table(config.outputs.interim_parquet)
        self.assertEqual(table.column("cell_id").to_pylist(), [1, 1, 2, 2])
        self.assertEqual(
            table.column("timestamp_ms").to_pylist(),
            [
                self.start_ms,
                self.start_ms + self.interval_ms,
                self.start_ms,
                self.start_ms + self.interval_ms,
            ],
        )
        self.assertEqual(table.column("internet").to_pylist(), [4.0, 0.0, 0.0, 3.0])
        self.assertEqual(manifest["statistics"]["raw_rows"], 4)
        self.assertEqual(manifest["statistics"]["activity_null_rows"]["internet"], 1)
        self.assertEqual(manifest["statistics"]["missing_pair_count"], 1)
        self.assertEqual(manifest["statistics"]["internet_all_null_pair_count"], 1)
        self.assertEqual(manifest["parquet"]["rows"], 4)

        repeated_manifest = preprocess_dataset(config, reference, self.data_directory)
        first_hashes = {
            key: value["sha256"] for key, value in manifest["outputs"].items()
        }
        repeated_hashes = {
            key: value["sha256"] for key, value in repeated_manifest["outputs"].items()
        }
        self.assertEqual(repeated_hashes, first_hashes)

    def test_input_row_order_does_not_change_outputs(self) -> None:
        rows = self.valid_rows()
        config_a, reference_a = self.write_case(
            "".join(rows).encode(), expected_rows=4, output_directory="run-a"
        )
        preprocess_dataset(config_a, reference_a, self.data_directory)
        traffic_a = np.load(config_a.outputs.traffic).copy()
        missing_a = np.load(config_a.outputs.missing_mask).copy()
        null_a = np.load(config_a.outputs.internet_null_mask).copy()

        config_b, reference_b = self.write_case(
            "".join(reversed(rows)).encode(),
            expected_rows=4,
            output_directory="run-b",
        )
        preprocess_dataset(config_b, reference_b, self.data_directory)

        np.testing.assert_array_equal(np.load(config_b.outputs.traffic), traffic_a)
        np.testing.assert_array_equal(np.load(config_b.outputs.missing_mask), missing_a)
        np.testing.assert_array_equal(
            np.load(config_b.outputs.internet_null_mask), null_a
        )

    def test_wrong_column_count_is_rejected_without_publishing_outputs(self) -> None:
        malformed = (
            row(1, self.start_ms, 39, "1.0").rstrip("\n") + "\textra\n"
        ).encode()
        config, reference = self.write_case(malformed, expected_rows=1)

        with self.assertRaisesRegex(PreprocessingError, "TSV 파싱에 실패"):
            preprocess_dataset(config, reference, self.data_directory)

        self.assertFalse(config.outputs.traffic.exists())
        self.assertFalse(config.outputs.manifest.exists())
        self.assertEqual(list(config.outputs.traffic.parent.glob("*.partial")), [])

    def test_out_of_range_cell_is_rejected(self) -> None:
        payload = row(3, self.start_ms, 39, "1.0").encode()
        config, reference = self.write_case(payload, expected_rows=1)

        with self.assertRaisesRegex(PreprocessingError, "cell_id"):
            preprocess_dataset(config, reference, self.data_directory)

    def test_misaligned_timestamp_is_rejected(self) -> None:
        payload = row(1, self.start_ms + 1, 39, "1.0").encode()
        config, reference = self.write_case(payload, expected_rows=1)

        with self.assertRaisesRegex(PreprocessingError, "timestamp"):
            preprocess_dataset(config, reference, self.data_directory)

    def test_negative_activity_is_rejected(self) -> None:
        payload = row(1, self.start_ms, 39, "-1.0").encode()
        config, reference = self.write_case(payload, expected_rows=1)

        with self.assertRaisesRegex(PreprocessingError, "음수"):
            preprocess_dataset(config, reference, self.data_directory)

    def test_checksum_mismatch_stops_before_creating_outputs(self) -> None:
        payload = row(1, self.start_ms, 39, "1.0").encode()
        config, reference = self.write_case(payload, expected_rows=1)
        (self.data_directory / "day-01.txt").write_bytes(
            row(1, self.start_ms, 39, "2.0").encode()
        )

        with self.assertRaisesRegex(PreprocessingError, "무결성"):
            preprocess_dataset(config, reference, self.data_directory)

        self.assertFalse(config.outputs.traffic.exists())
        self.assertFalse(config.outputs.manifest.exists())

    def test_config_rejects_inconsistent_cell_count(self) -> None:
        payload = row(1, self.start_ms, 39, "1.0").encode()
        self.write_case(payload, expected_rows=1)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["grid"]["expected_cell_count"] = 3
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        with self.assertRaisesRegex(PreprocessingError, "cell ID 범위"):
            load_preprocess_config(self.config_path, base_directory=self.root)


if __name__ == "__main__":
    unittest.main()
