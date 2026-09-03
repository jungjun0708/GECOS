from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.select_central_900 import (
    CentralSelectionError,
    GridReference,
    GridSpec,
    SelectionSpec,
    compute_sha256,
    load_grid_cells,
    load_selection_config,
    map_selected_indices,
    reconstruct_grid,
    run_selection,
    select_cells,
    verify_grid_source,
)


def md5_bytes(payload: bytes) -> str:
    try:
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(payload).hexdigest()


class CentralSelectionTests(unittest.TestCase):
    rows = 6
    columns = 6
    steps = 8

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.grid_path = self.root / "milano-grid.geojson"
        self.reference_path = self.root / "milano-grid-reference.json"
        self.parent_manifest_path = self.root / "processed-manifest.json"
        self.config_path = self.root / "selection.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def transform(column: float, row: float) -> list[float]:
        # 실제 Milano Grid처럼 위도·경도 축에 약간 비스듬한 정사각 격자를 만든다.
        return [
            9.0 + column * 0.01 + row * 0.0001,
            45.0 + row * 0.01 - column * 0.00005,
        ]

    def grid_payload(self, *, reverse: bool = True) -> dict[str, object]:
        features: list[dict[str, object]] = []
        for row in range(self.rows):
            for column in range(self.columns):
                cell_id = row * self.columns + column + 1
                ring = [
                    self.transform(column, row),
                    self.transform(column + 1, row),
                    self.transform(column + 1, row + 1),
                    self.transform(column, row + 1),
                    self.transform(column, row),
                ]
                features.append(
                    {
                        "type": "Feature",
                        "id": cell_id - 1,
                        "properties": {"cellId": cell_id},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                )
        if reverse:
            features.reverse()
        return {
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
            },
            "type": "FeatureCollection",
            "features": features,
        }

    def write_grid(self, payload: dict[str, object] | None = None) -> bytes:
        content = json.dumps(
            payload or self.grid_payload(), separators=(",", ":")
        ).encode()
        self.grid_path.write_bytes(content)
        return content

    def grid_spec(self) -> GridSpec:
        return GridSpec(
            crs_name="urn:ogc:def:crs:EPSG::4326",
            cell_id_property="cellId",
            cell_id_min=1,
            cell_id_max=self.rows * self.columns,
            expected_feature_count=self.rows * self.columns,
            expected_rows=self.rows,
            expected_columns=self.columns,
        )

    def write_reference(self, content: bytes) -> None:
        payload = {
            "schema_version": 1,
            "source": {
                "title": "Synthetic Grid",
                "citation": "Synthetic test fixture",
                "persistent_id": "doi:test/grid",
                "doi_url": "https://example.test/grid",
                "dataset_version": "1.0",
                "license": "test-only",
                "license_url": "https://example.test/license",
            },
            "file": {
                "name": self.grid_path.name,
                "size_bytes": len(content),
                "checksum": {"algorithm": "md5", "value": md5_bytes(content)},
            },
            "acquisition": {"method": "test-fixture"},
        }
        self.reference_path.write_text(json.dumps(payload), encoding="utf-8")

    def write_processed_inputs(self) -> dict[str, Path]:
        ids = np.arange(1, self.rows * self.columns + 1, dtype=np.int32)[::-1]
        traffic = np.repeat(ids[:, None].astype(np.float32), self.steps, axis=1)
        timestamps = np.arange(self.steps, dtype=np.int64) * 600_000
        missing = np.zeros_like(traffic, dtype=bool)
        internet_null = np.zeros_like(traffic, dtype=bool)
        missing[0, 0] = True
        internet_null[1, 1] = True

        values = {
            "traffic": traffic,
            "cell_ids": ids,
            "timestamps_ms": timestamps,
            "missing_mask": missing,
            "internet_null_mask": internet_null,
        }
        paths: dict[str, Path] = {}
        output_metadata: dict[str, object] = {}
        for key, value in values.items():
            path = self.root / f"{key}.npy"
            np.save(path, value, allow_pickle=False)
            paths[key] = path
            output_metadata[key] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": compute_sha256(path),
            }
        self.parent_manifest_path.write_text(
            json.dumps({"schema_version": 1, "outputs": output_metadata}),
            encoding="utf-8",
        )
        return paths

    def write_config(self, input_paths: dict[str, Path]) -> None:
        payload = {
            "schema_version": 1,
            "name": "synthetic-central",
            "protocol": "central-900-approximate",
            "grid_reference_manifest": str(self.reference_path),
            "inputs": {
                "grid_geojson": str(self.grid_path),
                "processed_manifest": str(self.parent_manifest_path),
                **{key: str(path) for key, path in input_paths.items()},
            },
            "grid": {
                "crs_name": "urn:ogc:def:crs:EPSG::4326",
                "cell_id_property": "cellId",
                "cell_id_min": 1,
                "cell_id_max": self.rows * self.columns,
                "expected_feature_count": self.rows * self.columns,
                "expected_rows": self.rows,
                "expected_columns": self.columns,
            },
            "selection": {
                "row_start": 2,
                "row_end_exclusive": 4,
                "column_start": 2,
                "column_end_exclusive": 4,
                "expected_cell_count": 4,
            },
            "time": {"expected_steps": self.steps, "interval_ms": 600_000},
            "outputs": {
                "central_cells_csv": str(self.root / "outputs" / "central.csv"),
                "central_traffic": str(self.root / "outputs" / "traffic.npy"),
                "central_missing_mask": str(self.root / "outputs" / "missing.npy"),
                "central_internet_null_mask": str(self.root / "outputs" / "null.npy"),
                "manifest": str(self.root / "outputs" / "manifest.json"),
                "map_png": str(self.root / "outputs" / "map.png"),
            },
            "visualization": {
                "dpi": 72,
                "figure_width_inches": 2,
                "figure_height_inches": 2,
            },
        }
        self.config_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_coordinate_reconstruction_is_independent_of_feature_order(self) -> None:
        self.write_grid()
        cells, validation = load_grid_cells(self.grid_path, self.grid_spec())
        indexed, reconstruction = reconstruct_grid(cells, self.grid_spec())

        self.assertEqual(validation["feature_count"], 36)
        self.assertEqual(reconstruction["cell_id_formula_match_count"], 36)
        self.assertEqual(
            [(item.grid_row, item.grid_column, item.cell_id) for item in indexed],
            [
                (row, column, row * self.columns + column + 1)
                for row in range(self.rows)
                for column in range(self.columns)
            ],
        )

    def test_central_selection_uses_half_open_row_and_column_ranges(self) -> None:
        self.write_grid()
        cells, _ = load_grid_cells(self.grid_path, self.grid_spec())
        indexed, _ = reconstruct_grid(cells, self.grid_spec())
        selected = select_cells(
            indexed,
            SelectionSpec(2, 4, 2, 4, 4),
        )

        self.assertEqual([item.cell_id for item in selected], [15, 16, 21, 22])

    def test_duplicate_cell_id_is_rejected(self) -> None:
        payload = self.grid_payload()
        payload["features"][-1]["properties"]["cellId"] = 36  # type: ignore[index]
        payload["features"][-1]["id"] = 35  # type: ignore[index]
        self.write_grid(payload)

        with self.assertRaisesRegex(CentralSelectionError, "중복 cell ID"):
            load_grid_cells(self.grid_path, self.grid_spec())

    def test_open_polygon_ring_is_rejected(self) -> None:
        payload = self.grid_payload()
        ring = payload["features"][0]["geometry"]["coordinates"][0]  # type: ignore[index]
        ring[-1] = [0.0, 0.0]
        self.write_grid(payload)

        with self.assertRaisesRegex(CentralSelectionError, "닫혀 있지 않습니다"):
            load_grid_cells(self.grid_path, self.grid_spec())

    def test_coordinate_and_cell_id_formula_mismatch_is_rejected(self) -> None:
        payload = self.grid_payload(reverse=False)
        first = payload["features"][0]  # type: ignore[index]
        second = payload["features"][1]  # type: ignore[index]
        first["properties"]["cellId"], second["properties"]["cellId"] = (  # type: ignore[index]
            second["properties"]["cellId"],  # type: ignore[index]
            first["properties"]["cellId"],  # type: ignore[index]
        )
        first["id"], second["id"] = second["id"], first["id"]  # type: ignore[index]
        self.write_grid(payload)
        cells, _ = load_grid_cells(self.grid_path, self.grid_spec())

        with self.assertRaisesRegex(CentralSelectionError, "cell ID 공식"):
            reconstruct_grid(cells, self.grid_spec())

    def test_official_size_and_md5_are_both_required(self) -> None:
        content = self.write_grid()
        reference = GridReference(
            path=self.reference_path,
            source={"doi_url": "https://example.test/grid"},
            acquisition={},
            filename=self.grid_path.name,
            size_bytes=len(content),
            checksum_algorithm="md5",
            checksum=md5_bytes(content),
        )
        report = verify_grid_source(self.grid_path, reference)
        self.assertTrue(report["official_checksum_matched"])

        self.grid_path.write_bytes(content[:-1] + b" ")
        with self.assertRaisesRegex(CentralSelectionError, "MD5"):
            verify_grid_source(self.grid_path, reference)

    def test_selected_ids_map_to_rows_even_when_source_order_is_reversed(self) -> None:
        source_ids = np.asarray([4, 3, 2, 1], dtype=np.int32)
        self.write_grid()
        cells, _ = load_grid_cells(self.grid_path, self.grid_spec())
        indexed, _ = reconstruct_grid(cells, self.grid_spec())
        selected = [indexed[0], indexed[1], indexed[2], indexed[3]]

        indices = map_selected_indices(source_ids, selected)

        np.testing.assert_array_equal(indices, np.asarray([3, 2, 1, 0]))

    def test_full_pipeline_is_deterministic_without_map(self) -> None:
        content = self.write_grid()
        self.write_reference(content)
        input_paths = self.write_processed_inputs()
        self.write_config(input_paths)
        config = load_selection_config(self.config_path, base_directory=self.root)

        first = run_selection(config, skip_map=True)
        first_digests = {
            key: value["sha256"] for key, value in first["outputs"].items()
        }
        second = run_selection(config, skip_map=True)
        second_digests = {
            key: value["sha256"] for key, value in second["outputs"].items()
        }

        self.assertEqual(first_digests, second_digests)
        self.assertEqual(first["selection"]["cell_count"], 4)
        self.assertEqual(first["selection"]["first_cell_id"], 15)
        self.assertEqual(first["selection"]["last_cell_id"], 22)
        output = np.load(config.outputs.central_traffic, allow_pickle=False)
        self.assertEqual(output.shape, (4, self.steps))
        np.testing.assert_array_equal(output[:, 0], np.asarray([15, 16, 21, 22]))

    def test_invalid_input_fails_before_publishing_outputs(self) -> None:
        content = self.write_grid()
        self.write_reference(content)
        input_paths = self.write_processed_inputs()
        self.write_config(input_paths)
        config = load_selection_config(self.config_path, base_directory=self.root)
        self.grid_path.write_bytes(content[:-1] + b" ")

        with self.assertRaisesRegex(CentralSelectionError, "MD5"):
            run_selection(config, skip_map=True)

        for path in config.outputs.as_dict().values():
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
