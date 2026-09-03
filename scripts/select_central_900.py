#!/usr/bin/env python3
"""Milano Grid에서 중앙 30×30 셀을 검증하고 학습용 부분집합을 만든다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from scripts.verify_raw_data import compute_digest

TOOL_VERSION = "1.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "select_central_900.json"
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CentralSelectionError(RuntimeError):
    """공간 데이터 또는 선택 계약을 만족하지 못할 때 발생한다."""


@dataclass(frozen=True)
class GridReference:
    path: Path
    source: Mapping[str, Any]
    acquisition: Mapping[str, Any]
    filename: str
    size_bytes: int
    checksum_algorithm: str
    checksum: str


@dataclass(frozen=True)
class InputPaths:
    grid_geojson: Path
    processed_manifest: Path
    traffic: Path
    cell_ids: Path
    timestamps_ms: Path
    missing_mask: Path
    internet_null_mask: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "grid_geojson": self.grid_geojson,
            "processed_manifest": self.processed_manifest,
            "traffic": self.traffic,
            "cell_ids": self.cell_ids,
            "timestamps_ms": self.timestamps_ms,
            "missing_mask": self.missing_mask,
            "internet_null_mask": self.internet_null_mask,
        }


@dataclass(frozen=True)
class OutputPaths:
    central_cells_csv: Path
    central_traffic: Path
    central_missing_mask: Path
    central_internet_null_mask: Path
    manifest: Path
    map_png: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "central_cells_csv": self.central_cells_csv,
            "central_traffic": self.central_traffic,
            "central_missing_mask": self.central_missing_mask,
            "central_internet_null_mask": self.central_internet_null_mask,
            "manifest": self.manifest,
            "map_png": self.map_png,
        }


@dataclass(frozen=True)
class GridSpec:
    crs_name: str
    cell_id_property: str
    cell_id_min: int
    cell_id_max: int
    expected_feature_count: int
    expected_rows: int
    expected_columns: int


@dataclass(frozen=True)
class SelectionSpec:
    row_start: int
    row_end_exclusive: int
    column_start: int
    column_end_exclusive: int
    expected_cell_count: int


@dataclass(frozen=True)
class VisualizationSpec:
    dpi: int
    figure_width_inches: float
    figure_height_inches: float


@dataclass(frozen=True)
class SelectionConfig:
    path: Path
    name: str
    protocol: str
    grid_reference_manifest: Path
    inputs: InputPaths
    outputs: OutputPaths
    grid: GridSpec
    selection: SelectionSpec
    expected_steps: int
    interval_ms: int
    visualization: VisualizationSpec


@dataclass(frozen=True)
class GridCell:
    cell_id: int
    feature_id: int
    ring: tuple[tuple[float, float], ...]
    centroid_lon: float
    centroid_lat: float
    signed_area: float


@dataclass(frozen=True)
class IndexedCell:
    cell: GridCell
    grid_row: int
    grid_column: int

    @property
    def cell_id(self) -> int:
        return self.cell.cell_id


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CentralSelectionError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CentralSelectionError(f"{field}는 JSON array여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CentralSelectionError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CentralSelectionError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CentralSelectionError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CentralSelectionError(
            f"{field}는 {minimum} 이상의 유한한 숫자여야 합니다."
        )
    return result


def _resolve_path(value: object, field: str, base_directory: Path) -> Path:
    raw_path = Path(_require_string(value, field))
    return (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (base_directory / raw_path).resolve()
    )


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CentralSelectionError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CentralSelectionError(f"{label}이 올바른 JSON이 아닙니다: {exc}") from exc
    return _require_mapping(payload, label)


def load_selection_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> SelectionConfig:
    """중앙 셀 선택 설정을 읽고 서로 모순되는 값을 차단한다."""

    root = _load_json(path, "선택 config")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise CentralSelectionError(
            f"지원하지 않는 schema_version입니다: {schema_version}"
        )

    name = _require_string(root.get("name"), "name")
    protocol = _require_string(root.get("protocol"), "protocol")
    if protocol != "central-900-approximate":
        raise CentralSelectionError(
            "현재 구현은 protocol=central-900-approximate만 지원합니다."
        )

    reference_path = _resolve_path(
        root.get("grid_reference_manifest"),
        "grid_reference_manifest",
        base_directory,
    )

    raw_inputs = _require_mapping(root.get("inputs"), "inputs")
    inputs = InputPaths(
        **{
            field: _resolve_path(
                raw_inputs.get(field), f"inputs.{field}", base_directory
            )
            for field in InputPaths.__dataclass_fields__
        }
    )

    raw_outputs = _require_mapping(root.get("outputs"), "outputs")
    outputs = OutputPaths(
        **{
            field: _resolve_path(
                raw_outputs.get(field), f"outputs.{field}", base_directory
            )
            for field in OutputPaths.__dataclass_fields__
        }
    )
    output_values = list(outputs.as_dict().values())
    if len(set(output_values)) != len(output_values):
        raise CentralSelectionError("outputs 경로가 서로 중복됩니다.")

    raw_grid = _require_mapping(root.get("grid"), "grid")
    grid = GridSpec(
        crs_name=_require_string(raw_grid.get("crs_name"), "grid.crs_name"),
        cell_id_property=_require_string(
            raw_grid.get("cell_id_property"), "grid.cell_id_property"
        ),
        cell_id_min=_require_int(raw_grid.get("cell_id_min"), "grid.cell_id_min"),
        cell_id_max=_require_int(raw_grid.get("cell_id_max"), "grid.cell_id_max"),
        expected_feature_count=_require_int(
            raw_grid.get("expected_feature_count"),
            "grid.expected_feature_count",
            minimum=1,
        ),
        expected_rows=_require_int(
            raw_grid.get("expected_rows"), "grid.expected_rows", minimum=1
        ),
        expected_columns=_require_int(
            raw_grid.get("expected_columns"), "grid.expected_columns", minimum=1
        ),
    )
    if grid.cell_id_max - grid.cell_id_min + 1 != grid.expected_feature_count:
        raise CentralSelectionError(
            "grid의 cell ID 범위와 feature 수가 일치하지 않습니다."
        )
    if grid.expected_rows * grid.expected_columns != grid.expected_feature_count:
        raise CentralSelectionError("grid의 행×열과 feature 수가 일치하지 않습니다.")

    raw_selection = _require_mapping(root.get("selection"), "selection")
    selection = SelectionSpec(
        row_start=_require_int(raw_selection.get("row_start"), "selection.row_start"),
        row_end_exclusive=_require_int(
            raw_selection.get("row_end_exclusive"), "selection.row_end_exclusive"
        ),
        column_start=_require_int(
            raw_selection.get("column_start"), "selection.column_start"
        ),
        column_end_exclusive=_require_int(
            raw_selection.get("column_end_exclusive"),
            "selection.column_end_exclusive",
        ),
        expected_cell_count=_require_int(
            raw_selection.get("expected_cell_count"),
            "selection.expected_cell_count",
            minimum=1,
        ),
    )
    if not 0 <= selection.row_start < selection.row_end_exclusive <= grid.expected_rows:
        raise CentralSelectionError("selection 행 범위가 grid 범위를 벗어납니다.")
    if not (
        0
        <= selection.column_start
        < selection.column_end_exclusive
        <= grid.expected_columns
    ):
        raise CentralSelectionError("selection 열 범위가 grid 범위를 벗어납니다.")
    calculated_count = (selection.row_end_exclusive - selection.row_start) * (
        selection.column_end_exclusive - selection.column_start
    )
    if calculated_count != selection.expected_cell_count:
        raise CentralSelectionError(
            "selection 범위로 계산한 셀 수와 expected_cell_count가 다릅니다."
        )

    raw_time = _require_mapping(root.get("time"), "time")
    expected_steps = _require_int(
        raw_time.get("expected_steps"), "time.expected_steps", minimum=2
    )
    interval_ms = _require_int(
        raw_time.get("interval_ms"), "time.interval_ms", minimum=1
    )

    raw_visualization = _require_mapping(root.get("visualization"), "visualization")
    visualization = VisualizationSpec(
        dpi=_require_int(raw_visualization.get("dpi"), "visualization.dpi", minimum=72),
        figure_width_inches=_require_number(
            raw_visualization.get("figure_width_inches"),
            "visualization.figure_width_inches",
            minimum=1.0,
        ),
        figure_height_inches=_require_number(
            raw_visualization.get("figure_height_inches"),
            "visualization.figure_height_inches",
            minimum=1.0,
        ),
    )

    return SelectionConfig(
        path=path.resolve(),
        name=name,
        protocol=protocol,
        grid_reference_manifest=reference_path,
        inputs=inputs,
        outputs=outputs,
        grid=grid,
        selection=selection,
        expected_steps=expected_steps,
        interval_ms=interval_ms,
        visualization=visualization,
    )


def load_grid_reference(path: Path) -> GridReference:
    """공식 Milano Grid 메타데이터에서 파일 검증 기준을 읽는다."""

    root = _load_json(path, "Grid 기준 manifest")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise CentralSelectionError(
            f"지원하지 않는 Grid manifest입니다: {schema_version}"
        )

    source = _require_mapping(root.get("source"), "source")
    for field in (
        "title",
        "citation",
        "persistent_id",
        "doi_url",
        "dataset_version",
        "license",
        "license_url",
    ):
        _require_string(source.get(field), f"source.{field}")

    file_info = _require_mapping(root.get("file"), "file")
    filename = _require_string(file_info.get("name"), "file.name")
    if Path(filename).name != filename:
        raise CentralSelectionError("file.name에는 디렉터리를 포함할 수 없습니다.")
    size_bytes = _require_int(file_info.get("size_bytes"), "file.size_bytes", minimum=1)
    checksum_info = _require_mapping(file_info.get("checksum"), "file.checksum")
    algorithm = _require_string(
        checksum_info.get("algorithm"), "file.checksum.algorithm"
    ).lower()
    checksum = _require_string(
        checksum_info.get("value"), "file.checksum.value"
    ).lower()
    if algorithm != "md5" or not MD5_PATTERN.fullmatch(checksum):
        raise CentralSelectionError("Grid 기준 checksum은 올바른 MD5여야 합니다.")

    acquisition = _require_mapping(root.get("acquisition"), "acquisition")
    return GridReference(
        path=path.resolve(),
        source=dict(source),
        acquisition=dict(acquisition),
        filename=filename,
        size_bytes=size_bytes,
        checksum_algorithm=algorithm,
        checksum=checksum,
    )


def compute_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(chunk_size)
    view = memoryview(buffer)
    try:
        with path.open("rb", buffering=0) as handle:
            while read_count := handle.readinto(buffer):
                digest.update(view[:read_count])
    except OSError as exc:
        raise CentralSelectionError(
            f"파일 checksum을 계산할 수 없습니다: {path}"
        ) from exc
    return digest.hexdigest()


def verify_grid_source(path: Path, reference: GridReference) -> dict[str, Any]:
    """로컬 GeoJSON이 공식 Dataverse 크기와 MD5에 일치하는지 확인한다."""

    if not path.is_file():
        raise CentralSelectionError(
            f"Milano Grid 파일이 없습니다: {path}\n"
            f"다운로드 안내: {reference.source['doi_url']}"
        )
    if path.name != reference.filename:
        raise CentralSelectionError(
            f"Grid 파일명이 기준과 다릅니다: {path.name} != {reference.filename}"
        )
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise CentralSelectionError(
            f"Grid 파일 정보를 읽을 수 없습니다: {path}"
        ) from exc
    if actual_size != reference.size_bytes:
        raise CentralSelectionError(
            "Grid 파일 크기가 공식 기준과 다릅니다: "
            f"{actual_size} != {reference.size_bytes}"
        )
    actual_md5 = compute_digest(path, reference.checksum_algorithm)
    if actual_md5 != reference.checksum:
        raise CentralSelectionError(
            f"Grid 파일 MD5가 공식 기준과 다릅니다: {actual_md5}"
        )
    return {
        "path": _display_path(path),
        "size_bytes": actual_size,
        "md5": actual_md5,
        "sha256": compute_sha256(path),
        "official_checksum_matched": True,
    }


def _coordinate(value: object, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) < 2:
        raise CentralSelectionError(f"{field}는 [longitude, latitude] 좌표여야 합니다.")
    lon = _require_number_allow_negative(value[0], f"{field}[0]")
    lat = _require_number_allow_negative(value[1], f"{field}[1]")
    if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
        raise CentralSelectionError(f"{field}가 유효한 경도·위도 범위를 벗어납니다.")
    return lon, lat


def _require_number_allow_negative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CentralSelectionError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise CentralSelectionError(f"{field}는 유한한 숫자여야 합니다.")
    return result


def polygon_centroid(
    raw_coordinates: object,
    *,
    feature_label: str,
) -> tuple[tuple[tuple[float, float], ...], float, float, float]:
    """구멍이 없는 GeoJSON Polygon의 면적 중심을 shoelace 공식으로 계산한다."""

    rings = _require_list(raw_coordinates, f"{feature_label}.geometry.coordinates")
    if len(rings) != 1:
        raise CentralSelectionError(
            f"{feature_label}는 구멍이 없는 단일 exterior ring이어야 합니다."
        )
    raw_ring = _require_list(rings[0], f"{feature_label}.geometry.coordinates[0]")
    if len(raw_ring) < 4:
        raise CentralSelectionError(f"{feature_label} polygon 꼭짓점이 부족합니다.")
    ring = tuple(
        _coordinate(point, f"{feature_label}.geometry.coordinates[0][{index}]")
        for index, point in enumerate(raw_ring)
    )
    if ring[0] != ring[-1]:
        raise CentralSelectionError(
            f"{feature_label} polygon ring이 닫혀 있지 않습니다."
        )

    cross_sum = 0.0
    centroid_x_numerator = 0.0
    centroid_y_numerator = 0.0
    for (x0, y0), (x1, y1) in pairwise(ring):
        cross = x0 * y1 - x1 * y0
        cross_sum += cross
        centroid_x_numerator += (x0 + x1) * cross
        centroid_y_numerator += (y0 + y1) * cross
    signed_area = cross_sum / 2.0
    if abs(signed_area) <= 1e-15:
        raise CentralSelectionError(f"{feature_label} polygon 면적이 0입니다.")
    centroid_lon = centroid_x_numerator / (6.0 * signed_area)
    centroid_lat = centroid_y_numerator / (6.0 * signed_area)
    if not (math.isfinite(centroid_lon) and math.isfinite(centroid_lat)):
        raise CentralSelectionError(f"{feature_label} centroid가 유효하지 않습니다.")
    return ring, centroid_lon, centroid_lat, signed_area


def load_grid_cells(
    path: Path, spec: GridSpec
) -> tuple[tuple[GridCell, ...], dict[str, Any]]:
    """GeoJSON 구조와 셀 ID를 검증하고 polygon centroid를 계산한다."""

    root = _load_json(path, "Milano Grid GeoJSON")
    if root.get("type") != "FeatureCollection":
        raise CentralSelectionError(
            "GeoJSON root.type은 FeatureCollection이어야 합니다."
        )

    crs = _require_mapping(root.get("crs"), "crs")
    crs_properties = _require_mapping(crs.get("properties"), "crs.properties")
    actual_crs = _require_string(crs_properties.get("name"), "crs.properties.name")
    if actual_crs != spec.crs_name:
        raise CentralSelectionError(f"GeoJSON CRS가 다릅니다: {actual_crs}")

    features = _require_list(root.get("features"), "features")
    if len(features) != spec.expected_feature_count:
        raise CentralSelectionError(
            f"GeoJSON feature 수가 다릅니다: {len(features)} != {spec.expected_feature_count}"
        )

    cells: list[GridCell] = []
    seen_ids: set[int] = set()
    seen_centroids: set[tuple[float, float]] = set()
    ring_vertex_counts: set[int] = set()
    for index, raw_feature in enumerate(features):
        label = f"features[{index}]"
        feature = _require_mapping(raw_feature, label)
        if feature.get("type") != "Feature":
            raise CentralSelectionError(f"{label}.type은 Feature여야 합니다.")
        feature_id = _require_int(feature.get("id"), f"{label}.id")
        properties = _require_mapping(feature.get("properties"), f"{label}.properties")
        cell_id = _require_int(
            properties.get(spec.cell_id_property),
            f"{label}.properties.{spec.cell_id_property}",
            minimum=spec.cell_id_min,
        )
        if cell_id > spec.cell_id_max:
            raise CentralSelectionError(
                f"{label} cell ID가 범위를 벗어납니다: {cell_id}"
            )
        if cell_id in seen_ids:
            raise CentralSelectionError(f"중복 cell ID가 있습니다: {cell_id}")
        if feature_id != cell_id - spec.cell_id_min:
            raise CentralSelectionError(
                f"{label} feature id와 cell ID 관계가 다릅니다: {feature_id}, {cell_id}"
            )

        geometry = _require_mapping(feature.get("geometry"), f"{label}.geometry")
        if geometry.get("type") != "Polygon":
            raise CentralSelectionError(
                f"{label}.geometry.type은 Polygon이어야 합니다."
            )
        ring, centroid_lon, centroid_lat, signed_area = polygon_centroid(
            geometry.get("coordinates"), feature_label=label
        )
        centroid_key = (round(centroid_lon, 12), round(centroid_lat, 12))
        if centroid_key in seen_centroids:
            raise CentralSelectionError(f"중복 centroid가 있습니다: cell ID {cell_id}")

        seen_ids.add(cell_id)
        seen_centroids.add(centroid_key)
        ring_vertex_counts.add(len(ring))
        cells.append(
            GridCell(
                cell_id=cell_id,
                feature_id=feature_id,
                ring=ring,
                centroid_lon=centroid_lon,
                centroid_lat=centroid_lat,
                signed_area=signed_area,
            )
        )

    expected_ids = set(range(spec.cell_id_min, spec.cell_id_max + 1))
    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)[:10]
        unexpected = sorted(seen_ids - expected_ids)[:10]
        raise CentralSelectionError(
            f"GeoJSON cell ID 집합이 다릅니다: missing={missing}, unexpected={unexpected}"
        )

    cells.sort(key=lambda cell: cell.cell_id)
    area_values = np.fromiter(
        (abs(cell.signed_area) for cell in cells), dtype=np.float64
    )
    return tuple(cells), {
        "crs": actual_crs,
        "feature_count": len(cells),
        "cell_id_min": min(seen_ids),
        "cell_id_max": max(seen_ids),
        "unique_cell_count": len(seen_ids),
        "geometry_type": "Polygon",
        "ring_vertex_counts": sorted(ring_vertex_counts),
        "polygon_area_degrees_squared_min": float(area_values.min()),
        "polygon_area_degrees_squared_max": float(area_values.max()),
    }


def reconstruct_grid(
    cells: Sequence[GridCell],
    spec: GridSpec,
) -> tuple[tuple[IndexedCell, ...], dict[str, Any]]:
    """centroid로 100×100 위치를 복원하고 공식 cell ID 순서와 교차 검증한다."""

    if len(cells) != spec.expected_rows * spec.expected_columns:
        raise CentralSelectionError("격자를 복원하기 위한 cell 수가 맞지 않습니다.")

    latitude_sorted = sorted(
        cells,
        key=lambda cell: (cell.centroid_lat, cell.centroid_lon, cell.cell_id),
    )
    row_groups: list[list[GridCell]] = []
    for row in range(spec.expected_rows):
        start = row * spec.expected_columns
        group = latitude_sorted[start : start + spec.expected_columns]
        if len(group) != spec.expected_columns:
            raise CentralSelectionError(f"복원한 row {row}의 cell 수가 맞지 않습니다.")
        row_groups.append(group)

    row_band_gaps: list[float] = []
    for row in range(spec.expected_rows - 1):
        current_max = max(cell.centroid_lat for cell in row_groups[row])
        next_min = min(cell.centroid_lat for cell in row_groups[row + 1])
        gap = next_min - current_max
        if gap <= 0:
            raise CentralSelectionError(
                f"centroid latitude만으로 row {row}와 {row + 1}을 분리할 수 없습니다."
            )
        row_band_gaps.append(gap)

    indexed: list[IndexedCell] = []
    for row, group in enumerate(row_groups):
        longitude_sorted = sorted(
            group,
            key=lambda cell: (cell.centroid_lon, cell.centroid_lat, cell.cell_id),
        )
        longitudes = [cell.centroid_lon for cell in longitude_sorted]
        if any(right <= left for left, right in pairwise(longitudes)):
            raise CentralSelectionError(
                f"복원한 row {row}의 longitude가 증가하지 않습니다."
            )
        indexed.extend(
            IndexedCell(cell=cell, grid_row=row, grid_column=column)
            for column, cell in enumerate(longitude_sorted)
        )

    mismatches: list[dict[str, int]] = []
    for item in indexed:
        expected_id = (
            item.grid_row * spec.expected_columns + item.grid_column + spec.cell_id_min
        )
        if item.cell_id != expected_id:
            mismatches.append(
                {
                    "cell_id": item.cell_id,
                    "expected_id": expected_id,
                    "grid_row": item.grid_row,
                    "grid_column": item.grid_column,
                }
            )
            if len(mismatches) == 10:
                break
    if mismatches:
        raise CentralSelectionError(
            "centroid로 복원한 위치와 cell ID 공식이 일치하지 않습니다: "
            + json.dumps(mismatches, ensure_ascii=False)
        )

    indexed.sort(key=lambda item: (item.grid_row, item.grid_column))
    return tuple(indexed), {
        "rows": spec.expected_rows,
        "columns": spec.expected_columns,
        "row_direction": "south-to-north (centroid latitude ascending)",
        "column_direction": "west-to-east (centroid longitude ascending per row)",
        "minimum_gap_between_latitude_bands": min(row_band_gaps),
        "cell_id_formula": "cell_id = grid_row * expected_columns + grid_column + cell_id_min",
        "cell_id_formula_match_count": len(indexed),
    }


def select_cells(
    indexed_cells: Sequence[IndexedCell],
    spec: SelectionSpec,
) -> tuple[IndexedCell, ...]:
    selected = tuple(
        item
        for item in indexed_cells
        if spec.row_start <= item.grid_row < spec.row_end_exclusive
        and spec.column_start <= item.grid_column < spec.column_end_exclusive
    )
    if len(selected) != spec.expected_cell_count:
        raise CentralSelectionError(
            f"중앙 선택 cell 수가 다릅니다: {len(selected)} != {spec.expected_cell_count}"
        )
    if len({item.cell_id for item in selected}) != len(selected):
        raise CentralSelectionError("중앙 선택 결과에 중복 cell ID가 있습니다.")
    expected_rows = set(range(spec.row_start, spec.row_end_exclusive))
    expected_columns = set(range(spec.column_start, spec.column_end_exclusive))
    if {item.grid_row for item in selected} != expected_rows:
        raise CentralSelectionError("중앙 선택 결과의 row 집합이 다릅니다.")
    if {item.grid_column for item in selected} != expected_columns:
        raise CentralSelectionError("중앙 선택 결과의 column 집합이 다릅니다.")
    return selected


def _load_npy(path: Path, label: str) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CentralSelectionError(f"{label}을 읽을 수 없습니다: {path}") from exc


def verify_processed_inputs(
    config: SelectionConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """전처리 manifest와 실제 NumPy 파일의 checksum·shape를 다시 확인한다."""

    parent_manifest = _load_json(config.inputs.processed_manifest, "전처리 manifest")
    raw_outputs = _require_mapping(parent_manifest.get("outputs"), "processed.outputs")
    input_keys = (
        "traffic",
        "cell_ids",
        "timestamps_ms",
        "missing_mask",
        "internet_null_mask",
    )
    input_digests: dict[str, Any] = {}
    for key in input_keys:
        path = getattr(config.inputs, key)
        if not path.is_file():
            raise CentralSelectionError(f"전처리 입력이 없습니다: {path}")
        metadata = _require_mapping(raw_outputs.get(key), f"processed.outputs.{key}")
        expected_size = _require_int(
            metadata.get("size_bytes"), f"processed.outputs.{key}.size_bytes", minimum=1
        )
        expected_sha256 = _require_string(
            metadata.get("sha256"), f"processed.outputs.{key}.sha256"
        ).lower()
        if not SHA256_PATTERN.fullmatch(expected_sha256):
            raise CentralSelectionError(
                f"processed.outputs.{key}.sha256이 잘못되었습니다."
            )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise CentralSelectionError(
                f"{key} 크기가 전처리 manifest와 다릅니다: {actual_size} != {expected_size}"
            )
        actual_sha256 = compute_sha256(path)
        if actual_sha256 != expected_sha256:
            raise CentralSelectionError(f"{key} checksum이 전처리 manifest와 다릅니다.")
        input_digests[key] = {
            "path": _display_path(path),
            "size_bytes": actual_size,
            "sha256": actual_sha256,
        }

    arrays = {key: _load_npy(getattr(config.inputs, key), key) for key in input_keys}
    traffic = arrays["traffic"]
    cell_ids = arrays["cell_ids"]
    timestamps = arrays["timestamps_ms"]
    missing_mask = arrays["missing_mask"]
    null_mask = arrays["internet_null_mask"]

    expected_shape = (config.grid.expected_feature_count, config.expected_steps)
    if traffic.shape != expected_shape or traffic.dtype != np.dtype("float32"):
        raise CentralSelectionError(
            f"traffic 계약이 다릅니다: shape={traffic.shape}, dtype={traffic.dtype}"
        )
    if (
        cell_ids.shape != (config.grid.expected_feature_count,)
        or cell_ids.dtype.kind not in "iu"
    ):
        raise CentralSelectionError(
            f"cell_ids 계약이 다릅니다: shape={cell_ids.shape}, dtype={cell_ids.dtype}"
        )
    expected_ids = np.arange(
        config.grid.cell_id_min,
        config.grid.cell_id_max + 1,
        dtype=np.int64,
    )
    if not np.array_equal(np.sort(np.asarray(cell_ids, dtype=np.int64)), expected_ids):
        raise CentralSelectionError("cell_ids 값이 Grid의 1~10,000 ID 집합과 다릅니다.")
    if (
        timestamps.shape != (config.expected_steps,)
        or timestamps.dtype.kind not in "iu"
    ):
        raise CentralSelectionError(
            f"timestamps_ms 계약이 다릅니다: shape={timestamps.shape}, dtype={timestamps.dtype}"
        )
    if not np.all(
        np.diff(np.asarray(timestamps, dtype=np.int64)) == config.interval_ms
    ):
        raise CentralSelectionError("timestamps_ms 간격이 config와 다릅니다.")
    for label, mask in (
        ("missing_mask", missing_mask),
        ("internet_null_mask", null_mask),
    ):
        if mask.shape != expected_shape or mask.dtype != np.dtype("bool"):
            raise CentralSelectionError(
                f"{label} 계약이 다릅니다: shape={mask.shape}, dtype={mask.dtype}"
            )

    for start in range(0, expected_shape[0], 256):
        stop = min(start + 256, expected_shape[0])
        traffic_block = np.asarray(traffic[start:stop])
        if not np.all(np.isfinite(traffic_block)) or np.any(traffic_block < 0):
            raise CentralSelectionError(
                f"traffic에 유효하지 않은 값이 있습니다: rows {start}:{stop}"
            )
        if np.any(
            np.asarray(missing_mask[start:stop]) & np.asarray(null_mask[start:stop])
        ):
            raise CentralSelectionError("두 결측 mask가 같은 위치에서 겹칩니다.")

    return arrays, {
        "manifest_path": _display_path(config.inputs.processed_manifest),
        "manifest_sha256": compute_sha256(config.inputs.processed_manifest),
        "inputs": input_digests,
        "traffic_shape": list(traffic.shape),
        "traffic_dtype": str(traffic.dtype),
        "timestamp_count": int(timestamps.size),
        "timestamp_interval_ms": config.interval_ms,
    }


def map_selected_indices(
    source_cell_ids: np.ndarray,
    selected: Sequence[IndexedCell],
) -> np.ndarray:
    id_to_index = {
        int(cell_id): index for index, cell_id in enumerate(np.asarray(source_cell_ids))
    }
    if len(id_to_index) != source_cell_ids.size:
        raise CentralSelectionError("전처리 cell_ids에 중복 값이 있습니다.")
    missing = [item.cell_id for item in selected if item.cell_id not in id_to_index]
    if missing:
        raise CentralSelectionError(
            f"전처리 행렬에 선택 cell ID가 없습니다: {missing[:10]}"
        )
    return np.asarray([id_to_index[item.cell_id] for item in selected], dtype=np.int64)


def calculate_traffic_statistics(
    arrays: Mapping[str, np.ndarray],
    selected_indices: np.ndarray,
) -> dict[str, Any]:
    """중앙과 외부의 트래픽·결측 통계를 메모리 제한 방식으로 계산한다."""

    traffic = arrays["traffic"]
    missing_mask = arrays["missing_mask"]
    null_mask = arrays["internet_null_mask"]
    row_count, step_count = traffic.shape
    selected_row_mask = np.zeros(row_count, dtype=bool)
    selected_row_mask[selected_indices] = True

    row_sums = np.empty(row_count, dtype=np.float64)
    missing_counts = np.empty(row_count, dtype=np.int64)
    null_counts = np.empty(row_count, dtype=np.int64)
    for start in range(0, row_count, 256):
        stop = min(start + 256, row_count)
        row_sums[start:stop] = np.sum(traffic[start:stop], axis=1, dtype=np.float64)
        missing_counts[start:stop] = np.count_nonzero(missing_mask[start:stop], axis=1)
        null_counts[start:stop] = np.count_nonzero(null_mask[start:stop], axis=1)

    outside_mask = ~selected_row_mask
    selected_total = float(row_sums[selected_row_mask].sum(dtype=np.float64))
    outside_total = float(row_sums[outside_mask].sum(dtype=np.float64))
    all_total = selected_total + outside_total
    selected_observations = int(selected_row_mask.sum()) * step_count
    outside_observations = int(outside_mask.sum()) * step_count

    return {
        "selected": {
            "cell_count": int(selected_row_mask.sum()),
            "traffic_sum": selected_total,
            "traffic_mean_per_observation": selected_total / selected_observations,
            "cell_mean_traffic_median": float(
                np.median(row_sums[selected_row_mask] / step_count)
            ),
            "missing_pair_count": int(missing_counts[selected_row_mask].sum()),
            "missing_pair_ratio": float(
                missing_counts[selected_row_mask].sum() / selected_observations
            ),
            "internet_all_null_pair_count": int(null_counts[selected_row_mask].sum()),
            "internet_all_null_pair_ratio": float(
                null_counts[selected_row_mask].sum() / selected_observations
            ),
        },
        "outside": {
            "cell_count": int(outside_mask.sum()),
            "traffic_sum": outside_total,
            "traffic_mean_per_observation": outside_total / outside_observations,
            "cell_mean_traffic_median": float(
                np.median(row_sums[outside_mask] / step_count)
            ),
            "missing_pair_count": int(missing_counts[outside_mask].sum()),
            "missing_pair_ratio": float(
                missing_counts[outside_mask].sum() / outside_observations
            ),
            "internet_all_null_pair_count": int(null_counts[outside_mask].sum()),
            "internet_all_null_pair_ratio": float(
                null_counts[outside_mask].sum() / outside_observations
            ),
        },
        "selected_traffic_share": selected_total / all_total,
        "selected_to_outside_mean_ratio": (
            (selected_total / selected_observations)
            / (outside_total / outside_observations)
        ),
    }


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.partial")


def _write_csv(path: Path, selected: Sequence[IndexedCell]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["cell_id", "grid_row", "grid_column", "centroid_lon", "centroid_lat"]
        )
        for item in selected:
            writer.writerow(
                [
                    item.cell_id,
                    item.grid_row,
                    item.grid_column,
                    f"{item.cell.centroid_lon:.12f}",
                    f"{item.cell.centroid_lat:.12f}",
                ]
            )


def _write_npy(path: Path, value: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)


def render_map(
    path: Path,
    indexed_cells: Sequence[IndexedCell],
    selected: Sequence[IndexedCell],
    spec: VisualizationSpec,
) -> str:
    """외부 basemap 없이 전체 Grid와 선택 영역을 PNG로 그린다."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise CentralSelectionError(
            "지도 생성에는 requirements/spatial.txt의 matplotlib이 필요합니다."
        ) from exc

    selected_ids = {item.cell_id for item in selected}
    all_polygons = [item.cell.ring[:-1] for item in indexed_cells]
    selected_polygons = [
        item.cell.ring[:-1] for item in indexed_cells if item.cell_id in selected_ids
    ]

    figure, axis = plt.subplots(
        figsize=(spec.figure_width_inches, spec.figure_height_inches)
    )
    axis.add_collection(
        PolyCollection(
            all_polygons,
            facecolors="#edf0f4",
            edgecolors="#a9b2bd",
            linewidths=0.08,
            rasterized=True,
        )
    )
    axis.add_collection(
        PolyCollection(
            selected_polygons,
            facecolors="#6f42c1",
            edgecolors="#4b258f",
            linewidths=0.12,
            rasterized=True,
        )
    )
    all_lon = [point[0] for polygon in all_polygons for point in polygon]
    all_lat = [point[1] for polygon in all_polygons for point in polygon]
    axis.set_xlim(min(all_lon), max(all_lon))
    axis.set_ylim(min(all_lat), max(all_lat))
    mean_latitude = sum(all_lat) / len(all_lat)
    axis.set_aspect(1.0 / math.cos(math.radians(mean_latitude)))
    axis.set_xlabel("Longitude (WGS84)")
    axis.set_ylabel("Latitude (WGS84)")
    axis.set_title("Milano Grid: central 30 × 30 cells (approximate)")
    axis.legend(
        handles=[
            Patch(facecolor="#6f42c1", edgecolor="#4b258f", label="Selected 900"),
            Patch(facecolor="#edf0f4", edgecolor="#a9b2bd", label="Other 9,100"),
        ],
        loc="upper right",
        frameon=True,
    )
    axis.text(
        0.0,
        -0.12,
        "Source: Telecom Italia Milano Grid (Harvard Dataverse, ODbL 1.0)",
        transform=axis.transAxes,
        fontsize=7,
        color="#4c566a",
    )
    figure.tight_layout()
    figure.savefig(
        path,
        format="png",
        dpi=spec.dpi,
        metadata={"Software": f"GECOS central selection {TOOL_VERSION}"},
    )
    plt.close(figure)
    return matplotlib.__version__


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_selection(config: SelectionConfig, *, skip_map: bool = False) -> dict[str, Any]:
    """전체 검증, 선택, 부분집합 생성과 manifest 기록을 수행한다."""

    started = time.perf_counter()
    reference = load_grid_reference(config.grid_reference_manifest)
    grid_integrity = verify_grid_source(config.inputs.grid_geojson, reference)
    cells, geojson_validation = load_grid_cells(config.inputs.grid_geojson, config.grid)
    indexed_cells, reconstruction = reconstruct_grid(cells, config.grid)
    selected = select_cells(indexed_cells, config.selection)
    arrays, processed_validation = verify_processed_inputs(config)
    selected_indices = map_selected_indices(arrays["cell_ids"], selected)

    selected_cell_ids = np.asarray([item.cell_id for item in selected], dtype="<i4")
    if not np.array_equal(
        np.asarray(arrays["cell_ids"])[selected_indices].astype("<i4"),
        selected_cell_ids,
    ):
        raise CentralSelectionError("선택 cell ID와 전처리 행렬의 행 매핑이 다릅니다.")

    central_traffic = np.asarray(arrays["traffic"][selected_indices], dtype=np.float32)
    central_missing = np.asarray(arrays["missing_mask"][selected_indices], dtype=bool)
    central_null = np.asarray(
        arrays["internet_null_mask"][selected_indices], dtype=bool
    )
    expected_output_shape = (
        config.selection.expected_cell_count,
        config.expected_steps,
    )
    for label, value in (
        ("central_traffic", central_traffic),
        ("central_missing_mask", central_missing),
        ("central_internet_null_mask", central_null),
    ):
        if value.shape != expected_output_shape:
            raise CentralSelectionError(f"{label} shape가 다릅니다: {value.shape}")
    if not np.all(np.isfinite(central_traffic)) or np.any(central_traffic < 0):
        raise CentralSelectionError("중앙 traffic에 유효하지 않은 값이 있습니다.")
    if np.any(central_missing & central_null):
        raise CentralSelectionError("중앙 결측 mask가 겹칩니다.")

    traffic_statistics = calculate_traffic_statistics(arrays, selected_indices)
    output_paths = config.outputs.as_dict()
    temporary_paths = {key: _temporary_path(path) for key, path in output_paths.items()}
    for output in output_paths.values():
        output.parent.mkdir(parents=True, exist_ok=True)

    output_metadata: dict[str, Any] = {}
    matplotlib_version: str | None = None
    try:
        _write_csv(temporary_paths["central_cells_csv"], selected)
        _write_npy(temporary_paths["central_traffic"], central_traffic)
        _write_npy(temporary_paths["central_missing_mask"], central_missing)
        _write_npy(temporary_paths["central_internet_null_mask"], central_null)
        if not skip_map:
            matplotlib_version = render_map(
                temporary_paths["map_png"],
                indexed_cells,
                selected,
                config.visualization,
            )

        generated_keys = [
            "central_cells_csv",
            "central_traffic",
            "central_missing_mask",
            "central_internet_null_mask",
        ]
        if not skip_map:
            generated_keys.append("map_png")
        for key in generated_keys:
            temporary = temporary_paths[key]
            output_metadata[key] = {
                "path": _display_path(output_paths[key]),
                "size_bytes": temporary.stat().st_size,
                "sha256": compute_sha256(temporary),
            }

        selected_id_digest = hashlib.sha256(selected_cell_ids.tobytes()).hexdigest()
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "tool": {
                "name": "scripts.select_central_900",
                "version": TOOL_VERSION,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": config.protocol,
            "warning_ko": (
                "논문이 정확한 중앙 900셀 ID를 공개하지 않았으므로 이 결과는 "
                "공식 Grid 좌표로 만든 재현 프로젝트의 근사 프로토콜이다."
            ),
            "git_commit": _git_commit(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "grid_source": {
                "reference_manifest": _display_path(reference.path),
                "reference_manifest_sha256": compute_sha256(reference.path),
                "source": dict(reference.source),
                "acquisition": dict(reference.acquisition),
                "integrity": grid_integrity,
            },
            "geojson_validation": geojson_validation,
            "grid_reconstruction": reconstruction,
            "selection": {
                "row_start": config.selection.row_start,
                "row_end_exclusive": config.selection.row_end_exclusive,
                "column_start": config.selection.column_start,
                "column_end_exclusive": config.selection.column_end_exclusive,
                "cell_count": len(selected),
                "first_cell_id": int(selected_cell_ids[0]),
                "last_cell_id": int(selected_cell_ids[-1]),
                "cell_ids_int32_sha256": selected_id_digest,
            },
            "processed_input_validation": processed_validation,
            "central_output_validation": {
                "shape": list(central_traffic.shape),
                "traffic_dtype": str(central_traffic.dtype),
                "traffic_min": float(central_traffic.min()),
                "traffic_max": float(central_traffic.max()),
                "missing_pair_count": int(np.count_nonzero(central_missing)),
                "internet_all_null_pair_count": int(np.count_nonzero(central_null)),
            },
            "traffic_statistics": traffic_statistics,
            "outputs": output_metadata,
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "numpy_version": np.__version__,
                "matplotlib_version": matplotlib_version,
                "peak_rss_bytes": _peak_rss_bytes(),
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
        temporary_paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        publish_keys = generated_keys + ["manifest"]
        for key in publish_keys:
            os.replace(temporary_paths[key], output_paths[key])
        return manifest
    finally:
        for temporary in temporary_paths.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milano Grid 중앙 30×30 셀을 선택하고 학습용 부분집합을 만듭니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"선택 config 경로 (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--grid",
        type=Path,
        help="config의 inputs.grid_geojson을 대신할 로컬 GeoJSON 경로",
    )
    parser.add_argument(
        "--skip-map",
        action="store_true",
        help="matplotlib 지도 PNG만 생략합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_selection_config(args.config)
        if args.grid is not None:
            config = SelectionConfig(
                **{
                    **config.__dict__,
                    "inputs": InputPaths(
                        **{
                            **config.inputs.__dict__,
                            "grid_geojson": args.grid.resolve(),
                        }
                    ),
                }
            )
        manifest = run_selection(config, skip_map=args.skip_map)
    except CentralSelectionError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"파일 시스템 오류: {exc}", file=sys.stderr)
        return 2

    selection = manifest["selection"]
    statistics = manifest["traffic_statistics"]
    print(
        "중앙 900셀 선택 완료: "
        f"protocol={manifest['protocol']}, "
        f"cells={selection['cell_count']}, "
        f"shape={manifest['central_output_validation']['shape']}"
    )
    print(
        "공간 검증: "
        f"official_md5={manifest['grid_source']['integrity']['official_checksum_matched']}, "
        f"grid={manifest['grid_reconstruction']['rows']}x"
        f"{manifest['grid_reconstruction']['columns']}"
    )
    print(
        "트래픽 비교: "
        f"central_share={statistics['selected_traffic_share']:.6f}, "
        f"mean_ratio={statistics['selected_to_outside_mean_ratio']:.6f}"
    )
    print(f"manifest={config.outputs.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
