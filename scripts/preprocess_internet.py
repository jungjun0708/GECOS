#!/usr/bin/env python3
"""Telecom Italia 원본을 GECOS 학습용 Internet traffic 행렬로 변환한다."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from scripts.verify_raw_data import (
    DEFAULT_DATA_DIRECTORY,
    ManifestError,
    ReferenceManifest,
    compute_digest,
    load_reference_manifest,
    verify_data_directory,
    write_report,
)

TOOL_VERSION = "1.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "preprocess_milan_nov2013.json"
COLUMN_NAMES = (
    "cell_id",
    "timestamp_ms",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
)
ACTIVITY_COLUMNS = ("sms_in", "sms_out", "call_in", "call_out", "internet")
OUTPUT_KEYS = (
    "interim_parquet",
    "traffic",
    "cell_ids",
    "timestamps_ms",
    "missing_mask",
    "internet_null_mask",
    "manifest",
)


class PreprocessingError(RuntimeError):
    """전처리 계약을 만족하지 못할 때 발생한다."""


@dataclass(frozen=True)
class OutputPaths:
    interim_parquet: Path
    traffic: Path
    cell_ids: Path
    timestamps_ms: Path
    missing_mask: Path
    internet_null_mask: Path
    manifest: Path

    def as_dict(self) -> dict[str, Path]:
        return {key: getattr(self, key) for key in OUTPUT_KEYS}


@dataclass(frozen=True)
class PreprocessConfig:
    path: Path
    name: str
    raw_reference_manifest: Path
    expected_total_rows: int
    cell_id_min: int
    cell_id_max: int
    expected_cell_count: int
    timezone_name: str
    start_local: datetime
    end_exclusive_local: datetime
    interval_ms: int
    steps_per_file: int
    expected_steps: int
    block_size_bytes: int
    compression: str
    compression_level: int
    row_group_rows: int
    outputs: OutputPaths


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PreprocessingError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreprocessingError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PreprocessingError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _resolve_path(value: object, field: str, base_directory: Path) -> Path:
    raw_path = Path(_require_string(value, field))
    return (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (base_directory / raw_path).resolve()
    )


def _parse_local_datetime(value: object, field: str) -> datetime:
    text = _require_string(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PreprocessingError(f"{field}가 ISO-8601 형식이 아닙니다: {text}") from exc
    if parsed.tzinfo is not None:
        raise PreprocessingError(
            f"{field}에는 timezone offset을 직접 넣지 마세요: {text}"
        )
    return parsed


def load_preprocess_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> PreprocessConfig:
    """전처리 설정 JSON을 읽고 상호 모순을 검사한다."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreprocessingError(f"전처리 config를 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreprocessingError(
            f"전처리 config가 올바른 JSON이 아닙니다: {exc}"
        ) from exc

    root = _require_mapping(payload, "root")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise PreprocessingError(
            f"지원하지 않는 schema_version입니다: {schema_version}"
        )

    name = _require_string(root.get("name"), "name")
    raw_reference_manifest = _resolve_path(
        root.get("raw_reference_manifest"),
        "raw_reference_manifest",
        base_directory,
    )
    expected_total_rows = _require_int(
        root.get("expected_total_rows"), "expected_total_rows", minimum=1
    )

    grid = _require_mapping(root.get("grid"), "grid")
    cell_id_min = _require_int(grid.get("cell_id_min"), "grid.cell_id_min")
    cell_id_max = _require_int(grid.get("cell_id_max"), "grid.cell_id_max")
    expected_cell_count = _require_int(
        grid.get("expected_cell_count"), "grid.expected_cell_count", minimum=1
    )
    if cell_id_max < cell_id_min:
        raise PreprocessingError("grid.cell_id_max는 cell_id_min 이상이어야 합니다.")
    if cell_id_max - cell_id_min + 1 != expected_cell_count:
        raise PreprocessingError(
            "cell ID 범위와 expected_cell_count가 일치하지 않습니다."
        )

    time_config = _require_mapping(root.get("time"), "time")
    timezone_name = _require_string(time_config.get("timezone"), "time.timezone")
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise PreprocessingError(f"알 수 없는 timezone입니다: {timezone_name}") from exc
    start_local = _parse_local_datetime(
        time_config.get("start_local"), "time.start_local"
    )
    end_exclusive_local = _parse_local_datetime(
        time_config.get("end_exclusive_local"), "time.end_exclusive_local"
    )
    interval_ms = _require_int(
        time_config.get("interval_ms"), "time.interval_ms", minimum=1
    )
    steps_per_file = _require_int(
        time_config.get("steps_per_file"), "time.steps_per_file", minimum=1
    )
    expected_steps = _require_int(
        time_config.get("expected_steps"), "time.expected_steps", minimum=1
    )

    start_ms = int(start_local.replace(tzinfo=zone).timestamp() * 1000)
    end_ms = int(end_exclusive_local.replace(tzinfo=zone).timestamp() * 1000)
    duration_ms = end_ms - start_ms
    if duration_ms <= 0 or duration_ms % interval_ms:
        raise PreprocessingError(
            "time 범위가 interval_ms로 정확히 나누어지지 않습니다."
        )
    if duration_ms // interval_ms != expected_steps:
        raise PreprocessingError("time 범위와 expected_steps가 일치하지 않습니다.")

    parser = _require_mapping(root.get("parser"), "parser")
    block_size_bytes = _require_int(
        parser.get("block_size_bytes"), "parser.block_size_bytes", minimum=1024
    )
    column_names = parser.get("column_names")
    if not isinstance(column_names, list) or tuple(column_names) != COLUMN_NAMES:
        raise PreprocessingError(
            "parser.column_names는 GECOS 원본의 고정된 8개 열과 정확히 일치해야 합니다."
        )

    aggregation = _require_mapping(root.get("aggregation"), "aggregation")
    if aggregation.get("target") != "internet":
        raise PreprocessingError("aggregation.target은 internet이어야 합니다.")
    if aggregation.get("null_value") != 0.0:
        raise PreprocessingError("aggregation.null_value는 0.0이어야 합니다.")
    if aggregation.get("accumulator_dtype") != "float64":
        raise PreprocessingError("aggregation.accumulator_dtype은 float64여야 합니다.")
    if aggregation.get("output_dtype") != "float32":
        raise PreprocessingError("aggregation.output_dtype은 float32여야 합니다.")

    parquet = _require_mapping(root.get("parquet"), "parquet")
    compression = _require_string(parquet.get("compression"), "parquet.compression")
    if compression != "zstd":
        raise PreprocessingError("현재 전처리 계약은 zstd Parquet만 지원합니다.")
    compression_level = _require_int(
        parquet.get("compression_level"), "parquet.compression_level"
    )
    row_group_rows = _require_int(
        parquet.get("row_group_rows"), "parquet.row_group_rows", minimum=1
    )

    raw_outputs = _require_mapping(root.get("outputs"), "outputs")
    missing_output_keys = [key for key in OUTPUT_KEYS if key not in raw_outputs]
    if missing_output_keys:
        raise PreprocessingError(
            f"outputs에 필요한 경로가 없습니다: {missing_output_keys}"
        )
    output_values = {
        key: _resolve_path(raw_outputs[key], f"outputs.{key}", base_directory)
        for key in OUTPUT_KEYS
    }
    if len(set(output_values.values())) != len(output_values):
        raise PreprocessingError("outputs 경로는 서로 달라야 합니다.")

    return PreprocessConfig(
        path=path.resolve(),
        name=name,
        raw_reference_manifest=raw_reference_manifest,
        expected_total_rows=expected_total_rows,
        cell_id_min=cell_id_min,
        cell_id_max=cell_id_max,
        expected_cell_count=expected_cell_count,
        timezone_name=timezone_name,
        start_local=start_local,
        end_exclusive_local=end_exclusive_local,
        interval_ms=interval_ms,
        steps_per_file=steps_per_file,
        expected_steps=expected_steps,
        block_size_bytes=block_size_bytes,
        compression=compression,
        compression_level=compression_level,
        row_group_rows=row_group_rows,
        outputs=OutputPaths(**output_values),
    )


def build_expected_timestamps(config: PreprocessConfig) -> np.ndarray:
    zone = ZoneInfo(config.timezone_name)
    start_ms = int(config.start_local.replace(tzinfo=zone).timestamp() * 1000)
    timestamps = (
        start_ms + np.arange(config.expected_steps, dtype=np.int64) * config.interval_ms
    )
    return timestamps


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _temporary_paths(outputs: OutputPaths) -> dict[str, Path]:
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temporary: dict[str, Path] = {}
    for key, final_path in outputs.as_dict().items():
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary[key] = final_path.with_name(f".{final_path.name}.{token}.partial")
    return temporary


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    try:
        array.flush()
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _arrow_column_types() -> dict[str, pa.DataType]:
    return {
        "cell_id": pa.int32(),
        "timestamp_ms": pa.int64(),
        "country_code": pa.int32(),
        "sms_in": pa.float64(),
        "sms_out": pa.float64(),
        "call_in": pa.float64(),
        "call_out": pa.float64(),
        "internet": pa.float64(),
    }


def _open_csv(path: Path, config: PreprocessConfig) -> pacsv.CSVStreamingReader:
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(
            column_names=list(COLUMN_NAMES),
            block_size=config.block_size_bytes,
            use_threads=True,
        ),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types=_arrow_column_types(),
            null_values=[""],
        ),
    )


def _parquet_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("cell_id", pa.int32(), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
            pa.field("internet", pa.float32(), nullable=False),
            pa.field("missing_pair", pa.bool_(), nullable=False),
            pa.field("internet_all_null", pa.bool_(), nullable=False),
        ]
    )


def _validate_output_arrays(
    temporary: Mapping[str, Path], config: PreprocessConfig
) -> dict[str, Any]:
    expected_shape = (config.expected_cell_count, config.expected_steps)
    specifications = {
        "traffic": (expected_shape, np.dtype("float32")),
        "missing_mask": (expected_shape, np.dtype("bool")),
        "internet_null_mask": (expected_shape, np.dtype("bool")),
        "cell_ids": ((config.expected_cell_count,), np.dtype("int32")),
        "timestamps_ms": ((config.expected_steps,), np.dtype("int64")),
    }
    loaded: dict[str, np.ndarray] = {}
    for key, (shape, dtype) in specifications.items():
        array = np.load(temporary[key], mmap_mode="r", allow_pickle=False)
        if array.shape != shape:
            raise PreprocessingError(f"{key} shape 불일치: {array.shape} != {shape}")
        if array.dtype != dtype:
            raise PreprocessingError(f"{key} dtype 불일치: {array.dtype} != {dtype}")
        loaded[key] = array

    traffic = loaded["traffic"]
    if not np.isfinite(traffic).all():
        raise PreprocessingError("traffic.npy에 NaN 또는 무한대가 있습니다.")
    traffic_min = float(traffic.min())
    traffic_max = float(traffic.max())
    if traffic_min < 0:
        raise PreprocessingError("traffic.npy에 음수가 있습니다.")

    expected_cell_ids = np.arange(
        config.cell_id_min, config.cell_id_max + 1, dtype=np.int32
    )
    if not np.array_equal(loaded["cell_ids"], expected_cell_ids):
        raise PreprocessingError("cell_ids.npy가 설정된 연속 범위와 다릅니다.")

    expected_timestamps = build_expected_timestamps(config)
    if not np.array_equal(loaded["timestamps_ms"], expected_timestamps):
        raise PreprocessingError("timestamps_ms.npy가 설정된 시간축과 다릅니다.")
    if not np.all(np.diff(loaded["timestamps_ms"]) == config.interval_ms):
        raise PreprocessingError("timestamps_ms.npy의 간격이 일정하지 않습니다.")

    missing_mask = loaded["missing_mask"]
    internet_null_mask = loaded["internet_null_mask"]
    if np.any(missing_mask & internet_null_mask):
        raise PreprocessingError("두 결측 mask가 동시에 True인 위치가 있습니다.")

    validation = {
        "shape": list(expected_shape),
        "traffic_dtype": str(traffic.dtype),
        "traffic_min": traffic_min,
        "traffic_max": traffic_max,
        "missing_pair_count": int(missing_mask.sum()),
        "internet_all_null_pair_count": int(internet_null_mask.sum()),
        "timestamp_interval_ms": config.interval_ms,
    }
    loaded.clear()
    gc.collect()
    return validation


def _output_metadata(
    temporary: Mapping[str, Path], outputs: OutputPaths
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for key, final_path in outputs.as_dict().items():
        if key == "manifest":
            continue
        partial_path = temporary[key]
        records[key] = {
            "path": _portable_path(final_path),
            "size_bytes": partial_path.stat().st_size,
            "sha256": compute_digest(partial_path, "sha256"),
        }
    return records


def preprocess_dataset(
    config: PreprocessConfig,
    reference: ReferenceManifest,
    data_directory: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """원본 30개를 검증하고 학습용 행렬과 Parquet을 생성한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    if reference.expected_file_count * config.steps_per_file != config.expected_steps:
        raise PreprocessingError(
            "기준 파일 수, steps_per_file과 expected_steps가 일치하지 않습니다."
        )

    if progress:
        progress("원본 파일의 크기와 MD5를 다시 검증합니다.")
    raw_report = verify_data_directory(
        reference,
        data_directory,
        quick=False,
        progress=progress,
    )
    if not raw_report["verification"]["integrity_verified"]:
        raise PreprocessingError("원본 데이터 무결성 검증에 실패했습니다.")

    timestamps = build_expected_timestamps(config)
    cell_ids = np.arange(config.cell_id_min, config.cell_id_max + 1, dtype=np.int32)
    expected_shape = (config.expected_cell_count, config.expected_steps)
    day_size = config.expected_cell_count * config.steps_per_file
    repeated_cell_ids = np.repeat(cell_ids, config.steps_per_file)

    temporary = _temporary_paths(config.outputs)
    traffic_map: np.memmap | None = None
    missing_map: np.memmap | None = None
    internet_null_map: np.memmap | None = None
    parquet_writer: pq.ParquetWriter | None = None
    outputs_committed = False

    total_rows = 0
    null_counts = {name: 0 for name in ACTIVITY_COLUMNS}
    global_cells_seen = np.zeros(config.expected_cell_count, dtype=bool)
    file_statistics: list[dict[str, Any]] = []

    try:
        traffic_map = np.lib.format.open_memmap(
            temporary["traffic"], mode="w+", dtype=np.float32, shape=expected_shape
        )
        missing_map = np.lib.format.open_memmap(
            temporary["missing_mask"], mode="w+", dtype=bool, shape=expected_shape
        )
        internet_null_map = np.lib.format.open_memmap(
            temporary["internet_null_mask"],
            mode="w+",
            dtype=bool,
            shape=expected_shape,
        )
        parquet_writer = pq.ParquetWriter(
            temporary["interim_parquet"],
            _parquet_schema(),
            compression=config.compression,
            compression_level=config.compression_level,
            use_dictionary=["cell_id"],
            write_statistics=True,
            version="2.6",
        )

        for file_index, expected_file in enumerate(reference.files):
            input_path = data_directory / expected_file.name
            slice_start = file_index * config.steps_per_file
            slice_stop = slice_start + config.steps_per_file
            expected_day_timestamps = timestamps[slice_start:slice_stop]

            daily_sum = np.zeros(
                (config.expected_cell_count, config.steps_per_file), dtype=np.float64
            )
            daily_pair_present = np.zeros(daily_sum.shape, dtype=bool)
            daily_internet_present = np.zeros(daily_sum.shape, dtype=bool)
            day_timestamps_seen = np.zeros(config.steps_per_file, dtype=bool)
            day_cells_seen = np.zeros(config.expected_cell_count, dtype=bool)
            day_null_counts = {name: 0 for name in ACTIVITY_COLUMNS}
            day_rows = 0

            if progress:
                progress(
                    f"[{file_index + 1:02d}/{reference.expected_file_count}] "
                    f"{expected_file.name}: TSV 파싱 중"
                )

            try:
                batches = _open_csv(input_path, config)
                for batch in batches:
                    batch_rows = batch.num_rows
                    if batch_rows == 0:
                        continue
                    day_rows += batch_rows
                    total_rows += batch_rows

                    cell_column = batch.column(batch.schema.get_field_index("cell_id"))
                    timestamp_column = batch.column(
                        batch.schema.get_field_index("timestamp_ms")
                    )
                    country_column = batch.column(
                        batch.schema.get_field_index("country_code")
                    )
                    if (
                        cell_column.null_count
                        or timestamp_column.null_count
                        or country_column.null_count
                    ):
                        raise PreprocessingError(
                            f"{expected_file.name}의 식별자 열에 공란이 있습니다."
                        )

                    raw_cell_ids = cell_column.to_numpy(zero_copy_only=False)
                    raw_timestamps = timestamp_column.to_numpy(zero_copy_only=False)
                    country_codes = country_column.to_numpy(zero_copy_only=False)

                    invalid_cell = (raw_cell_ids < config.cell_id_min) | (
                        raw_cell_ids > config.cell_id_max
                    )
                    if invalid_cell.any():
                        bad = int(raw_cell_ids[np.flatnonzero(invalid_cell)[0]])
                        raise PreprocessingError(
                            f"{expected_file.name}에 범위를 벗어난 cell_id가 있습니다: {bad}"
                        )
                    if (country_codes < 0).any():
                        bad = int(country_codes[np.flatnonzero(country_codes < 0)[0]])
                        raise PreprocessingError(
                            f"{expected_file.name}에 음수 country_code가 있습니다: {bad}"
                        )

                    delta = raw_timestamps - timestamps[0]
                    misaligned = delta % config.interval_ms != 0
                    if misaligned.any():
                        bad = int(raw_timestamps[np.flatnonzero(misaligned)[0]])
                        raise PreprocessingError(
                            f"{expected_file.name}에 10분 축과 맞지 않는 timestamp가 있습니다: {bad}"
                        )
                    global_time_index = delta // config.interval_ms
                    invalid_time = (global_time_index < slice_start) | (
                        global_time_index >= slice_stop
                    )
                    if invalid_time.any():
                        bad = int(raw_timestamps[np.flatnonzero(invalid_time)[0]])
                        raise PreprocessingError(
                            f"{expected_file.name}의 날짜 범위를 벗어난 timestamp가 있습니다: {bad}"
                        )

                    local_time_index = (global_time_index - slice_start).astype(
                        np.int64, copy=False
                    )
                    cell_index = (raw_cell_ids - config.cell_id_min).astype(
                        np.int64, copy=False
                    )
                    flat_index = cell_index * config.steps_per_file + local_time_index
                    day_timestamps_seen[local_time_index] = True
                    day_cells_seen[cell_index] = True
                    global_cells_seen[cell_index] = True

                    internet_values: np.ndarray | None = None
                    internet_is_null: np.ndarray | None = None
                    for activity_name in ACTIVITY_COLUMNS:
                        column = batch.column(
                            batch.schema.get_field_index(activity_name)
                        )
                        is_null = column.is_null().to_numpy(zero_copy_only=False)
                        values = column.to_numpy(zero_copy_only=False)
                        null_count = int(column.null_count)
                        day_null_counts[activity_name] += null_count
                        null_counts[activity_name] += null_count

                        non_null = ~is_null
                        if non_null.any() and not np.isfinite(values[non_null]).all():
                            raise PreprocessingError(
                                f"{expected_file.name}의 {activity_name}에 NaN 또는 무한대가 있습니다."
                            )
                        if non_null.any() and (values[non_null] < 0).any():
                            raise PreprocessingError(
                                f"{expected_file.name}의 {activity_name}에 음수가 있습니다."
                            )
                        if activity_name == "internet":
                            internet_values = values.copy()
                            internet_is_null = is_null

                    if internet_values is None or internet_is_null is None:
                        raise PreprocessingError("internet 열을 읽지 못했습니다.")
                    internet_values[internet_is_null] = 0.0
                    daily_sum.ravel()[:] += np.bincount(
                        flat_index,
                        weights=internet_values,
                        minlength=day_size,
                    )
                    daily_pair_present.ravel()[flat_index] = True
                    daily_internet_present.ravel()[flat_index[~internet_is_null]] = True
            except (
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                OSError,
                ValueError,
            ) as exc:
                raise PreprocessingError(
                    f"{expected_file.name} TSV 파싱에 실패했습니다: {exc}"
                ) from exc

            if day_rows == 0:
                raise PreprocessingError(
                    f"{expected_file.name}에 데이터 행이 없습니다."
                )
            if not day_timestamps_seen.all():
                missing_steps = np.flatnonzero(~day_timestamps_seen).tolist()
                raise PreprocessingError(
                    f"{expected_file.name}에 누락된 10분 timestamp가 있습니다: {missing_steps}"
                )

            missing_pair = ~daily_pair_present
            internet_all_null = daily_pair_present & ~daily_internet_present
            traffic_day = daily_sum.astype(np.float32)
            traffic_map[:, slice_start:slice_stop] = traffic_day
            missing_map[:, slice_start:slice_stop] = missing_pair
            internet_null_map[:, slice_start:slice_stop] = internet_all_null

            day_table = pa.Table.from_arrays(
                [
                    pa.array(repeated_cell_ids, type=pa.int32()),
                    pa.array(
                        np.tile(expected_day_timestamps, config.expected_cell_count),
                        type=pa.int64(),
                    ),
                    pa.array(traffic_day.ravel(), type=pa.float32()),
                    pa.array(missing_pair.ravel(), type=pa.bool_()),
                    pa.array(internet_all_null.ravel(), type=pa.bool_()),
                ],
                schema=_parquet_schema(),
            )
            parquet_writer.write_table(day_table, row_group_size=config.row_group_rows)

            file_statistics.append(
                {
                    "name": expected_file.name,
                    "rows": day_rows,
                    "unique_cells": int(day_cells_seen.sum()),
                    "unique_timestamps": int(day_timestamps_seen.sum()),
                    "activity_null_rows": day_null_counts,
                    "observed_pair_count": int(daily_pair_present.sum()),
                    "missing_pair_count": int(missing_pair.sum()),
                    "internet_all_null_pair_count": int(internet_all_null.sum()),
                    "traffic_sum_float64": float(daily_sum.sum()),
                }
            )
            if progress:
                progress(
                    f"[{file_index + 1:02d}/{reference.expected_file_count}] "
                    f"{expected_file.name}: rows={day_rows}, "
                    f"missing_pairs={int(missing_pair.sum())}"
                )

        if total_rows != config.expected_total_rows:
            raise PreprocessingError(
                f"전체 행 수 불일치: {total_rows} != {config.expected_total_rows}"
            )
        if int(global_cells_seen.sum()) != config.expected_cell_count:
            raise PreprocessingError(
                "전체 기간에 등장한 cell 수가 expected_cell_count와 다릅니다."
            )

        parquet_writer.close()
        parquet_writer = None
        _close_memmap(traffic_map)
        _close_memmap(missing_map)
        _close_memmap(internet_null_map)
        traffic_map = None
        missing_map = None
        internet_null_map = None

        with temporary["cell_ids"].open("wb") as handle:
            np.save(handle, cell_ids, allow_pickle=False)
        with temporary["timestamps_ms"].open("wb") as handle:
            np.save(handle, timestamps, allow_pickle=False)

        validation = _validate_output_arrays(temporary, config)
        parquet_metadata = pq.ParquetFile(temporary["interim_parquet"]).metadata
        expected_parquet_rows = config.expected_cell_count * config.expected_steps
        if parquet_metadata.num_rows != expected_parquet_rows:
            raise PreprocessingError(
                f"Parquet 행 수 불일치: {parquet_metadata.num_rows} != {expected_parquet_rows}"
            )
        if parquet_metadata.num_columns != len(_parquet_schema()):
            raise PreprocessingError("Parquet 열 수가 계약과 다릅니다.")

        output_metadata = _output_metadata(temporary, config.outputs)
        finished_at = datetime.now(timezone.utc)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "dataset": config.name,
            "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at_utc": finished_at.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": round(time.perf_counter() - started_counter, 3),
            "git": _git_state(),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "platform": platform.platform(),
            },
            "inputs": {
                "raw_reference_manifest": {
                    "path": _portable_path(config.raw_reference_manifest),
                    "sha256": compute_digest(config.raw_reference_manifest, "sha256"),
                },
                "preprocess_config": {
                    "path": _portable_path(config.path),
                    "sha256": compute_digest(config.path, "sha256"),
                },
                "raw_integrity": {
                    "status": raw_report["verification"]["status"],
                    "integrity_verified": raw_report["verification"][
                        "integrity_verified"
                    ],
                    "summary": raw_report["summary"],
                },
            },
            "contract": {
                "shape": list(expected_shape),
                "cell_id_range": [config.cell_id_min, config.cell_id_max],
                "timezone": config.timezone_name,
                "interval_ms": config.interval_ms,
                "start_timestamp_ms": int(timestamps[0]),
                "end_timestamp_ms": int(timestamps[-1]),
                "row_order": (
                    "source day ascending; within each day, cell_id ascending, "
                    "then timestamp_ms ascending"
                ),
                "aggregation": "sum(fill_null(internet, 0)) by cell_id and timestamp_ms",
                "missing_mask": "True when no raw row exists for the cell-time pair",
                "internet_null_mask": (
                    "True when raw rows exist but every internet value is null"
                ),
            },
            "statistics": {
                "raw_rows": total_rows,
                "unique_cells": int(global_cells_seen.sum()),
                "unique_timestamps": config.expected_steps,
                "activity_null_rows": null_counts,
                "missing_pair_count": validation["missing_pair_count"],
                "internet_all_null_pair_count": validation[
                    "internet_all_null_pair_count"
                ],
                "traffic_min": validation["traffic_min"],
                "traffic_max": validation["traffic_max"],
                "files": file_statistics,
            },
            "validation": validation,
            "parquet": {
                "rows": parquet_metadata.num_rows,
                "columns": parquet_metadata.num_columns,
                "row_groups": parquet_metadata.num_row_groups,
                "compression": config.compression,
            },
            "outputs": output_metadata,
        }
        write_report(manifest, temporary["manifest"])

        final_paths = config.outputs.as_dict()
        for key in OUTPUT_KEYS:
            if key == "manifest":
                continue
            os.replace(temporary[key], final_paths[key])
        os.replace(temporary["manifest"], final_paths["manifest"])
        outputs_committed = True
        return manifest
    finally:
        if parquet_writer is not None:
            try:
                parquet_writer.close()
            except (OSError, pa.ArrowException) as cleanup_error:
                if progress:
                    progress(f"partial Parquet 종료 중 오류: {cleanup_error}")
        _close_memmap(traffic_map)
        _close_memmap(missing_map)
        _close_memmap(internet_null_map)
        if not outputs_committed:
            gc.collect()
            for path in temporary.values():
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    if progress:
                        progress(f"partial 파일 삭제 중 오류: {cleanup_error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "검증된 Telecom Italia 2013년 11월 원본을 GECOS 학습용 행렬로 변환합니다."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"전처리 config JSON (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help=f"원본 파일 디렉터리 (기본값: {DEFAULT_DATA_DIRECTORY})",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_preprocess_config(args.config)
        reference = load_reference_manifest(config.raw_reference_manifest)
        manifest = preprocess_dataset(
            config,
            reference,
            args.data_dir,
            progress=lambda message: print(message, flush=True),
        )
    except (ManifestError, PreprocessingError, OSError, ValueError) as exc:
        print(f"전처리에 실패했습니다: {exc}", file=sys.stderr)
        return 1

    print(
        "전처리 완료: "
        f"shape={tuple(manifest['contract']['shape'])}, "
        f"rows={manifest['statistics']['raw_rows']}, "
        f"manifest={_portable_path(config.outputs.manifest)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
