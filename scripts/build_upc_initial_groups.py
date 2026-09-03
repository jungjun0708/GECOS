#!/usr/bin/env python3
"""GECOS 논문의 UPC 1단계인 24개 peak-hour 초기 그룹을 만든다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

TOOL_VERSION = "1.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "upc_milan_nov2013.json"
PROTOCOL_NAMES = ("paper_faithful", "train_only")
HOURS_PER_DAY = 24


class UpcInitialGroupError(RuntimeError):
    """UPC 초기 그룹 계약 또는 입력 무결성이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class InputPaths:
    processed_manifest: Path
    traffic: Path
    cell_ids: Path
    timestamps_ms: Path
    missing_mask: Path
    internet_null_mask: Path
    central_manifest: Path
    central_cells_csv: Path

    def processed_arrays(self) -> dict[str, Path]:
        return {
            "traffic": self.traffic,
            "cell_ids": self.cell_ids,
            "timestamps_ms": self.timestamps_ms,
            "missing_mask": self.missing_mask,
            "internet_null_mask": self.internet_null_mask,
        }


@dataclass(frozen=True)
class OutputPaths:
    paper_faithful_peak_hours: Path
    train_only_peak_hours: Path
    figure4_diagnostic_peak_hours: Path
    all_cell_memberships_csv: Path
    central_900_memberships_csv: Path
    group_counts_json: Path
    manifest: Path

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    start_local: datetime
    end_exclusive_local: datetime
    weekdays_only: bool


@dataclass(frozen=True)
class UpcConfig:
    path: Path
    name: str
    inputs: InputPaths
    outputs: OutputPaths
    expected_cell_count: int
    expected_step_count: int
    expected_central_cell_count: int
    timezone_name: str
    interval_ms: int
    observations_per_hour: int
    protocols: dict[str, ProtocolSpec]
    figure4_diagnostic: ProtocolSpec
    paper_fingerprint: tuple[int, ...]
    cell_chunk_size: int
    require_exact_paper_fingerprint: bool


@dataclass(frozen=True)
class LocalTimeAxis:
    timezone_name: str
    hour_indices: dict[tuple[date, int], np.ndarray]
    dates: tuple[date, ...]
    first_local_timestamp: str
    last_local_timestamp: str


@dataclass(frozen=True)
class ProtocolHours:
    name: str
    dates: tuple[date, ...]
    indices: np.ndarray


@dataclass(frozen=True)
class ProtocolResult:
    peak_hours: np.ndarray
    diagnostics: dict[str, Any]


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise UpcInitialGroupError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise UpcInitialGroupError(f"{field}는 JSON array여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpcInitialGroupError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UpcInitialGroupError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise UpcInitialGroupError(f"{field}는 boolean이어야 합니다.")
    return value


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
        raise UpcInitialGroupError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UpcInitialGroupError(f"{label}이 올바른 JSON이 아닙니다: {exc}") from exc
    return _require_mapping(payload, label)


def _parse_local_datetime(value: object, field: str) -> datetime:
    text = _require_string(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise UpcInitialGroupError(
            f"{field}가 ISO-8601 형식이 아닙니다: {text}"
        ) from exc
    if parsed.tzinfo is not None:
        raise UpcInitialGroupError(
            f"{field}에는 timezone offset을 직접 넣지 마세요: {text}"
        )
    if parsed.time() != datetime.min.time():
        raise UpcInitialGroupError(f"{field}는 local 자정이어야 합니다: {text}")
    return parsed


def load_upc_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> UpcConfig:
    """UPC 초기 그룹 설정을 읽고 상호 모순을 검사한다."""

    root = _load_json(path, "UPC config")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise UpcInitialGroupError(
            f"지원하지 않는 schema_version입니다: {schema_version}"
        )

    name = _require_string(root.get("name"), "name")
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
    if len(set(outputs.as_dict().values())) != len(outputs.as_dict()):
        raise UpcInitialGroupError("outputs 경로는 서로 달라야 합니다.")

    grid = _require_mapping(root.get("grid"), "grid")
    expected_cell_count = _require_int(
        grid.get("expected_cell_count"), "grid.expected_cell_count", minimum=1
    )
    expected_central_cell_count = _require_int(
        grid.get("expected_central_cell_count"),
        "grid.expected_central_cell_count",
        minimum=1,
    )
    if expected_central_cell_count > expected_cell_count:
        raise UpcInitialGroupError("중앙 cell 수는 전체 cell 수보다 클 수 없습니다.")

    time_config = _require_mapping(root.get("time"), "time")
    timezone_name = _require_string(time_config.get("timezone"), "time.timezone")
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise UpcInitialGroupError(
            f"알 수 없는 timezone입니다: {timezone_name}"
        ) from exc
    interval_ms = _require_int(
        time_config.get("interval_ms"), "time.interval_ms", minimum=1
    )
    observations_per_hour = _require_int(
        time_config.get("observations_per_hour"),
        "time.observations_per_hour",
        minimum=1,
    )
    expected_step_count = _require_int(
        time_config.get("expected_step_count"),
        "time.expected_step_count",
        minimum=1,
    )
    if interval_ms * observations_per_hour != 3_600_000:
        raise UpcInitialGroupError(
            "interval_ms와 observations_per_hour의 곱은 정확히 한 시간이어야 합니다."
        )

    raw_protocols = _require_mapping(root.get("protocols"), "protocols")
    if set(raw_protocols) != set(PROTOCOL_NAMES):
        raise UpcInitialGroupError(
            f"protocols에는 {list(PROTOCOL_NAMES)}만 있어야 합니다."
        )
    protocols: dict[str, ProtocolSpec] = {}
    for protocol_name in PROTOCOL_NAMES:
        raw_protocol = _require_mapping(
            raw_protocols[protocol_name], f"protocols.{protocol_name}"
        )
        start_local = _parse_local_datetime(
            raw_protocol.get("start_local"),
            f"protocols.{protocol_name}.start_local",
        )
        end_exclusive_local = _parse_local_datetime(
            raw_protocol.get("end_exclusive_local"),
            f"protocols.{protocol_name}.end_exclusive_local",
        )
        if end_exclusive_local <= start_local:
            raise UpcInitialGroupError(
                f"{protocol_name}의 종료 시각은 시작 시각보다 뒤여야 합니다."
            )
        weekdays_only = _require_bool(
            raw_protocol.get("weekdays_only"),
            f"protocols.{protocol_name}.weekdays_only",
        )
        if not weekdays_only:
            raise UpcInitialGroupError(
                f"{protocol_name}은 논문 계약에 따라 평일만 사용해야 합니다."
            )
        protocols[protocol_name] = ProtocolSpec(
            name=protocol_name,
            start_local=start_local,
            end_exclusive_local=end_exclusive_local,
            weekdays_only=weekdays_only,
        )

    raw_diagnostics = _require_mapping(root.get("diagnostics"), "diagnostics")
    raw_figure4 = _require_mapping(
        raw_diagnostics.get("figure4_complete_weeks_mean_profile"),
        "diagnostics.figure4_complete_weeks_mean_profile",
    )
    diagnostic_method = _require_string(
        raw_figure4.get("method"),
        "diagnostics.figure4_complete_weeks_mean_profile.method",
    )
    if diagnostic_method != "mean_hourly_profile_then_argmax":
        raise UpcInitialGroupError(
            "Fig. 4 진단 method는 mean_hourly_profile_then_argmax여야 합니다."
        )
    diagnostic_start = _parse_local_datetime(
        raw_figure4.get("start_local"),
        "diagnostics.figure4_complete_weeks_mean_profile.start_local",
    )
    diagnostic_end = _parse_local_datetime(
        raw_figure4.get("end_exclusive_local"),
        "diagnostics.figure4_complete_weeks_mean_profile.end_exclusive_local",
    )
    if diagnostic_end <= diagnostic_start:
        raise UpcInitialGroupError(
            "Fig. 4 진단 종료 시각은 시작 시각보다 뒤여야 합니다."
        )
    diagnostic_weekdays_only = _require_bool(
        raw_figure4.get("weekdays_only"),
        "diagnostics.figure4_complete_weeks_mean_profile.weekdays_only",
    )
    if not diagnostic_weekdays_only:
        raise UpcInitialGroupError("Fig. 4 진단은 평일만 사용해야 합니다.")
    figure4_diagnostic = ProtocolSpec(
        name="figure4_complete_weeks_mean_profile",
        start_local=diagnostic_start,
        end_exclusive_local=diagnostic_end,
        weekdays_only=diagnostic_weekdays_only,
    )

    validation = _require_mapping(root.get("validation"), "validation")
    raw_fingerprint = _require_list(
        validation.get("paper_group_counts"), "validation.paper_group_counts"
    )
    if len(raw_fingerprint) != HOURS_PER_DAY:
        raise UpcInitialGroupError(
            "validation.paper_group_counts는 hour 0~23의 24개 값이어야 합니다."
        )
    paper_fingerprint = tuple(
        _require_int(value, f"validation.paper_group_counts[{index}]")
        for index, value in enumerate(raw_fingerprint)
    )
    if sum(paper_fingerprint) != expected_cell_count:
        raise UpcInitialGroupError(
            "논문 fingerprint의 합이 expected_cell_count와 다릅니다."
        )
    require_exact = _require_bool(
        validation.get("require_exact_paper_fingerprint"),
        "validation.require_exact_paper_fingerprint",
    )

    execution = _require_mapping(root.get("execution"), "execution")
    cell_chunk_size = _require_int(
        execution.get("cell_chunk_size"),
        "execution.cell_chunk_size",
        minimum=1,
    )

    return UpcConfig(
        path=path.resolve(),
        name=name,
        inputs=inputs,
        outputs=outputs,
        expected_cell_count=expected_cell_count,
        expected_step_count=expected_step_count,
        expected_central_cell_count=expected_central_cell_count,
        timezone_name=timezone_name,
        interval_ms=interval_ms,
        observations_per_hour=observations_per_hour,
        protocols=protocols,
        figure4_diagnostic=figure4_diagnostic,
        paper_fingerprint=paper_fingerprint,
        cell_chunk_size=cell_chunk_size,
        require_exact_paper_fingerprint=require_exact,
    )


def compute_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise UpcInitialGroupError(f"파일을 읽을 수 없습니다: {path}") from exc
    return digest.hexdigest()


def _verify_file_metadata(
    path: Path,
    metadata: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    expected_size = _require_int(metadata.get("size_bytes"), f"{label}.size_bytes")
    expected_sha = _require_string(metadata.get("sha256"), f"{label}.sha256")
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise UpcInitialGroupError(f"{label}.sha256 형식이 올바르지 않습니다.")
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise UpcInitialGroupError(f"입력 파일을 읽을 수 없습니다: {path}") from exc
    if actual_size != expected_size:
        raise UpcInitialGroupError(
            f"{label} 크기가 manifest와 다릅니다: {actual_size} != {expected_size}"
        )
    actual_sha = compute_sha256(path)
    if actual_sha != expected_sha:
        raise UpcInitialGroupError(f"{label} checksum이 manifest와 다릅니다.")
    return {
        "path": _display_path(path),
        "size_bytes": actual_size,
        "sha256": actual_sha,
    }


def _load_npy(path: Path, label: str) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise UpcInitialGroupError(f"{label} NumPy 파일을 읽을 수 없습니다.") from exc


def verify_processed_inputs(
    config: UpcConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """전처리 manifest와 실제 배열의 checksum, shape, dtype을 검증한다."""

    manifest = _load_json(config.inputs.processed_manifest, "전처리 manifest")
    if manifest.get("status") != "complete":
        raise UpcInitialGroupError("전처리 manifest의 status가 complete가 아닙니다.")
    outputs = _require_mapping(manifest.get("outputs"), "processed.outputs")
    verified_files: dict[str, Any] = {}
    for key, path in config.inputs.processed_arrays().items():
        metadata = _require_mapping(outputs.get(key), f"processed.outputs.{key}")
        verified_files[key] = _verify_file_metadata(
            path, metadata, f"processed.outputs.{key}"
        )

    arrays = {
        key: _load_npy(path, key)
        for key, path in config.inputs.processed_arrays().items()
    }
    expected_matrix_shape = (
        config.expected_cell_count,
        config.expected_step_count,
    )
    specifications = {
        "traffic": (expected_matrix_shape, np.dtype("float32")),
        "cell_ids": ((config.expected_cell_count,), np.dtype("int32")),
        "timestamps_ms": ((config.expected_step_count,), np.dtype("int64")),
        "missing_mask": (expected_matrix_shape, np.dtype("bool")),
        "internet_null_mask": (expected_matrix_shape, np.dtype("bool")),
    }
    for key, (expected_shape, expected_dtype) in specifications.items():
        array = arrays[key]
        if array.shape != expected_shape:
            raise UpcInitialGroupError(
                f"{key} shape 불일치: {array.shape} != {expected_shape}"
            )
        if array.dtype != expected_dtype:
            raise UpcInitialGroupError(
                f"{key} dtype 불일치: {array.dtype} != {expected_dtype}"
            )

    cell_ids = arrays["cell_ids"]
    if np.any(np.diff(cell_ids.astype(np.int64)) <= 0):
        raise UpcInitialGroupError("cell_ids는 오름차순의 고유한 값이어야 합니다.")
    timestamps = arrays["timestamps_ms"]
    if not np.all(np.diff(timestamps) == config.interval_ms):
        raise UpcInitialGroupError(
            "timestamps_ms 간격이 config의 interval_ms와 다릅니다."
        )

    contract = _require_mapping(manifest.get("contract"), "processed.contract")
    if contract.get("shape") != list(expected_matrix_shape):
        raise UpcInitialGroupError("전처리 manifest의 shape 계약이 config와 다릅니다.")
    if contract.get("timezone") != config.timezone_name:
        raise UpcInitialGroupError(
            "전처리 manifest의 timezone이 UPC config와 다릅니다."
        )
    if contract.get("interval_ms") != config.interval_ms:
        raise UpcInitialGroupError(
            "전처리 manifest의 interval이 UPC config와 다릅니다."
        )

    return arrays, {
        "manifest_path": _display_path(config.inputs.processed_manifest),
        "manifest_sha256": compute_sha256(config.inputs.processed_manifest),
        "files": verified_files,
        "shape": list(expected_matrix_shape),
        "cell_id_min": int(cell_ids[0]),
        "cell_id_max": int(cell_ids[-1]),
        "timestamp_start_ms": int(timestamps[0]),
        "timestamp_end_ms": int(timestamps[-1]),
    }


def load_central_cells(
    config: UpcConfig,
    all_cell_ids: np.ndarray,
) -> tuple[list[dict[str, str]], np.ndarray, dict[str, Any]]:
    """중앙 900셀 CSV의 무결성과 전체 행렬 매핑을 검증한다."""

    manifest = _load_json(config.inputs.central_manifest, "중앙 900셀 manifest")
    outputs = _require_mapping(manifest.get("outputs"), "central.outputs")
    csv_metadata = _require_mapping(
        outputs.get("central_cells_csv"), "central.outputs.central_cells_csv"
    )
    verified_csv = _verify_file_metadata(
        config.inputs.central_cells_csv,
        csv_metadata,
        "central.outputs.central_cells_csv",
    )
    try:
        with config.inputs.central_cells_csv.open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            required_fields = {
                "cell_id",
                "grid_row",
                "grid_column",
                "centroid_lon",
                "centroid_lat",
            }
            if reader.fieldnames is None or not required_fields.issubset(
                reader.fieldnames
            ):
                raise UpcInitialGroupError("중앙 900셀 CSV에 필요한 열이 없습니다.")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise UpcInitialGroupError("중앙 900셀 CSV를 읽을 수 없습니다.") from exc

    if len(rows) != config.expected_central_cell_count:
        raise UpcInitialGroupError(
            "중앙 900셀 CSV 행 수가 config와 다릅니다: "
            f"{len(rows)} != {config.expected_central_cell_count}"
        )
    try:
        central_ids = np.asarray([int(row["cell_id"]) for row in rows], dtype=np.int32)
    except (TypeError, ValueError) as exc:
        raise UpcInitialGroupError(
            "중앙 900셀 CSV의 cell_id가 정수가 아닙니다."
        ) from exc
    if len(np.unique(central_ids)) != len(central_ids):
        raise UpcInitialGroupError("중앙 900셀 CSV에 중복 cell_id가 있습니다.")

    positions = np.searchsorted(all_cell_ids, central_ids)
    if np.any(positions >= len(all_cell_ids)) or not np.array_equal(
        all_cell_ids[positions], central_ids
    ):
        raise UpcInitialGroupError(
            "중앙 900셀 cell_id를 전체 전처리 행렬에 매핑할 수 없습니다."
        )

    selection = _require_mapping(manifest.get("selection"), "central.selection")
    if selection.get("cell_count") != config.expected_central_cell_count:
        raise UpcInitialGroupError("중앙 manifest의 cell_count가 config와 다릅니다.")
    expected_id_sha = _require_string(
        selection.get("cell_ids_int32_sha256"),
        "central.selection.cell_ids_int32_sha256",
    )
    actual_id_sha = hashlib.sha256(central_ids.astype("<i4").tobytes()).hexdigest()
    if actual_id_sha != expected_id_sha:
        raise UpcInitialGroupError("중앙 900셀 ID 순서 checksum이 manifest와 다릅니다.")

    return (
        rows,
        positions.astype(np.int64),
        {
            "manifest_path": _display_path(config.inputs.central_manifest),
            "manifest_sha256": compute_sha256(config.inputs.central_manifest),
            "cells_csv": verified_csv,
            "cell_count": len(rows),
            "cell_ids_int32_sha256": actual_id_sha,
        },
    )


def build_local_time_axis(
    timestamps_ms: np.ndarray,
    *,
    timezone_name: str,
    interval_ms: int,
    observations_per_hour: int,
) -> LocalTimeAxis:
    """epoch millisecond 축을 local date/hour별 10분 index로 변환한다."""

    if timestamps_ms.ndim != 1 or len(timestamps_ms) == 0:
        raise UpcInitialGroupError(
            "timestamps_ms는 비어 있지 않은 1차원 배열이어야 합니다."
        )
    if len(timestamps_ms) % observations_per_hour:
        raise UpcInitialGroupError(
            "timestamp 수가 observations_per_hour로 나누어지지 않습니다."
        )
    if not np.all(np.diff(timestamps_ms) == interval_ms):
        raise UpcInitialGroupError("timestamp 간격이 일정하지 않습니다.")

    zone = ZoneInfo(timezone_name)
    local_times = tuple(
        datetime.fromtimestamp(int(timestamp) / 1000, tz=zone)
        for timestamp in timestamps_ms
    )
    interval_minutes = interval_ms // 60_000
    expected_minutes = tuple(
        index * interval_minutes for index in range(observations_per_hour)
    )
    hour_indices: dict[tuple[date, int], np.ndarray] = {}
    all_dates: list[date] = []
    for start in range(0, len(local_times), observations_per_hour):
        stop = start + observations_per_hour
        hour_times = local_times[start:stop]
        first = hour_times[0]
        key = (first.date(), first.hour)
        if key in hour_indices:
            raise UpcInitialGroupError(
                "같은 local date/hour가 두 번 나타납니다. DST 중복 시간을 별도 "
                f"정책 없이 처리할 수 없습니다: {key}"
            )
        if any(
            value.date() != first.date() or value.hour != first.hour
            for value in hour_times
        ):
            raise UpcInitialGroupError(
                f"한 시간 묶음이 local hour 경계를 넘습니다: {first.isoformat()}"
            )
        actual_minutes = tuple(value.minute for value in hour_times)
        if actual_minutes != expected_minutes or any(
            value.second != 0 or value.microsecond != 0 for value in hour_times
        ):
            raise UpcInitialGroupError(
                "local hour의 관측 minute 구성이 계약과 다릅니다: "
                f"{first.date()} hour={first.hour}, minutes={actual_minutes}"
            )
        hour_indices[key] = np.arange(start, stop, dtype=np.int64)
        if not all_dates or all_dates[-1] != first.date():
            all_dates.append(first.date())

    for local_date in all_dates:
        hours = [hour for day, hour in hour_indices if day == local_date]
        if sorted(hours) != list(range(HOURS_PER_DAY)):
            raise UpcInitialGroupError(
                f"{local_date.isoformat()}의 local hour가 0~23 전체가 아닙니다."
            )

    return LocalTimeAxis(
        timezone_name=timezone_name,
        hour_indices=hour_indices,
        dates=tuple(all_dates),
        first_local_timestamp=local_times[0].isoformat(),
        last_local_timestamp=local_times[-1].isoformat(),
    )


def select_protocol_hours(
    axis: LocalTimeAxis,
    protocol: ProtocolSpec,
) -> ProtocolHours:
    """프로토콜 기간에 속한 평일을 (day, hour, observation) index로 만든다."""

    available_dates = set(axis.dates)
    selected_dates: list[date] = []
    current = protocol.start_local.date()
    end = protocol.end_exclusive_local.date()
    while current < end:
        if current not in available_dates:
            raise UpcInitialGroupError(
                f"{protocol.name} 기간의 날짜가 timestamp 축에 없습니다: {current}"
            )
        if not protocol.weekdays_only or current.weekday() < 5:
            selected_dates.append(current)
        current += timedelta(days=1)
    if not selected_dates:
        raise UpcInitialGroupError(f"{protocol.name}에 선택된 평일이 없습니다.")

    day_indices: list[np.ndarray] = []
    for selected_date in selected_dates:
        try:
            day_indices.append(
                np.stack(
                    [
                        axis.hour_indices[(selected_date, hour)]
                        for hour in range(HOURS_PER_DAY)
                    ]
                )
            )
        except KeyError as exc:
            raise UpcInitialGroupError(
                f"{selected_date.isoformat()}의 시간 index가 완전하지 않습니다."
            ) from exc
    indices = np.stack(day_indices).astype(np.int64, copy=False)
    return ProtocolHours(
        name=protocol.name,
        dates=tuple(selected_dates),
        indices=indices,
    )


def aggregate_hourly_traffic(
    traffic: np.ndarray,
    hour_indices: np.ndarray,
) -> np.ndarray:
    """선택된 10분 traffic을 (cell, day, 24 hour) 합계로 변환한다."""

    if traffic.ndim != 2:
        raise UpcInitialGroupError("traffic은 (cell, time) 2차원 배열이어야 합니다.")
    if hour_indices.ndim != 3 or hour_indices.shape[1] != HOURS_PER_DAY:
        raise UpcInitialGroupError(
            "hour_indices는 (day, 24, observations_per_hour)여야 합니다."
        )
    flat_indices = hour_indices.reshape(-1)
    if len(flat_indices) == 0 or int(flat_indices.min()) < 0:
        raise UpcInitialGroupError(
            "hour_indices가 비어 있거나 음수 index를 포함합니다."
        )
    if int(flat_indices.max()) >= traffic.shape[1]:
        raise UpcInitialGroupError("hour_indices가 traffic 시간축을 벗어납니다.")
    selected = np.asarray(traffic[:, flat_indices], dtype=np.float64)
    shaped = selected.reshape(
        traffic.shape[0],
        hour_indices.shape[0],
        HOURS_PER_DAY,
        hour_indices.shape[2],
    )
    return shaped.sum(axis=3, dtype=np.float64)


def _representative_peak_hours(daily_peaks: np.ndarray) -> tuple[np.ndarray, int]:
    """날짜별 peak의 최빈 hour를 고르며 동률이면 가장 이른 hour를 택한다."""

    if daily_peaks.ndim != 2 or daily_peaks.shape[1] == 0:
        raise UpcInitialGroupError(
            "daily_peaks는 한 개 이상의 날짜를 가진 2차원 배열이어야 합니다."
        )
    hours = np.arange(HOURS_PER_DAY, dtype=np.int8)
    counts = np.count_nonzero(daily_peaks[:, :, None] == hours, axis=1)
    maxima = counts.max(axis=1, keepdims=True)
    tie_count = int(np.count_nonzero(np.count_nonzero(counts == maxima, axis=1) > 1))
    return np.argmax(counts, axis=1).astype(np.int8), tie_count


def compute_protocol_peak_hours(
    traffic: np.ndarray,
    missing_mask: np.ndarray,
    internet_null_mask: np.ndarray,
    protocol_hours: ProtocolHours,
    *,
    cell_chunk_size: int,
) -> ProtocolResult:
    """셀별 scaling, 시간 합산, 일별 peak와 대표 peak를 메모리 제한으로 계산한다."""

    if traffic.shape != missing_mask.shape or traffic.shape != internet_null_mask.shape:
        raise UpcInitialGroupError("traffic과 결측 mask shape가 서로 다릅니다.")
    if traffic.ndim != 2:
        raise UpcInitialGroupError("traffic과 결측 mask는 2차원이어야 합니다.")
    if cell_chunk_size < 1:
        raise UpcInitialGroupError("cell_chunk_size는 1 이상이어야 합니다.")

    flat_indices = protocol_hours.indices.reshape(-1)
    cell_count = traffic.shape[0]
    day_count = protocol_hours.indices.shape[0]
    observations_per_hour = protocol_hours.indices.shape[2]
    peak_hours = np.empty(cell_count, dtype=np.int8)
    constant_cell_count = 0
    daily_peak_tie_count = 0
    representative_tie_count = 0
    scaling_invariance_mismatch_count = 0
    missing_pair_count = 0
    internet_null_pair_count = 0

    for start in range(0, cell_count, cell_chunk_size):
        stop = min(start + cell_chunk_size, cell_count)
        raw_selected = np.asarray(
            traffic[start:stop, flat_indices], dtype=np.float64
        ).reshape(
            stop - start,
            day_count,
            HOURS_PER_DAY,
            observations_per_hour,
        )
        minimum = raw_selected.min(axis=(1, 2, 3), keepdims=True)
        maximum = raw_selected.max(axis=(1, 2, 3), keepdims=True)
        span = maximum - minimum
        constant_cell_count += int(np.count_nonzero(span.reshape(-1) == 0.0))

        raw_hourly = raw_selected.sum(axis=3, dtype=np.float64)
        scaled_hourly = np.zeros_like(raw_hourly)
        numerator = raw_hourly - observations_per_hour * minimum.reshape(-1, 1, 1)
        np.divide(
            numerator,
            span.reshape(-1, 1, 1),
            out=scaled_hourly,
            where=span.reshape(-1, 1, 1) != 0.0,
        )
        raw_daily_peaks = np.argmax(raw_hourly, axis=2).astype(np.int8)
        scaled_daily_peaks = np.argmax(scaled_hourly, axis=2).astype(np.int8)
        scaling_invariance_mismatch_count += int(
            np.count_nonzero(raw_daily_peaks != scaled_daily_peaks)
        )

        daily_maxima = scaled_hourly.max(axis=2, keepdims=True)
        daily_peak_tie_count += int(
            np.count_nonzero(
                np.count_nonzero(scaled_hourly == daily_maxima, axis=2) > 1
            )
        )
        representative, mode_ties = _representative_peak_hours(scaled_daily_peaks)
        peak_hours[start:stop] = representative
        representative_tie_count += mode_ties

        missing_pair_count += int(
            np.count_nonzero(missing_mask[start:stop, flat_indices])
        )
        internet_null_pair_count += int(
            np.count_nonzero(internet_null_mask[start:stop, flat_indices])
        )

    if scaling_invariance_mismatch_count:
        raise UpcInitialGroupError(
            "min-max scaling 전후의 일별 peak hour가 달라졌습니다: "
            f"{scaling_invariance_mismatch_count}개 cell-day"
        )
    group_counts = np.bincount(peak_hours, minlength=HOURS_PER_DAY).astype(np.int64)
    if len(group_counts) != HOURS_PER_DAY or int(group_counts.sum()) != cell_count:
        raise UpcInitialGroupError("24개 초기 그룹에 모든 cell이 배정되지 않았습니다.")

    selected_observations = cell_count * len(flat_indices)
    diagnostics: dict[str, Any] = {
        "start_local_date": protocol_hours.dates[0].isoformat(),
        "end_local_date_inclusive": protocol_hours.dates[-1].isoformat(),
        "weekday_count": day_count,
        "weekdays": [value.isoformat() for value in protocol_hours.dates],
        "selected_10min_steps_per_cell": len(flat_indices),
        "selected_observations": selected_observations,
        "scaling": "per-cell min-max fitted on the protocol's selected weekdays",
        "constant_cell_count": constant_cell_count,
        "daily_peak_tie_cell_day_count": daily_peak_tie_count,
        "representative_mode_tie_cell_count": representative_tie_count,
        "scaling_invariance_mismatch_cell_day_count": (
            scaling_invariance_mismatch_count
        ),
        "missing_pair_count": missing_pair_count,
        "missing_pair_ratio": missing_pair_count / selected_observations,
        "internet_all_null_pair_count": internet_null_pair_count,
        "internet_all_null_pair_ratio": (
            internet_null_pair_count / selected_observations
        ),
        "tie_break": "earliest hour",
        "group_counts_hour_0_to_23": group_counts.tolist(),
        "assigned_cell_count": int(group_counts.sum()),
    }
    return ProtocolResult(peak_hours=peak_hours, diagnostics=diagnostics)


def compute_mean_profile_peak_hours(
    traffic: np.ndarray,
    missing_mask: np.ndarray,
    internet_null_mask: np.ndarray,
    protocol_hours: ProtocolHours,
    *,
    cell_chunk_size: int,
) -> ProtocolResult:
    """Fig. 4와 가까운 비명시적 평균-profile 가설을 진단용으로 계산한다."""

    if traffic.shape != missing_mask.shape or traffic.shape != internet_null_mask.shape:
        raise UpcInitialGroupError("traffic과 결측 mask shape가 서로 다릅니다.")
    if traffic.ndim != 2:
        raise UpcInitialGroupError("traffic과 결측 mask는 2차원이어야 합니다.")
    flat_indices = protocol_hours.indices.reshape(-1)
    cell_count = traffic.shape[0]
    day_count = protocol_hours.indices.shape[0]
    observations_per_hour = protocol_hours.indices.shape[2]
    peak_hours = np.empty(cell_count, dtype=np.int8)
    constant_cell_count = 0
    profile_peak_tie_count = 0
    scaling_invariance_mismatch_count = 0
    missing_pair_count = 0
    internet_null_pair_count = 0

    for start in range(0, cell_count, cell_chunk_size):
        stop = min(start + cell_chunk_size, cell_count)
        raw_selected = np.asarray(
            traffic[start:stop, flat_indices], dtype=np.float64
        ).reshape(
            stop - start,
            day_count,
            HOURS_PER_DAY,
            observations_per_hour,
        )
        minimum = raw_selected.min(axis=(1, 2, 3), keepdims=True)
        maximum = raw_selected.max(axis=(1, 2, 3), keepdims=True)
        span = maximum - minimum
        constant_cell_count += int(np.count_nonzero(span.reshape(-1) == 0.0))

        raw_hourly = raw_selected.sum(axis=3, dtype=np.float64)
        scaled_hourly = np.zeros_like(raw_hourly)
        numerator = raw_hourly - observations_per_hour * minimum.reshape(-1, 1, 1)
        np.divide(
            numerator,
            span.reshape(-1, 1, 1),
            out=scaled_hourly,
            where=span.reshape(-1, 1, 1) != 0.0,
        )
        raw_profile = raw_hourly.mean(axis=1)
        scaled_profile = scaled_hourly.mean(axis=1)
        raw_peak = np.argmax(raw_profile, axis=1).astype(np.int8)
        scaled_peak = np.argmax(scaled_profile, axis=1).astype(np.int8)
        scaling_invariance_mismatch_count += int(
            np.count_nonzero(raw_peak != scaled_peak)
        )
        maxima = scaled_profile.max(axis=1, keepdims=True)
        profile_peak_tie_count += int(
            np.count_nonzero(np.count_nonzero(scaled_profile == maxima, axis=1) > 1)
        )
        peak_hours[start:stop] = scaled_peak
        missing_pair_count += int(
            np.count_nonzero(missing_mask[start:stop, flat_indices])
        )
        internet_null_pair_count += int(
            np.count_nonzero(internet_null_mask[start:stop, flat_indices])
        )

    if scaling_invariance_mismatch_count:
        raise UpcInitialGroupError(
            "Fig. 4 진단에서 min-max scaling 전후 peak가 달라졌습니다: "
            f"{scaling_invariance_mismatch_count}개 cell"
        )
    group_counts = np.bincount(peak_hours, minlength=HOURS_PER_DAY).astype(np.int64)
    if len(group_counts) != HOURS_PER_DAY or int(group_counts.sum()) != cell_count:
        raise UpcInitialGroupError(
            "Fig. 4 진단 그룹에 모든 cell이 배정되지 않았습니다."
        )

    selected_observations = cell_count * len(flat_indices)
    diagnostics: dict[str, Any] = {
        "status": "diagnostic_hypothesis_not_algorithm_1",
        "method": "mean hourly profile over complete-week weekdays, then argmax",
        "reason": (
            "This unreported variant is retained only because it closely matches "
            "the published Fig. 4 counts."
        ),
        "start_local_date": protocol_hours.dates[0].isoformat(),
        "end_local_date_inclusive": protocol_hours.dates[-1].isoformat(),
        "weekday_count": day_count,
        "weekdays": [value.isoformat() for value in protocol_hours.dates],
        "selected_10min_steps_per_cell": len(flat_indices),
        "selected_observations": selected_observations,
        "scaling": "per-cell min-max fitted on the diagnostic weekdays",
        "constant_cell_count": constant_cell_count,
        "profile_peak_tie_cell_count": profile_peak_tie_count,
        "scaling_invariance_mismatch_cell_count": (scaling_invariance_mismatch_count),
        "missing_pair_count": missing_pair_count,
        "missing_pair_ratio": missing_pair_count / selected_observations,
        "internet_all_null_pair_count": internet_null_pair_count,
        "internet_all_null_pair_ratio": (
            internet_null_pair_count / selected_observations
        ),
        "tie_break": "earliest hour",
        "group_counts_hour_0_to_23": group_counts.tolist(),
        "assigned_cell_count": int(group_counts.sum()),
    }
    return ProtocolResult(peak_hours=peak_hours, diagnostics=diagnostics)


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.partial")


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _write_all_memberships(
    path: Path,
    cell_ids: np.ndarray,
    paper_peak_hours: np.ndarray,
    train_peak_hours: np.ndarray,
    figure4_diagnostic_peak_hours: np.ndarray,
    central_positions: np.ndarray,
) -> None:
    central_mask = np.zeros(len(cell_ids), dtype=bool)
    central_mask[central_positions] = True
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "cell_id",
                "paper_faithful_peak_hour",
                "train_only_peak_hour",
                "figure4_diagnostic_peak_hour",
                "is_central_900",
            ]
        )
        for index, cell_id in enumerate(cell_ids):
            writer.writerow(
                [
                    int(cell_id),
                    int(paper_peak_hours[index]),
                    int(train_peak_hours[index]),
                    int(figure4_diagnostic_peak_hours[index]),
                    int(central_mask[index]),
                ]
            )


def _write_central_memberships(
    path: Path,
    central_rows: Sequence[Mapping[str, str]],
    central_positions: np.ndarray,
    paper_peak_hours: np.ndarray,
    train_peak_hours: np.ndarray,
    figure4_diagnostic_peak_hours: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "cell_id",
                "grid_row",
                "grid_column",
                "centroid_lon",
                "centroid_lat",
                "paper_faithful_peak_hour",
                "train_only_peak_hour",
                "figure4_diagnostic_peak_hour",
            ]
        )
        for row, position in zip(central_rows, central_positions, strict=True):
            writer.writerow(
                [
                    row["cell_id"],
                    row["grid_row"],
                    row["grid_column"],
                    row["centroid_lon"],
                    row["centroid_lat"],
                    int(paper_peak_hours[position]),
                    int(train_peak_hours[position]),
                    int(figure4_diagnostic_peak_hours[position]),
                ]
            )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
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
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def _estimated_chunk_working_set_bytes(
    cell_chunk_size: int,
    maximum_weekday_count: int,
    observations_per_hour: int,
) -> int:
    selected_values = (
        cell_chunk_size * maximum_weekday_count * HOURS_PER_DAY * observations_per_hour
    )
    hourly_values = cell_chunk_size * maximum_weekday_count * HOURS_PER_DAY
    return selected_values * (4 + 8 + 2) + hourly_values * (8 * 2 + 1)


def run_upc_initial_groups(config: UpcConfig) -> dict[str, Any]:
    """입력을 검증하고 두 프로토콜의 초기 그룹과 진단 manifest를 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    arrays, processed_validation = verify_processed_inputs(config)
    central_rows, central_positions, central_validation = load_central_cells(
        config, arrays["cell_ids"]
    )
    axis = build_local_time_axis(
        arrays["timestamps_ms"],
        timezone_name=config.timezone_name,
        interval_ms=config.interval_ms,
        observations_per_hour=config.observations_per_hour,
    )
    protocol_hours = {
        name: select_protocol_hours(axis, config.protocols[name])
        for name in PROTOCOL_NAMES
    }
    results = {
        name: compute_protocol_peak_hours(
            arrays["traffic"],
            arrays["missing_mask"],
            arrays["internet_null_mask"],
            protocol_hours[name],
            cell_chunk_size=config.cell_chunk_size,
        )
        for name in PROTOCOL_NAMES
    }
    figure4_hours = select_protocol_hours(axis, config.figure4_diagnostic)
    figure4_result = compute_mean_profile_peak_hours(
        arrays["traffic"],
        arrays["missing_mask"],
        arrays["internet_null_mask"],
        figure4_hours,
        cell_chunk_size=config.cell_chunk_size,
    )

    expected = np.asarray(config.paper_fingerprint, dtype=np.int64)
    actual = np.asarray(
        results["paper_faithful"].diagnostics["group_counts_hour_0_to_23"],
        dtype=np.int64,
    )
    difference = actual - expected
    paper_match = bool(np.array_equal(actual, expected))
    figure4_actual = np.asarray(
        figure4_result.diagnostics["group_counts_hour_0_to_23"],
        dtype=np.int64,
    )
    figure4_difference = figure4_actual - expected
    figure4_match = bool(np.array_equal(figure4_actual, expected))
    paper_result = results["paper_faithful"].peak_hours
    train_result = results["train_only"].peak_hours
    figure4_peaks = figure4_result.peak_hours
    agreement_count = int(np.count_nonzero(paper_result == train_result))
    central_paper_counts = np.bincount(
        paper_result[central_positions], minlength=HOURS_PER_DAY
    ).tolist()
    central_train_counts = np.bincount(
        train_result[central_positions], minlength=HOURS_PER_DAY
    ).tolist()
    central_figure4_counts = np.bincount(
        figure4_peaks[central_positions], minlength=HOURS_PER_DAY
    ).tolist()

    counts_report: dict[str, Any] = {
        "schema_version": 1,
        "paper_fingerprint": {
            "expected_hour_0_to_23": expected.tolist(),
            "actual_hour_0_to_23": actual.tolist(),
            "difference_actual_minus_expected": difference.tolist(),
            "l1_difference": int(np.abs(difference).sum()),
            "exact_match": paper_match,
        },
        "figure4_diagnostic": {
            **figure4_result.diagnostics,
            "expected_hour_0_to_23": expected.tolist(),
            "difference_actual_minus_expected": figure4_difference.tolist(),
            "l1_difference": int(np.abs(figure4_difference).sum()),
            "exact_match": figure4_match,
        },
        "protocols": {name: results[name].diagnostics for name in PROTOCOL_NAMES},
        "protocol_membership_comparison": {
            "same_peak_hour_cell_count": agreement_count,
            "different_peak_hour_cell_count": config.expected_cell_count
            - agreement_count,
            "agreement_ratio": agreement_count / config.expected_cell_count,
        },
        "central_900": {
            "cell_count": config.expected_central_cell_count,
            "paper_faithful_group_counts_hour_0_to_23": central_paper_counts,
            "train_only_group_counts_hour_0_to_23": central_train_counts,
            "figure4_diagnostic_group_counts_hour_0_to_23": (central_figure4_counts),
        },
    }

    output_paths = config.outputs.as_dict()
    temporary_paths = {key: _temporary_path(path) for key, path in output_paths.items()}
    for output in output_paths.values():
        output.parent.mkdir(parents=True, exist_ok=True)

    published = False
    try:
        _write_npy(temporary_paths["paper_faithful_peak_hours"], paper_result)
        _write_npy(temporary_paths["train_only_peak_hours"], train_result)
        _write_npy(temporary_paths["figure4_diagnostic_peak_hours"], figure4_peaks)
        _write_all_memberships(
            temporary_paths["all_cell_memberships_csv"],
            arrays["cell_ids"],
            paper_result,
            train_result,
            figure4_peaks,
            central_positions,
        )
        _write_central_memberships(
            temporary_paths["central_900_memberships_csv"],
            central_rows,
            central_positions,
            paper_result,
            train_result,
            figure4_peaks,
        )
        temporary_paths["group_counts_json"].write_text(
            json.dumps(counts_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        deterministic_keys = [
            "paper_faithful_peak_hours",
            "train_only_peak_hours",
            "figure4_diagnostic_peak_hours",
            "all_cell_memberships_csv",
            "central_900_memberships_csv",
            "group_counts_json",
        ]
        output_metadata = {
            key: {
                "path": _display_path(output_paths[key]),
                "size_bytes": temporary_paths[key].stat().st_size,
                "sha256": compute_sha256(temporary_paths[key]),
            }
            for key in deterministic_keys
        }
        finished_at = datetime.now(timezone.utc)
        maximum_weekdays = max(
            [len(hours.dates) for hours in protocol_hours.values()]
            + [len(figure4_hours.dates)]
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete" if paper_match else "diagnostic_mismatch",
            "tool": {
                "name": "scripts.build_upc_initial_groups",
                "version": TOOL_VERSION,
            },
            "dataset": config.name,
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "inputs": {
                "processed": processed_validation,
                "central_900": central_validation,
            },
            "time_axis": {
                "timezone": axis.timezone_name,
                "first_local_timestamp": axis.first_local_timestamp,
                "last_local_timestamp": axis.last_local_timestamp,
                "calendar_day_count": len(axis.dates),
                "interval_ms": config.interval_ms,
                "observations_per_hour": config.observations_per_hour,
            },
            "algorithm_contract": {
                "hourly_aggregation": "sum six consecutive 10-minute values",
                "day_filter": "Monday through Friday in Europe/Rome",
                "scaling": (
                    "per-cell min-max fitted independently within each protocol's "
                    "selected weekdays"
                ),
                "daily_peak": "argmax over local hour 0..23",
                "representative_peak": "mode of daily peak hours",
                "tie_break": "earliest hour",
                "filled_value_policy": (
                    "preprocessing traffic value 0 is retained; missing and "
                    "internet-all-null masks are counted for diagnostics"
                ),
            },
            "paper_fingerprint": counts_report["paper_fingerprint"],
            "figure4_diagnostic": counts_report["figure4_diagnostic"],
            "protocols": counts_report["protocols"],
            "protocol_membership_comparison": counts_report[
                "protocol_membership_comparison"
            ],
            "central_900": counts_report["central_900"],
            "outputs": output_metadata,
            "runtime": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "cell_chunk_size": config.cell_chunk_size,
                "estimated_chunk_working_set_bytes": (
                    _estimated_chunk_working_set_bytes(
                        config.cell_chunk_size,
                        maximum_weekdays,
                        config.observations_per_hour,
                    )
                ),
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        }
        temporary_paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for key in deterministic_keys + ["manifest"]:
            os.replace(temporary_paths[key], output_paths[key])
        published = True
        return manifest
    finally:
        if not published:
            for temporary in temporary_paths.values():
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GECOS UPC의 24개 peak-hour 초기 그룹을 만들고 논문 분포와 비교합니다."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"UPC 초기 그룹 config (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_upc_config(args.config)
        manifest = run_upc_initial_groups(config)
    except UpcInitialGroupError as exc:
        print(f"UPC 초기 그룹 생성 실패: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"파일 시스템 오류: {exc}", file=sys.stderr)
        return 2

    fingerprint = manifest["paper_fingerprint"]
    print(
        "UPC 초기 그룹 생성 완료: "
        f"paper_match={fingerprint['exact_match']}, "
        f"l1_difference={fingerprint['l1_difference']}"
    )
    for name in PROTOCOL_NAMES:
        protocol = manifest["protocols"][name]
        print(
            f"{name}: weekdays={protocol['weekday_count']}, "
            f"counts={protocol['group_counts_hour_0_to_23']}"
        )
    figure4 = manifest["figure4_diagnostic"]
    print(
        "figure4_complete_weeks_mean_profile: "
        f"weekdays={figure4['weekday_count']}, "
        f"l1_difference={figure4['l1_difference']}, "
        f"counts={figure4['group_counts_hour_0_to_23']}"
    )
    print(f"manifest={_display_path(config.outputs.manifest)}")
    if config.require_exact_paper_fingerprint and not fingerprint["exact_match"]:
        print(
            "논문 fingerprint와 다르므로 다음 UPC 단계로 진행하지 않습니다. "
            "manifest의 시간대별 차이를 진단하세요.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
