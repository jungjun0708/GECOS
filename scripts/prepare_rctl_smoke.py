#!/usr/bin/env python3
"""실제 Milano Train 구간에서 작고 결정적인 RCTL smoke 입력을 만든다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_upc_initial_groups import (
    _display_path,
    _git_state,
    _peak_rss_bytes,
    _temporary_path,
    compute_sha256,
    load_central_cells,
    load_upc_config,
    verify_processed_inputs,
)
from scripts.forecast_contract import (
    ForecastIndexContract,
    build_forecast_index_contract,
    load_forecast_config,
)
from scripts.rctl_contract import (
    DEFAULT_CONFIG,
    RctlContractError,
    RctlSmokeConfig,
    SelectionSpec,
    load_rctl_smoke_config,
)

TOOL_VERSION = "1.0.0"


def evenly_spaced_positions(item_count: int, selected_count: int) -> np.ndarray:
    """양 끝을 포함한 결정적 위치를 반올림으로 선택한다."""

    if item_count < 1 or selected_count < 1 or selected_count > item_count:
        raise RctlContractError("선택 수는 1 이상 전체 항목 수 이하여야 합니다.")
    positions = np.rint(np.linspace(0, item_count - 1, selected_count)).astype(
        np.int64
    )
    if len(np.unique(positions)) != selected_count:
        raise RctlContractError("등간격 선택 결과에 중복 위치가 생겼습니다.")
    return positions


def select_spatially_spread_cells(
    central_rows: Sequence[Mapping[str, str]],
    central_positions: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """중앙 격자의 가로·세로 등간격 교차점에서 셀을 선택한다."""

    if len(central_rows) != len(central_positions):
        raise RctlContractError("중앙 셀 행과 전체 행렬 위치 수가 다릅니다.")
    try:
        grid_rows = sorted({int(row["grid_row"]) for row in central_rows})
        grid_columns = sorted({int(row["grid_column"]) for row in central_rows})
        coordinate_to_index = {
            (int(row["grid_row"]), int(row["grid_column"])): index
            for index, row in enumerate(central_rows)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RctlContractError("중앙 셀 좌표를 정수로 읽을 수 없습니다.") from exc
    if len(coordinate_to_index) != len(central_rows):
        raise RctlContractError("중앙 셀 CSV에 중복 격자 좌표가 있습니다.")
    if len(grid_rows) < side or len(grid_columns) < side:
        raise RctlContractError("중앙 격자가 요청한 smoke grid보다 작습니다.")

    selected_rows = [grid_rows[index] for index in evenly_spaced_positions(len(grid_rows), side)]
    selected_columns = [
        grid_columns[index]
        for index in evenly_spaced_positions(len(grid_columns), side)
    ]
    central_indices: list[int] = []
    coordinates: list[dict[str, int]] = []
    for grid_row in selected_rows:
        for grid_column in selected_columns:
            key = (grid_row, grid_column)
            if key not in coordinate_to_index:
                raise RctlContractError(f"중앙 격자 교차점 {key}에 셀이 없습니다.")
            central_index = coordinate_to_index[key]
            central_indices.append(central_index)
            coordinates.append(
                {
                    "cell_id": int(central_rows[central_index]["cell_id"]),
                    "grid_row": grid_row,
                    "grid_column": grid_column,
                    "all_cell_matrix_position": int(central_positions[central_index]),
                }
            )
    selected_central_indices = np.asarray(central_indices, dtype=np.int64)
    selected_matrix_positions = np.asarray(
        central_positions[selected_central_indices], dtype=np.int64
    )
    return selected_central_indices, selected_matrix_positions, coordinates


def build_smoke_arrays(
    *,
    traffic: np.ndarray,
    missing_mask: np.ndarray,
    internet_null_mask: np.ndarray,
    cell_ids: np.ndarray,
    timestamps_ms: np.ndarray,
    central_rows: Sequence[Mapping[str, str]],
    central_positions: np.ndarray,
    train_target_indices: np.ndarray,
    input_length: int,
    selection: SelectionSpec,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """선택한 16셀 × 64 target을 미래 누수 없는 window로 변환한다."""

    matrix_shape = traffic.shape
    if traffic.ndim != 2 or any(
        value.shape != matrix_shape for value in (missing_mask, internet_null_mask)
    ):
        raise RctlContractError("traffic과 두 결측 mask의 shape가 다릅니다.")
    if len(cell_ids) != matrix_shape[0] or len(timestamps_ms) != matrix_shape[1]:
        raise RctlContractError("traffic 축과 cell/timestamp 축 길이가 다릅니다.")
    if len(train_target_indices) < selection.windows_per_cell:
        raise RctlContractError("Train target 수가 smoke window 수보다 작습니다.")

    _, matrix_positions, coordinates = select_spatially_spread_cells(
        central_rows, central_positions, selection.central_grid_side
    )
    target_positions = evenly_spaced_positions(
        len(train_target_indices), selection.windows_per_cell
    )
    selected_targets = np.asarray(train_target_indices[target_positions], dtype=np.int64)
    if int(selected_targets[0]) < input_length:
        raise RctlContractError("첫 target 앞에 필요한 과거 입력이 없습니다.")
    offsets = np.arange(-input_length, 0, dtype=np.int64)
    window_indices = selected_targets[:, None] + offsets[None, :]
    if np.any(window_indices >= selected_targets[:, None]):
        raise RctlContractError("smoke 입력 window에 target 또는 미래값이 포함됩니다.")

    sample_count = selection.sample_count
    x = np.empty((sample_count, input_length, 1), dtype=np.float32)
    y = np.empty((sample_count, 1), dtype=np.float32)
    persistence = np.empty((sample_count, 1), dtype=np.float32)
    target_missing = np.empty((sample_count,), dtype=bool)
    target_internet_null = np.empty((sample_count,), dtype=bool)
    input_missing = np.empty((sample_count, input_length), dtype=bool)
    input_internet_null = np.empty((sample_count, input_length), dtype=bool)
    sample_cell_ids = np.repeat(
        np.asarray(cell_ids[matrix_positions], dtype=np.int32),
        selection.windows_per_cell,
    )
    sample_target_indices = np.tile(selected_targets, selection.cell_count)
    sample_target_timestamps_ms = np.asarray(
        timestamps_ms[sample_target_indices], dtype=np.int64
    )

    cursor = 0
    for matrix_position in matrix_positions:
        stop = cursor + selection.windows_per_cell
        x[cursor:stop, :, 0] = np.asarray(
            traffic[matrix_position, window_indices], dtype=np.float32
        )
        y[cursor:stop, 0] = np.asarray(
            traffic[matrix_position, selected_targets], dtype=np.float32
        )
        persistence[cursor:stop, 0] = np.asarray(
            traffic[matrix_position, selected_targets - 1], dtype=np.float32
        )
        target_missing[cursor:stop] = np.asarray(
            missing_mask[matrix_position, selected_targets], dtype=bool
        )
        target_internet_null[cursor:stop] = np.asarray(
            internet_null_mask[matrix_position, selected_targets], dtype=bool
        )
        input_missing[cursor:stop] = np.asarray(
            missing_mask[matrix_position, window_indices], dtype=bool
        )
        input_internet_null[cursor:stop] = np.asarray(
            internet_null_mask[matrix_position, window_indices], dtype=bool
        )
        cursor = stop

    arrays = {
        "x": x,
        "y": y,
        "persistence": persistence,
        "cell_ids": sample_cell_ids,
        "target_indices": sample_target_indices,
        "target_timestamps_ms": sample_target_timestamps_ms,
        "target_missing_mask": target_missing,
        "target_internet_null_mask": target_internet_null,
        "input_missing_mask": input_missing,
        "input_internet_null_mask": input_internet_null,
    }
    for name in ("x", "y", "persistence"):
        if not np.all(np.isfinite(arrays[name])):
            raise RctlContractError(f"smoke {name}에 NaN 또는 무한대가 있습니다.")
    if np.any(x < 0) or np.any(y < 0) or np.any(persistence < 0):
        raise RctlContractError("smoke traffic 값은 음수일 수 없습니다.")
    metadata = {
        "selection_protocol": {
            "cell_policy": selection.cell_policy,
            "target_policy": selection.target_policy,
            "selected_grid_rows": sorted({item["grid_row"] for item in coordinates}),
            "selected_grid_columns": sorted(
                {item["grid_column"] for item in coordinates}
            ),
            "cells": coordinates,
            "target_position_indices_within_train": target_positions.tolist(),
            "target_indices": selected_targets.tolist(),
        },
        "sample_count": sample_count,
        "cell_count": selection.cell_count,
        "windows_per_cell": selection.windows_per_cell,
        "input_length": input_length,
        "first_target_index": int(selected_targets[0]),
        "last_target_index": int(selected_targets[-1]),
        "target_missing_count": int(target_missing.sum()),
        "target_internet_null_count": int(target_internet_null.sum()),
        "input_missing_count": int(input_missing.sum()),
        "input_internet_null_count": int(input_internet_null.sum()),
        "traffic_statistics": {
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "x_mean": float(np.mean(x, dtype=np.float64)),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_mean": float(np.mean(y, dtype=np.float64)),
            "persistence_mae": float(
                np.mean(np.abs(y.astype(np.float64) - persistence), dtype=np.float64)
            ),
        },
    }
    return arrays, metadata


def _array_contract(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def prepare_rctl_smoke(config: RctlSmokeConfig) -> dict[str, Any]:
    """검증된 원본 배열에서 작은 입력 bundle과 manifest를 원자적으로 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    forecast_config = load_forecast_config(config.forecast_config_path)
    if forecast_config.input_length != config.input_length:
        raise RctlContractError("forecast와 RCTL input_length가 다릅니다.")
    upc_config = load_upc_config(forecast_config.upc_config_path)
    arrays, processed_validation = verify_processed_inputs(upc_config)
    central_rows, central_positions, central_validation = load_central_cells(
        upc_config, arrays["cell_ids"]
    )
    index_contract: ForecastIndexContract = build_forecast_index_contract(
        arrays["timestamps_ms"],
        forecast_config,
        timezone_name=upc_config.timezone_name,
        interval_ms=upc_config.interval_ms,
    )
    bundle_arrays, sample_metadata = build_smoke_arrays(
        traffic=arrays["traffic"],
        missing_mask=arrays["missing_mask"],
        internet_null_mask=arrays["internet_null_mask"],
        cell_ids=arrays["cell_ids"],
        timestamps_ms=arrays["timestamps_ms"],
        central_rows=central_rows,
        central_positions=central_positions,
        train_target_indices=index_contract.target_indices["train"],
        input_length=config.input_length,
        selection=config.selection,
    )

    input_path = config.outputs.input_npz
    manifest_path = config.outputs.input_manifest
    input_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_input = _temporary_path(input_path)
    temporary_manifest = _temporary_path(manifest_path)
    published = False
    try:
        with temporary_input.open("wb") as handle:
            np.savez_compressed(handle, **bundle_arrays)
        bundle_metadata = {
            "path": _display_path(input_path),
            "size_bytes": temporary_input.stat().st_size,
            "sha256": compute_sha256(temporary_input),
            "arrays": {
                name: _array_contract(value) for name, value in bundle_arrays.items()
            },
        }
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "tool": {"name": "scripts.prepare_rctl_smoke", "version": TOOL_VERSION},
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "source_inputs": {
                "processed": processed_validation,
                "central_900": central_validation,
                "forecast_config": {
                    "path": _display_path(forecast_config.path),
                    "sha256": compute_sha256(forecast_config.path),
                },
                "upc_config": {
                    "path": _display_path(upc_config.path),
                    "sha256": compute_sha256(upc_config.path),
                },
            },
            "forecast_contract": {
                "input_length": forecast_config.input_length,
                "horizon": forecast_config.horizon,
                "split_assignment": "target local timestamp",
                "train": index_contract.split_metadata["train"],
                "future_leakage_check": "every input index is strictly less than target index",
            },
            "smoke": sample_metadata,
            "output": bundle_metadata,
            "runtime": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_input, input_path)
        os.replace(temporary_manifest, manifest_path)
        published = True
        return manifest
    finally:
        if not published:
            temporary_input.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RCTL Colab smoke용 16셀 실제 데이터 bundle을 준비합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"RCTL smoke config 경로 (기본값: {DEFAULT_CONFIG})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_rctl_smoke_config(args.config)
        manifest = prepare_rctl_smoke(config)
    except (RctlContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    smoke = manifest["smoke"]
    print(
        "RCTL smoke 입력 준비 완료: "
        f"{smoke['cell_count']}셀 x {smoke['windows_per_cell']} window = "
        f"{smoke['sample_count']}표본"
    )
    print(f"입력: {manifest['output']['path']}")
    print(f"SHA-256: {manifest['output']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
