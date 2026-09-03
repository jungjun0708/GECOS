#!/usr/bin/env python3
"""중앙 900셀 LSTM·UPC pipeline smoke용 결정적 입력 bundle을 만든다."""

from __future__ import annotations

import argparse
import csv
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
    build_forecast_index_contract,
    load_forecast_config,
)
from scripts.lstm_contract import (
    DEFAULT_CONFIG,
    LstmSmokeContractError,
    LstmUpcSmokeConfig,
    load_lstm_smoke_config,
)
from scripts.validate_upc_training_policy import (
    load_policy_config,
    require_training_allowed,
    run_training_policy,
)

TOOL_VERSION = "1.0.0"


def evenly_spaced_positions(item_count: int, selected_count: int) -> np.ndarray:
    """양 끝을 포함해 중복 없이 결정적인 위치를 선택한다."""

    if item_count < 1 or selected_count < 2 or selected_count > item_count:
        raise LstmSmokeContractError(
            "target 선택 수는 2 이상이고 전체 target 수 이하여야 합니다."
        )
    positions = np.rint(np.linspace(0, item_count - 1, selected_count)).astype(np.int64)
    if len(np.unique(positions)) != selected_count:
        raise LstmSmokeContractError("등간격 target 선택에 중복 위치가 생겼습니다.")
    if int(positions[0]) != 0 or int(positions[-1]) != item_count - 1:
        raise LstmSmokeContractError("등간격 target 선택은 양 끝을 포함해야 합니다.")
    return positions


def load_central_cluster_memberships(
    path: Path,
    *,
    expected_cell_ids: np.ndarray,
    protocol: str,
    expected_cluster_counts: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """PCC central membership CSV를 셀 순서와 cluster 수까지 검증한다."""

    cluster_field = f"{protocol}_cluster"
    required_fields = {"cell_id", cluster_field}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_fields.issubset(
                reader.fieldnames
            ):
                raise LstmSmokeContractError(
                    f"central membership CSV에 {sorted(required_fields)} 열이 없습니다."
                )
            rows = list(reader)
    except OSError as exc:
        raise LstmSmokeContractError(
            f"central membership CSV를 읽을 수 없습니다: {path}"
        ) from exc
    if len(rows) != len(expected_cell_ids):
        raise LstmSmokeContractError(
            "central membership 행 수가 중앙 셀 수와 다릅니다: "
            f"{len(rows)} != {len(expected_cell_ids)}"
        )
    try:
        row_cell_ids = np.asarray([int(row["cell_id"]) for row in rows], dtype=np.int32)
        memberships = np.asarray(
            [int(row[cluster_field]) for row in rows], dtype=np.int8
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LstmSmokeContractError(
            "central membership의 cell ID 또는 cluster를 정수로 읽을 수 없습니다."
        ) from exc
    if not np.array_equal(row_cell_ids, np.asarray(expected_cell_ids, dtype=np.int32)):
        raise LstmSmokeContractError(
            "central membership의 cell ID 순서가 중앙 traffic 순서와 다릅니다."
        )
    unique, counts = np.unique(memberships, return_counts=True)
    actual_counts = tuple(
        (int(cluster_id), int(count))
        for cluster_id, count in zip(unique, counts, strict=True)
    )
    if actual_counts != expected_cluster_counts:
        raise LstmSmokeContractError(
            f"central cluster 수가 사전 등록값과 다릅니다: {actual_counts}"
        )
    return memberships, {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
        "protocol": protocol,
        "row_count": len(rows),
        "cluster_counts": {str(key): value for key, value in actual_counts},
    }


def build_lstm_smoke_arrays(
    *,
    traffic: np.ndarray,
    missing_mask: np.ndarray,
    internet_null_mask: np.ndarray,
    cell_ids: np.ndarray,
    timestamps_ms: np.ndarray,
    central_positions: np.ndarray,
    memberships: np.ndarray,
    target_indices_by_split: Mapping[str, np.ndarray],
    input_length: int,
    targets_per_split: int,
    split_order: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """900셀과 각 split의 64 target을 cell-major 배열로 변환한다."""

    if traffic.ndim != 2:
        raise LstmSmokeContractError("traffic은 2차원이어야 합니다.")
    if any(
        value.shape != traffic.shape for value in (missing_mask, internet_null_mask)
    ):
        raise LstmSmokeContractError("traffic과 두 결측 mask shape가 다릅니다.")
    if len(cell_ids) != traffic.shape[0] or len(timestamps_ms) != traffic.shape[1]:
        raise LstmSmokeContractError("traffic 축과 cell/timestamp 축 길이가 다릅니다.")
    if central_positions.ndim != 1 or len(central_positions) != len(memberships):
        raise LstmSmokeContractError("중앙 셀 위치와 membership 길이가 다릅니다.")
    if len(np.unique(central_positions)) != len(central_positions):
        raise LstmSmokeContractError("중앙 셀 위치에 중복이 있습니다.")
    if np.any(central_positions < 0) or np.any(central_positions >= traffic.shape[0]):
        raise LstmSmokeContractError("중앙 셀 위치가 traffic 범위를 벗어납니다.")
    if tuple(target_indices_by_split) != split_order:
        raise LstmSmokeContractError("target split 순서가 config와 다릅니다.")

    central_traffic = np.asarray(traffic[central_positions, :], dtype=np.float32)
    central_missing = np.asarray(missing_mask[central_positions, :], dtype=bool)
    central_null = np.asarray(internet_null_mask[central_positions, :], dtype=bool)
    if not np.all(np.isfinite(central_traffic)) or np.any(central_traffic < 0):
        raise LstmSmokeContractError("중앙 traffic에 NaN, 무한대 또는 음수가 있습니다.")
    if np.any(central_missing & central_null):
        raise LstmSmokeContractError("중앙 셀의 두 결측 mask가 겹칩니다.")

    arrays: dict[str, np.ndarray] = {
        "cell_ids": np.asarray(cell_ids[central_positions], dtype=np.int32),
        "memberships": np.asarray(memberships, dtype=np.int8),
    }
    split_metadata: dict[str, Any] = {}
    offsets = np.arange(-input_length, 0, dtype=np.int64)
    for split_name in split_order:
        available_targets = np.asarray(
            target_indices_by_split[split_name], dtype=np.int64
        )
        positions = evenly_spaced_positions(len(available_targets), targets_per_split)
        selected_targets = np.asarray(available_targets[positions], dtype=np.int64)
        if int(selected_targets[0]) < input_length:
            raise LstmSmokeContractError(
                f"{split_name} 첫 target 앞에 입력 window가 부족합니다."
            )
        window_indices = selected_targets[:, None] + offsets[None, :]
        if np.any(window_indices >= selected_targets[:, None]) or np.any(
            window_indices < 0
        ):
            raise LstmSmokeContractError(
                f"{split_name} 입력 window에 target·미래값 또는 음수 index가 있습니다."
            )

        x = np.asarray(central_traffic[:, window_indices], dtype=np.float32)[..., None]
        y = np.asarray(central_traffic[:, selected_targets], dtype=np.float32)[
            ..., None
        ]
        persistence = np.asarray(
            central_traffic[:, selected_targets - 1], dtype=np.float32
        )[..., None]
        target_missing = np.asarray(central_missing[:, selected_targets], dtype=bool)
        target_null = np.asarray(central_null[:, selected_targets], dtype=bool)
        input_missing = np.asarray(central_missing[:, window_indices], dtype=bool)
        input_null = np.asarray(central_null[:, window_indices], dtype=bool)
        if x.shape != (len(central_positions), targets_per_split, input_length, 1):
            raise LstmSmokeContractError(f"{split_name} x shape가 계약과 다릅니다.")
        if y.shape != (len(central_positions), targets_per_split, 1):
            raise LstmSmokeContractError(f"{split_name} y shape가 계약과 다릅니다.")
        if not all(np.all(np.isfinite(value)) for value in (x, y, persistence)):
            raise LstmSmokeContractError(
                f"{split_name} 값에 NaN 또는 무한대가 있습니다."
            )

        arrays.update(
            {
                f"x_{split_name}": x,
                f"y_{split_name}": y,
                f"persistence_{split_name}": persistence,
                f"target_indices_{split_name}": selected_targets,
                f"target_timestamps_ms_{split_name}": np.asarray(
                    timestamps_ms[selected_targets], dtype=np.int64
                ),
                f"target_missing_mask_{split_name}": target_missing,
                f"target_internet_null_mask_{split_name}": target_null,
                f"input_missing_mask_{split_name}": input_missing,
                f"input_internet_null_mask_{split_name}": input_null,
            }
        )
        split_metadata[split_name] = {
            "available_target_count_per_cell": len(available_targets),
            "selected_target_count_per_cell": targets_per_split,
            "sample_count": len(central_positions) * targets_per_split,
            "target_position_indices_within_split": positions.tolist(),
            "target_indices": selected_targets.tolist(),
            "first_target_index": int(selected_targets[0]),
            "last_target_index": int(selected_targets[-1]),
            "first_input_index": int(window_indices[0, 0]),
            "last_input_index_for_first_target": int(window_indices[0, -1]),
            "target_missing_count": int(target_missing.sum()),
            "target_internet_null_count": int(target_null.sum()),
            "input_missing_count": int(input_missing.sum()),
            "input_internet_null_count": int(input_null.sum()),
            "traffic": {
                "x_min": float(x.min()),
                "x_max": float(x.max()),
                "y_min": float(y.min()),
                "y_max": float(y.max()),
                "persistence_mae": float(
                    np.mean(
                        np.abs(y.astype(np.float64) - persistence.astype(np.float64)),
                        dtype=np.float64,
                    )
                ),
            },
        }
    return arrays, {
        "cell_count": len(central_positions),
        "targets_per_cell_per_split": targets_per_split,
        "sample_count_per_split": len(central_positions) * targets_per_split,
        "input_length": input_length,
        "cluster_counts": {
            str(cluster_id): int((memberships == cluster_id).sum())
            for cluster_id in sorted(np.unique(memberships).tolist())
        },
        "splits": split_metadata,
        "future_leakage_check": "every input index is strictly less than its target index",
    }


def _array_contract(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def prepare_lstm_upc_smoke(config: LstmUpcSmokeConfig) -> dict[str, Any]:
    """검증된 900셀·membership에서 작은 Colab 입력을 원자적으로 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    forecast_config = load_forecast_config(config.forecast_config_path)
    if forecast_config.input_length != config.architecture.input_length:
        raise LstmSmokeContractError("forecast와 LSTM input_length가 다릅니다.")
    upc_config = load_upc_config(forecast_config.upc_config_path)
    arrays, processed_validation = verify_processed_inputs(upc_config)
    central_rows, central_positions, central_validation = load_central_cells(
        upc_config, arrays["cell_ids"]
    )
    central_cell_ids = np.asarray(
        [int(row["cell_id"]) for row in central_rows], dtype=np.int32
    )
    if len(central_cell_ids) != config.selection.expected_central_cell_count:
        raise LstmSmokeContractError("중앙 셀 수가 smoke config와 다릅니다.")
    if not np.array_equal(arrays["cell_ids"][central_positions], central_cell_ids):
        raise LstmSmokeContractError("중앙 셀 ID와 traffic 행 순서가 다릅니다.")

    policy_config = load_policy_config(
        config.policy_config_path, base_directory=config.base_directory
    )
    policy = run_training_policy(policy_config)
    require_training_allowed(policy, config.upc_protocol)
    policy_output_path = policy_config.outputs.policy
    if not policy_output_path.is_file():
        raise LstmSmokeContractError(
            "검증된 training_policy.json이 생성되지 않았습니다."
        )
    protected_membership = policy["evidence"]["protected_outputs"][
        "central_900_memberships_csv"
    ]
    membership_path = Path(str(protected_membership["path"]))
    if not membership_path.is_absolute():
        membership_path = config.base_directory / membership_path
    membership_path = membership_path.resolve()
    if compute_sha256(membership_path) != protected_membership["sha256"]:
        raise LstmSmokeContractError(
            "training policy가 보호한 membership checksum이 다릅니다."
        )
    memberships, membership_validation = load_central_cluster_memberships(
        membership_path,
        expected_cell_ids=central_cell_ids,
        protocol=config.upc_protocol,
        expected_cluster_counts=config.selection.expected_cluster_counts,
    )

    index_contract = build_forecast_index_contract(
        arrays["timestamps_ms"],
        forecast_config,
        timezone_name=upc_config.timezone_name,
        interval_ms=upc_config.interval_ms,
    )
    selected_target_indices = {
        split_name: index_contract.target_indices[split_name]
        for split_name in config.selection.splits
    }
    bundle_arrays, smoke_metadata = build_lstm_smoke_arrays(
        traffic=arrays["traffic"],
        missing_mask=arrays["missing_mask"],
        internet_null_mask=arrays["internet_null_mask"],
        cell_ids=arrays["cell_ids"],
        timestamps_ms=arrays["timestamps_ms"],
        central_positions=central_positions,
        memberships=memberships,
        target_indices_by_split=selected_target_indices,
        input_length=config.architecture.input_length,
        targets_per_split=config.selection.targets_per_cell_per_split,
        split_order=config.selection.splits,
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
            "tool": {
                "name": "scripts.prepare_lstm_upc_smoke",
                "version": TOOL_VERSION,
            },
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
                "upc_training_policy_config": {
                    "path": _display_path(policy_config.path),
                    "sha256": compute_sha256(policy_config.path),
                },
                "upc_training_policy": {
                    "path": _display_path(policy_output_path),
                    "sha256": compute_sha256(policy_output_path),
                    "decision_stage": policy["decision_stage"],
                    "protocol": config.upc_protocol,
                    "model_training_allowed": policy["training_gates"][
                        config.upc_protocol
                    ]["model_training_allowed"],
                },
                "central_membership": membership_validation,
            },
            "forecast_contract": {
                "input_length": forecast_config.input_length,
                "horizon": forecast_config.horizon,
                "evaluation_mode": forecast_config.evaluation_mode,
                "full_split_metadata": {
                    name: index_contract.split_metadata[name]
                    for name in config.selection.splits
                },
            },
            "smoke": smoke_metadata,
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
        description="중앙 900셀 LSTM·UPC Colab smoke 입력을 준비합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_smoke_config(args.config)
        manifest = prepare_lstm_upc_smoke(config)
    except (LstmSmokeContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    smoke = manifest["smoke"]
    print(
        "LSTM·UPC smoke 입력 준비 완료: "
        f"{smoke['cell_count']}셀 x split당 "
        f"{smoke['targets_per_cell_per_split']} target = "
        f"{smoke['sample_count_per_split']}표본"
    )
    print(f"입력: {manifest['output']['path']}")
    print(f"SHA-256: {manifest['output']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
