#!/usr/bin/env python3
"""Train-only 셀별 Min-Max LSTM pilot의 결정적 입력 bundle을 만든다."""

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
)
from scripts.lstm_scaling_contract import (
    DEFAULT_CONFIG,
    LstmScalingContractError,
    LstmScalingPilotConfig,
    load_lstm_scaling_config,
)

TOOL_VERSION = "1.0.0"
COPIED_FIELDS = (
    "target_indices_{split}",
    "target_timestamps_ms_{split}",
    "target_missing_mask_{split}",
    "target_internet_null_mask_{split}",
    "input_missing_mask_{split}",
    "input_internet_null_mask_{split}",
)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LstmScalingContractError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LstmScalingContractError(
            f"{label}이 올바른 JSON이 아닙니다: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LstmScalingContractError(f"{label}의 최상위 값은 object여야 합니다.")
    return value


def _cell_parameters(
    values: np.ndarray, cell_minimum: np.ndarray, cell_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float32)
    minimum = np.asarray(cell_minimum, dtype=np.float32)
    scale = np.asarray(cell_range, dtype=np.float32)
    if (
        array.ndim < 2
        or minimum.shape != (array.shape[0],)
        or scale.shape != minimum.shape
    ):
        raise LstmScalingContractError("셀별 scaling 배열의 cell 축 shape가 다릅니다.")
    if not all(np.all(np.isfinite(value)) for value in (array, minimum, scale)):
        raise LstmScalingContractError(
            "셀별 scaling 입력에 NaN 또는 무한대가 있습니다."
        )
    if np.any(scale <= 0):
        raise LstmScalingContractError("셀별 scaling range는 모두 양수여야 합니다.")
    shape = (array.shape[0],) + (1,) * (array.ndim - 1)
    return array, minimum.reshape(shape), scale.reshape(shape)


def fit_per_cell_minmax(train_traffic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """오직 전달된 Train 시계열에서 셀별 최솟값과 범위를 적합한다."""

    values = np.asarray(train_traffic, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1:
        raise LstmScalingContractError(
            "scaler 적합용 Train traffic은 2차원이어야 합니다."
        )
    if not np.all(np.isfinite(values)):
        raise LstmScalingContractError(
            "scaler 적합용 Train traffic에 비유한값이 있습니다."
        )
    minimum = np.asarray(values.min(axis=1), dtype=np.float32)
    maximum = np.asarray(values.max(axis=1), dtype=np.float32)
    cell_range = np.asarray(maximum - minimum, dtype=np.float32)
    if np.any(cell_range <= 0):
        count = int((cell_range <= 0).sum())
        raise LstmScalingContractError(f"range가 0 이하인 셀이 있습니다: {count}")
    return minimum, cell_range


def transform_cellwise(
    values: np.ndarray, cell_minimum: np.ndarray, cell_range: np.ndarray
) -> np.ndarray:
    """클리핑 없이 셀별 Min-Max 변환한다."""

    array, minimum, scale = _cell_parameters(values, cell_minimum, cell_range)
    return np.asarray((array - minimum) / scale, dtype=np.float32)


def inverse_transform_cellwise(
    values: np.ndarray, cell_minimum: np.ndarray, cell_range: np.ndarray
) -> np.ndarray:
    """클리핑 없이 셀별 Min-Max 역변환한다."""

    array, minimum, scale = _cell_parameters(values, cell_minimum, cell_range)
    return np.asarray(array * scale + minimum, dtype=np.float32)


def _content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _array_contract(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": _content_sha256(array),
    }


def _base_array_names(config: LstmScalingPilotConfig) -> set[str]:
    names = {"cell_ids", "memberships"}
    for split in config.selection.splits:
        names.update({f"x_{split}", f"y_{split}", f"persistence_{split}"})
        names.update(template.format(split=split) for template in COPIED_FIELDS)
    return names


def _load_base_arrays(
    config: LstmScalingPilotConfig,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    base = config.base_smoke
    reference = config.base_reference
    if compute_sha256(base.outputs.input_npz) != reference.input_npz_sha256:
        raise LstmScalingContractError("기준 raw smoke input NPZ checksum이 다릅니다.")
    manifest = _load_json(base.outputs.input_manifest, "기준 raw smoke input manifest")
    if manifest.get("status") != "complete":
        raise LstmScalingContractError(
            "기준 raw smoke input manifest가 complete가 아닙니다."
        )
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("sha256") != reference.input_npz_sha256
    ):
        raise LstmScalingContractError("기준 input manifest의 NPZ checksum이 다릅니다.")
    contracts = output.get("arrays")
    if not isinstance(contracts, dict):
        raise LstmScalingContractError("기준 input manifest에 배열 계약이 없습니다.")
    required_names = _base_array_names(config)
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(base.outputs.input_npz, allow_pickle=False) as archive:
            if not required_names.issubset(archive.files):
                raise LstmScalingContractError(
                    "기준 input NPZ에 필요한 배열이 없습니다."
                )
            arrays = {
                name: np.asarray(archive[name]) for name in sorted(required_names)
            }
    except (OSError, ValueError) as exc:
        raise LstmScalingContractError("기준 input NPZ를 읽을 수 없습니다.") from exc
    for name, array in arrays.items():
        contract = contracts.get(name)
        if (
            not isinstance(contract, dict)
            or contract.get("shape") != list(array.shape)
            or contract.get("dtype") != str(array.dtype)
            or contract.get("content_sha256") != _content_sha256(array)
        ):
            raise LstmScalingContractError(f"기준 input {name} 배열 계약이 다릅니다.")
    return arrays, manifest


def _verify_raw_reference(config: LstmScalingPilotConfig) -> Mapping[str, Any]:
    reference = config.base_reference
    if (
        compute_sha256(reference.evaluation_report_path)
        != reference.evaluation_report_sha256
    ):
        raise LstmScalingContractError(
            "기준 raw evaluation report checksum이 다릅니다."
        )
    report = _load_json(reference.evaluation_report_path, "기준 raw evaluation report")
    if report.get("status") != reference.required_status:
        raise LstmScalingContractError("기준 raw evaluation report status가 다릅니다.")
    results = report.get("results")
    if not isinstance(results, list):
        raise LstmScalingContractError("기준 raw evaluation results가 없습니다.")
    found: dict[str, tuple[float, float]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        if (
            row.get("split") == config.raw_reference.split
            and row.get("target_policy") == config.raw_reference.target_policy
        ):
            micro = row.get("micro")
            if isinstance(micro, dict):
                found[str(row.get("baseline"))] = (micro.get("mae"), micro.get("wape"))
    for metric in config.raw_reference.metrics:
        if found.get(metric.model) != (metric.mae, metric.wape):
            raise LstmScalingContractError(
                f"기준 raw metric이 사전 등록값과 다릅니다: {metric.model}"
            )
    return report


def _load_scaler_source(
    config: LstmScalingPilotConfig, cell_ids: np.ndarray
) -> tuple[np.ndarray, Mapping[str, Any]]:
    source = config.scaler_source
    if (
        compute_sha256(source.central_manifest_path)
        != source.expected_central_manifest_sha256
    ):
        raise LstmScalingContractError("central manifest checksum이 다릅니다.")
    manifest = _load_json(source.central_manifest_path, "central manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(
        outputs.get("central_traffic"), dict
    ):
        raise LstmScalingContractError("central manifest에 traffic 계약이 없습니다.")
    traffic_contract = outputs["central_traffic"]
    if traffic_contract.get("sha256") != source.expected_central_traffic_sha256:
        raise LstmScalingContractError(
            "central manifest의 traffic checksum이 다릅니다."
        )
    if (
        compute_sha256(source.central_traffic_path)
        != source.expected_central_traffic_sha256
    ):
        raise LstmScalingContractError("central traffic checksum이 다릅니다.")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise LstmScalingContractError("central manifest에 selection이 없습니다.")
    if selection.get("cell_ids_int32_sha256") != _content_sha256(
        np.asarray(cell_ids, dtype=np.int32)
    ):
        raise LstmScalingContractError(
            "central traffic과 smoke cell 순서를 확인할 수 없습니다."
        )
    try:
        traffic = np.load(
            source.central_traffic_path, mmap_mode="r", allow_pickle=False
        )
    except (OSError, ValueError) as exc:
        raise LstmScalingContractError("central traffic을 읽을 수 없습니다.") from exc
    expected_shape = (config.selection.expected_central_cell_count, 4320)
    if traffic.shape != expected_shape or traffic.dtype != np.float32:
        raise LstmScalingContractError("central traffic shape/dtype이 다릅니다.")
    if not np.all(np.isfinite(traffic)) or np.any(traffic < 0):
        raise LstmScalingContractError(
            "central traffic에 비유한값 또는 음수가 있습니다."
        )
    return traffic, manifest


def _roundtrip_error(
    raw: np.ndarray, scaled: np.ndarray, minimum: np.ndarray, cell_range: np.ndarray
) -> float:
    restored = inverse_transform_cellwise(scaled, minimum, cell_range)
    return float(np.max(np.abs(restored.astype(np.float64) - raw.astype(np.float64))))


def _split_statistics(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    scaled_x: np.ndarray,
    scaled_y: np.ndarray,
    minimum: np.ndarray,
    cell_range: np.ndarray,
) -> dict[str, Any]:
    return {
        "sample_count": int(raw_y.shape[0] * raw_y.shape[1]),
        "raw": {
            "x_min": float(raw_x.min()),
            "x_max": float(raw_x.max()),
            "y_min": float(raw_y.min()),
            "y_max": float(raw_y.max()),
        },
        "scaled": {
            "x_min": float(scaled_x.min()),
            "x_max": float(scaled_x.max()),
            "y_min": float(scaled_y.min()),
            "y_max": float(scaled_y.max()),
            "x_below_zero_count": int((scaled_x < 0).sum()),
            "x_above_one_count": int((scaled_x > 1).sum()),
            "y_below_zero_count": int((scaled_y < 0).sum()),
            "y_above_one_count": int((scaled_y > 1).sum()),
        },
        "roundtrip_max_absolute_error": {
            "x": _roundtrip_error(raw_x, scaled_x, minimum, cell_range),
            "y": _roundtrip_error(raw_y, scaled_y, minimum, cell_range),
        },
    }


def prepare_lstm_scaling_pilot(config: LstmScalingPilotConfig) -> dict[str, Any]:
    """Train-only scaler를 적합하고 Test 없는 pilot 입력을 원자적으로 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    base_arrays, base_manifest = _load_base_arrays(config)
    _verify_raw_reference(config)
    cell_ids = np.asarray(base_arrays["cell_ids"], dtype=np.int32)
    central_traffic, central_manifest = _load_scaler_source(config, cell_ids)
    scaling = config.scaling
    fit_slice = central_traffic[
        :, scaling.fit_start_index_inclusive : scaling.fit_end_index_exclusive
    ]
    minimum, cell_range = fit_per_cell_minmax(fit_slice)
    zero_count = int((cell_range <= 0).sum())
    if zero_count != scaling.expected_zero_range_cell_count:
        raise LstmScalingContractError("zero-range 셀 수가 사전 등록값과 다릅니다.")

    bundle_arrays: dict[str, np.ndarray] = {
        "cell_ids": cell_ids,
        "memberships": np.asarray(base_arrays["memberships"], dtype=np.int8),
        "scaler_min": minimum,
        "scaler_range": cell_range,
    }
    split_statistics: dict[str, Any] = {}
    for split in config.selection.splits:
        raw_x = np.asarray(base_arrays[f"x_{split}"], dtype=np.float32)
        raw_y = np.asarray(base_arrays[f"y_{split}"], dtype=np.float32)
        scaled_x = transform_cellwise(raw_x, minimum, cell_range)
        scaled_y = transform_cellwise(raw_y, minimum, cell_range)
        bundle_arrays.update(
            {
                f"x_{split}": scaled_x,
                f"y_{split}": scaled_y,
                f"raw_y_{split}": raw_y,
                f"raw_persistence_{split}": np.asarray(
                    base_arrays[f"persistence_{split}"], dtype=np.float32
                ),
            }
        )
        for template in COPIED_FIELDS:
            name = template.format(split=split)
            bundle_arrays[name] = np.asarray(base_arrays[name])
        split_statistics[split] = _split_statistics(
            raw_x, raw_y, scaled_x, scaled_y, minimum, cell_range
        )

    train_scaled = [bundle_arrays["x_train"], bundle_arrays["y_train"]]
    if (
        min(float(value.min()) for value in train_scaled) < -1e-6
        or max(float(value.max()) for value in train_scaled) > 1.0 + 1e-6
    ):
        raise LstmScalingContractError("Train 변환값이 [0, 1] 범위를 벗어났습니다.")
    roundtrip_errors = [
        value
        for row in split_statistics.values()
        for value in row["roundtrip_max_absolute_error"].values()
    ]
    if max(roundtrip_errors) > scaling.roundtrip_max_absolute_error:
        raise LstmScalingContractError("scaling 역변환 오차 허용치를 초과했습니다.")
    if any("test" in name.lower() for name in bundle_arrays):
        raise LstmScalingContractError("pilot bundle에 Test 배열이 포함됐습니다.")
    if not all(np.all(np.isfinite(value)) for value in bundle_arrays.values()):
        raise LstmScalingContractError("pilot bundle에 NaN 또는 무한대가 있습니다.")

    input_path = config.outputs.input_npz
    manifest_path = config.outputs.input_manifest
    input_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_input = _temporary_path(input_path)
    temporary_manifest = _temporary_path(manifest_path)
    published = False
    try:
        with temporary_input.open("wb") as handle:
            np.savez_compressed(handle, **bundle_arrays)
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "tool": {
                "name": "scripts.prepare_lstm_scaling_pilot",
                "version": TOOL_VERSION,
            },
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "base_smoke": {
                "config_sha256": config.base_reference.config_sha256,
                "input_npz_sha256": config.base_reference.input_npz_sha256,
                "input_manifest_sha256": compute_sha256(
                    config.base_smoke.outputs.input_manifest
                ),
                "evaluation_report_sha256": config.base_reference.evaluation_report_sha256,
                "source_git": base_manifest.get("git"),
            },
            "scaler_source": {
                "central_manifest": {
                    "path": _display_path(config.scaler_source.central_manifest_path),
                    "sha256": config.scaler_source.expected_central_manifest_sha256,
                },
                "central_traffic": {
                    "path": _display_path(config.scaler_source.central_traffic_path),
                    "sha256": config.scaler_source.expected_central_traffic_sha256,
                    "shape": list(central_traffic.shape),
                    "dtype": str(central_traffic.dtype),
                },
                "central_protocol": central_manifest.get("protocol"),
            },
            "scaling": {
                "name": scaling.name,
                "fit_partition": scaling.fit_partition,
                "fit_indices": {
                    "start_inclusive": scaling.fit_start_index_inclusive,
                    "end_exclusive": scaling.fit_end_index_exclusive,
                    "count": scaling.fit_end_index_exclusive
                    - scaling.fit_start_index_inclusive,
                },
                "fit_used_validation": False,
                "fit_used_test": False,
                "clip_transform": scaling.clip_transform,
                "clip_inverse_prediction": scaling.clip_inverse_prediction,
                "zero_range_cell_count": zero_count,
                "cell_min_quantiles": np.quantile(
                    minimum, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                "cell_range_quantiles": np.quantile(
                    cell_range, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                "split_statistics": split_statistics,
                "roundtrip_gate": {
                    "maximum_observed": max(roundtrip_errors),
                    "maximum_allowed": scaling.roundtrip_max_absolute_error,
                    "passed": True,
                },
            },
            "selection": {
                "splits": list(config.selection.splits),
                "test_policy": config.test_policy,
                "test_arrays_in_bundle": False,
                "cell_count": len(cell_ids),
                "targets_per_cell_per_split": config.selection.targets_per_cell_per_split,
                "cluster_counts": {
                    str(key): int((bundle_arrays["memberships"] == key).sum())
                    for key, _ in config.selection.expected_cluster_counts
                },
            },
            "raw_validation_reference": {
                metric.model: {"mae": metric.mae, "wape": metric.wape}
                for metric in config.raw_reference.metrics
            },
            "output": {
                "path": _display_path(input_path),
                "size_bytes": temporary_input.stat().st_size,
                "sha256": compute_sha256(temporary_input),
                "arrays": {
                    name: _array_contract(value)
                    for name, value in bundle_arrays.items()
                },
            },
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
        description="LSTM Train-only scaling pilot 입력을 준비합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_scaling_config(args.config)
        manifest = prepare_lstm_scaling_pilot(config)
    except (LstmScalingContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("LSTM scaling pilot 입력 준비 완료 (Train/Validation only)")
    print(f"입력: {manifest['output']['path']}")
    print(f"SHA-256: {manifest['output']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "fit_per_cell_minmax",
    "inverse_transform_cellwise",
    "prepare_lstm_scaling_pilot",
    "transform_cellwise",
]
