#!/usr/bin/env python3
"""Test 없는 중앙 900셀 LSTM 전체 Train·Validation 입력을 준비한다."""

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
)
from scripts.forecast_contract import (
    build_forecast_index_contract,
    load_forecast_config,
)
from scripts.lstm_full_contract import (
    DEFAULT_CONFIG,
    FullJobSpec,
    LstmFullContractError,
    LstmFullTrainingConfig,
    load_lstm_full_config,
)
from scripts.prepare_lstm_scaling_pilot import (
    fit_per_cell_minmax,
    inverse_transform_cellwise,
    transform_cellwise,
)
from scripts.prepare_lstm_upc_smoke import load_central_cluster_memberships
from scripts.validate_upc_training_policy import require_training_allowed

TOOL_VERSION = "1.0.0"
BUNDLE_ARRAY_NAMES = (
    "cell_ids",
    "memberships",
    "traffic_train_validation",
    "missing_mask_train_validation",
    "internet_null_mask_train_validation",
    "timestamps_ms_train_validation",
    "scaler_min",
    "scaler_range",
    "target_indices_train",
    "target_indices_validation",
)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LstmFullContractError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LstmFullContractError(f"{label}이 올바른 JSON이 아닙니다: {exc}") from exc
    if not isinstance(value, dict):
        raise LstmFullContractError(f"{label}의 최상위 값은 object여야 합니다.")
    return value


def _content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _array_contract(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_sha256": _content_sha256(array),
    }


def _file_contract(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": compute_sha256(path),
    }


def _verify_sources(config: LstmFullTrainingConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in config.sources:
        if not source.path.is_file():
            raise LstmFullContractError(f"필수 source가 없습니다: {source.name}")
        actual = compute_sha256(source.path)
        if actual != source.sha256:
            raise LstmFullContractError(
                f"{source.name} checksum이 config와 다릅니다: {actual}"
            )
        metadata[source.name] = _file_contract(source.path)

    scaling_report = _load_json(
        config.source("scaling_pilot_evaluation").path,
        "scaling pilot evaluation",
    )
    decision = scaling_report.get("decision")
    if (
        scaling_report.get("status") != "pass"
        or not isinstance(decision, dict)
        or decision.get("outcome") != config.required_scaling_outcome
        or decision.get("test_used") is not False
    ):
        raise LstmFullContractError("scaling pilot 채택 근거가 config와 다릅니다.")

    policy = _load_json(config.source("upc_training_policy").path, "UPC policy")
    require_training_allowed(policy, config.upc.protocol)
    metadata["scaling_pilot_decision"] = {
        "status": scaling_report["status"],
        "outcome": decision["outcome"],
        "test_used": decision["test_used"],
    }
    metadata["upc_policy_decision"] = {
        "protocol": config.upc.protocol,
        "model_training_allowed": True,
    }
    return metadata


def _load_central_cell_ids(
    central_manifest: Mapping[str, Any], config: LstmFullTrainingConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    outputs = central_manifest.get("outputs")
    selection = central_manifest.get("selection")
    if not isinstance(outputs, dict) or not isinstance(selection, dict):
        raise LstmFullContractError(
            "central manifest에 output/selection 계약이 없습니다."
        )
    cells_contract = outputs.get("central_cells_csv")
    if not isinstance(cells_contract, dict):
        raise LstmFullContractError("central manifest에 central cell CSV가 없습니다.")
    cells_path = Path(str(cells_contract.get("path", "")))
    if not cells_path.is_absolute():
        cells_path = config.base_directory / cells_path
    cells_path = cells_path.resolve()
    if compute_sha256(cells_path) != cells_contract.get("sha256"):
        raise LstmFullContractError("central cell CSV checksum이 다릅니다.")
    try:
        with cells_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        cell_ids = np.asarray([int(row["cell_id"]) for row in rows], dtype=np.int32)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise LstmFullContractError("central cell ID를 읽을 수 없습니다.") from exc
    if cell_ids.shape != (config.data.expected_cell_count,):
        raise LstmFullContractError("central cell 수가 config와 다릅니다.")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise LstmFullContractError("central cell ID에 중복이 있습니다.")
    if _content_sha256(cell_ids) != selection.get("cell_ids_int32_sha256"):
        raise LstmFullContractError("central cell ID 순서 checksum이 다릅니다.")
    return cell_ids, _file_contract(cells_path)


def _load_source_arrays(
    config: LstmFullTrainingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    central_manifest = _load_json(
        config.source("central_manifest").path, "central manifest"
    )
    cell_ids, cells_metadata = _load_central_cell_ids(central_manifest, config)
    arrays: dict[str, np.ndarray] = {"cell_ids": cell_ids}
    source_map = {
        "traffic": "central_traffic",
        "missing_mask": "central_missing_mask",
        "internet_null_mask": "central_internet_null_mask",
        "timestamps_ms": "timestamps_ms",
    }
    for output_name, source_name in source_map.items():
        try:
            arrays[output_name] = np.load(
                config.source(source_name).path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError) as exc:
            raise LstmFullContractError(
                f"{source_name} 배열을 읽을 수 없습니다."
            ) from exc
    expected_matrix_shape = (config.data.expected_cell_count, 4320)
    if (
        arrays["traffic"].shape != expected_matrix_shape
        or arrays["traffic"].dtype != np.float32
    ):
        raise LstmFullContractError("central traffic shape/dtype이 다릅니다.")
    for name in ("missing_mask", "internet_null_mask"):
        if (
            arrays[name].shape != expected_matrix_shape
            or arrays[name].dtype != np.bool_
        ):
            raise LstmFullContractError(f"{name} shape/dtype이 다릅니다.")
    if (
        arrays["timestamps_ms"].shape != (4320,)
        or arrays["timestamps_ms"].dtype != np.int64
    ):
        raise LstmFullContractError("timestamp shape/dtype이 다릅니다.")
    if not np.all(np.isfinite(arrays["traffic"])) or np.any(arrays["traffic"] < 0):
        raise LstmFullContractError("central traffic에 비유한값 또는 음수가 있습니다.")
    if np.any(arrays["missing_mask"] & arrays["internet_null_mask"]):
        raise LstmFullContractError("두 결측 mask가 겹칩니다.")
    memberships, membership_metadata = load_central_cluster_memberships(
        config.source("central_memberships").path,
        expected_cell_ids=cell_ids,
        protocol=config.upc.protocol,
        expected_cluster_counts=config.upc.expected_cluster_counts,
    )
    arrays["memberships"] = memberships
    return arrays, {
        "central_cells_csv": cells_metadata,
        "central_memberships": membership_metadata,
    }


def build_compact_train_validation_arrays(
    *,
    config: LstmFullTrainingConfig,
    traffic: np.ndarray,
    missing_mask: np.ndarray,
    internet_null_mask: np.ndarray,
    timestamps_ms: np.ndarray,
    cell_ids: np.ndarray,
    memberships: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """전역 index 3,600 미만의 원시 시계열과 Train-only scaler만 묶는다."""

    start = config.data.bundle_global_start_index_inclusive
    end = config.data.bundle_global_end_index_exclusive
    if start != 0 or end != config.data.test_target_start_index_inclusive:
        raise LstmFullContractError("compact bundle 경계가 Test 시작과 맞지 않습니다.")
    minimum, cell_range = fit_per_cell_minmax(
        traffic[
            :,
            config.scaling.fit_start_index_inclusive : config.scaling.fit_end_index_exclusive,
        ]
    )
    if int((cell_range <= 0).sum()) != config.scaling.expected_zero_range_cell_count:
        raise LstmFullContractError("zero-range 셀 수가 config와 다릅니다.")
    traffic_slice = np.asarray(traffic[:, start:end], dtype=np.float32)
    missing_slice = np.asarray(missing_mask[:, start:end], dtype=bool)
    null_slice = np.asarray(internet_null_mask[:, start:end], dtype=bool)
    timestamp_slice = np.asarray(timestamps_ms[start:end], dtype=np.int64)
    target_indices = {
        split.name: np.arange(
            split.target_start_index_inclusive,
            split.target_end_index_exclusive,
            dtype=np.int64,
        )
        for split in config.data.splits
    }
    arrays = {
        "cell_ids": np.asarray(cell_ids, dtype=np.int32),
        "memberships": np.asarray(memberships, dtype=np.int8),
        "traffic_train_validation": traffic_slice,
        "missing_mask_train_validation": missing_slice,
        "internet_null_mask_train_validation": null_slice,
        "timestamps_ms_train_validation": timestamp_slice,
        "scaler_min": minimum,
        "scaler_range": cell_range,
        "target_indices_train": target_indices["train"],
        "target_indices_validation": target_indices["validation"],
    }
    if tuple(arrays) != BUNDLE_ARRAY_NAMES:
        raise LstmFullContractError("compact bundle 배열 순서가 계약과 다릅니다.")
    if any("test" in name.lower() for name in arrays):
        raise LstmFullContractError("compact bundle 이름에 Test가 포함됐습니다.")
    if max(int(value.max()) for value in target_indices.values()) >= end:
        raise LstmFullContractError("target index가 Test 경계에 도달했습니다.")
    if int(timestamp_slice[-1]) != int(timestamps_ms[end - 1]):
        raise LstmFullContractError(
            "compact timestamp 끝이 Validation 경계와 다릅니다."
        )
    scaled = transform_cellwise(traffic_slice, minimum, cell_range)
    restored = inverse_transform_cellwise(scaled, minimum, cell_range)
    roundtrip_error = float(
        np.max(np.abs(restored.astype(np.float64) - traffic_slice.astype(np.float64)))
    )
    train_scaled = scaled[:, : config.scaling.fit_end_index_exclusive]
    validation = config.data.split("validation")
    validation_scaled = scaled[
        :,
        validation.target_start_index_inclusive : validation.target_end_index_exclusive,
    ]
    if float(train_scaled.min()) < -1e-6 or float(train_scaled.max()) > 1.0 + 1e-6:
        raise LstmFullContractError(
            "전체 Train scaled traffic이 [0, 1]을 벗어났습니다."
        )
    if roundtrip_error > config.scaling.roundtrip_max_absolute_error:
        raise LstmFullContractError(
            "compact traffic 역변환 오차가 허용치를 넘었습니다."
        )
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise LstmFullContractError("compact bundle에 NaN 또는 무한대가 있습니다.")
    metadata = {
        "global_index_range": {
            "start_inclusive": start,
            "end_exclusive": end,
        },
        "test_start_index": config.data.test_target_start_index_inclusive,
        "test_arrays_present": False,
        "traffic_shape": list(traffic_slice.shape),
        "traffic_dtype": str(traffic_slice.dtype),
        "last_timestamp_ms": int(timestamp_slice[-1]),
        "scaling": {
            "fit_start_index_inclusive": config.scaling.fit_start_index_inclusive,
            "fit_end_index_exclusive": config.scaling.fit_end_index_exclusive,
            "fit_used_validation": False,
            "fit_used_test": False,
            "zero_range_cell_count": int((cell_range <= 0).sum()),
            "train_scaled_min": float(train_scaled.min()),
            "train_scaled_max": float(train_scaled.max()),
            "validation_scaled_below_zero_count": int((validation_scaled < 0).sum()),
            "validation_scaled_above_one_count": int((validation_scaled > 1).sum()),
            "roundtrip_max_absolute_error": roundtrip_error,
        },
        "splits": {
            split.name: {
                "target_count_per_cell": len(target_indices[split.name]),
                "sample_count": len(target_indices[split.name])
                * config.data.expected_cell_count,
                "first_target_index": int(target_indices[split.name][0]),
                "last_target_index": int(target_indices[split.name][-1]),
                "first_input_index": int(target_indices[split.name][0])
                - config.data.input_length,
                "last_input_index": int(target_indices[split.name][-1]) - 1,
            }
            for split in config.data.splits
        },
    }
    return arrays, metadata


def job_descriptor(
    *,
    config: LstmFullTrainingConfig,
    job: FullJobSpec,
    config_sha256: str,
    input_npz_sha256: str,
    source_git: Mapping[str, Any],
) -> dict[str, Any]:
    """한 seed·조건만 허용하는 결정적 Colab job descriptor를 만든다."""

    return {
        "schema_version": 1,
        "status": "ready",
        "job_id": job.job_id,
        "seed": job.seed,
        "condition": job.condition,
        "cluster_id": job.cluster_id,
        "expected_cell_count": job.expected_cell_count,
        "config_sha256": config_sha256,
        "input_npz_sha256": input_npz_sha256,
        "source_git": dict(source_git),
        "test_allowed": False,
        "output_relative_directory": (
            Path("data/processed/lstm_full_training/jobs") / job.job_id
        ).as_posix(),
    }


def prepare_lstm_full_training(config: LstmFullTrainingConfig) -> dict[str, Any]:
    """source를 검증하고 Test 없는 compact bundle·9개 descriptor를 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    source_metadata = _verify_sources(config)
    source_arrays, central_metadata = _load_source_arrays(config)
    forecast_config = load_forecast_config(config.source("forecast_config").path)
    index_contract = build_forecast_index_contract(
        source_arrays["timestamps_ms"],
        forecast_config,
        timezone_name="Europe/Rome",
        interval_ms=600_000,
    )
    for split in config.data.splits:
        if not np.array_equal(
            index_contract.target_indices[split.name],
            np.arange(
                split.target_start_index_inclusive,
                split.target_end_index_exclusive,
                dtype=np.int64,
            ),
        ):
            raise LstmFullContractError(f"{split.name} forecast index 계약이 다릅니다.")
    bundle_arrays, bundle_contract = build_compact_train_validation_arrays(
        config=config,
        traffic=source_arrays["traffic"],
        missing_mask=source_arrays["missing_mask"],
        internet_null_mask=source_arrays["internet_null_mask"],
        timestamps_ms=source_arrays["timestamps_ms"],
        cell_ids=source_arrays["cell_ids"],
        memberships=source_arrays["memberships"],
    )
    source_git = _git_state()
    if (
        config.pass_criteria.require_clean_source_git
        and source_git.get("dirty") is not False
    ):
        raise LstmFullContractError(
            "전체 학습 입력은 clean Git commit에서만 준비할 수 있습니다."
        )
    config_sha256 = compute_sha256(config.path)
    input_path = config.outputs.input_npz
    manifest_path = config.outputs.input_manifest
    descriptors_dir = config.outputs.job_descriptors_dir
    input_path.parent.mkdir(parents=True, exist_ok=True)
    descriptors_dir.mkdir(parents=True, exist_ok=True)
    temporary_input = _temporary_path(input_path)
    temporary_manifest = _temporary_path(manifest_path)
    descriptor_temporaries: dict[Path, Path] = {}
    published = False
    try:
        with temporary_input.open("wb") as handle:
            np.savez_compressed(handle, **bundle_arrays)
        input_npz_sha256 = compute_sha256(temporary_input)
        descriptor_metadata: dict[str, Any] = {}
        for job in config.jobs:
            final_path = descriptors_dir / f"{job.job_id}.json"
            temporary_path = _temporary_path(final_path)
            descriptor_temporaries[final_path] = temporary_path
            descriptor = job_descriptor(
                config=config,
                job=job,
                config_sha256=config_sha256,
                input_npz_sha256=input_npz_sha256,
                source_git=source_git,
            )
            temporary_path.write_text(
                json.dumps(descriptor, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            descriptor_metadata[job.job_id] = {
                "path": _display_path(final_path),
                "sha256": compute_sha256(temporary_path),
                "seed": job.seed,
                "condition": job.condition,
                "cluster_id": job.cluster_id,
                "expected_cell_count": job.expected_cell_count,
            }
        peak_rss_bytes = _peak_rss_bytes()
        if peak_rss_bytes > config.resources.local_peak_rss_limit_bytes:
            raise LstmFullContractError(
                "로컬 입력 준비 peak RSS가 사전 등록 상한을 초과했습니다: "
                f"{peak_rss_bytes} > {config.resources.local_peak_rss_limit_bytes}"
            )
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "tool": {
                "name": "scripts.prepare_lstm_full_training",
                "version": TOOL_VERSION,
            },
            "created_at_utc": finished_at.isoformat(),
            "git": source_git,
            "config": {
                "path": _display_path(config.path),
                "sha256": config_sha256,
            },
            "sources": {**source_metadata, **central_metadata},
            "forecast_contract": {
                "input_length": forecast_config.input_length,
                "horizon": forecast_config.horizon,
                "evaluation_mode": forecast_config.evaluation_mode,
                "splits": {
                    split.name: index_contract.split_metadata[split.name]
                    for split in config.data.splits
                },
            },
            "bundle_contract": bundle_contract,
            "jobs": {
                "expected_job_count": config.training.expected_job_count,
                "descriptors": descriptor_metadata,
            },
            "output": {
                "path": _display_path(input_path),
                "size_bytes": temporary_input.stat().st_size,
                "sha256": input_npz_sha256,
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
                "peak_rss_bytes": peak_rss_bytes,
                "local_peak_rss_limit_bytes": config.resources.local_peak_rss_limit_bytes,
                "local_tensorflow_training": False,
            },
            "test_seal": {
                "policy": config.data.test_policy,
                "known_prior_exposure": config.data.known_prior_test_exposure,
                "future_claim": config.data.future_test_claim,
                "test_arrays_present": False,
                "test_evaluated": False,
            },
        }
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_input, input_path)
        for final_path, temporary_path in descriptor_temporaries.items():
            os.replace(temporary_path, final_path)
        os.replace(temporary_manifest, manifest_path)
        published = True
        return manifest
    finally:
        if not published:
            temporary_input.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
            for temporary_path in descriptor_temporaries.values():
                temporary_path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test 없는 LSTM 전체 Train·Validation 입력을 준비합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_full_config(args.config)
        manifest = prepare_lstm_full_training(config)
    except (LstmFullContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("LSTM 전체 Train·Validation compact 입력 준비 완료")
    print(f"입력: {manifest['output']['path']}")
    print(f"SHA-256: {manifest['output']['sha256']}")
    print(f"job descriptor: {manifest['jobs']['expected_job_count']}개")
    print(f"Test 배열: {manifest['test_seal']['test_arrays_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_ARRAY_NAMES",
    "build_compact_train_validation_arrays",
    "job_descriptor",
    "prepare_lstm_full_training",
]
