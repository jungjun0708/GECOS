#!/usr/bin/env python3
"""Colab T4에서 LSTM 전체 학습의 immutable job 하나를 실행한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_upc_initial_groups import (
    _display_path,
    _peak_rss_bytes,
    _temporary_path,
    compute_sha256,
)
from scripts.lstm_full_contract import (
    DEFAULT_CONFIG,
    FullJobSpec,
    LstmFullContractError,
    LstmFullTrainingConfig,
    load_lstm_full_config,
)
from scripts.lstm_model import (
    audit_lstm_model,
    build_lstm_model,
    configure_determinism,
    require_tensorflow,
)
from scripts.prepare_lstm_full_training import BUNDLE_ARRAY_NAMES
from scripts.prepare_lstm_scaling_pilot import fit_per_cell_minmax
from scripts.run_lstm_upc_smoke import _runtime_environment, verified_source_git

TOOL_VERSION = "1.0.0"
JOB_OUTPUT_NAMES = (
    "training_report.json",
    "validation_predictions.npz",
    "best_weights.npz",
    "run_manifest.json",
)
PREDICTION_ARRAY_NAMES = (
    "central_positions",
    "cell_ids",
    "target_indices_validation",
    "raw_y_validation",
    "prediction_scaled_validation",
    "prediction_raw_validation",
    "target_missing_mask_validation",
    "target_internet_null_mask_validation",
    "lag_missing_mask_validation",
    "lag_internet_null_mask_validation",
    "scaler_min",
    "scaler_range",
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _temporary_file_contract(final_path: Path, temporary_path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(final_path),
        "size_bytes": temporary_path.stat().st_size,
        "sha256": compute_sha256(temporary_path),
    }


def _expected_job(config: LstmFullTrainingConfig, job_id: str) -> FullJobSpec:
    matches = [job for job in config.jobs if job.job_id == job_id]
    if len(matches) != 1:
        raise LstmFullContractError(f"등록되지 않았거나 중복된 job ID입니다: {job_id}")
    return matches[0]


def load_and_verify_full_bundle(
    config: LstmFullTrainingConfig,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """Test 없는 compact NPZ와 manifest의 provenance를 전부 검증한다."""

    manifest = _load_json(config.outputs.input_manifest, "full training input manifest")
    if manifest.get("status") != "complete":
        raise LstmFullContractError("input manifest가 complete 상태가 아닙니다.")
    verified_source_git(manifest)
    config_metadata = manifest.get("config")
    if not isinstance(config_metadata, dict) or config_metadata.get(
        "sha256"
    ) != compute_sha256(config.path):
        raise LstmFullContractError("full training config checksum이 다릅니다.")
    seal = manifest.get("test_seal")
    if (
        not isinstance(seal, dict)
        or seal.get("policy") != config.data.test_policy
        or seal.get("test_arrays_present") is not False
        or seal.get("test_evaluated") is not False
    ):
        raise LstmFullContractError("input manifest의 Test 봉인 계약이 다릅니다.")
    bundle = manifest.get("bundle_contract")
    if (
        not isinstance(bundle, dict)
        or bundle.get("test_start_index")
        != config.data.test_target_start_index_inclusive
        or bundle.get("test_arrays_present") is not False
        or bundle.get("global_index_range")
        != {
            "start_inclusive": config.data.bundle_global_start_index_inclusive,
            "end_exclusive": config.data.bundle_global_end_index_exclusive,
        }
    ):
        raise LstmFullContractError("compact bundle의 시간/Test 경계가 다릅니다.")

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise LstmFullContractError("input manifest.output이 없습니다.")
    if compute_sha256(config.outputs.input_npz) != output.get("sha256"):
        raise LstmFullContractError("compact input NPZ checksum이 다릅니다.")
    contracts = output.get("arrays")
    if not isinstance(contracts, dict):
        raise LstmFullContractError("compact input 배열 계약이 없습니다.")
    try:
        with np.load(config.outputs.input_npz, allow_pickle=False) as archive:
            if tuple(archive.files) != BUNDLE_ARRAY_NAMES:
                raise LstmFullContractError("compact input 배열 이름/순서가 다릅니다.")
            if any("test" in name.lower() for name in archive.files):
                raise LstmFullContractError("compact input에 Test 배열이 있습니다.")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise LstmFullContractError("compact input NPZ를 읽을 수 없습니다.") from exc
    for name, array in arrays.items():
        contract = contracts.get(name)
        if (
            not isinstance(contract, dict)
            or contract.get("shape") != list(array.shape)
            or contract.get("dtype") != str(array.dtype)
            or contract.get("content_sha256") != _content_sha256(array)
        ):
            raise LstmFullContractError(f"compact input {name} 배열 계약이 다릅니다.")

    cell_count = config.data.expected_cell_count
    end = config.data.bundle_global_end_index_exclusive
    matrix_shape = (cell_count, end)
    if (
        arrays["cell_ids"].shape != (cell_count,)
        or arrays["cell_ids"].dtype != np.int32
    ):
        raise LstmFullContractError("cell ID shape/dtype이 다릅니다.")
    if len(np.unique(arrays["cell_ids"])) != cell_count:
        raise LstmFullContractError("cell ID에 중복이 있습니다.")
    if (
        arrays["memberships"].shape != (cell_count,)
        or arrays["memberships"].dtype != np.int8
    ):
        raise LstmFullContractError("membership shape/dtype이 다릅니다.")
    actual_counts = tuple(
        (cluster_id, int((arrays["memberships"] == cluster_id).sum()))
        for cluster_id, _ in config.upc.expected_cluster_counts
    )
    if actual_counts != config.upc.expected_cluster_counts:
        raise LstmFullContractError("membership cluster 수가 config와 다릅니다.")
    if (
        arrays["traffic_train_validation"].shape != matrix_shape
        or arrays["traffic_train_validation"].dtype != np.float32
    ):
        raise LstmFullContractError("compact traffic shape/dtype이 다릅니다.")
    for name in (
        "missing_mask_train_validation",
        "internet_null_mask_train_validation",
    ):
        if arrays[name].shape != matrix_shape or arrays[name].dtype != np.bool_:
            raise LstmFullContractError(f"{name} shape/dtype이 다릅니다.")
    if (
        arrays["timestamps_ms_train_validation"].shape != (end,)
        or arrays["timestamps_ms_train_validation"].dtype != np.int64
    ):
        raise LstmFullContractError("compact timestamp shape/dtype이 다릅니다.")
    if not np.all(np.diff(arrays["timestamps_ms_train_validation"]) == 600_000):
        raise LstmFullContractError("compact timestamp가 정확한 10분 간격이 아닙니다.")

    for split in config.data.splits:
        expected_targets = np.arange(
            split.target_start_index_inclusive,
            split.target_end_index_exclusive,
            dtype=np.int64,
        )
        if not np.array_equal(arrays[f"target_indices_{split.name}"], expected_targets):
            raise LstmFullContractError(f"{split.name} target index가 다릅니다.")
        if int(expected_targets[-1]) >= config.data.test_target_start_index_inclusive:
            raise LstmFullContractError(f"{split.name} target이 Test에 닿았습니다.")
    minimum, cell_range = fit_per_cell_minmax(
        arrays["traffic_train_validation"][
            :,
            config.scaling.fit_start_index_inclusive : config.scaling.fit_end_index_exclusive,
        ]
    )
    if not np.array_equal(arrays["scaler_min"], minimum) or not np.array_equal(
        arrays["scaler_range"], cell_range
    ):
        raise LstmFullContractError(
            "Train-only scaler parameter가 재계산값과 다릅니다."
        )
    if np.any(cell_range <= 0):
        raise LstmFullContractError("zero-range 셀이 있습니다.")
    if np.any(
        arrays["missing_mask_train_validation"]
        & arrays["internet_null_mask_train_validation"]
    ):
        raise LstmFullContractError("두 결측 mask가 겹칩니다.")
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise LstmFullContractError("compact input에 NaN 또는 무한대가 있습니다.")
    return arrays, manifest


def load_and_verify_job_descriptor(
    config: LstmFullTrainingConfig,
    descriptor_path: Path,
    input_manifest: Mapping[str, Any],
) -> tuple[FullJobSpec, Mapping[str, Any]]:
    """descriptor가 config의 job 하나와 input manifest에 정확히 묶였는지 확인한다."""

    descriptor = _load_json(descriptor_path, "full training job descriptor")
    expected_keys = {
        "schema_version",
        "status",
        "job_id",
        "seed",
        "condition",
        "cluster_id",
        "expected_cell_count",
        "config_sha256",
        "input_npz_sha256",
        "source_git",
        "test_allowed",
        "output_relative_directory",
    }
    if set(descriptor) != expected_keys:
        raise LstmFullContractError("job descriptor key 집합이 다릅니다.")
    if descriptor.get("schema_version") != 1 or descriptor.get("status") != "ready":
        raise LstmFullContractError("job descriptor 상태가 ready가 아닙니다.")
    job = _expected_job(config, str(descriptor.get("job_id")))
    expected_values = {
        "seed": job.seed,
        "condition": job.condition,
        "cluster_id": job.cluster_id,
        "expected_cell_count": job.expected_cell_count,
        "config_sha256": compute_sha256(config.path),
        "input_npz_sha256": compute_sha256(config.outputs.input_npz),
        "source_git": input_manifest.get("git"),
        "test_allowed": False,
        "output_relative_directory": (
            Path("data/processed/lstm_full_training/jobs") / job.job_id
        ).as_posix(),
    }
    for field, expected in expected_values.items():
        if descriptor.get(field) != expected:
            raise LstmFullContractError(f"job descriptor {field} 값이 다릅니다.")
    jobs_metadata = input_manifest.get("jobs")
    if (
        not isinstance(jobs_metadata, dict)
        or jobs_metadata.get("expected_job_count") != config.training.expected_job_count
    ):
        raise LstmFullContractError("input manifest job 수가 다릅니다.")
    descriptors = jobs_metadata.get("descriptors")
    metadata = descriptors.get(job.job_id) if isinstance(descriptors, dict) else None
    if not isinstance(metadata, dict) or metadata.get("sha256") != compute_sha256(
        descriptor_path
    ):
        raise LstmFullContractError("job descriptor checksum이 manifest와 다릅니다.")
    if any(
        metadata.get(field) != expected_values[field]
        for field in (
            "seed",
            "condition",
            "cluster_id",
            "expected_cell_count",
        )
    ):
        raise LstmFullContractError("job descriptor metadata가 config와 다릅니다.")
    return job, descriptor


def build_job_arrays(
    config: LstmFullTrainingConfig,
    arrays: Mapping[str, np.ndarray],
    job: FullJobSpec,
) -> dict[str, np.ndarray]:
    """해당 job의 셀만 골라 Train·Validation window를 cell-major로 만든다."""

    memberships = np.asarray(arrays["memberships"], dtype=np.int8)
    if job.cluster_id is None:
        central_positions = np.arange(len(memberships), dtype=np.int32)
    else:
        central_positions = np.flatnonzero(memberships == job.cluster_id).astype(
            np.int32
        )
    if len(central_positions) != job.expected_cell_count:
        raise LstmFullContractError(f"{job.job_id} 선택 셀 수가 config와 다릅니다.")

    traffic = np.asarray(
        arrays["traffic_train_validation"][central_positions], dtype=np.float32
    )
    minimum = np.asarray(arrays["scaler_min"][central_positions], dtype=np.float32)
    cell_range = np.asarray(arrays["scaler_range"][central_positions], dtype=np.float32)
    scaled = np.asarray(
        (traffic - minimum[:, None]) / cell_range[:, None], dtype=np.float32
    )
    output: dict[str, np.ndarray] = {
        "central_positions": central_positions,
        "cell_ids": np.asarray(arrays["cell_ids"][central_positions], dtype=np.int32),
        "scaler_min": minimum,
        "scaler_range": cell_range,
    }
    offsets = np.arange(-config.data.input_length, 0, dtype=np.int64)
    for split in config.data.splits:
        targets = np.asarray(arrays[f"target_indices_{split.name}"], dtype=np.int64)
        window_indices = targets[:, None] + offsets[None, :]
        if (
            int(window_indices.min()) < config.data.bundle_global_start_index_inclusive
            or int(targets.max()) >= config.data.test_target_start_index_inclusive
            or np.any(window_indices >= targets[:, None])
        ):
            raise LstmFullContractError(
                f"{split.name} window가 인과/Test 경계를 위반합니다."
            )
        output[f"x_{split.name}"] = np.ascontiguousarray(
            scaled[:, window_indices, None], dtype=np.float32
        )
        output[f"y_{split.name}"] = np.ascontiguousarray(
            scaled[:, targets, None], dtype=np.float32
        )
        expected_samples = job.expected_cell_count * split.targets_per_cell
        if output[f"x_{split.name}"].shape != (
            job.expected_cell_count,
            split.targets_per_cell,
            config.data.input_length,
            1,
        ) or output[f"y_{split.name}"].shape != (
            job.expected_cell_count,
            split.targets_per_cell,
            1,
        ):
            raise LstmFullContractError(f"{split.name} window shape가 다릅니다.")
        if (
            output[f"x_{split.name}"].shape[0] * split.targets_per_cell
            != expected_samples
        ):
            raise LstmFullContractError(f"{split.name} sample 수가 다릅니다.")

    validation_targets = np.asarray(arrays["target_indices_validation"], dtype=np.int64)
    missing = np.asarray(arrays["missing_mask_train_validation"], dtype=bool)
    internet_null = np.asarray(
        arrays["internet_null_mask_train_validation"], dtype=bool
    )
    output.update(
        {
            "target_indices_validation": validation_targets,
            "raw_y_validation": np.ascontiguousarray(
                traffic[:, validation_targets, None], dtype=np.float32
            ),
            "target_missing_mask_validation": np.ascontiguousarray(
                missing[central_positions][:, validation_targets], dtype=bool
            ),
            "target_internet_null_mask_validation": np.ascontiguousarray(
                internet_null[central_positions][:, validation_targets], dtype=bool
            ),
            "lag_missing_mask_validation": np.ascontiguousarray(
                missing[central_positions][:, validation_targets - 1], dtype=bool
            ),
            "lag_internet_null_mask_validation": np.ascontiguousarray(
                internet_null[central_positions][:, validation_targets - 1], dtype=bool
            ),
        }
    )
    train_values = (output["x_train"], output["y_train"])
    if (
        min(float(value.min()) for value in train_values) < -1e-6
        or max(float(value.max()) for value in train_values) > 1.0 + 1e-6
    ):
        raise LstmFullContractError("job Train scaled x/y가 [0, 1]을 벗어났습니다.")
    if not all(np.all(np.isfinite(value)) for value in output.values()):
        raise LstmFullContractError("job array에 NaN 또는 무한대가 있습니다.")
    return output


def _flatten(values: np.ndarray) -> np.ndarray:
    return values.reshape((-1, *values.shape[2:]))


def _mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(targets.astype(np.float64) - predictions.astype(np.float64)),
            dtype=np.float64,
        )
    )


def _weights_finite(weights: Sequence[np.ndarray]) -> bool:
    return all(np.all(np.isfinite(value)) for value in weights)


def _prediction_diagnostics(values: np.ndarray) -> dict[str, Any]:
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(np.mean(values, dtype=np.float64)),
        "negative_count": int((values < 0).sum()),
        "finite": bool(np.all(np.isfinite(values))),
    }


def run_lstm_full_training_job(
    config: LstmFullTrainingConfig,
    descriptor_path: Path,
) -> dict[str, Any]:
    """사전 등록한 early stopping으로 job 하나를 학습하고 best 산출물을 저장한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    tf = require_tensorflow()
    runtime_environment = _runtime_environment(tf)
    required_gpu = config.resources.required_colab_gpu_name_contains
    if not runtime_environment["gpu_available"] or not any(
        required_gpu in row["name"]
        for row in runtime_environment["nvidia_smi_inventory"]
    ):
        raise LstmFullContractError(f"이 job은 Colab {required_gpu} GPU가 필요합니다.")
    arrays, input_manifest = load_and_verify_full_bundle(config)
    source_git = verified_source_git(input_manifest)
    job, descriptor = load_and_verify_job_descriptor(
        config, descriptor_path, input_manifest
    )
    job_arrays = build_job_arrays(config, arrays, job)
    del arrays

    tf.keras.backend.clear_session()
    configure_determinism(job.seed)
    audit_model = build_lstm_model(
        spec=config.architecture,
        dropout=config.training.dropout,
        learning_rate=config.training.learning_rate,
        compile_model=False,
    )
    architecture_report = audit_lstm_model(
        audit_model, spec=config.architecture, seed=job.seed
    )
    if not architecture_report["required_gates_passed"]:
        raise LstmFullContractError("LSTM architecture 필수 gate가 실패했습니다.")
    del audit_model
    tf.keras.backend.clear_session()

    determinism = configure_determinism(job.seed)
    model = build_lstm_model(
        spec=config.architecture,
        dropout=config.training.dropout,
        learning_rate=config.training.learning_rate,
    )
    x_train = _flatten(job_arrays["x_train"])
    y_train = _flatten(job_arrays["y_train"])
    x_validation = _flatten(job_arrays["x_validation"])
    y_validation = _flatten(job_arrays["y_validation"])

    class EpochTimer(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.seconds: list[float] = []
            self._epoch_started = 0.0

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            self._epoch_started = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            self.seconds.append(time.perf_counter() - self._epoch_started)

    class WallClockLimit(tf.keras.callbacks.Callback):
        def __init__(self, deadline: float) -> None:
            super().__init__()
            self.deadline = deadline
            self.exceeded = False

        def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
            if time.perf_counter() >= self.deadline:
                self.exceeded = True
                self.model.stop_training = True

    early_spec = config.training.early_stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=early_spec.monitor,
        mode=early_spec.mode,
        patience=early_spec.patience,
        min_delta=early_spec.min_delta,
        restore_best_weights=early_spec.restore_best_weights,
        start_from_epoch=early_spec.start_from_epoch,
        verbose=1,
    )
    timer = EpochTimer()
    wall_clock = WallClockLimit(
        started_counter + config.training.maximum_wall_clock_seconds_per_job
    )
    fit_started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        callbacks=[timer, wall_clock, early_stopping],
        verbose=2,
    )
    fit_seconds = time.perf_counter() - fit_started
    if wall_clock.exceeded:
        raise LstmFullContractError(
            "job wall-clock 상한을 넘어 incomplete로 종료했습니다; 결과로 게시하지 않습니다."
        )

    losses = np.asarray(history.history.get("loss", []), dtype=np.float64)
    validation_losses = np.asarray(
        history.history.get("val_loss", []), dtype=np.float64
    )
    if not len(losses) or losses.shape != validation_losses.shape:
        raise LstmFullContractError("학습 history에 loss/val_loss가 없습니다.")
    best_epoch_index = int(np.argmin(validation_losses))
    best_epoch = best_epoch_index + 1
    best_validation_loss = float(validation_losses[best_epoch_index])
    callback_best_weights = getattr(early_stopping, "best_weights", None)
    current_weights = [np.asarray(value) for value in model.get_weights()]
    restored_exact = bool(
        isinstance(callback_best_weights, list)
        and len(callback_best_weights) == len(current_weights)
        and all(
            np.array_equal(current, best)
            for current, best in zip(
                current_weights, callback_best_weights, strict=True
            )
        )
    )
    flat_prediction_scaled = np.asarray(
        model.predict(x_validation, batch_size=config.training.batch_size, verbose=0),
        dtype=np.float32,
    )
    restored_validation_mae = _mae(y_validation, flat_prediction_scaled)
    restored_loss_difference = abs(restored_validation_mae - best_validation_loss)
    prediction_scaled = flat_prediction_scaled.reshape(
        (job.expected_cell_count, config.data.split("validation").targets_per_cell, 1)
    )
    prediction_raw = np.asarray(
        prediction_scaled * job_arrays["scaler_range"][:, None, None]
        + job_arrays["scaler_min"][:, None, None],
        dtype=np.float32,
    )
    peak_rss_bytes = _peak_rss_bytes()
    if peak_rss_bytes is None:
        raise LstmFullContractError("Colab peak RSS를 측정할 수 없습니다.")
    gates = {
        "architecture": bool(architecture_report["required_gates_passed"]),
        "exact_parameter_count": int(model.count_params())
        == config.architecture.expected_parameter_count,
        "test_absent": bool(
            descriptor["test_allowed"] is False
            and int(job_arrays["target_indices_validation"].max())
            < config.data.test_target_start_index_inclusive
            and all("test" not in name.lower() for name in job_arrays)
        ),
        "finite_history_weights_and_predictions": bool(
            np.all(np.isfinite(losses))
            and np.all(np.isfinite(validation_losses))
            and _weights_finite(current_weights)
            and np.all(np.isfinite(prediction_scaled))
            and np.all(np.isfinite(prediction_raw))
        ),
        "best_weights_restored_exactly": restored_exact,
        "restored_prediction_matches_best_validation_loss": restored_loss_difference
        <= 1e-5,
        "wall_clock_within_limit": not wall_clock.exceeded,
        "peak_rss_within_soft_limit": peak_rss_bytes
        <= config.resources.colab_peak_rss_soft_limit_bytes,
        "performance_not_used_as_gate": bool(
            not config.pass_criteria.require_better_than_persistence
            and not config.pass_criteria.require_upc_improvement
        ),
    }
    if not all(gates.values()):
        failure = json.dumps(gates, ensure_ascii=False, allow_nan=False)
        raise LstmFullContractError(f"job 필수 gate가 실패했습니다: {failure}")

    training_report = {
        "schema_version": 1,
        "status": "pass",
        "job": {
            "job_id": job.job_id,
            "seed": job.seed,
            "condition": job.condition,
            "cluster_id": job.cluster_id,
            "cell_count": job.expected_cell_count,
        },
        "source": {
            "git": source_git,
            "config_sha256": compute_sha256(config.path),
            "input_npz_sha256": compute_sha256(config.outputs.input_npz),
            "descriptor_sha256": compute_sha256(descriptor_path),
        },
        "model": {
            "name": config.architecture.name,
            "parameter_count": int(model.count_params()),
            "author_implementation_confirmed": False,
            "architecture_audit": architecture_report,
        },
        "training_contract": {
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "loss": config.training.loss,
            "loss_domain": config.training.loss_domain,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "dropout": config.training.dropout,
            "shuffle": config.training.shuffle,
            "early_stopping": {
                "monitor": early_spec.monitor,
                "monitor_domain": early_spec.monitor_domain,
                "mode": early_spec.mode,
                "patience": early_spec.patience,
                "min_delta": early_spec.min_delta,
                "restore_best_weights": early_spec.restore_best_weights,
                "start_from_epoch": early_spec.start_from_epoch,
            },
        },
        "samples": {
            split.name: job.expected_cell_count * split.targets_per_cell
            for split in config.data.splits
        },
        "determinism": determinism,
        "history": {
            "loss": [float(value) for value in losses],
            "val_loss": [float(value) for value in validation_losses],
        },
        "selection": {
            "completed_epochs": len(losses),
            "best_epoch": best_epoch,
            "best_scaled_validation_mae": best_validation_loss,
            "restored_scaled_validation_mae": restored_validation_mae,
            "absolute_loss_recalculation_difference": restored_loss_difference,
            "stopped_before_max_epochs": len(losses) < config.training.max_epochs,
            "checkpoint_selection": config.training.checkpoint_selection,
        },
        "prediction_diagnostics": {
            "scaled_validation": _prediction_diagnostics(prediction_scaled),
            "raw_validation": _prediction_diagnostics(prediction_raw),
        },
        "gates": gates,
        "gates_passed": True,
        "test_evaluated": False,
    }
    prediction_arrays = {
        "central_positions": job_arrays["central_positions"],
        "cell_ids": job_arrays["cell_ids"],
        "target_indices_validation": job_arrays["target_indices_validation"],
        "raw_y_validation": job_arrays["raw_y_validation"],
        "prediction_scaled_validation": prediction_scaled,
        "prediction_raw_validation": prediction_raw,
        "target_missing_mask_validation": job_arrays["target_missing_mask_validation"],
        "target_internet_null_mask_validation": job_arrays[
            "target_internet_null_mask_validation"
        ],
        "lag_missing_mask_validation": job_arrays["lag_missing_mask_validation"],
        "lag_internet_null_mask_validation": job_arrays[
            "lag_internet_null_mask_validation"
        ],
        "scaler_min": job_arrays["scaler_min"],
        "scaler_range": job_arrays["scaler_range"],
    }
    if tuple(prediction_arrays) != PREDICTION_ARRAY_NAMES or any(
        "test" in name.lower() for name in prediction_arrays
    ):
        raise LstmFullContractError(
            "Validation prediction output 이름 계약이 다릅니다."
        )
    weight_arrays = {
        f"weight_{index:03d}": np.asarray(value)
        for index, value in enumerate(current_weights)
    }

    output_dir = config.outputs.jobs_root / job.job_id
    output_paths = {
        "training_report": output_dir / "training_report.json",
        "validation_predictions": output_dir / "validation_predictions.npz",
        "best_weights": output_dir / "best_weights.npz",
        "run_manifest": output_dir / "run_manifest.json",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: _temporary_path(path) for name, path in output_paths.items()
    }
    published = False
    try:
        _write_json(temporary_paths["training_report"], training_report)
        with temporary_paths["validation_predictions"].open("wb") as handle:
            np.savez_compressed(handle, **prediction_arrays)
        with temporary_paths["best_weights"].open("wb") as handle:
            np.savez_compressed(handle, **weight_arrays)
        deterministic_names = (
            "training_report",
            "validation_predictions",
            "best_weights",
        )
        output_metadata = {
            name: _temporary_file_contract(output_paths[name], temporary_paths[name])
            for name in deterministic_names
        }
        output_metadata["validation_predictions"]["arrays"] = {
            name: _array_contract(value) for name, value in prediction_arrays.items()
        }
        output_metadata["best_weights"]["arrays"] = {
            name: _array_contract(value) for name, value in weight_arrays.items()
        }
        finished_at = datetime.now(timezone.utc)
        run_manifest = {
            "schema_version": 1,
            "status": "pass",
            "tool": {
                "name": "scripts.run_lstm_full_training_job",
                "version": TOOL_VERSION,
            },
            "created_at_utc": finished_at.isoformat(),
            "job": training_report["job"],
            "git": source_git,
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "input": {
                "npz_path": _display_path(config.outputs.input_npz),
                "npz_sha256": compute_sha256(config.outputs.input_npz),
                "manifest_path": _display_path(config.outputs.input_manifest),
                "manifest_sha256": compute_sha256(config.outputs.input_manifest),
                "descriptor_path": str(descriptor_path),
                "descriptor_sha256": compute_sha256(descriptor_path),
            },
            "gates": gates,
            "outputs": output_metadata,
            "runtime": {
                **runtime_environment,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "fit_seconds": fit_seconds,
                "epoch_seconds": timer.seconds,
                "peak_rss_bytes": peak_rss_bytes,
                "peak_rss_soft_limit_bytes": config.resources.colab_peak_rss_soft_limit_bytes,
            },
            "test_seal": {
                "policy": config.data.test_policy,
                "test_arrays_present": False,
                "test_evaluated": False,
            },
        }
        _write_json(temporary_paths["run_manifest"], run_manifest)
        for name in deterministic_names:
            os.replace(temporary_paths[name], output_paths[name])
        os.replace(temporary_paths["run_manifest"], output_paths["run_manifest"])
        published = True
        return run_manifest
    finally:
        if not published:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab T4에서 LSTM 전체 학습 job 하나를 실행합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--job", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_full_config(args.config)
        manifest = run_lstm_full_training_job(config, args.job.resolve())
    except (LstmFullContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"LSTM 전체 학습 job 상태: {manifest['status']}")
    print(f"job: {manifest['job']['job_id']}")
    print(f"best epoch: {manifest['outputs']['training_report']['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOB_OUTPUT_NAMES",
    "PREDICTION_ARRAY_NAMES",
    "build_job_arrays",
    "load_and_verify_full_bundle",
    "load_and_verify_job_descriptor",
    "run_lstm_full_training_job",
]
