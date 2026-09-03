#!/usr/bin/env python3
"""Colab T4에서 중앙 900셀 LSTM의 UPC off/on pipeline smoke를 실행한다."""

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
from scripts.evaluate_naive_baselines import (
    EvaluationScope,
    PerCellMetricParts,
    compute_per_cell_metric_parts,
    finalize_metric_parts,
)
from scripts.lstm_contract import (
    DEFAULT_CONFIG,
    LstmSmokeContractError,
    LstmUpcSmokeConfig,
    load_lstm_smoke_config,
)
from scripts.lstm_model import (
    audit_lstm_model,
    build_lstm_model,
    configure_determinism,
    require_tensorflow,
)
from scripts.validate_upc_training_policy import require_training_allowed

TOOL_VERSION = "1.0.0"
MODEL_LABELS = ("lstm_upc_off", "lstm_upc_on")
EVALUATION_SPLITS = ("validation", "test")
TARGET_POLICIES = ("all_targets", "observed_targets_only")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LstmSmokeContractError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LstmSmokeContractError(
            f"{label}이 올바른 JSON이 아닙니다: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise LstmSmokeContractError(f"{label}의 최상위 값은 object여야 합니다.")
    return value


def _content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _expected_bundle_array_names(config: LstmUpcSmokeConfig) -> set[str]:
    names = {"cell_ids", "memberships"}
    for split_name in config.selection.splits:
        names.update(
            {
                f"x_{split_name}",
                f"y_{split_name}",
                f"persistence_{split_name}",
                f"target_indices_{split_name}",
                f"target_timestamps_ms_{split_name}",
                f"target_missing_mask_{split_name}",
                f"target_internet_null_mask_{split_name}",
                f"input_missing_mask_{split_name}",
                f"input_internet_null_mask_{split_name}",
            }
        )
    return names


def load_and_verify_bundle(
    config: LstmUpcSmokeConfig,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """입력 manifest, 정책, 설정과 NPZ 배열 checksum을 모두 확인한다."""

    manifest = _load_json(config.outputs.input_manifest, "LSTM input manifest")
    if manifest.get("status") != "complete":
        raise LstmSmokeContractError("LSTM input manifest가 complete 상태가 아닙니다.")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise LstmSmokeContractError("LSTM input manifest에 config 계약이 없습니다.")
    if compute_sha256(config.path) != manifest_config.get("sha256"):
        raise LstmSmokeContractError(
            "LSTM config checksum이 input manifest와 다릅니다."
        )

    source_inputs = manifest.get("source_inputs")
    if not isinstance(source_inputs, dict):
        raise LstmSmokeContractError("LSTM input manifest에 source_inputs가 없습니다.")
    source_paths = {
        "forecast_config": config.forecast_config_path,
        "upc_training_policy_config": config.policy_config_path,
    }
    for key, path in source_paths.items():
        metadata = source_inputs.get(key)
        if not isinstance(metadata, dict) or compute_sha256(path) != metadata.get(
            "sha256"
        ):
            raise LstmSmokeContractError(f"{key} checksum이 input manifest와 다릅니다.")

    policy_metadata = source_inputs.get("upc_training_policy")
    if not isinstance(policy_metadata, dict):
        raise LstmSmokeContractError("검증된 UPC training policy metadata가 없습니다.")
    policy_path = Path(str(policy_metadata.get("path", "")))
    if not policy_path.is_absolute():
        policy_path = config.base_directory / policy_path
    policy_path = policy_path.resolve()
    if compute_sha256(policy_path) != policy_metadata.get("sha256"):
        raise LstmSmokeContractError("UPC training policy checksum이 다릅니다.")
    policy = _load_json(policy_path, "UPC training policy")
    require_training_allowed(policy, config.upc_protocol)

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise LstmSmokeContractError("LSTM input manifest.output이 없습니다.")
    if compute_sha256(config.outputs.input_npz) != output.get("sha256"):
        raise LstmSmokeContractError("LSTM input NPZ checksum이 manifest와 다릅니다.")
    expected_contracts = output.get("arrays")
    if not isinstance(expected_contracts, dict):
        raise LstmSmokeContractError("LSTM input 배열 계약이 없습니다.")
    try:
        with np.load(config.outputs.input_npz, allow_pickle=False) as archive:
            if set(archive.files) != _expected_bundle_array_names(config):
                raise LstmSmokeContractError(
                    "LSTM input NPZ 배열 이름 계약이 다릅니다."
                )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise LstmSmokeContractError("LSTM input NPZ를 읽을 수 없습니다.") from exc

    for name, array in arrays.items():
        contract = expected_contracts.get(name)
        if not isinstance(contract, dict):
            raise LstmSmokeContractError(f"LSTM input {name} 계약이 없습니다.")
        if list(array.shape) != contract.get("shape") or str(
            array.dtype
        ) != contract.get("dtype"):
            raise LstmSmokeContractError(f"LSTM input {name} shape/dtype이 다릅니다.")
        if _content_sha256(array) != contract.get("content_sha256"):
            raise LstmSmokeContractError(f"LSTM input {name} 내용 checksum이 다릅니다.")

    cell_count = config.selection.expected_central_cell_count
    target_count = config.selection.targets_per_cell_per_split
    if arrays["cell_ids"].shape != (cell_count,) or arrays["memberships"].shape != (
        cell_count,
    ):
        raise LstmSmokeContractError("중앙 cell ID 또는 membership shape가 다릅니다.")
    if len(np.unique(arrays["cell_ids"])) != cell_count:
        raise LstmSmokeContractError("중앙 cell ID에 중복이 있습니다.")
    actual_counts = tuple(
        (cluster_id, int((arrays["memberships"] == cluster_id).sum()))
        for cluster_id, _ in config.selection.expected_cluster_counts
    )
    if actual_counts != config.selection.expected_cluster_counts:
        raise LstmSmokeContractError("bundle cluster 수가 config와 다릅니다.")

    for split_name in config.selection.splits:
        x = arrays[f"x_{split_name}"]
        y = arrays[f"y_{split_name}"]
        expected_x_shape = (
            cell_count,
            target_count,
            config.architecture.input_length,
            1,
        )
        if x.shape != expected_x_shape or y.shape != (cell_count, target_count, 1):
            raise LstmSmokeContractError(f"{split_name} x/y shape가 config와 다릅니다.")
        target_indices = arrays[f"target_indices_{split_name}"]
        if target_indices.shape != (target_count,) or not np.all(
            np.diff(target_indices) > 0
        ):
            raise LstmSmokeContractError(
                f"{split_name} target index가 증가하지 않습니다."
            )
        if not all(
            np.all(np.isfinite(arrays[name]))
            for name in (
                f"x_{split_name}",
                f"y_{split_name}",
                f"persistence_{split_name}",
            )
        ):
            raise LstmSmokeContractError(
                f"{split_name} 입력에 NaN 또는 무한대가 있습니다."
            )
    return arrays, manifest


def recombine_cluster_predictions(
    *,
    memberships: np.ndarray,
    predictions_by_cluster: Mapping[int, np.ndarray],
    expected_cluster_counts: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """cluster별 예측을 원래 중앙 셀 순서로 scatter하고 정확성을 확인한다."""

    if memberships.ndim != 1:
        raise LstmSmokeContractError("membership은 1차원이어야 합니다.")
    expected_ids = tuple(cluster_id for cluster_id, _ in expected_cluster_counts)
    if tuple(sorted(predictions_by_cluster)) != expected_ids:
        raise LstmSmokeContractError("cluster prediction key가 config와 다릅니다.")
    first = np.asarray(predictions_by_cluster[expected_ids[0]])
    if first.ndim < 2:
        raise LstmSmokeContractError(
            "cluster prediction은 cell 축과 target 축이 필요합니다."
        )
    result = np.empty((len(memberships), *first.shape[1:]), dtype=first.dtype)
    filled = np.zeros(len(memberships), dtype=bool)
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, expected_count in expected_cluster_counts:
        indices = np.flatnonzero(memberships == cluster_id)
        predictions = np.asarray(predictions_by_cluster[cluster_id])
        if len(indices) != expected_count or predictions.shape != (
            expected_count,
            *first.shape[1:],
        ):
            raise LstmSmokeContractError(
                f"cluster {cluster_id} prediction shape 또는 셀 수가 다릅니다."
            )
        if np.any(filled[indices]):
            raise LstmSmokeContractError("cluster 재결합 중 셀이 중복 배정됐습니다.")
        result[indices] = predictions
        filled[indices] = True
        exact = np.array_equal(result[indices], predictions)
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "cell_count": len(indices),
                "first_central_position": int(indices[0]),
                "last_central_position": int(indices[-1]),
                "scatter_exact": exact,
            }
        )
    complete = bool(np.all(filled))
    finite = bool(np.all(np.isfinite(result)))
    report = {
        "cell_count": len(memberships),
        "filled_cell_count": int(filled.sum()),
        "unfilled_cell_count": int((~filled).sum()),
        "duplicate_cell_count": 0,
        "finite": finite,
        "clusters": cluster_rows,
        "exact": complete
        and finite
        and all(row["scatter_exact"] for row in cluster_rows),
    }
    if not report["exact"]:
        raise LstmSmokeContractError(
            "cluster 예측 재결합 exact gate를 통과하지 못했습니다."
        )
    return result, report


def _runtime_environment(tf: Any) -> dict[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    try:
        inventory_rows = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        inventory_rows = []
    inventory: list[dict[str, Any]] = []
    for row in inventory_rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 3:
            inventory.append(
                {
                    "name": fields[0],
                    "memory_total_mib": int(fields[1]),
                    "driver_version": fields[2],
                }
            )
    try:
        build_info = {
            key: str(value) for key, value in tf.sysconfig.get_build_info().items()
        }
    except (AttributeError, TypeError):
        build_info = {}
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", None),
        "tensorflow_build_info": build_info,
        "physical_gpus": [
            {"name": item.name, "device_type": item.device_type}
            for item in physical_gpus
        ],
        "nvidia_smi_inventory": inventory,
        "gpu_available": bool(physical_gpus),
    }


def _flatten_for_cells(
    arrays: Mapping[str, np.ndarray], split_name: str, cell_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(arrays[f"x_{split_name}"][cell_indices], dtype=np.float32)
    y = np.asarray(arrays[f"y_{split_name}"][cell_indices], dtype=np.float32)
    return x.reshape((-1, *x.shape[2:])), y.reshape((-1, 1))


def _mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(targets.astype(np.float64) - predictions.astype(np.float64)),
            dtype=np.float64,
        )
    )


def _train_one_model(
    config: LstmUpcSmokeConfig,
    arrays: Mapping[str, np.ndarray],
    *,
    label: str,
    cell_indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """한 공간 범위에 동일 초기화의 LSTM을 학습하고 세 split을 예측한다."""

    tf = require_tensorflow()
    tf.keras.backend.clear_session()
    determinism = configure_determinism(config.seed)
    model = build_lstm_model(
        spec=config.architecture,
        dropout=config.training.dropout,
        learning_rate=config.training.learning_rate,
    )
    x_train, y_train = _flatten_for_cells(arrays, "train", cell_indices)
    x_validation, y_validation = _flatten_for_cells(arrays, "validation", cell_indices)
    prefit = np.asarray(
        model.predict(x_train, batch_size=config.training.batch_size, verbose=0),
        dtype=np.float32,
    )
    prefit_mae = _mae(y_train, prefit)

    class EpochTimer(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.seconds: list[float] = []
            self._started = 0.0

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            self.seconds.append(time.perf_counter() - self._started)

    timer = EpochTimer()
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        callbacks=[timer],
        verbose=0,
    )
    training_seconds = time.perf_counter() - started
    predictions: dict[str, np.ndarray] = {}
    flat_predictions: dict[str, np.ndarray] = {}
    for split_name in config.selection.splits:
        x_split, _ = _flatten_for_cells(arrays, split_name, cell_indices)
        flat = np.asarray(
            model.predict(x_split, batch_size=config.training.batch_size, verbose=0),
            dtype=np.float32,
        )
        flat_predictions[split_name] = flat
        predictions[split_name] = flat.reshape(
            (len(cell_indices), config.selection.targets_per_cell_per_split, 1)
        )
    _, y_train_check = _flatten_for_cells(arrays, "train", cell_indices)
    final_train_mae = _mae(y_train_check, flat_predictions["train"])
    finite = all(np.all(np.isfinite(value)) for value in predictions.values())
    losses = [float(value) for value in history.history["loss"]]
    validation_losses = [float(value) for value in history.history["val_loss"]]
    gates = {
        "finite_predictions_and_history": bool(
            finite
            and np.all(np.isfinite(losses))
            and np.all(np.isfinite(validation_losses))
        ),
        "train_mae_decreased": final_train_mae < prefit_mae,
        "fixed_epoch_count_completed": len(losses) == config.training.max_epochs,
    }
    report = {
        "label": label,
        "cell_count": len(cell_indices),
        "sample_counts": {
            split_name: len(cell_indices) * config.selection.targets_per_cell_per_split
            for split_name in config.selection.splits
        },
        "seed_reset_before_model_build": True,
        "determinism": determinism,
        "metrics": {
            "prefit_train_mae": prefit_mae,
            "final_train_mae": final_train_mae,
            "prefit_to_final_reduction_fraction": (
                (prefit_mae - final_train_mae) / prefit_mae if prefit_mae else None
            ),
            "first_epoch_loss": losses[0],
            "last_epoch_loss": losses[-1],
            "first_epoch_validation_loss": validation_losses[0],
            "last_epoch_validation_loss": validation_losses[-1],
        },
        "history": {"loss": losses, "validation_loss": validation_losses},
        "prediction_diagnostics": {
            split_name: {
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "mean": float(np.mean(value, dtype=np.float64)),
                "negative_count": int((value < 0).sum()),
            }
            for split_name, value in predictions.items()
        },
        "gates": gates,
        "gates_passed": all(gates.values()),
    }
    runtime = {
        "label": label,
        "training_seconds": training_seconds,
        "epoch_seconds": timer.seconds,
    }
    return predictions, report, runtime


def _evaluate_prediction_sets(
    config: LstmUpcSmokeConfig,
    arrays: Mapping[str, np.ndarray],
    prediction_sets: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """기존 기준선과 동일한 metric 함수로 smoke 예측을 평가한다."""

    cell_ids = np.asarray(arrays["cell_ids"], dtype=np.int32)
    scope = EvaluationScope(
        name="central_900_smoke",
        cell_ids=cell_ids,
        positions=np.arange(len(cell_ids), dtype=np.int64),
        protocol="central-900-approximate-selected-target-smoke",
    )
    summaries: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    for split_name in EVALUATION_SPLITS:
        targets = np.asarray(arrays[f"y_{split_name}"][:, :, 0], dtype=np.float32)
        missing = np.asarray(arrays[f"target_missing_mask_{split_name}"], dtype=bool)
        internet_null = np.asarray(
            arrays[f"target_internet_null_mask_{split_name}"], dtype=bool
        )
        observed = ~(missing | internet_null)
        last_input_missing = np.asarray(
            arrays[f"input_missing_mask_{split_name}"][:, :, -1], dtype=bool
        )
        last_input_null = np.asarray(
            arrays[f"input_internet_null_mask_{split_name}"][:, :, -1], dtype=bool
        )
        for label, values_by_split in prediction_sets.items():
            predictions = np.asarray(values_by_split[split_name], dtype=np.float32)
            if predictions.shape == (*targets.shape, 1):
                predictions = predictions[:, :, 0]
            if predictions.shape != targets.shape:
                raise LstmSmokeContractError(
                    f"{label} {split_name} prediction shape가 target과 다릅니다."
                )
            for target_policy in TARGET_POLICIES:
                eligible = (
                    np.ones_like(observed)
                    if target_policy == "all_targets"
                    else observed
                )
                parts = PerCellMetricParts()
                parts.append(
                    compute_per_cell_metric_parts(
                        targets,
                        predictions,
                        eligible,
                        missing,
                        internet_null,
                        last_input_missing,
                        last_input_null,
                    )
                )
                summary, rows = finalize_metric_parts(
                    parts,
                    scope=scope,
                    split=split_name,
                    baseline=label,
                    target_policy=target_policy,
                    target_count_per_cell=targets.shape[1],
                )
                summaries.append(summary)
                if split_name == "test":
                    per_cell_rows.extend(rows)
    return summaries, per_cell_rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_per_cell_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "scope",
        "scope_protocol",
        "split",
        "baseline",
        "target_policy",
        "cell_id",
        "candidate_target_count",
        "eligible_target_count",
        "excluded_missing_target_count",
        "positive_target_count_for_mape",
        "zero_target_count_excluded_from_mape",
        "missing_target_count",
        "internet_all_null_target_count",
        "lag_source_missing_count",
        "lag_source_internet_all_null_count",
        "mae",
        "mape_ratio",
        "mape_percent",
        "wape",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_contract(final_path: Path, temporary_path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(final_path),
        "size_bytes": temporary_path.stat().st_size,
        "sha256": compute_sha256(temporary_path),
    }


def run_lstm_upc_smoke(config: LstmUpcSmokeConfig) -> dict[str, Any]:
    """T4 확인 후 LSTM UPC off/on 학습·재결합·평가 산출물을 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    tf = require_tensorflow()
    runtime_environment = _runtime_environment(tf)
    if not runtime_environment["gpu_available"]:
        raise LstmSmokeContractError("LSTM smoke는 Colab T4 GPU 환경에서만 실행합니다.")
    if not any(
        "T4" in item["name"] for item in runtime_environment["nvidia_smi_inventory"]
    ):
        raise LstmSmokeContractError(
            "요청한 Tesla T4가 아니므로 LSTM smoke를 중단합니다."
        )
    arrays, input_manifest = load_and_verify_bundle(config)

    tf.keras.backend.clear_session()
    configure_determinism(config.seed)
    audit_model = build_lstm_model(
        spec=config.architecture,
        dropout=config.training.dropout,
        learning_rate=config.training.learning_rate,
        compile_model=False,
    )
    architecture_report = audit_lstm_model(
        audit_model, spec=config.architecture, seed=config.seed
    )
    if not architecture_report["required_gates_passed"]:
        architecture_failure = {
            "required_gates": architecture_report["required_gates"],
            "actual_parameter_count": architecture_report["actual_parameter_count"],
            "actual_output_shape": architecture_report["actual_output_shape"],
            "gradient_counts": {
                key: architecture_report["gradient_audit"][key]
                for key in (
                    "trainable_variable_count",
                    "missing_gradient_count",
                    "nonfinite_gradient_count",
                    "nonzero_gradient_count",
                )
            },
            "zero_gradient_variables": [
                row["variable"]
                for row in architecture_report["gradient_audit"]["variables"]
                if row["gradient_present"]
                and row["finite"]
                and row["maximum_absolute_gradient"] == 0
            ],
        }
        failure_json = json.dumps(
            architecture_failure, ensure_ascii=False, allow_nan=False
        )
        print(f"LSTM_ARCHITECTURE_AUDIT_FAILURE={failure_json}", flush=True)
        raise LstmSmokeContractError(
            "LSTM architecture 필수 gate를 통과하지 못했습니다: " + failure_json
        )
    del audit_model
    tf.keras.backend.clear_session()

    all_indices = np.arange(
        config.selection.expected_central_cell_count, dtype=np.int64
    )
    off_predictions, off_training, off_runtime = _train_one_model(
        config, arrays, label="lstm_upc_off", cell_indices=all_indices
    )
    memberships = np.asarray(arrays["memberships"], dtype=np.int8)
    cluster_predictions: dict[int, dict[str, np.ndarray]] = {}
    cluster_training_reports: list[dict[str, Any]] = []
    model_runtimes = [off_runtime]
    for cluster_id, _ in config.selection.expected_cluster_counts:
        indices = np.flatnonzero(memberships == cluster_id)
        predictions, report, model_runtime = _train_one_model(
            config,
            arrays,
            label=f"lstm_upc_on_cluster_{cluster_id}",
            cell_indices=indices,
        )
        cluster_predictions[cluster_id] = predictions
        cluster_training_reports.append(report)
        model_runtimes.append(model_runtime)

    on_predictions: dict[str, np.ndarray] = {}
    recombination_reports: dict[str, Any] = {}
    for split_name in config.selection.splits:
        combined, report = recombine_cluster_predictions(
            memberships=memberships,
            predictions_by_cluster={
                cluster_id: values[split_name]
                for cluster_id, values in cluster_predictions.items()
            },
            expected_cluster_counts=config.selection.expected_cluster_counts,
        )
        on_predictions[split_name] = combined
        recombination_reports[split_name] = report

    prediction_sets = {
        "persistence_selected_smoke": {
            split_name: arrays[f"persistence_{split_name}"]
            for split_name in EVALUATION_SPLITS
        },
        "lstm_upc_off": {
            split_name: off_predictions[split_name] for split_name in EVALUATION_SPLITS
        },
        "lstm_upc_on": {
            split_name: on_predictions[split_name] for split_name in EVALUATION_SPLITS
        },
    }
    metric_results, per_cell_rows = _evaluate_prediction_sets(
        config, arrays, prediction_sets
    )
    training_reports = [off_training, *cluster_training_reports]
    gates = {
        "architecture": bool(architecture_report["required_gates_passed"]),
        "all_three_models_trained": len(training_reports) == 3,
        "all_training_gates": all(
            report["gates_passed"] for report in training_reports
        ),
        "all_recombination_gates": all(
            report["exact"] for report in recombination_reports.values()
        ),
        "finite_metric_results": all(
            all(
                value is None or np.isfinite(value)
                for aggregate in (row["micro"], row["cell_macro"])
                for value in aggregate.values()
            )
            for row in metric_results
        ),
        "persistence_is_not_a_performance_gate": not config.pass_criteria.require_better_than_persistence,
    }
    overall_pass = all(gates.values())
    evaluation_report = {
        "schema_version": 1,
        "status": "pass" if overall_pass else "failed",
        "scope": "central 900 cells, 64 deterministic targets per split",
        "not_a_performance_result": True,
        "paper_table_ii_directly_comparable": False,
        "model_contract": {
            "name": config.architecture.name,
            "author_implementation_confirmed": False,
            "seed": config.seed,
            "same_seed_reset_before_each_model": True,
            "upc_protocol": config.upc_protocol,
            "upc_off_model_count": 1,
            "upc_on_model_count": len(config.selection.expected_cluster_counts),
        },
        "training_contract": {
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "loss": config.training.loss,
            "batch_size": config.training.batch_size,
            "max_epochs": config.training.max_epochs,
            "dropout": config.training.dropout,
            "shuffle": config.training.shuffle,
            "input_scaling": config.training.input_scaling,
            "validation_role": config.training.validation_role,
            "test_role": config.training.test_role,
        },
        "selection_contract": input_manifest["smoke"],
        "training": training_reports,
        "recombination": recombination_reports,
        "metric_contract": {
            "implementation": "scripts.evaluate_naive_baselines shared metric functions",
            "target_policies": list(TARGET_POLICIES),
            "mape_zero_handling": "MAPE uses only eligible targets with y > 0",
            "aggregations": ["micro", "cell_macro"],
            "lag_source_fields_for_models": "most recent value in each eight-step input window",
        },
        "results": metric_results,
        "gates": gates,
        "gates_passed": overall_pass,
    }

    deterministic_outputs = {
        "architecture_report": config.outputs.architecture_report,
        "evaluation_report": config.outputs.evaluation_report,
        "predictions_npz": config.outputs.predictions_npz,
        "per_cell_metrics_csv": config.outputs.per_cell_metrics_csv,
    }
    all_output_paths = {
        **deterministic_outputs,
        "run_manifest": config.outputs.run_manifest,
    }
    temporary_paths = {
        name: _temporary_path(path) for name, path in all_output_paths.items()
    }
    for path in all_output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        _write_json(temporary_paths["architecture_report"], architecture_report)
        _write_json(temporary_paths["evaluation_report"], evaluation_report)
        prediction_arrays = {
            "cell_ids": np.asarray(arrays["cell_ids"], dtype=np.int32),
            "memberships": memberships,
        }
        for split_name in EVALUATION_SPLITS:
            prediction_arrays[f"target_indices_{split_name}"] = np.asarray(
                arrays[f"target_indices_{split_name}"], dtype=np.int64
            )
            prediction_arrays[f"persistence_{split_name}"] = np.asarray(
                arrays[f"persistence_{split_name}"], dtype=np.float32
            )
            prediction_arrays[f"lstm_upc_off_{split_name}"] = np.asarray(
                off_predictions[split_name], dtype=np.float32
            )
            prediction_arrays[f"lstm_upc_on_{split_name}"] = np.asarray(
                on_predictions[split_name], dtype=np.float32
            )
        with temporary_paths["predictions_npz"].open("wb") as handle:
            np.savez_compressed(handle, **prediction_arrays)
        _write_per_cell_csv(temporary_paths["per_cell_metrics_csv"], per_cell_rows)
        output_metadata = {
            name: _file_contract(deterministic_outputs[name], temporary_paths[name])
            for name in deterministic_outputs
        }
        finished_at = datetime.now(timezone.utc)
        run_manifest = {
            "schema_version": 1,
            "status": "pass" if overall_pass else "failed",
            "tool": {"name": "scripts.run_lstm_upc_smoke", "version": TOOL_VERSION},
            "created_at_utc": finished_at.isoformat(),
            "scope": "central-900 UPC off/on selected-target pipeline diagnostic",
            "not_a_performance_result": True,
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "input": {
                "npz_path": _display_path(config.outputs.input_npz),
                "npz_sha256": compute_sha256(config.outputs.input_npz),
                "manifest_path": _display_path(config.outputs.input_manifest),
                "manifest_sha256": compute_sha256(config.outputs.input_manifest),
            },
            "gates": gates,
            "outputs": output_metadata,
            "runtime": {
                **runtime_environment,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "peak_rss_bytes": _peak_rss_bytes(),
                "models": model_runtimes,
            },
            "checkpoint_policy": "No checkpoint is retained for this fixed-epoch pipeline smoke.",
        }
        _write_json(temporary_paths["run_manifest"], run_manifest)
        for name in deterministic_outputs:
            os.replace(temporary_paths[name], all_output_paths[name])
        os.replace(temporary_paths["run_manifest"], config.outputs.run_manifest)
        published = True
        return run_manifest
    finally:
        if not published:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab T4에서 중앙 900셀 LSTM UPC off/on smoke를 실행합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_smoke_config(args.config)
        manifest = run_lstm_upc_smoke(config)
    except (LstmSmokeContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"LSTM·UPC smoke 상태: {manifest['status']}")
    print(f"manifest: {_display_path(config.outputs.run_manifest)}")
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
