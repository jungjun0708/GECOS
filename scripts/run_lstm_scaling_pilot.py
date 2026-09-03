#!/usr/bin/env python3
"""Colab T4에서 Train-only 셀별 Min-Max LSTM pilot을 실행한다."""

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
from scripts.lstm_model import (
    audit_lstm_model,
    build_lstm_model,
    configure_determinism,
    require_tensorflow,
)
from scripts.lstm_scaling_contract import (
    DEFAULT_CONFIG,
    LstmScalingContractError,
    LstmScalingPilotConfig,
    load_lstm_scaling_config,
)
from scripts.prepare_lstm_scaling_pilot import inverse_transform_cellwise
from scripts.run_lstm_upc_smoke import (
    _file_contract,
    _runtime_environment,
    _train_one_model,
    _write_json,
    _write_per_cell_csv,
    recombine_cluster_predictions,
    verified_source_git,
)

TOOL_VERSION = "1.0.0"
MODEL_LABELS = ("lstm_scaled_upc_off", "lstm_scaled_upc_on")
TARGET_POLICIES = ("all_targets", "observed_targets_only")
EVALUATION_SPLIT = "validation"
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


def _content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def expected_bundle_array_names(config: LstmScalingPilotConfig) -> set[str]:
    names = {"cell_ids", "memberships", "scaler_min", "scaler_range"}
    for split in config.selection.splits:
        names.update(
            {
                f"x_{split}",
                f"y_{split}",
                f"raw_y_{split}",
                f"raw_persistence_{split}",
            }
        )
        names.update(template.format(split=split) for template in COPIED_FIELDS)
    return names


def load_and_verify_scaling_bundle(
    config: LstmScalingPilotConfig,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """Test가 없는 scaling bundle의 provenance와 모든 배열 checksum을 검증한다."""

    manifest = _load_json(config.outputs.input_manifest, "scaling input manifest")
    if manifest.get("status") != "complete":
        raise LstmScalingContractError("scaling input manifest가 complete가 아닙니다.")
    verified_source_git(manifest)
    config_metadata = manifest.get("config")
    if not isinstance(config_metadata, dict) or config_metadata.get(
        "sha256"
    ) != compute_sha256(config.path):
        raise LstmScalingContractError(
            "scaling config checksum이 input manifest와 다릅니다."
        )
    base_metadata = manifest.get("base_smoke")
    if not isinstance(base_metadata, dict):
        raise LstmScalingContractError("input manifest에 기준 smoke 계약이 없습니다.")
    expected_base_hashes = {
        "config_sha256": config.base_reference.config_sha256,
        "input_npz_sha256": config.base_reference.input_npz_sha256,
        "evaluation_report_sha256": config.base_reference.evaluation_report_sha256,
    }
    if any(
        base_metadata.get(key) != value for key, value in expected_base_hashes.items()
    ):
        raise LstmScalingContractError(
            "input manifest의 기준 smoke checksum이 다릅니다."
        )
    scaler_metadata = manifest.get("scaler_source")
    if not isinstance(scaler_metadata, dict):
        raise LstmScalingContractError("input manifest에 scaler source가 없습니다.")
    central_manifest = scaler_metadata.get("central_manifest")
    central_traffic = scaler_metadata.get("central_traffic")
    if (
        not isinstance(central_manifest, dict)
        or central_manifest.get("sha256")
        != config.scaler_source.expected_central_manifest_sha256
        or not isinstance(central_traffic, dict)
        or central_traffic.get("sha256")
        != config.scaler_source.expected_central_traffic_sha256
    ):
        raise LstmScalingContractError(
            "input manifest의 scaler source checksum이 다릅니다."
        )
    selection = manifest.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("splits") != list(config.selection.splits)
        or selection.get("test_policy") != config.test_policy
        or selection.get("test_arrays_in_bundle") is not False
    ):
        raise LstmScalingContractError("input manifest의 Test 봉인 계약이 다릅니다.")
    scaling_metadata = manifest.get("scaling")
    if (
        not isinstance(scaling_metadata, dict)
        or scaling_metadata.get("name") != config.scaling.name
        or scaling_metadata.get("fit_used_validation") is not False
        or scaling_metadata.get("fit_used_test") is not False
        or scaling_metadata.get("clip_transform") is not False
        or scaling_metadata.get("clip_inverse_prediction") is not False
    ):
        raise LstmScalingContractError(
            "input manifest의 Train-only scaling 계약이 다릅니다."
        )
    fit_indices = scaling_metadata.get("fit_indices")
    if not isinstance(fit_indices, dict) or (
        fit_indices.get("start_inclusive"),
        fit_indices.get("end_exclusive"),
    ) != (
        config.scaling.fit_start_index_inclusive,
        config.scaling.fit_end_index_exclusive,
    ):
        raise LstmScalingContractError("input manifest의 scaler 적합 구간이 다릅니다.")

    output = manifest.get("output")
    if not isinstance(output, dict):
        raise LstmScalingContractError("input manifest.output이 없습니다.")
    if compute_sha256(config.outputs.input_npz) != output.get("sha256"):
        raise LstmScalingContractError("scaling input NPZ checksum이 다릅니다.")
    contracts = output.get("arrays")
    if not isinstance(contracts, dict):
        raise LstmScalingContractError("scaling input 배열 계약이 없습니다.")
    try:
        with np.load(config.outputs.input_npz, allow_pickle=False) as archive:
            expected_names = expected_bundle_array_names(config)
            if set(archive.files) != expected_names:
                raise LstmScalingContractError(
                    "scaling input NPZ 배열 이름이 다릅니다."
                )
            if any("test" in name.lower() for name in archive.files):
                raise LstmScalingContractError(
                    "scaling input NPZ에 Test 배열이 있습니다."
                )
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise LstmScalingContractError("scaling input NPZ를 읽을 수 없습니다.") from exc
    for name, array in arrays.items():
        contract = contracts.get(name)
        if (
            not isinstance(contract, dict)
            or contract.get("shape") != list(array.shape)
            or contract.get("dtype") != str(array.dtype)
            or contract.get("content_sha256") != _content_sha256(array)
        ):
            raise LstmScalingContractError(
                f"scaling input {name} 배열 계약이 다릅니다."
            )

    cell_count = config.selection.expected_central_cell_count
    target_count = config.selection.targets_per_cell_per_split
    if arrays["cell_ids"].shape != (cell_count,) or arrays["memberships"].shape != (
        cell_count,
    ):
        raise LstmScalingContractError("cell ID 또는 membership shape가 다릅니다.")
    actual_counts = tuple(
        (cluster, int((arrays["memberships"] == cluster).sum()))
        for cluster, _ in config.selection.expected_cluster_counts
    )
    if actual_counts != config.selection.expected_cluster_counts:
        raise LstmScalingContractError("bundle cluster 수가 다릅니다.")
    minimum = arrays["scaler_min"]
    cell_range = arrays["scaler_range"]
    if minimum.shape != (cell_count,) or cell_range.shape != (cell_count,):
        raise LstmScalingContractError("scaler parameter shape가 다릅니다.")
    if np.any(cell_range <= 0) or not all(
        np.all(np.isfinite(value)) for value in (minimum, cell_range)
    ):
        raise LstmScalingContractError("scaler parameter가 유효하지 않습니다.")
    for split in config.selection.splits:
        if arrays[f"x_{split}"].shape != (
            cell_count,
            target_count,
            config.architecture.input_length,
            1,
        ) or arrays[f"y_{split}"].shape != (cell_count, target_count, 1):
            raise LstmScalingContractError(f"{split} scaled x/y shape가 다릅니다.")
        if arrays[f"raw_y_{split}"].shape != (cell_count, target_count, 1):
            raise LstmScalingContractError(f"{split} raw y shape가 다릅니다.")
        restored = inverse_transform_cellwise(arrays[f"y_{split}"], minimum, cell_range)
        error = float(
            np.max(
                np.abs(
                    restored.astype(np.float64)
                    - arrays[f"raw_y_{split}"].astype(np.float64)
                )
            )
        )
        if error > config.scaling.roundtrip_max_absolute_error:
            raise LstmScalingContractError(
                f"{split} y 역변환 오차가 허용치를 넘었습니다."
            )
    if (
        min(float(arrays[name].min()) for name in ("x_train", "y_train")) < -1e-6
        or max(float(arrays[name].max()) for name in ("x_train", "y_train"))
        > 1.0 + 1e-6
    ):
        raise LstmScalingContractError("scaled Train 값이 [0, 1] 범위를 벗어났습니다.")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise LstmScalingContractError("scaling input에 NaN 또는 무한대가 있습니다.")
    return arrays, manifest


def classify_scaling_result(
    config: LstmScalingPilotConfig, validation_mae: float
) -> dict[str, Any]:
    """결과를 보지 않고 고정한 MAE 규칙으로 scaling 후보를 판정한다."""

    raw_mae = config.raw_reference.metric_for(config.decision_rule.primary_model).mae
    if not np.isfinite(validation_mae) or validation_mae < 0:
        raise LstmScalingContractError("판정할 Validation MAE가 유효하지 않습니다.")
    improvement = (raw_mae - validation_mae) / raw_mae
    if validation_mae <= config.decision_rule.material_improvement_max_mae:
        category = "material_improvement"
        outcome = config.decision_rule.material_improvement_outcome
    elif validation_mae < raw_mae:
        category = "positive_but_below_material"
        outcome = config.decision_rule.positive_but_below_material_outcome
    else:
        category = "no_improvement"
        outcome = config.decision_rule.no_improvement_outcome
    return {
        "metric": config.decision_rule.metric,
        "primary_model": config.decision_rule.primary_model,
        "raw_validation_mae": raw_mae,
        "scaled_validation_mae": validation_mae,
        "absolute_mae_change": validation_mae - raw_mae,
        "improvement_fraction": improvement,
        "improvement_percent": improvement * 100,
        "material_improvement_fraction_required": config.decision_rule.material_improvement_fraction,
        "material_improvement_max_mae": config.decision_rule.material_improvement_max_mae,
        "category": category,
        "outcome": outcome,
        "test_used": False,
    }


def _evaluate_validation(
    config: LstmScalingPilotConfig,
    arrays: Mapping[str, np.ndarray],
    prediction_sets: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_ids = np.asarray(arrays["cell_ids"], dtype=np.int32)
    scope = EvaluationScope(
        name="central_900_scaling_pilot",
        cell_ids=cell_ids,
        positions=np.arange(len(cell_ids), dtype=np.int64),
        protocol="central-900-approximate-selected-validation-scaling-pilot",
    )
    targets = np.asarray(arrays["raw_y_validation"][:, :, 0], dtype=np.float32)
    missing = np.asarray(arrays["target_missing_mask_validation"], dtype=bool)
    internet_null = np.asarray(
        arrays["target_internet_null_mask_validation"], dtype=bool
    )
    observed = ~(missing | internet_null)
    last_input_missing = np.asarray(
        arrays["input_missing_mask_validation"][:, :, -1], dtype=bool
    )
    last_input_null = np.asarray(
        arrays["input_internet_null_mask_validation"][:, :, -1], dtype=bool
    )
    summaries: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    for label, raw_predictions in prediction_sets.items():
        predictions = np.asarray(raw_predictions, dtype=np.float32)
        if predictions.shape == (*targets.shape, 1):
            predictions = predictions[:, :, 0]
        if predictions.shape != targets.shape:
            raise LstmScalingContractError(f"{label} prediction shape가 다릅니다.")
        for target_policy in TARGET_POLICIES:
            eligible = (
                np.ones_like(observed) if target_policy == "all_targets" else observed
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
                    require_nonnegative_predictions=(
                        label == "persistence_selected_smoke"
                    ),
                )
            )
            summary, rows = finalize_metric_parts(
                parts,
                scope=scope,
                split=EVALUATION_SPLIT,
                baseline=label,
                target_policy=target_policy,
                target_count_per_cell=targets.shape[1],
            )
            summaries.append(summary)
            per_cell_rows.extend(rows)
    return summaries, per_cell_rows


def _primary_mae(results: Sequence[Mapping[str, Any]]) -> float:
    for row in results:
        if (
            row.get("split") == EVALUATION_SPLIT
            and row.get("baseline") == "lstm_scaled_upc_off"
            and row.get("target_policy") == "all_targets"
        ):
            micro = row.get("micro")
            if isinstance(micro, dict) and isinstance(micro.get("mae"), (int, float)):
                return float(micro["mae"])
    raise LstmScalingContractError("primary Validation MAE 결과를 찾을 수 없습니다.")


def _prediction_diagnostics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(np.mean(array, dtype=np.float64)),
        "negative_count": int((array < 0).sum()),
        "finite": bool(np.all(np.isfinite(array))),
    }


def run_lstm_scaling_pilot(config: LstmScalingPilotConfig) -> dict[str, Any]:
    """T4에서 세 모델을 학습하고 raw 단위 Validation 지표로 후보를 판정한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    tf = require_tensorflow()
    runtime_environment = _runtime_environment(tf)
    if not runtime_environment["gpu_available"]:
        raise LstmScalingContractError(
            "LSTM scaling pilot은 Colab T4에서만 실행합니다."
        )
    if not any(
        "T4" in row["name"] for row in runtime_environment["nvidia_smi_inventory"]
    ):
        raise LstmScalingContractError("요청한 Tesla T4가 아니므로 pilot을 중단합니다.")
    arrays, input_manifest = load_and_verify_scaling_bundle(config)
    source_git = verified_source_git(input_manifest)

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
        raise LstmScalingContractError("LSTM architecture 필수 gate가 실패했습니다.")
    del audit_model
    tf.keras.backend.clear_session()

    all_indices = np.arange(
        config.selection.expected_central_cell_count, dtype=np.int64
    )
    off_scaled, off_training, off_runtime = _train_one_model(
        config, arrays, label="lstm_scaled_upc_off", cell_indices=all_indices
    )
    memberships = np.asarray(arrays["memberships"], dtype=np.int8)
    cluster_predictions: dict[int, dict[str, np.ndarray]] = {}
    cluster_training: list[dict[str, Any]] = []
    model_runtimes = [off_runtime]
    for cluster, _ in config.selection.expected_cluster_counts:
        indices = np.flatnonzero(memberships == cluster)
        predictions, report, model_runtime = _train_one_model(
            config,
            arrays,
            label=f"lstm_scaled_upc_on_cluster_{cluster}",
            cell_indices=indices,
        )
        cluster_predictions[cluster] = predictions
        cluster_training.append(report)
        model_runtimes.append(model_runtime)

    on_scaled: dict[str, np.ndarray] = {}
    recombination: dict[str, Any] = {}
    for split in config.selection.splits:
        combined, report = recombine_cluster_predictions(
            memberships=memberships,
            predictions_by_cluster={
                cluster: values[split]
                for cluster, values in cluster_predictions.items()
            },
            expected_cluster_counts=config.selection.expected_cluster_counts,
        )
        on_scaled[split] = combined
        recombination[split] = report

    minimum = np.asarray(arrays["scaler_min"], dtype=np.float32)
    cell_range = np.asarray(arrays["scaler_range"], dtype=np.float32)
    off_raw = inverse_transform_cellwise(
        off_scaled[EVALUATION_SPLIT], minimum, cell_range
    )
    on_raw = inverse_transform_cellwise(
        on_scaled[EVALUATION_SPLIT], minimum, cell_range
    )
    prediction_sets = {
        "persistence_selected_smoke": arrays["raw_persistence_validation"],
        "lstm_scaled_upc_off": off_raw,
        "lstm_scaled_upc_on": on_raw,
    }
    metric_results, per_cell_rows = _evaluate_validation(
        config, arrays, prediction_sets
    )
    decision = classify_scaling_result(config, _primary_mae(metric_results))
    training_reports = [off_training, *cluster_training]
    gates = {
        "architecture": bool(architecture_report["required_gates_passed"]),
        "train_only_scaler_fit": bool(
            input_manifest["scaling"]["fit_used_validation"] is False
            and input_manifest["scaling"]["fit_used_test"] is False
        ),
        "test_withheld": bool(
            input_manifest["selection"]["test_arrays_in_bundle"] is False
            and all("test" not in name.lower() for name in arrays)
        ),
        "scaling_roundtrip": bool(
            input_manifest["scaling"]["roundtrip_gate"]["passed"]
        ),
        "all_three_models_trained": len(training_reports) == 3,
        "all_training_gates": all(row["gates_passed"] for row in training_reports),
        "all_recombination_gates": all(row["exact"] for row in recombination.values()),
        "finite_raw_predictions": all(
            np.all(np.isfinite(value)) for value in (off_raw, on_raw)
        ),
        "finite_metric_results": all(
            all(
                value is None or np.isfinite(value)
                for aggregate in (row["micro"], row["cell_macro"])
                for value in aggregate.values()
            )
            for row in metric_results
        ),
        "decision_does_not_use_test": decision["test_used"] is False,
        "performance_not_used_as_pipeline_pass_gate": True,
    }
    overall_pass = all(gates.values())
    evaluation_report = {
        "schema_version": 1,
        "status": "pass" if overall_pass else "failed",
        "scope": "central 900 cells, 64 deterministic Validation targets per cell",
        "not_a_full_performance_result": True,
        "paper_table_ii_directly_comparable": False,
        "test_policy": config.test_policy,
        "test_evaluated": False,
        "only_changed_factor": config.scaling.only_changed_factor,
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
        "scaling_contract": input_manifest["scaling"],
        "training": training_reports,
        "recombination": recombination,
        "prediction_diagnostics_raw_units": {
            "lstm_scaled_upc_off": _prediction_diagnostics(off_raw),
            "lstm_scaled_upc_on": _prediction_diagnostics(on_raw),
        },
        "metric_contract": {
            "implementation": "scripts.evaluate_naive_baselines shared metric functions",
            "evaluation_split": EVALUATION_SPLIT,
            "target_policies": list(TARGET_POLICIES),
            "aggregations": ["micro", "cell_macro"],
            "unit": "original internet traffic after cellwise inverse transform",
            "prediction_clipping": False,
            "mape_zero_handling": "MAPE uses only eligible targets with y > 0",
        },
        "raw_validation_reference": input_manifest["raw_validation_reference"],
        "results": metric_results,
        "decision": decision,
        "gates": gates,
        "gates_passed": overall_pass,
    }

    deterministic_outputs = {
        "architecture_report": config.outputs.architecture_report,
        "evaluation_report": config.outputs.evaluation_report,
        "predictions_npz": config.outputs.predictions_npz,
        "per_cell_metrics_csv": config.outputs.per_cell_metrics_csv,
    }
    all_outputs = {**deterministic_outputs, "run_manifest": config.outputs.run_manifest}
    temporary_paths = {
        name: _temporary_path(path) for name, path in all_outputs.items()
    }
    for path in all_outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        _write_json(temporary_paths["architecture_report"], architecture_report)
        _write_json(temporary_paths["evaluation_report"], evaluation_report)
        prediction_arrays = {
            "cell_ids": np.asarray(arrays["cell_ids"], dtype=np.int32),
            "memberships": memberships,
            "scaler_min": minimum,
            "scaler_range": cell_range,
            "target_indices_validation": np.asarray(
                arrays["target_indices_validation"], dtype=np.int64
            ),
            "raw_y_validation": np.asarray(
                arrays["raw_y_validation"], dtype=np.float32
            ),
            "raw_persistence_validation": np.asarray(
                arrays["raw_persistence_validation"], dtype=np.float32
            ),
            "lstm_scaled_upc_off_prediction_scaled_validation": np.asarray(
                off_scaled["validation"], dtype=np.float32
            ),
            "lstm_scaled_upc_on_prediction_scaled_validation": np.asarray(
                on_scaled["validation"], dtype=np.float32
            ),
            "lstm_scaled_upc_off_prediction_raw_validation": off_raw,
            "lstm_scaled_upc_on_prediction_raw_validation": on_raw,
        }
        if any("test" in name.lower() for name in prediction_arrays):
            raise LstmScalingContractError("prediction output에 Test 배열이 있습니다.")
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
            "tool": {"name": "scripts.run_lstm_scaling_pilot", "version": TOOL_VERSION},
            "created_at_utc": finished_at.isoformat(),
            "scope": "central-900 Train-only scaling selected-Validation diagnostic",
            "not_a_full_performance_result": True,
            "git": source_git,
            "execution_workspace_git": _git_state(),
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
            "decision": decision,
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
            "checkpoint_policy": "No checkpoint is retained for this fixed-epoch pilot.",
        }
        _write_json(temporary_paths["run_manifest"], run_manifest)
        for name in deterministic_outputs:
            os.replace(temporary_paths[name], all_outputs[name])
        os.replace(temporary_paths["run_manifest"], config.outputs.run_manifest)
        published = True
        return run_manifest
    finally:
        if not published:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab T4에서 LSTM scaling pilot을 실행합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_scaling_config(args.config)
        manifest = run_lstm_scaling_pilot(config)
    except (LstmScalingContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"LSTM scaling pilot 상태: {manifest['status']}")
    print(f"판정: {manifest['decision']['outcome']}")
    print(f"manifest: {_display_path(config.outputs.run_manifest)}")
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "classify_scaling_result",
    "expected_bundle_array_names",
    "load_and_verify_scaling_bundle",
    "run_lstm_scaling_pilot",
]
