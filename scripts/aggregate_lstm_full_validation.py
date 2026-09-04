#!/usr/bin/env python3
"""9개 LSTM full job을 Test 없이 중앙 900셀 Validation 결과로 집계한다."""

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
from scripts.evaluate_naive_baselines import (
    EvaluationScope,
    PerCellMetricParts,
    compute_per_cell_metric_parts,
    finalize_metric_parts,
)
from scripts.lstm_full_contract import (
    DEFAULT_CONFIG,
    FullJobSpec,
    LstmFullContractError,
    LstmFullTrainingConfig,
    load_lstm_full_config,
)
from scripts.run_lstm_full_training_job import (
    PREDICTION_ARRAY_NAMES,
    load_and_verify_full_bundle,
)
from scripts.run_lstm_upc_smoke import recombine_cluster_predictions

TOOL_VERSION = "1.0.0"
MODEL_LABELS = ("lstm_full_upc_off", "lstm_full_upc_on")


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


def _contract_matches(array: np.ndarray, value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("shape") == list(array.shape)
        and value.get("dtype") == str(array.dtype)
        and value.get("content_sha256") == _content_sha256(array)
    )


def _expected_positions(memberships: np.ndarray, job: FullJobSpec) -> np.ndarray:
    if job.cluster_id is None:
        return np.arange(len(memberships), dtype=np.int32)
    return np.flatnonzero(memberships == job.cluster_id).astype(np.int32)


def load_and_verify_completed_job(
    config: LstmFullTrainingConfig,
    job: FullJobSpec,
    source_arrays: Mapping[str, np.ndarray],
    input_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """한 job의 manifest, report, prediction, checkpoint를 상호 검증한다."""

    output_dir = config.outputs.jobs_root / job.job_id
    paths = {
        "training_report": output_dir / "training_report.json",
        "validation_predictions": output_dir / "validation_predictions.npz",
        "best_weights": output_dir / "best_weights.npz",
        "run_manifest": output_dir / "run_manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise LstmFullContractError(f"{job.job_id} output이 부족합니다: {missing}")
    manifest = _load_json(paths["run_manifest"], f"{job.job_id} run manifest")
    report = _load_json(paths["training_report"], f"{job.job_id} training report")
    if manifest.get("status") != "pass" or report.get("status") != "pass":
        raise LstmFullContractError(f"{job.job_id}가 pass 상태가 아닙니다.")
    manifest_job = manifest.get("job")
    report_job = report.get("job")
    expected_job = {
        "job_id": job.job_id,
        "seed": job.seed,
        "condition": job.condition,
        "cluster_id": job.cluster_id,
        "cell_count": job.expected_cell_count,
    }
    if manifest_job != expected_job or report_job != expected_job:
        raise LstmFullContractError(f"{job.job_id} identity가 config와 다릅니다.")
    source_git = input_manifest.get("git")
    if manifest.get("git") != {
        **source_git,
        "provenance": "locally_prepared_input_manifest",
    }:
        raise LstmFullContractError(f"{job.job_id} source Git provenance가 다릅니다.")
    config_metadata = manifest.get("config")
    input_metadata = manifest.get("input")
    if (
        not isinstance(config_metadata, dict)
        or config_metadata.get("sha256") != compute_sha256(config.path)
        or not isinstance(input_metadata, dict)
        or input_metadata.get("npz_sha256") != compute_sha256(config.outputs.input_npz)
        or input_metadata.get("manifest_sha256")
        != compute_sha256(config.outputs.input_manifest)
    ):
        raise LstmFullContractError(f"{job.job_id} config/input provenance가 다릅니다.")
    descriptor_metadata = (
        input_manifest.get("jobs", {}).get("descriptors", {}).get(job.job_id)
    )
    if not isinstance(descriptor_metadata, dict) or input_metadata.get(
        "descriptor_sha256"
    ) != descriptor_metadata.get("sha256"):
        raise LstmFullContractError(f"{job.job_id} descriptor provenance가 다릅니다.")
    seal = manifest.get("test_seal")
    if (
        not isinstance(seal, dict)
        or seal.get("policy") != config.data.test_policy
        or seal.get("test_arrays_present") is not False
        or seal.get("test_evaluated") is not False
        or report.get("test_evaluated") is not False
    ):
        raise LstmFullContractError(f"{job.job_id} Test 봉인이 깨졌습니다.")
    gates = report.get("gates")
    if (
        report.get("gates_passed") is not True
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise LstmFullContractError(f"{job.job_id} training gate가 실패했습니다.")

    output_contracts = manifest.get("outputs")
    if not isinstance(output_contracts, dict):
        raise LstmFullContractError(f"{job.job_id} output 계약이 없습니다.")
    for name in ("training_report", "validation_predictions", "best_weights"):
        contract = output_contracts.get(name)
        if not isinstance(contract, dict) or contract.get("sha256") != compute_sha256(
            paths[name]
        ):
            raise LstmFullContractError(f"{job.job_id} {name} checksum이 다릅니다.")

    try:
        with np.load(paths["validation_predictions"], allow_pickle=False) as archive:
            if tuple(archive.files) != PREDICTION_ARRAY_NAMES or any(
                "test" in name.lower() for name in archive.files
            ):
                raise LstmFullContractError(
                    f"{job.job_id} prediction 배열 이름 계약이 다릅니다."
                )
            predictions = {name: np.asarray(archive[name]) for name in archive.files}
        with np.load(paths["best_weights"], allow_pickle=False) as archive:
            expected_weight_names = tuple(
                f"weight_{index:03d}" for index in range(len(archive.files))
            )
            if tuple(archive.files) != expected_weight_names:
                raise LstmFullContractError(
                    f"{job.job_id} checkpoint 배열 이름이 순차적이지 않습니다."
                )
            weights = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise LstmFullContractError(f"{job.job_id} NPZ를 읽을 수 없습니다.") from exc
    prediction_contracts = output_contracts["validation_predictions"].get("arrays")
    weight_contracts = output_contracts["best_weights"].get("arrays")
    if not isinstance(prediction_contracts, dict) or set(prediction_contracts) != set(
        predictions
    ):
        raise LstmFullContractError(f"{job.job_id} prediction array 계약이 없습니다.")
    if not isinstance(weight_contracts, dict) or set(weight_contracts) != set(weights):
        raise LstmFullContractError(f"{job.job_id} weight array 계약이 없습니다.")
    if any(
        not _contract_matches(value, prediction_contracts[name])
        for name, value in predictions.items()
    ) or any(
        not _contract_matches(value, weight_contracts[name])
        for name, value in weights.items()
    ):
        raise LstmFullContractError(f"{job.job_id} NPZ 배열 계약이 다릅니다.")
    if not all(
        np.all(np.isfinite(value))
        for value in (*predictions.values(), *weights.values())
    ):
        raise LstmFullContractError(
            f"{job.job_id} output에 NaN 또는 무한대가 있습니다."
        )

    memberships = np.asarray(source_arrays["memberships"], dtype=np.int8)
    positions = _expected_positions(memberships, job)
    validation_targets = np.asarray(
        source_arrays["target_indices_validation"], dtype=np.int64
    )
    expected_values = {
        "central_positions": positions,
        "cell_ids": np.asarray(source_arrays["cell_ids"][positions], dtype=np.int32),
        "target_indices_validation": validation_targets,
        "raw_y_validation": np.asarray(
            source_arrays["traffic_train_validation"][positions][
                :, validation_targets, None
            ],
            dtype=np.float32,
        ),
        "target_missing_mask_validation": np.asarray(
            source_arrays["missing_mask_train_validation"][positions][
                :, validation_targets
            ],
            dtype=bool,
        ),
        "target_internet_null_mask_validation": np.asarray(
            source_arrays["internet_null_mask_train_validation"][positions][
                :, validation_targets
            ],
            dtype=bool,
        ),
        "lag_missing_mask_validation": np.asarray(
            source_arrays["missing_mask_train_validation"][positions][
                :, validation_targets - 1
            ],
            dtype=bool,
        ),
        "lag_internet_null_mask_validation": np.asarray(
            source_arrays["internet_null_mask_train_validation"][positions][
                :, validation_targets - 1
            ],
            dtype=bool,
        ),
        "scaler_min": np.asarray(
            source_arrays["scaler_min"][positions], dtype=np.float32
        ),
        "scaler_range": np.asarray(
            source_arrays["scaler_range"][positions], dtype=np.float32
        ),
    }
    if any(
        not np.array_equal(predictions[name], expected)
        for name, expected in expected_values.items()
    ):
        raise LstmFullContractError(f"{job.job_id} source-aligned 배열이 다릅니다.")
    expected_prediction_shape = (
        job.expected_cell_count,
        config.data.split("validation").targets_per_cell,
        1,
    )
    if (
        predictions["prediction_scaled_validation"].shape != expected_prediction_shape
        or predictions["prediction_raw_validation"].shape != expected_prediction_shape
    ):
        raise LstmFullContractError(f"{job.job_id} prediction shape가 다릅니다.")
    restored = (
        predictions["prediction_scaled_validation"]
        * predictions["scaler_range"][:, None, None]
        + predictions["scaler_min"][:, None, None]
    )
    inverse_error = float(
        np.max(
            np.abs(
                restored.astype(np.float64)
                - predictions["prediction_raw_validation"].astype(np.float64)
            )
        )
    )
    if inverse_error > config.scaling.roundtrip_max_absolute_error:
        raise LstmFullContractError(
            f"{job.job_id} prediction 역변환 오차가 허용치를 넘었습니다."
        )
    selection = report.get("selection")
    if (
        not isinstance(selection, dict)
        or not isinstance(selection.get("best_epoch"), int)
        or selection["best_epoch"] < 1
        or selection["best_epoch"] > config.training.max_epochs
    ):
        raise LstmFullContractError(f"{job.job_id} best epoch가 유효하지 않습니다.")
    return {
        "job": job,
        "manifest": manifest,
        "report": report,
        "predictions": predictions,
        "paths": paths,
        "inverse_error": inverse_error,
    }


def evaluate_seed_predictions(
    config: LstmFullTrainingConfig,
    source_arrays: Mapping[str, np.ndarray],
    predictions_by_seed: Mapping[int, Mapping[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """공통 지표 함수로 seed별 UPC off/on Validation을 raw 단위 평가한다."""

    cell_ids = np.asarray(source_arrays["cell_ids"], dtype=np.int32)
    targets_indices = np.asarray(
        source_arrays["target_indices_validation"], dtype=np.int64
    )
    targets = np.asarray(
        source_arrays["traffic_train_validation"][:, targets_indices],
        dtype=np.float32,
    )
    missing = np.asarray(
        source_arrays["missing_mask_train_validation"][:, targets_indices], dtype=bool
    )
    internet_null = np.asarray(
        source_arrays["internet_null_mask_train_validation"][:, targets_indices],
        dtype=bool,
    )
    lag_missing = np.asarray(
        source_arrays["missing_mask_train_validation"][:, targets_indices - 1],
        dtype=bool,
    )
    lag_internet_null = np.asarray(
        source_arrays["internet_null_mask_train_validation"][:, targets_indices - 1],
        dtype=bool,
    )
    observed = ~(missing | internet_null)
    scope = EvaluationScope(
        name="central_900_lstm_full_validation",
        cell_ids=cell_ids,
        positions=np.arange(len(cell_ids), dtype=np.int64),
        protocol="central-900-approximate-all-validation-targets",
    )
    summaries: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    for seed in config.training.seeds:
        values_by_label = predictions_by_seed.get(seed)
        if (
            not isinstance(values_by_label, dict)
            or tuple(values_by_label) != MODEL_LABELS
        ):
            raise LstmFullContractError(f"seed {seed} prediction 조건이 다릅니다.")
        for label, raw_predictions in values_by_label.items():
            predictions = np.asarray(raw_predictions, dtype=np.float32)
            if predictions.shape == (*targets.shape, 1):
                predictions = predictions[:, :, 0]
            if predictions.shape != targets.shape:
                raise LstmFullContractError(f"seed {seed} {label} shape가 다릅니다.")
            for target_policy in config.validation.target_policies:
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
                        lag_missing,
                        lag_internet_null,
                        require_nonnegative_predictions=False,
                    )
                )
                summary, rows = finalize_metric_parts(
                    parts,
                    scope=scope,
                    split="validation",
                    baseline=label,
                    target_policy=target_policy,
                    target_count_per_cell=targets.shape[1],
                )
                summary["seed"] = seed
                for row in rows:
                    row["seed"] = seed
                summaries.append(summary)
                per_cell_rows.extend(rows)
    return summaries, per_cell_rows


def summarize_seed_metrics(
    results: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """각 metric의 seed 개별값·평균·표본 표준편차(ddof=1)를 만든다."""

    output: list[dict[str, Any]] = []
    for model in MODEL_LABELS:
        for target_policy in ("all_targets", "observed_targets_only"):
            rows = {
                int(row["seed"]): row
                for row in results
                if row.get("baseline") == model
                and row.get("target_policy") == target_policy
            }
            if tuple(rows) != tuple(seeds):
                raise LstmFullContractError(
                    f"{model} {target_policy} seed 결과가 완전하지 않습니다."
                )
            for aggregation in ("micro", "cell_macro"):
                for metric in ("mae", "mape_ratio", "mape_percent", "wape"):
                    values = np.asarray(
                        [rows[seed][aggregation][metric] for seed in seeds],
                        dtype=np.float64,
                    )
                    if not np.all(np.isfinite(values)):
                        raise LstmFullContractError(
                            "seed summary metric이 유한하지 않습니다."
                        )
                    output.append(
                        {
                            "model": model,
                            "target_policy": target_policy,
                            "aggregation": aggregation,
                            "metric": metric,
                            "values_by_seed": {
                                str(seed): float(value)
                                for seed, value in zip(seeds, values, strict=True)
                            },
                            "seed_count": len(values),
                            "mean": float(format(np.mean(values), ".12g")),
                            "sample_standard_deviation_ddof_1": float(
                                format(np.std(values, ddof=1), ".12g")
                            ),
                        }
                    )
    return output


def _write_training_jobs_csv(
    path: Path, completed_jobs: Sequence[Mapping[str, Any]]
) -> None:
    fieldnames = [
        "job_id",
        "seed",
        "condition",
        "cluster_id",
        "cell_count",
        "completed_epochs",
        "best_epoch",
        "best_scaled_validation_mae",
        "restored_scaled_validation_mae",
        "stopped_before_max_epochs",
        "fit_seconds",
        "peak_rss_bytes",
        "checkpoint_sha256",
        "prediction_sha256",
        "run_manifest_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for completed in completed_jobs:
            report = completed["report"]
            manifest = completed["manifest"]
            job = completed["job"]
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "seed": job.seed,
                    "condition": job.condition,
                    "cluster_id": "" if job.cluster_id is None else job.cluster_id,
                    "cell_count": job.expected_cell_count,
                    "completed_epochs": report["selection"]["completed_epochs"],
                    "best_epoch": report["selection"]["best_epoch"],
                    "best_scaled_validation_mae": report["selection"][
                        "best_scaled_validation_mae"
                    ],
                    "restored_scaled_validation_mae": report["selection"][
                        "restored_scaled_validation_mae"
                    ],
                    "stopped_before_max_epochs": report["selection"][
                        "stopped_before_max_epochs"
                    ],
                    "fit_seconds": manifest["runtime"]["fit_seconds"],
                    "peak_rss_bytes": manifest["runtime"]["peak_rss_bytes"],
                    "checkpoint_sha256": manifest["outputs"]["best_weights"]["sha256"],
                    "prediction_sha256": manifest["outputs"]["validation_predictions"][
                        "sha256"
                    ],
                    "run_manifest_sha256": compute_sha256(
                        completed["paths"]["run_manifest"]
                    ),
                }
            )


def _write_per_cell_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "seed",
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


def aggregate_lstm_full_validation(config: LstmFullTrainingConfig) -> dict[str, Any]:
    """9개 immutable job을 검증하고 Test 없는 Validation release를 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    source_git_now = _git_state()
    if (
        config.pass_criteria.require_clean_source_git
        and source_git_now.get("dirty") is not False
    ):
        raise LstmFullContractError(
            "Validation 집계는 clean Git commit에서만 실행합니다."
        )
    source_arrays, input_manifest = load_and_verify_full_bundle(config)
    completed_jobs = [
        load_and_verify_completed_job(config, job, source_arrays, input_manifest)
        for job in config.jobs
    ]
    if len(completed_jobs) != config.training.expected_job_count:
        raise LstmFullContractError("9개 job이 모두 완료되지 않았습니다.")

    completed_by_id = {item["job"].job_id: item for item in completed_jobs}
    memberships = np.asarray(source_arrays["memberships"], dtype=np.int8)
    predictions_by_seed: dict[int, dict[str, np.ndarray]] = {}
    recombination: dict[str, Any] = {}
    prediction_output: dict[str, np.ndarray] = {
        "cell_ids": np.asarray(source_arrays["cell_ids"], dtype=np.int32),
        "memberships": memberships,
        "target_indices_validation": np.asarray(
            source_arrays["target_indices_validation"], dtype=np.int64
        ),
        "raw_y_validation": np.asarray(
            source_arrays["traffic_train_validation"][
                :, source_arrays["target_indices_validation"], None
            ],
            dtype=np.float32,
        ),
        "target_missing_mask_validation": np.asarray(
            source_arrays["missing_mask_train_validation"][
                :, source_arrays["target_indices_validation"]
            ],
            dtype=bool,
        ),
        "target_internet_null_mask_validation": np.asarray(
            source_arrays["internet_null_mask_train_validation"][
                :, source_arrays["target_indices_validation"]
            ],
            dtype=bool,
        ),
    }
    for seed in config.training.seeds:
        off = completed_by_id[f"seed_{seed}_upc_off"]["predictions"][
            "prediction_raw_validation"
        ]
        cluster_predictions = {
            cluster_id: completed_by_id[f"seed_{seed}_upc_on_cluster_{cluster_id}"][
                "predictions"
            ]["prediction_raw_validation"]
            for cluster_id, _ in config.upc.expected_cluster_counts
        }
        on, report = recombine_cluster_predictions(
            memberships=memberships,
            predictions_by_cluster=cluster_predictions,
            expected_cluster_counts=config.upc.expected_cluster_counts,
        )
        predictions_by_seed[seed] = {
            "lstm_full_upc_off": np.asarray(off, dtype=np.float32),
            "lstm_full_upc_on": np.asarray(on, dtype=np.float32),
        }
        recombination[str(seed)] = report
        prediction_output[f"seed_{seed}_upc_off_validation"] = np.asarray(
            off, dtype=np.float32
        )
        prediction_output[f"seed_{seed}_upc_on_validation"] = np.asarray(
            on, dtype=np.float32
        )
    if any("test" in name.lower() for name in prediction_output):
        raise LstmFullContractError("집계 prediction output에 Test 배열이 있습니다.")

    metric_results, per_cell_rows = evaluate_seed_predictions(
        config, source_arrays, predictions_by_seed
    )
    seed_summary = summarize_seed_metrics(metric_results, config.training.seeds)
    metrics_finite = all(
        value is not None and np.isfinite(value)
        for row in metric_results
        for aggregation in config.validation.aggregations
        for value in row[aggregation].values()
    )
    gates = {
        "clean_aggregation_git": source_git_now.get("dirty") is False,
        "source_git_matches_prepared_bundle": source_git_now.get("commit")
        == input_manifest["git"]["commit"],
        "all_nine_jobs": len(completed_jobs) == config.training.expected_job_count,
        "all_job_gates": all(item["report"]["gates_passed"] for item in completed_jobs),
        "all_best_weights_restored": all(
            item["report"]["gates"]["best_weights_restored_exactly"]
            for item in completed_jobs
        ),
        "exact_cluster_recombination": all(
            value["exact"] for value in recombination.values()
        ),
        "finite_validation_metrics": bool(metrics_finite),
        "test_absent": bool(
            input_manifest["test_seal"]["test_arrays_present"] is False
            and all("test" not in name.lower() for name in prediction_output)
        ),
        "performance_not_used_as_gate": bool(
            not config.validation.performance_used_as_pipeline_gate
            and not config.pass_criteria.require_better_than_persistence
            and not config.pass_criteria.require_upc_improvement
        ),
    }
    if not all(gates.values()):
        raise LstmFullContractError(
            "Validation 집계 gate가 실패했습니다: "
            + json.dumps(gates, ensure_ascii=False, allow_nan=False)
        )

    validation_report = {
        "schema_version": 1,
        "status": "pass",
        "scope": "central 900 cells, all 720 Validation targets per cell",
        "paper_table_ii_directly_comparable": False,
        "model_contract": {
            "name": config.architecture.name,
            "parameter_count": config.architecture.expected_parameter_count,
            "author_implementation_confirmed": False,
            "seeds": list(config.training.seeds),
            "upc_protocol": config.upc.protocol,
        },
        "selection_contract": {
            "best_epoch_unit": config.training.early_stopping.monitor_domain,
            "checkpoint_selection": config.training.checkpoint_selection,
            "validation_targets_per_cell": config.data.split(
                "validation"
            ).targets_per_cell,
            "performance_used_as_pipeline_gate": False,
        },
        "metric_contract": {
            "implementation": "scripts.evaluate_naive_baselines shared metric functions",
            "unit": config.validation.report_unit,
            "target_policies": list(config.validation.target_policies),
            "aggregations": list(config.validation.aggregations),
            "mape_zero_handling": "eligible targets with y > 0 only",
            "negative_prediction_policy": "linear LSTM predictions retained without clipping",
            "seed_summary": config.validation.seed_summary,
        },
        "results": metric_results,
        "seed_summary": seed_summary,
        "recombination": recombination,
        "gates": gates,
        "gates_passed": True,
        "test_seal": {
            "policy": config.data.test_policy,
            "known_prior_exposure": config.data.known_prior_test_exposure,
            "future_claim": config.data.future_test_claim,
            "test_evaluated": False,
        },
    }

    output_paths = {
        "training_jobs_csv": config.outputs.training_jobs_csv,
        "validation_report": config.outputs.validation_report,
        "validation_predictions_npz": config.outputs.validation_predictions_npz,
        "validation_per_cell_metrics_csv": config.outputs.validation_per_cell_metrics_csv,
        "release_manifest": config.outputs.release_manifest,
        "aggregation_manifest": config.outputs.aggregation_manifest,
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: _temporary_path(path) for name, path in output_paths.items()
    }
    published = False
    try:
        _write_training_jobs_csv(temporary_paths["training_jobs_csv"], completed_jobs)
        _write_json(temporary_paths["validation_report"], validation_report)
        with temporary_paths["validation_predictions_npz"].open("wb") as handle:
            np.savez_compressed(handle, **prediction_output)
        _write_per_cell_csv(
            temporary_paths["validation_per_cell_metrics_csv"], per_cell_rows
        )
        primary_output_names = (
            "training_jobs_csv",
            "validation_report",
            "validation_predictions_npz",
            "validation_per_cell_metrics_csv",
        )
        primary_metadata = {
            name: _temporary_file_contract(output_paths[name], temporary_paths[name])
            for name in primary_output_names
        }
        primary_metadata["validation_predictions_npz"]["arrays"] = {
            name: _array_contract(value) for name, value in prediction_output.items()
        }
        release_manifest = {
            "schema_version": 1,
            "status": "ready_for_locked_test_evaluation",
            "git": input_manifest["git"],
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "input": {
                "npz_sha256": compute_sha256(config.outputs.input_npz),
                "manifest_sha256": compute_sha256(config.outputs.input_manifest),
            },
            "jobs": {
                item["job"].job_id: {
                    "descriptor_sha256": item["manifest"]["input"]["descriptor_sha256"],
                    "training_report_sha256": item["manifest"]["outputs"][
                        "training_report"
                    ]["sha256"],
                    "validation_predictions_sha256": item["manifest"]["outputs"][
                        "validation_predictions"
                    ]["sha256"],
                    "best_weights_sha256": item["manifest"]["outputs"]["best_weights"][
                        "sha256"
                    ],
                    "run_manifest_sha256": compute_sha256(
                        item["paths"]["run_manifest"]
                    ),
                }
                for item in completed_jobs
            },
            "validation_outputs": primary_metadata,
            "gates": gates,
            "test_seal": validation_report["test_seal"],
        }
        _write_json(temporary_paths["release_manifest"], release_manifest)
        release_metadata = _temporary_file_contract(
            output_paths["release_manifest"], temporary_paths["release_manifest"]
        )
        finished_at = datetime.now(timezone.utc)
        aggregation_manifest = {
            "schema_version": 1,
            "status": "pass",
            "tool": {
                "name": "scripts.aggregate_lstm_full_validation",
                "version": TOOL_VERSION,
            },
            "created_at_utc": finished_at.isoformat(),
            "git": input_manifest["git"],
            "aggregation_git": source_git_now,
            "config": release_manifest["config"],
            "input": release_manifest["input"],
            "jobs": {
                item["job"].job_id: {
                    "run_manifest_sha256": compute_sha256(
                        item["paths"]["run_manifest"]
                    ),
                    "fit_seconds": item["manifest"]["runtime"]["fit_seconds"],
                    "peak_rss_bytes": item["manifest"]["runtime"]["peak_rss_bytes"],
                }
                for item in completed_jobs
            },
            "gates": gates,
            "outputs": {**primary_metadata, "release_manifest": release_metadata},
            "runtime": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "peak_rss_bytes": _peak_rss_bytes(),
                "local_tensorflow_training": False,
            },
            "test_evaluated": False,
        }
        _write_json(temporary_paths["aggregation_manifest"], aggregation_manifest)
        for name in primary_output_names:
            os.replace(temporary_paths[name], output_paths[name])
        os.replace(
            temporary_paths["release_manifest"], output_paths["release_manifest"]
        )
        os.replace(
            temporary_paths["aggregation_manifest"],
            output_paths["aggregation_manifest"],
        )
        published = True
        return aggregation_manifest
    finally:
        if not published:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="9개 LSTM full job의 Validation 결과를 Test 없이 집계합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_lstm_full_config(args.config)
        manifest = aggregate_lstm_full_validation(config)
    except (LstmFullContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"LSTM full Validation 집계 상태: {manifest['status']}")
    print(f"release: {_display_path(config.outputs.release_manifest)}")
    print(f"Test 평가: {manifest['test_evaluated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_lstm_full_validation",
    "evaluate_seed_predictions",
    "load_and_verify_completed_job",
    "summarize_seed_metrics",
]
