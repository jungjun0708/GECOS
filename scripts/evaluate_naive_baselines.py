#!/usr/bin/env python3
"""GECOS 예측 계약 위에서 학습 없는 두 기준선을 평가한다."""

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_upc_initial_groups import (
    UpcInitialGroupError,
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
    DEFAULT_CONFIG,
    ForecastConfig,
    ForecastContractError,
    ForecastIndexContract,
    build_forecast_index_contract,
    load_forecast_config,
)

TOOL_VERSION = "1.0.0"
METRIC_SIGNIFICANT_DIGITS = 12


@dataclass(frozen=True)
class EvaluationScope:
    name: str
    cell_ids: np.ndarray
    positions: np.ndarray
    protocol: str


@dataclass
class PerCellMetricParts:
    """Chunk와 무관한 최종 집계를 위해 셀 단위 부분합을 순서대로 보존한다."""

    sample_count: list[np.ndarray] = field(default_factory=list)
    absolute_error_sum: list[np.ndarray] = field(default_factory=list)
    target_sum: list[np.ndarray] = field(default_factory=list)
    positive_target_count: list[np.ndarray] = field(default_factory=list)
    absolute_percentage_error_sum: list[np.ndarray] = field(default_factory=list)
    missing_target_count: list[np.ndarray] = field(default_factory=list)
    internet_null_target_count: list[np.ndarray] = field(default_factory=list)
    excluded_missing_target_count: list[np.ndarray] = field(default_factory=list)
    lag_source_missing_count: list[np.ndarray] = field(default_factory=list)
    lag_source_internet_null_count: list[np.ndarray] = field(default_factory=list)

    def append(self, values: Mapping[str, np.ndarray]) -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name).append(np.asarray(values[name]))

    def concatenate(self) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate(getattr(self, name))
            for name in self.__dataclass_fields__
        }


def _canonical_metric(value: float) -> float:
    """동일 지표의 미세한 SIMD 합산 차이를 출력 정밀도 밖으로 정규화한다."""

    return float(format(value, f".{METRIC_SIGNIFICANT_DIGITS}g"))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return _canonical_metric(numerator / denominator)


def compute_per_cell_metric_parts(
    targets: np.ndarray,
    predictions: np.ndarray,
    eligible: np.ndarray,
    missing_targets: np.ndarray,
    internet_null_targets: np.ndarray,
    lag_missing: np.ndarray,
    lag_internet_null: np.ndarray,
    *,
    require_nonnegative_predictions: bool = True,
) -> dict[str, np.ndarray]:
    """한 chunk의 셀별 지표 부분합을 float64로 계산한다."""

    arrays = (
        predictions,
        eligible,
        missing_targets,
        internet_null_targets,
        lag_missing,
        lag_internet_null,
    )
    if targets.ndim != 2 or any(value.shape != targets.shape for value in arrays):
        raise ForecastContractError("target, prediction과 mask shape가 서로 다릅니다.")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise ForecastContractError("target 또는 prediction에 NaN/무한대가 있습니다.")
    if np.any(targets < 0):
        raise ForecastContractError("target은 음수일 수 없습니다.")
    if require_nonnegative_predictions and np.any(predictions < 0):
        raise ForecastContractError("기준선 prediction은 음수일 수 없습니다.")
    if np.any(missing_targets & internet_null_targets):
        raise ForecastContractError("두 target 결측 mask가 겹칩니다.")

    target64 = np.asarray(targets, dtype=np.float64)
    prediction64 = np.asarray(predictions, dtype=np.float64)
    absolute_error = np.abs(target64 - prediction64)
    eligible_bool = np.asarray(eligible, dtype=bool)
    positive = eligible_bool & (target64 > 0)
    ape = np.zeros_like(absolute_error)
    np.divide(absolute_error, target64, out=ape, where=positive)
    excluded_missing = (~eligible_bool) & (missing_targets | internet_null_targets)
    return {
        "sample_count": eligible_bool.sum(axis=1, dtype=np.int64),
        "absolute_error_sum": np.where(eligible_bool, absolute_error, 0).sum(
            axis=1, dtype=np.float64
        ),
        "target_sum": np.where(eligible_bool, target64, 0).sum(
            axis=1, dtype=np.float64
        ),
        "positive_target_count": positive.sum(axis=1, dtype=np.int64),
        "absolute_percentage_error_sum": ape.sum(axis=1, dtype=np.float64),
        "missing_target_count": missing_targets.sum(axis=1, dtype=np.int64),
        "internet_null_target_count": internet_null_targets.sum(axis=1, dtype=np.int64),
        "excluded_missing_target_count": excluded_missing.sum(axis=1, dtype=np.int64),
        "lag_source_missing_count": lag_missing.sum(axis=1, dtype=np.int64),
        "lag_source_internet_null_count": lag_internet_null.sum(axis=1, dtype=np.int64),
    }


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return {
            "defined_cell_count": 0,
            "mean": None,
            "q25": None,
            "median": None,
            "q75": None,
        }
    return {
        "defined_cell_count": len(valid),
        "mean": _canonical_metric(np.mean(valid, dtype=np.float64)),
        "q25": _canonical_metric(np.quantile(valid, 0.25)),
        "median": _canonical_metric(np.quantile(valid, 0.5)),
        "q75": _canonical_metric(np.quantile(valid, 0.75)),
    }


def finalize_metric_parts(
    parts: PerCellMetricParts,
    *,
    scope: EvaluationScope,
    split: str,
    baseline: str,
    target_policy: str,
    target_count_per_cell: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """셀 부분합으로 micro, cell-macro와 셀별 행을 결정론적으로 만든다."""

    values = parts.concatenate()
    if any(len(value) != len(scope.cell_ids) for value in values.values()):
        raise ForecastContractError("셀별 지표 수가 scope의 cell 수와 다릅니다.")
    sample_count = values["sample_count"]
    positive_count = values["positive_target_count"]
    absolute_sum = values["absolute_error_sum"]
    target_sum = values["target_sum"]
    ape_sum = values["absolute_percentage_error_sum"]

    cell_mae = np.full(len(scope.cell_ids), np.nan, dtype=np.float64)
    cell_mape = np.full(len(scope.cell_ids), np.nan, dtype=np.float64)
    cell_wape = np.full(len(scope.cell_ids), np.nan, dtype=np.float64)
    np.divide(absolute_sum, sample_count, out=cell_mae, where=sample_count > 0)
    np.divide(ape_sum, positive_count, out=cell_mape, where=positive_count > 0)
    np.divide(absolute_sum, target_sum, out=cell_wape, where=target_sum > 0)

    total_samples = int(np.sum(sample_count, dtype=np.int64))
    total_positive = int(np.sum(positive_count, dtype=np.int64))
    total_absolute = float(np.sum(absolute_sum, dtype=np.float64))
    total_target = float(np.sum(target_sum, dtype=np.float64))
    total_ape = float(np.sum(ape_sum, dtype=np.float64))
    candidate_targets = len(scope.cell_ids) * target_count_per_cell
    excluded_zero = total_samples - total_positive
    mae_distribution = _distribution(cell_mae)
    mape_distribution = _distribution(cell_mape)
    wape_distribution = _distribution(cell_wape)
    summary = {
        "scope": scope.name,
        "scope_protocol": scope.protocol,
        "cell_count": len(scope.cell_ids),
        "split": split,
        "baseline": baseline,
        "target_policy": target_policy,
        "target_count_per_cell": target_count_per_cell,
        "candidate_target_count": candidate_targets,
        "eligible_target_count": total_samples,
        "excluded_missing_target_count": int(
            np.sum(values["excluded_missing_target_count"], dtype=np.int64)
        ),
        "missing_target_count": int(
            np.sum(values["missing_target_count"], dtype=np.int64)
        ),
        "internet_all_null_target_count": int(
            np.sum(values["internet_null_target_count"], dtype=np.int64)
        ),
        "lag_source_missing_count": int(
            np.sum(values["lag_source_missing_count"], dtype=np.int64)
        ),
        "lag_source_internet_all_null_count": int(
            np.sum(values["lag_source_internet_null_count"], dtype=np.int64)
        ),
        "positive_target_count_for_mape": total_positive,
        "zero_target_count_excluded_from_mape": excluded_zero,
        "micro": {
            "mae": _safe_ratio(total_absolute, total_samples),
            "mape_ratio": _safe_ratio(total_ape, total_positive),
            "mape_percent": (
                None
                if total_positive == 0
                else _canonical_metric(100 * total_ape / total_positive)
            ),
            "wape": _safe_ratio(total_absolute, total_target),
        },
        "cell_macro": {
            "mae": mae_distribution["mean"],
            "mape_ratio": mape_distribution["mean"],
            "mape_percent": (
                None
                if mape_distribution["mean"] is None
                else _canonical_metric(100 * mape_distribution["mean"])
            ),
            "wape": wape_distribution["mean"],
        },
        "cell_distribution": {
            "mae": mae_distribution,
            "mape_ratio": mape_distribution,
            "wape": wape_distribution,
        },
    }
    per_cell_rows = []
    for index, cell_id in enumerate(scope.cell_ids):
        per_cell_rows.append(
            {
                "scope": scope.name,
                "scope_protocol": scope.protocol,
                "split": split,
                "baseline": baseline,
                "target_policy": target_policy,
                "cell_id": int(cell_id),
                "candidate_target_count": target_count_per_cell,
                "eligible_target_count": int(sample_count[index]),
                "excluded_missing_target_count": int(
                    values["excluded_missing_target_count"][index]
                ),
                "positive_target_count_for_mape": int(positive_count[index]),
                "zero_target_count_excluded_from_mape": int(
                    sample_count[index] - positive_count[index]
                ),
                "missing_target_count": int(values["missing_target_count"][index]),
                "internet_all_null_target_count": int(
                    values["internet_null_target_count"][index]
                ),
                "lag_source_missing_count": int(
                    values["lag_source_missing_count"][index]
                ),
                "lag_source_internet_all_null_count": int(
                    values["lag_source_internet_null_count"][index]
                ),
                "mae": (
                    None
                    if np.isnan(cell_mae[index])
                    else _canonical_metric(cell_mae[index])
                ),
                "mape_ratio": (
                    None
                    if np.isnan(cell_mape[index])
                    else _canonical_metric(cell_mape[index])
                ),
                "mape_percent": (
                    None
                    if np.isnan(cell_mape[index])
                    else _canonical_metric(cell_mape[index] * 100)
                ),
                "wape": (
                    None
                    if np.isnan(cell_wape[index])
                    else _canonical_metric(cell_wape[index])
                ),
            }
        )
    return summary, per_cell_rows


def _scope_sha256(cell_ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(cell_ids, dtype="<i4").tobytes()).hexdigest()


def load_evaluation_inputs(
    config: ForecastConfig,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, EvaluationScope],
    ForecastIndexContract,
    dict[str, Any],
]:
    """기존 manifest 계약을 재검증하고 두 평가 scope를 구성한다."""

    upc_config = load_upc_config(config.upc_config_path)
    arrays, processed_validation = verify_processed_inputs(upc_config)
    central_rows, central_positions, central_validation = load_central_cells(
        upc_config, arrays["cell_ids"]
    )
    central_ids = np.asarray(
        [int(row["cell_id"]) for row in central_rows], dtype=np.int32
    )
    if not np.array_equal(arrays["cell_ids"][central_positions], central_ids):
        raise ForecastContractError("중앙 900셀 순서와 전체 행렬 위치가 다릅니다.")
    all_positions = np.arange(len(arrays["cell_ids"]), dtype=np.int64)
    scopes = {
        "central_900": EvaluationScope(
            name="central_900",
            cell_ids=central_ids,
            positions=central_positions,
            protocol="central-900-approximate",
        ),
        "all_10000": EvaluationScope(
            name="all_10000",
            cell_ids=np.asarray(arrays["cell_ids"], dtype=np.int32),
            positions=all_positions,
            protocol="all-preprocessed-cells",
        ),
    }
    if tuple(scopes) != config.scopes:
        raise ForecastContractError("평가 scope 순서가 config와 다릅니다.")
    index_contract = build_forecast_index_contract(
        arrays["timestamps_ms"],
        config,
        timezone_name=upc_config.timezone_name,
        interval_ms=upc_config.interval_ms,
    )
    validation = {
        "upc_config": {
            "path": _display_path(config.upc_config_path),
            "sha256": compute_sha256(config.upc_config_path),
        },
        "processed": processed_validation,
        "central_900": central_validation,
        "scopes": {
            name: {
                "cell_count": len(scope.cell_ids),
                "cell_ids_int32_sha256": _scope_sha256(scope.cell_ids),
                "protocol": scope.protocol,
            }
            for name, scope in scopes.items()
        },
    }
    return arrays, scopes, index_contract, validation


def _evaluate(
    config: ForecastConfig,
    arrays: Mapping[str, np.ndarray],
    scopes: Mapping[str, EvaluationScope],
    index_contract: ForecastIndexContract,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accumulators = {
        (scope, split, baseline.name, policy): PerCellMetricParts()
        for scope in config.scopes
        for split in config.evaluated_splits
        for baseline in config.baselines
        for policy in config.target_policies
    }

    for scope_name in config.scopes:
        scope = scopes[scope_name]
        for start in range(0, len(scope.positions), config.cell_chunk_size):
            stop = min(start + config.cell_chunk_size, len(scope.positions))
            positions = scope.positions[start:stop]
            traffic = np.asarray(arrays["traffic"][positions, :], dtype=np.float32)
            missing = np.asarray(arrays["missing_mask"][positions, :], dtype=bool)
            internet_null = np.asarray(
                arrays["internet_null_mask"][positions, :], dtype=bool
            )
            if not np.all(np.isfinite(traffic)) or np.any(traffic < 0):
                raise ForecastContractError(
                    f"{scope_name} traffic chunk에 NaN, 무한대 또는 음수가 있습니다."
                )
            if np.any(missing & internet_null):
                raise ForecastContractError(
                    f"{scope_name}의 두 입력 결측 mask가 겹칩니다."
                )

            for split_name in config.evaluated_splits:
                target_indices = index_contract.target_indices[split_name]
                targets = traffic[:, target_indices]
                missing_targets = missing[:, target_indices]
                null_targets = internet_null[:, target_indices]
                observed_targets = ~(missing_targets | null_targets)
                for baseline in config.baselines:
                    lag_indices = target_indices - baseline.lag_steps
                    predictions = traffic[:, lag_indices]
                    lag_missing = missing[:, lag_indices]
                    lag_null = internet_null[:, lag_indices]
                    for policy in config.target_policies:
                        eligible = (
                            np.ones_like(observed_targets)
                            if policy == "all_targets"
                            else observed_targets
                        )
                        parts = compute_per_cell_metric_parts(
                            targets,
                            predictions,
                            eligible,
                            missing_targets,
                            null_targets,
                            lag_missing,
                            lag_null,
                        )
                        accumulators[
                            (scope_name, split_name, baseline.name, policy)
                        ].append(parts)

    summaries: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    for scope_name in config.scopes:
        for split_name in config.evaluated_splits:
            target_count = len(index_contract.target_indices[split_name])
            for baseline in config.baselines:
                for policy in config.target_policies:
                    summary, cell_rows = finalize_metric_parts(
                        accumulators[(scope_name, split_name, baseline.name, policy)],
                        scope=scopes[scope_name],
                        split=split_name,
                        baseline=baseline.name,
                        target_policy=policy,
                        target_count_per_cell=target_count,
                    )
                    summaries.append(summary)
                    if split_name in config.per_cell_output_splits:
                        per_cell_rows.extend(cell_rows)
    return summaries, per_cell_rows


def _write_per_cell_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _estimated_chunk_working_set_bytes(
    cell_chunk_size: int,
    total_step_count: int,
    maximum_evaluation_steps: int,
) -> int:
    source_arrays = cell_chunk_size * total_step_count * (4 + 1 + 1)
    evaluation_arrays = cell_chunk_size * maximum_evaluation_steps * (4 * 2 + 8 * 3 + 6)
    return source_arrays + evaluation_arrays


def run_naive_baselines(config: ForecastConfig) -> dict[str, Any]:
    """입력을 검증하고 두 scope의 naive 기준선 결과를 원자적으로 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    arrays, scopes, index_contract, input_validation = load_evaluation_inputs(config)
    result_rows, per_cell_rows = _evaluate(config, arrays, scopes, index_contract)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": config.name,
        "forecast_contract": {
            "input_length": config.input_length,
            "horizon": config.horizon,
            "evaluation_mode": config.evaluation_mode,
            "split_assignment": "target local timestamp",
            "first_local_timestamp": index_contract.first_local_timestamp,
            "last_local_timestamp": index_contract.last_local_timestamp,
            "total_targets_per_cell": index_contract.total_targets_per_cell,
            "primary_split": config.primary_split,
            "paper_comparison_split": config.paper_comparison_split,
            "splits": index_contract.split_metadata,
            "test_history_policy": (
                "Past observed values from earlier validation/test timestamps may be "
                "used for rolling one-step prediction; no future value is used."
            ),
        },
        "baseline_contract": [
            {
                "name": baseline.name,
                "lag_steps": baseline.lag_steps,
                "description": baseline.description,
                "reported_in_paper_table_ii": False,
            }
            for baseline in config.baselines
        ],
        "metric_contract": {
            "primary_target_policy": config.primary_target_policy,
            "target_policies": list(config.target_policies),
            "all_targets": "filled zero values are retained",
            "observed_targets_only": (
                "targets flagged by missing_mask or internet_null_mask are excluded"
            ),
            "mae": "mean absolute error over eligible targets",
            "mape_ratio": "mean abs(y-yhat)/y over eligible targets with y > 0",
            "mape_percent": "100 * mape_ratio",
            "wape": "sum abs(y-yhat) / sum y over eligible targets",
            "micro": "all eligible cell-target pairs pooled",
            "cell_macro": "unweighted distribution across defined per-cell metrics",
            "serialization_precision": (
                f"{METRIC_SIGNIFICANT_DIGITS} significant digits; stabilizes "
                "outputs across equivalent cell chunk sizes"
            ),
        },
        "scopes": input_validation["scopes"],
        "results": result_rows,
    }

    output_paths = config.outputs.as_dict()
    temporary_paths = {key: _temporary_path(path) for key, path in output_paths.items()}
    for output in output_paths.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        temporary_paths["summary_json"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_per_cell_metrics(temporary_paths["per_cell_metrics_csv"], per_cell_rows)
        deterministic_keys = ["summary_json", "per_cell_metrics_csv"]
        output_metadata = {
            key: {
                "path": _display_path(output_paths[key]),
                "size_bytes": temporary_paths[key].stat().st_size,
                "sha256": compute_sha256(temporary_paths[key]),
            }
            for key in deterministic_keys
        }
        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "tool": {
                "name": "scripts.evaluate_naive_baselines",
                "version": TOOL_VERSION,
            },
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "inputs": input_validation,
            "forecast_contract": summary["forecast_contract"],
            "baseline_contract": summary["baseline_contract"],
            "metric_contract": summary["metric_contract"],
            "result_row_count": len(result_rows),
            "per_cell_result_row_count": len(per_cell_rows),
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
                        arrays["traffic"].shape[1],
                        max(
                            len(index_contract.target_indices[name])
                            for name in config.evaluated_splits
                        ),
                    )
                ),
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        }
        temporary_paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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


def _find_primary_results(
    summary: Mapping[str, Any], config: ForecastConfig
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in summary["results"]
        if row["split"] == config.primary_split
        and row["target_policy"] == config.primary_target_policy
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GECOS 시간 분할 계약으로 Persistence와 일간 naive를 평가합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"naive baseline config (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_forecast_config(args.config)
        manifest = run_naive_baselines(config)
        summary = json.loads(config.outputs.summary_json.read_text(encoding="utf-8"))
    except UpcInitialGroupError as exc:
        print(f"Naive 기준선 평가 실패: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"파일 시스템 오류: {exc}", file=sys.stderr)
        return 2

    print(
        "Naive 기준선 평가 완료: "
        f"result_rows={manifest['result_row_count']}, "
        f"per_cell_rows={manifest['per_cell_result_row_count']}"
    )
    for row in _find_primary_results(summary, config):
        micro = row["micro"]
        print(
            f"{row['scope']} {row['baseline']}: "
            f"MAE={micro['mae']:.6f}, "
            f"MAPE_ratio={micro['mape_ratio']:.6f}, "
            f"WAPE={micro['wape']:.6f}"
        )
    print(f"manifest={_display_path(config.outputs.manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
