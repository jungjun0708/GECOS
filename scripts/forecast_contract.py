#!/usr/bin/env python3
"""GECOS 예측 표본의 target 정렬, 시간 분할과 공통 계약을 정의한다."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from scripts.build_upc_initial_groups import (
    REPOSITORY_ROOT,
    UpcInitialGroupError,
    _parse_local_datetime,
    _require_bool,
    _require_int,
    _require_list,
    _require_mapping,
    _require_string,
    _resolve_path,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "naive_baselines_milan_nov2013.json"
PARTITION_NAMES = ("train", "validation", "test")
AUXILIARY_SPLIT_NAMES = ("paper_holdout_10d",)
EVALUATED_SPLIT_NAMES = ("validation", "test", "paper_holdout_10d")
BASELINE_NAMES = ("persistence", "daily_seasonal_naive")
SCOPE_NAMES = ("central_900", "all_10000")
TARGET_POLICY_NAMES = ("all_targets", "observed_targets_only")


class ForecastContractError(UpcInitialGroupError):
    """예측 표본, 분할 또는 기준선 설정 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class TimeRangeSpec:
    name: str
    start_local: datetime
    end_exclusive_local: datetime
    expected_targets_per_cell: int


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    lag_steps: int
    description: str


@dataclass(frozen=True)
class BaselineOutputPaths:
    summary_json: Path
    per_cell_metrics_csv: Path
    manifest: Path

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ForecastConfig:
    path: Path
    name: str
    upc_config_path: Path
    input_length: int
    horizon: int
    evaluation_mode: str
    partitions: dict[str, TimeRangeSpec]
    auxiliary_splits: dict[str, TimeRangeSpec]
    evaluated_splits: tuple[str, ...]
    primary_split: str
    paper_comparison_split: str
    expected_total_targets_per_cell: int
    baselines: tuple[BaselineSpec, ...]
    scopes: tuple[str, ...]
    primary_target_policy: str
    target_policies: tuple[str, ...]
    mape_positive_targets_only: bool
    report_mape_ratio_and_percent: bool
    report_micro_and_cell_macro: bool
    per_cell_output_splits: tuple[str, ...]
    cell_chunk_size: int
    outputs: BaselineOutputPaths


@dataclass(frozen=True)
class ForecastIndexContract:
    target_indices: dict[str, np.ndarray]
    split_metadata: dict[str, dict[str, Any]]
    first_local_timestamp: str
    last_local_timestamp: str
    total_targets_per_cell: int


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ForecastContractError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ForecastContractError(f"{label}이 올바른 JSON이 아닙니다: {exc}") from exc
    return _require_mapping(value, label)


def _parse_time_range(raw: object, name: str, label: str) -> TimeRangeSpec:
    value = _require_mapping(raw, label)
    start = _parse_local_datetime(value.get("start_local"), f"{label}.start_local")
    end = _parse_local_datetime(
        value.get("end_exclusive_local"), f"{label}.end_exclusive_local"
    )
    if end <= start:
        raise ForecastContractError(
            f"{label}의 종료 시각은 시작 시각보다 뒤여야 합니다."
        )
    expected = _require_int(
        value.get("expected_targets_per_cell"),
        f"{label}.expected_targets_per_cell",
        minimum=1,
    )
    return TimeRangeSpec(
        name=name,
        start_local=start,
        end_exclusive_local=end,
        expected_targets_per_cell=expected,
    )


def load_forecast_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> ForecastConfig:
    """기준선 설정을 읽고 평가 계약의 상호 모순을 검사한다."""

    root = _load_json(path, "forecast baseline config")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise ForecastContractError(
            f"지원하지 않는 schema_version입니다: {schema_version}"
        )
    name = _require_string(root.get("name"), "name")
    upc_config_path = _resolve_path(
        root.get("upc_config"), "upc_config", base_directory
    )

    forecast = _require_mapping(root.get("forecast"), "forecast")
    input_length = _require_int(
        forecast.get("input_length"), "forecast.input_length", minimum=1
    )
    if input_length != 8:
        raise ForecastContractError(
            "이번 GECOS 계약의 forecast.input_length는 8이어야 합니다."
        )
    horizon = _require_int(forecast.get("horizon"), "forecast.horizon", minimum=1)
    if horizon != 1:
        raise ForecastContractError("이번 계약의 forecast.horizon은 1이어야 합니다.")
    evaluation_mode = _require_string(
        forecast.get("evaluation_mode"), "forecast.evaluation_mode"
    )
    if evaluation_mode != "rolling_one_step_with_observed_history":
        raise ForecastContractError(
            "forecast.evaluation_mode는 rolling_one_step_with_observed_history여야 "
            "합니다."
        )

    raw_partitions = _require_mapping(forecast.get("partitions"), "forecast.partitions")
    if set(raw_partitions) != set(PARTITION_NAMES):
        raise ForecastContractError(
            f"forecast.partitions에는 {list(PARTITION_NAMES)}만 있어야 합니다."
        )
    partitions = {
        name: _parse_time_range(
            raw_partitions[name], name, f"forecast.partitions.{name}"
        )
        for name in PARTITION_NAMES
    }
    if any(
        partitions[left].end_exclusive_local != partitions[right].start_local
        for left, right in pairwise(PARTITION_NAMES)
    ):
        raise ForecastContractError(
            "Train, Validation, Test 기간은 빈틈없이 이어져야 합니다."
        )

    raw_auxiliary = _require_mapping(
        forecast.get("auxiliary_splits"), "forecast.auxiliary_splits"
    )
    if set(raw_auxiliary) != set(AUXILIARY_SPLIT_NAMES):
        raise ForecastContractError(
            "forecast.auxiliary_splits에는 paper_holdout_10d만 있어야 합니다."
        )
    auxiliary_splits = {
        name: _parse_time_range(
            raw_auxiliary[name], name, f"forecast.auxiliary_splits.{name}"
        )
        for name in AUXILIARY_SPLIT_NAMES
    }
    paper_holdout = auxiliary_splits["paper_holdout_10d"]
    if (
        paper_holdout.start_local != partitions["validation"].start_local
        or paper_holdout.end_exclusive_local != partitions["test"].end_exclusive_local
    ):
        raise ForecastContractError(
            "paper_holdout_10d는 Validation 시작부터 Test 종료까지여야 합니다."
        )

    evaluated_splits = tuple(
        _require_string(value, f"forecast.evaluated_splits[{index}]")
        for index, value in enumerate(
            _require_list(forecast.get("evaluated_splits"), "forecast.evaluated_splits")
        )
    )
    if evaluated_splits != EVALUATED_SPLIT_NAMES:
        raise ForecastContractError(
            f"forecast.evaluated_splits는 {list(EVALUATED_SPLIT_NAMES)}여야 합니다."
        )
    primary_split = _require_string(
        forecast.get("primary_split"), "forecast.primary_split"
    )
    paper_comparison_split = _require_string(
        forecast.get("paper_comparison_split"),
        "forecast.paper_comparison_split",
    )
    if primary_split != "test" or paper_comparison_split != "paper_holdout_10d":
        raise ForecastContractError(
            "주 결과는 test, 논문 비교 보조값은 paper_holdout_10d여야 합니다."
        )
    expected_total = _require_int(
        forecast.get("expected_total_targets_per_cell"),
        "forecast.expected_total_targets_per_cell",
        minimum=1,
    )

    raw_baselines = _require_list(root.get("baselines"), "baselines")
    parsed_baselines: list[BaselineSpec] = []
    for index, item in enumerate(raw_baselines):
        value = _require_mapping(item, f"baselines[{index}]")
        parsed_baselines.append(
            BaselineSpec(
                name=_require_string(value.get("name"), f"baselines[{index}].name"),
                lag_steps=_require_int(
                    value.get("lag_steps"),
                    f"baselines[{index}].lag_steps",
                    minimum=1,
                ),
                description=_require_string(
                    value.get("description"), f"baselines[{index}].description"
                ),
            )
        )
    baselines = tuple(parsed_baselines)
    if tuple(item.name for item in baselines) != BASELINE_NAMES:
        raise ForecastContractError(
            f"baselines 순서는 {list(BASELINE_NAMES)}여야 합니다."
        )
    expected_lags = {"persistence": 1, "daily_seasonal_naive": 144}
    if any(item.lag_steps != expected_lags[item.name] for item in baselines):
        raise ForecastContractError("기준선 lag는 persistence=1, daily=144여야 합니다.")

    scopes = tuple(
        _require_string(value, f"scopes[{index}]")
        for index, value in enumerate(_require_list(root.get("scopes"), "scopes"))
    )
    if scopes != SCOPE_NAMES:
        raise ForecastContractError(f"scopes는 {list(SCOPE_NAMES)}여야 합니다.")

    metrics = _require_mapping(root.get("metrics"), "metrics")
    primary_target_policy = _require_string(
        metrics.get("primary_target_policy"), "metrics.primary_target_policy"
    )
    if primary_target_policy != "all_targets":
        raise ForecastContractError(
            "metrics.primary_target_policy는 all_targets여야 합니다."
        )
    target_policies = tuple(
        _require_string(value, f"metrics.target_policies[{index}]")
        for index, value in enumerate(
            _require_list(metrics.get("target_policies"), "metrics.target_policies")
        )
    )
    if target_policies != TARGET_POLICY_NAMES:
        raise ForecastContractError(
            f"metrics.target_policies는 {list(TARGET_POLICY_NAMES)}여야 합니다."
        )
    mape_positive = _require_bool(
        metrics.get("mape_positive_targets_only"),
        "metrics.mape_positive_targets_only",
    )
    report_mape_units = _require_bool(
        metrics.get("report_mape_ratio_and_percent"),
        "metrics.report_mape_ratio_and_percent",
    )
    report_averages = _require_bool(
        metrics.get("report_micro_and_cell_macro"),
        "metrics.report_micro_and_cell_macro",
    )
    if not (mape_positive and report_mape_units and report_averages):
        raise ForecastContractError("세 metrics 안전 계약은 모두 true여야 합니다.")
    per_cell_output_splits = tuple(
        _require_string(value, f"metrics.per_cell_output_splits[{index}]")
        for index, value in enumerate(
            _require_list(
                metrics.get("per_cell_output_splits"),
                "metrics.per_cell_output_splits",
            )
        )
    )
    if per_cell_output_splits != ("test",):
        raise ForecastContractError("셀별 지표 출력 split은 test만 허용합니다.")

    execution = _require_mapping(root.get("execution"), "execution")
    cell_chunk_size = _require_int(
        execution.get("cell_chunk_size"),
        "execution.cell_chunk_size",
        minimum=1,
    )
    raw_outputs = _require_mapping(root.get("outputs"), "outputs")
    outputs = BaselineOutputPaths(
        **{
            field: _resolve_path(
                raw_outputs.get(field), f"outputs.{field}", base_directory
            )
            for field in BaselineOutputPaths.__dataclass_fields__
        }
    )
    if len(set(outputs.as_dict().values())) != len(outputs.as_dict()):
        raise ForecastContractError("outputs 경로는 서로 달라야 합니다.")

    return ForecastConfig(
        path=path.resolve(),
        name=name,
        upc_config_path=upc_config_path,
        input_length=input_length,
        horizon=horizon,
        evaluation_mode=evaluation_mode,
        partitions=partitions,
        auxiliary_splits=auxiliary_splits,
        evaluated_splits=evaluated_splits,
        primary_split=primary_split,
        paper_comparison_split=paper_comparison_split,
        expected_total_targets_per_cell=expected_total,
        baselines=baselines,
        scopes=scopes,
        primary_target_policy=primary_target_policy,
        target_policies=target_policies,
        mape_positive_targets_only=mape_positive,
        report_mape_ratio_and_percent=report_mape_units,
        report_micro_and_cell_macro=report_averages,
        per_cell_output_splits=per_cell_output_splits,
        cell_chunk_size=cell_chunk_size,
        outputs=outputs,
    )


def _indices_for_range(
    local_times: tuple[datetime, ...],
    spec: TimeRangeSpec,
    *,
    first_eligible_target: int,
) -> np.ndarray:
    values = [
        index
        for index, value in enumerate(local_times)
        if index >= first_eligible_target
        and spec.start_local <= value.replace(tzinfo=None) < spec.end_exclusive_local
    ]
    return np.asarray(values, dtype=np.int64)


def build_forecast_index_contract(
    timestamps_ms: np.ndarray,
    config: ForecastConfig,
    *,
    timezone_name: str,
    interval_ms: int,
) -> ForecastIndexContract:
    """과거 window와 다음 target이 겹치지 않는 target-index split을 만든다."""

    if timestamps_ms.ndim != 1 or len(timestamps_ms) <= config.input_length:
        raise ForecastContractError(
            "timestamp 축이 예측 window를 만들기에 너무 짧습니다."
        )
    if not np.all(np.diff(timestamps_ms) == interval_ms):
        raise ForecastContractError(
            "timestamp 간격이 config와 다르거나 일정하지 않습니다."
        )
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ForecastContractError(
            f"알 수 없는 timezone입니다: {timezone_name}"
        ) from exc
    local_times = tuple(
        datetime.fromtimestamp(int(timestamp) / 1000, tz=zone)
        for timestamp in timestamps_ms
    )
    first_eligible_target = config.input_length + config.horizon - 1
    if first_eligible_target != config.input_length:
        raise ForecastContractError("현재 구현은 horizon=1 계약만 지원합니다.")

    targets: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    all_ranges = {**config.partitions, **config.auxiliary_splits}
    for name, spec in all_ranges.items():
        indices = _indices_for_range(
            local_times,
            spec,
            first_eligible_target=first_eligible_target,
        )
        if len(indices) != spec.expected_targets_per_cell:
            raise ForecastContractError(
                f"{name} target 수가 config와 다릅니다: "
                f"{len(indices)} != {spec.expected_targets_per_cell}"
            )
        input_start = indices - config.input_length
        input_end = indices - config.horizon
        if np.any(input_start < 0) or np.any(input_end >= indices):
            raise ForecastContractError(f"{name} window에 미래 target이 포함됩니다.")
        targets[name] = indices
        metadata[name] = {
            "target_count_per_cell": len(indices),
            "first_target_index": int(indices[0]),
            "last_target_index": int(indices[-1]),
            "first_target_local": local_times[int(indices[0])].isoformat(),
            "last_target_local": local_times[int(indices[-1])].isoformat(),
            "first_input_index": int(input_start[0]),
            "last_input_index_for_first_target": int(input_end[0]),
            "target_assignment_basis": "target local timestamp",
        }

    core_targets = np.concatenate([targets[name] for name in PARTITION_NAMES])
    expected_core = np.arange(first_eligible_target, len(timestamps_ms), dtype=np.int64)
    if not np.array_equal(core_targets, expected_core):
        raise ForecastContractError(
            "Train, Validation, Test target이 전체 eligible target을 정확히 분할하지 "
            "않습니다."
        )
    if len(core_targets) != config.expected_total_targets_per_cell:
        raise ForecastContractError(
            "전체 target 수가 config와 다릅니다: "
            f"{len(core_targets)} != {config.expected_total_targets_per_cell}"
        )
    expected_holdout = np.concatenate([targets["validation"], targets["test"]])
    if not np.array_equal(targets["paper_holdout_10d"], expected_holdout):
        raise ForecastContractError(
            "paper_holdout_10d target은 Validation과 Test의 결합이어야 합니다."
        )
    for baseline in config.baselines:
        for split_name in config.evaluated_splits:
            if int(targets[split_name][0]) - baseline.lag_steps < 0:
                raise ForecastContractError(
                    f"{baseline.name}은 {split_name} 첫 target의 과거 값을 참조할 수 "
                    "없습니다."
                )

    return ForecastIndexContract(
        target_indices=targets,
        split_metadata=metadata,
        first_local_timestamp=local_times[0].isoformat(),
        last_local_timestamp=local_times[-1].isoformat(),
        total_targets_per_cell=len(core_targets),
    )
