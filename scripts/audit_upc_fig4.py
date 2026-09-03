#!/usr/bin/env python3
"""논문 Fig. 4 불일치를 사전 고정한 유한한 변형만으로 감사한다."""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_upc_initial_groups import (
    HOURS_PER_DAY,
    REPOSITORY_ROOT,
    ProtocolHours,
    ProtocolSpec,
    UpcInitialGroupError,
    _display_path,
    _git_state,
    _parse_local_datetime,
    _peak_rss_bytes,
    _require_bool,
    _require_int,
    _require_list,
    _require_mapping,
    _require_string,
    _resolve_path,
    _temporary_path,
    build_local_time_axis,
    compute_sha256,
    load_upc_config,
    select_protocol_hours,
    verify_processed_inputs,
)

TOOL_VERSION = "1.0.0"
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "upc_fig4_audit_milan_nov2013.json"

REFERENCE_VARIANT = "algorithm1_full_month_sum_daily_mode_f64_zero"
DIAGNOSTIC_VARIANT = "figure4_probe_complete_weeks_sum_mean_profile_f64_zero"
AUDIT_VARIANT_NAMES = (
    REFERENCE_VARIANT,
    "full_month_sum_mean_profile_f64_zero",
    "full_month_max_daily_mode_f64_zero",
    "full_month_max_mean_profile_f64_zero",
    "complete_weeks_sum_daily_mode_f64_zero",
    DIAGNOSTIC_VARIANT,
    "complete_weeks_max_daily_mode_f64_zero",
    "complete_weeks_max_mean_profile_f64_zero",
    "figure4_probe_complete_weeks_sum_mean_profile_f32_zero",
    "figure4_probe_complete_weeks_sum_mean_profile_f64_exclude",
)
FULL_MONTH_START = "2013-11-01T00:00:00"
FULL_MONTH_END = "2013-12-01T00:00:00"
COMPLETE_WEEKS_START = "2013-11-04T00:00:00"
COMPLETE_WEEKS_END = "2013-11-30T00:00:00"
EXPECTED_VARIANT_CONTRACTS = (
    (
        AUDIT_VARIANT_NAMES[0],
        FULL_MONTH_START,
        FULL_MONTH_END,
        "sum",
        "daily_peak_mode",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[1],
        FULL_MONTH_START,
        FULL_MONTH_END,
        "sum",
        "mean_hourly_profile",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[2],
        FULL_MONTH_START,
        FULL_MONTH_END,
        "max",
        "daily_peak_mode",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[3],
        FULL_MONTH_START,
        FULL_MONTH_END,
        "max",
        "mean_hourly_profile",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[4],
        COMPLETE_WEEKS_START,
        COMPLETE_WEEKS_END,
        "sum",
        "daily_peak_mode",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[5],
        COMPLETE_WEEKS_START,
        COMPLETE_WEEKS_END,
        "sum",
        "mean_hourly_profile",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[6],
        COMPLETE_WEEKS_START,
        COMPLETE_WEEKS_END,
        "max",
        "daily_peak_mode",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[7],
        COMPLETE_WEEKS_START,
        COMPLETE_WEEKS_END,
        "max",
        "mean_hourly_profile",
        "zero_filled",
        "float64",
    ),
    (
        AUDIT_VARIANT_NAMES[8],
        COMPLETE_WEEKS_START,
        COMPLETE_WEEKS_END,
        "sum",
        "mean_hourly_profile",
        "zero_filled",
        "float32",
    ),
    (
        AUDIT_VARIANT_NAMES[9],
        COMPLETE_WEEKS_START,
        COMPLETE_WEEKS_END,
        "sum",
        "mean_hourly_profile",
        "exclude_flagged_and_renormalize",
        "float64",
    ),
)
ALLOWED_HOURLY_REDUCERS = {"sum", "max"}
ALLOWED_REPRESENTATIVE_METHODS = {"daily_peak_mode", "mean_hourly_profile"}
ALLOWED_MISSING_POLICIES = {
    "zero_filled",
    "exclude_flagged_and_renormalize",
}
ALLOWED_DTYPES = {"float32", "float64"}


class Fig4AuditError(UpcInitialGroupError):
    """Fig. 4 감사 계약이나 입력이 유효하지 않을 때 발생한다."""


@dataclass(frozen=True)
class AuditVariant:
    name: str
    start_local: datetime
    end_exclusive_local: datetime
    weekdays_only: bool
    hourly_reducer: str
    representative_method: str
    missing_policy: str
    calculation_dtype: str


@dataclass(frozen=True)
class AuditOutputPaths:
    report_json: Path
    comparison_csv: Path
    manifest: Path

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class Fig4AuditConfig:
    path: Path
    name: str
    upc_config_path: Path
    variants: tuple[AuditVariant, ...]
    outputs: AuditOutputPaths
    scope_contract: dict[str, Any]
    decision: dict[str, Any]


@dataclass(frozen=True)
class AuditVariantResult:
    peak_hours: np.ndarray
    diagnostics: dict[str, Any]


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Fig4AuditError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Fig4AuditError(f"{label}이 올바른 JSON이 아닙니다: {exc}") from exc
    return _require_mapping(value, label)


def _parse_variant(raw: object, index: int) -> AuditVariant:
    label = f"variants[{index}]"
    value = _require_mapping(raw, label)
    name = _require_string(value.get("name"), f"{label}.name")
    start_local = _parse_local_datetime(
        value.get("start_local"), f"{label}.start_local"
    )
    end_exclusive_local = _parse_local_datetime(
        value.get("end_exclusive_local"), f"{label}.end_exclusive_local"
    )
    if end_exclusive_local <= start_local:
        raise Fig4AuditError(f"{label}의 종료 시각은 시작 시각보다 뒤여야 합니다.")
    weekdays_only = _require_bool(value.get("weekdays_only"), f"{label}.weekdays_only")
    if not weekdays_only:
        raise Fig4AuditError(f"{label}은 논문 계약에 따라 평일만 사용해야 합니다.")

    choices = {
        "hourly_reducer": ALLOWED_HOURLY_REDUCERS,
        "representative_method": ALLOWED_REPRESENTATIVE_METHODS,
        "missing_policy": ALLOWED_MISSING_POLICIES,
        "calculation_dtype": ALLOWED_DTYPES,
    }
    parsed: dict[str, str] = {}
    for field, allowed in choices.items():
        parsed[field] = _require_string(value.get(field), f"{label}.{field}")
        if parsed[field] not in allowed:
            raise Fig4AuditError(
                f"{label}.{field}가 허용 목록에 없습니다: {parsed[field]}"
            )
    return AuditVariant(
        name=name,
        start_local=start_local,
        end_exclusive_local=end_exclusive_local,
        weekdays_only=weekdays_only,
        **parsed,
    )


def load_fig4_audit_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> Fig4AuditConfig:
    """감사 config와 사전 등록 범위를 읽고 변경·확장을 차단한다."""

    root = _load_json(path, "Fig. 4 audit config")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise Fig4AuditError(f"지원하지 않는 schema_version입니다: {schema_version}")
    name = _require_string(root.get("name"), "name")
    upc_config_path = _resolve_path(
        root.get("upc_config"), "upc_config", base_directory
    )

    raw_scope = _require_mapping(root.get("scope_contract"), "scope_contract")
    required_scope = {
        "pre_registered_before_execution": True,
        "allow_post_result_variant_expansion": False,
        "stop_after_declared_variants": True,
    }
    for field, expected in required_scope.items():
        actual = _require_bool(raw_scope.get(field), f"scope_contract.{field}")
        if actual is not expected:
            raise Fig4AuditError(
                f"scope_contract.{field}는 {expected}로 고정되어야 합니다."
            )
    scope_contract = {
        **required_scope,
        "rationale": _require_string(
            raw_scope.get("rationale"), "scope_contract.rationale"
        ),
        "analytical_equivalence": _require_string(
            raw_scope.get("analytical_equivalence"),
            "scope_contract.analytical_equivalence",
        ),
    }

    variants = tuple(
        _parse_variant(value, index)
        for index, value in enumerate(_require_list(root.get("variants"), "variants"))
    )
    names = tuple(variant.name for variant in variants)
    if names != AUDIT_VARIANT_NAMES:
        raise Fig4AuditError(
            "감사 변형의 이름과 순서는 사전 등록 목록과 정확히 같아야 합니다: "
            f"{list(AUDIT_VARIANT_NAMES)}"
        )
    actual_contracts = tuple(
        (
            variant.name,
            variant.start_local.isoformat(),
            variant.end_exclusive_local.isoformat(),
            variant.hourly_reducer,
            variant.representative_method,
            variant.missing_policy,
            variant.calculation_dtype,
        )
        for variant in variants
    )
    if actual_contracts != EXPECTED_VARIANT_CONTRACTS:
        raise Fig4AuditError(
            "감사 변형의 기간과 계산 요인은 사전 등록 계약과 정확히 같아야 합니다."
        )

    raw_decision = _require_mapping(root.get("decision"), "decision")
    decision = {
        "primary_model_protocol": _require_string(
            raw_decision.get("primary_model_protocol"),
            "decision.primary_model_protocol",
        ),
        "sensitivity_model_protocol": _require_string(
            raw_decision.get("sensitivity_model_protocol"),
            "decision.sensitivity_model_protocol",
        ),
        "diagnostic_variant": _require_string(
            raw_decision.get("diagnostic_variant"),
            "decision.diagnostic_variant",
        ),
        "diagnostic_eligible_for_model_input": _require_bool(
            raw_decision.get("diagnostic_eligible_for_model_input"),
            "decision.diagnostic_eligible_for_model_input",
        ),
        "exact_figure4_match_required_for_baselines": _require_bool(
            raw_decision.get("exact_figure4_match_required_for_baselines"),
            "decision.exact_figure4_match_required_for_baselines",
        ),
        "exact_figure4_match_required_for_reproduction_claim": _require_bool(
            raw_decision.get("exact_figure4_match_required_for_reproduction_claim"),
            "decision.exact_figure4_match_required_for_reproduction_claim",
        ),
    }
    expected_decision = {
        "primary_model_protocol": "train_only",
        "sensitivity_model_protocol": "algorithm1_full_month",
        "diagnostic_variant": DIAGNOSTIC_VARIANT,
        "diagnostic_eligible_for_model_input": False,
        "exact_figure4_match_required_for_baselines": False,
        "exact_figure4_match_required_for_reproduction_claim": True,
    }
    if decision != expected_decision:
        raise Fig4AuditError(
            "decision은 학습 전 고정한 프로토콜 결정과 정확히 같아야 합니다."
        )

    raw_outputs = _require_mapping(root.get("outputs"), "outputs")
    outputs = AuditOutputPaths(
        **{
            field: _resolve_path(
                raw_outputs.get(field), f"outputs.{field}", base_directory
            )
            for field in AuditOutputPaths.__dataclass_fields__
        }
    )
    if len(set(outputs.as_dict().values())) != len(outputs.as_dict()):
        raise Fig4AuditError("outputs 경로는 서로 달라야 합니다.")
    return Fig4AuditConfig(
        path=path.resolve(),
        name=name,
        upc_config_path=upc_config_path,
        variants=variants,
        outputs=outputs,
        scope_contract=scope_contract,
        decision=decision,
    )


def _mode_peak_hours(daily_peaks: np.ndarray) -> tuple[np.ndarray, int]:
    hours = np.arange(HOURS_PER_DAY, dtype=np.int8)
    counts = np.count_nonzero(daily_peaks[:, :, None] == hours, axis=1)
    maxima = counts.max(axis=1, keepdims=True)
    tie_count = int(np.count_nonzero(np.count_nonzero(counts == maxima, axis=1) > 1))
    return np.argmax(counts, axis=1).astype(np.int8), tie_count


def _hourly_values(
    raw: np.ndarray,
    valid: np.ndarray,
    *,
    hourly_reducer: str,
    missing_policy: str,
    calculation_dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """raw/scaled 시간값과 유효 hour를 같은 정책으로 계산한다."""

    axes = (1, 2, 3)
    if missing_policy == "zero_filled":
        minimum = raw.min(axis=axes, keepdims=True)
        maximum = raw.max(axis=axes, keepdims=True)
        no_valid_cell_count = 0
    else:
        minimum = np.min(np.where(valid, raw, np.inf), axis=axes, keepdims=True)
        maximum = np.max(np.where(valid, raw, -np.inf), axis=axes, keepdims=True)
        no_valid = ~np.any(valid, axis=axes, keepdims=True)
        no_valid_cell_count = int(np.count_nonzero(no_valid))
        minimum = np.where(no_valid, 0, minimum).astype(calculation_dtype)
        maximum = np.where(no_valid, 0, maximum).astype(calculation_dtype)
    span = maximum - minimum
    constant_cell_count = int(np.count_nonzero(span.reshape(-1) == 0))

    if missing_policy == "zero_filled":
        hour_valid = np.ones(raw.shape[:3], dtype=bool)
        if hourly_reducer == "sum":
            raw_hourly = raw.sum(axis=3, dtype=calculation_dtype)
            numerator = raw_hourly - raw.shape[3] * minimum.reshape(-1, 1, 1)
        else:
            raw_hourly = raw.max(axis=3)
            numerator = raw_hourly - minimum.reshape(-1, 1, 1)
    else:
        valid_count = valid.sum(axis=3)
        hour_valid = valid_count > 0
        if hourly_reducer == "sum":
            raw_sum = np.where(valid, raw, 0).sum(axis=3, dtype=calculation_dtype)
            raw_hourly = np.zeros_like(raw_sum)
            np.divide(
                raw_sum * raw.shape[3],
                valid_count,
                out=raw_hourly,
                where=hour_valid,
            )
            numerator = raw_hourly - raw.shape[3] * minimum.reshape(-1, 1, 1)
        else:
            raw_hourly = np.max(np.where(valid, raw, -np.inf), axis=3)
            raw_hourly = np.where(hour_valid, raw_hourly, 0)
            numerator = raw_hourly - minimum.reshape(-1, 1, 1)

    scaled_hourly = np.zeros_like(raw_hourly)
    nonconstant = span.reshape(-1, 1, 1) != 0
    np.divide(
        numerator,
        span.reshape(-1, 1, 1),
        out=scaled_hourly,
        where=hour_valid & nonconstant,
    )
    return (
        raw_hourly,
        scaled_hourly,
        hour_valid,
        constant_cell_count,
        no_valid_cell_count,
    )


def _representative_from_hourly(
    raw_hourly: np.ndarray,
    scaled_hourly: np.ndarray,
    hour_valid: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    invalid_day_count = int(np.count_nonzero(~np.any(hour_valid, axis=2)))
    raw_scores = np.where(hour_valid, raw_hourly, -np.inf)
    scaled_scores = np.where(hour_valid, scaled_hourly, -np.inf)

    if method == "daily_peak_mode":
        raw_daily = np.argmax(raw_scores, axis=2).astype(np.int8)
        scaled_daily = np.argmax(scaled_scores, axis=2).astype(np.int8)
        raw_peak, _ = _mode_peak_hours(raw_daily)
        scaled_peak, representative_ties = _mode_peak_hours(scaled_daily)
        maxima = scaled_scores.max(axis=2, keepdims=True)
        peak_tie_count = (
            int(
                np.count_nonzero(
                    np.count_nonzero(
                        (scaled_scores == maxima) & hour_valid,
                        axis=2,
                    )
                    > 1
                )
            )
            + representative_ties
        )
        return raw_peak, scaled_peak, peak_tie_count, invalid_day_count

    valid_count = hour_valid.sum(axis=1)
    profile_valid = valid_count > 0
    profile_shape = (raw_hourly.shape[0], raw_hourly.shape[2])
    raw_profile = np.zeros(profile_shape, dtype=raw_hourly.dtype)
    scaled_profile = np.zeros_like(raw_profile)
    np.divide(
        np.where(hour_valid, raw_hourly, 0).sum(axis=1, dtype=raw_hourly.dtype),
        valid_count,
        out=raw_profile,
        where=profile_valid,
    )
    np.divide(
        np.where(hour_valid, scaled_hourly, 0).sum(axis=1, dtype=scaled_hourly.dtype),
        valid_count,
        out=scaled_profile,
        where=profile_valid,
    )
    raw_profile = np.where(profile_valid, raw_profile, -np.inf)
    scaled_profile = np.where(profile_valid, scaled_profile, -np.inf)
    raw_peak = np.argmax(raw_profile, axis=1).astype(np.int8)
    scaled_peak = np.argmax(scaled_profile, axis=1).astype(np.int8)
    maxima = scaled_profile.max(axis=1, keepdims=True)
    peak_tie_count = int(
        np.count_nonzero(
            np.count_nonzero(
                (scaled_profile == maxima) & profile_valid,
                axis=1,
            )
            > 1
        )
    )
    return raw_peak, scaled_peak, peak_tie_count, invalid_day_count


def compute_audit_variant(
    traffic: np.ndarray,
    missing_mask: np.ndarray,
    internet_null_mask: np.ndarray,
    protocol_hours: ProtocolHours,
    variant: AuditVariant,
    *,
    cell_chunk_size: int,
) -> AuditVariantResult:
    """한 개의 사전 등록 변형을 chunk 단위로 계산한다."""

    if traffic.shape != missing_mask.shape or traffic.shape != internet_null_mask.shape:
        raise Fig4AuditError("traffic과 결측 mask shape가 서로 다릅니다.")
    if traffic.ndim != 2 or cell_chunk_size < 1:
        raise Fig4AuditError(
            "traffic은 2차원이고 cell_chunk_size는 1 이상이어야 합니다."
        )

    calculation_dtype = np.dtype(variant.calculation_dtype)
    flat_indices = protocol_hours.indices.reshape(-1)
    day_count = protocol_hours.indices.shape[0]
    observations_per_hour = protocol_hours.indices.shape[2]
    peak_hours = np.empty(traffic.shape[0], dtype=np.int8)
    constant_cell_count = 0
    no_valid_cell_count = 0
    invalid_cell_day_count = 0
    peak_tie_count = 0
    scaling_mismatch_count = 0
    missing_pair_count = 0
    internet_null_pair_count = 0

    for start in range(0, traffic.shape[0], cell_chunk_size):
        stop = min(start + cell_chunk_size, traffic.shape[0])
        shape = (
            stop - start,
            day_count,
            HOURS_PER_DAY,
            observations_per_hour,
        )
        raw = np.asarray(
            traffic[start:stop, flat_indices], dtype=calculation_dtype
        ).reshape(shape)
        missing = np.asarray(missing_mask[start:stop, flat_indices]).reshape(shape)
        internet_null = np.asarray(
            internet_null_mask[start:stop, flat_indices]
        ).reshape(shape)
        valid = ~(missing | internet_null)
        (
            raw_hourly,
            scaled_hourly,
            hour_valid,
            chunk_constant_count,
            chunk_no_valid_count,
        ) = _hourly_values(
            raw,
            valid,
            hourly_reducer=variant.hourly_reducer,
            missing_policy=variant.missing_policy,
            calculation_dtype=calculation_dtype,
        )
        raw_peak, scaled_peak, chunk_ties, chunk_invalid_days = (
            _representative_from_hourly(
                raw_hourly,
                scaled_hourly,
                hour_valid,
                variant.representative_method,
            )
        )
        peak_hours[start:stop] = scaled_peak
        constant_cell_count += chunk_constant_count
        no_valid_cell_count += chunk_no_valid_count
        invalid_cell_day_count += chunk_invalid_days
        peak_tie_count += chunk_ties
        scaling_mismatch_count += int(np.count_nonzero(raw_peak != scaled_peak))
        missing_pair_count += int(np.count_nonzero(missing))
        internet_null_pair_count += int(np.count_nonzero(internet_null))

    counts = np.bincount(peak_hours, minlength=HOURS_PER_DAY).astype(np.int64)
    if len(counts) != HOURS_PER_DAY or int(counts.sum()) != traffic.shape[0]:
        raise Fig4AuditError(
            "감사 변형에서 모든 cell이 24개 그룹에 배정되지 않았습니다."
        )
    selected_observations = traffic.shape[0] * len(flat_indices)
    diagnostics = {
        "variant": variant.name,
        "date_start": protocol_hours.dates[0].isoformat(),
        "date_end_inclusive": protocol_hours.dates[-1].isoformat(),
        "weekday_count": day_count,
        "weekdays": [value.isoformat() for value in protocol_hours.dates],
        "hourly_reducer": variant.hourly_reducer,
        "representative_method": variant.representative_method,
        "missing_policy": variant.missing_policy,
        "calculation_dtype": variant.calculation_dtype,
        "tie_break": "earliest hour",
        "constant_cell_count": constant_cell_count,
        "no_valid_cell_count": no_valid_cell_count,
        "invalid_cell_day_count": invalid_cell_day_count,
        "peak_tie_count": peak_tie_count,
        "scaling_invariance_mismatch_cell_count": scaling_mismatch_count,
        "missing_pair_count": missing_pair_count,
        "internet_all_null_pair_count": internet_null_pair_count,
        "selected_observations": selected_observations,
        "group_counts_hour_0_to_23": counts.tolist(),
        "assigned_cell_count": int(counts.sum()),
        "membership_int8_sha256": hashlib.sha256(peak_hours.tobytes()).hexdigest(),
        "eligible_for_model_input": False,
    }
    return AuditVariantResult(peak_hours=peak_hours, diagnostics=diagnostics)


def _score_variant(
    result: AuditVariantResult,
    expected: np.ndarray,
    reference_peaks: np.ndarray,
    diagnostic_peaks: np.ndarray,
) -> dict[str, Any]:
    actual = np.asarray(result.diagnostics["group_counts_hour_0_to_23"], dtype=np.int64)
    difference = actual - expected
    l1_difference = int(np.abs(difference).sum())
    return {
        **result.diagnostics,
        "expected_hour_0_to_23": expected.tolist(),
        "difference_actual_minus_expected": difference.tolist(),
        "matching_group_count_hours": int(np.count_nonzero(difference == 0)),
        "l1_difference": l1_difference,
        "minimum_cell_reassignments_lower_bound": (l1_difference + 1) // 2,
        "exact_match": bool(np.array_equal(actual, expected)),
        "membership_difference_from_algorithm1_full_month": int(
            np.count_nonzero(result.peak_hours != reference_peaks)
        ),
        "membership_difference_from_designated_figure4_probe": int(
            np.count_nonzero(result.peak_hours != diagnostic_peaks)
        ),
    }


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "variant",
        "weekday_count",
        "hourly_reducer",
        "representative_method",
        "missing_policy",
        "calculation_dtype",
        "matching_group_count_hours",
        "l1_difference",
        "minimum_cell_reassignments_lower_bound",
        "exact_match",
        "membership_difference_from_algorithm1_full_month",
        "membership_difference_from_designated_figure4_probe",
        "group_counts_hour_0_to_23",
        "difference_actual_minus_expected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field], separators=(",", ":"))
                    if isinstance(row[field], list)
                    else row[field]
                    for field in fields
                }
            )


def run_fig4_audit(config: Fig4AuditConfig) -> dict[str, Any]:
    """등록한 변형을 한 번씩 실행하고 결정론적 보고서와 manifest를 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    upc_config = load_upc_config(config.upc_config_path)
    arrays, processed_validation = verify_processed_inputs(upc_config)
    axis = build_local_time_axis(
        arrays["timestamps_ms"],
        timezone_name=upc_config.timezone_name,
        interval_ms=upc_config.interval_ms,
        observations_per_hour=upc_config.observations_per_hour,
    )

    raw_results: dict[str, AuditVariantResult] = {}
    for variant in config.variants:
        hours = select_protocol_hours(
            axis,
            ProtocolSpec(
                name=variant.name,
                start_local=variant.start_local,
                end_exclusive_local=variant.end_exclusive_local,
                weekdays_only=variant.weekdays_only,
            ),
        )
        raw_results[variant.name] = compute_audit_variant(
            arrays["traffic"],
            arrays["missing_mask"],
            arrays["internet_null_mask"],
            hours,
            variant,
            cell_chunk_size=upc_config.cell_chunk_size,
        )

    expected = np.asarray(upc_config.paper_fingerprint, dtype=np.int64)
    reference_peaks = raw_results[REFERENCE_VARIANT].peak_hours
    diagnostic_peaks = raw_results[DIAGNOSTIC_VARIANT].peak_hours
    scored = [
        _score_variant(
            raw_results[variant.name],
            expected,
            reference_peaks,
            diagnostic_peaks,
        )
        for variant in config.variants
    ]
    best_l1 = min(row["l1_difference"] for row in scored)
    exact_matches = [row["variant"] for row in scored if row["exact_match"]]
    report = {
        "schema_version": 1,
        "audit": config.name,
        "scope_contract": config.scope_contract,
        "expected_figure4_group_counts_hour_0_to_23": expected.tolist(),
        "reference_variant": REFERENCE_VARIANT,
        "designated_diagnostic_variant": DIAGNOSTIC_VARIANT,
        "variants": scored,
        "summary": {
            "executed_variant_count": len(scored),
            "best_l1_difference": best_l1,
            "best_variants": [
                row["variant"] for row in scored if row["l1_difference"] == best_l1
            ],
            "exact_match_found": bool(exact_matches),
            "exact_match_variants": exact_matches,
            "audit_stopped_after_declared_variants": True,
            "interpretation": (
                "An exact match would identify a compatible calculation, not prove "
                "the authors used it. A mismatch does not block independent baselines."
            ),
        },
        "decision": config.decision,
        "author_questions": [
            "Was 2013-11-01 excluded so that the analysis used four complete weeks?",
            "Was a mean hourly profile used instead of the mode of daily peak hours?",
            "How were six 10-minute observations reduced to each hour?",
            "How were missing and all-null Internet records treated?",
            "Can the UPC membership or Fig. 4 generation code be shared?",
        ],
    }

    output_paths = config.outputs.as_dict()
    temporary_paths = {key: _temporary_path(path) for key, path in output_paths.items()}
    for output in output_paths.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        temporary_paths["report_json"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_comparison_csv(temporary_paths["comparison_csv"], scored)
        deterministic_keys = ["report_json", "comparison_csv"]
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
            "tool": {"name": "scripts.audit_upc_fig4", "version": TOOL_VERSION},
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "upc_config": {
                "path": _display_path(config.upc_config_path),
                "sha256": compute_sha256(config.upc_config_path),
            },
            "inputs": processed_validation,
            "scope_contract": config.scope_contract,
            "summary": report["summary"],
            "decision": config.decision,
            "outputs": output_metadata,
            "runtime": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "cell_chunk_size": upc_config.cell_chunk_size,
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        }
        temporary_paths["manifest"].write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GECOS 논문 Fig. 4 불일치를 사전 등록한 변형으로 감사합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Fig. 4 audit config (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_fig4_audit_config(args.config)
        manifest = run_fig4_audit(config)
    except UpcInitialGroupError as exc:
        print(f"Fig. 4 감사 실패: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"파일 시스템 오류: {exc}", file=sys.stderr)
        return 2
    summary = manifest["summary"]
    print(
        "Fig. 4 제한 감사 완료: "
        f"variants={summary['executed_variant_count']}, "
        f"best_l1={summary['best_l1_difference']}, "
        f"exact_match={summary['exact_match_found']}"
    )
    print(f"report={_display_path(config.outputs.report_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
