#!/usr/bin/env python3
"""GECOS UPC의 PCC 기반 최종 N=2 클러스터를 결정론적으로 만든다."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from scripts.build_upc_initial_groups import (
    HOURS_PER_DAY,
    UpcInitialGroupError,
    _display_path,
    _git_state,
    _load_json,
    _peak_rss_bytes,
    _require_bool,
    _require_int,
    _require_mapping,
    _require_string,
    _temporary_path,
    _verify_file_metadata,
    build_local_time_axis,
    compute_sha256,
    load_central_cells,
    load_upc_config,
    select_protocol_hours,
    verify_processed_inputs,
)

TOOL_VERSION = "1.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "upc_pcc_milan_nov2013.json"
PROTOCOL_NAMES = ("train_only", "algorithm1_full_month")
PRIMARY_ORDER = "ascending_group_id"
SENSITIVITY_ORDER = "descending_group_id"


class UpcFinalClusterError(RuntimeError):
    """PCC 최종 군집 계약이나 입력 무결성이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class FinalClusterConfig:
    path: Path
    name: str
    upstream_config_path: Path
    upstream_config_sha256: str
    upstream_manifest_path: Path
    accepted_upstream_status: str
    primary_protocol: str
    sensitivity_protocol: str
    forbidden_protocol: str
    cluster_count: int
    theta: int
    pcc_tie_tolerance: float
    primary_order: str
    sensitivity_order: str
    paper_group_start: int
    paper_group_end: int
    minimum_order_agreement: float
    require_all_cells_assigned_once: bool
    require_all_central_cells_assigned_once: bool
    cell_chunk_size: int
    output_directory: Path


@dataclass(frozen=True)
class GroupProfileResult:
    profiles: np.ndarray
    valid: np.ndarray
    counts: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AssignmentResult:
    group_to_cluster: np.ndarray
    cluster_members: tuple[tuple[int, ...], ...]
    trace: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProtocolClusterResult:
    name: str
    group_profiles: np.ndarray
    profile_valid: np.ndarray
    nonempty_group_ids: np.ndarray
    pcc: np.ndarray
    group_counts: np.ndarray
    eligible_seed_group_ids: np.ndarray
    seed_pair: tuple[int, int]
    seed_pcc: float
    primary: AssignmentResult
    order_sensitivity: AssignmentResult
    diagnostics: dict[str, Any]


def _resolve_path(value: object, field: str, base_directory: Path) -> Path:
    raw = Path(_require_string(value, field))
    return raw.resolve() if raw.is_absolute() else (base_directory / raw).resolve()


def _require_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpcFinalClusterError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if not np.isfinite(result):
        raise UpcFinalClusterError(f"{field}는 유한한 숫자여야 합니다.")
    if minimum is not None and result < minimum:
        raise UpcFinalClusterError(f"{field}는 {minimum} 이상이어야 합니다.")
    if maximum is not None and result > maximum:
        raise UpcFinalClusterError(f"{field}는 {maximum} 이하여야 합니다.")
    return result


def _expect_literal(
    mapping: Mapping[str, Any], field: str, expected: object, prefix: str
) -> None:
    value = mapping.get(field)
    if value != expected:
        raise UpcFinalClusterError(
            f"{prefix}.{field}는 {expected!r}이어야 합니다: {value!r}"
        )


def load_final_cluster_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> FinalClusterConfig:
    """사전 등록된 PCC 군집 설정을 읽고 허용된 계약만 받아들인다."""

    try:
        root = _load_json(path, "UPC PCC config")
    except UpcInitialGroupError as exc:
        raise UpcFinalClusterError(str(exc)) from exc
    if _require_int(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise UpcFinalClusterError("지원하지 않는 schema_version입니다.")
    name = _require_string(root.get("name"), "name")

    inputs = _require_mapping(root.get("inputs"), "inputs")
    upstream_config_path = _resolve_path(
        inputs.get("upstream_upc_config"),
        "inputs.upstream_upc_config",
        base_directory,
    )
    upstream_config_sha256 = _require_string(
        inputs.get("upstream_upc_config_sha256"),
        "inputs.upstream_upc_config_sha256",
    )
    if len(upstream_config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in upstream_config_sha256
    ):
        raise UpcFinalClusterError("upstream config SHA-256 형식이 올바르지 않습니다.")
    upstream_manifest_path = _resolve_path(
        inputs.get("upstream_upc_manifest"),
        "inputs.upstream_upc_manifest",
        base_directory,
    )
    accepted_upstream_status = _require_string(
        inputs.get("accepted_upstream_status"),
        "inputs.accepted_upstream_status",
    )
    if accepted_upstream_status != "diagnostic_mismatch":
        raise UpcFinalClusterError(
            "현재 실행은 Fig. 4 불일치를 명시한 diagnostic_mismatch만 받습니다."
        )

    roles = _require_mapping(root.get("protocol_roles"), "protocol_roles")
    primary_protocol = _require_string(
        roles.get("primary_model_protocol"),
        "protocol_roles.primary_model_protocol",
    )
    sensitivity_protocol = _require_string(
        roles.get("sensitivity_model_protocol"),
        "protocol_roles.sensitivity_model_protocol",
    )
    forbidden_protocol = _require_string(
        roles.get("forbidden_model_protocol"),
        "protocol_roles.forbidden_model_protocol",
    )
    if (primary_protocol, sensitivity_protocol) != PROTOCOL_NAMES:
        raise UpcFinalClusterError(
            "프로토콜 역할은 train_only 주 분석, algorithm1_full_month 민감도여야 합니다."
        )
    if forbidden_protocol != "figure4_probe_complete_weeks_mean_profile":
        raise UpcFinalClusterError("Fig. 4 probe 금지 이름이 계약과 다릅니다.")

    clustering = _require_mapping(root.get("clustering"), "clustering")
    cluster_count = _require_int(
        clustering.get("cluster_count"), "clustering.cluster_count", minimum=2
    )
    if cluster_count != 2:
        raise UpcFinalClusterError("이번 재현의 cluster_count는 2로 고정합니다.")
    theta = _require_int(clustering.get("theta"), "clustering.theta", minimum=0)
    pcc_vector_length = _require_int(
        clustering.get("pcc_vector_length"),
        "clustering.pcc_vector_length",
        minimum=2,
    )
    if pcc_vector_length != HOURS_PER_DAY:
        raise UpcFinalClusterError("PCC profile 길이는 24여야 합니다.")
    expected_literals = {
        "seed_size_rule": "strictly_greater_than_theta",
        "group_profile": ("mean_over_cells_and_weekdays_of_per_cell_scaled_hourly_sum"),
        "seed_pair_rule": "minimum_pcc",
        "assignment_score": ("unweighted_mean_pcc_to_current_cluster_member_groups"),
        "small_nonempty_group_policy": ("profile_and_assign_but_exclude_from_seed"),
        "empty_group_policy": "exclude_from_pcc_and_assignment",
        "canonical_cluster_labels": "ascending_seed_group_id",
        "tie_break": ("lexicographically_smallest_seed_pair_then_smallest_cluster_id"),
    }
    for field, expected in expected_literals.items():
        _expect_literal(clustering, field, expected, "clustering")
    primary_order = _require_string(
        clustering.get("primary_remaining_group_order"),
        "clustering.primary_remaining_group_order",
    )
    sensitivity_order = _require_string(
        clustering.get("sensitivity_remaining_group_order"),
        "clustering.sensitivity_remaining_group_order",
    )
    if primary_order != PRIMARY_ORDER or sensitivity_order != SENSITIVITY_ORDER:
        raise UpcFinalClusterError(
            "남은 그룹 순서는 오름차순 주 분석, 내림차순 민감도로 고정합니다."
        )
    pcc_tie_tolerance = _require_float(
        clustering.get("pcc_tie_tolerance"),
        "clustering.pcc_tie_tolerance",
        minimum=0.0,
        maximum=1e-6,
    )

    paper = _require_mapping(root.get("paper_diagnostic"), "paper_diagnostic")
    _expect_literal(
        paper, "role", "diagnostic_only_not_a_tuning_target", "paper_diagnostic"
    )
    paper_group_start = _require_int(
        paper.get("figure5_first_cluster_group_start_inclusive"),
        "paper_diagnostic.figure5_first_cluster_group_start_inclusive",
    )
    paper_group_end = _require_int(
        paper.get("figure5_first_cluster_group_end_inclusive"),
        "paper_diagnostic.figure5_first_cluster_group_end_inclusive",
    )
    if (paper_group_start, paper_group_end) != (8, 18):
        raise UpcFinalClusterError("Fig. 5 진단 범위는 group 8~18이어야 합니다.")

    validation = _require_mapping(root.get("validation"), "validation")
    minimum_order_agreement = _require_float(
        validation.get("minimum_order_sensitivity_agreement"),
        "validation.minimum_order_sensitivity_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    require_all_cells = _require_bool(
        validation.get("require_all_cells_assigned_once"),
        "validation.require_all_cells_assigned_once",
    )
    require_all_central = _require_bool(
        validation.get("require_all_central_cells_assigned_once"),
        "validation.require_all_central_cells_assigned_once",
    )
    if not require_all_cells or not require_all_central:
        raise UpcFinalClusterError(
            "전체 셀과 중앙 900셀의 완전 배정 검사는 필수입니다."
        )

    execution = _require_mapping(root.get("execution"), "execution")
    cell_chunk_size = _require_int(
        execution.get("cell_chunk_size"),
        "execution.cell_chunk_size",
        minimum=1,
    )
    _expect_literal(execution, "multiprocessing", False, "execution")
    _expect_literal(execution, "execution_location", "local_wsl_cpu", "execution")

    outputs = _require_mapping(root.get("outputs"), "outputs")
    output_directory = _resolve_path(
        outputs.get("directory"), "outputs.directory", base_directory
    )
    return FinalClusterConfig(
        path=path.resolve(),
        name=name,
        upstream_config_path=upstream_config_path,
        upstream_config_sha256=upstream_config_sha256,
        upstream_manifest_path=upstream_manifest_path,
        accepted_upstream_status=accepted_upstream_status,
        primary_protocol=primary_protocol,
        sensitivity_protocol=sensitivity_protocol,
        forbidden_protocol=forbidden_protocol,
        cluster_count=cluster_count,
        theta=theta,
        pcc_tie_tolerance=pcc_tie_tolerance,
        primary_order=primary_order,
        sensitivity_order=sensitivity_order,
        paper_group_start=paper_group_start,
        paper_group_end=paper_group_end,
        minimum_order_agreement=minimum_order_agreement,
        require_all_cells_assigned_once=require_all_cells,
        require_all_central_cells_assigned_once=require_all_central,
        cell_chunk_size=cell_chunk_size,
        output_directory=output_directory,
    )


def verify_upstream_initial_groups(
    config: FinalClusterConfig,
) -> tuple[Any, Mapping[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """초기 UPC config, manifest, membership 파일을 연결해 검증한다."""

    actual_config_sha = compute_sha256(config.upstream_config_path)
    if actual_config_sha != config.upstream_config_sha256:
        raise UpcFinalClusterError(
            "초기 UPC config checksum이 PCC config의 사전 등록값과 다릅니다."
        )
    try:
        upstream_config = load_upc_config(config.upstream_config_path)
        manifest = _load_json(config.upstream_manifest_path, "초기 UPC manifest")
    except UpcInitialGroupError as exc:
        raise UpcFinalClusterError(str(exc)) from exc
    if manifest.get("status") != config.accepted_upstream_status:
        raise UpcFinalClusterError(
            "초기 UPC manifest status가 명시적으로 허용한 상태와 다릅니다: "
            f"{manifest.get('status')!r}"
        )
    manifest_config = _require_mapping(manifest.get("config"), "upstream.config")
    if manifest_config.get("sha256") != actual_config_sha:
        raise UpcFinalClusterError(
            "초기 UPC manifest가 가리키는 config checksum이 다릅니다."
        )
    roles = _require_mapping(manifest.get("protocol_roles"), "upstream.protocol_roles")
    if roles.get("primary_model_protocol") != config.primary_protocol:
        raise UpcFinalClusterError("초기 UPC의 주 프로토콜 역할이 다릅니다.")
    if roles.get("sensitivity_model_protocol") != config.sensitivity_protocol:
        raise UpcFinalClusterError("초기 UPC의 민감도 프로토콜 역할이 다릅니다.")
    if roles.get("diagnostic_only") != config.forbidden_protocol:
        raise UpcFinalClusterError("초기 UPC의 진단 전용 프로토콜 이름이 다릅니다.")
    continuation = _require_mapping(
        manifest.get("continuation_decision"), "upstream.continuation_decision"
    )
    if (
        continuation.get("exact_figure4_match_required_for_independent_baselines")
        is not False
    ):
        raise UpcFinalClusterError(
            "Fig. 4 불일치 후속 진행 결정이 초기 UPC manifest에 없습니다."
        )

    outputs = _require_mapping(manifest.get("outputs"), "upstream.outputs")
    memberships: dict[str, np.ndarray] = {}
    verified: dict[str, Any] = {}
    for protocol_name in PROTOCOL_NAMES:
        key = f"{protocol_name}_peak_hours"
        metadata = _require_mapping(outputs.get(key), f"upstream.outputs.{key}")
        path = getattr(upstream_config.outputs, key)
        try:
            verified[key] = _verify_file_metadata(
                path, metadata, f"upstream.outputs.{key}"
            )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, UpcInitialGroupError) as exc:
            raise UpcFinalClusterError(
                f"{protocol_name} 초기 membership 검증에 실패했습니다: {exc}"
            ) from exc
        expected_shape = (upstream_config.expected_cell_count,)
        if array.shape != expected_shape or array.dtype != np.dtype("int8"):
            raise UpcFinalClusterError(
                f"{key} shape/dtype이 다릅니다: {array.shape}, {array.dtype}"
            )
        if np.any(array < 0) or np.any(array >= HOURS_PER_DAY):
            raise UpcFinalClusterError(f"{key}에 0~23 밖의 group ID가 있습니다.")
        expected_counts = _require_mapping(
            manifest.get("protocols"), "upstream.protocols"
        )[protocol_name]["group_counts_hour_0_to_23"]
        actual_counts = np.bincount(array, minlength=HOURS_PER_DAY).tolist()
        if actual_counts != expected_counts:
            raise UpcFinalClusterError(
                f"{key}의 실제 그룹 수가 초기 manifest와 다릅니다."
            )
        memberships[protocol_name] = array

    return (
        upstream_config,
        manifest,
        memberships,
        {
            "config": {
                "path": _display_path(config.upstream_config_path),
                "sha256": actual_config_sha,
            },
            "manifest": {
                "path": _display_path(config.upstream_manifest_path),
                "sha256": compute_sha256(config.upstream_manifest_path),
                "status": manifest.get("status"),
            },
            "membership_files": verified,
        },
    )


def compute_group_profiles(
    traffic: np.ndarray,
    protocol_hour_indices: np.ndarray,
    initial_group_ids: np.ndarray,
    *,
    cell_chunk_size: int,
) -> GroupProfileResult:
    """셀별 scaling 후 셀·평일 평균인 24시간 group profile을 계산한다."""

    if traffic.ndim != 2:
        raise UpcFinalClusterError("traffic은 (cell, time) 2차원이어야 합니다.")
    if protocol_hour_indices.ndim != 3 or protocol_hour_indices.shape[1] != 24:
        raise UpcFinalClusterError(
            "hour index는 (weekday, 24, observation)이어야 합니다."
        )
    if initial_group_ids.shape != (traffic.shape[0],):
        raise UpcFinalClusterError("초기 group ID 수가 traffic cell 수와 다릅니다.")
    if np.any(initial_group_ids < 0) or np.any(initial_group_ids >= HOURS_PER_DAY):
        raise UpcFinalClusterError("초기 group ID는 0~23이어야 합니다.")
    if cell_chunk_size < 1:
        raise UpcFinalClusterError("cell_chunk_size는 1 이상이어야 합니다.")

    flat_indices = protocol_hour_indices.reshape(-1)
    if len(flat_indices) == 0 or int(flat_indices.min()) < 0:
        raise UpcFinalClusterError("선택된 시간 index가 비어 있거나 음수입니다.")
    if int(flat_indices.max()) >= traffic.shape[1]:
        raise UpcFinalClusterError("선택된 시간 index가 traffic 범위를 벗어납니다.")
    day_count = protocol_hour_indices.shape[0]
    observations_per_hour = protocol_hour_indices.shape[2]
    counts = np.bincount(initial_group_ids, minlength=HOURS_PER_DAY).astype(np.int64)
    sums = np.zeros((HOURS_PER_DAY, HOURS_PER_DAY), dtype=np.float64)
    constant_cell_count = 0

    for start in range(0, traffic.shape[0], cell_chunk_size):
        stop = min(start + cell_chunk_size, traffic.shape[0])
        raw = np.asarray(traffic[start:stop, flat_indices], dtype=np.float64).reshape(
            stop - start,
            day_count,
            HOURS_PER_DAY,
            observations_per_hour,
        )
        minimum = raw.min(axis=(1, 2, 3))
        maximum = raw.max(axis=(1, 2, 3))
        span = maximum - minimum
        constant_cell_count += int(np.count_nonzero(span == 0.0))
        raw_hourly = raw.sum(axis=3, dtype=np.float64)
        scaled_hourly = np.zeros_like(raw_hourly)
        np.divide(
            raw_hourly - observations_per_hour * minimum[:, None, None],
            span[:, None, None],
            out=scaled_hourly,
            where=span[:, None, None] != 0.0,
        )
        cell_profiles = scaled_hourly.mean(axis=1, dtype=np.float64)
        np.add.at(
            sums,
            np.asarray(initial_group_ids[start:stop], dtype=np.int64),
            cell_profiles,
        )

    profiles = np.zeros_like(sums)
    valid = counts > 0
    np.divide(sums, counts[:, None], out=profiles, where=valid[:, None])
    if not np.all(np.isfinite(profiles[valid])):
        raise UpcFinalClusterError("비어 있지 않은 group profile에 NaN/Inf가 있습니다.")
    if int(counts.sum()) != traffic.shape[0]:
        raise UpcFinalClusterError("group profile 계산에서 일부 cell이 누락됐습니다.")
    return GroupProfileResult(
        profiles=profiles,
        valid=valid,
        counts=counts,
        diagnostics={
            "weekday_count": int(day_count),
            "observations_per_hour": int(observations_per_hour),
            "constant_cell_count": constant_cell_count,
            "nonempty_group_count": int(np.count_nonzero(valid)),
            "empty_group_ids": np.flatnonzero(~valid).astype(int).tolist(),
            "profile_definition": (
                "mean over cells and selected weekdays of each cell's per-protocol "
                "min-max-scaled six-observation hourly sum"
            ),
        },
    )


def pearson_correlation_matrix(
    profiles: np.ndarray,
    group_ids: np.ndarray | None = None,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """행별 profile의 PCC 행렬을 계산하고 수치적 불변식을 검사한다."""

    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise UpcFinalClusterError(
            "PCC에는 2개 이상의 길이 2 이상 profile이 필요합니다."
        )
    if not np.all(np.isfinite(values)):
        raise UpcFinalClusterError("PCC profile에 NaN/Inf가 있습니다.")
    ids = (
        np.arange(values.shape[0], dtype=np.int64)
        if group_ids is None
        else np.asarray(group_ids, dtype=np.int64)
    )
    if ids.shape != (values.shape[0],) or len(np.unique(ids)) != len(ids):
        raise UpcFinalClusterError(
            "PCC group ID가 profile 행과 일대일 대응하지 않습니다."
        )
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    invalid = np.flatnonzero(norms <= np.finfo(np.float64).eps)
    if len(invalid):
        invalid_ids = ids[invalid].astype(int).tolist()
        raise UpcFinalClusterError(
            f"분산이 0인 group profile은 PCC를 계산할 수 없습니다: {invalid_ids}"
        )
    normalized = centered / norms[:, None]
    matrix = normalized @ normalized.T
    if not np.all(np.isfinite(matrix)):
        raise UpcFinalClusterError("PCC 행렬에 NaN/Inf가 있습니다.")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise UpcFinalClusterError("PCC 행렬이 대칭이 아닙니다.")
    if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=tolerance):
        raise UpcFinalClusterError("PCC 행렬 대각선이 1이 아닙니다.")
    if float(matrix.min()) < -1.0 - tolerance or float(matrix.max()) > 1.0 + tolerance:
        raise UpcFinalClusterError("PCC 값이 [-1, 1] 범위를 벗어났습니다.")
    matrix = np.clip((matrix + matrix.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def eligible_seed_groups(group_counts: np.ndarray, theta: int) -> np.ndarray:
    """논문의 엄격한 |G_k| > theta 조건을 적용한다."""

    counts = np.asarray(group_counts)
    if counts.shape != (HOURS_PER_DAY,) or np.any(counts < 0):
        raise UpcFinalClusterError(
            "group_counts는 음수가 없는 길이 24 벡터여야 합니다."
        )
    if theta < 0:
        raise UpcFinalClusterError("theta는 0 이상이어야 합니다.")
    return np.flatnonzero(counts > theta).astype(np.int8)


def _pcc_index(nonempty_group_ids: np.ndarray) -> dict[int, int]:
    return {int(group_id): index for index, group_id in enumerate(nonempty_group_ids)}


def select_seed_pair(
    pcc: np.ndarray,
    nonempty_group_ids: np.ndarray,
    eligible_group_ids: np.ndarray,
    *,
    tolerance: float,
) -> tuple[tuple[int, int], float]:
    """eligible group 중 PCC가 최소인 두 group을 결정론적으로 선택한다."""

    nonempty = np.asarray(nonempty_group_ids, dtype=np.int64)
    eligible = np.sort(np.asarray(eligible_group_ids, dtype=np.int64))
    if pcc.shape != (len(nonempty), len(nonempty)):
        raise UpcFinalClusterError("PCC shape가 nonempty group 수와 다릅니다.")
    if len(eligible) < 2:
        raise UpcFinalClusterError("seed 후보 group이 2개보다 적습니다.")
    index = _pcc_index(nonempty)
    if any(int(group_id) not in index for group_id in eligible):
        raise UpcFinalClusterError("seed 후보가 nonempty PCC 행렬에 없습니다.")
    scored = [
        ((int(left), int(right)), float(pcc[index[int(left)], index[int(right)]]))
        for left, right in combinations(eligible.tolist(), 2)
    ]
    minimum = min(score for _, score in scored)
    tied = [pair for pair, score in scored if abs(score - minimum) <= tolerance]
    return min(tied), minimum


def assign_groups(
    pcc: np.ndarray,
    nonempty_group_ids: np.ndarray,
    seed_pair: tuple[int, int],
    *,
    order: str,
    tolerance: float,
) -> AssignmentResult:
    """각 남은 group을 현재 cluster member와의 평균 PCC가 큰 곳에 배정한다."""

    if order not in {PRIMARY_ORDER, SENSITIVITY_ORDER}:
        raise UpcFinalClusterError(f"지원하지 않는 group 순서입니다: {order}")
    nonempty = np.sort(np.asarray(nonempty_group_ids, dtype=np.int64))
    if len(nonempty) < 2 or len(np.unique(nonempty)) != len(nonempty):
        raise UpcFinalClusterError("nonempty group ID가 올바르지 않습니다.")
    if pcc.shape != (len(nonempty), len(nonempty)):
        raise UpcFinalClusterError("PCC shape가 nonempty group 수와 다릅니다.")
    seeds = tuple(sorted(int(value) for value in seed_pair))
    if len(set(seeds)) != 2 or any(
        seed not in set(nonempty.tolist()) for seed in seeds
    ):
        raise UpcFinalClusterError("seed pair가 서로 다른 nonempty group이어야 합니다.")
    index = _pcc_index(nonempty)
    members: list[list[int]] = [[seeds[0]], [seeds[1]]]
    assignments = np.full(HOURS_PER_DAY, -1, dtype=np.int8)
    assignments[seeds[0]] = 0
    assignments[seeds[1]] = 1
    remaining = [int(value) for value in nonempty if int(value) not in seeds]
    remaining.sort(reverse=order == SENSITIVITY_ORDER)
    trace: list[dict[str, Any]] = []

    for group_id in remaining:
        scores = [
            float(
                np.mean(
                    [pcc[index[group_id], index[member]] for member in cluster_members],
                    dtype=np.float64,
                )
            )
            for cluster_members in members
        ]
        maximum = max(scores)
        candidates = [
            cluster_id
            for cluster_id, score in enumerate(scores)
            if abs(score - maximum) <= tolerance
        ]
        chosen = min(candidates)
        assignments[group_id] = chosen
        trace.append(
            {
                "group_id": group_id,
                "mean_pcc_by_cluster_before_assignment": scores,
                "chosen_cluster": chosen,
                "tie": len(candidates) > 1,
            }
        )
        members[chosen].append(group_id)

    if np.any(assignments[nonempty] < 0):
        raise UpcFinalClusterError(
            "비어 있지 않은 초기 group 일부가 배정되지 않았습니다."
        )
    canonical_members = tuple(tuple(sorted(group)) for group in members)
    return AssignmentResult(
        group_to_cluster=assignments,
        cluster_members=canonical_members,
        trace=tuple(trace),
    )


def label_invariant_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    """두 N=2 membership을 label swap에 불변인 방식으로 비교한다."""

    left = np.asarray(reference, dtype=np.int8)
    right = np.asarray(candidate, dtype=np.int8)
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise UpcFinalClusterError("membership 비교 배열의 shape가 올바르지 않습니다.")
    if np.any((left < 0) | (left > 1)) or np.any((right < 0) | (right > 1)):
        raise UpcFinalClusterError("membership 비교 label은 0 또는 1이어야 합니다.")
    direct = int(np.count_nonzero(left == right))
    swapped_values = 1 - right
    swapped = int(np.count_nonzero(left == swapped_values))
    use_swap = swapped > direct
    matched = swapped if use_swap else direct
    return {
        "item_count": len(left),
        "matched_item_count": matched,
        "agreement_ratio": matched / len(left),
        "candidate_label_mapping_to_reference": (
            {"0": 1, "1": 0} if use_swap else {"0": 0, "1": 1}
        ),
        "direct_match_count": direct,
        "swapped_match_count": swapped,
    }


def _cell_clusters(
    initial_group_ids: np.ndarray,
    group_to_cluster: np.ndarray,
    expected_cell_count: int,
) -> np.ndarray:
    values = np.asarray(group_to_cluster[np.asarray(initial_group_ids, dtype=np.int64)])
    if values.shape != (expected_cell_count,) or np.any((values < 0) | (values > 1)):
        raise UpcFinalClusterError(
            "모든 cell이 최종 cluster 하나에 배정되지 않았습니다."
        )
    return values.astype(np.int8, copy=False)


def _cluster_counts(cell_clusters: np.ndarray) -> list[int]:
    counts = np.bincount(cell_clusters, minlength=2).astype(np.int64)
    if len(counts) != 2 or int(counts.sum()) != len(cell_clusters):
        raise UpcFinalClusterError(
            "최종 cluster 수 또는 cell 합계가 올바르지 않습니다."
        )
    return counts.astype(int).tolist()


def _assignment_payload(result: AssignmentResult) -> dict[str, Any]:
    return {
        "cluster_group_ids": {
            str(cluster_id): list(group_ids)
            for cluster_id, group_ids in enumerate(result.cluster_members)
        },
        "assignment_trace": list(result.trace),
    }


def _paper_figure5_diagnostic(
    assignment: np.ndarray,
    group_counts: np.ndarray,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    nonempty = np.flatnonzero(group_counts > 0)
    actual = assignment[nonempty]
    expected = np.asarray(
        [0 if start <= int(group_id) <= end else 1 for group_id in nonempty],
        dtype=np.int8,
    )
    group_comparison = label_invariant_agreement(expected, actual)
    weighted_expected = np.repeat(expected, group_counts[nonempty])
    weighted_actual = np.repeat(actual, group_counts[nonempty])
    cell_comparison = label_invariant_agreement(weighted_expected, weighted_actual)
    return {
        "nonempty_group_agreement": group_comparison,
        "cell_weighted_agreement": cell_comparison,
    }


def build_protocol_clusters(
    name: str,
    traffic: np.ndarray,
    hour_indices: np.ndarray,
    initial_group_ids: np.ndarray,
    central_positions: np.ndarray,
    config: FinalClusterConfig,
) -> ProtocolClusterResult:
    profile_result = compute_group_profiles(
        traffic,
        hour_indices,
        initial_group_ids,
        cell_chunk_size=config.cell_chunk_size,
    )
    nonempty = np.flatnonzero(profile_result.valid).astype(np.int8)
    pcc = pearson_correlation_matrix(
        profile_result.profiles[nonempty],
        nonempty,
        tolerance=config.pcc_tie_tolerance,
    )
    eligible = eligible_seed_groups(profile_result.counts, config.theta)
    seed_pair, seed_pcc = select_seed_pair(
        pcc,
        nonempty,
        eligible,
        tolerance=config.pcc_tie_tolerance,
    )
    primary = assign_groups(
        pcc,
        nonempty,
        seed_pair,
        order=config.primary_order,
        tolerance=config.pcc_tie_tolerance,
    )
    order_sensitivity = assign_groups(
        pcc,
        nonempty,
        seed_pair,
        order=config.sensitivity_order,
        tolerance=config.pcc_tie_tolerance,
    )
    primary_cells = _cell_clusters(
        initial_group_ids, primary.group_to_cluster, len(initial_group_ids)
    )
    descending_cells = _cell_clusters(
        initial_group_ids, order_sensitivity.group_to_cluster, len(initial_group_ids)
    )
    order_all = label_invariant_agreement(primary_cells, descending_cells)
    order_central = label_invariant_agreement(
        primary_cells[central_positions], descending_cells[central_positions]
    )
    small_nonempty = np.flatnonzero(
        (profile_result.counts > 0) & (profile_result.counts <= config.theta)
    ).astype(int)
    diagnostics: dict[str, Any] = {
        **profile_result.diagnostics,
        "initial_group_counts_hour_0_to_23": profile_result.counts.astype(int).tolist(),
        "seed_size_rule": f"size > {config.theta}",
        "eligible_seed_group_ids": eligible.astype(int).tolist(),
        "small_nonempty_group_ids_excluded_only_from_seed": small_nonempty.tolist(),
        "seed_pair": list(seed_pair),
        "seed_pcc": seed_pcc,
        "primary": {
            **_assignment_payload(primary),
            "all_cell_cluster_counts": _cluster_counts(primary_cells),
            "central_900_cluster_counts": _cluster_counts(
                primary_cells[central_positions]
            ),
        },
        "remaining_order_sensitivity": {
            "order": config.sensitivity_order,
            **_assignment_payload(order_sensitivity),
            "all_cell_cluster_counts": _cluster_counts(descending_cells),
            "central_900_cluster_counts": _cluster_counts(
                descending_cells[central_positions]
            ),
            "all_cells_label_invariant_agreement": order_all,
            "central_900_label_invariant_agreement": order_central,
            "minimum_agreement_threshold": config.minimum_order_agreement,
            "review_required": (
                order_all["agreement_ratio"] < config.minimum_order_agreement
            ),
        },
        "paper_figure5_diagnostic": {
            "role": "diagnostic_only_not_a_tuning_target",
            "published_first_cluster_group_ids": list(
                range(config.paper_group_start, config.paper_group_end + 1)
            ),
            "primary_ascending_order": _paper_figure5_diagnostic(
                primary.group_to_cluster,
                profile_result.counts,
                start=config.paper_group_start,
                end=config.paper_group_end,
            ),
            "descending_order_sensitivity": _paper_figure5_diagnostic(
                order_sensitivity.group_to_cluster,
                profile_result.counts,
                start=config.paper_group_start,
                end=config.paper_group_end,
            ),
        },
    }
    return ProtocolClusterResult(
        name=name,
        group_profiles=profile_result.profiles,
        profile_valid=profile_result.valid,
        nonempty_group_ids=nonempty,
        pcc=pcc,
        group_counts=profile_result.counts,
        eligible_seed_group_ids=eligible,
        seed_pair=seed_pair,
        seed_pcc=seed_pcc,
        primary=primary,
        order_sensitivity=order_sensitivity,
        diagnostics=diagnostics,
    )


def _output_paths(config: FinalClusterConfig) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for protocol_name in PROTOCOL_NAMES:
        paths[f"{protocol_name}_group_profiles"] = (
            config.output_directory / f"{protocol_name}_group_profiles.npy"
        )
        paths[f"{protocol_name}_profile_valid"] = (
            config.output_directory / f"{protocol_name}_profile_valid.npy"
        )
        paths[f"{protocol_name}_nonempty_group_ids"] = (
            config.output_directory / f"{protocol_name}_nonempty_group_ids.npy"
        )
        paths[f"{protocol_name}_pcc"] = (
            config.output_directory / f"{protocol_name}_pcc.npy"
        )
    paths.update(
        {
            "group_assignments_csv": config.output_directory / "group_assignments.csv",
            "all_cell_memberships_csv": (
                config.output_directory / "all_cell_memberships.csv"
            ),
            "central_900_memberships_csv": (
                config.output_directory / "central_900_memberships.csv"
            ),
            "summary_json": config.output_directory / "summary.json",
            "manifest": config.output_directory / "manifest.json",
        }
    )
    return paths


def _write_npy(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def _write_group_assignments(
    path: Path,
    results: Mapping[str, ProtocolClusterResult],
    config: FinalClusterConfig,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "protocol",
                "initial_group",
                "cell_count",
                "profile_valid",
                "seed_eligible_size_gt_theta",
                "is_seed",
                "primary_cluster",
                "descending_order_cluster",
                "paper_figure5_expected_cluster",
            ]
        )
        for protocol_name in PROTOCOL_NAMES:
            result = results[protocol_name]
            seed_set = set(result.seed_pair)
            eligible_set = set(result.eligible_seed_group_ids.astype(int).tolist())
            for group_id in range(HOURS_PER_DAY):
                valid = bool(result.profile_valid[group_id])
                writer.writerow(
                    [
                        protocol_name,
                        group_id,
                        int(result.group_counts[group_id]),
                        int(valid),
                        int(group_id in eligible_set),
                        int(group_id in seed_set),
                        int(result.primary.group_to_cluster[group_id]) if valid else "",
                        (
                            int(result.order_sensitivity.group_to_cluster[group_id])
                            if valid
                            else ""
                        ),
                        0
                        if config.paper_group_start
                        <= group_id
                        <= config.paper_group_end
                        else 1,
                    ]
                )


def _write_all_cell_memberships(
    path: Path,
    cell_ids: np.ndarray,
    central_positions: np.ndarray,
    memberships: Mapping[str, np.ndarray],
    results: Mapping[str, ProtocolClusterResult],
) -> None:
    central_mask = np.zeros(len(cell_ids), dtype=bool)
    central_mask[central_positions] = True
    cell_clusters = {
        name: _cell_clusters(
            memberships[name], results[name].primary.group_to_cluster, len(cell_ids)
        )
        for name in PROTOCOL_NAMES
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "cell_id",
                "is_central_900",
                "train_only_initial_group",
                "train_only_cluster",
                "algorithm1_full_month_initial_group",
                "algorithm1_full_month_cluster",
            ]
        )
        for index, cell_id in enumerate(cell_ids):
            writer.writerow(
                [
                    int(cell_id),
                    int(central_mask[index]),
                    int(memberships["train_only"][index]),
                    int(cell_clusters["train_only"][index]),
                    int(memberships["algorithm1_full_month"][index]),
                    int(cell_clusters["algorithm1_full_month"][index]),
                ]
            )


def _write_central_memberships(
    path: Path,
    central_rows: Sequence[Mapping[str, str]],
    central_positions: np.ndarray,
    memberships: Mapping[str, np.ndarray],
    results: Mapping[str, ProtocolClusterResult],
) -> None:
    clusters = {
        name: _cell_clusters(
            memberships[name],
            results[name].primary.group_to_cluster,
            len(memberships[name]),
        )
        for name in PROTOCOL_NAMES
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "cell_id",
                "grid_row",
                "grid_column",
                "centroid_lon",
                "centroid_lat",
                "train_only_initial_group",
                "train_only_cluster",
                "algorithm1_full_month_initial_group",
                "algorithm1_full_month_cluster",
            ]
        )
        for row, position in zip(central_rows, central_positions, strict=True):
            writer.writerow(
                [
                    row["cell_id"],
                    row["grid_row"],
                    row["grid_column"],
                    row["centroid_lon"],
                    row["centroid_lat"],
                    int(memberships["train_only"][position]),
                    int(clusters["train_only"][position]),
                    int(memberships["algorithm1_full_month"][position]),
                    int(clusters["algorithm1_full_month"][position]),
                ]
            )


def _estimated_chunk_working_set_bytes(
    cell_chunk_size: int,
    maximum_weekday_count: int,
    observations_per_hour: int,
) -> int:
    selected = (
        cell_chunk_size * maximum_weekday_count * HOURS_PER_DAY * observations_per_hour
    )
    hourly = cell_chunk_size * maximum_weekday_count * HOURS_PER_DAY
    return selected * (4 + 8) + hourly * 8 * 2


def run_upc_final_clusters(config: FinalClusterConfig) -> dict[str, Any]:
    """검증된 초기 그룹으로 PCC 군집을 만들고 원자적으로 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    upstream_config, upstream_manifest, memberships, upstream_validation = (
        verify_upstream_initial_groups(config)
    )
    try:
        arrays, processed_validation = verify_processed_inputs(upstream_config)
        central_rows, central_positions, central_validation = load_central_cells(
            upstream_config, arrays["cell_ids"]
        )
        axis = build_local_time_axis(
            arrays["timestamps_ms"],
            timezone_name=upstream_config.timezone_name,
            interval_ms=upstream_config.interval_ms,
            observations_per_hour=upstream_config.observations_per_hour,
        )
        protocol_hours = {
            name: select_protocol_hours(axis, upstream_config.protocols[name])
            for name in PROTOCOL_NAMES
        }
    except UpcInitialGroupError as exc:
        raise UpcFinalClusterError(str(exc)) from exc

    results = {
        name: build_protocol_clusters(
            name,
            arrays["traffic"],
            protocol_hours[name].indices,
            memberships[name],
            central_positions,
            config,
        )
        for name in PROTOCOL_NAMES
    }
    cell_clusters = {
        name: _cell_clusters(
            memberships[name],
            results[name].primary.group_to_cluster,
            len(arrays["cell_ids"]),
        )
        for name in PROTOCOL_NAMES
    }
    protocol_comparison = {
        "all_cells_label_invariant_agreement": label_invariant_agreement(
            cell_clusters[config.primary_protocol],
            cell_clusters[config.sensitivity_protocol],
        ),
        "central_900_label_invariant_agreement": label_invariant_agreement(
            cell_clusters[config.primary_protocol][central_positions],
            cell_clusters[config.sensitivity_protocol][central_positions],
        ),
    }
    order_review_required = any(
        result.diagnostics["remaining_order_sensitivity"]["review_required"]
        for result in results.values()
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "complete_with_order_sensitivity_review"
            if order_review_required
            else "complete"
        ),
        "dataset": config.name,
        "protocol_roles": {
            "primary_model_protocol": config.primary_protocol,
            "sensitivity_model_protocol": config.sensitivity_protocol,
            "forbidden_model_protocol": config.forbidden_protocol,
        },
        "algorithm_contract": {
            "cluster_count": config.cluster_count,
            "theta": config.theta,
            "seed_eligibility": "initial group size strictly greater than theta",
            "profile": (
                "length-24 mean over cells and weekdays after per-cell, "
                "per-protocol min-max scaling and six-step hourly sum"
            ),
            "seed_pair": "minimum PCC; lexicographically smallest pair on tie",
            "remaining_assignment": (
                "unweighted mean PCC to current cluster member groups; maximum wins"
            ),
            "primary_remaining_order": config.primary_order,
            "sensitivity_remaining_order": config.sensitivity_order,
            "assignment_tie_break": "smallest cluster ID",
            "canonical_cluster_labels": "ascending seed group ID",
            "small_nonempty_groups": "profiled and assigned but excluded from seeds",
            "empty_groups": "excluded from PCC and assignment",
            "pcc_tie_tolerance": config.pcc_tie_tolerance,
        },
        "protocols": {name: results[name].diagnostics for name in PROTOCOL_NAMES},
        "protocol_comparison": protocol_comparison,
        "engineering_gate": {
            "minimum_order_sensitivity_agreement": config.minimum_order_agreement,
            "order_sensitivity_review_required": order_review_required,
            "ready_for_expensive_model_training": not order_review_required,
            "rule": (
                "do not choose a better-looking assignment after model evaluation; "
                "review the order ambiguity before expensive training"
            ),
        },
    }

    output_paths = _output_paths(config)
    temporary_paths = {key: _temporary_path(path) for key, path in output_paths.items()}
    config.output_directory.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        for protocol_name in PROTOCOL_NAMES:
            result = results[protocol_name]
            _write_npy(
                temporary_paths[f"{protocol_name}_group_profiles"],
                result.group_profiles,
            )
            _write_npy(
                temporary_paths[f"{protocol_name}_profile_valid"],
                result.profile_valid,
            )
            _write_npy(
                temporary_paths[f"{protocol_name}_nonempty_group_ids"],
                result.nonempty_group_ids,
            )
            _write_npy(temporary_paths[f"{protocol_name}_pcc"], result.pcc)
        _write_group_assignments(
            temporary_paths["group_assignments_csv"], results, config
        )
        _write_all_cell_memberships(
            temporary_paths["all_cell_memberships_csv"],
            arrays["cell_ids"],
            central_positions,
            memberships,
            results,
        )
        _write_central_memberships(
            temporary_paths["central_900_memberships_csv"],
            central_rows,
            central_positions,
            memberships,
            results,
        )
        temporary_paths["summary_json"].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        deterministic_keys = [key for key in output_paths if key != "manifest"]
        output_metadata = {
            key: {
                "path": _display_path(output_paths[key]),
                "size_bytes": temporary_paths[key].stat().st_size,
                "sha256": compute_sha256(temporary_paths[key]),
            }
            for key in deterministic_keys
        }
        finished_at = datetime.now(timezone.utc)
        maximum_weekdays = max(
            result.diagnostics["weekday_count"] for result in results.values()
        )
        manifest_status = "complete_with_upstream_diagnostic_mismatch"
        if order_review_required:
            manifest_status += "_and_order_sensitivity_review"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": manifest_status,
            "tool": {
                "name": "scripts.build_upc_final_clusters",
                "version": TOOL_VERSION,
            },
            "dataset": config.name,
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "inputs": {
                "upstream_upc": upstream_validation,
                "processed": processed_validation,
                "central_900": central_validation,
            },
            "upstream_disclosure": {
                "status": upstream_manifest.get("status"),
                "figure4_exact_match": upstream_manifest["paper_fingerprint"][
                    "exact_match"
                ],
                "continuation_decision": upstream_manifest["continuation_decision"],
                "figure4_probe_used_as_input": False,
            },
            "summary": summary,
            "outputs": output_metadata,
            "runtime": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "execution_location": "local_wsl_cpu",
                "multiprocessing": False,
                "cell_chunk_size": config.cell_chunk_size,
                "estimated_chunk_working_set_bytes": (
                    _estimated_chunk_working_set_bytes(
                        config.cell_chunk_size,
                        maximum_weekdays,
                        upstream_config.observations_per_hour,
                    )
                ),
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
        description="GECOS UPC 초기 그룹을 PCC 기반 최종 N=2 cluster로 병합합니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"UPC PCC config (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_final_cluster_config(args.config)
        manifest = run_upc_final_clusters(config)
    except (UpcFinalClusterError, UpcInitialGroupError) as exc:
        print(f"UPC 최종 cluster 생성 실패: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"파일 시스템 오류: {exc}", file=sys.stderr)
        return 2

    summary = manifest["summary"]
    print(f"UPC 최종 cluster 생성 완료: status={manifest['status']}")
    for protocol_name in PROTOCOL_NAMES:
        protocol = summary["protocols"][protocol_name]
        print(
            f"{protocol_name}: seeds={protocol['seed_pair']}, "
            f"seed_pcc={protocol['seed_pcc']:.6f}, "
            f"all_cells={protocol['primary']['all_cell_cluster_counts']}, "
            "order_agreement="
            f"{protocol['remaining_order_sensitivity']['all_cells_label_invariant_agreement']['agreement_ratio']:.6f}"
        )
    print(
        "protocol_agreement="
        f"{summary['protocol_comparison']['all_cells_label_invariant_agreement']['agreement_ratio']:.6f}"
    )
    print(f"manifest={_display_path(_output_paths(config)['manifest'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
