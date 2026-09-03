#!/usr/bin/env python3
"""UPC 순서 민감도 근거를 검증하고 프로토콜별 학습 허용 정책을 게시한다."""

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

TOOL_VERSION = "1.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "upc_training_policy_milan_nov2013.json"
PRIMARY_PROTOCOL = "train_only"
SENSITIVITY_PROTOCOL = "algorithm1_full_month"
FORBIDDEN_PROTOCOL = "figure4_probe_complete_weeks_mean_profile"
PROTOCOL_NAMES = (PRIMARY_PROTOCOL, SENSITIVITY_PROTOCOL)
PROTECTED_OUTPUT_KEYS = (
    "group_assignments_csv",
    "all_cell_memberships_csv",
    "central_900_memberships_csv",
    "summary_json",
)
EXPECTED_GROUP_ASSIGNMENT_FIELDS = (
    "protocol",
    "initial_group",
    "cell_count",
    "profile_valid",
    "seed_eligible_size_gt_theta",
    "is_seed",
    "primary_cluster",
    "descending_order_cluster",
    "paper_figure5_expected_cluster",
)


class UpcTrainingPolicyError(RuntimeError):
    """학습 정책 계약이나 근거 파일의 무결성이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class PolicyOutputPaths:
    policy: Path
    manifest: Path


@dataclass(frozen=True)
class ObservedEvidence:
    primary_order_agreement: float
    sensitivity_order_agreement: float
    sensitivity_changed_group_ids: tuple[int, ...]
    sensitivity_changed_cell_count: int


@dataclass(frozen=True)
class UpcTrainingPolicyConfig:
    path: Path
    base_directory: Path
    name: str
    decision_stage: str
    pcc_config_path: Path
    pcc_config_sha256: str
    pcc_manifest_path: Path
    required_pcc_manifest_status: str
    protected_output_sha256: dict[str, str]
    minimum_order_agreement: float
    numeric_tolerance: float
    observed: ObservedEvidence
    primary_training_protocol: str
    allowed_protocols: tuple[str, ...]
    blocked_protocols: tuple[str, ...]
    reproduction_assignment_order: str
    order_sensitivity_role: str
    full_month_neural_training: str
    figure4_probe_training: str
    order_invariant_extension: str
    model_performance_used_for_decision: bool
    outputs: PolicyOutputPaths


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UpcTrainingPolicyError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UpcTrainingPolicyError(
            f"{label}이 올바른 JSON이 아닙니다: {exc}"
        ) from exc
    return _require_mapping(payload, label)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise UpcTrainingPolicyError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise UpcTrainingPolicyError(f"{field}는 JSON array여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpcTrainingPolicyError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise UpcTrainingPolicyError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UpcTrainingPolicyError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if not np.isfinite(result):
        raise UpcTrainingPolicyError(f"{field}는 유한한 숫자여야 합니다.")
    if minimum is not None and result < minimum:
        raise UpcTrainingPolicyError(f"{field}는 {minimum} 이상이어야 합니다.")
    if maximum is not None and result > maximum:
        raise UpcTrainingPolicyError(f"{field}는 {maximum} 이하여야 합니다.")
    return result


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise UpcTrainingPolicyError(f"{field}는 boolean이어야 합니다.")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = _require_string(value, field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise UpcTrainingPolicyError(f"{field}의 SHA-256 형식이 올바르지 않습니다.")
    return text


def _resolve_path(value: object, field: str, base_directory: Path) -> Path:
    raw = Path(_require_string(value, field))
    return raw.resolve() if raw.is_absolute() else (base_directory / raw).resolve()


def _expect_literal(
    mapping: Mapping[str, Any], field: str, expected: object, prefix: str
) -> None:
    actual = mapping.get(field)
    if actual != expected:
        raise UpcTrainingPolicyError(
            f"{prefix}.{field}는 {expected!r}이어야 합니다: {actual!r}"
        )


def load_policy_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> UpcTrainingPolicyConfig:
    """사후 설계 결정을 명시한 정책 config를 엄격하게 읽는다."""

    root = _load_json(path, "UPC training policy config")
    if _require_int(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise UpcTrainingPolicyError("지원하지 않는 schema_version입니다.")
    name = _require_string(root.get("name"), "name")
    decision_stage = _require_string(root.get("decision_stage"), "decision_stage")
    if decision_stage != "post_clustering_pre_model":
        raise UpcTrainingPolicyError(
            "decision_stage는 post_clustering_pre_model이어야 합니다."
        )

    inputs = _require_mapping(root.get("inputs"), "inputs")
    pcc_config_path = _resolve_path(
        inputs.get("pcc_config"), "inputs.pcc_config", base_directory
    )
    pcc_config_sha256 = _require_sha256(
        inputs.get("pcc_config_sha256"), "inputs.pcc_config_sha256"
    )
    pcc_manifest_path = _resolve_path(
        inputs.get("pcc_manifest"), "inputs.pcc_manifest", base_directory
    )
    required_pcc_manifest_status = _require_string(
        inputs.get("required_pcc_manifest_status"),
        "inputs.required_pcc_manifest_status",
    )
    if (
        required_pcc_manifest_status
        != "complete_with_upstream_diagnostic_mismatch_and_order_sensitivity_review"
    ):
        raise UpcTrainingPolicyError(
            "PCC manifest의 불일치·순서 검토 상태를 명시적으로 유지해야 합니다."
        )
    raw_protected = _require_mapping(
        inputs.get("protected_output_sha256"), "inputs.protected_output_sha256"
    )
    if set(raw_protected) != set(PROTECTED_OUTPUT_KEYS):
        raise UpcTrainingPolicyError(
            "protected_output_sha256는 group/cell membership과 summary 4개만 "
            "정확히 포함해야 합니다."
        )
    protected_output_sha256 = {
        key: _require_sha256(
            raw_protected.get(key), f"inputs.protected_output_sha256.{key}"
        )
        for key in PROTECTED_OUTPUT_KEYS
    }

    thresholds = _require_mapping(root.get("thresholds"), "thresholds")
    minimum_order_agreement = _require_float(
        thresholds.get("minimum_order_sensitivity_agreement"),
        "thresholds.minimum_order_sensitivity_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    numeric_tolerance = _require_float(
        thresholds.get("numeric_tolerance"),
        "thresholds.numeric_tolerance",
        minimum=0.0,
        maximum=1e-6,
    )

    evidence = _require_mapping(root.get("observed_evidence"), "observed_evidence")
    primary_order_agreement = _require_float(
        evidence.get("train_only_order_agreement"),
        "observed_evidence.train_only_order_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    sensitivity_order_agreement = _require_float(
        evidence.get("algorithm1_full_month_order_agreement"),
        "observed_evidence.algorithm1_full_month_order_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    raw_changed_groups = _require_list(
        evidence.get("algorithm1_full_month_changed_group_ids"),
        "observed_evidence.algorithm1_full_month_changed_group_ids",
    )
    changed_groups = tuple(
        _require_int(
            value,
            f"observed_evidence.algorithm1_full_month_changed_group_ids[{index}]",
        )
        for index, value in enumerate(raw_changed_groups)
    )
    if (
        tuple(sorted(changed_groups)) != changed_groups
        or len(set(changed_groups)) != len(changed_groups)
        or any(group_id >= 24 for group_id in changed_groups)
    ):
        raise UpcTrainingPolicyError(
            "변경 group ID는 0~23의 고유한 오름차순 값이어야 합니다."
        )
    changed_cell_count = _require_int(
        evidence.get("algorithm1_full_month_changed_cell_count"),
        "observed_evidence.algorithm1_full_month_changed_cell_count",
        minimum=1,
    )

    policy = _require_mapping(root.get("policy"), "policy")
    primary_training_protocol = _require_string(
        policy.get("primary_training_protocol"), "policy.primary_training_protocol"
    )
    if primary_training_protocol != PRIMARY_PROTOCOL:
        raise UpcTrainingPolicyError("주 학습 프로토콜은 train_only여야 합니다.")
    allowed_protocols = tuple(
        _require_string(value, f"policy.allowed_model_training_protocols[{index}]")
        for index, value in enumerate(
            _require_list(
                policy.get("allowed_model_training_protocols"),
                "policy.allowed_model_training_protocols",
            )
        )
    )
    blocked_protocols = tuple(
        _require_string(value, f"policy.blocked_model_training_protocols[{index}]")
        for index, value in enumerate(
            _require_list(
                policy.get("blocked_model_training_protocols"),
                "policy.blocked_model_training_protocols",
            )
        )
    )
    if allowed_protocols != (PRIMARY_PROTOCOL,):
        raise UpcTrainingPolicyError(
            "현재 학습 허용 목록에는 train_only만 있어야 합니다."
        )
    if blocked_protocols != (SENSITIVITY_PROTOCOL, FORBIDDEN_PROTOCOL):
        raise UpcTrainingPolicyError(
            "차단 목록에는 전체 월 민감도와 Fig. 4 probe가 순서대로 있어야 합니다."
        )
    reproduction_assignment_order = _require_string(
        policy.get("reproduction_assignment_order"),
        "policy.reproduction_assignment_order",
    )
    order_sensitivity_role = _require_string(
        policy.get("order_sensitivity_role"), "policy.order_sensitivity_role"
    )
    full_month_neural_training = _require_string(
        policy.get("full_month_neural_training"),
        "policy.full_month_neural_training",
    )
    figure4_probe_training = _require_string(
        policy.get("figure4_probe_training"), "policy.figure4_probe_training"
    )
    order_invariant_extension = _require_string(
        policy.get("order_invariant_extension"), "policy.order_invariant_extension"
    )
    model_performance_used = _require_bool(
        policy.get("model_performance_used_for_decision"),
        "policy.model_performance_used_for_decision",
    )
    expected_policy_literals = {
        "reproduction_assignment_order": "ascending_group_id",
        "order_sensitivity_role": "diagnostic_only",
        "full_month_neural_training": "deferred_until_separate_explicit_decision",
        "figure4_probe_training": "forbidden",
        "order_invariant_extension": ("deferred_separate_non_reproduction_experiment"),
        "model_performance_used_for_decision": False,
    }
    for field, expected in expected_policy_literals.items():
        _expect_literal(policy, field, expected, "policy")

    outputs = _require_mapping(root.get("outputs"), "outputs")
    output_paths = PolicyOutputPaths(
        policy=_resolve_path(outputs.get("policy"), "outputs.policy", base_directory),
        manifest=_resolve_path(
            outputs.get("manifest"), "outputs.manifest", base_directory
        ),
    )
    if output_paths.policy == output_paths.manifest:
        raise UpcTrainingPolicyError("정책과 manifest 출력 경로는 달라야 합니다.")

    return UpcTrainingPolicyConfig(
        path=path.resolve(),
        base_directory=base_directory.resolve(),
        name=name,
        decision_stage=decision_stage,
        pcc_config_path=pcc_config_path,
        pcc_config_sha256=pcc_config_sha256,
        pcc_manifest_path=pcc_manifest_path,
        required_pcc_manifest_status=required_pcc_manifest_status,
        protected_output_sha256=protected_output_sha256,
        minimum_order_agreement=minimum_order_agreement,
        numeric_tolerance=numeric_tolerance,
        observed=ObservedEvidence(
            primary_order_agreement=primary_order_agreement,
            sensitivity_order_agreement=sensitivity_order_agreement,
            sensitivity_changed_group_ids=changed_groups,
            sensitivity_changed_cell_count=changed_cell_count,
        ),
        primary_training_protocol=primary_training_protocol,
        allowed_protocols=allowed_protocols,
        blocked_protocols=blocked_protocols,
        reproduction_assignment_order=reproduction_assignment_order,
        order_sensitivity_role=order_sensitivity_role,
        full_month_neural_training=full_month_neural_training,
        figure4_probe_training=figure4_probe_training,
        order_invariant_extension=order_invariant_extension,
        model_performance_used_for_decision=model_performance_used,
        outputs=output_paths,
    )


def _metadata_path(
    metadata: Mapping[str, Any], label: str, base_directory: Path
) -> Path:
    return _resolve_path(metadata.get("path"), f"{label}.path", base_directory)


def _verify_protected_file(
    metadata: Mapping[str, Any],
    expected_sha256: str,
    label: str,
    base_directory: Path,
) -> tuple[Path, dict[str, Any]]:
    path = _metadata_path(metadata, label, base_directory)
    metadata_size = _require_int(metadata.get("size_bytes"), f"{label}.size_bytes")
    metadata_sha = _require_sha256(metadata.get("sha256"), f"{label}.sha256")
    if metadata_sha != expected_sha256:
        raise UpcTrainingPolicyError(
            f"{label} checksum이 정책 config에 고정한 근거와 다릅니다."
        )
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise UpcTrainingPolicyError(
            f"{label} 파일을 읽을 수 없습니다: {path}"
        ) from exc
    if actual_size != metadata_size:
        raise UpcTrainingPolicyError(
            f"{label} 크기가 PCC manifest와 다릅니다: {actual_size} != {metadata_size}"
        )
    actual_sha = compute_sha256(path)
    if actual_sha != metadata_sha:
        raise UpcTrainingPolicyError(f"{label} 실제 checksum이 manifest와 다릅니다.")
    return path, {
        "path": _display_path(path),
        "size_bytes": actual_size,
        "sha256": actual_sha,
    }


def load_group_assignment_rows(path: Path) -> list[dict[str, Any]]:
    """24개 group의 두 순서 assignment를 엄격하게 읽는다."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_GROUP_ASSIGNMENT_FIELDS:
                raise UpcTrainingPolicyError(
                    "group assignment CSV 열이 생성 계약과 다릅니다."
                )
            raw_rows = [dict(row) for row in reader]
    except OSError as exc:
        raise UpcTrainingPolicyError(
            f"group assignment CSV를 읽을 수 없습니다: {path}"
        ) from exc
    if len(raw_rows) != len(PROTOCOL_NAMES) * 24:
        raise UpcTrainingPolicyError("group assignment CSV는 정확히 48행이어야 합니다.")

    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row_index, row in enumerate(raw_rows):
        protocol = row["protocol"]
        if protocol not in PROTOCOL_NAMES:
            raise UpcTrainingPolicyError(
                f"알 수 없는 protocol이 assignment CSV에 있습니다: {protocol}"
            )
        try:
            group_id = int(row["initial_group"])
            cell_count = int(row["cell_count"])
            profile_valid = int(row["profile_valid"])
        except (TypeError, ValueError) as exc:
            raise UpcTrainingPolicyError(
                f"assignment CSV {row_index + 2}행의 정수 필드가 잘못됐습니다."
            ) from exc
        if not 0 <= group_id < 24 or cell_count < 0 or profile_valid not in {0, 1}:
            raise UpcTrainingPolicyError(
                f"assignment CSV {row_index + 2}행의 값 범위가 잘못됐습니다."
            )
        key = (protocol, group_id)
        if key in seen:
            raise UpcTrainingPolicyError(f"중복 protocol/group 행입니다: {key}")
        seen.add(key)
        primary_text = row["primary_cluster"]
        descending_text = row["descending_order_cluster"]
        if profile_valid:
            try:
                primary_cluster = int(primary_text)
                descending_cluster = int(descending_text)
            except (TypeError, ValueError) as exc:
                raise UpcTrainingPolicyError(
                    f"유효 group {key}의 cluster label이 없습니다."
                ) from exc
            if primary_cluster not in {0, 1} or descending_cluster not in {0, 1}:
                raise UpcTrainingPolicyError(
                    f"group {key}의 cluster는 0 또는 1이어야 합니다."
                )
            if cell_count == 0:
                raise UpcTrainingPolicyError(
                    f"유효 group {key}의 cell_count가 0입니다."
                )
        else:
            if primary_text or descending_text or cell_count != 0:
                raise UpcTrainingPolicyError(
                    f"빈 group {key}에 cluster 또는 cell이 기록됐습니다."
                )
            primary_cluster = None
            descending_cluster = None
        parsed.append(
            {
                "protocol": protocol,
                "group_id": group_id,
                "cell_count": cell_count,
                "profile_valid": bool(profile_valid),
                "primary_cluster": primary_cluster,
                "descending_cluster": descending_cluster,
            }
        )
    expected_keys = {
        (protocol, group_id) for protocol in PROTOCOL_NAMES for group_id in range(24)
    }
    if seen != expected_keys:
        raise UpcTrainingPolicyError("protocol별 group 0~23 행이 완전하지 않습니다.")
    return parsed


def _nested_mapping(root: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any]:
    value: object = root
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        value = _require_mapping(value, ".".join(traversed[:-1]) or "root").get(key)
    return _require_mapping(value, ".".join(keys))


def analyze_order_changes(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    protocol: str,
) -> dict[str, Any]:
    """label mapping을 적용해 순서 변경 group과 가중 cell 수를 계산한다."""

    protocol_summary = _nested_mapping(summary, ("protocols", protocol))
    sensitivity = _nested_mapping(protocol_summary, ("remaining_order_sensitivity",))
    comparison = _nested_mapping(sensitivity, ("all_cells_label_invariant_agreement",))
    mapping_raw = _require_mapping(
        comparison.get("candidate_label_mapping_to_reference"),
        f"protocols.{protocol}.candidate_label_mapping_to_reference",
    )
    try:
        label_mapping = {int(key): int(value) for key, value in mapping_raw.items()}
    except (TypeError, ValueError) as exc:
        raise UpcTrainingPolicyError(
            "label mapping은 정수 0/1 대응이어야 합니다."
        ) from exc
    if set(label_mapping) != {0, 1} or set(label_mapping.values()) != {0, 1}:
        raise UpcTrainingPolicyError("label mapping은 0과 1의 순열이어야 합니다.")

    protocol_rows = sorted(
        (row for row in rows if row["protocol"] == protocol),
        key=lambda row: int(row["group_id"]),
    )
    changed_group_ids: list[int] = []
    changed_cell_count = 0
    assigned_cell_count = 0
    for row in protocol_rows:
        if not row["profile_valid"]:
            continue
        assigned_cell_count += int(row["cell_count"])
        mapped_descending = label_mapping[int(row["descending_cluster"])]
        if int(row["primary_cluster"]) != mapped_descending:
            changed_group_ids.append(int(row["group_id"]))
            changed_cell_count += int(row["cell_count"])
    item_count = _require_int(
        comparison.get("item_count"),
        f"protocols.{protocol}.order_comparison.item_count",
        minimum=1,
    )
    matched_count = _require_int(
        comparison.get("matched_item_count"),
        f"protocols.{protocol}.order_comparison.matched_item_count",
    )
    agreement_ratio = _require_float(
        comparison.get("agreement_ratio"),
        f"protocols.{protocol}.order_comparison.agreement_ratio",
        minimum=0.0,
        maximum=1.0,
    )
    if assigned_cell_count != item_count:
        raise UpcTrainingPolicyError(
            f"{protocol} group cell 합이 order 비교 item 수와 다릅니다."
        )
    if changed_cell_count != item_count - matched_count:
        raise UpcTrainingPolicyError(
            f"{protocol} 변경 group의 cell 합이 membership 비교와 다릅니다."
        )
    if not np.isclose(
        agreement_ratio,
        matched_count / item_count,
        rtol=0.0,
        atol=1e-12,
    ):
        raise UpcTrainingPolicyError(f"{protocol} agreement ratio 계산이 다릅니다.")
    return {
        "protocol": protocol,
        "label_mapping_descending_to_primary": {
            str(key): label_mapping[key] for key in sorted(label_mapping)
        },
        "assigned_cell_count": assigned_cell_count,
        "matched_cell_count": matched_count,
        "changed_cell_count": changed_cell_count,
        "changed_group_ids": changed_group_ids,
        "agreement_ratio": agreement_ratio,
    }


def evaluate_training_policy(
    config: UpcTrainingPolicyConfig,
    summary: Mapping[str, Any],
    group_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """검증된 clustering 사실을 프로토콜별 학습 gate로 변환한다."""

    if summary.get("status") != "complete_with_order_sensitivity_review":
        raise UpcTrainingPolicyError("PCC summary가 순서 민감도 검토 상태가 아닙니다.")
    roles = _require_mapping(summary.get("protocol_roles"), "summary.protocol_roles")
    if roles.get("primary_model_protocol") != PRIMARY_PROTOCOL:
        raise UpcTrainingPolicyError(
            "PCC summary의 주 프로토콜이 train_only가 아닙니다."
        )
    if roles.get("sensitivity_model_protocol") != SENSITIVITY_PROTOCOL:
        raise UpcTrainingPolicyError("PCC summary의 민감도 프로토콜이 다릅니다.")
    if roles.get("forbidden_model_protocol") != FORBIDDEN_PROTOCOL:
        raise UpcTrainingPolicyError("PCC summary의 Fig. 4 probe 금지 계약이 다릅니다.")
    contract = _require_mapping(
        summary.get("algorithm_contract"), "summary.algorithm_contract"
    )
    if contract.get("primary_remaining_order") != "ascending_group_id":
        raise UpcTrainingPolicyError("PCC 주 배정 순서가 오름차순이 아닙니다.")
    if contract.get("sensitivity_remaining_order") != "descending_group_id":
        raise UpcTrainingPolicyError("PCC 민감도 배정 순서가 내림차순이 아닙니다.")
    clustering_gate = _require_mapping(
        summary.get("engineering_gate"), "summary.engineering_gate"
    )
    source_threshold = _require_float(
        clustering_gate.get("minimum_order_sensitivity_agreement"),
        "summary.engineering_gate.minimum_order_sensitivity_agreement",
        minimum=0.0,
        maximum=1.0,
    )
    if not np.isclose(
        source_threshold,
        config.minimum_order_agreement,
        rtol=0.0,
        atol=config.numeric_tolerance,
    ):
        raise UpcTrainingPolicyError(
            "정책과 PCC summary의 순서 민감도 임계값이 다릅니다."
        )
    if clustering_gate.get("ready_for_expensive_model_training") is not False:
        raise UpcTrainingPolicyError(
            "PCC clustering의 보수적 전역 gate가 false가 아닙니다."
        )

    analyses = {
        protocol: analyze_order_changes(summary, group_rows, protocol)
        for protocol in PROTOCOL_NAMES
    }
    primary = analyses[PRIMARY_PROTOCOL]
    sensitivity = analyses[SENSITIVITY_PROTOCOL]
    observed_checks = (
        (
            primary["agreement_ratio"],
            config.observed.primary_order_agreement,
            "train_only order agreement",
        ),
        (
            sensitivity["agreement_ratio"],
            config.observed.sensitivity_order_agreement,
            "algorithm1_full_month order agreement",
        ),
    )
    for actual, expected, label in observed_checks:
        if not np.isclose(actual, expected, rtol=0.0, atol=config.numeric_tolerance):
            raise UpcTrainingPolicyError(f"{label}가 정책에 기록한 근거와 다릅니다.")
    if tuple(sensitivity["changed_group_ids"]) != (
        config.observed.sensitivity_changed_group_ids
    ):
        raise UpcTrainingPolicyError("전체 월 변경 group 목록이 정책 근거와 다릅니다.")
    if (
        sensitivity["changed_cell_count"]
        != config.observed.sensitivity_changed_cell_count
    ):
        raise UpcTrainingPolicyError("전체 월 변경 cell 수가 정책 근거와 다릅니다.")
    if primary["changed_group_ids"] or primary["changed_cell_count"]:
        raise UpcTrainingPolicyError(
            "train_only은 순서 변경에 완전히 안정적이어야 합니다."
        )

    primary_stability_passed = (
        primary["agreement_ratio"] >= config.minimum_order_agreement
    )
    sensitivity_stability_passed = (
        sensitivity["agreement_ratio"] >= config.minimum_order_agreement
    )
    if not primary_stability_passed:
        raise UpcTrainingPolicyError(
            "train_only이 학습 허용 안정성 임계값을 통과하지 못했습니다."
        )
    if sensitivity_stability_passed:
        raise UpcTrainingPolicyError(
            "전체 월 결과가 임계값을 통과해 현재 차단 결정의 근거와 모순됩니다."
        )
    return {
        "analyses": analyses,
        "gates": {
            PRIMARY_PROTOCOL: {
                "role": "primary_no_future_leakage",
                "order_stability_passed": primary_stability_passed,
                "model_training_allowed": True,
                "reason": (
                    "primary protocol, train-period-only scaling and 100% "
                    "ascending/descending membership agreement"
                ),
            },
            SENSITIVITY_PROTOCOL: {
                "role": "clustering_sensitivity_only",
                "order_stability_passed": sensitivity_stability_passed,
                "model_training_allowed": False,
                "reason": (
                    "50.48% order agreement is below the preregistered 95% "
                    "threshold; neural training is deferred"
                ),
            },
            FORBIDDEN_PROTOCOL: {
                "role": "figure4_diagnostic_only",
                "order_stability_passed": None,
                "model_training_allowed": False,
                "reason": "unreported Fig. 4 probe is forbidden as model input",
            },
        },
        "aggregate_gate": {
            "ready_for_primary_model_training": True,
            "ready_for_all_preregistered_protocol_training": False,
            "allowed_model_training_protocols": list(config.allowed_protocols),
            "blocked_model_training_protocols": list(config.blocked_protocols),
        },
    }


def verify_policy_evidence(
    config: UpcTrainingPolicyConfig,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    """PCC config, manifest와 고정한 4개 산출물의 checksum을 검증한다."""

    actual_pcc_config_sha = compute_sha256(config.pcc_config_path)
    if actual_pcc_config_sha != config.pcc_config_sha256:
        raise UpcTrainingPolicyError(
            "PCC config checksum이 학습 정책에 고정한 값과 다릅니다."
        )
    pcc_manifest = _load_json(config.pcc_manifest_path, "PCC manifest")
    if pcc_manifest.get("status") != config.required_pcc_manifest_status:
        raise UpcTrainingPolicyError("PCC manifest status가 정책 계약과 다릅니다.")
    manifest_config = _require_mapping(
        pcc_manifest.get("config"), "pcc_manifest.config"
    )
    if manifest_config.get("sha256") != actual_pcc_config_sha:
        raise UpcTrainingPolicyError("PCC manifest의 config checksum이 다릅니다.")
    outputs = _require_mapping(pcc_manifest.get("outputs"), "pcc_manifest.outputs")
    verified_outputs: dict[str, Any] = {}
    verified_paths: dict[str, Path] = {}
    for key in PROTECTED_OUTPUT_KEYS:
        metadata = _require_mapping(outputs.get(key), f"pcc_manifest.outputs.{key}")
        path, verified = _verify_protected_file(
            metadata,
            config.protected_output_sha256[key],
            f"pcc_manifest.outputs.{key}",
            config.base_directory,
        )
        verified_paths[key] = path
        verified_outputs[key] = verified

    summary = _load_json(verified_paths["summary_json"], "PCC summary")
    manifest_summary = _require_mapping(
        pcc_manifest.get("summary"), "pcc_manifest.summary"
    )
    if summary != manifest_summary:
        raise UpcTrainingPolicyError(
            "PCC summary 파일과 manifest 내 summary가 다릅니다."
        )
    group_rows = load_group_assignment_rows(verified_paths["group_assignments_csv"])
    evaluated = evaluate_training_policy(config, summary, group_rows)
    evidence_metadata = {
        "pcc_config": {
            "path": _display_path(config.pcc_config_path),
            "sha256": actual_pcc_config_sha,
        },
        "pcc_manifest": {
            "path": _display_path(config.pcc_manifest_path),
            "status": pcc_manifest.get("status"),
        },
        "protected_outputs": verified_outputs,
    }
    return pcc_manifest, evaluated, evidence_metadata


def build_policy_payload(
    config: UpcTrainingPolicyConfig,
    pcc_manifest: Mapping[str, Any],
    evaluated: Mapping[str, Any],
    evidence_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """실행 시각을 제외한 결정론적 machine-readable 정책을 만든다."""

    upstream_disclosure = _require_mapping(
        pcc_manifest.get("upstream_disclosure"), "pcc_manifest.upstream_disclosure"
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "decision_stage": config.decision_stage,
        "decision_character": "post_clustering_pre_model_scope_decision",
        "model_performance_used_for_decision": (
            config.model_performance_used_for_decision
        ),
        "source_disclosure": {
            "pcc_manifest_status": pcc_manifest.get("status"),
            "upstream_initial_group_status": upstream_disclosure.get("status"),
            "figure4_exact_match": upstream_disclosure.get("figure4_exact_match"),
            "clustering_global_ready_for_expensive_model_training": False,
            "clustering_global_gate_value_preserved": True,
            "interpretation": (
                "the clustering manifest keeps its conservative global false gate; "
                "this policy scopes training eligibility by protocol without "
                "changing any clustering membership"
            ),
        },
        "evidence": {
            **evidence_metadata,
            "minimum_order_sensitivity_agreement": config.minimum_order_agreement,
            "order_change_analysis": evaluated["analyses"],
        },
        "training_gates": evaluated["gates"],
        "aggregate_gate": evaluated["aggregate_gate"],
        "decision_rules": {
            "reproduction_assignment_order": config.reproduction_assignment_order,
            "order_sensitivity_role": config.order_sensitivity_role,
            "full_month_neural_training": config.full_month_neural_training,
            "figure4_probe_training": config.figure4_probe_training,
            "order_invariant_extension": config.order_invariant_extension,
            "post_result_order_search_permitted": False,
            "membership_selection_by_model_performance_permitted": False,
        },
    }


def require_training_allowed(policy: Mapping[str, Any], protocol: str) -> None:
    """후속 학습기가 호출할 수 있는 protocol allowlist 검사다."""

    gates = _require_mapping(policy.get("training_gates"), "policy.training_gates")
    if protocol not in gates:
        raise UpcTrainingPolicyError(f"학습 정책에 없는 protocol입니다: {protocol}")
    gate = _require_mapping(gates[protocol], f"policy.training_gates.{protocol}")
    if gate.get("model_training_allowed") is not True:
        raise UpcTrainingPolicyError(
            f"학습이 차단된 protocol입니다: {protocol}; reason={gate.get('reason')}"
        )


def run_training_policy(config: UpcTrainingPolicyConfig) -> dict[str, Any]:
    """근거를 검증하고 정책과 실행 manifest를 원자적으로 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    pcc_manifest, evaluated, evidence_metadata = verify_policy_evidence(config)
    policy = build_policy_payload(config, pcc_manifest, evaluated, evidence_metadata)
    output_paths = {
        "policy": config.outputs.policy,
        "manifest": config.outputs.manifest,
    }
    temporary_paths = {key: _temporary_path(path) for key, path in output_paths.items()}
    for output in output_paths.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    published = False
    try:
        temporary_paths["policy"].write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        policy_metadata = {
            "path": _display_path(config.outputs.policy),
            "size_bytes": temporary_paths["policy"].stat().st_size,
            "sha256": compute_sha256(temporary_paths["policy"]),
        }
        finished_at = datetime.now(timezone.utc)
        execution_manifest = {
            "schema_version": 1,
            "status": "complete",
            "tool": {
                "name": "scripts.validate_upc_training_policy",
                "version": TOOL_VERSION,
            },
            "created_at_utc": finished_at.isoformat(),
            "git": _git_state(),
            "config": {
                "path": _display_path(config.path),
                "sha256": compute_sha256(config.path),
            },
            "inputs": {
                "pcc_manifest": {
                    "path": _display_path(config.pcc_manifest_path),
                    "sha256": compute_sha256(config.pcc_manifest_path),
                    "status": pcc_manifest.get("status"),
                },
                "protected_outputs": evidence_metadata["protected_outputs"],
            },
            "decision": {
                "ready_for_primary_model_training": True,
                "ready_for_all_preregistered_protocol_training": False,
                "allowed_model_training_protocols": list(config.allowed_protocols),
                "blocked_model_training_protocols": list(config.blocked_protocols),
            },
            "outputs": {"policy": policy_metadata},
            "runtime": {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started_counter,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "execution_location": "local_wsl_cpu",
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        }
        temporary_paths["manifest"].write_text(
            json.dumps(execution_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_paths["policy"], config.outputs.policy)
        os.replace(temporary_paths["manifest"], config.outputs.manifest)
        published = True
        return policy
    finally:
        if not published:
            for temporary in temporary_paths.values():
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "UPC 순서 민감도 근거를 검증하고 프로토콜별 모델 학습 정책을 게시합니다."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"UPC training policy config (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--check-protocol",
        help="정책 생성 후 특정 protocol의 학습 허용 여부를 확인합니다.",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_policy_config(args.config)
        policy = run_training_policy(config)
        if args.check_protocol:
            require_training_allowed(policy, args.check_protocol)
    except UpcTrainingPolicyError as exc:
        print(f"UPC 학습 정책 검증 실패: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"파일 시스템 오류: {exc}", file=sys.stderr)
        return 2
    print(
        "UPC 학습 정책 생성 완료: "
        f"primary_ready={policy['aggregate_gate']['ready_for_primary_model_training']}, "
        "all_protocols_ready="
        f"{policy['aggregate_gate']['ready_for_all_preregistered_protocol_training']}"
    )
    if args.check_protocol:
        print(f"학습 허용 protocol: {args.check_protocol}")
    print(f"policy={_display_path(config.outputs.policy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
