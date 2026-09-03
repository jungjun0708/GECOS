#!/usr/bin/env python3
"""LSTM Train-only 셀별 Min-Max 제한 pilot의 사전 등록 계약."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from scripts.build_upc_initial_groups import REPOSITORY_ROOT, compute_sha256
from scripts.lstm_contract import (
    LstmArchitectureSpec,
    LstmPassCriteria,
    LstmSelectionSpec,
    LstmSmokeContractError,
    LstmTrainingSpec,
    LstmUpcSmokeConfig,
    load_lstm_smoke_config,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "lstm_scaling_pilot_milan_nov2013.json"
EXPECTED_NAME = "milan-internet-2013-11-lstm-train-only-per-cell-minmax-pilot"
EXPECTED_STAGE = "post_raw_smoke_pre_full_training"
EXPECTED_SCALING_NAME = "per_cell_train_only_minmax"
EXPECTED_SPLITS = ("train", "validation")
EXPECTED_TEST_POLICY = "withheld_not_bundled_or_evaluated"
EXPECTED_RAW_METRICS = (
    ("persistence_selected_smoke", 32.2166927947, 0.116521760244),
    ("lstm_upc_off", 236.285288592, 0.854599754295),
    ("lstm_upc_on", 247.777719229, 0.896165729294),
)


class LstmScalingContractError(LstmSmokeContractError):
    """Scaling pilot 설정·입력·판정 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class BaseSmokeReference:
    config_path: Path
    config_sha256: str
    input_npz_sha256: str
    evaluation_report_path: Path
    evaluation_report_sha256: str
    required_status: str


@dataclass(frozen=True)
class ScalerSourceSpec:
    central_manifest_path: Path
    central_traffic_path: Path
    expected_central_manifest_sha256: str
    expected_central_traffic_sha256: str


@dataclass(frozen=True)
class ScalingSpec:
    name: str
    only_changed_factor: str
    fit_partition: str
    fit_start_index_inclusive: int
    fit_end_index_exclusive: int
    fit_values: str
    formula: str
    inverse_formula: str
    dtype: str
    clip_transform: bool
    clip_inverse_prediction: bool
    zero_range_policy: str
    expected_zero_range_cell_count: int
    roundtrip_max_absolute_error: float


@dataclass(frozen=True)
class RawMetricReference:
    model: str
    mae: float
    wape: float


@dataclass(frozen=True)
class RawValidationReference:
    split: str
    target_policy: str
    aggregation: str
    primary_model: str
    metrics: tuple[RawMetricReference, ...]

    def metric_for(self, model: str) -> RawMetricReference:
        for metric in self.metrics:
            if metric.model == model:
                return metric
        raise LstmScalingContractError(f"raw reference에 없는 모델입니다: {model}")


@dataclass(frozen=True)
class ScalingDecisionRule:
    metric: str
    primary_model: str
    material_improvement_fraction: float
    material_improvement_max_mae: float
    material_improvement_outcome: str
    positive_but_below_material_outcome: str
    no_improvement_outcome: str
    test_metric_used_for_decision: bool
    persistence_used_as_pass_gate: bool
    upc_on_off_difference_used_for_scaling_decision: bool


@dataclass(frozen=True)
class ScalingPilotOutputPaths:
    input_npz: Path
    input_manifest: Path
    architecture_report: Path
    evaluation_report: Path
    predictions_npz: Path
    per_cell_metrics_csv: Path
    run_manifest: Path

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class LstmScalingPilotConfig:
    path: Path
    base_directory: Path
    name: str
    decision_stage: str
    question: str
    base_reference: BaseSmokeReference
    base_smoke: LstmUpcSmokeConfig
    scaler_source: ScalerSourceSpec
    scaling: ScalingSpec
    selection: LstmSelectionSpec
    test_policy: str
    raw_reference: RawValidationReference
    decision_rule: ScalingDecisionRule
    outputs: ScalingPilotOutputPaths
    training: LstmTrainingSpec

    @property
    def architecture(self) -> LstmArchitectureSpec:
        return self.base_smoke.architecture

    @property
    def seed(self) -> int:
        return self.base_smoke.seed

    @property
    def upc_protocol(self) -> str:
        return self.base_smoke.upc_protocol

    @property
    def pass_criteria(self) -> LstmPassCriteria:
        return self.base_smoke.pass_criteria


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LstmScalingContractError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LstmScalingContractError(
            f"{label}이 올바른 JSON이 아닙니다: {exc}"
        ) from exc
    return _require_mapping(value, label)


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise LstmScalingContractError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise LstmScalingContractError(f"{field}는 JSON array여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LstmScalingContractError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LstmScalingContractError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LstmScalingContractError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise LstmScalingContractError(f"{field}는 유한한 숫자여야 합니다.")
    if minimum is not None and result < minimum:
        raise LstmScalingContractError(f"{field}는 {minimum} 이상이어야 합니다.")
    if maximum is not None and result > maximum:
        raise LstmScalingContractError(f"{field}는 {maximum} 이하여야 합니다.")
    return result


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise LstmScalingContractError(f"{field}는 boolean이어야 합니다.")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = _require_string(value, field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise LstmScalingContractError(f"{field}의 SHA-256 형식이 올바르지 않습니다.")
    return text


def _resolve_path(value: object, field: str, base_directory: Path) -> Path:
    path = Path(_require_string(value, field))
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _expect(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise LstmScalingContractError(
            f"{field}는 사전 등록값 {expected!r}이어야 합니다: {actual!r}"
        )


def _load_raw_metrics(value: object) -> tuple[RawMetricReference, ...]:
    mapping = _require_mapping(value, "raw_validation_reference.metrics")
    metrics: list[RawMetricReference] = []
    for model, expected_mae, expected_wape in EXPECTED_RAW_METRICS:
        row = _require_mapping(
            mapping.get(model), f"raw_validation_reference.metrics.{model}"
        )
        metric = RawMetricReference(
            model=model,
            mae=_require_number(
                row.get("mae"), f"raw_validation_reference.metrics.{model}.mae"
            ),
            wape=_require_number(
                row.get("wape"), f"raw_validation_reference.metrics.{model}.wape"
            ),
        )
        _expect(metric.mae, expected_mae, f"raw reference {model} MAE")
        _expect(metric.wape, expected_wape, f"raw reference {model} WAPE")
        metrics.append(metric)
    if set(mapping) != {row[0] for row in EXPECTED_RAW_METRICS}:
        raise LstmScalingContractError(
            "raw reference 모델 집합이 사전 등록값과 다릅니다."
        )
    return tuple(metrics)


def load_lstm_scaling_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> LstmScalingPilotConfig:
    """결과 전 고정한 셀별 Train-only Min-Max pilot 설정을 엄격하게 읽는다."""

    root = _load_json(path, "LSTM scaling pilot config")
    _expect(
        _require_int(root.get("schema_version"), "schema_version"), 1, "schema_version"
    )
    name = _require_string(root.get("name"), "name")
    decision_stage = _require_string(root.get("decision_stage"), "decision_stage")
    question = _require_string(root.get("question"), "question")
    _expect(name, EXPECTED_NAME, "name")
    _expect(decision_stage, EXPECTED_STAGE, "decision_stage")

    raw_base = _require_mapping(root.get("base_smoke"), "base_smoke")
    base_reference = BaseSmokeReference(
        config_path=_resolve_path(
            raw_base.get("config"), "base_smoke.config", base_directory
        ),
        config_sha256=_require_sha256(
            raw_base.get("config_sha256"), "base_smoke.config_sha256"
        ),
        input_npz_sha256=_require_sha256(
            raw_base.get("input_npz_sha256"), "base_smoke.input_npz_sha256"
        ),
        evaluation_report_path=_resolve_path(
            raw_base.get("evaluation_report"),
            "base_smoke.evaluation_report",
            base_directory,
        ),
        evaluation_report_sha256=_require_sha256(
            raw_base.get("evaluation_report_sha256"),
            "base_smoke.evaluation_report_sha256",
        ),
        required_status=_require_string(
            raw_base.get("required_status"), "base_smoke.required_status"
        ),
    )
    if compute_sha256(base_reference.config_path) != base_reference.config_sha256:
        raise LstmScalingContractError(
            "기준 LSTM smoke config checksum이 사전 등록값과 다릅니다."
        )
    base_smoke = load_lstm_smoke_config(
        base_reference.config_path, base_directory=base_directory
    )
    _expect(base_reference.required_status, "pass", "base_smoke.required_status")

    raw_source = _require_mapping(root.get("scaler_source"), "scaler_source")
    scaler_source = ScalerSourceSpec(
        central_manifest_path=_resolve_path(
            raw_source.get("central_manifest"),
            "scaler_source.central_manifest",
            base_directory,
        ),
        central_traffic_path=_resolve_path(
            raw_source.get("central_traffic"),
            "scaler_source.central_traffic",
            base_directory,
        ),
        expected_central_manifest_sha256=_require_sha256(
            raw_source.get("expected_central_manifest_sha256"),
            "scaler_source.expected_central_manifest_sha256",
        ),
        expected_central_traffic_sha256=_require_sha256(
            raw_source.get("expected_central_traffic_sha256"),
            "scaler_source.expected_central_traffic_sha256",
        ),
    )

    raw_scaling = _require_mapping(root.get("scaling"), "scaling")
    scaling = ScalingSpec(
        name=_require_string(raw_scaling.get("name"), "scaling.name"),
        only_changed_factor=_require_string(
            raw_scaling.get("only_changed_factor"), "scaling.only_changed_factor"
        ),
        fit_partition=_require_string(
            raw_scaling.get("fit_partition"), "scaling.fit_partition"
        ),
        fit_start_index_inclusive=_require_int(
            raw_scaling.get("fit_start_index_inclusive"),
            "scaling.fit_start_index_inclusive",
        ),
        fit_end_index_exclusive=_require_int(
            raw_scaling.get("fit_end_index_exclusive"),
            "scaling.fit_end_index_exclusive",
            minimum=1,
        ),
        fit_values=_require_string(raw_scaling.get("fit_values"), "scaling.fit_values"),
        formula=_require_string(raw_scaling.get("formula"), "scaling.formula"),
        inverse_formula=_require_string(
            raw_scaling.get("inverse_formula"), "scaling.inverse_formula"
        ),
        dtype=_require_string(raw_scaling.get("dtype"), "scaling.dtype"),
        clip_transform=_require_bool(
            raw_scaling.get("clip_transform"), "scaling.clip_transform"
        ),
        clip_inverse_prediction=_require_bool(
            raw_scaling.get("clip_inverse_prediction"),
            "scaling.clip_inverse_prediction",
        ),
        zero_range_policy=_require_string(
            raw_scaling.get("zero_range_policy"), "scaling.zero_range_policy"
        ),
        expected_zero_range_cell_count=_require_int(
            raw_scaling.get("expected_zero_range_cell_count"),
            "scaling.expected_zero_range_cell_count",
        ),
        roundtrip_max_absolute_error=_require_number(
            raw_scaling.get("roundtrip_max_absolute_error"),
            "scaling.roundtrip_max_absolute_error",
            minimum=0.0,
        ),
    )
    expected_scaling_values = {
        "name": EXPECTED_SCALING_NAME,
        "only_changed_factor": "model_input_and_target_scaling",
        "fit_partition": "train",
        "fit_start_index_inclusive": 0,
        "fit_end_index_exclusive": 2880,
        "fit_values": "existing_preprocessed_filled_traffic",
        "formula": "scaled=(value-cell_train_min)/(cell_train_max-cell_train_min)",
        "inverse_formula": "value=scaled*(cell_train_max-cell_train_min)+cell_train_min",
        "dtype": "float32",
        "clip_transform": False,
        "clip_inverse_prediction": False,
        "zero_range_policy": "reject",
        "expected_zero_range_cell_count": 0,
        "roundtrip_max_absolute_error": 0.001,
    }
    for field, expected in expected_scaling_values.items():
        _expect(getattr(scaling, field), expected, f"scaling.{field}")

    raw_selection = _require_mapping(root.get("selection"), "selection")
    splits = tuple(
        _require_string(value, f"selection.splits[{index}]")
        for index, value in enumerate(
            _require_list(raw_selection.get("splits"), "selection.splits")
        )
    )
    raw_cluster_counts = _require_mapping(
        raw_selection.get("expected_cluster_counts"),
        "selection.expected_cluster_counts",
    )
    try:
        cluster_counts = tuple(
            sorted(
                (
                    int(cluster_id),
                    _require_int(
                        count,
                        f"selection.expected_cluster_counts.{cluster_id}",
                        minimum=1,
                    ),
                )
                for cluster_id, count in raw_cluster_counts.items()
            )
        )
    except ValueError as exc:
        raise LstmScalingContractError(
            "cluster ID는 정수 문자열이어야 합니다."
        ) from exc
    selection = LstmSelectionSpec(
        spatial_scope=_require_string(
            raw_selection.get("spatial_scope"), "selection.spatial_scope"
        ),
        splits=splits,
        targets_per_cell_per_split=_require_int(
            raw_selection.get("targets_per_cell_per_split"),
            "selection.targets_per_cell_per_split",
            minimum=2,
        ),
        target_policy=base_smoke.selection.target_policy,
        expected_central_cell_count=_require_int(
            raw_selection.get("expected_central_cell_count"),
            "selection.expected_central_cell_count",
            minimum=1,
        ),
        expected_cluster_counts=cluster_counts,
    )
    test_policy = _require_string(
        raw_selection.get("test_policy"), "selection.test_policy"
    )
    if (
        selection.spatial_scope != "same_all_central_900_as_base_smoke"
        or selection.splits != EXPECTED_SPLITS
        or selection.targets_per_cell_per_split
        != base_smoke.selection.targets_per_cell_per_split
        or selection.expected_central_cell_count
        != base_smoke.selection.expected_central_cell_count
        or selection.expected_cluster_counts
        != base_smoke.selection.expected_cluster_counts
        or test_policy != EXPECTED_TEST_POLICY
    ):
        raise LstmScalingContractError("pilot selection이 기준 smoke와 다릅니다.")

    raw_reference_mapping = _require_mapping(
        root.get("raw_validation_reference"), "raw_validation_reference"
    )
    raw_reference = RawValidationReference(
        split=_require_string(
            raw_reference_mapping.get("split"), "raw_validation_reference.split"
        ),
        target_policy=_require_string(
            raw_reference_mapping.get("target_policy"),
            "raw_validation_reference.target_policy",
        ),
        aggregation=_require_string(
            raw_reference_mapping.get("aggregation"),
            "raw_validation_reference.aggregation",
        ),
        primary_model=_require_string(
            raw_reference_mapping.get("primary_model"),
            "raw_validation_reference.primary_model",
        ),
        metrics=_load_raw_metrics(raw_reference_mapping.get("metrics")),
    )
    _expect(raw_reference.split, "validation", "raw_validation_reference.split")
    _expect(
        raw_reference.target_policy,
        "all_targets",
        "raw_validation_reference.target_policy",
    )
    _expect(raw_reference.aggregation, "micro", "raw_validation_reference.aggregation")
    _expect(raw_reference.primary_model, "lstm_upc_off", "raw primary_model")

    raw_decision = _require_mapping(root.get("decision_rule"), "decision_rule")
    decision = ScalingDecisionRule(
        metric=_require_string(raw_decision.get("metric"), "decision_rule.metric"),
        primary_model=_require_string(
            raw_decision.get("primary_model"), "decision_rule.primary_model"
        ),
        material_improvement_fraction=_require_number(
            raw_decision.get("material_improvement_fraction"),
            "decision_rule.material_improvement_fraction",
            minimum=0.0,
            maximum=1.0,
        ),
        material_improvement_max_mae=_require_number(
            raw_decision.get("material_improvement_max_mae"),
            "decision_rule.material_improvement_max_mae",
            minimum=0.0,
        ),
        material_improvement_outcome=_require_string(
            raw_decision.get("material_improvement_outcome"),
            "decision_rule.material_improvement_outcome",
        ),
        positive_but_below_material_outcome=_require_string(
            raw_decision.get("positive_but_below_material_outcome"),
            "decision_rule.positive_but_below_material_outcome",
        ),
        no_improvement_outcome=_require_string(
            raw_decision.get("no_improvement_outcome"),
            "decision_rule.no_improvement_outcome",
        ),
        test_metric_used_for_decision=_require_bool(
            raw_decision.get("test_metric_used_for_decision"),
            "decision_rule.test_metric_used_for_decision",
        ),
        persistence_used_as_pass_gate=_require_bool(
            raw_decision.get("persistence_used_as_pass_gate"),
            "decision_rule.persistence_used_as_pass_gate",
        ),
        upc_on_off_difference_used_for_scaling_decision=_require_bool(
            raw_decision.get("upc_on_off_difference_used_for_scaling_decision"),
            "decision_rule.upc_on_off_difference_used_for_scaling_decision",
        ),
    )
    _expect(decision.metric, "validation_all_targets_micro_mae", "decision metric")
    _expect(decision.primary_model, "lstm_upc_off", "decision primary_model")
    _expect(decision.material_improvement_fraction, 0.2, "material fraction")
    raw_primary_mae = raw_reference.metric_for(decision.primary_model).mae
    expected_max_mae = raw_primary_mae * (1 - decision.material_improvement_fraction)
    if not math.isclose(
        decision.material_improvement_max_mae,
        expected_max_mae,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise LstmScalingContractError(
            "material improvement MAE가 raw 기준과 20% 규칙에서 계산되지 않았습니다."
        )
    if (
        decision.test_metric_used_for_decision
        or decision.persistence_used_as_pass_gate
        or decision.upc_on_off_difference_used_for_scaling_decision
    ):
        raise LstmScalingContractError(
            "Test, Persistence 또는 UPC 차이를 scaling 선택 gate로 사용할 수 없습니다."
        )

    raw_outputs = _require_mapping(root.get("outputs"), "outputs")
    outputs = ScalingPilotOutputPaths(
        **{
            field: _resolve_path(
                raw_outputs.get(field), f"outputs.{field}", base_directory
            )
            for field in ScalingPilotOutputPaths.__dataclass_fields__
        }
    )
    if len(set(outputs.as_dict().values())) != len(outputs.as_dict()):
        raise LstmScalingContractError(
            "scaling pilot output 경로는 서로 달라야 합니다."
        )
    if any(
        "lstm_scaling_pilot" not in str(value) for value in outputs.as_dict().values()
    ):
        raise LstmScalingContractError(
            "scaling pilot output은 별도 lstm_scaling_pilot 경로를 사용해야 합니다."
        )

    training = replace(
        base_smoke.training,
        input_scaling=EXPECTED_SCALING_NAME,
        test_role=EXPECTED_TEST_POLICY,
    )
    return LstmScalingPilotConfig(
        path=path.resolve(),
        base_directory=base_directory.resolve(),
        name=name,
        decision_stage=decision_stage,
        question=question,
        base_reference=base_reference,
        base_smoke=base_smoke,
        scaler_source=scaler_source,
        scaling=scaling,
        selection=selection,
        test_policy=test_policy,
        raw_reference=raw_reference,
        decision_rule=decision,
        outputs=outputs,
        training=training,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "LstmScalingContractError",
    "LstmScalingPilotConfig",
    "RawMetricReference",
    "ScalingDecisionRule",
    "ScalingSpec",
    "load_lstm_scaling_config",
]
