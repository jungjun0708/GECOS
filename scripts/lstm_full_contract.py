#!/usr/bin/env python3
"""중앙 900셀 LSTM 전체 Train·Validation 학습의 사전 등록 계약."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_upc_initial_groups import REPOSITORY_ROOT, compute_sha256
from scripts.lstm_contract import LstmArchitectureSpec, LstmSmokeContractError
from scripts.lstm_scaling_contract import load_lstm_scaling_config

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "lstm_full_training_milan_nov2013.json"
EXPECTED_NAME = "milan-internet-2013-11-lstm-full-train-validation"
EXPECTED_STAGE = "post_scaling_pilot_pre_test_evaluation"
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_CONDITIONS = (
    "upc_off",
    "upc_on_cluster_0",
    "upc_on_cluster_1",
)
EXPECTED_CLUSTER_COUNTS = ((0, 611), (1, 289))
EXPECTED_SPLITS = ("train", "validation")
EXPECTED_TARGET_COUNTS = {"train": 2872, "validation": 720}
EXPECTED_SOURCE_KEYS = (
    "scaling_pilot_config",
    "scaling_pilot_evaluation",
    "forecast_config",
    "central_manifest",
    "central_traffic",
    "central_missing_mask",
    "central_internet_null_mask",
    "timestamps_ms",
    "upc_training_policy",
    "central_memberships",
    "baseline_summary",
)


class LstmFullContractError(LstmSmokeContractError):
    """전체 LSTM Train·Validation 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class SourceReference:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class FullSplitSpec:
    name: str
    target_start_index_inclusive: int
    target_end_index_exclusive: int
    targets_per_cell: int
    samples: int


@dataclass(frozen=True)
class FullDataSpec:
    spatial_scope: str
    expected_cell_count: int
    input_length: int
    horizon: int
    bundle_global_start_index_inclusive: int
    bundle_global_end_index_exclusive: int
    splits: tuple[FullSplitSpec, ...]
    test_policy: str
    test_target_start_index_inclusive: int
    known_prior_test_exposure: str
    future_test_claim: str

    def split(self, name: str) -> FullSplitSpec:
        for row in self.splits:
            if row.name == name:
                return row
        raise LstmFullContractError(f"등록되지 않은 split입니다: {name}")


@dataclass(frozen=True)
class FullScalingSpec:
    name: str
    fit_start_index_inclusive: int
    fit_end_index_exclusive: int
    fit_values: str
    dtype: str
    clip_transform: bool
    clip_inverse_prediction: bool
    zero_range_policy: str
    expected_zero_range_cell_count: int
    roundtrip_max_absolute_error: float


@dataclass(frozen=True)
class FullUpcSpec:
    protocol: str
    conditions: tuple[str, ...]
    expected_cluster_counts: tuple[tuple[int, int], ...]
    prediction_recombination: str


@dataclass(frozen=True)
class EarlyStoppingSpec:
    monitor: str
    monitor_domain: str
    mode: str
    patience: int
    min_delta: float
    restore_best_weights: bool
    start_from_epoch: int


@dataclass(frozen=True)
class FullTrainingSpec:
    seeds: tuple[int, ...]
    expected_job_count: int
    optimizer: str
    learning_rate: float
    loss: str
    loss_domain: str
    batch_size: int
    max_epochs: int
    dropout: float
    shuffle: bool
    early_stopping: EarlyStoppingSpec
    checkpoint_format: str
    checkpoint_selection: str
    maximum_wall_clock_seconds_per_job: int
    wall_clock_limit_outcome: str


@dataclass(frozen=True)
class ValidationEvaluationSpec:
    evaluated_split: str
    target_policies: tuple[str, ...]
    metrics: tuple[str, ...]
    aggregations: tuple[str, ...]
    report_unit: str
    mape_positive_targets_only: bool
    seed_summary: str
    performance_used_as_pipeline_gate: bool
    upc_condition_used_for_exclusion: bool


@dataclass(frozen=True)
class ResourcePolicy:
    local_tensorflow_training: bool
    local_peak_rss_limit_bytes: int
    required_colab_gpu_name_contains: str
    colab_peak_rss_soft_limit_bytes: int
    job_execution: str
    rerun_policy: str


@dataclass(frozen=True)
class FullPassCriteria:
    require_clean_source_git: bool
    require_source_checksums: bool
    require_test_absent: bool
    require_exact_parameter_count: bool
    require_all_nine_jobs: bool
    require_finite_history_weights_and_predictions: bool
    require_best_weights_restored: bool
    require_exact_cluster_recombination: bool
    require_finite_validation_metrics: bool
    require_better_than_persistence: bool
    require_upc_improvement: bool


@dataclass(frozen=True)
class FullOutputPaths:
    input_npz: Path
    input_manifest: Path
    job_descriptors_dir: Path
    jobs_root: Path
    training_jobs_csv: Path
    validation_report: Path
    validation_predictions_npz: Path
    validation_per_cell_metrics_csv: Path
    release_manifest: Path
    aggregation_manifest: Path

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class FullJobSpec:
    job_id: str
    seed: int
    condition: str
    cluster_id: int | None
    expected_cell_count: int


@dataclass(frozen=True)
class LstmFullTrainingConfig:
    path: Path
    base_directory: Path
    name: str
    decision_stage: str
    question: str
    sources: tuple[SourceReference, ...]
    required_scaling_outcome: str
    data: FullDataSpec
    scaling: FullScalingSpec
    upc: FullUpcSpec
    architecture: LstmArchitectureSpec
    training: FullTrainingSpec
    validation: ValidationEvaluationSpec
    resources: ResourcePolicy
    pass_criteria: FullPassCriteria
    outputs: FullOutputPaths

    def source(self, name: str) -> SourceReference:
        for row in self.sources:
            if row.name == name:
                return row
        raise LstmFullContractError(f"등록되지 않은 source입니다: {name}")

    @property
    def jobs(self) -> tuple[FullJobSpec, ...]:
        rows: list[FullJobSpec] = []
        counts = dict(self.upc.expected_cluster_counts)
        for seed in self.training.seeds:
            rows.append(
                FullJobSpec(
                    job_id=f"seed_{seed}_upc_off",
                    seed=seed,
                    condition="upc_off",
                    cluster_id=None,
                    expected_cell_count=self.data.expected_cell_count,
                )
            )
            for cluster_id in sorted(counts):
                rows.append(
                    FullJobSpec(
                        job_id=f"seed_{seed}_upc_on_cluster_{cluster_id}",
                        seed=seed,
                        condition=f"upc_on_cluster_{cluster_id}",
                        cluster_id=cluster_id,
                        expected_cell_count=counts[cluster_id],
                    )
                )
        return tuple(rows)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise LstmFullContractError(f"{field}는 JSON object여야 합니다.")
    return value


def _list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise LstmFullContractError(f"{field}는 JSON array여야 합니다.")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LstmFullContractError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LstmFullContractError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LstmFullContractError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise LstmFullContractError(f"{field}는 유한해야 합니다.")
    if minimum is not None and result < minimum:
        raise LstmFullContractError(f"{field}는 {minimum} 이상이어야 합니다.")
    if maximum is not None and result > maximum:
        raise LstmFullContractError(f"{field}는 {maximum} 이하여야 합니다.")
    return result


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise LstmFullContractError(f"{field}는 boolean이어야 합니다.")
    return value


def _sha256(value: object, field: str) -> str:
    text = _string(value, field)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise LstmFullContractError(f"{field}의 SHA-256 형식이 올바르지 않습니다.")
    return text


def _path(value: object, field: str, base_directory: Path) -> Path:
    result = Path(_string(value, field))
    if not result.is_absolute():
        result = base_directory / result
    return result.resolve()


def _expect(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise LstmFullContractError(
            f"{field}는 사전 등록값 {expected!r}이어야 합니다: {actual!r}"
        )


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LstmFullContractError(f"config를 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LstmFullContractError(f"config JSON이 올바르지 않습니다: {exc}") from exc
    return _mapping(value, "config")


def _load_sources(
    raw: Mapping[str, Any], base_directory: Path
) -> tuple[tuple[SourceReference, ...], str]:
    allowed = {
        "required_scaling_outcome",
        *(name for key in EXPECTED_SOURCE_KEYS for name in (key, f"{key}_sha256")),
    }
    if set(raw) != allowed:
        raise LstmFullContractError("sources key 집합이 사전 등록값과 다릅니다.")
    rows = tuple(
        SourceReference(
            name=name,
            path=_path(raw.get(name), f"sources.{name}", base_directory),
            sha256=_sha256(raw.get(f"{name}_sha256"), f"sources.{name}_sha256"),
        )
        for name in EXPECTED_SOURCE_KEYS
    )
    outcome = _string(
        raw.get("required_scaling_outcome"), "sources.required_scaling_outcome"
    )
    _expect(outcome, "adopt_as_full_training_scaling_candidate", "scaling outcome")
    scaling_reference = next(row for row in rows if row.name == "scaling_pilot_config")
    if compute_sha256(scaling_reference.path) != scaling_reference.sha256:
        raise LstmFullContractError("scaling pilot config checksum이 다릅니다.")
    scaling_config = load_lstm_scaling_config(
        scaling_reference.path, base_directory=base_directory
    )
    _expect(scaling_config.scaling.name, "per_cell_train_only_minmax", "scaling name")
    return rows, outcome


def load_lstm_full_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> LstmFullTrainingConfig:
    """결과 전 고정한 전체 LSTM Train·Validation 설정을 엄격하게 읽는다."""

    root = _read_json(path)
    _expect(
        _integer(root.get("schema_version"), "schema_version", minimum=1), 1, "schema"
    )
    name = _string(root.get("name"), "name")
    stage = _string(root.get("decision_stage"), "decision_stage")
    question = _string(root.get("question"), "question")
    _expect(name, EXPECTED_NAME, "name")
    _expect(stage, EXPECTED_STAGE, "decision_stage")
    sources, required_outcome = _load_sources(
        _mapping(root.get("sources"), "sources"), base_directory
    )

    raw_data = _mapping(root.get("data"), "data")
    raw_splits = _mapping(raw_data.get("splits"), "data.splits")
    if tuple(raw_splits) != EXPECTED_SPLITS:
        raise LstmFullContractError("Train·Validation 외 split을 등록할 수 없습니다.")
    splits: list[FullSplitSpec] = []
    expected_boundaries = {"train": (8, 2880), "validation": (2880, 3600)}
    for split_name in EXPECTED_SPLITS:
        row = _mapping(raw_splits.get(split_name), f"data.splits.{split_name}")
        split = FullSplitSpec(
            name=split_name,
            target_start_index_inclusive=_integer(
                row.get("target_start_index_inclusive"),
                f"data.splits.{split_name}.target_start_index_inclusive",
            ),
            target_end_index_exclusive=_integer(
                row.get("target_end_index_exclusive"),
                f"data.splits.{split_name}.target_end_index_exclusive",
                minimum=1,
            ),
            targets_per_cell=_integer(
                row.get("targets_per_cell"),
                f"data.splits.{split_name}.targets_per_cell",
                minimum=1,
            ),
            samples=_integer(
                row.get("samples"), f"data.splits.{split_name}.samples", minimum=1
            ),
        )
        _expect(
            (
                split.target_start_index_inclusive,
                split.target_end_index_exclusive,
            ),
            expected_boundaries[split_name],
            f"{split_name} boundaries",
        )
        _expect(
            split.targets_per_cell,
            EXPECTED_TARGET_COUNTS[split_name],
            f"{split_name} target count",
        )
        _expect(split.samples, 900 * split.targets_per_cell, f"{split_name} samples")
        splits.append(split)
    data = FullDataSpec(
        spatial_scope=_string(raw_data.get("spatial_scope"), "data.spatial_scope"),
        expected_cell_count=_integer(
            raw_data.get("expected_cell_count"), "data.expected_cell_count", minimum=1
        ),
        input_length=_integer(
            raw_data.get("input_length"), "data.input_length", minimum=1
        ),
        horizon=_integer(raw_data.get("horizon"), "data.horizon", minimum=1),
        bundle_global_start_index_inclusive=_integer(
            raw_data.get("bundle_global_start_index_inclusive"),
            "data.bundle_global_start_index_inclusive",
        ),
        bundle_global_end_index_exclusive=_integer(
            raw_data.get("bundle_global_end_index_exclusive"),
            "data.bundle_global_end_index_exclusive",
            minimum=1,
        ),
        splits=tuple(splits),
        test_policy=_string(raw_data.get("test_policy"), "data.test_policy"),
        test_target_start_index_inclusive=_integer(
            raw_data.get("test_target_start_index_inclusive"),
            "data.test_target_start_index_inclusive",
            minimum=1,
        ),
        known_prior_test_exposure=_string(
            raw_data.get("known_prior_test_exposure"), "data.known_prior_test_exposure"
        ),
        future_test_claim=_string(
            raw_data.get("future_test_claim"), "data.future_test_claim"
        ),
    )
    expected_data = {
        "spatial_scope": "central_900_approximate_all_cells",
        "expected_cell_count": 900,
        "input_length": 8,
        "horizon": 1,
        "bundle_global_start_index_inclusive": 0,
        "bundle_global_end_index_exclusive": 3600,
        "test_policy": "excluded_from_bundle_training_jobs_and_validation_aggregation",
        "test_target_start_index_inclusive": 3600,
        "known_prior_test_exposure": "raw_fixed_epoch_pipeline_smoke_only",
        "future_test_claim": "locked_final_evaluation_with_prior_raw_smoke_exposure_not_pristine",
    }
    for field, expected in expected_data.items():
        _expect(getattr(data, field), expected, f"data.{field}")

    raw_scaling = _mapping(root.get("scaling"), "scaling")
    scaling = FullScalingSpec(
        name=_string(raw_scaling.get("name"), "scaling.name"),
        fit_start_index_inclusive=_integer(
            raw_scaling.get("fit_start_index_inclusive"),
            "scaling.fit_start_index_inclusive",
        ),
        fit_end_index_exclusive=_integer(
            raw_scaling.get("fit_end_index_exclusive"),
            "scaling.fit_end_index_exclusive",
            minimum=1,
        ),
        fit_values=_string(raw_scaling.get("fit_values"), "scaling.fit_values"),
        dtype=_string(raw_scaling.get("dtype"), "scaling.dtype"),
        clip_transform=_boolean(
            raw_scaling.get("clip_transform"), "scaling.clip_transform"
        ),
        clip_inverse_prediction=_boolean(
            raw_scaling.get("clip_inverse_prediction"),
            "scaling.clip_inverse_prediction",
        ),
        zero_range_policy=_string(
            raw_scaling.get("zero_range_policy"), "scaling.zero_range_policy"
        ),
        expected_zero_range_cell_count=_integer(
            raw_scaling.get("expected_zero_range_cell_count"),
            "scaling.expected_zero_range_cell_count",
        ),
        roundtrip_max_absolute_error=_number(
            raw_scaling.get("roundtrip_max_absolute_error"),
            "scaling.roundtrip_max_absolute_error",
            minimum=0.0,
        ),
    )
    expected_scaling = {
        "name": "per_cell_train_only_minmax",
        "fit_start_index_inclusive": 0,
        "fit_end_index_exclusive": 2880,
        "fit_values": "existing_preprocessed_filled_traffic",
        "dtype": "float32",
        "clip_transform": False,
        "clip_inverse_prediction": False,
        "zero_range_policy": "reject",
        "expected_zero_range_cell_count": 0,
        "roundtrip_max_absolute_error": 0.001,
    }
    for field, expected in expected_scaling.items():
        _expect(getattr(scaling, field), expected, f"scaling.{field}")

    raw_upc = _mapping(root.get("upc"), "upc")
    raw_counts = _mapping(raw_upc.get("expected_cluster_counts"), "upc counts")
    counts = tuple(
        sorted(
            (int(key), _integer(value, f"cluster {key}"))
            for key, value in raw_counts.items()
        )
    )
    upc = FullUpcSpec(
        protocol=_string(raw_upc.get("protocol"), "upc.protocol"),
        conditions=tuple(
            _string(value, f"upc.conditions[{index}]")
            for index, value in enumerate(
                _list(raw_upc.get("conditions"), "upc.conditions")
            )
        ),
        expected_cluster_counts=counts,
        prediction_recombination=_string(
            raw_upc.get("prediction_recombination"), "upc.prediction_recombination"
        ),
    )
    _expect(upc.protocol, "train_only", "upc.protocol")
    _expect(upc.conditions, EXPECTED_CONDITIONS, "upc.conditions")
    _expect(upc.expected_cluster_counts, EXPECTED_CLUSTER_COUNTS, "upc counts")
    _expect(
        upc.prediction_recombination,
        "scatter_to_exact_central_manifest_order",
        "upc recombination",
    )

    scaling_config = load_lstm_scaling_config(
        next(row.path for row in sources if row.name == "scaling_pilot_config"),
        base_directory=base_directory,
    )
    raw_architecture = _mapping(root.get("architecture"), "architecture")
    architecture_values = {
        "name": _string(raw_architecture.get("name"), "architecture.name"),
        "units": tuple(
            _integer(value, f"architecture.lstm_units[{index}]", minimum=1)
            for index, value in enumerate(
                _list(raw_architecture.get("lstm_units"), "architecture.lstm_units")
            )
        ),
        "return_sequences": tuple(
            _boolean(value, f"architecture.return_sequences[{index}]")
            for index, value in enumerate(
                _list(
                    raw_architecture.get("return_sequences"),
                    "architecture.return_sequences",
                )
            )
        ),
        "dropout_placement": _string(
            raw_architecture.get("dropout_placement"), "architecture.dropout_placement"
        ),
        "output_units": _integer(
            raw_architecture.get("output_units"), "architecture.output_units", minimum=1
        ),
        "expected_parameter_count": _integer(
            raw_architecture.get("expected_parameter_count"),
            "architecture.expected_parameter_count",
            minimum=1,
        ),
        "author_implementation_confirmed": _boolean(
            raw_architecture.get("author_implementation_confirmed"),
            "architecture.author_implementation_confirmed",
        ),
    }
    base_architecture = scaling_config.architecture
    for field, actual in architecture_values.items():
        _expect(actual, getattr(base_architecture, field), f"architecture.{field}")
    architecture = base_architecture

    raw_training = _mapping(root.get("training"), "training")
    raw_early = _mapping(raw_training.get("early_stopping"), "early_stopping")
    early = EarlyStoppingSpec(
        monitor=_string(raw_early.get("monitor"), "early_stopping.monitor"),
        monitor_domain=_string(
            raw_early.get("monitor_domain"), "early_stopping.monitor_domain"
        ),
        mode=_string(raw_early.get("mode"), "early_stopping.mode"),
        patience=_integer(raw_early.get("patience"), "early_stopping.patience"),
        min_delta=_number(
            raw_early.get("min_delta"), "early_stopping.min_delta", minimum=0.0
        ),
        restore_best_weights=_boolean(
            raw_early.get("restore_best_weights"), "early_stopping.restore_best_weights"
        ),
        start_from_epoch=_integer(
            raw_early.get("start_from_epoch"), "early_stopping.start_from_epoch"
        ),
    )
    training = FullTrainingSpec(
        seeds=tuple(
            _integer(value, f"training.seeds[{index}]")
            for index, value in enumerate(
                _list(raw_training.get("seeds"), "training.seeds")
            )
        ),
        expected_job_count=_integer(
            raw_training.get("expected_job_count"),
            "training.expected_job_count",
            minimum=1,
        ),
        optimizer=_string(raw_training.get("optimizer"), "training.optimizer"),
        learning_rate=_number(
            raw_training.get("learning_rate"), "training.learning_rate", minimum=0.0
        ),
        loss=_string(raw_training.get("loss"), "training.loss"),
        loss_domain=_string(raw_training.get("loss_domain"), "training.loss_domain"),
        batch_size=_integer(
            raw_training.get("batch_size"), "training.batch_size", minimum=1
        ),
        max_epochs=_integer(
            raw_training.get("max_epochs"), "training.max_epochs", minimum=1
        ),
        dropout=_number(
            raw_training.get("dropout"), "training.dropout", minimum=0.0, maximum=1.0
        ),
        shuffle=_boolean(raw_training.get("shuffle"), "training.shuffle"),
        early_stopping=early,
        checkpoint_format=_string(
            raw_training.get("checkpoint_format"), "training.checkpoint_format"
        ),
        checkpoint_selection=_string(
            raw_training.get("checkpoint_selection"), "training.checkpoint_selection"
        ),
        maximum_wall_clock_seconds_per_job=_integer(
            raw_training.get("maximum_wall_clock_seconds_per_job"),
            "training.maximum_wall_clock_seconds_per_job",
            minimum=1,
        ),
        wall_clock_limit_outcome=_string(
            raw_training.get("wall_clock_limit_outcome"),
            "training.wall_clock_limit_outcome",
        ),
    )
    expected_training = {
        "seeds": EXPECTED_SEEDS,
        "expected_job_count": 9,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "mae",
        "loss_domain": "cellwise_scaled",
        "batch_size": 512,
        "max_epochs": 1000,
        "dropout": 0.05,
        "shuffle": False,
        "checkpoint_format": "numpy_weights_npz",
        "checkpoint_selection": "best_scaled_validation_mae",
        "maximum_wall_clock_seconds_per_job": 7200,
        "wall_clock_limit_outcome": "incomplete_not_a_result",
    }
    for field, expected in expected_training.items():
        _expect(getattr(training, field), expected, f"training.{field}")
    expected_early = {
        "monitor": "val_loss",
        "monitor_domain": "cellwise_scaled_mae",
        "mode": "min",
        "patience": 5,
        "min_delta": 0.0,
        "restore_best_weights": True,
        "start_from_epoch": 0,
    }
    for field, expected in expected_early.items():
        _expect(getattr(early, field), expected, f"early_stopping.{field}")

    raw_validation = _mapping(root.get("validation_evaluation"), "validation")
    validation = ValidationEvaluationSpec(
        evaluated_split=_string(
            raw_validation.get("evaluated_split"), "evaluated_split"
        ),
        target_policies=tuple(
            _string(value, f"target_policies[{index}]")
            for index, value in enumerate(
                _list(raw_validation.get("target_policies"), "target_policies")
            )
        ),
        metrics=tuple(
            _string(value, f"metrics[{index}]")
            for index, value in enumerate(
                _list(raw_validation.get("metrics"), "metrics")
            )
        ),
        aggregations=tuple(
            _string(value, f"aggregations[{index}]")
            for index, value in enumerate(
                _list(raw_validation.get("aggregations"), "aggregations")
            )
        ),
        report_unit=_string(raw_validation.get("report_unit"), "report_unit"),
        mape_positive_targets_only=_boolean(
            raw_validation.get("mape_positive_targets_only"),
            "mape_positive_targets_only",
        ),
        seed_summary=_string(raw_validation.get("seed_summary"), "seed_summary"),
        performance_used_as_pipeline_gate=_boolean(
            raw_validation.get("performance_used_as_pipeline_gate"),
            "performance_used_as_pipeline_gate",
        ),
        upc_condition_used_for_exclusion=_boolean(
            raw_validation.get("upc_condition_used_for_exclusion"),
            "upc_condition_used_for_exclusion",
        ),
    )
    expected_validation = {
        "evaluated_split": "validation",
        "target_policies": ("all_targets", "observed_targets_only"),
        "metrics": ("mae", "mape_ratio", "mape_percent", "wape"),
        "aggregations": ("micro", "cell_macro"),
        "report_unit": "original_traffic_after_cellwise_inverse_transform",
        "mape_positive_targets_only": True,
        "seed_summary": "individual_mean_and_sample_standard_deviation_ddof_1",
        "performance_used_as_pipeline_gate": False,
        "upc_condition_used_for_exclusion": False,
    }
    for field, expected in expected_validation.items():
        _expect(getattr(validation, field), expected, f"validation.{field}")

    raw_resource = _mapping(root.get("resource_policy"), "resource_policy")
    resources = ResourcePolicy(
        local_tensorflow_training=_boolean(
            raw_resource.get("local_tensorflow_training"), "local_tensorflow_training"
        ),
        local_peak_rss_limit_bytes=_integer(
            raw_resource.get("local_peak_rss_limit_bytes"),
            "local_peak_rss_limit_bytes",
            minimum=1,
        ),
        required_colab_gpu_name_contains=_string(
            raw_resource.get("required_colab_gpu_name_contains"),
            "required_colab_gpu_name_contains",
        ),
        colab_peak_rss_soft_limit_bytes=_integer(
            raw_resource.get("colab_peak_rss_soft_limit_bytes"),
            "colab_peak_rss_soft_limit_bytes",
            minimum=1,
        ),
        job_execution=_string(raw_resource.get("job_execution"), "job_execution"),
        rerun_policy=_string(raw_resource.get("rerun_policy"), "rerun_policy"),
    )
    expected_resources = {
        "local_tensorflow_training": False,
        "local_peak_rss_limit_bytes": 268435456,
        "required_colab_gpu_name_contains": "T4",
        "colab_peak_rss_soft_limit_bytes": 4294967296,
        "job_execution": "independent_seed_and_condition_jobs",
        "rerun_policy": "rerun_only_same_immutable_job_after_infrastructure_failure",
    }
    for field, expected in expected_resources.items():
        _expect(getattr(resources, field), expected, f"resource_policy.{field}")

    raw_pass = _mapping(root.get("pass_criteria"), "pass_criteria")
    pass_criteria = FullPassCriteria(
        **{
            field: _boolean(raw_pass.get(field), f"pass_criteria.{field}")
            for field in FullPassCriteria.__dataclass_fields__
        }
    )
    if (
        not all(
            getattr(pass_criteria, field)
            for field in (
                "require_clean_source_git",
                "require_source_checksums",
                "require_test_absent",
                "require_exact_parameter_count",
                "require_all_nine_jobs",
                "require_finite_history_weights_and_predictions",
                "require_best_weights_restored",
                "require_exact_cluster_recombination",
                "require_finite_validation_metrics",
            )
        )
        or pass_criteria.require_better_than_persistence
        or pass_criteria.require_upc_improvement
    ):
        raise LstmFullContractError("pass criteria가 사전 등록한 구조 gate와 다릅니다.")

    raw_outputs = _mapping(root.get("outputs"), "outputs")
    outputs = FullOutputPaths(
        **{
            field: _path(raw_outputs.get(field), f"outputs.{field}", base_directory)
            for field in FullOutputPaths.__dataclass_fields__
        }
    )
    if len(set(outputs.as_dict().values())) != len(outputs.as_dict()):
        raise LstmFullContractError("output 경로는 서로 달라야 합니다.")
    if any(
        "lstm_full_training" not in str(value) for value in outputs.as_dict().values()
    ):
        raise LstmFullContractError(
            "모든 output은 lstm_full_training 전용 경로여야 합니다."
        )

    config = LstmFullTrainingConfig(
        path=path.resolve(),
        base_directory=base_directory.resolve(),
        name=name,
        decision_stage=stage,
        question=question,
        sources=sources,
        required_scaling_outcome=required_outcome,
        data=data,
        scaling=scaling,
        upc=upc,
        architecture=architecture,
        training=training,
        validation=validation,
        resources=resources,
        pass_criteria=pass_criteria,
        outputs=outputs,
    )
    _expect(len(config.jobs), training.expected_job_count, "job count")
    return config


__all__ = [
    "DEFAULT_CONFIG",
    "FullJobSpec",
    "LstmFullContractError",
    "LstmFullTrainingConfig",
    "load_lstm_full_config",
]
