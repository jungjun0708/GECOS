#!/usr/bin/env python3
"""중앙 900셀 LSTM·UPC pipeline smoke의 고정 계약을 정의한다."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_upc_initial_groups import REPOSITORY_ROOT

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "lstm_upc_smoke_milan_nov2013.json"
EXPECTED_MODEL_NAME = "paper_parameter_reconstruction"
EXPECTED_PROTOCOL = "train_only"
EXPECTED_UNITS = (64, 128, 64)
EXPECTED_RETURN_SEQUENCES = (True, True, False)
EXPECTED_PARAMETER_COUNT = 165_185
EXPECTED_SPLITS = ("train", "validation", "test")
EXPECTED_CLUSTER_COUNTS = ((0, 611), (1, 289))


class LstmSmokeContractError(RuntimeError):
    """LSTM smoke의 설정·입력·실행 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class LstmArchitectureSpec:
    name: str
    input_length: int
    units: tuple[int, ...]
    return_sequences: tuple[bool, ...]
    dropout_placement: str
    output_units: int
    expected_parameter_count: int
    author_implementation_confirmed: bool
    evidence: str


@dataclass(frozen=True)
class LstmSelectionSpec:
    spatial_scope: str
    splits: tuple[str, ...]
    targets_per_cell_per_split: int
    target_policy: str
    expected_central_cell_count: int
    expected_cluster_counts: tuple[tuple[int, int], ...]

    @property
    def expected_samples_per_split(self) -> int:
        return self.expected_central_cell_count * self.targets_per_cell_per_split


@dataclass(frozen=True)
class LstmTrainingSpec:
    optimizer: str
    learning_rate: float
    loss: str
    batch_size: int
    max_epochs: int
    dropout: float
    shuffle: bool
    input_scaling: str
    validation_role: str
    test_role: str


@dataclass(frozen=True)
class LstmPassCriteria:
    require_exact_parameter_count: bool
    require_finite_values: bool
    require_train_mae_decrease: bool
    require_exact_cluster_recombination: bool
    require_better_than_persistence: bool


@dataclass(frozen=True)
class LstmOutputPaths:
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
class LstmUpcSmokeConfig:
    path: Path
    base_directory: Path
    name: str
    forecast_config_path: Path
    policy_config_path: Path
    upc_protocol: str
    seed: int
    architecture: LstmArchitectureSpec
    selection: LstmSelectionSpec
    training: LstmTrainingSpec
    pass_criteria: LstmPassCriteria
    outputs: LstmOutputPaths


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LstmSmokeContractError(
            f"LSTM smoke config를 읽을 수 없습니다: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise LstmSmokeContractError(
            f"LSTM smoke config가 올바른 JSON이 아닙니다: {exc}"
        ) from exc
    return _require_mapping(value, "LSTM smoke config")


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise LstmSmokeContractError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise LstmSmokeContractError(f"{field}는 JSON array여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LstmSmokeContractError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LstmSmokeContractError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LstmSmokeContractError(f"{field}는 숫자여야 합니다.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise LstmSmokeContractError(f"{field}는 {minimum} 이상이어야 합니다.")
    if maximum is not None and result > maximum:
        raise LstmSmokeContractError(f"{field}는 {maximum} 이하여야 합니다.")
    return result


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise LstmSmokeContractError(f"{field}는 boolean이어야 합니다.")
    return value


def _resolve_path(value: object, field: str, base_directory: Path) -> Path:
    raw = _require_string(value, field)
    path = Path(raw)
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def reconstructed_lstm_parameter_count(
    units: tuple[int, ...],
    *,
    input_features: int = 1,
    output_units: int = 1,
) -> int:
    """Keras LSTM bias를 포함한 stacked-LSTM과 Dense의 parameter 수를 계산한다."""

    if (
        not units
        or input_features < 1
        or output_units < 1
        or any(value < 1 for value in units)
    ):
        raise LstmSmokeContractError(
            "LSTM parameter 계산 차원은 모두 1 이상이어야 합니다."
        )
    total = 0
    feature_count = input_features
    for unit_count in units:
        total += 4 * unit_count * (feature_count + unit_count + 1)
        feature_count = unit_count
    total += (feature_count + 1) * output_units
    return total


def load_lstm_smoke_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> LstmUpcSmokeConfig:
    """사전 등록한 LSTM·UPC pipeline smoke 설정을 엄격하게 읽는다."""

    root = _load_json(path)
    if _require_int(root.get("schema_version"), "schema_version", minimum=1) != 1:
        raise LstmSmokeContractError("지원하지 않는 schema_version입니다.")
    name = _require_string(root.get("name"), "name")
    forecast_config_path = _resolve_path(
        root.get("forecast_config"), "forecast_config", base_directory
    )
    policy_config_path = _resolve_path(
        root.get("upc_training_policy_config"),
        "upc_training_policy_config",
        base_directory,
    )
    upc_protocol = _require_string(root.get("upc_protocol"), "upc_protocol")
    seed = _require_int(root.get("seed"), "seed")

    raw_architecture = _require_mapping(root.get("architecture"), "architecture")
    units = tuple(
        _require_int(value, f"architecture.lstm_units[{index}]", minimum=1)
        for index, value in enumerate(
            _require_list(raw_architecture.get("lstm_units"), "architecture.lstm_units")
        )
    )
    return_sequences = tuple(
        _require_bool(value, f"architecture.return_sequences[{index}]")
        for index, value in enumerate(
            _require_list(
                raw_architecture.get("return_sequences"),
                "architecture.return_sequences",
            )
        )
    )
    architecture = LstmArchitectureSpec(
        name=_require_string(raw_architecture.get("name"), "architecture.name"),
        input_length=_require_int(
            raw_architecture.get("input_length"),
            "architecture.input_length",
            minimum=1,
        ),
        units=units,
        return_sequences=return_sequences,
        dropout_placement=_require_string(
            raw_architecture.get("dropout_placement"),
            "architecture.dropout_placement",
        ),
        output_units=_require_int(
            raw_architecture.get("output_units"),
            "architecture.output_units",
            minimum=1,
        ),
        expected_parameter_count=_require_int(
            raw_architecture.get("expected_parameter_count"),
            "architecture.expected_parameter_count",
            minimum=1,
        ),
        author_implementation_confirmed=_require_bool(
            raw_architecture.get("author_implementation_confirmed"),
            "architecture.author_implementation_confirmed",
        ),
        evidence=_require_string(
            raw_architecture.get("evidence"), "architecture.evidence"
        ),
    )

    raw_smoke = _require_mapping(root.get("smoke"), "smoke")
    raw_selection = _require_mapping(raw_smoke.get("selection"), "smoke.selection")
    splits = tuple(
        _require_string(value, f"smoke.selection.splits[{index}]")
        for index, value in enumerate(
            _require_list(raw_selection.get("splits"), "smoke.selection.splits")
        )
    )
    raw_cluster_counts = _require_mapping(
        raw_selection.get("expected_cluster_counts"),
        "smoke.selection.expected_cluster_counts",
    )
    try:
        cluster_counts = tuple(
            sorted(
                (
                    int(cluster_id),
                    _require_int(
                        count,
                        f"smoke.selection.expected_cluster_counts.{cluster_id}",
                        minimum=1,
                    ),
                )
                for cluster_id, count in raw_cluster_counts.items()
            )
        )
    except ValueError as exc:
        raise LstmSmokeContractError("cluster ID는 정수 문자열이어야 합니다.") from exc
    selection = LstmSelectionSpec(
        spatial_scope=_require_string(
            raw_selection.get("spatial_scope"), "smoke.selection.spatial_scope"
        ),
        splits=splits,
        targets_per_cell_per_split=_require_int(
            raw_selection.get("targets_per_cell_per_split"),
            "smoke.selection.targets_per_cell_per_split",
            minimum=2,
        ),
        target_policy=_require_string(
            raw_selection.get("target_policy"), "smoke.selection.target_policy"
        ),
        expected_central_cell_count=_require_int(
            raw_selection.get("expected_central_cell_count"),
            "smoke.selection.expected_central_cell_count",
            minimum=1,
        ),
        expected_cluster_counts=cluster_counts,
    )

    raw_training = _require_mapping(raw_smoke.get("training"), "smoke.training")
    training = LstmTrainingSpec(
        optimizer=_require_string(
            raw_training.get("optimizer"), "smoke.training.optimizer"
        ),
        learning_rate=_require_number(
            raw_training.get("learning_rate"),
            "smoke.training.learning_rate",
            minimum=0.0,
        ),
        loss=_require_string(raw_training.get("loss"), "smoke.training.loss"),
        batch_size=_require_int(
            raw_training.get("batch_size"), "smoke.training.batch_size", minimum=1
        ),
        max_epochs=_require_int(
            raw_training.get("max_epochs"), "smoke.training.max_epochs", minimum=1
        ),
        dropout=_require_number(
            raw_training.get("dropout"),
            "smoke.training.dropout",
            minimum=0.0,
            maximum=0.999999,
        ),
        shuffle=_require_bool(raw_training.get("shuffle"), "smoke.training.shuffle"),
        input_scaling=_require_string(
            raw_training.get("input_scaling"), "smoke.training.input_scaling"
        ),
        validation_role=_require_string(
            raw_training.get("validation_role"), "smoke.training.validation_role"
        ),
        test_role=_require_string(
            raw_training.get("test_role"), "smoke.training.test_role"
        ),
    )

    raw_criteria = _require_mapping(
        raw_smoke.get("pass_criteria"), "smoke.pass_criteria"
    )
    pass_criteria = LstmPassCriteria(
        require_exact_parameter_count=_require_bool(
            raw_criteria.get("require_exact_parameter_count"),
            "smoke.pass_criteria.require_exact_parameter_count",
        ),
        require_finite_values=_require_bool(
            raw_criteria.get("require_finite_values"),
            "smoke.pass_criteria.require_finite_values",
        ),
        require_train_mae_decrease=_require_bool(
            raw_criteria.get("require_train_mae_decrease"),
            "smoke.pass_criteria.require_train_mae_decrease",
        ),
        require_exact_cluster_recombination=_require_bool(
            raw_criteria.get("require_exact_cluster_recombination"),
            "smoke.pass_criteria.require_exact_cluster_recombination",
        ),
        require_better_than_persistence=_require_bool(
            raw_criteria.get("require_better_than_persistence"),
            "smoke.pass_criteria.require_better_than_persistence",
        ),
    )

    raw_outputs = _require_mapping(root.get("outputs"), "outputs")
    outputs = LstmOutputPaths(
        **{
            field: _resolve_path(
                raw_outputs.get(field), f"outputs.{field}", base_directory
            )
            for field in LstmOutputPaths.__dataclass_fields__
        }
    )

    if upc_protocol != EXPECTED_PROTOCOL:
        raise LstmSmokeContractError(
            "LSTM smoke의 UPC protocol은 train_only여야 합니다."
        )
    if seed != 42:
        raise LstmSmokeContractError("최초 LSTM smoke seed는 42여야 합니다.")
    if (
        architecture.name != EXPECTED_MODEL_NAME
        or architecture.input_length != 8
        or architecture.units != EXPECTED_UNITS
        or architecture.return_sequences != EXPECTED_RETURN_SEQUENCES
        or architecture.dropout_placement != "after_each_lstm"
        or architecture.output_units != 1
        or architecture.expected_parameter_count != EXPECTED_PARAMETER_COUNT
        or architecture.author_implementation_confirmed
    ):
        raise LstmSmokeContractError(
            "LSTM architecture가 사전 등록한 복원 후보와 다릅니다."
        )
    reconstructed = reconstructed_lstm_parameter_count(
        architecture.units,
        output_units=architecture.output_units,
    )
    if reconstructed != architecture.expected_parameter_count:
        raise LstmSmokeContractError(
            "LSTM layer 산술 parameter 수가 Table III 기대값과 다릅니다."
        )
    if (
        selection.spatial_scope != "all_central_900_in_manifest_order"
        or selection.splits != EXPECTED_SPLITS
        or selection.targets_per_cell_per_split != 64
        or selection.target_policy
        != "each_split_target_indices_evenly_spaced_including_endpoints"
        or selection.expected_central_cell_count != 900
        or selection.expected_cluster_counts != EXPECTED_CLUSTER_COUNTS
    ):
        raise LstmSmokeContractError("LSTM smoke selection이 사전 등록값과 다릅니다.")
    if (
        training.optimizer != "adam"
        or training.learning_rate != 0.001
        or training.loss != "mae"
        or training.batch_size != 512
        or training.max_epochs != 5
        or training.dropout != 0.05
        or training.shuffle
        or training.input_scaling != "none_raw_traffic"
        or training.validation_role != "monitor_only_fixed_epoch_no_early_stopping"
        or training.test_role != "evaluate_once_after_training"
    ):
        raise LstmSmokeContractError(
            "LSTM smoke training 설정이 사전 등록값과 다릅니다."
        )
    if (
        not pass_criteria.require_exact_parameter_count
        or not pass_criteria.require_finite_values
        or not pass_criteria.require_train_mae_decrease
        or not pass_criteria.require_exact_cluster_recombination
        or pass_criteria.require_better_than_persistence
    ):
        raise LstmSmokeContractError(
            "LSTM smoke pass criteria가 사전 등록값과 다릅니다."
        )
    if len(set(outputs.as_dict().values())) != len(outputs.as_dict()):
        raise LstmSmokeContractError("LSTM smoke output 경로는 서로 달라야 합니다.")

    return LstmUpcSmokeConfig(
        path=path.resolve(),
        base_directory=base_directory.resolve(),
        name=name,
        forecast_config_path=forecast_config_path,
        policy_config_path=policy_config_path,
        upc_protocol=upc_protocol,
        seed=seed,
        architecture=architecture,
        selection=selection,
        training=training,
        pass_criteria=pass_criteria,
        outputs=outputs,
    )
