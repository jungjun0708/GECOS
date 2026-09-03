#!/usr/bin/env python3
"""RCTL 아키텍처 감사와 소규모 과적합 진단의 사전 등록 계약."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.build_upc_initial_groups import (
    REPOSITORY_ROOT,
    _require_bool,
    _require_int,
    _require_mapping,
    _require_string,
    _resolve_path,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "rctl_smoke_milan_nov2013.json"
VARIANT_NAMES = ("paper_interpretation", "public_reference")
EXPECTED_CHANNELS = (16, 32, 64, 64, 32, 16)


class RctlContractError(RuntimeError):
    """RCTL smoke 설정 또는 결과 계약이 깨졌을 때 발생한다."""


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    channels: tuple[int, ...]
    kernel_size: int
    dilations: tuple[int, ...]
    tcn_shortcut_merge: str
    rcc1: str
    rcc2_routes: tuple[tuple[int, int], ...]
    final_input_shortcut: bool
    expected_parameter_count: int | None
    evidence: str

    @property
    def rcc2_route_map(self) -> dict[int, int]:
        return dict(self.rcc2_routes)


@dataclass(frozen=True)
class SelectionSpec:
    central_grid_side: int
    windows_per_cell: int
    cell_policy: str
    target_policy: str

    @property
    def cell_count(self) -> int:
        return self.central_grid_side**2

    @property
    def sample_count(self) -> int:
        return self.cell_count * self.windows_per_cell


@dataclass(frozen=True)
class TrainingSpec:
    optimizer: str
    learning_rate: float
    loss: str
    batch_size: int
    max_epochs: int
    dropout: float
    shuffle: bool
    input_scaling: str


@dataclass(frozen=True)
class PassCriteria:
    loss_reduction_basis: str
    minimum_loss_reduction_fraction: float
    require_better_than_persistence: bool
    require_finite_values: bool


@dataclass(frozen=True)
class RctlOutputPaths:
    input_npz: Path
    input_manifest: Path
    architecture_report: Path
    overfit_report: Path
    run_manifest: Path
    checkpoint_directory: Path

    def as_dict(self) -> dict[str, Path]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class RctlSmokeConfig:
    path: Path
    name: str
    forecast_config_path: Path
    seed: int
    input_length: int
    variants: dict[str, ArchitectureSpec]
    selected_variant: str
    selection: SelectionSpec
    training: TrainingSpec
    pass_criteria: PassCriteria
    outputs: RctlOutputPaths


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RctlContractError(f"RCTL config를 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RctlContractError(f"RCTL config가 올바른 JSON이 아닙니다: {exc}") from exc
    try:
        return _require_mapping(value, "RCTL config")
    except RuntimeError as exc:
        raise RctlContractError(str(exc)) from exc


def _require_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RctlContractError(f"{field}은 숫자여야 합니다.")
    return float(value)


def _parse_int_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RctlContractError(f"{field}은 배열이어야 합니다.")
    try:
        return tuple(_require_int(item, f"{field}[{index}]", minimum=1) for index, item in enumerate(value))
    except RuntimeError as exc:
        raise RctlContractError(str(exc)) from exc


def _parse_variant(
    name: str,
    raw: object,
    channels: tuple[int, ...],
) -> ArchitectureSpec:
    try:
        value = _require_mapping(raw, f"architecture.variants.{name}")
        routes_raw = _require_mapping(
            value.get("rcc2_routes"), f"architecture.variants.{name}.rcc2_routes"
        )
        routes: list[tuple[int, int]] = []
        for destination, source in routes_raw.items():
            try:
                destination_index = int(destination)
            except (TypeError, ValueError) as exc:
                raise RctlContractError(
                    f"{name} rcc2 destination은 정수 문자열이어야 합니다."
                ) from exc
            source_index = _require_int(
                source, f"architecture.variants.{name}.rcc2_routes.{destination}"
            )
            if destination_index >= len(channels) or source_index >= len(channels):
                raise RctlContractError(f"{name} rcc2 route index가 block 범위를 벗어납니다.")
            if channels[destination_index] != channels[source_index]:
                raise RctlContractError(
                    f"{name} rcc2 route {source_index}->{destination_index}의 channel이 다릅니다."
                )
            routes.append((destination_index, source_index))
        expected_raw = value.get("expected_parameter_count")
        expected = None
        if expected_raw is not None:
            expected = _require_int(
                expected_raw,
                f"architecture.variants.{name}.expected_parameter_count",
                minimum=1,
            )
        spec = ArchitectureSpec(
            name=name,
            channels=channels,
            kernel_size=_require_int(
                value.get("kernel_size"),
                f"architecture.variants.{name}.kernel_size",
                minimum=1,
            ),
            dilations=_parse_int_tuple(
                value.get("dilations"), f"architecture.variants.{name}.dilations"
            ),
            tcn_shortcut_merge=_require_string(
                value.get("tcn_shortcut_merge"),
                f"architecture.variants.{name}.tcn_shortcut_merge",
            ),
            rcc1=_require_string(value.get("rcc1"), f"architecture.variants.{name}.rcc1"),
            rcc2_routes=tuple(sorted(routes)),
            final_input_shortcut=_require_bool(
                value.get("final_input_shortcut"),
                f"architecture.variants.{name}.final_input_shortcut",
            ),
            expected_parameter_count=expected,
            evidence=_require_string(
                value.get("evidence"), f"architecture.variants.{name}.evidence"
            ),
        )
    except RuntimeError as exc:
        if isinstance(exc, RctlContractError):
            raise
        raise RctlContractError(str(exc)) from exc
    if len(spec.dilations) != len(channels):
        raise RctlContractError(f"{name} dilation 수가 block 수와 다릅니다.")
    if spec.tcn_shortcut_merge not in {"add", "concatenate"}:
        raise RctlContractError(f"{name} tcn_shortcut_merge 값이 지원되지 않습니다.")
    if spec.rcc1 != "project_block_input_then_add_after_lstm":
        raise RctlContractError(f"{name} rcc1 계약이 사전 등록값과 다릅니다.")
    if not spec.final_input_shortcut:
        raise RctlContractError(f"{name} final input shortcut은 true여야 합니다.")
    return spec


def load_rctl_smoke_config(
    path: Path,
    *,
    base_directory: Path = REPOSITORY_ROOT,
) -> RctlSmokeConfig:
    """RCTL smoke config를 읽고 논문형/공개형 계약을 엄격하게 검증한다."""

    root = _load_json(path)
    try:
        schema_version = _require_int(root.get("schema_version"), "schema_version", minimum=1)
        if schema_version != 1:
            raise RctlContractError(f"지원하지 않는 schema_version입니다: {schema_version}")
        name = _require_string(root.get("name"), "name")
        forecast_path = _resolve_path(
            root.get("forecast_config"), "forecast_config", base_directory
        )
        seed = _require_int(root.get("seed"), "seed", minimum=0)
        architecture = _require_mapping(root.get("architecture"), "architecture")
        input_length = _require_int(
            architecture.get("input_length"), "architecture.input_length", minimum=1
        )
        channels = _parse_int_tuple(architecture.get("channels"), "architecture.channels")
        variants_raw = _require_mapping(architecture.get("variants"), "architecture.variants")
        smoke = _require_mapping(root.get("smoke"), "smoke")
        selection_raw = _require_mapping(smoke.get("selection"), "smoke.selection")
        training_raw = _require_mapping(smoke.get("training"), "smoke.training")
        criteria_raw = _require_mapping(smoke.get("pass_criteria"), "smoke.pass_criteria")
        outputs_raw = _require_mapping(root.get("outputs"), "outputs")
    except RuntimeError as exc:
        if isinstance(exc, RctlContractError):
            raise
        raise RctlContractError(str(exc)) from exc

    if input_length != 8:
        raise RctlContractError("RCTL smoke input_length는 8이어야 합니다.")
    if channels != EXPECTED_CHANNELS:
        raise RctlContractError(f"RCTL channels는 {list(EXPECTED_CHANNELS)}여야 합니다.")
    if tuple(variants_raw) != VARIANT_NAMES:
        raise RctlContractError(f"RCTL variants 순서는 {list(VARIANT_NAMES)}여야 합니다.")
    variants = {
        variant_name: _parse_variant(variant_name, variants_raw[variant_name], channels)
        for variant_name in VARIANT_NAMES
    }
    paper = variants["paper_interpretation"]
    public = variants["public_reference"]
    if (
        paper.kernel_size != 4
        or paper.dilations != (1, 2, 4, 8, 16, 32)
        or paper.tcn_shortcut_merge != "concatenate"
        or paper.rcc2_routes != ((3, 2), (4, 1), (5, 0))
        or paper.expected_parameter_count != 173633
    ):
        raise RctlContractError("paper_interpretation 계약이 사전 등록값과 다릅니다.")
    if (
        public.kernel_size != 3
        or public.dilations != (1, 2, 4, 6, 8, 10)
        or public.tcn_shortcut_merge != "add"
        or public.rcc2_routes != ((2, 2), (3, 2), (4, 1), (5, 0))
        or public.expected_parameter_count is not None
    ):
        raise RctlContractError("public_reference 계약이 공개 코드 전사값과 다릅니다.")

    try:
        selected_variant = _require_string(smoke.get("selected_variant"), "smoke.selected_variant")
        selection = SelectionSpec(
            central_grid_side=_require_int(
                selection_raw.get("central_grid_side"),
                "smoke.selection.central_grid_side",
                minimum=1,
            ),
            windows_per_cell=_require_int(
                selection_raw.get("windows_per_cell"),
                "smoke.selection.windows_per_cell",
                minimum=1,
            ),
            cell_policy=_require_string(
                selection_raw.get("cell_policy"), "smoke.selection.cell_policy"
            ),
            target_policy=_require_string(
                selection_raw.get("target_policy"), "smoke.selection.target_policy"
            ),
        )
        training = TrainingSpec(
            optimizer=_require_string(training_raw.get("optimizer"), "smoke.training.optimizer"),
            learning_rate=_require_number(
                training_raw.get("learning_rate"), "smoke.training.learning_rate"
            ),
            loss=_require_string(training_raw.get("loss"), "smoke.training.loss"),
            batch_size=_require_int(
                training_raw.get("batch_size"), "smoke.training.batch_size", minimum=1
            ),
            max_epochs=_require_int(
                training_raw.get("max_epochs"), "smoke.training.max_epochs", minimum=1
            ),
            dropout=_require_number(training_raw.get("dropout"), "smoke.training.dropout"),
            shuffle=_require_bool(training_raw.get("shuffle"), "smoke.training.shuffle"),
            input_scaling=_require_string(
                training_raw.get("input_scaling"), "smoke.training.input_scaling"
            ),
        )
        criteria = PassCriteria(
            loss_reduction_basis=_require_string(
                criteria_raw.get("loss_reduction_basis"),
                "smoke.pass_criteria.loss_reduction_basis",
            ),
            minimum_loss_reduction_fraction=_require_number(
                criteria_raw.get("minimum_loss_reduction_fraction"),
                "smoke.pass_criteria.minimum_loss_reduction_fraction",
            ),
            require_better_than_persistence=_require_bool(
                criteria_raw.get("require_better_than_persistence"),
                "smoke.pass_criteria.require_better_than_persistence",
            ),
            require_finite_values=_require_bool(
                criteria_raw.get("require_finite_values"),
                "smoke.pass_criteria.require_finite_values",
            ),
        )
        outputs = RctlOutputPaths(
            **{
                field: _resolve_path(outputs_raw.get(field), f"outputs.{field}", base_directory)
                for field in RctlOutputPaths.__dataclass_fields__
            }
        )
    except RuntimeError as exc:
        if isinstance(exc, RctlContractError):
            raise
        raise RctlContractError(str(exc)) from exc

    if selected_variant != "paper_interpretation":
        raise RctlContractError("최초 smoke의 selected_variant는 paper_interpretation이어야 합니다.")
    if selection.central_grid_side != 4 or selection.windows_per_cell != 64:
        raise RctlContractError("최초 smoke 표본은 중앙 4x4셀과 셀당 64 window여야 합니다.")
    if selection.cell_policy != "central_30x30_grid_evenly_spaced_4x4":
        raise RctlContractError("smoke cell 선택 정책이 사전 등록값과 다릅니다.")
    if selection.target_policy != "train_target_indices_evenly_spaced":
        raise RctlContractError("smoke target 선택 정책이 사전 등록값과 다릅니다.")
    if (
        training.optimizer != "adam"
        or training.learning_rate != 0.001
        or training.loss != "mae"
        or training.batch_size != 512
        or training.max_epochs != 200
        or training.dropout != 0.05
        or training.shuffle
        or training.input_scaling != "none_raw_traffic"
    ):
        raise RctlContractError("최초 smoke 학습 설정이 사전 등록값과 다릅니다.")
    if (
        criteria.loss_reduction_basis != "prefit_eval_mae_to_final_eval_mae"
        or criteria.minimum_loss_reduction_fraction != 0.8
        or not criteria.require_better_than_persistence
        or not criteria.require_finite_values
    ):
        raise RctlContractError("smoke 통과 기준이 사전 등록값과 다릅니다.")
    output_values = list(outputs.as_dict().values())
    if len(set(output_values)) != len(output_values):
        raise RctlContractError("RCTL output 경로는 서로 달라야 합니다.")

    return RctlSmokeConfig(
        path=path.resolve(),
        name=name,
        forecast_config_path=forecast_path,
        seed=seed,
        input_length=input_length,
        variants=variants,
        selected_variant=selected_variant,
        selection=selection,
        training=training,
        pass_criteria=criteria,
        outputs=outputs,
    )
