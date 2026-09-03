#!/usr/bin/env python3
"""Colab GPU에서 RCTL 구조 감사와 의도적 과적합 smoke를 실행한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
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
from scripts.rctl_contract import (
    DEFAULT_CONFIG,
    RctlContractError,
    RctlSmokeConfig,
    load_rctl_smoke_config,
)
from scripts.rctl_model import (
    audit_rctl_model,
    build_rctl_model,
    configure_determinism,
    require_tensorflow,
)

TOOL_VERSION = "1.0.0"
REQUIRED_BUNDLE_ARRAYS = {
    "x",
    "y",
    "persistence",
    "cell_ids",
    "target_indices",
    "target_timestamps_ms",
    "target_missing_mask",
    "target_internet_null_mask",
    "input_missing_mask",
    "input_internet_null_mask",
}


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RctlContractError(f"{label}을 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RctlContractError(f"{label}이 올바른 JSON이 아닙니다: {exc}") from exc
    if not isinstance(value, dict):
        raise RctlContractError(f"{label}의 최상위 값은 객체여야 합니다.")
    return value


def _content_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_and_verify_bundle(config: RctlSmokeConfig) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    """준비 단계 manifest와 NPZ의 파일·배열 checksum을 모두 확인한다."""

    manifest = _load_json(config.outputs.input_manifest, "RCTL input manifest")
    if manifest.get("status") != "complete":
        raise RctlContractError("RCTL input manifest가 complete 상태가 아닙니다.")
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RctlContractError("RCTL input manifest.output이 없습니다.")
    expected_file_sha = output.get("sha256")
    actual_file_sha = compute_sha256(config.outputs.input_npz)
    if actual_file_sha != expected_file_sha:
        raise RctlContractError("RCTL input NPZ checksum이 manifest와 다릅니다.")
    expected_arrays = output.get("arrays")
    if not isinstance(expected_arrays, dict):
        raise RctlContractError("RCTL input manifest의 배열 계약이 없습니다.")
    try:
        with np.load(config.outputs.input_npz, allow_pickle=False) as archive:
            if set(archive.files) != REQUIRED_BUNDLE_ARRAYS:
                raise RctlContractError("RCTL input NPZ의 배열 이름 계약이 다릅니다.")
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise RctlContractError("RCTL input NPZ를 읽을 수 없습니다.") from exc
    for name, array in arrays.items():
        contract = expected_arrays.get(name)
        if not isinstance(contract, dict):
            raise RctlContractError(f"RCTL input manifest에 {name} 계약이 없습니다.")
        if list(array.shape) != contract.get("shape") or str(array.dtype) != contract.get("dtype"):
            raise RctlContractError(f"RCTL input {name}의 shape 또는 dtype이 다릅니다.")
        if _content_sha256(array) != contract.get("content_sha256"):
            raise RctlContractError(f"RCTL input {name}의 내용 checksum이 다릅니다.")

    x = arrays["x"]
    y = arrays["y"]
    if x.shape != (config.selection.sample_count, config.input_length, 1):
        raise RctlContractError("RCTL x shape가 사전 등록된 sample 계약과 다릅니다.")
    if y.shape != (config.selection.sample_count, 1):
        raise RctlContractError("RCTL y shape가 사전 등록된 sample 계약과 다릅니다.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise RctlContractError("RCTL x 또는 y에 NaN/무한대가 있습니다.")
    return arrays, manifest


def _runtime_environment(tf: Any) -> dict[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    gpu_details: list[dict[str, Any]] = []
    for device in physical_gpus:
        try:
            details = tf.config.experimental.get_device_details(device)
        except (RuntimeError, ValueError):
            details = {}
        gpu_details.append(
            {
                "name": device.name,
                "device_type": device.device_type,
                "details": {key: str(value) for key, value in details.items()},
            }
        )
    keras_version = getattr(tf.keras, "__version__", None)
    try:
        build_info = {
            key: str(value) for key, value in tf.sysconfig.get_build_info().items()
        }
    except (AttributeError, TypeError):
        build_info = {}
    try:
        nvidia_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        nvidia_rows = []
    nvidia_inventory = []
    for row in nvidia_rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 3:
            nvidia_inventory.append(
                {
                    "name": fields[0],
                    "memory_total_mib": int(fields[1]),
                    "driver_version": fields[2],
                }
            )
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "tensorflow_version": tf.__version__,
        "keras_version": keras_version,
        "tensorflow_build_info": build_info,
        "physical_gpus": gpu_details,
        "nvidia_smi_inventory": nvidia_inventory,
        "gpu_available": bool(physical_gpus),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    published = False
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        published = True
    finally:
        if not published:
            temporary.unlink(missing_ok=True)


def _architecture_audit(config: RctlSmokeConfig) -> dict[str, Any]:
    tf = require_tensorflow()
    variants: list[dict[str, Any]] = []
    for spec in config.variants.values():
        tf.keras.backend.clear_session()
        configure_determinism(config.seed)
        model = build_rctl_model(
            steps=config.input_length,
            spec=spec,
            dropout=config.training.dropout,
            learning_rate=config.training.learning_rate,
        )
        variants.append(
            audit_rctl_model(
                model,
                spec=spec,
                steps=config.input_length,
                seed=config.seed,
            )
        )
        del model
    required_pass = all(item["required_gates_passed"] for item in variants)
    public_result = next(
        item for item in variants if item["variant"] == "public_reference"
    )
    paper_expected = config.variants[
        "paper_interpretation"
    ].expected_parameter_count
    final_projection = next(
        layer
        for layer in public_result["layers"]
        if layer["name"] == "final_input_projection"
    )
    public_count = public_result["parameter_count"]["actual"]
    count_without_projection = public_count - final_projection["parameter_count"]
    return {
        "schema_version": 1,
        "status": "pass" if required_pass else "failed",
        "purpose": "architecture contract audit; parameter mismatch is diagnostic only",
        "selected_training_variant": config.selected_variant,
        "required_gates_passed": required_pass,
        "post_hoc_parameter_reconciliation": {
            "classification": "diagnostic_clue_not_an_implementation_change",
            "paper_reported_parameter_count": paper_expected,
            "public_reference_parameter_count": public_count,
            "final_input_projection_parameter_count": final_projection[
                "parameter_count"
            ],
            "public_count_without_final_input_projection": count_without_projection,
            "equals_paper_reported_count": count_without_projection == paper_expected,
            "interpretation": (
                "The arithmetic equality is a clue about the unpublished graph or "
                "counting method. It does not authorize removing a layer after seeing "
                "the result."
            ),
        },
        "variants": variants,
    }


def _overfit_smoke(config: RctlSmokeConfig, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tf = require_tensorflow()
    tf.keras.backend.clear_session()
    determinism = configure_determinism(config.seed)
    spec = config.variants[config.selected_variant]
    model = build_rctl_model(
        steps=config.input_length,
        spec=spec,
        dropout=config.training.dropout,
        learning_rate=config.training.learning_rate,
    )
    x = np.asarray(arrays["x"], dtype=np.float32)
    y = np.asarray(arrays["y"], dtype=np.float32)
    persistence = np.asarray(arrays["persistence"], dtype=np.float32)
    prefit_predictions = np.asarray(model(x, training=False).numpy(), dtype=np.float32)
    prefit_mae = float(
        np.mean(np.abs(y.astype(np.float64) - prefit_predictions), dtype=np.float64)
    )
    persistence_mae = float(
        np.mean(np.abs(y.astype(np.float64) - persistence), dtype=np.float64)
    )

    class EpochTimer(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.seconds: list[float] = []
            self._started = 0.0

        def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
            del epoch, logs
            self._started = time.perf_counter()

        def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
            del epoch, logs
            self.seconds.append(time.perf_counter() - self._started)

    epoch_timer = EpochTimer()
    started = time.perf_counter()
    history = model.fit(
        x,
        y,
        batch_size=config.training.batch_size,
        epochs=config.training.max_epochs,
        shuffle=config.training.shuffle,
        callbacks=[epoch_timer],
        verbose=0,
    )
    training_seconds = time.perf_counter() - started
    final_predictions = np.asarray(model(x, training=False).numpy(), dtype=np.float32)
    final_mae = float(
        np.mean(np.abs(y.astype(np.float64) - final_predictions), dtype=np.float64)
    )
    reduction = (prefit_mae - final_mae) / prefit_mae if prefit_mae else 0.0
    losses = [float(value) for value in history.history["loss"]]
    finite = bool(
        np.all(np.isfinite(prefit_predictions))
        and np.all(np.isfinite(final_predictions))
        and np.all(np.isfinite(losses))
        and np.isfinite(prefit_mae)
        and np.isfinite(final_mae)
    )
    gates = {
        "finite_values": finite,
        "loss_reduction_at_least_configured_fraction": (
            reduction >= config.pass_criteria.minimum_loss_reduction_fraction
        ),
        "final_train_mae_below_persistence": final_mae < persistence_mae,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "status": "pass" if passed else "failed",
        "purpose": (
            "implementation diagnostic on the same training subset; this is not a "
            "validation/test performance result"
        ),
        "variant": config.selected_variant,
        "sample_contract": {
            "sample_count": len(x),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "unique_cell_count": len(np.unique(arrays["cell_ids"])),
            "unique_target_count_per_cell": len(
                np.unique(arrays["target_indices"])
            ),
            "target_missing_count": int(np.asarray(arrays["target_missing_mask"]).sum()),
            "target_internet_null_count": int(
                np.asarray(arrays["target_internet_null_mask"]).sum()
            ),
        },
        "training_contract": {
            "seed": config.seed,
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "loss": config.training.loss,
            "batch_size": config.training.batch_size,
            "epochs_requested": config.training.max_epochs,
            "epochs_completed": len(losses),
            "dropout": config.training.dropout,
            "shuffle": config.training.shuffle,
            "input_scaling": config.training.input_scaling,
        },
        "metrics": {
            "prefit_train_mae": prefit_mae,
            "final_train_mae": final_mae,
            "prefit_to_final_loss_reduction_fraction": reduction,
            "persistence_train_mae": persistence_mae,
            "final_to_persistence_mae_ratio": (
                final_mae / persistence_mae if persistence_mae else None
            ),
            "first_epoch_training_loss": losses[0],
            "last_epoch_training_loss": losses[-1],
        },
        "pass_criteria": {
            "loss_reduction_basis": config.pass_criteria.loss_reduction_basis,
            "minimum_loss_reduction_fraction": (
                config.pass_criteria.minimum_loss_reduction_fraction
            ),
            "require_better_than_persistence": (
                config.pass_criteria.require_better_than_persistence
            ),
            "require_finite_values": config.pass_criteria.require_finite_values,
        },
        "gates": gates,
        "gates_passed": passed,
        "history": {"loss": losses, "epoch_seconds": epoch_timer.seconds},
        "prediction_diagnostics": {
            "minimum": float(final_predictions.min()),
            "maximum": float(final_predictions.max()),
            "mean": float(np.mean(final_predictions, dtype=np.float64)),
            "negative_prediction_count": int((final_predictions < 0).sum()),
        },
        "determinism": determinism,
        "training_seconds": training_seconds,
    }


def run_rctl_smoke(config: RctlSmokeConfig) -> dict[str, Any]:
    """GPU 존재를 확인한 뒤 구조 감사와 overfit 진단 결과를 게시한다."""

    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    tf = require_tensorflow()
    runtime = _runtime_environment(tf)
    if not runtime["gpu_available"]:
        raise RctlContractError("RCTL smoke는 Colab T4 GPU가 보이는 환경에서만 실행합니다.")
    if not any(
        "T4" in item["name"] for item in runtime["nvidia_smi_inventory"]
    ):
        raise RctlContractError("요청한 Tesla T4가 아니므로 RCTL smoke를 중단합니다.")
    arrays, input_manifest = load_and_verify_bundle(config)
    architecture_report = _architecture_audit(config)
    _write_json_atomic(config.outputs.architecture_report, architecture_report)
    if architecture_report["required_gates_passed"]:
        overfit_report = _overfit_smoke(config, arrays)
    else:
        overfit_report = {
            "schema_version": 1,
            "status": "not_run",
            "reason": "required architecture gate failed",
            "gates_passed": False,
        }
    _write_json_atomic(config.outputs.overfit_report, overfit_report)
    finished_at = datetime.now(timezone.utc)
    overall_pass = bool(
        architecture_report["required_gates_passed"]
        and overfit_report.get("gates_passed")
    )
    manifest = {
        "schema_version": 1,
        "status": "pass" if overall_pass else "failed",
        "tool": {"name": "scripts.run_rctl_smoke", "version": TOOL_VERSION},
        "created_at_utc": finished_at.isoformat(),
        "scope": "16-cell real-data training-subset implementation diagnostic",
        "not_a_performance_result": True,
        "git": _git_state(),
        "config": {
            "path": _display_path(config.path),
            "sha256": compute_sha256(config.path),
        },
        "input": {
            "npz_path": _display_path(config.outputs.input_npz),
            "npz_sha256": compute_sha256(config.outputs.input_npz),
            "manifest_path": _display_path(config.outputs.input_manifest),
            "manifest_sha256": compute_sha256(config.outputs.input_manifest),
            "prepared_sample_count": input_manifest["smoke"]["sample_count"],
        },
        "gates": {
            "architecture_required_gates": architecture_report[
                "required_gates_passed"
            ],
            "overfit_required_gates": bool(overfit_report.get("gates_passed")),
        },
        "outputs": {
            "architecture_report": {
                "path": _display_path(config.outputs.architecture_report),
                "size_bytes": config.outputs.architecture_report.stat().st_size,
                "sha256": compute_sha256(config.outputs.architecture_report),
            },
            "overfit_report": {
                "path": _display_path(config.outputs.overfit_report),
                "size_bytes": config.outputs.overfit_report.stat().st_size,
                "sha256": compute_sha256(config.outputs.overfit_report),
            },
        },
        "runtime": {
            **runtime,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - started_counter,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "checkpoint_policy": (
            "No checkpoint is retained for this disposable overfit diagnostic."
        ),
    }
    _write_json_atomic(config.outputs.run_manifest, manifest)
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Colab T4에서 RCTL architecture/overfit smoke를 실행합니다."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_rctl_smoke_config(args.config)
        manifest = run_rctl_smoke(config)
    except (RctlContractError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"RCTL smoke 상태: {manifest['status']}")
    print(f"manifest: {_display_path(config.outputs.run_manifest)}")
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
