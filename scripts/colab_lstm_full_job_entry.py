#!/usr/bin/env python3
"""업로드한 immutable LSTM full job 하나를 Colab에서 실행해 ZIP으로 묶는다."""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

BUNDLE_PATH = Path("/content/gecos_lstm_full_training_bundle.zip")
JOB_PATH = Path("/content/gecos_lstm_full_job.json")
WORKSPACE = Path("/content/gecos")
OUTPUT_ZIP = Path("/content/gecos_lstm_full_job_outputs.zip")


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if (
            destination_resolved not in target.parents
            and target != destination_resolved
        ):
            raise RuntimeError(
                f"bundle에 안전하지 않은 경로가 있습니다: {member.filename}"
            )
    archive.extractall(destination)


def main() -> int:
    if not BUNDLE_PATH.is_file():
        raise RuntimeError(f"업로드 bundle이 없습니다: {BUNDLE_PATH}")
    if not JOB_PATH.is_file():
        raise RuntimeError(f"업로드 job descriptor가 없습니다: {JOB_PATH}")
    if (
        WORKSPACE.resolve().parent != Path("/content").resolve()
        or WORKSPACE.name != "gecos"
    ):
        raise RuntimeError(f"삭제 가능한 임시 작업 경로가 아닙니다: {WORKSPACE}")
    for module_name in tuple(sys.modules):
        if module_name == "scripts" or module_name.startswith("scripts."):
            del sys.modules[module_name]
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        _safe_extract(archive, WORKSPACE)
    os.chdir(WORKSPACE)
    sys.path.insert(0, str(WORKSPACE))

    from scripts.lstm_full_contract import load_lstm_full_config
    from scripts.run_lstm_full_training_job import run_lstm_full_training_job

    config = load_lstm_full_config(
        WORKSPACE / "configs" / "lstm_full_training_milan_nov2013.json",
        base_directory=WORKSPACE,
    )
    manifest = run_lstm_full_training_job(config, JOB_PATH)
    output_dir = config.outputs.jobs_root / manifest["job"]["job_id"]
    output_paths = tuple(
        output_dir / name
        for name in (
            "training_report.json",
            "validation_predictions.npz",
            "best_weights.npz",
            "run_manifest.json",
        )
    )
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_paths:
            if not path.is_file():
                raise RuntimeError(f"job output이 없습니다: {path}")
            archive.write(path, path.relative_to(WORKSPACE).as_posix())
    report = json.loads(output_paths[0].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "job_id": manifest["job"]["job_id"],
                "completed_epochs": report["selection"]["completed_epochs"],
                "best_epoch": report["selection"]["best_epoch"],
                "best_scaled_validation_mae": report["selection"][
                    "best_scaled_validation_mae"
                ],
                "output_zip": str(OUTPUT_ZIP),
                "tensorflow_version": manifest["runtime"]["tensorflow_version"],
                "gpu": manifest["runtime"]["nvidia_smi_inventory"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["status"] == "pass" else 2


if __name__ == "__main__":
    exit_status = main()
    if exit_status:
        raise RuntimeError("LSTM full training job 필수 gate를 통과하지 못했습니다.")
