#!/usr/bin/env python3
"""업로드한 scaling pilot bundle을 Colab에서 실행하고 결과 ZIP을 만든다."""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

BUNDLE_PATH = Path("/content/gecos_lstm_scaling_pilot_bundle.zip")
WORKSPACE = Path("/content/gecos")
OUTPUT_ZIP = Path("/content/gecos_lstm_scaling_pilot_outputs.zip")


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

    from scripts.lstm_scaling_contract import load_lstm_scaling_config
    from scripts.run_lstm_scaling_pilot import run_lstm_scaling_pilot

    config = load_lstm_scaling_config(
        WORKSPACE / "configs" / "lstm_scaling_pilot_milan_nov2013.json",
        base_directory=WORKSPACE,
    )
    manifest = run_lstm_scaling_pilot(config)
    output_paths = (
        config.outputs.architecture_report,
        config.outputs.evaluation_report,
        config.outputs.predictions_npz,
        config.outputs.per_cell_metrics_csv,
        config.outputs.run_manifest,
    )
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_paths:
            archive.write(path, path.relative_to(WORKSPACE).as_posix())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "decision": manifest["decision"]["outcome"],
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
        raise RuntimeError("LSTM scaling pilot 필수 gate를 통과하지 못했습니다.")
