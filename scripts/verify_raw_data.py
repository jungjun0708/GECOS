#!/usr/bin/env python3
"""Telecom Italia 원본 파일을 공식 manifest와 대조한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "1.0.0"
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = REPOSITORY_ROOT / "metadata" / "telecom_italia_mi_2013_11.json"
DEFAULT_DATA_DIRECTORY = REPOSITORY_ROOT / "dataverse_files"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "interim" / "raw_integrity_report.json"
MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class ManifestError(ValueError):
    """기준 manifest가 유효하지 않을 때 발생한다."""


@dataclass(frozen=True)
class ExpectedFile:
    """공식 메타데이터에서 고정한 파일 정보."""

    name: str
    size_bytes: int
    checksum: str


@dataclass(frozen=True)
class ReferenceManifest:
    """검증에 필요한 기준 manifest의 정규화된 표현."""

    schema_version: int
    source: Mapping[str, Any]
    file_glob: str
    expected_file_count: int
    expected_total_bytes: int
    checksum_algorithm: str
    files: tuple[ExpectedFile, ...]


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field}는 JSON object여야 합니다.")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{field}는 {minimum} 이상의 정수여야 합니다.")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field}는 비어 있지 않은 문자열이어야 합니다.")
    return value


def load_reference_manifest(path: Path) -> ReferenceManifest:
    """JSON 기준 manifest를 읽고 내부 일관성을 검증한다."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"기준 manifest를 읽을 수 없습니다: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"기준 manifest가 올바른 JSON이 아닙니다: {exc}") from exc

    root = _require_mapping(payload, "root")
    schema_version = _require_int(
        root.get("schema_version"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise ManifestError(f"지원하지 않는 schema_version입니다: {schema_version}")

    source = _require_mapping(root.get("source"), "source")
    for field in ("title", "persistent_id", "doi_url", "dataset_version", "license"):
        _require_string(source.get(field), f"source.{field}")

    selection = _require_mapping(root.get("selection"), "selection")
    file_glob = _require_string(selection.get("file_glob"), "selection.file_glob")
    expected_file_count = _require_int(
        selection.get("expected_file_count"),
        "selection.expected_file_count",
        minimum=1,
    )
    expected_total_bytes = _require_int(
        selection.get("expected_total_bytes"),
        "selection.expected_total_bytes",
    )

    integrity = _require_mapping(root.get("integrity"), "integrity")
    checksum_algorithm = _require_string(
        integrity.get("algorithm"), "integrity.algorithm"
    ).lower()
    if checksum_algorithm != "md5":
        raise ManifestError(
            "현재 검증기는 공식 Dataverse가 제공한 MD5 manifest만 지원합니다."
        )

    raw_files = root.get("files")
    if not isinstance(raw_files, list):
        raise ManifestError("files는 JSON array여야 합니다.")

    files: list[ExpectedFile] = []
    seen_names: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        item = _require_mapping(raw_file, f"files[{index}]")
        name = _require_string(item.get("name"), f"files[{index}].name")
        if Path(name).name != name:
            raise ManifestError(f"파일명에는 디렉터리를 포함할 수 없습니다: {name}")
        if name in seen_names:
            raise ManifestError(f"중복 파일명이 있습니다: {name}")

        size_bytes = _require_int(item.get("size_bytes"), f"files[{index}].size_bytes")
        checksum = _require_string(
            item.get("checksum"), f"files[{index}].checksum"
        ).lower()
        if not MD5_PATTERN.fullmatch(checksum):
            raise ManifestError(f"올바르지 않은 MD5 값입니다: {name}")

        seen_names.add(name)
        files.append(ExpectedFile(name=name, size_bytes=size_bytes, checksum=checksum))

    files.sort(key=lambda item: item.name)
    if len(files) != expected_file_count:
        raise ManifestError(
            "파일 수가 manifest 요약과 다릅니다: "
            f"files={len(files)}, expected_file_count={expected_file_count}"
        )

    calculated_total = sum(item.size_bytes for item in files)
    if calculated_total != expected_total_bytes:
        raise ManifestError(
            "파일 크기 합계가 manifest 요약과 다릅니다: "
            f"files={calculated_total}, expected_total_bytes={expected_total_bytes}"
        )

    return ReferenceManifest(
        schema_version=schema_version,
        source=dict(source),
        file_glob=file_glob,
        expected_file_count=expected_file_count,
        expected_total_bytes=expected_total_bytes,
        checksum_algorithm=checksum_algorithm,
        files=tuple(files),
    )


def compute_digest(
    path: Path, algorithm: str, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> str:
    """파일 전체를 메모리에 올리지 않고 digest를 계산한다."""

    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")

    if algorithm.lower() == "md5":
        try:
            digest = hashlib.md5(usedforsecurity=False)
        except TypeError:  # usedforsecurity 인자를 지원하지 않는 hashlib 구현 대응
            digest = hashlib.md5()
    else:
        digest = hashlib.new(algorithm)

    buffer = bytearray(chunk_size)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as handle:
        while read_size := handle.readinto(buffer):
            digest.update(view[:read_size])
    return digest.hexdigest()


def verify_data_directory(
    reference: ReferenceManifest,
    data_directory: Path,
    *,
    quick: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """데이터 디렉터리를 검증하고 직렬화 가능한 결과를 반환한다."""

    if chunk_size <= 0:
        raise ValueError("chunk_size는 0보다 커야 합니다.")

    directory_exists = data_directory.is_dir()
    observed_paths = (
        {
            path.name: path
            for path in data_directory.glob(reference.file_glob)
            if path.is_file()
        }
        if directory_exists
        else {}
    )
    expected_names = {item.name for item in reference.files}
    observed_names = set(observed_paths)
    missing_files = sorted(expected_names - observed_names)
    unexpected_files = sorted(observed_names - expected_names)

    file_results: list[dict[str, Any]] = []
    size_mismatches = 0
    checksum_mismatches = 0
    checksum_verified = 0
    size_verified = 0
    observed_expected_bytes = 0

    for index, expected in enumerate(reference.files, start=1):
        path = observed_paths.get(expected.name)
        if path is None:
            file_results.append(
                {
                    "name": expected.name,
                    "status": "missing",
                    "expected_size_bytes": expected.size_bytes,
                    "actual_size_bytes": None,
                    "expected_checksum": expected.checksum,
                    "actual_checksum": None,
                }
            )
            if progress:
                progress(
                    f"[{index:02d}/{reference.expected_file_count}] "
                    f"{expected.name}: 누락"
                )
            continue

        actual_size = path.stat().st_size
        observed_expected_bytes += actual_size
        if actual_size != expected.size_bytes:
            size_mismatches += 1
            status = "size_mismatch"
            actual_checksum = None
        elif quick:
            size_verified += 1
            status = "size_only_ok"
            actual_checksum = None
        else:
            size_verified += 1
            if progress:
                progress(
                    f"[{index:02d}/{reference.expected_file_count}] "
                    f"{expected.name}: MD5 계산 중"
                )
            actual_checksum = compute_digest(
                path, reference.checksum_algorithm, chunk_size=chunk_size
            )
            if actual_checksum == expected.checksum:
                checksum_verified += 1
                status = "ok"
            else:
                checksum_mismatches += 1
                status = "checksum_mismatch"

        file_results.append(
            {
                "name": expected.name,
                "status": status,
                "expected_size_bytes": expected.size_bytes,
                "actual_size_bytes": actual_size,
                "expected_checksum": expected.checksum,
                "actual_checksum": actual_checksum,
            }
        )
        if progress:
            progress(
                f"[{index:02d}/{reference.expected_file_count}] "
                f"{expected.name}: {status}"
            )

    unexpected_results = [
        {"name": name, "size_bytes": observed_paths[name].stat().st_size}
        for name in unexpected_files
    ]
    checks_passed = not any(
        (
            not directory_exists,
            missing_files,
            unexpected_files,
            size_mismatches,
            checksum_mismatches,
        )
    )
    if not checks_passed:
        status = "failed"
    elif quick:
        status = "passed_size_only"
    else:
        status = "passed"

    return {
        "report_schema_version": 1,
        "tool": {"name": "verify_raw_data", "version": TOOL_VERSION},
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": dict(reference.source),
        "verification": {
            "mode": "size_only" if quick else "size_and_checksum",
            "status": status,
            "checks_passed": checks_passed,
            "integrity_verified": checks_passed and not quick,
            "checksum_algorithm": reference.checksum_algorithm,
            "chunk_size_bytes": chunk_size,
            "data_directory_exists": directory_exists,
        },
        "summary": {
            "expected_file_count": reference.expected_file_count,
            "observed_expected_file_count": len(observed_names & expected_names),
            "size_verified_file_count": size_verified,
            "checksum_verified_file_count": checksum_verified,
            "missing_file_count": len(missing_files),
            "unexpected_file_count": len(unexpected_files),
            "size_mismatch_count": size_mismatches,
            "checksum_mismatch_count": checksum_mismatches,
            "expected_total_bytes": reference.expected_total_bytes,
            "observed_expected_total_bytes": observed_expected_bytes,
        },
        "missing_files": missing_files,
        "unexpected_files": unexpected_results,
        "files": file_results,
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    """완성되지 않은 보고서가 남지 않도록 JSON을 원자적으로 저장한다."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Telecom Italia 2013년 11월 Milano 원본 30개를 공식 크기와 MD5로 검증합니다."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help=f"원본 파일 디렉터리 (기본값: {DEFAULT_DATA_DIRECTORY})",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"기준 manifest JSON (기본값: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"검증 보고서 경로 (기본값: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="파일명과 크기만 확인합니다. 이 결과는 무결성 검증으로 간주하지 않습니다.",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=DEFAULT_CHUNK_SIZE // (1024 * 1024),
        help="checksum 계산 버퍼 크기 MiB (기본값: 8)",
    )
    parser.add_argument("--version", action="version", version=TOOL_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_size_mib <= 0:
        parser.error("--chunk-size-mib는 0보다 커야 합니다.")

    try:
        reference = load_reference_manifest(args.reference)
        report = verify_data_directory(
            reference,
            args.data_dir,
            quick=args.quick,
            chunk_size=args.chunk_size_mib * 1024 * 1024,
            progress=lambda message: print(message, flush=True),
        )
        report["reference_manifest"] = {
            "path": _display_path(args.reference),
            "sha256": compute_digest(args.reference, "sha256"),
        }
        write_report(report, args.output)
    except (ManifestError, OSError, ValueError) as exc:
        print(f"검증을 실행할 수 없습니다: {exc}", file=sys.stderr)
        return 2

    summary = report["summary"]
    status = report["verification"]["status"]
    print(
        "요약: "
        f"status={status}, "
        f"files={summary['observed_expected_file_count']}/"
        f"{summary['expected_file_count']}, "
        f"bytes={summary['observed_expected_total_bytes']}/"
        f"{summary['expected_total_bytes']}"
    )
    print(f"보고서: {_display_path(args.output)}")

    if status == "passed_size_only":
        print(
            "주의: 빠른 검사는 MD5를 확인하지 않았으므로 무결성이 확정되지 않았습니다."
        )
    elif status == "passed":
        print("원본 데이터 무결성 검증에 성공했습니다.")
    else:
        print("원본 데이터 검증에 실패했습니다. 보고서의 실패 파일을 확인하세요.")
    return 0 if report["verification"]["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
