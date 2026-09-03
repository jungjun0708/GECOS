#!/usr/bin/env python3
"""공식 checksum과 같은 Milano Grid 사본을 원자적으로 내려받는다."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO

from scripts.select_central_900 import (
    DEFAULT_CONFIG,
    CentralSelectionError,
    GridReference,
    load_grid_reference,
    load_selection_config,
    verify_grid_source,
)

TOOL_VERSION = "1.0.0"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> int:
    copied = 0
    while chunk := source.read(DOWNLOAD_CHUNK_SIZE):
        destination.write(chunk)
        copied += len(chunk)
    return copied


def fetch_grid(
    reference: GridReference,
    target: Path,
    source_url: str,
) -> tuple[str, dict[str, object]]:
    """정상 파일은 재사용하고, 없을 때만 임시 경로에서 검증한 뒤 공개한다."""

    if target.exists():
        return "already_present", verify_grid_source(target, reference)

    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": f"GECOS-reproduction/{TOOL_VERSION}"},
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix=".milano-grid-download-", dir=target.parent
        ) as temporary_directory:
            temporary_path = Path(temporary_directory) / reference.filename
            try:
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    temporary_path.open("wb") as handle,
                ):
                    copied = _copy_stream(response, handle)
            except (OSError, urllib.error.URLError) as exc:
                raise CentralSelectionError(
                    f"Milano Grid를 내려받지 못했습니다: {source_url}"
                ) from exc
            if copied != reference.size_bytes:
                raise CentralSelectionError(
                    f"다운로드 크기가 공식 기준과 다릅니다: {copied} != {reference.size_bytes}"
                )
            report = verify_grid_source(temporary_path, reference)
            os.replace(temporary_path, target)
    except OSError as exc:
        raise CentralSelectionError(
            f"다운로드 파일을 게시할 수 없습니다: {target}"
        ) from exc
    return "downloaded", {**report, "path": str(target.resolve())}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="공식 MD5와 같은 공개 미러의 Milano Grid를 내려받습니다."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"중앙 셀 선택 config 경로 (기본값: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--output", type=Path, help="기본 data/raw 출력 경로 대신 사용")
    parser.add_argument(
        "--source-url",
        help="metadata에 고정한 공개 미러 대신 사용할 다운로드 URL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_selection_config(args.config)
        reference = load_grid_reference(config.grid_reference_manifest)
        target = args.output.resolve() if args.output else config.inputs.grid_geojson
        source_url = args.source_url or reference.acquisition.get("verified_mirror_url")
        if not isinstance(source_url, str) or not source_url.strip():
            raise CentralSelectionError("metadata에 verified_mirror_url이 없습니다.")
        status, report = fetch_grid(reference, target, source_url)
    except CentralSelectionError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1

    print(
        f"Milano Grid 준비 완료: status={status}, "
        f"bytes={report['size_bytes']}, md5={report['md5']}"
    )
    print(f"path={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
