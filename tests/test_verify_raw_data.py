from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from scripts.verify_raw_data import (
    ManifestError,
    load_reference_manifest,
    main,
    verify_data_directory,
    write_report,
)


def md5_bytes(payload: bytes) -> str:
    try:
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()
    except TypeError:  # usedforsecurity 인자를 지원하지 않는 hashlib 구현 대응
        return hashlib.md5(payload).hexdigest()


class VerifyRawDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_directory = self.root / "raw"
        self.data_directory.mkdir()
        self.reference_path = self.root / "reference.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_reference(self, files: dict[str, bytes]) -> None:
        manifest = {
            "schema_version": 1,
            "source": {
                "title": "test dataset",
                "persistent_id": "doi:test/example",
                "doi_url": "https://example.test/dataset",
                "dataset_version": "1.0",
                "license": "test-only",
            },
            "selection": {
                "file_glob": "sample-*.txt",
                "expected_file_count": len(files),
                "expected_total_bytes": sum(len(payload) for payload in files.values()),
            },
            "integrity": {"algorithm": "md5"},
            "files": [
                {
                    "name": name,
                    "size_bytes": len(payload),
                    "checksum": md5_bytes(payload),
                }
                for name, payload in files.items()
            ],
        }
        self.reference_path.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )

    def create_files(self, files: dict[str, bytes]) -> None:
        for name, payload in files.items():
            (self.data_directory / name).write_bytes(payload)

    def test_full_verification_passes_and_report_is_writable(self) -> None:
        files = {"sample-01.txt": b"alpha\n", "sample-02.txt": b"beta\n"}
        self.write_reference(files)
        self.create_files(files)

        reference = load_reference_manifest(self.reference_path)
        report = verify_data_directory(reference, self.data_directory, chunk_size=3)
        output_path = self.root / "reports" / "result.json"
        write_report(report, output_path)

        written_report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(written_report["verification"]["status"], "passed")
        self.assertTrue(written_report["verification"]["integrity_verified"])
        self.assertEqual(written_report["summary"]["checksum_verified_file_count"], 2)

    def test_missing_and_unexpected_files_fail(self) -> None:
        expected = {"sample-01.txt": b"alpha", "sample-02.txt": b"beta"}
        self.write_reference(expected)
        self.create_files({"sample-01.txt": b"alpha", "sample-03.txt": b"extra"})

        report = verify_data_directory(
            load_reference_manifest(self.reference_path), self.data_directory
        )

        self.assertEqual(report["verification"]["status"], "failed")
        self.assertEqual(report["missing_files"], ["sample-02.txt"])
        self.assertEqual(report["unexpected_files"][0]["name"], "sample-03.txt")

    def test_size_mismatch_skips_checksum(self) -> None:
        self.write_reference({"sample-01.txt": b"abc"})
        self.create_files({"sample-01.txt": b"too-long"})

        report = verify_data_directory(
            load_reference_manifest(self.reference_path), self.data_directory
        )

        self.assertEqual(report["summary"]["size_mismatch_count"], 1)
        self.assertEqual(report["files"][0]["status"], "size_mismatch")
        self.assertIsNone(report["files"][0]["actual_checksum"])

    def test_checksum_mismatch_fails_when_size_matches(self) -> None:
        self.write_reference({"sample-01.txt": b"abc"})
        self.create_files({"sample-01.txt": b"xyz"})

        report = verify_data_directory(
            load_reference_manifest(self.reference_path), self.data_directory
        )

        self.assertEqual(report["summary"]["checksum_mismatch_count"], 1)
        self.assertEqual(report["files"][0]["status"], "checksum_mismatch")

    def test_quick_mode_does_not_claim_integrity(self) -> None:
        self.write_reference({"sample-01.txt": b"abc"})
        self.create_files({"sample-01.txt": b"xyz"})

        report = verify_data_directory(
            load_reference_manifest(self.reference_path),
            self.data_directory,
            quick=True,
        )

        self.assertEqual(report["verification"]["status"], "passed_size_only")
        self.assertTrue(report["verification"]["checks_passed"])
        self.assertFalse(report["verification"]["integrity_verified"])
        self.assertEqual(report["summary"]["checksum_verified_file_count"], 0)

    def test_manifest_rejects_duplicate_filenames(self) -> None:
        self.write_reference({"sample-01.txt": b"abc"})
        manifest = json.loads(self.reference_path.read_text(encoding="utf-8"))
        manifest["files"].append(dict(manifest["files"][0]))
        manifest["selection"]["expected_file_count"] = 2
        manifest["selection"]["expected_total_bytes"] = 6
        self.reference_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ManifestError, "중복 파일명"):
            load_reference_manifest(self.reference_path)

    def test_cli_exit_codes_distinguish_pass_failure_and_configuration_error(self) -> None:
        files = {"sample-01.txt": b"abc"}
        self.write_reference(files)
        self.create_files(files)
        output_path = self.root / "report.json"
        arguments = [
            "--data-dir",
            str(self.data_directory),
            "--reference",
            str(self.reference_path),
            "--output",
            str(output_path),
        ]

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(arguments), 0)

            (self.data_directory / "sample-01.txt").unlink()
            self.assertEqual(main(arguments), 1)

            self.reference_path.write_text("not-json", encoding="utf-8")
            self.assertEqual(main(arguments), 2)


if __name__ == "__main__":
    unittest.main()
