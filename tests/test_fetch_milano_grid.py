from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.fetch_milano_grid import fetch_grid
from scripts.select_central_900 import CentralSelectionError, GridReference


def md5_bytes(payload: bytes) -> str:
    try:
        return hashlib.md5(payload, usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(payload).hexdigest()


class FetchMilanoGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.payload = b'{"type":"FeatureCollection","features":[]}'
        self.source = self.root / "source" / "milano-grid.geojson"
        self.target = self.root / "target" / "milano-grid.geojson"
        self.source.parent.mkdir()
        self.source.write_bytes(self.payload)
        self.reference = GridReference(
            path=self.root / "reference.json",
            source={"doi_url": "https://example.test/grid"},
            acquisition={},
            filename="milano-grid.geojson",
            size_bytes=len(self.payload),
            checksum_algorithm="md5",
            checksum=md5_bytes(self.payload),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_download_is_verified_before_target_is_published(self) -> None:
        status, report = fetch_grid(self.reference, self.target, self.source.as_uri())

        self.assertEqual(status, "downloaded")
        self.assertTrue(report["official_checksum_matched"])
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_valid_existing_target_is_reused(self) -> None:
        self.target.parent.mkdir()
        self.target.write_bytes(self.payload)

        status, _ = fetch_grid(
            self.reference,
            self.target,
            "https://invalid.example.test/not-requested",
        )

        self.assertEqual(status, "already_present")

    def test_checksum_failure_does_not_publish_target(self) -> None:
        bad_source = self.root / "bad" / "milano-grid.geojson"
        bad_source.parent.mkdir()
        bad_source.write_bytes(b"x" * len(self.payload))

        with self.assertRaisesRegex(CentralSelectionError, "MD5"):
            fetch_grid(self.reference, self.target, bad_source.as_uri())

        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
