from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_rctl_smoke import (
    build_smoke_arrays,
    evenly_spaced_positions,
    select_spatially_spread_cells,
)
from scripts.rctl_contract import (
    RctlContractError,
    SelectionSpec,
    load_rctl_smoke_config,
)
from scripts.rctl_model import audit_rctl_model, build_rctl_model

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rctl_smoke_milan_nov2013.json"


class RctlConfigTests(unittest.TestCase):
    def test_registered_paper_and_public_variants_remain_distinct(self) -> None:
        config = load_rctl_smoke_config(CONFIG_PATH)

        paper = config.variants["paper_interpretation"]
        public = config.variants["public_reference"]
        self.assertEqual(paper.kernel_size, 4)
        self.assertEqual(paper.dilations, (1, 2, 4, 8, 16, 32))
        self.assertEqual(paper.tcn_shortcut_merge, "concatenate")
        self.assertEqual(paper.rcc2_routes, ((3, 2), (4, 1), (5, 0)))
        self.assertEqual(paper.expected_parameter_count, 173633)
        self.assertEqual(public.kernel_size, 3)
        self.assertEqual(public.dilations, (1, 2, 4, 6, 8, 10))
        self.assertEqual(public.tcn_shortcut_merge, "add")
        self.assertEqual(public.rcc2_routes, ((2, 2), (3, 2), (4, 1), (5, 0)))

    def test_changed_paper_merge_is_rejected(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["architecture"]["variants"]["paper_interpretation"][
            "tcn_shortcut_merge"
        ] = "add"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RctlContractError, "paper_interpretation"):
                load_rctl_smoke_config(path, base_directory=REPOSITORY_ROOT)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"), "TensorFlow는 Colab 모델 감사에서 실행"
    )
    def test_tensorflow_model_shapes_gradients_causality_and_counts(self) -> None:
        config = load_rctl_smoke_config(CONFIG_PATH)
        expected_counts = {
            "paper_interpretation": 236657,
            "public_reference": 173665,
        }
        for name, spec in config.variants.items():
            model = build_rctl_model(
                steps=config.input_length,
                spec=spec,
                dropout=config.training.dropout,
                learning_rate=config.training.learning_rate,
                compile_model=False,
            )
            report = audit_rctl_model(
                model,
                spec=spec,
                steps=config.input_length,
                seed=config.seed,
            )
            self.assertTrue(report["required_gates_passed"], name)
            self.assertEqual(model.output_shape, (None, 1))
            self.assertEqual(model.count_params(), expected_counts[name])


class RctlSmokeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.central_rows = [
            {
                "cell_id": str(100 + row * 6 + column),
                "grid_row": str(row),
                "grid_column": str(column),
                "centroid_lon": "9.0",
                "centroid_lat": "45.0",
            }
            for row in range(6)
            for column in range(6)
        ]
        self.central_positions = np.arange(36, dtype=np.int64)

    def test_evenly_spaced_selection_includes_both_ends(self) -> None:
        np.testing.assert_array_equal(
            evenly_spaced_positions(30, 4), np.array([0, 10, 19, 29])
        )

    def test_spatial_selection_is_row_major_and_deterministic(self) -> None:
        _, positions, coordinates = select_spatially_spread_cells(
            self.central_rows, self.central_positions, 4
        )

        self.assertEqual([0, 2, 3, 5], sorted({row["grid_row"] for row in coordinates}))
        self.assertEqual(
            [0, 2, 3, 5], sorted({row["grid_column"] for row in coordinates})
        )
        self.assertEqual(coordinates[0]["cell_id"], 100)
        self.assertEqual(coordinates[-1]["cell_id"], 135)
        self.assertEqual(positions.tolist(), [row["all_cell_matrix_position"] for row in coordinates])

    def test_window_target_and_persistence_alignment(self) -> None:
        steps = 20
        traffic = np.stack(
            [np.arange(steps, dtype=np.float32) + 1000 * cell for cell in range(36)]
        )
        masks = np.zeros_like(traffic, dtype=bool)
        cell_ids = np.arange(100, 136, dtype=np.int32)
        timestamps = np.arange(steps, dtype=np.int64) * 600_000
        selection = SelectionSpec(
            central_grid_side=4,
            windows_per_cell=3,
            cell_policy="synthetic_spatial",
            target_policy="synthetic_train",
        )

        arrays, metadata = build_smoke_arrays(
            traffic=traffic,
            missing_mask=masks,
            internet_null_mask=masks,
            cell_ids=cell_ids,
            timestamps_ms=timestamps,
            central_rows=self.central_rows,
            central_positions=self.central_positions,
            train_target_indices=np.arange(8, 15, dtype=np.int64),
            input_length=8,
            selection=selection,
        )

        self.assertEqual(arrays["x"].shape, (48, 8, 1))
        self.assertEqual(arrays["y"].shape, (48, 1))
        np.testing.assert_array_equal(arrays["x"][0, :, 0], np.arange(8))
        self.assertEqual(arrays["y"][0, 0], 8)
        self.assertEqual(arrays["persistence"][0, 0], 7)
        np.testing.assert_array_equal(
            arrays["target_indices"][:3], np.array([8, 11, 14])
        )
        self.assertEqual(metadata["sample_count"], 48)
        self.assertEqual(metadata["target_missing_count"], 0)


if __name__ == "__main__":
    unittest.main()
