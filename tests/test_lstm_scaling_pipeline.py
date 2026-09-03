from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from scripts.lstm_scaling_contract import (
    LstmScalingContractError,
    load_lstm_scaling_config,
)
from scripts.prepare_lstm_scaling_pilot import (
    fit_per_cell_minmax,
    inverse_transform_cellwise,
    transform_cellwise,
)
from scripts.run_lstm_scaling_pilot import (
    classify_scaling_result,
    expected_bundle_array_names,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "lstm_scaling_pilot_milan_nov2013.json"


class CellwiseMinMaxTests(unittest.TestCase):
    def test_fit_uses_only_values_passed_as_train(self) -> None:
        train = np.array([[0, 5, 10], [100, 120, 140]], dtype=np.float32)
        validation_extreme = np.array([[1000], [-1000]], dtype=np.float32)

        minimum, cell_range = fit_per_cell_minmax(train)
        scaled_validation = transform_cellwise(validation_extreme, minimum, cell_range)

        np.testing.assert_array_equal(minimum, [0, 100])
        np.testing.assert_array_equal(cell_range, [10, 40])
        self.assertGreater(float(scaled_validation[0, 0]), 1.0)
        self.assertLess(float(scaled_validation[1, 0]), 0.0)

    def test_transform_is_cell_specific_and_roundtrips_without_clipping(self) -> None:
        raw = np.array(
            [
                [[[0.0], [5.0], [10.0]], [[15.0], [-5.0], [2.5]]],
                [[[100.0], [120.0], [140.0]], [[180.0], [80.0], [110.0]]],
            ],
            dtype=np.float32,
        )
        minimum = np.array([0, 100], dtype=np.float32)
        cell_range = np.array([10, 40], dtype=np.float32)

        scaled = transform_cellwise(raw, minimum, cell_range)
        restored = inverse_transform_cellwise(scaled, minimum, cell_range)

        self.assertEqual(float(scaled[0, 1, 0, 0]), 1.5)
        self.assertEqual(float(scaled[1, 1, 0, 0]), 2.0)
        self.assertEqual(float(scaled[0, 1, 1, 0]), -0.5)
        np.testing.assert_allclose(restored, raw, rtol=0.0, atol=1e-6)

    def test_zero_range_cell_is_rejected(self) -> None:
        with self.assertRaisesRegex(LstmScalingContractError, "range"):
            fit_per_cell_minmax(np.array([[1, 1, 1], [1, 2, 3]], dtype=np.float32))

    def test_mismatched_cell_axis_is_rejected(self) -> None:
        with self.assertRaisesRegex(LstmScalingContractError, "cell 축"):
            transform_cellwise(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.ones(3, dtype=np.float32),
            )


class ScalingPilotDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_lstm_scaling_config(CONFIG_PATH)

    def test_material_positive_and_reject_categories_are_exclusive(self) -> None:
        material = classify_scaling_result(self.config, 180.0)
        positive = classify_scaling_result(self.config, 220.0)
        rejected = classify_scaling_result(self.config, 240.0)

        self.assertEqual(material["category"], "material_improvement")
        self.assertEqual(positive["category"], "positive_but_below_material")
        self.assertEqual(rejected["category"], "no_improvement")
        self.assertFalse(material["test_used"])

    def test_bundle_contract_has_only_train_and_validation(self) -> None:
        names = expected_bundle_array_names(self.config)

        self.assertIn("x_train", names)
        self.assertIn("raw_y_validation", names)
        self.assertFalse(any("test" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main()
