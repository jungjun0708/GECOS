from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.lstm_scaling_contract import (
    LstmScalingContractError,
    load_lstm_scaling_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "lstm_scaling_pilot_milan_nov2013.json"


class LstmScalingConfigTests(unittest.TestCase):
    def test_registered_train_only_contract(self) -> None:
        config = load_lstm_scaling_config(CONFIG_PATH)

        self.assertEqual(config.selection.splits, ("train", "validation"))
        self.assertEqual(config.test_policy, "withheld_not_bundled_or_evaluated")
        self.assertEqual(config.scaling.fit_start_index_inclusive, 0)
        self.assertEqual(config.scaling.fit_end_index_exclusive, 2880)
        self.assertFalse(config.scaling.clip_transform)
        self.assertFalse(config.scaling.clip_inverse_prediction)
        self.assertEqual(config.training.max_epochs, 5)
        self.assertEqual(config.training.input_scaling, "per_cell_train_only_minmax")
        self.assertEqual(config.raw_reference.primary_model, "lstm_upc_off")
        self.assertAlmostEqual(
            config.decision_rule.material_improvement_max_mae,
            189.0282308736,
            places=10,
        )

    def _assert_mutation_rejected(self, mutate: object) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutate(payload)  # type: ignore[operator]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(LstmScalingContractError):
                load_lstm_scaling_config(path, base_directory=REPOSITORY_ROOT)

    def test_validation_cannot_enter_scaler_fit(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["scaling"].__setitem__(
                "fit_end_index_exclusive", 3600
            )
        )

    def test_transform_or_inverse_clipping_cannot_be_enabled(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["scaling"].__setitem__("clip_transform", True)
        )
        self._assert_mutation_rejected(
            lambda payload: payload["scaling"].__setitem__(
                "clip_inverse_prediction", True
            )
        )

    def test_test_split_cannot_be_selected(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["selection"]["splits"].append("test")
        )

    def test_decision_threshold_cannot_be_changed_after_registration(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["decision_rule"].__setitem__(
                "material_improvement_max_mae", 200.0
            )
        )


if __name__ == "__main__":
    unittest.main()
