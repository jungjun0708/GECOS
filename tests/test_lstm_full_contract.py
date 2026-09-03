from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts.lstm_full_contract import LstmFullContractError, load_lstm_full_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "lstm_full_training_milan_nov2013.json"


class LstmFullConfigTests(unittest.TestCase):
    def test_registered_jobs_data_and_training_contract(self) -> None:
        config = load_lstm_full_config(CONFIG_PATH)

        self.assertEqual(config.training.seeds, (42, 43, 44))
        self.assertEqual(len(config.jobs), 9)
        self.assertEqual(
            [job.expected_cell_count for job in config.jobs[:3]], [900, 611, 289]
        )
        self.assertEqual(config.data.split("train").targets_per_cell, 2872)
        self.assertEqual(config.data.split("validation").targets_per_cell, 720)
        self.assertEqual(config.data.bundle_global_end_index_exclusive, 3600)
        self.assertEqual(config.data.test_target_start_index_inclusive, 3600)
        self.assertIn("excluded", config.data.test_policy)
        self.assertEqual(config.scaling.fit_end_index_exclusive, 2880)
        self.assertEqual(config.training.early_stopping.monitor, "val_loss")
        self.assertEqual(
            config.training.early_stopping.monitor_domain,
            "cellwise_scaled_mae",
        )
        self.assertTrue(config.training.early_stopping.restore_best_weights)
        self.assertFalse(config.pass_criteria.require_better_than_persistence)
        self.assertFalse(config.pass_criteria.require_upc_improvement)

    def _assert_mutation_rejected(
        self, mutate: Callable[[dict[str, Any]], None]
    ) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mutate(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(LstmFullContractError):
                load_lstm_full_config(path, base_directory=REPOSITORY_ROOT)

    def test_test_data_cannot_enter_train_validation_bundle(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["data"].__setitem__(
                "bundle_global_end_index_exclusive", 4320
            )
        )
        self._assert_mutation_rejected(
            lambda payload: payload["data"]["splits"].__setitem__(
                "test",
                {
                    "target_start_index_inclusive": 3600,
                    "target_end_index_exclusive": 4320,
                    "targets_per_cell": 720,
                    "samples": 648000,
                },
            )
        )

    def test_scaler_fit_cannot_include_validation(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["scaling"].__setitem__(
                "fit_end_index_exclusive", 3600
            )
        )

    def test_seeds_and_job_count_cannot_change(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["training"].__setitem__("seeds", [42])
        )
        self._assert_mutation_rejected(
            lambda payload: payload["training"].__setitem__("expected_job_count", 3)
        )

    def test_early_stopping_selection_cannot_change(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["training"]["early_stopping"].__setitem__(
                "monitor_domain", "raw_traffic_mae"
            )
        )
        self._assert_mutation_rejected(
            lambda payload: payload["training"]["early_stopping"].__setitem__(
                "patience", 10
            )
        )

    def test_performance_cannot_become_pipeline_gate(self) -> None:
        self._assert_mutation_rejected(
            lambda payload: payload["pass_criteria"].__setitem__(
                "require_upc_improvement", True
            )
        )


if __name__ == "__main__":
    unittest.main()
