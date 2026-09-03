from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scripts.lstm_contract import (
    EXPECTED_PARAMETER_COUNT,
    LstmSmokeContractError,
    load_lstm_smoke_config,
    reconstructed_lstm_parameter_count,
)
from scripts.lstm_model import audit_lstm_model, build_lstm_model, configure_determinism

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "lstm_upc_smoke_milan_nov2013.json"


class LstmSmokeConfigTests(unittest.TestCase):
    def test_registered_contract_and_parameter_arithmetic(self) -> None:
        config = load_lstm_smoke_config(CONFIG_PATH)

        self.assertEqual(config.upc_protocol, "train_only")
        self.assertEqual(config.architecture.name, "paper_parameter_reconstruction")
        self.assertEqual(config.architecture.units, (64, 128, 64))
        self.assertEqual(config.architecture.return_sequences, (True, True, False))
        self.assertEqual(config.architecture.expected_parameter_count, 165185)
        self.assertFalse(config.architecture.author_implementation_confirmed)
        self.assertEqual(config.selection.expected_cluster_counts, ((0, 611), (1, 289)))
        self.assertEqual(config.selection.expected_samples_per_split, 57600)
        self.assertEqual(config.training.max_epochs, 5)
        self.assertFalse(config.training.shuffle)
        self.assertFalse(config.pass_criteria.require_better_than_persistence)
        self.assertEqual(
            reconstructed_lstm_parameter_count((64, 128, 64)),
            EXPECTED_PARAMETER_COUNT,
        )

    def test_architecture_change_is_rejected(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["architecture"]["lstm_units"] = [64, 64, 64]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LstmSmokeContractError, "architecture"):
                load_lstm_smoke_config(path, base_directory=REPOSITORY_ROOT)

    def test_smoke_cannot_claim_author_confirmation(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["architecture"]["author_implementation_confirmed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LstmSmokeContractError, "architecture"):
                load_lstm_smoke_config(path, base_directory=REPOSITORY_ROOT)

    def test_full_month_protocol_is_rejected_by_config(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        payload["upc_protocol"] = "algorithm1_full_month"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(LstmSmokeContractError, "train_only"):
                load_lstm_smoke_config(path, base_directory=REPOSITORY_ROOT)

    @unittest.skipUnless(
        importlib.util.find_spec("tensorflow"), "TensorFlow 구조 감사는 Colab에서 실행"
    )
    def test_tensorflow_parameter_count_shape_and_gradients(self) -> None:
        config = load_lstm_smoke_config(CONFIG_PATH)
        configure_determinism(config.seed)
        model = build_lstm_model(
            spec=config.architecture,
            dropout=config.training.dropout,
            learning_rate=config.training.learning_rate,
            compile_model=False,
        )
        report = audit_lstm_model(model, spec=config.architecture, seed=config.seed)

        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(model.count_params(), 165185)
        self.assertTrue(report["required_gates_passed"])


if __name__ == "__main__":
    unittest.main()
