#!/usr/bin/env python3
"""Table III parameter 수를 재구성한 LSTM 기준선과 구조 감사 도구."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.lstm_contract import LstmArchitectureSpec, LstmSmokeContractError
from scripts.rctl_model import configure_determinism, require_tensorflow


def build_lstm_model(
    *,
    spec: LstmArchitectureSpec,
    dropout: float,
    learning_rate: float,
    compile_model: bool = True,
) -> Any:
    """64→128→64 stacked-LSTM과 선형 Dense 출력을 만든다."""

    tf = require_tensorflow()
    keras = tf.keras
    if len(spec.units) != len(spec.return_sequences):
        raise LstmSmokeContractError("LSTM units와 return_sequences 길이가 다릅니다.")
    if not 0 <= dropout < 1:
        raise LstmSmokeContractError("dropout은 0 이상 1 미만이어야 합니다.")

    inputs = keras.Input(
        shape=(spec.input_length, 1), dtype=tf.float32, name="traffic_window"
    )
    x = inputs
    for index, (unit_count, return_sequences) in enumerate(
        zip(spec.units, spec.return_sequences, strict=True)
    ):
        x = keras.layers.LSTM(
            unit_count,
            return_sequences=return_sequences,
            recurrent_dropout=0.0,
            name=f"lstm_{index + 1}_{unit_count}",
        )(x)
        x = keras.layers.Dropout(dropout, name=f"dropout_{index + 1}")(x)
    outputs = keras.layers.Dense(spec.output_units, name="forecast")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=spec.name)
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mae",
        )
    return model


def _shape(value: Any) -> list[int | None]:
    return [None if item is None else int(item) for item in value.shape]


def audit_lstm_model(
    model: Any,
    *,
    spec: LstmArchitectureSpec,
    seed: int,
) -> dict[str, Any]:
    """출력 shape, parameter 수, forward와 gradient 건전성을 검사한다."""

    tf = require_tensorflow()
    configure_determinism(seed)
    rng = np.random.default_rng(seed)
    sample = rng.normal(size=(4, spec.input_length, 1)).astype(np.float32)
    target = rng.normal(size=(4, spec.output_units)).astype(np.float32)
    predictions = model(sample, training=False)
    output_shape = list(predictions.shape)
    output_shape_passed = output_shape == [4, spec.output_units]
    finite_forward = bool(tf.reduce_all(tf.math.is_finite(predictions)).numpy())

    with tf.GradientTape() as tape:
        training_predictions = model(sample, training=True)
        loss = tf.reduce_mean(tf.abs(target - training_predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_rows: list[dict[str, Any]] = []
    missing_gradient_count = 0
    nonfinite_gradient_count = 0
    nonzero_gradient_count = 0
    for variable, gradient in zip(model.trainable_variables, gradients, strict=True):
        variable_name = getattr(variable, "path", variable.name)
        if gradient is None:
            missing_gradient_count += 1
            gradient_rows.append(
                {
                    "variable": variable_name,
                    "gradient_present": False,
                    "finite": False,
                    "maximum_absolute_gradient": None,
                }
            )
            continue
        values = tf.convert_to_tensor(gradient)
        finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
        maximum_absolute = float(tf.reduce_max(tf.abs(values)).numpy())
        nonfinite_gradient_count += int(not finite)
        nonzero_gradient_count += int(maximum_absolute > 0)
        gradient_rows.append(
            {
                "variable": variable_name,
                "gradient_present": True,
                "finite": finite,
                "maximum_absolute_gradient": maximum_absolute,
            }
        )
    gradients_passed = (
        missing_gradient_count == 0
        and nonfinite_gradient_count == 0
        and nonzero_gradient_count == len(model.trainable_variables)
    )
    parameter_count = int(model.count_params())
    parameter_count_passed = parameter_count == spec.expected_parameter_count
    required_gates = {
        "output_shape": output_shape_passed,
        "finite_forward": finite_forward,
        "finite_nonzero_backward_for_every_variable": gradients_passed,
        "exact_table_iii_parameter_count": parameter_count_passed,
    }
    layer_rows = [
        {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "input_shape": (
                [_shape(item) for item in layer.input]
                if isinstance(layer.input, (list, tuple))
                else _shape(layer.input)
            ),
            "output_shape": _shape(layer.output),
            "parameter_count": int(layer.count_params()),
        }
        for layer in model.layers
    ]
    return {
        "schema_version": 1,
        "status": "pass" if all(required_gates.values()) else "failed",
        "model_name": spec.name,
        "author_implementation_confirmed": spec.author_implementation_confirmed,
        "evidence": spec.evidence,
        "contract": {
            "input_shape": [None, spec.input_length, 1],
            "output_shape": [None, spec.output_units],
            "lstm_units": list(spec.units),
            "return_sequences": list(spec.return_sequences),
            "dropout_placement": spec.dropout_placement,
            "expected_parameter_count": spec.expected_parameter_count,
        },
        "actual_output_shape": output_shape,
        "actual_parameter_count": parameter_count,
        "parameter_count_difference": parameter_count - spec.expected_parameter_count,
        "gradient_audit": {
            "loss": float(loss.numpy()),
            "trainable_variable_count": len(model.trainable_variables),
            "missing_gradient_count": missing_gradient_count,
            "nonfinite_gradient_count": nonfinite_gradient_count,
            "nonzero_gradient_count": nonzero_gradient_count,
            "variables": gradient_rows,
        },
        "required_gates": required_gates,
        "required_gates_passed": all(required_gates.values()),
        "layers": layer_rows,
    }


__all__ = [
    "audit_lstm_model",
    "build_lstm_model",
    "configure_determinism",
    "require_tensorflow",
]
