#!/usr/bin/env python3
"""논문 해석형과 공개 코드형 RCTL 모델 및 구조 감사 도구."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.rctl_contract import ArchitectureSpec, RctlContractError


def require_tensorflow() -> Any:
    """로컬 데이터 준비에는 TensorFlow를 강제하지 않고 모델 실행 시에만 불러온다."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RctlContractError(
            "RCTL 모델 실행에는 TensorFlow가 필요합니다. Colab T4 환경에서 실행하세요."
        ) from exc
    return tf


def configure_determinism(seed: int) -> dict[str, Any]:
    """지원되는 범위에서 Python/NumPy/TensorFlow 난수를 고정한다."""

    tf = require_tensorflow()
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    deterministic_ops = False
    deterministic_error: str | None = None
    try:
        tf.config.experimental.enable_op_determinism()
        deterministic_ops = True
    except (AttributeError, RuntimeError) as exc:
        deterministic_error = str(exc)
    return {
        "seed": seed,
        "tensorflow_op_determinism_enabled": deterministic_ops,
        "tensorflow_op_determinism_error": deterministic_error,
    }


def build_rctl_model(
    *,
    steps: int,
    spec: ArchitectureSpec,
    dropout: float,
    learning_rate: float,
    compile_model: bool = True,
) -> Any:
    """사전 등록된 variant에 따라 RCTL 계산 그래프를 만든다."""

    tf = require_tensorflow()
    keras = tf.keras
    if steps < 1:
        raise RctlContractError("RCTL steps는 1 이상이어야 합니다.")
    if not 0 <= dropout < 1:
        raise RctlContractError("dropout은 0 이상 1 미만이어야 합니다.")

    inputs = keras.Input(shape=(steps, 1), dtype=tf.float32, name="traffic_window")
    x = inputs
    outputs_store: list[Any] = []
    route_map = spec.rcc2_route_map

    for block_index, (filters, dilation) in enumerate(
        zip(spec.channels, spec.dilations, strict=True)
    ):
        prefix = f"b{block_index}"
        block_input = x
        fx = keras.layers.Conv1D(
            filters,
            spec.kernel_size,
            padding="causal",
            dilation_rate=dilation,
            name=f"{prefix}_causal_conv1",
        )(block_input)
        fx = keras.layers.BatchNormalization(name=f"{prefix}_bn1")(fx)
        fx = keras.layers.ReLU(name=f"{prefix}_relu1")(fx)
        fx = keras.layers.Dropout(dropout, name=f"{prefix}_dropout1")(fx)
        fx = keras.layers.Conv1D(
            filters,
            spec.kernel_size,
            padding="causal",
            dilation_rate=dilation,
            name=f"{prefix}_causal_conv2",
        )(fx)
        fx = keras.layers.BatchNormalization(name=f"{prefix}_bn2")(fx)
        fx = keras.layers.ReLU(name=f"{prefix}_relu2")(fx)
        fx = keras.layers.Dropout(dropout, name=f"{prefix}_dropout2")(fx)

        tcn_shortcut = keras.layers.Conv1D(
            filters, 1, padding="causal", name=f"{prefix}_tcn_shortcut"
        )(block_input)
        if spec.tcn_shortcut_merge == "concatenate":
            merged = keras.layers.Concatenate(name=f"{prefix}_tcn_concat")(
                [tcn_shortcut, fx]
            )
        elif spec.tcn_shortcut_merge == "add":
            merged = keras.layers.Add(name=f"{prefix}_tcn_add")([tcn_shortcut, fx])
        else:
            raise RctlContractError(
                f"지원하지 않는 TCN shortcut merge입니다: {spec.tcn_shortcut_merge}"
            )
        block_output = keras.layers.LSTM(
            filters, return_sequences=True, name=f"{prefix}_lstm"
        )(merged)

        rcc1_shortcut = keras.layers.Conv1D(
            filters, 1, padding="same", name=f"{prefix}_rcc1_projection"
        )(block_input)
        block_output = keras.layers.Add(name=f"{prefix}_rcc1_add")(
            [rcc1_shortcut, block_output]
        )
        stored_output = keras.layers.Conv1D(
            filters, 1, padding="same", name=f"{prefix}_rcc2_projection"
        )(block_output)
        outputs_store.append(stored_output)

        if block_index in route_map:
            source_index = route_map[block_index]
            x = keras.layers.Add(name=f"{prefix}_rcc2_add")(
                [outputs_store[source_index], block_output]
            )
        else:
            x = block_output

    if spec.final_input_shortcut:
        final_shortcut = keras.layers.Conv1D(
            spec.channels[-1],
            1,
            padding="same",
            name="final_input_projection",
        )(inputs)
        x = keras.layers.Add(name="final_input_add")([x, final_shortcut])
    x = keras.layers.Flatten(name="flatten")(x)
    outputs = keras.layers.Dense(1, name="forecast")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"rctl_{spec.name}")
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mae",
        )
    return model


def _shape(tensor: Any) -> list[int | None]:
    return [None if value is None else int(value) for value in tensor.shape]


def _input_shapes(layer: Any) -> list[list[int | None]]:
    inputs = layer.input if isinstance(layer.input, (list, tuple)) else [layer.input]
    return [_shape(value) for value in inputs]


def _layer_shape_contract(model: Any) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    residual_shapes_pass = True
    for layer in model.layers:
        class_name = layer.__class__.__name__
        input_shapes = _input_shapes(layer)
        output_shape = _shape(layer.output)
        row = {
            "name": layer.name,
            "type": class_name,
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "parameter_count": int(layer.count_params()),
        }
        if class_name == "Add":
            shapes_equal = all(shape == input_shapes[0] for shape in input_shapes[1:])
            row["add_inputs_exactly_equal"] = shapes_equal
            residual_shapes_pass = residual_shapes_pass and shapes_equal
        elif class_name == "Concatenate":
            non_channel_shapes_equal = all(
                shape[:-1] == input_shapes[0][:-1] for shape in input_shapes[1:]
            )
            row["non_channel_dimensions_equal"] = non_channel_shapes_equal
            residual_shapes_pass = residual_shapes_pass and non_channel_shapes_equal
        rows.append(row)
    return rows, residual_shapes_pass


def _causality_audit(model: Any, *, steps: int, seed: int) -> dict[str, Any]:
    tf = require_tensorflow()
    causal_layers = [
        layer for layer in model.layers if "_causal_conv" in layer.name
    ]
    if not causal_layers:
        raise RctlContractError("감사할 causal convolution layer가 없습니다.")
    probe = tf.keras.Model(
        inputs=model.input,
        outputs=[layer.output for layer in causal_layers],
        name=f"{model.name}_causality_probe",
    )
    rng = np.random.default_rng(seed)
    original = rng.normal(size=(2, steps, 1)).astype(np.float32)
    changed = original.copy()
    cut_index = max(0, steps // 2 - 1)
    changed[:, cut_index + 1 :, :] += rng.normal(
        loc=100.0,
        scale=10.0,
        size=changed[:, cut_index + 1 :, :].shape,
    ).astype(np.float32)
    original_outputs = probe(original, training=False)
    changed_outputs = probe(changed, training=False)
    if not isinstance(original_outputs, (list, tuple)):
        original_outputs = [original_outputs]
        changed_outputs = [changed_outputs]
    layer_results: list[dict[str, Any]] = []
    tolerance = 1e-5
    for layer, left, right in zip(
        causal_layers, original_outputs, changed_outputs, strict=True
    ):
        maximum_delta = float(
            tf.reduce_max(tf.abs(left[:, : cut_index + 1] - right[:, : cut_index + 1])).numpy()
        )
        layer_results.append(
            {
                "layer": layer.name,
                "maximum_earlier_output_delta": maximum_delta,
                "passed": maximum_delta <= tolerance,
            }
        )
    return {
        "passed": all(item["passed"] for item in layer_results),
        "cut_index": cut_index,
        "changed_input_indices": list(range(cut_index + 1, steps)),
        "compared_output_indices": list(range(cut_index + 1)),
        "absolute_tolerance": tolerance,
        "layers": layer_results,
    }


def _gradient_audit(model: Any, *, steps: int, seed: int) -> dict[str, Any]:
    tf = require_tensorflow()
    rng = np.random.default_rng(seed + 1)
    x = tf.convert_to_tensor(rng.normal(size=(4, steps, 1)).astype(np.float32))
    y = tf.convert_to_tensor(rng.normal(size=(4, 1)).astype(np.float32))
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.reduce_mean(tf.abs(y - predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    rows: list[dict[str, Any]] = []
    finite = True
    nonzero_count = 0
    missing_count = 0
    for variable, gradient in zip(model.trainable_variables, gradients, strict=True):
        variable_name = getattr(variable, "path", variable.name)
        if gradient is None:
            rows.append(
                {"variable": variable_name, "gradient_present": False, "finite": False}
            )
            finite = False
            missing_count += 1
            continue
        values = tf.convert_to_tensor(gradient)
        gradient_finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
        maximum_absolute = float(tf.reduce_max(tf.abs(values)).numpy())
        if maximum_absolute > 0:
            nonzero_count += 1
        finite = finite and gradient_finite
        rows.append(
            {
                "variable": variable_name,
                "gradient_present": True,
                "finite": gradient_finite,
                "maximum_absolute_gradient": maximum_absolute,
            }
        )
    return {
        "passed": finite and missing_count == 0 and nonzero_count > 0,
        "loss": float(loss.numpy()),
        "trainable_variable_count": len(model.trainable_variables),
        "missing_gradient_count": missing_count,
        "nonzero_gradient_variable_count": nonzero_count,
        "variables": rows,
    }


def audit_rctl_model(
    model: Any,
    *,
    spec: ArchitectureSpec,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """shape, residual, causal, forward/backward와 parameter 수를 함께 감사한다."""

    tf = require_tensorflow()
    layer_contract, residual_shapes_pass = _layer_shape_contract(model)
    rng = np.random.default_rng(seed + 2)
    sample = rng.normal(size=(3, steps, 1)).astype(np.float32)
    predictions = model(sample, training=False)
    output_shape = list(predictions.shape)
    output_shape_pass = output_shape == [3, 1]
    forward_finite = bool(tf.reduce_all(tf.math.is_finite(predictions)).numpy())
    causality = _causality_audit(model, steps=steps, seed=seed)
    gradients = _gradient_audit(model, steps=steps, seed=seed)
    parameter_count = int(model.count_params())
    expected_count = spec.expected_parameter_count
    if expected_count is None:
        parameter_comparison = {
            "status": "not_specified_for_public_code",
            "expected": None,
            "actual": parameter_count,
            "difference": None,
            "included_in_smoke_gate": False,
        }
    else:
        parameter_comparison = {
            "status": "match" if parameter_count == expected_count else "diagnostic_mismatch",
            "expected": expected_count,
            "actual": parameter_count,
            "difference": parameter_count - expected_count,
            "included_in_smoke_gate": False,
        }
    required_gates = {
        "output_shape": output_shape_pass,
        "residual_shapes_without_broadcast": residual_shapes_pass,
        "finite_forward": forward_finite,
        "causal_convolutions": causality["passed"],
        "finite_nonzero_backward": gradients["passed"],
    }
    return {
        "variant": spec.name,
        "evidence": spec.evidence,
        "contract": {
            "input_shape": [None, steps, 1],
            "output_shape": [None, 1],
            "channels": list(spec.channels),
            "kernel_size": spec.kernel_size,
            "dilations": list(spec.dilations),
            "tcn_shortcut_merge": spec.tcn_shortcut_merge,
            "rcc1": spec.rcc1,
            "rcc2_routes_destination_to_source": {
                str(destination): source for destination, source in spec.rcc2_routes
            },
            "final_input_shortcut": spec.final_input_shortcut,
        },
        "actual_forward_output_shape": output_shape,
        "required_gates": required_gates,
        "required_gates_passed": all(required_gates.values()),
        "parameter_count": parameter_comparison,
        "causality_audit": causality,
        "gradient_audit": gradients,
        "layers": layer_contract,
    }
