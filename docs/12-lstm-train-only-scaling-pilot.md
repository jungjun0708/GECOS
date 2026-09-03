# LSTM Train-only 셀별 Min-Max scaling pilot

## 1. 목적과 현재 상태

중앙 900셀 LSTM·UPC pipeline smoke는 연결 계약을 통과했지만, raw traffic을 5
epoch만 학습한 LSTM의 Validation MAE가 `236.2853`으로 Persistence `32.2167`보다
크게 나빴다. 이 단계에서는 모델 구조나 표본을 동시에 바꾸지 않고 **입력과 target의
scaling 하나만** 바꿔 과소학습이 실질적으로 줄어드는지 확인한다.

이 문서의 실험 계약과 판정 기준은 Colab 결과를 보기 전에 고정한다. 이 pilot은
Validation을 이용한 다음 본 학습 설정 선택 단계이므로 Test는 입력 bundle에도 넣지
않고 평가하지 않는다. 논문 Table II와 직접 비교할 수 있는 성능 결과도 아니다.

## 2. 왜 이 실험이 다음 순서인가

raw smoke에서는 세 모델 모두 Train MAE가 감소했지만 예측 평균이 실제 traffic
규모에 충분히 도달하지 못했다. 셀별 traffic 범위가 약 `3.55`부터 `7,939.19`까지
크게 다른 상태에서 하나의 LSTM이 raw MAE를 최적화한 영향인지 먼저 분리해야 한다.

바로 전체 시점·여러 seed 학습을 실행하면 다음 문제가 생긴다.

1. scaling 문제와 epoch 부족 문제를 구분하지 못한 채 Colab 비용을 늘릴 수 있다.
2. Validation을 scaler 적합에 포함하는 시간 누수를 뒤늦게 발견할 수 있다.
3. 결과를 본 뒤 scaler나 clipping을 바꾸는 선택 편향이 생길 수 있다.
4. Test를 반복 확인해 최종 평가 구간이 사실상 Validation처럼 사용될 수 있다.

따라서 같은 선택 표본에서 단일 후보를 먼저 검사하고, 미리 정한 문턱으로 다음
작업을 결정한다.

## 3. 고정한 비교 계약

raw smoke와 동일하게 유지하는 항목은 다음과 같다.

| 항목 | 고정값 |
|---|---|
| 공간 | `central-900-approximate` 전체 900셀 |
| target | Train/Validation 각각 셀당 같은 64개 |
| UPC | `train_only`, cluster 0은 611셀, cluster 1은 289셀 |
| 구조 | `LSTM(64) → LSTM(128) → LSTM(64) → Dense(1)` |
| parameter | 165,185 |
| seed | 42, 각 모델 생성 직전 재설정 |
| optimizer / learning rate | Adam / 0.001 |
| loss / batch | MAE / 512 |
| dropout / shuffle | 0.05 / `false` |
| epoch | 고정 5, early stopping 없음 |
| 모델 수 | UPC off 1개, UPC on cluster별 2개 |

유일하게 바꾸는 요인은 모델 입력과 target의 scaling이다. raw smoke 결과를 덮어쓰지
않고 `lstm_scaling_pilot` 전용 설정·입력·결과 경로를 사용한다.

## 4. Train-only 셀별 Min-Max 계약

각 중앙 셀 `c`에서 11월 1~20일 Train 전체 2,880시점만 사용해 다음 값을 적합한다.

```text
minimum[c] = min(traffic[c, 0:2880])
range[c]   = max(traffic[c, 0:2880]) - minimum[c]
scaled     = (value - minimum[c]) / range[c]
restored   = scaled * range[c] + minimum[c]
```

- dtype은 `float32`다.
- 선택된 64개 Train target만이 아니라 Train의 전체 2,880시점으로 적합한다.
- Validation과 Test 값은 scaler 적합에 사용하지 않는다.
- input window와 target에 같은 셀별 parameter를 사용한다.
- 변환값과 역변환 예측을 `[0, 1]` 또는 0 이상으로 clipping하지 않는다.
- range가 0인 셀이 하나라도 있으면 중단한다.
- raw 단위 역변환 오차의 최대 허용값은 `0.001`이다.
- 모든 MAE·MAPE·WAPE는 예측을 셀별로 역변환한 뒤 원래 traffic 단위에서 계산한다.

Validation 값이 Train 범위를 벗어나면 0 미만 또는 1 초과가 될 수 있다. 이는 시간상
새로운 범위를 보존하는 정상 동작이며, 이를 자르면 모델뿐 아니라 정답도 바뀌므로
clipping하지 않는다.

## 5. 로컬 입력 검증 결과

Colab 전 사전 점검에서 중앙 traffic과 기존 raw smoke 입력·결과의 SHA-256을 다시
검증했다. 실제 scaler 진단은 다음과 같다.

| 검사 | 결과 |
|---|---:|
| scaler 적합 index | `[0, 2880)` |
| zero-range 셀 | 0 |
| 셀별 Train minimum의 min / median / max | 0 / 65.4349 / 277.6404 |
| 셀별 Train range의 min / median / max | 3.5527 / 646.2198 / 7,939.1909 |
| Train scaled input·target 범위 | 정확히 `[0, 1]` |
| Validation input 0 미만 / 1 초과 | 665 / 185개 |
| Validation target 0 미만 / 1 초과 | 85 / 22개 |
| 최대 float32 역변환 오차 | 0.00048828125, 통과 |
| 새 bundle의 Test 배열 | 0개 |
| 입력 NPZ 크기 | 약 4.31MiB |
| 로컬 입력 준비 최대 RSS | 약 72.2MiB |

물리 RAM 32GB 전체를 작업 전용으로 가정하지 않는다. 로컬에서는 memory map과 선택
배열만 사용해 추가 peak RSS를 약 72MiB로 제한하고, TensorFlow 학습은 Colab T4로
분리한다. 따라서 Codex, WSL, 브라우저와 다른 앱이 동시에 메모리와 CPU를 사용하는
상황에서도 로컬 부담이 작다.

## 6. 결과 전 판정 규칙

주 판정값은 `lstm_scaled_upc_off`의 Validation `all_targets` micro MAE 하나다. raw
기준값 `236.285288592`에서 최소 20% 개선되는지를 본다.

```text
20% 개선 문턱 = 236.285288592 × 0.8 = 189.0282308736
```

| scaled Validation MAE | 판정 | 다음 작업 |
|---|---|---|
| `<= 189.0282308736` | 실질 개선 | 본 학습 scaling 후보로 채택 |
| raw보다 작지만 문턱 초과 | 불충분한 증거 | 전체 학습 전 epoch 통제 실험 |
| raw 이상 | 개선 없음 | 후보 기각, 최적화·출력 계약 재검토 |

Persistence보다 좋아지는지, UPC on/off 중 어느 쪽이 좋은지, Test 수치는 이 scaling
후보 판정이나 pipeline 합격 조건으로 사용하지 않는다. pipeline 합격은 구조, 유한값,
세 모델의 Train MAE 감소, 900셀 exact 재결합, 역변환과 Test 봉인으로만 정한다.

## 7. 결정성 확인 계획

clean Git commit에서 로컬 입력을 다시 만든 뒤 같은 Colab T4 세션에서 최종 코드를
두 번 실행한다. 실행 시각과 소요 시간이 있는 manifest는 달라도 정상이다. 다음 네
핵심 산출물의 SHA-256이 두 실행에서 같아야 한다.

- `architecture_report.json`
- `evaluation_report.json`
- `predictions.npz`
- `per_cell_metrics.csv`

## 8. 실행 방법

```bash
.venv/bin/python -m scripts.prepare_lstm_scaling_pilot \
  --config configs/lstm_scaling_pilot_milan_nov2013.json

.venv/bin/python -m unittest \
  tests.test_lstm_scaling_contract \
  tests.test_lstm_scaling_pipeline -v
```

Colab bundle에는 필요한 Python 코드, 두 LSTM 설정, 기존 공통 설정, 입력 NPZ와
manifest만 넣는다. 원시 데이터, 전체 전처리 배열, 기존 Test 배열, 논문 PDF와
checkpoint는 올리지 않는다.

## 9. Colab 결과

결과 전 사전 등록 상태다. clean commit 번들의 T4 반복 실행이 끝난 뒤 이 절에는
환경, 세 모델 학습 gate, raw 단위 Validation 지표, 사전 판정 결과와 두 실행의
핵심 SHA-256만 추가한다. 위 계약과 문턱은 결과에 맞춰 바꾸지 않는다.
