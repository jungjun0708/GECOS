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

### 9.1 실행 환경과 gate

사전 등록 계약을 commit `30fce1ffd32225307f2b307c91b36f5fe61506ec`에 고정한
뒤 clean 상태에서 입력을 다시 만들었다. 같은 Colab T4 세션에서 최종 코드를 두 번
실행했다.

| 항목 | 값 |
|---|---|
| GPU | Tesla T4, 15,360MiB |
| Python | 3.13.15 |
| NumPy | 2.1.3 |
| TensorFlow / Keras | 2.20.0 / 3.13.2 |
| 입력 NPZ SHA-256 | `c575bb60316b5bf8b4b6b2ed516f305a6d760118f189071edb90814fcab9c197` |
| 입력 manifest SHA-256 | `3bddb9aaf3b33fc559a6f4775bf0fdf9c6478fd75d333a9eabb8bbc68fc7d54a` |
| Colab bundle SHA-256 | `308c5a7a2d76be418dfa8728f43a3014abb6267f3f2482eccdbecd23b269c067` |
| 1차 / 2차 전체 실행 시간 | 약 35.14초 / 31.50초 |
| 1차 / 2차 peak RSS | 약 1.41GiB / 1.46GiB |

두 실행 모두 다음 gate를 통과했다.

- 실제 parameter 수 `165,185`, 출력 shape와 gradient 감사 통과
- Train-only scaler 적합과 최대 역변환 오차 통과
- Test 배열 0개, Test 평가 0회
- UPC off 1개와 cluster별 UPC on 2개가 모두 정확히 5 epoch 완료
- 세 모델 모두 prefit보다 final Train MAE 감소
- Train/Validation에서 cluster 예측을 900셀 순서로 정확히 재결합
- raw 단위 예측과 모든 평가 지표가 유한함

scaled-domain 학습 진단은 다음과 같다. 이 값은 셀별로 정규화된 단위라 raw smoke의
Train MAE와 직접 비교하지 않는다.

| 모델 | 셀 | prefit Train MAE | final Train MAE | 감소율 |
|---|---:|---:|---:|---:|
| scaled UPC off | 900 | 0.247221 | 0.037702 | 84.75% |
| scaled UPC on, cluster 0 | 611 | 0.249636 | 0.037991 | 84.78% |
| scaled UPC on, cluster 1 | 289 | 0.242114 | 0.047587 | 80.35% |

### 9.2 원래 traffic 단위 Validation 결과

아래 값은 모든 예측을 셀별로 역변환한 뒤 계산한 `all_targets` micro average다.

| 모델 | MAE | MAPE ratio | MAPE percent | WAPE |
|---|---:|---:|---:|---:|
| Persistence | 32.2167 | 0.131736 | 13.1736% | 0.116522 |
| raw LSTM UPC off | 236.2853 | 0.722282 | 72.2282% | 0.854600 |
| **scaled LSTM UPC off** | **30.4308** | **0.129584** | **12.9584%** | **0.110062** |
| scaled LSTM UPC on | 33.4268 | 0.134007 | 13.4007% | 0.120899 |

주 판정값인 UPC off MAE는 raw `236.285288592`에서 scaled `30.4307535912`로
`87.1212%` 감소했다. 사전 등록한 20% 개선 문턱 `189.0282308736`보다 충분히
낮으므로 판정은 다음과 같다.

```text
category = material_improvement
outcome  = adopt_as_full_training_scaling_candidate
```

따라서 **Train-only 셀별 Min-Max를 본 학습의 scaling 후보로 채택**한다. 이 결과는
raw-scale 5 epoch 과소학습의 주된 원인이 수치 스케일에 있었다는 강한 진단 증거다.
다만 64개 Validation target과 seed 하나의 pilot이므로, scaling이 논문 저자의 실제
설정이었다거나 최종 Test 성능이 확정됐다고 해석하지 않는다.

scaled UPC off가 이 Validation subset에서는 Persistence보다 MAE가 약 `5.54%`
낮았고, scaled UPC on은 off보다 약 `9.85%` 높았다. 두 비교는 사전 scaling 선택
gate가 아니며 UPC 효과에 대한 결론으로 사용하지 않는다. UPC 비교는 전체 Train
target, early stopping과 여러 seed를 사용하는 본 실험에서 수행한다.

선형 출력과 무클리핑 계약에 따라 raw 단위 Validation 예측 중 UPC off 2개, UPC on
41개가 0보다 조금 작았다. 최솟값은 각각 약 `-0.00659`, `-0.03244`다. 이를 0으로
자르지 않고 그대로 평가했으며 출력 activation 문제는 계속 `GAP-LSTM-02`로 남긴다.

### 9.3 반복 결정성

실행 시각과 소요 시간이 들어가는 `manifest.json`은 두 실행에서 달랐다. 다음 네
핵심 산출물은 바이트 단위로 같았다.

| 산출물 | 두 실행의 동일 SHA-256 |
|---|---|
| `architecture_report.json` | `cae46c163e652a133c4ee5e1b22fae39d1635d97201a01d2753a16c7d86698f9` |
| `evaluation_report.json` | `96069a20813e1abd5e65648bad958cf8ea6226455bd410ae5d3196d791cc9195` |
| `predictions.npz` | `2c9e46fe2afd1eb161d919cc197848432be98a62856bbb9bed25bcae9daab579` |
| `per_cell_metrics.csv` | `6f36f8cd17cf92f9e2f0a32d0aa4e8332a82de2dc38c1b58c33d9d1c32c6a015` |

## 10. 결론과 다음 단계

scaling 후보 판정이 명확히 통과했으므로 epoch 수만 바꾸는 추가 pilot은 필요하지
않다. 다음 작업은 이 셀별 Train-only scaler를 고정하고 중앙 900셀의 전체 target을
사용하는 본 학습 계약을 구현하는 것이다.

본 학습에서는 Validation MAE early stopping, 최적 가중치 복원, seed `42, 43, 44`를
사용한다. Test는 모델·epoch·scaling 선택이 모두 끝난 뒤 seed별로 한 번만 평가한다.
먼저 LSTM UPC off/on 전체 비교를 완성해 데이터와 cluster별 학습 경로를 검증한 뒤,
같은 split·scaler·seed 계약으로 RCTL 비교를 확장한다. 이 pilot의 raw 결과와 scaled
결과는 모두 진단 기록으로 보존하며 서로 덮어쓰지 않는다.
