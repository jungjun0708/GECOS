# 중앙 900셀 LSTM·UPC Colab T4 pipeline smoke

## 1. 결론

중앙 900셀 전체를 유지하되 각 시간 분할에서 셀당 64개 target을 결정적으로
선택하여, LSTM 기준선의 UPC 적용 전·후 데이터 흐름을 Colab T4에서 끝까지
검증했다.

- 논문 Table III의 LSTM parameter 수 `165,185`를 정확히 만드는
  `LSTM(64) → LSTM(128) → LSTM(64) → Dense(1)` 후보를 구현했다.
- 이 구조는 **parameter 표를 만족하는 재구성 후보**이며 원저자가 사용한 정확한
  구조로 확인된 것은 아니다. 코드와 결과에서는
  `paper_parameter_reconstruction`으로만 부른다.
- UPC 미적용 모델 1개와 `train_only` 최종 cluster별 모델 2개를 같은 seed와
  설정으로 각각 5 epoch 학습했다.
- 구조, 유한값, 세 모델의 Train MAE 감소, 900셀 예측 재결합이 모두 통과했다.
- 최종 코드를 같은 T4에서 두 번 실행했을 때 구조 보고서, 평가 보고서, 예측 NPZ,
  셀별 CSV의 SHA-256이 모두 같았다.
- Test micro MAE는 Persistence `28.5222`, UPC 미적용 LSTM `234.9030`, UPC 적용
  LSTM `246.1071`이었다. UPC 적용 모델이 미적용 모델보다 약 `4.77%` 나빴다.

마지막 수치는 UPC가 해롭다는 결론이 아니다. 셀당 64개 target, seed 1개, raw
traffic, 고정 5 epoch를 사용한 **pipeline smoke**에서 LSTM이 심하게 과소학습했다는
진단이다. 논문 Table II와 직접 비교하거나 모델 우열의 근거로 사용하지 않는다.

## 2. 이 smoke를 먼저 수행한 이유

UPC 적용 실험은 cluster마다 별도 모델을 학습한 뒤 예측을 원래 900셀 순서로
되돌려야 한다. 전체 시점과 3개 seed를 바로 실행하면 다음 오류를 비싼 학습이 끝난
뒤에야 발견할 수 있다.

1. 논문 LSTM parameter 수를 잘못 해석함
2. window와 target이 한 시점 어긋남
3. Validation/Test 경계에서 미래값이 입력에 들어감
4. 중앙 900셀 membership 순서와 예측 순서가 달라짐
5. 두 cluster 중 일부 셀이 누락되거나 중복됨
6. 로컬 준비 과정이 Codex와 다른 앱이 사용하는 메모리까지 압박함
7. Colab 코드와 로컬 입력의 commit 또는 checksum이 맞지 않음

이 단계는 성능을 고르는 실험이 아니라 위 연결 계약을 작은 비용으로 검증하는
실험이다. 따라서 Persistence보다 좋아야 한다는 조건은 의도적으로 합격 기준에서
제외했다.

## 3. LSTM 구조를 재구성한 방법

Keras LSTM의 parameter 수는 입력 feature 수 `d`, unit 수 `u`에 대해 다음과 같다.

```text
LSTM parameters = 4 × u × (d + u + 1)
```

이 식으로 다음 stacked-LSTM을 구성하면 논문 Table III의 `165,185`와 정확히
일치한다.

| layer | 입력 feature | unit | parameter |
|---|---:|---:|---:|
| LSTM 1, sequence 반환 | 1 | 64 | 16,896 |
| LSTM 2, sequence 반환 | 64 | 128 | 98,816 |
| LSTM 3, 마지막 출력만 반환 | 128 | 64 | 49,408 |
| 선형 Dense | 64 | 1 | 65 |
| 합계 |  |  | **165,185** |

각 LSTM 뒤에 dropout `0.05`를 적용한다. Dropout layer는 학습 parameter를 추가하지
않는다. 구조 감사에서는 실제 `model.count_params()`도 `165,185`였고 출력 shape
`(batch, 1)`, 유한한 forward, 11개 trainable variable 모두에 도달하는 유한하고
0이 아닌 gradient를 확인했다.

gradient 검사는 학습 loss를 바꾸는 과정이 아니다. 작은 짝수 표본에서 MAE의 양·음
부호가 정확히 상쇄되어 최종 bias gradient가 0이 되는 거짓 실패가 실제로 발견됐다.
따라서 계산 그래프 연결성 probe에는 MSE를 사용하고, 실제 모델 학습에는 계속 MAE를
사용한다. 두 목적은 `architecture_report.json`에 별도로 기록한다.

논문은 LSTM의 layer별 unit, sequence 반환 방식, dropout 위치와 출력 activation을
공개하지 않았다. 같은 parameter 수만으로 저자 구조를 유일하게 확정할 수 없으므로
이 결과는 `GAP-LSTM-01`로 추적한다.

## 4. 데이터와 선택 계약

### 4.1 공간과 UPC membership

- 공간 범위: `central-900-approximate` 전 셀
- 셀 순서: `central_900.csv`와 보호된 membership CSV 순서
- UPC 프로토콜: 누수 없고 순서 민감도 검사를 통과한 `train_only`만 허용
- cluster 0: 611셀
- cluster 1: 289셀

입력 준비 시 UPC 학습 정책을 다시 생성·검증하고, 보호된 membership checksum이
바뀌지 않았는지 확인한다. `algorithm1_full_month`와 Fig. 4 probe는 이 학습기에
전달할 수 없다.

### 4.2 시간 표본

공통 예측 계약과 같이 과거 8개 10분 값을 사용해 바로 다음 값을 예측한다.

```text
X[cell, t-8:t] -> y[cell, t]
```

각 분할의 모든 target 중 첫 target과 마지막 target을 포함해 64개를 등간격으로
선택한다. 모든 셀에 같은 target index를 사용한다.

| 분할 | 전체 target/셀 | smoke target/셀 | 전체 smoke 표본 | 선택 index 범위 |
|---|---:|---:|---:|---:|
| Train | 2,872 | 64 | 57,600 | 8~2,879 |
| Validation | 720 | 64 | 57,600 | 2,880~3,599 |
| Test | 720 | 64 | 57,600 | 3,600~4,319 |

UPC 적용 학습 표본은 cluster 0이 `611 × 64 = 39,104`, cluster 1이
`289 × 64 = 18,496`이다. 두 집합의 합은 UPC 미적용 모델의 57,600개와 같아 공간
범위가 달라지지 않는다.

전처리에서 0으로 채운 target을 포함하는 `all_targets`와, 원본 행 누락 또는 모든
Internet 값이 공란인 target을 제외하는 `observed_targets_only`를 모두 평가한다.
입력 window의 결측 표시는 예측에서 제거하지 않고 진단 개수로 남긴다.

## 5. 학습·평가 계약

| 항목 | 값 |
|---|---|
| seed | 42, 각 모델 생성 직전에 동일하게 재설정 |
| optimizer / learning rate | Adam / 0.001 |
| loss / batch | MAE / 512 |
| dropout / shuffle | 0.05 / `false` |
| 모델 입력·target scaling | 없음, `float32` raw traffic |
| epoch | 고정 5, early stopping 없음 |
| Validation | 고정 epoch 동안 감시만 함 |
| Test | 학습 종료 후 한 번 평가 |
| checkpoint | 저장하지 않음 |

합격 조건은 다음 여섯 가지다.

1. 실제 parameter 수가 정확히 `165,185`
2. forward, backward, history와 예측이 모두 유한함
3. 세 모델 모두 prefit보다 final Train MAE가 감소함
4. 세 모델 모두 정확히 5 epoch를 완료함
5. cluster 예측을 900셀 원래 순서로 누락·중복 없이 정확히 재결합함
6. MAE·MAPE·WAPE 결과가 모두 유한함

선형 Dense 출력은 음수 예측을 만들 수 있다. 이를 결과 확인 후 0으로 잘라 성능을
바꾸지 않고 원시 예측 그대로 평가하며, 분할별 최솟값과 음수 개수를 남긴다. 최종
실행에서 cluster 1은 Train 1개, Validation 4개의 작은 음수 예측을 만들었고 Test에는
없었다. 출력 activation과 후처리가 공개되지 않은 문제는 `GAP-LSTM-02`로 추적한다.

MAE·MAPE·WAPE 계산은 기존 기준선과 같은 함수를 사용한다. 다만 Persistence에는
기존의 비음수 계약을 유지하고, LSTM 호출에서만 음수 prediction을 명시적으로
허용한다. MAPE는 `y > 0`인 eligible target만 사용하며 ratio와 percent를 함께
출력한다.

## 6. 실행 결과

### 6.1 구조와 pipeline gate

| 검사 | 결과 |
|---|---|
| 실제 parameter 수 | 165,185, 통과 |
| 출력 shape | `(4, 1)` probe, 통과 |
| 유한한 forward | 통과 |
| gradient 연결 | 11 / 11 trainable variable, 통과 |
| 세 모델 학습 | 3 / 3, 통과 |
| 세 모델 Train MAE 감소 | 통과 |
| 900셀 재결합 | Train/Validation/Test 모두 exact, 통과 |
| metric 유한성 | 통과 |

학습 진단은 다음과 같다.

| 모델 | 셀 | prefit Train MAE | final Train MAE | 감소율 |
|---|---:|---:|---:|---:|
| UPC off | 900 | 278.5584 | 236.7829 | 14.9970% |
| UPC on, cluster 0 | 611 | 298.6522 | 265.4328 | 11.1231% |
| UPC on, cluster 1 | 289 | 236.0765 | 214.1233 | 9.2992% |

### 6.2 Validation/Test 진단 지표

아래 값은 주 target 정책인 `all_targets`의 micro average다.

| 분할 | 모델 | MAE | MAPE ratio | MAPE percent | WAPE |
|---|---|---:|---:|---:|---:|
| Validation | Persistence | 32.2167 | 0.131736 | 13.1736% | 0.116522 |
| Validation | LSTM UPC off | 236.2853 | 0.722282 | 72.2282% | 0.854600 |
| Validation | LSTM UPC on | 247.7777 | 0.731690 | 73.1690% | 0.896166 |
| Test | Persistence | 28.5222 | 0.126767 | 12.6767% | 0.103867 |
| Test | LSTM UPC off | 234.9030 | 0.703412 | 70.3412% | 0.855427 |
| Test | LSTM UPC on | 246.1071 | 0.718142 | 71.8142% | 0.896228 |

Test에서 UPC on의 MAE는 off보다 `11.2040` 크고 상대적으로 약 `4.77%` 나빴다.
그러나 세 LSTM 모두 Persistence보다 훨씬 나쁘며 출력 평균도 실제 traffic 규모보다
작다. 이 상태에서 UPC 효과를 해석하면 “clustering 효과”보다 “짧은 raw-scale
최적화의 과소학습”을 측정하게 된다.

이 진단은 다음을 알려준다.

- 데이터 준비, 모델 학습, cluster별 분기와 재결합 코드는 작동한다.
- 단 5 epoch의 raw-scale LSTM 수치로 논문의 LSTM 성능을 판단할 수 없다.
- 대규모 3-seed 비교 전에 모델 입력·target scaling의 미공개 선택을 제한된 별도
  실험으로 검증해야 한다.
- scaling을 시험하더라도 이번 raw 결과를 덮어쓰지 않고 별도 config와 결과 이름을
  사용해야 한다.

## 7. 로컬과 Colab 자원 사용

로컬에서는 전체 학습을 하지 않고 memory map으로 필요한 값만 선택해 압축 NPZ를
만들었다.

| 항목 | 실측값 |
|---|---:|
| 로컬 입력 준비 script 시간 | 약 0.82초 |
| 로컬 최대 RSS | 101,765,120 bytes, 약 97.1MiB |
| 입력 NPZ 크기 | 6,105,482 bytes, 약 5.82MiB |
| Colab 최종 1차 전체 시간 | 약 41.74초 |
| Colab 최종 2차 전체 시간 | 약 33.42초 |
| Colab 최대 RSS | 약 1.46GiB |

노트북의 물리 RAM은 32GB이지만 WSL이 관측한 메모리와 Codex·다른 앱의 사용량을
별도로 고려했다. 로컬 추가 사용량을 약 100MiB로 제한하고 GPU 학습 및 약 1.5GiB의
프로세스 메모리는 Colab로 옮겼으므로, “32GB 전체를 이 작업이 사용할 수 있다”는
가정을 하지 않는다.

## 8. Colab 환경과 결정성

| 항목 | 값 |
|---|---|
| GPU | Tesla T4, 15,360MiB |
| Python | 3.13.15 |
| NumPy | 2.1.3 |
| TensorFlow / Keras | 2.20.0 / 3.13.2 |
| CUDA / cuDNN | 12.5.1 / 9 |
| 입력을 준비한 Git commit | `07bdb4f3f7d8a373b48b94e50bea1a4eb813dc74` |
| 입력 Git 상태 | clean |

최종 코드를 같은 T4에서 두 번 실행했다. 실행시각과 소요 시간이 들어가는
`manifest.json`은 달라지는 것이 정상이며, 다음 네 핵심 산출물은 바이트 단위로
같았다.

| 산출물 | 두 실행의 동일 SHA-256 |
|---|---|
| `architecture_report.json` | `cae46c163e652a133c4ee5e1b22fae39d1635d97201a01d2753a16c7d86698f9` |
| `evaluation_report.json` | `d2c97e5c370e8acf1d5274b8a8ec1981b2a57bcd466f4edd5d110384c98be153` |
| `predictions.npz` | `18b39d6e353a61ef071ca0c1f69f706a1cc278a4b4dc63a2934c7e70f42fad7d` |
| `per_cell_metrics.csv` | `9b633be645bfa6f484e51b160a729eac8d2e899b922f7244f88c27f68e4b55bb` |

입력 NPZ SHA-256은
`efe0dc5d6aa3ff558914f34ae7021e67266c402b8a88cf50426cc7caf244572e`다.
Colab 작업 디렉터리에는 `.git`을 업로드하지 않으므로 그 위치의 Git 조회값은
`null`이다. 대신 로컬의 clean commit을 입력 manifest에서 검증해 최종 실행
manifest의 provenance로 기록한다.

## 9. 실행 방법

### 9.1 로컬 입력 준비

```bash
.venv/bin/python -m scripts.prepare_lstm_upc_smoke \
  --config configs/lstm_upc_smoke_milan_nov2013.json
.venv/bin/python -m unittest discover -s tests -v
```

입력 준비기는 전처리·중앙 900셀·예측 분할·UPC 정책·membership checksum을 모두
검증한 뒤에만 NPZ와 manifest를 원자적으로 게시한다. Colab 실행기는 입력 manifest가
clean Git commit에서 만들어졌는지도 다시 검사한다.

### 9.2 최소 Colab bundle

실제 bundle에는 실행에 필요한 코드와 설정, 약 5.82MiB 입력 NPZ, UPC 정책만 넣는다.
원시 데이터, 전체 처리 배열, 논문 PDF, 기존 결과 ZIP과 checkpoint는 넣지 않는다.

```bash
zip -FS data/interim/lstm_upc_smoke/colab_bundle.zip \
  scripts/__init__.py \
  scripts/build_upc_initial_groups.py \
  scripts/evaluate_naive_baselines.py \
  scripts/forecast_contract.py \
  scripts/lstm_contract.py \
  scripts/lstm_model.py \
  scripts/rctl_contract.py \
  scripts/rctl_model.py \
  scripts/run_lstm_upc_smoke.py \
  scripts/validate_upc_training_policy.py \
  configs/lstm_upc_smoke_milan_nov2013.json \
  configs/naive_baselines_milan_nov2013.json \
  configs/upc_training_policy_milan_nov2013.json \
  requirements/model.txt \
  data/interim/lstm_upc_smoke/input.npz \
  data/interim/lstm_upc_smoke/input_manifest.json \
  data/processed/upc/training_policy.json
```

### 9.3 `google-colab-cli` 실행과 회수

```bash
colab new --session gecos-lstm-upc-smoke --gpu T4
colab exec --session gecos-lstm-upc-smoke \
  --file scripts/probe_colab_runtime.py --timeout 120
colab upload --session gecos-lstm-upc-smoke \
  data/interim/lstm_upc_smoke/colab_bundle.zip \
  /content/gecos_lstm_upc_smoke_bundle.zip
colab exec --session gecos-lstm-upc-smoke \
  --file scripts/colab_lstm_upc_entry.py --timeout 1200
colab download --session gecos-lstm-upc-smoke \
  /content/gecos_lstm_upc_smoke_outputs.zip \
  data/interim/lstm_upc_smoke/colab_outputs.zip
colab stop --session gecos-lstm-upc-smoke
```

성공 여부와 관계없이 마지막에 세션을 종료한다. 기존 세션에 수정된 bundle을 다시
올려도 entry가 이전 `scripts.*` 모듈 캐시를 제거하고 새 코드를 import한다.
다운로드 ZIP은 먼저 `unzip -t`와 `unzip -l`로 검사하고, 아래 다섯 경로만 있는지
확인한 뒤 푼다.

## 10. 산출물과 다음 단계

모든 산출물은 재생성 가능한 중간 데이터 또는 실험 결과이므로 Git에서 제외한다.

| 파일 | 내용 |
|---|---|
| `data/interim/lstm_upc_smoke/input.npz` | 900셀 × 세 분할의 선택 window·target·mask |
| `data/interim/lstm_upc_smoke/input_manifest.json` | 선택 규칙, Git·입력·배열 checksum |
| `data/processed/lstm_upc_smoke/architecture_report.json` | layer, parameter, forward/backward 감사 |
| `data/processed/lstm_upc_smoke/evaluation_report.json` | 학습 history, 예측 진단, 지표와 gate |
| `data/processed/lstm_upc_smoke/predictions.npz` | Persistence와 UPC off/on 예측 |
| `data/processed/lstm_upc_smoke/per_cell_metrics.csv` | Test 셀별·정책별 MAE/MAPE/WAPE |
| `data/processed/lstm_upc_smoke/manifest.json` | provenance, 환경, 실행시간, 출력 checksum |

후속 작업으로 같은 선택 표본과 고정 epoch에서 **Train 기간에만 적합한 모델
scaling**을 별도 config로 사전 등록해 raw 결과와 비교했다. Test를 봉인한 제한
pilot에서 셀별 Min-Max가 사전 개선 문턱을 통과했으며, 구현·판정 근거는
[LSTM Train-only 셀별 Min-Max scaling pilot](12-lstm-train-only-scaling-pilot.md)에
기록한다. 다음은 이 scaling 후보를 고정하고 전체 시간축, early stopping과 seed
`42, 43, 44`를 사용하는 중앙 900셀 LSTM 본 비교다.
