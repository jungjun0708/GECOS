# 예측 표본 계약과 학습 없는 기준선

## 1. 결론

공통 예측 표본, 시간 분할, 결측 및 지표 계약을 코드로 고정하고 다음 두 기준선을
실제 Milan 10,000셀 데이터에서 평가했다.

- `persistence`: 직전 10분 값으로 다음 값을 예측
- `daily_seasonal_naive`: 24시간 전 같은 시각 값으로 다음 값을 예측

주 Test 결과에서 Persistence가 두 공간 범위의 MAE, MAPE, WAPE 모두 일간
계절성 기준선보다 좋았다. 중앙 900셀의 micro MAE는 `28.589375`, 전체
10,000셀은 `7.733168`이었다. 이는 뒤에 구현할 LSTM과 RCTL이 최소한 넘어야 할
데이터 기반 기준점이다.

이 두 기준선은 논문 Table II에 보고된 모델이 아니다. 따라서 논문의 수치와 직접
일치시키는 대상이 아니라, 같은 표본과 평가 코드가 상식적으로 동작하는지 확인하고
후속 모델의 실질적인 이득을 판단하기 위한 대조군이다.

## 2. 이 단계를 모델보다 먼저 한 이유

신경망을 먼저 학습하면 낮은 오차가 모델 구조 때문인지, 시간 분할이나 MAPE 구현이
달라서인지 구분하기 어렵다. 학습이 없는 기준선은 다음 계약을 빠르고 싸게 검증한다.

1. 과거 8개 값과 다음 target의 위치가 한 칸 밀리지 않았는가?
2. Validation과 Test 표본이 target 시각 기준으로 정확히 분리됐는가?
3. 입력 window나 기준선 참조값에 미래 정보가 들어가지 않았는가?
4. 0 target과 원본 결측을 어떤 방식으로 지표에 반영했는가?
5. 중앙 900셀과 전체 10,000셀에서 같은 지표 구현을 사용했는가?

이 계약은 이후 LSTM과 RCTL에서도 그대로 재사용한다. 모델이 바뀌어도 target과
평가 방식이 바뀌지 않아야 공정한 비교가 된다.

## 3. 예측 표본과 시간 분할

논문의 sequence length `q=8`과 one-step 예측 해석에 따라 다음 표본을 만든다.

```text
입력: X[cell, t-8:t]
정답: y[cell, t]
예측 시점: 10분 뒤 한 시점
```

길이 4,320인 월간 시계열의 첫 8개 값은 과거 window를 만드는 데 사용되므로 target은
셀당 `4,320 - 8 = 4,312`개다. 표본을 먼저 섞거나 임의 비율로 나누지 않고,
`Europe/Rome`의 **target local timestamp**로 다음처럼 배정한다.

| 구간 | target 기간 | 셀당 target 수 | 역할 |
|---|---|---:|---|
| Train | 11월 1일 01:20 ~ 11월 20일 23:50 | 2,872 | 후속 모델 학습 |
| Validation | 11월 21일 00:00 ~ 11월 25일 23:50 | 720 | 후속 모델 선택 |
| Test | 11월 26일 00:00 ~ 11월 30일 23:50 | 720 | 주 결과 |
| Paper holdout | 11월 21일 00:00 ~ 11월 30일 23:50 | 1,440 | 논문 설명과 맞춘 보조값 |

Train, Validation, Test는 전체 4,312개 target을 중복이나 빈틈 없이 정확히 분할한다.
Paper holdout은 Validation과 Test를 합친 보조 구간이며 주 Test 결과와 섞지 않는다.

### 3.1 rolling one-step history

평가는 `rolling_one_step_with_observed_history`다. 각 target을 예측할 때 그 시각보다
앞선 실제 관측값은 Validation 또는 Test 구간에 속하더라도 사용할 수 있다.
Persistence는 `t-1`, 일간 기준선은 `t-144`를 참조한다. 두 경우 모두 참조 index가
target index보다 작은지 검사한다.

이는 매 10분마다 새 관측을 받은 뒤 다음 10분만 예측하는 온라인 문제다. Test 5일
전체를 한 번에 예측하고 자기 예측을 다음 입력으로 넣는 재귀 multi-step 문제와는
다르다. 후속 모델도 주 비교에서는 같은 rolling one-step 조건을 사용한다.

## 4. 공간 범위와 기준선

| 공간 범위 | 셀 수 | 의미 |
|---|---:|---|
| `central_900` | 900 | 공식 Grid에서 선택한 `central-900-approximate` 영역 |
| `all_10000` | 10,000 | 전처리를 통과한 전체 Milan Grid |

기준선 수식은 다음과 같다.

```text
persistence:             y_hat[cell, t] = y[cell, t-1]
daily_seasonal_naive:    y_hat[cell, t] = y[cell, t-144]
```

두 기준선에는 학습 파라미터, optimizer, seed가 없다. `q=8`은 공통 target 집합을
만드는 계약이고, 각 기준선은 그 동일한 target에 필요한 과거 lag만 참조한다.

## 5. 결측과 지표 계약

전처리는 원본에 없는 셀-시점과 Internet 전체 공란을 각각 mask로 기록하면서 학습
행렬에는 0을 채운다. 그 영향을 숨기지 않기 위해 두 target 정책을 동시에 출력한다.

| 정책 | 처리 | 용도 |
|---|---|---|
| `all_targets` | 전처리에서 채운 0을 포함 | 주 데이터 계약 |
| `observed_targets_only` | 두 mask가 표시된 target만 제외 | 결측 민감도 검사 |

`observed_targets_only`도 과거 lag가 결측인 예측을 제거하지 않는다. 기준선은 이미
채워진 0을 참조하며, 대신 `lag_source_missing_count`와
`lag_source_internet_all_null_count`를 별도로 보고한다. 이 정책을 나중에 바꾸려면
기존 결과를 덮어쓰지 않고 별도 민감도 실험으로 추가해야 한다.

필수 지표는 다음과 같다.

- MAE: eligible target의 평균 절대오차
- MAPE ratio: eligible target 중 `y > 0`인 항목의 평균 `|y-y_hat|/y`
- MAPE percent: MAPE ratio에 100을 곱한 값
- WAPE: eligible target의 `sum(|y-y_hat|) / sum(y)`
- micro: 모든 cell-target 쌍을 합쳐 계산
- cell-macro: 셀별 지표의 동일 가중 평균

MAPE에서 빠진 0 target 수, 결측 target 수, eligible target 수를 결과에 함께 기록한다.
파일 출력은 같은 계산의 SIMD 및 chunk 배치에 따른 double 정밀도 끝자리 차이를 없애기
위해 유효숫자 12자리로 정규화한다.

## 6. 실행 방법

UPC 단계와 같은 Python 3.12 환경을 사용한다.

```bash
python3 -m pip --python .venv/bin/python install \
  --requirement requirements/upc.txt
```

계약과 지표 테스트를 실행한다.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

실제 데이터를 평가한다.

```bash
.venv/bin/python -m scripts.evaluate_naive_baselines \
  --config configs/naive_baselines_milan_nov2013.json
```

실행 전에 전처리 manifest, 다섯 NumPy 입력 파일, 중앙 900셀 manifest와 CSV의 크기,
SHA-256, shape, dtype, 시간 간격과 ID 매핑을 다시 검증한다. 검증에 실패하면 결과를
게시하지 않는다.

## 7. 실제 Test 결과

### 7.1 주 결과: `all_targets`, micro

| 범위 | 기준선 | MAE | MAPE ratio | MAPE percent | WAPE |
|---|---|---:|---:|---:|---:|
| 중앙 900셀 | Persistence | 28.589375 | 0.126581 | 12.6581% | 0.103794 |
| 중앙 900셀 | Daily seasonal | 50.711360 | 1.486162 | 148.6162% | 0.184108 |
| 전체 10,000셀 | Persistence | 7.733168 | 0.138741 | 13.8741% | 0.116508 |
| 전체 10,000셀 | Daily seasonal | 12.789351 | 0.367648 | 36.7648% | 0.192685 |

Persistence는 Daily seasonal 대비 중앙 900셀에서 MAE와 WAPE가 약 `43.62%`, MAPE가
약 `91.48%` 낮았다. 전체 10,000셀에서는 MAE와 WAPE가 약 `39.53%`, MAPE가 약
`62.26%` 낮았다. 이 데이터와 10분 예측에서는 “어제 같은 시각”보다 “바로 직전
시각”이 훨씬 강한 기준선이라는 뜻이다.

중앙 영역의 MAE가 전체보다 큰 것은 중앙 셀의 traffic 규모가 더 크기 때문일 수 있다.
규모에 덜 직접적으로 좌우되는 WAPE도 중앙 Persistence `0.103794`, 전체 Persistence
`0.116508`로 서로 다르므로 MAE 하나만으로 공간 범위를 비교하지 않는다.

### 7.2 micro와 cell-macro

| 범위 | 기준선 | micro MAPE | cell-macro MAPE | micro WAPE | cell-macro WAPE |
|---|---|---:|---:|---:|---:|
| 중앙 900셀 | Persistence | 0.126581 | 0.132502 | 0.103794 | 0.121604 |
| 중앙 900셀 | Daily seasonal | 1.486162 | 1.511788 | 0.184108 | 0.200386 |
| 전체 10,000셀 | Persistence | 0.138741 | 0.139404 | 0.116508 | 0.136702 |
| 전체 10,000셀 | Daily seasonal | 0.367648 | 0.371645 | 0.192685 | 0.210293 |

모든 경우 cell-macro WAPE가 micro WAPE보다 높다. traffic 합이 작은 셀도 동일한
가중치를 받으면 상대적으로 예측하기 어려운 셀의 영향이 커진다는 신호다. 후속 모델은
전체 micro 개선만 아니라 셀별 분포가 함께 좋아지는지 확인해야 한다.

### 7.3 결측 target 민감도

| 범위 | Test 후보 수 | 관측 target 수 | 제외 결측 수 | 전체/관측 Persistence MAE |
|---|---:|---:|---:|---:|
| 중앙 900셀 | 648,000 | 646,655 | 1,345 | 28.589375 / 28.648603 |
| 전체 10,000셀 | 7,200,000 | 7,196,470 | 3,530 | 7.733168 / 7.736912 |

결측 제외 비율은 중앙 약 `0.208%`, 전체 약 `0.049%`로 작고 MAE/WAPE 변화도 작다.
MAPE는 두 정책에서 동일했다. 해당 결측 target은 전처리에서 0으로 채워졌고 MAPE가
원래부터 `y=0`을 제외하기 때문이다. 이는 두 정책이 중복이라는 뜻이 아니라, 왜 같은
수치가 나왔는지 mask count로 설명할 수 있다는 뜻이다.

### 7.4 논문 설명에 맞춘 10일 보조 구간

| 범위 | 기준선 | micro MAE | MAPE percent | WAPE |
|---|---|---:|---:|---:|
| 중앙 900셀 | Persistence | 30.260167 | 12.7969% | 0.109646 |
| 중앙 900셀 | Daily seasonal | 59.369150 | 87.0051% | 0.215121 |
| 전체 10,000셀 | Persistence | 8.227778 | 14.3464% | 0.122660 |
| 전체 10,000셀 | Daily seasonal | 14.614338 | 32.0830% | 0.217872 |

이 값에는 Validation이 포함되므로 모델 선택 후 최종 성능을 주장하는 주 지표로
사용하지 않는다.

## 8. 산출물과 결정성

생성 파일은 모두 `data/processed/baselines/` 아래에 있어 Git에서 제외된다.

| 파일 | 내용 | 크기 | SHA-256 |
|---|---|---:|---|
| `summary.json` | 2개 범위 × 3개 구간 × 2개 기준선 × 2개 정책의 24개 결과 | 43,077 bytes | `aa6e1a20cb058fb33bd923abcd422fb759d4c7eab4db9b4d2993b37865cf905f` |
| `per_cell_metrics.csv` | Test 셀별 지표 43,600행 | 6,909,864 bytes | `2d5dc0cfd9fc06610338b8db28497dc559b975a8cb3856ea6cbb4b59f02ff1ed` |
| `manifest.json` | 입력·config·Git·환경·실행시간·출력 checksum | 실행별 metadata 포함 | 실행마다 생성 |

실제 데이터에서 연속 두 번 실행했을 때 동적 실행 metadata가 있는 manifest를 제외한
JSON과 CSV의 SHA-256이 같았다. 합성 테스트에서는 cell chunk를 `1`과 `3`으로 바꿔도
두 파일의 checksum이 같은지 확인한다.

## 9. 노트북 자원과 Colab 역할

최종 WSL 로컬 실행의 측정값은 다음과 같다.

| 항목 | 측정값 |
|---|---:|
| wall time | 약 2.5초 |
| 최대 RSS | 약 348MiB |
| swap | 0 |
| cell chunk | 256개 |
| 계산용 chunk 추정치 | 약 19.7MiB |
| 출력 크기 | 약 6.63MiB |

입력 배열은 memory map으로 열고 현재 256개 셀만 계산 배열로 만든다. 최대 RSS가
노트북의 32GB RAM보다 훨씬 작아 Codex와 다른 앱이 메모리를 사용 중이어도 안전한
여유가 있다. multiprocessing을 사용하지 않아 다른 작업과 CPU 경합도 제한적이다.

이 단계는 학습도 GPU 연산도 없고 Colab 업로드 시간이 로컬 계산보다 길다. 따라서
`google-colab-cli`나 Colab 세션을 사용하지 않는다. Colab은 TensorFlow가 필요한
LSTM/RCTL smoke test와 본 학습에서 GPU, 런타임 유지, 산출물 회수에 사용한다.

## 10. 테스트가 보장하는 사항

- target 시각 기준 Train/Validation/Test 경계와 정확한 표본 수
- 첫 Test target 입력의 마지막 index가 target보다 과거인지 여부
- timestamp 10분 간격 훼손 차단
- 잘못된 split target 수와 기준선 lag 설정 차단
- MAE, MAPE, WAPE의 micro 및 cell-macro 수식
- MAPE의 0 target 제외와 두 결측 mask 집계
- 실제 manifest checksum 변조 시 산출물 게시 차단
- 중앙 셀 ID와 전체 행렬 위치 매핑
- 일간 반복 합성 신호에서 daily seasonal 오차가 0인지 여부
- cell chunk 크기와 무관한 결과 파일 checksum

## 11. 다음 단계의 통과 조건

다음 PCC 및 모델 단계는 이 기준선 계약을 새로 정의하지 않고 가져다 써야 한다.

1. LSTM/RCTL 예측을 같은 Test target index에 맞춘다.
2. 결과에 split, 공간 범위, target 정책, micro/cell-macro를 항상 표시한다.
3. 중앙 900셀 Test에서 최소한 Persistence보다 나은지를 먼저 확인한다.
4. Persistence보다 못하면 대규모 학습보다 scaling, window 정렬, 학습 수렴을 먼저
   진단한다.
5. 논문 Table II와 비교할 때도 우리의 엄격한 5일 Test와 논문 설명형 10일 holdout을
   구분한다.

이번 단계의 중요한 학습은 복잡한 모델이 반드시 강한 기준선보다 낫지는 않다는 점과,
MAPE 하나만 보면 결측 처리와 셀별 편차를 놓칠 수 있다는 점이다.
