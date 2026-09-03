# 중앙 900셀 LSTM 전체 Train·Validation 학습

## 1. 목적과 상태

Train-only 셀별 Min-Max scaling pilot은 같은 900셀·선택 target·LSTM·seed·5
epoch에서 UPC off Validation MAE를 raw `236.2853`에서 `30.4308`로 낮춰 결과 전에
고정한 20% 개선 문턱을 통과했다. 다음 단계는 이 scaling을 고정하고 중앙 900셀의
모든 Train·Validation target으로 LSTM UPC off/on을 seed 3개에서 학습하는 것이다.

이 문서의 계약은 전체 학습 결과를 보기 전에 작성한다. 현재 단계는 **Train과
Validation을 이용한 학습·모델 선택**이며 Test 데이터, Test prediction과 Test
지표를 포함하지 않는다. 9개 학습 job과 Validation 집계가 통과한 뒤 별도 변경에서
Test 평가 코드를 추가한다.

## 2. LSTM을 RCTL보다 먼저 실행하는 이유

- LSTM 구조, scaling과 UPC 재결합 경로는 작은 smoke에서 이미 검증됐다.
- full LSTM은 scaling pilot이 전체 target에서도 유지되는지 확인하는 가장 단순한
  학습 기준점이다.
- 이 기준점이 있어야 뒤의 RCTL 차이를 clustering 효과와 모델 구조 효과로 나눠
  해석할 수 있다.
- RCTL은 논문 parameter 수 `173,633`과 현재 두 구현의 수가 일치하지 않는 gap이
  남아 있다. 두 모델을 동시에 확장하면 실패 원인이 섞인다.

scaler 변형을 더 탐색하거나 raw-scale 전체 학습을 반복하지 않는다. raw smoke와
scaling pilot은 각각 구조 검증과 scaling 선택 근거로 보존한다.

## 3. 데이터와 Test 봉인 계약

### 3.1 공간과 시간

| 항목 | 값 |
|---|---:|
| 공간 | `central-900-approximate` 전체 900셀 |
| 입력 | 과거 8개 10분 traffic |
| horizon | 다음 1개 시점, 10분 |
| Train target index | `[8, 2880)` |
| Train target/셀 | 2,872 |
| Train 전체 표본 | 2,584,800 |
| Validation target index | `[2880, 3600)` |
| Validation target/셀 | 720 |
| Validation 전체 표본 | 648,000 |

Train·Validation bundle은 전역 index `[0, 3600)`의 중앙 traffic과 두 결측 mask만
포함한다. 첫 Train target의 입력에 필요한 index 0부터 Validation 마지막 target
index 3,599까지만 허용한다. 이름뿐 아니라 배열 shape, timestamp 최댓값과 target
index를 검사하여 Test 시작 index 3,600 이상 값이 들어가면 중단한다.

### 3.2 Test의 해석 한계

이전 raw 고정 5 epoch pipeline smoke에서 Test를 진단용으로 한 번 평가했다. 따라서
향후 5일 Test는 완전히 보지 않은 pristine holdout이라고 부르지 않고
`locked final evaluation with prior raw-smoke exposure`로 표시한다.

이번 scaling 선택과 본 학습의 best epoch는 Test를 사용하지 않는다. 9개 학습과
Validation 결과, config와 checkpoint checksum이 동결된 뒤에만 Test 전용 bundle과
평가기를 만든다. Test 결과를 확인한 뒤 scaler, 구조, seed, epoch 선택 또는
clipping을 바꾸지 않는다.

## 4. scaling 계약

각 셀 `c`의 Train 2,880시점 전체로만 minimum과 range를 적합한다.

```text
minimum[c] = min(traffic[c, 0:2880])
range[c]   = max(traffic[c, 0:2880]) - minimum[c]
scaled     = (value - minimum[c]) / range[c]
restored   = scaled * range[c] + minimum[c]
```

- input과 target에 같은 셀별 parameter를 사용한다.
- dtype은 `float32`다.
- Validation은 scaler 적합에 사용하지 않는다.
- transform과 inverse prediction을 clipping하지 않는다.
- zero-range 셀이 하나라도 있으면 중단한다.
- 최대 허용 역변환 오차는 `0.001`이다.

## 5. 모델과 학습 계약

논문 Table III의 `165,185` parameter를 정확히 만드는 재구성 후보를 유지한다.

```text
LSTM(64, return_sequences=True)
Dropout(0.05)
LSTM(128, return_sequences=True)
Dropout(0.05)
LSTM(64)
Dropout(0.05)
Dense(1, linear)
```

| 항목 | 값 |
|---|---|
| seed | 42, 43, 44 |
| optimizer / learning rate | Adam / 0.001 |
| loss | 셀별 scaled target의 MAE |
| batch size | 512 |
| shuffle | `false` |
| 최대 epoch | 1,000 |
| early stopping monitor | `val_loss`, scaled Validation MAE |
| mode / patience / min_delta | `min` / 5 / 0 |
| start_from_epoch | 0 |
| best weight | 복원 후 NumPy NPZ로 저장 |
| prediction clipping | 없음 |

scaled `val_loss`를 감시하는 이유는 실제 최적화 loss와 모델 선택 단위를 일치시키기
위해서다. 원래 traffic 단위 Validation 지표는 역변환 후 보고하지만 best epoch를
고르는 데 사용하지 않는다. 논문에 이 선택이 공개되지 않은 한계는 `GAP-LSTM-03`으로
추적한다.

## 6. 독립 job 계약

한 Colab 호출에서 모든 모델을 학습하지 않고 seed와 조건별로 다음 9개 job을
독립 실행한다.

| seed | UPC off | UPC on cluster 0 | UPC on cluster 1 |
|---:|---|---|---|
| 42 | 900셀 | 611셀 | 289셀 |
| 43 | 900셀 | 611셀 | 289셀 |
| 44 | 900셀 | 611셀 | 289셀 |

각 job은 동일한 clean Git commit, config와 입력 NPZ checksum을 요구한다. 출력은
best weights, Validation scaled/raw prediction, 결정적인 학습 보고서와 실행시간을
분리한 manifest로 구성한다. 인프라 실패는 같은 immutable job만 다시 실행할 수
있고, 설정을 바꾼 재실행은 새 실험 버전으로 취급한다.

job당 wall-clock 7,200초를 넘으면 성능 결과가 아니라 `incomplete`로 처리한다.
성능이 Persistence보다 좋아야 하거나 UPC on이 off보다 좋아야 한다는 조건은
pipeline 합격 gate가 아니다.

## 7. Validation 집계 계약

9개 job이 모두 완료되면 다음 순서로 집계한다.

1. job ID, seed, 역할, config·입력 checksum의 완전성 확인
2. history, weights와 prediction의 유한성 확인
3. cluster 0·1 예측을 중앙 manifest의 원래 900셀 순서로 scatter
4. 누락·중복과 잘못된 cell ID가 없는지 exact 검사
5. 셀별 scaler로 prediction을 원래 traffic 단위로 역변환
6. seed별 UPC off/on Validation MAE·MAPE·WAPE 계산
7. `all_targets`와 `observed_targets_only`를 분리
8. micro와 cell-macro 및 셀별 분포 기록
9. seed 3개의 개별값, 평균과 표본 표준편차(`ddof=1`) 기록

Validation 결과가 좋거나 나쁜지는 실행 계약의 통과 여부를 바꾸지 않는다. UPC on이
off보다 나쁘더라도 두 조건 모두 향후 잠긴 Test 평가에 포함한다.

## 8. 자원 계획

로컬 32GB RAM 전체를 전용 자원으로 가정하지 않는다. compact 입력은 약 20MiB
수준으로 만들고 로컬 peak RSS 상한을 256MiB로 둔다. 로컬에서는 TensorFlow
학습을 하지 않는다.

Colab에서는 job에 필요한 셀만 window로 만들고 모델 종료 후 배열과 TensorFlow
session을 정리한다. peak RSS soft limit은 4GiB다. 초과 시 표본이나 batch를 조용히
바꾸지 않고, 결정적 batch 생성 방식으로 메모리 구현을 수정한 새 commit에서 해당
job을 다시 시작한다.

## 9. 합격 조건

- 모든 source와 배열 checksum 일치
- clean Git commit에서 입력 생성
- bundle과 job 실행 경로에 Test 없음
- 실제 모델 parameter 수 `165,185`
- 9개 job 모두 존재하고 중복 없음
- history, best weights와 Validation prediction이 모두 유한함
- best scaled Validation epoch의 weights가 복원됨
- UPC on 예측이 seed별 900셀 순서로 exact 재결합됨
- 모든 Validation metric이 유한함
- 성능 결과가 구조 gate에 사용되지 않음

## 10. 실행 결과

결과 전 사전 등록 상태다. 코드와 합성 테스트를 구현하고 clean commit에서 입력을
만든 뒤 Colab T4로 9개 job을 실행한다. 결과가 나온 뒤 이 절에 실제 best epoch,
실행시간, peak RSS, Validation 지표와 output checksum을 추가하되 위 계약은 바꾸지
않는다.
