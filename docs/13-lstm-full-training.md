# 중앙 900셀 LSTM 전체 Train·Validation 학습

## 1. 목적과 상태

Train-only 셀별 Min-Max scaling pilot은 같은 900셀·선택 target·LSTM·seed·5
epoch에서 UPC off Validation MAE를 raw `236.2853`에서 `30.4308`로 낮춰 결과 전에
고정한 20% 개선 문턱을 통과했다. 사전 등록 당시 다음 단계는 이 scaling을 고정하고
중앙 900셀의 모든 Train·Validation target으로 LSTM UPC off/on을 seed 3개에서
학습하는 것이었다.

이 문서의 1~9절 계약은 전체 학습 결과를 보기 전에 작성했고 12절에 실제 실행
결과를 추가했다. 범위는 **Train과 Validation을 이용한 학습·모델 선택**이며 Test
데이터, Test prediction과 Test 지표를 포함하지 않는다. 9개 학습 job과 Validation
집계는 완료했으며, 학습용 프로젝트의 최종 결정에 따라 Test 평가 코드는 추가하지
않았다.

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

## 10. 구현 상태와 산출물

사전 등록 뒤 다음 실행 경로를 구현했다.

- `prepare_lstm_full_training.py`: source checksum을 재검증하고 전역 index
  `[0, 3600)`만 담은 compact NPZ와 9개 immutable descriptor를 만든다.
- `run_lstm_full_training_job.py`: descriptor 하나에 해당하는 셀만 window로 만들고
  T4, 구조, early stopping, best-weight 복원과 Test 부재를 검사한다.
- `colab_lstm_full_job_entry.py`: Colab 임시 workspace를 매 job 새로 만들고 네 결과
  파일만 ZIP으로 회수한다.
- `aggregate_lstm_full_validation.py`: 9개 job과 checkpoint checksum을 검증하고
  cluster 예측을 원래 900셀 순서로 재결합한 뒤 Validation 지표를 집계한다.
- `test_lstm_full_pipeline.py`: Test 경계, scaler, window, cluster 순서, 음수 선형 출력,
  seed 표본표준편차를 작은 합성 배열로 검사한다.

실행 중 생성되는 파일은 모두 `.gitignore`로 보호된 전용 경로에 둔다.

| 경로 | 역할 |
|---|---|
| `data/interim/lstm_full_training/train_validation_input.npz` | Test 없는 compact 입력 |
| `data/interim/lstm_full_training/train_validation_input_manifest.json` | source·배열 checksum과 Test seal |
| `data/interim/lstm_full_training/job_descriptors/*.json` | seed·조건별 immutable job 9개 |
| `data/processed/lstm_full_training/jobs/<job_id>/best_weights.npz` | best scaled Validation epoch의 weights |
| `data/processed/lstm_full_training/jobs/<job_id>/validation_predictions.npz` | 해당 셀의 Validation 예측·mask |
| `data/processed/lstm_full_training/jobs/<job_id>/training_report.json` | history, best epoch와 구조 gate |
| `data/processed/lstm_full_training/jobs/<job_id>/run_manifest.json` | T4 환경·시간·출력 checksum |
| `data/processed/lstm_full_training/validation_report.json` | seed별 결과와 평균·표본표준편차 |
| `data/processed/lstm_full_training/validation_release_manifest.json` | 향후 잠긴 Test 평가 입력을 고정할 release |

## 11. 실행 방법

### 11.1 로컬 입력 준비

먼저 구현을 commit하여 Git 상태를 clean으로 만든 뒤 실행한다. 입력 준비기가 dirty
상태를 거부하므로 결과 전 등록한 코드와 실제 Colab 코드가 달라지는 일을 막는다.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.prepare_lstm_full_training \
  --config configs/lstm_full_training_milan_nov2013.json
```

### 11.2 최소 Colab bundle

원시 데이터, 전체 4,320시점 배열, 논문 PDF와 기존 결과는 업로드하지 않는다. 아래
코드 의존성과 Test가 없는 compact 입력만 ZIP에 넣는다.

```bash
zip -FS data/interim/lstm_full_training/colab_bundle.zip \
  scripts/__init__.py \
  scripts/build_upc_initial_groups.py \
  scripts/evaluate_naive_baselines.py \
  scripts/forecast_contract.py \
  scripts/lstm_contract.py \
  scripts/lstm_full_contract.py \
  scripts/lstm_model.py \
  scripts/lstm_scaling_contract.py \
  scripts/prepare_lstm_full_training.py \
  scripts/prepare_lstm_scaling_pilot.py \
  scripts/prepare_lstm_upc_smoke.py \
  scripts/rctl_contract.py \
  scripts/rctl_model.py \
  scripts/run_lstm_full_training_job.py \
  scripts/run_lstm_upc_smoke.py \
  scripts/validate_upc_training_policy.py \
  configs/lstm_full_training_milan_nov2013.json \
  configs/lstm_scaling_pilot_milan_nov2013.json \
  configs/lstm_upc_smoke_milan_nov2013.json \
  requirements/model.txt \
  data/interim/lstm_full_training/train_validation_input.npz \
  data/interim/lstm_full_training/train_validation_input_manifest.json
```

### 11.3 `google-colab-cli` job 실행

T4 세션 하나에 bundle을 한 번 올리고 descriptor를 하나씩 교체하여 순차 실행한다.
각 결과 ZIP은 job ID가 들어간 로컬 파일명으로 즉시 회수하고 검사한다. 한 job의
실패가 다른 job의 산출물을 덮어쓰지 않는다.

```bash
colab new --session gecos-lstm-full --gpu T4
colab exec --session gecos-lstm-full \
  --file scripts/probe_colab_runtime.py --timeout 120
colab upload --session gecos-lstm-full \
  data/interim/lstm_full_training/colab_bundle.zip \
  /content/gecos_lstm_full_training_bundle.zip

for job in data/interim/lstm_full_training/job_descriptors/*.json; do
  job_id="$(basename "$job" .json)"
  colab upload --session gecos-lstm-full \
    "$job" /content/gecos_lstm_full_job.json
  colab exec --session gecos-lstm-full \
    --file scripts/colab_lstm_full_job_entry.py --timeout 7500
  colab download --session gecos-lstm-full \
    /content/gecos_lstm_full_job_outputs.zip \
    "data/interim/lstm_full_training/${job_id}_outputs.zip"
  unzip -t "data/interim/lstm_full_training/${job_id}_outputs.zip"
  unzip -o "data/interim/lstm_full_training/${job_id}_outputs.zip" -d .
done

colab stop --session gecos-lstm-full
```

세션은 성공·실패와 관계없이 종료한다. 인프라 오류가 난 job은 descriptor와 bundle을
바꾸지 않고 그 job만 재실행한다. wall-clock 상한이나 4GiB soft RSS를 넘으면 표본,
batch 또는 epoch를 즉석에서 축소하지 않고 구현 변경을 별도 commit으로 남긴다.

### 11.4 Validation 집계

```bash
.venv/bin/python -m scripts.aggregate_lstm_full_validation \
  --config configs/lstm_full_training_milan_nov2013.json
```

집계기는 9개 job이 전부 존재하지 않으면 실패한다. 이번 명령은 Test 배열을 읽거나
Test 지표를 계산하지 않는다.

## 12. 실행 결과

### 12.1 실행 provenance와 자원

사전 등록 뒤 commit `57030bd6dd6757ca7ebfac3ea8700e9a1bbcbc16`의 clean
상태에서 입력을 만들었다. compact 입력 SHA-256은
`bdcafbfa1ee04e12d01f64e120648dc70b3b778a3206f722e24c2031a8a87321`이며 크기는
11,231,547 bytes다. 로컬 준비 peak RSS는 151,740,416 bytes, 약 144.7MiB로
256MiB 상한을 통과했다.

모든 job은 Colab Tesla T4 15,360MiB, Python 3.13.15, NumPy 2.1.3,
TensorFlow 2.20.0, Keras 3.13.2에서 실행했다. 성공해 보존한 9개 job의 순수 fit
시간 합은 3,902.57초, 약 65.0분이다. 관측한 세션 누적 peak RSS 최댓값은
2,324,791,296 bytes, 약 2.17GiB로 4GiB soft limit 안이다.

seed 44 cluster 0의 첫 실행은 학습 gate를 모두 통과하고 best epoch를 출력한 직후
Colab VM이 404/401로 소실되어 결과 ZIP을 회수하지 못했다. 서버에 남은 이름 없는
orphan T4를 해제한 뒤 동일 commit·bundle·descriptor로 그 job만 다시 실행했다.
재실행의 best epoch와 scaled Validation MAE는 첫 실행과 동일한 3과
`0.03194784000515938`이었다. 이는 등록한
`rerun_only_same_immutable_job_after_infrastructure_failure` 정책에 해당하며, 결과를
보고 설정을 바꾼 재시도가 아니다.

### 12.2 job별 early stopping 결과

| seed | 조건 | 셀 | 완료 epoch | best epoch | best scaled Validation MAE | fit 초 |
|---:|---|---:|---:|---:|---:|---:|
| 42 | UPC off | 900 | 13 | 8 | 0.03338347 | 708.25 |
| 42 | UPC on cluster 0 | 611 | 8 | 3 | 0.03200935 | 279.36 |
| 42 | UPC on cluster 1 | 289 | 19 | 14 | 0.03578634 | 324.98 |
| 43 | UPC off | 900 | 13 | 8 | 0.03330174 | 739.90 |
| 43 | UPC on cluster 0 | 611 | 8 | 3 | 0.03200490 | 273.40 |
| 43 | UPC on cluster 1 | 289 | 20 | 15 | 0.03572622 | 329.45 |
| 44 | UPC off | 900 | 12 | 7 | 0.03340585 | 648.48 |
| 44 | UPC on cluster 0 | 611 | 8 | 3 | 0.03194784 | 248.99 |
| 44 | UPC on cluster 1 | 289 | 23 | 18 | 0.03571745 | 349.76 |

조건별 최적 epoch가 3부터 18까지 달라 고정 5 epoch만으로 UPC off/on을 비교하면
조건에 따라 과소학습 또는 이미 악화가 시작된 weights를 비교할 수 있음을 확인했다.
scaled MAE는 셀 범위가 다른 cluster 모델의 선택값일 뿐이므로 cluster 간 성능값으로
직접 비교하지 않는다.

### 12.3 전체 Validation 결과

다음은 720개 Validation target/셀, `all_targets`, micro 집계의 seed 평균 ± 표본
표준편차다. MAPE는 읽기 쉬운 percent 단위이며 WAPE는 ratio 단위다.

| 모델 | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| LSTM UPC off | 28.3164 ± 0.0610 | 11.4516% ± 0.1371%p | 0.102404 ± 0.000221 |
| LSTM UPC on | 28.2098 ± 0.0407 | 10.7346% ± 0.0377%p | 0.102019 ± 0.000147 |

결측·Internet 전체 null target 796개를 제외한 `observed_targets_only` micro 결과는
다음과 같다. MAPE는 원래도 양수 target만 사용하므로 이 데이터에서는
`all_targets`와 같다.

| 모델 | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| LSTM UPC off | 28.3512 ± 0.0611 | 11.4516% ± 0.1371%p | 0.102404 ± 0.000221 |
| LSTM UPC on | 28.2445 ± 0.0408 | 10.7346% ± 0.0377%p | 0.102019 ± 0.000147 |

`all_targets`의 cell-macro 평균은 UPC off/on 순서로 MAE `28.3164/28.2098`, MAPE
`12.0447%/11.1765%`, WAPE `0.115408/0.111791`이다. 모든 셀이 같은 720개 target을
갖는 `all_targets`에서는 micro MAE와 cell-macro MAE가 같지만, WAPE와 MAPE는
분모와 셀별 비율 평균 방식이 달라 서로 같지 않다.

### 12.4 해석: scaling과 UPC에서 배운 점

같은 중앙 900셀 Validation의 기존 Persistence는 MAE `31.9310`, MAPE
`12.9357%`, WAPE `0.115476`이었다. 이에 비해 3-seed 평균 LSTM UPC off는 MAE와
WAPE가 약 11.32%, UPC on은 약 11.65% 낮았다. 이 비교는 full LSTM이 raw 5 epoch
smoke의 심한 과소학습을 벗어났고 Train-only 셀별 scaling과 early stopping이 실제
전체 target에서도 유효했음을 보여준다.

UPC on은 세 seed 모두 off보다 낮았다. on−off paired 차이의 평균 ± 표본표준편차는
MAE `-0.1066 ± 0.0922`, MAPE `-0.7170 ± 0.0993%p`, WAPE
`-0.000385 ± 0.000334`다. 상대 개선은 MAE와 WAPE에서 약 0.38%, MAPE에서 약
6.26%다. 따라서 현재 Validation에서는 UPC가 일관된 방향의 작은 MAE/WAPE 개선과
더 큰 MAPE 개선을 보였다고 말할 수 있다. 다만 seed가 3개뿐이고 Validation을 모델
선택에 사용했으므로 “일반화 성능 향상이 확정됐다”거나 논문의 clustering 효과를
그대로 재현했다고 주장하지 않는다.

특히 MAPE는 traffic이 작은 양수 target의 상대오차 영향을 크게 받는다. UPC 효과를
MAPE 하나로 과장하지 않고 MAE·WAPE와 함께 해석한다. 이번 성능값은 pipeline 통과
gate가 아니며, UPC on이 나빴더라도 둘 다 다음 잠긴 평가 후보로 보존했을 것이다.

선형 출력과 무클리핑 계약에서 raw 단위 음수 Validation 예측은 seed 42/43/44의 UPC
off가 각각 `2/1/1`개, 재결합한 UPC on이 `26/22/21`개였다. 최솟값은 off
`-0.07198`, on `-0.55696`으로 전체 648,000개 예측 중 매우 드물지만 0은 아니다.
결과를 좋게 만들기 위해 0으로 자르지 않았고 `GAP-LSTM-02`의 저자 출력 activation
불명확성은 유지한다.

### 12.5 gate, 결정성 및 출력 고정

다음 gate가 모두 통과했다.

- clean source commit과 입력·config·descriptor checksum 일치
- 9개 job의 `165,185` parameter, 유한 history·weights·prediction
- callback best weights와 저장 직전 모델 weights의 배열 단위 exact 일치
- cluster별 예측의 900셀 원래 순서 exact 재결합
- 두 target 정책과 micro/cell-macro의 유한 Validation 지표
- 입력, job output과 집계 output에 Test 배열·평가 없음
- Persistence 우위 또는 UPC 개선 여부를 통과 조건으로 사용하지 않음

집계를 두 번 실행했을 때 다음 핵심 파일은 바이트 단위로 같았다.

| 산출물 | SHA-256 |
|---|---|
| `training_jobs.csv` | `1efded3de1401c77a508db8855a8240655402b5c5bc835c4aa46be5665538ae1` |
| `validation_report.json` | `fb3ec426f1db258754af5c4fb01f2efb957014db2770c9a3e79e5a9493ffdd60` |
| `validation_predictions.npz` | `bac2a783dfe514cd268b070495d5ca40ae3978b887770b9bd081350325341e1b` |
| `validation_per_cell_metrics.csv` | `bb816deec0bc3555feb066a93a3d8657b3e9b529fca2ca59ac38f14fd32cea05` |
| `validation_release_manifest.json` | `5cb81e6c20411369069f6b9821b1f22938bdf3c096f927d39a61fa07c839ee88` |

`aggregation_manifest.json`은 실행시각과 소요 시간을 포함하므로 재실행 때 hash가
달라지는 것이 정상이다. release manifest는 9개 descriptor, training report,
Validation prediction, best checkpoint와 run manifest hash를 모두 고정하며 상태는
`ready_for_locked_test_evaluation`이다.

### 12.6 결론과 최종 경계

이번 단계의 결론은 **전체 Train·Validation LSTM 학습 경로가 재현 가능하게
완료됐고, UPC off/on 두 후보가 잠겼다**는 것이다. 아직 Test를 실행하지 않았으므로
이 문서의 수치는 최종 성능이 아니다. 또한 이전 raw smoke의 Test 노출 때문에 다음
평가도 pristine holdout이 아니라
`locked final evaluation with prior raw-smoke exposure`로 표시해야 한다.

사전 등록 시에는 다음 변경에서 위 release SHA를 고정한 Test 전용 bundle과 일회성
평가기를 추가할 계획이었다. 최종적으로는 과거 raw smoke의 Test 노출과 추가 학습
가치를 함께 고려해 **release를 평가 준비 상태로 보존하고 Test를 실행하지 않는 것**을
선택했다. 이 결정은 Validation 결과를 Test 결과로 바꾸지 않으며, 문서의 지표도
계속 Validation으로만 표시한다.

향후 새로운 학습 목표로 재개한다면 이 release SHA를 변경하지 않은 별도 Test 전용
변경에서만 평가한다. Test를 본 뒤 scaler, epoch, seed, clipping, UPC membership
또는 모델 구조를 바꾸지 않는다. 프로젝트 전체의 종료 판단은
[학습용 논문 재현 최종 정리](14-study-reproduction-conclusion.md)를 따른다.
