# GECOS 핵심 재현 범위와 실험 계약

## 1. 문서 목적

이 문서는 GECOS 논문 재현에서 무엇을 구현하고, 어떤 데이터와 평가 규칙을
사용하며, 어디까지를 성공으로 판단할지 사전에 고정한다. 결과를 확인한 뒤
전처리 방식이나 평가 기준을 바꾸는 일을 방지하고, 논문에 공개되지 않은 선택은
명시적인 가정으로 남기는 것이 목적이다.

- 대상 논문: *Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*
- 논문 DOI: <https://doi.org/10.1109/TNSM.2025.3599168>
- 원저자 공개 코드: <https://github.com/Superint-Lab/GECOS>
- 데이터: <https://doi.org/10.7910/DVN/EGZHFV>
- 문서 상태: 실행 계약 v1 (LSTM Train-only scaling pilot 완료)

이 문서에서 정하지 못한 사항은
[재현 가능성 차이 및 처리 방침](01-reproducibility-gaps.md)에 등록한다.

## 2. 판단 근거의 우선순위

논문과 공개 코드가 충돌할 때는 다음 순서로 판단한다.

1. 출판된 논문의 알고리즘, 표, 그림 및 본문
2. Telecom Italia 데이터의 공식 설명과 실제 파일 구조
3. 원저자 공개 코드
4. 이 저장소에서 명시하고 검증한 구현 가정

하위 근거로 상위 근거를 조용히 덮어쓰지 않는다. 논문만으로 구현할 수 없는
부분은 추정값을 숨기지 않고 설정과 결과에 `approximate`로 표시한다.

## 3. 재현 목표

### 3.1 1차 필수 범위

- 2013년 11월 Milan 통신 데이터의 검증 및 전처리
- 10,000개 셀, 10분 간격, 30일 행렬 생성
- 중앙 900셀 실험 데이터 생성
- 논문의 Urbanflow Peak Clustering(UPC) 구현
- 논문의 Residual Convolutional TCN-LSTM(RCTL) 구현
- 학습이 없는 기준선과 LSTM 기준선 구현
- UPC 적용 여부와 RCC 적용 여부의 통제 비교
- 로컬 전처리 및 Google Colab GPU 학습 절차 문서화

### 3.2 1차 실험에서 제외하는 범위

- Transformer, MAMBA, GASTN의 완전 재현
- 논문의 모든 클러스터 수와 sequence length 탐색
- Attention 구조 추가 실험
- 10,000셀에서 모든 모델과 seed를 반복하는 대규모 비교
- 논문과 동일한 99% 신뢰구간 주장
- Intel Arc GPU 또는 NPU를 위한 별도 TensorFlow 환경 구축
- RAN Intelligent Controller 연동

제외 항목은 핵심 파이프라인이 검증된 뒤 별도 실험으로 추가할 수 있다.

## 4. 재현 수준

| 수준 | 의미 | 이번 프로젝트의 적용 |
|---|---|---|
| 구조 재현 | 논문의 데이터 흐름과 알고리즘 구조를 구현 | 필수 |
| 설정 재현 | 공개된 하이퍼파라미터를 적용 | 필수 |
| 수치 재현 | 논문 표의 수치와 오차 범위 내 일치 | 참고 목표 |
| 확장 재현 | 10,000셀 및 추가 모델 비교 | 선택 |

논문 수치와 일치하지 않아도 원인을 추적하고 구현 선택과 실행 환경을 공개하면
프로젝트는 실패로 보지 않는다. 설명되지 않은 임의 조정으로 수치만 맞추는 것을
금지한다.

## 5. 예측 문제 정의

각 셀의 과거 8개 10분 시점으로 바로 다음 10분의 Internet traffic을 예측한다.

```text
입력:  X[cell, t-8:t]
정답:  y[cell, t]
형상:  X=(8, 1), y=(1,)
예측:  one-step ahead, 10분 후
```

전체 시계열에 window를 먼저 정의한 뒤 **정답 시각**을 기준으로 split을 배정한다.
따라서 validation과 test의 첫 시점은 직전 구간에서 관측된 과거 8개 값을 입력으로
사용할 수 있지만, 입력에 미래 값은 들어가지 않는다. 길이 4,320의 시계열에서
다음 시점 예측 표본 수는 셀당 `4320 - 8 = 4312`개다.

## 6. 데이터 계약

### 6.1 원본 범위

- 파일: `sms-call-internet-mi-2013-11-01.txt`부터
  `sms-call-internet-mi-2013-11-30.txt`까지 30개
- 셀: ID 1부터 10,000까지
- 시간 간격: 600,000ms, 즉 10분
- 시간대: `Europe/Rome`
- 목표 열: `internet`

원본 TSV는 header가 없으며 다음 8개 열로 해석한다.

| 순서 | 이름 | 사용 |
|---:|---|---|
| 1 | `cell_id` | 셀 식별자 |
| 2 | `timestamp_ms` | 시점 식별자 |
| 3 | `country_code` | 집계 차원 |
| 4 | `sms_in` | 이번 재현에서는 미사용 |
| 5 | `sms_out` | 이번 재현에서는 미사용 |
| 6 | `call_in` | 이번 재현에서는 미사용 |
| 7 | `call_out` | 이번 재현에서는 미사용 |
| 8 | `internet` | 예측 대상 |

### 6.2 집계와 결측 처리

1. 활동값의 공란은 해당 행에서 관측된 활동이 없는 것으로 보고 0으로 변환한다.
2. 국가코드별 행은 `(cell_id, timestamp_ms)` 기준으로 `internet`을 합산한다.
3. 10,000개 셀과 4,320개 시점의 완전한 곱으로 reindex한다.
4. 원본에 없는 셀-시점 조합도 0으로 채운다.
5. 학습 행렬은 `float32`, 셀 ID와 timestamp는 정수형으로 저장한다.
6. `missing_mask`는 `(cell_id, timestamp_ms)`에 원본 행이 하나도 없을 때만
   `True`로 저장한다.
7. `internet_null_mask`는 원본 행은 있지만 모든 `internet` 값이 공란일 때만
   `True`로 저장한다. 두 mask는 서로 겹치지 않는다.
8. 활동 열별 원본 공란 행 수와 두 결측 유형의 개수를 manifest에 남긴다.

핵심 집계는 다음 수식으로 정의한다.

```text
traffic[cell, time] = sum(fill_null(internet, 0) for each country_code)
```

### 6.3 전처리 산출물

```text
data/interim/internet_10min.parquet
data/processed/traffic.npy
data/processed/cell_ids.npy
data/processed/timestamps_ms.npy
data/processed/missing_mask.npy
data/processed/internet_null_mask.npy
data/processed/manifest.json
```

실제 산출물은 Git에 포함하지 않는다. `manifest.json`에는 원본 파일명, 크기,
MD5, 행 수, 변환 설정, 출력 shape와 checksum을 기록한다. 향후 공식 checksum
목록처럼 코드 검증에 필요한 작은 metadata는 별도의 추적 가능한 경로에 둔다.

### 6.4 전처리 통과 조건

- shape가 정확히 `(10000, 4320)`임
- 셀 ID가 중복 없이 10,000개임
- timestamp가 중복 없이 4,320개임
- 인접 timestamp 차이가 모두 600,000ms임
- NaN 및 무한대가 없음
- 모든 traffic 값이 0 이상임
- `missing_mask`와 `internet_null_mask`가 서로 겹치지 않음
- 같은 원본과 설정으로 실행했을 때 출력 checksum이 같음

이 조건을 통과하기 전에는 UPC 또는 모델 학습을 시작하지 않는다.

## 7. 시간 분할 계약

### 7.1 주 결과: 엄격한 평가 프로토콜

| 구간 | 날짜 | 용도 |
|---|---|---|
| Train | 11월 1~20일 | 파라미터 학습 및 scaler 적합 |
| Validation | 11월 21~25일 | early stopping 및 모델 선택 |
| Test | 11월 26~30일 | 최종 성능 보고 |

주 결과와 결론은 test 구간만으로 작성한다.

### 7.2 논문 비교용 보조 프로토콜

논문은 앞 20일을 학습에 사용하고 나머지 10일을 validation과 testing에
사용했다고 설명하지만 두 구간의 경계를 공개하지 않았다. 따라서 11월 21~30일
전체 지표도 논문 비교용으로 출력하되 validation이 포함된 **보조 지표**로만
표시한다. 주 test 지표와 섞어서 보고하지 않는다.

## 8. 중앙 900셀 계약

Milano Grid 데이터(<https://doi.org/10.7910/DVN/QJWLFU>)의 각 셀 centroid를
이용해 100x100 격자 인덱스를 복원한다. x 좌표는 오름차순, y 좌표는 일관된
방향으로 정렬하고 기하학적 중앙의 30x30 셀을 선택한다.

- 0-based 행 범위: `[35, 65)`
- 0-based 열 범위: `[35, 65)`
- 기대 셀 수: 900
- 결과 ID 파일: `data/processed/central_900.csv`

논문이 정확한 ID 목록을 공개하지 않았으므로 이 범위의 결과는
`central-900-approximate`로 표시한다. 추후 원저자 또는 선행 연구의 정확한
목록을 확인하면 기존 목록을 덮어쓰지 않고 별도 프로토콜로 추가한다.

구현과 실제 검증 결과는
[중앙 900셀 공간 선택과 검증](04-central-900-selection.md)에 기록한다. 좌표로 복원한
100×100 위치와 cell ID 공식이 전체 10,000개에서 일치한 뒤에만 `(900, 4320)`
traffic 부분집합을 게시한다.

## 9. UPC 계약

### 9.1 24개 초기 그룹

1. 10분 traffic 6개를 합산해 셀별 1시간 traffic을 만든다.
2. `Europe/Rome` 기준 월요일부터 금요일만 사용한다.
3. 각 셀과 날짜에서 traffic이 최대인 시간을 peak hour로 선택한다.
4. 날짜별 peak hour 중 가장 자주 등장한 시간을 셀의 대표 peak hour로 정한다.
5. 대표 peak hour 0~23에 따라 24개 그룹을 만든다.

동률이면 가장 이른 hour를 선택한다. 이는 일반적인 첫 `argmax` 동작을 명시적으로
고정한 것이다.

논문 전체 30일을 사용한 초기 그룹의 검증 기준은 다음과 같다.

```text
[17, 4, 0, 0, 0, 0, 1, 3, 940, 207, 133, 208,
 383, 1306, 283, 321, 644, 681, 2155, 603, 850, 1114, 129, 18]
```

합계는 10,000이어야 한다. 이 분포와 다르면 시간대, 시간 집계, scaling,
결측 처리 순으로 원인을 확인한다.

실제 구현 결과, 위 Algorithm 1 해석은 Fig. 4와 L1 차이 `1,180`이었고,
11월 4~29일의 완전한 4주를 사용해 시간별 평균 profile의 최대값을 고르는 미공개
가설은 L1 차이 `4`였다. 주 계약을 결과에 맞춰 바꾸지 않고 두 경로를 분리한다.
근거와 전체 시간대별 비교는
[UPC 24개 초기 그룹 생성과 논문 지문 검증](05-upc-initial-groups.md)에 기록한다.

### 9.2 PCC와 최종 클러스터

- UPC 입력은 셀별 min-max scaling을 적용한다.
- scaler는 해당 UPC 프로토콜이 허용하는 기간에만 적합한다.
- 각 초기 그룹의 profile은 셀과 평일을 평균한 길이 24의 시간별 벡터로 정의한다.
- 크기가 `theta=10`보다 큰 그룹만 seed 후보로 사용한다.
- 그룹 간 profile의 Pearson correlation coefficient(PCC)를 계산한다.
- PCC가 가장 낮은 두 그룹을 최초 seed로 선택한다.
- 나머지 그룹은 기존 cluster와의 평균 PCC가 가장 높은 곳에 결정론적으로 배정한다.
- 동률이면 hour ID가 작은 그룹 또는 cluster ID가 작은 쪽을 선택한다.
- 최종 cluster 수는 `N=2`다.

크기가 작아 seed에서 제외된 그룹도 profile을 계산하여 최종 배정에는 포함한다.
이 정책은 논문 의사코드에서 불명확한 부분에 대한 명시적 구현 가정이다.

남은 그룹의 반복 순서는 논문에 명시되지 않았으므로 group ID 오름차순을 주 구현,
내림차순을 사전 등록한 민감도로 사용한다. 실제 결과 `train_only`은 두 순서의
membership이 100% 같았지만 `algorithm1_full_month`는 label-swap 불변 기준
50.48%만 같았다. 구현과 결과는
[UPC PCC 기반 최종 2개 클러스터](09-upc-pcc-final-clusters.md)에 기록하며,
이 차이는 `GAP-UPC-07`로 추적한다.

### 9.3 두 가지 UPC 프로토콜

| 이름 | clustering에 사용하는 기간 | 용도 |
|---|---|---|
| `train_only` | Train 20일의 평일 | 주 결과, 정보 누수 방지 |
| `algorithm1_full_month` | 30일 전체 평일 | 논문 Algorithm 1 민감도 비교 |

주 성능 결과에는 `train_only`를 사용한다. 두 프로토콜의 초기 그룹 수,
최종 cluster 크기와 membership 일치율을 함께 기록한다.

Fig. 4와 가까운 완전한 4주 평균-profile 계산은 `figure4_probe`로 분리하며 모델
입력으로 사용하지 않는다. 사전 등록한 제한 감사 결과 정확 일치는 없었지만, 이
불일치는 독립 기준선 진행을 막지 않는다. 결정 근거는
[UPC Fig. 4 불일치 제한 감사](06-upc-fig4-bounded-audit.md)에 기록한다.

### 9.4 프로토콜별 학습 허용 정책

PCC 순서 민감도 검사 후 clustering manifest의 보수적인 전역 학습 gate는 그대로
보존하고, 후속 모델 학습 허용 여부를 별도 정책으로 관리한다.

| 프로토콜 | 학습 허용 | 근거 |
|---|---|---|
| `train_only` | 예 | 누수 없는 주 프로토콜이며 오름·내림 membership 100% 일치 |
| `algorithm1_full_month` | 아니요 | 순서 일치율 50.48%, clustering 민감도로만 보존 |
| Fig. 4 probe | 아니요 | 미공개 진단 가설로 모델 입력 금지 |

이 결정은 PCC 결과 확인 후 모델 결과 확인 전에 내린
`post_clustering_pre_model` 범위 결정이다. 모델 성능이나 cluster 균형으로
membership을 선택하지 않는다. 설정, 자동 검증과 근거는
[UPC 순서 민감도 검토와 프로토콜별 학습 정책](10-upc-order-training-policy.md)에
기록한다.

## 10. 모델 계약

### 10.1 공통 학습 설정

| 항목 | 값 |
|---|---:|
| 입력 길이 | 8 |
| Batch size | 512 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | MAE |
| Dropout | 0.05 |
| 최대 epoch | 1,000 |
| Early stopping patience | 5 |
| Cluster 수 | 2 |

early stopping은 validation MAE를 감시하고 최적 가중치를 복원한다. 논문에는
patience가 공개되지 않았으므로 5를 구현 가정으로 사용하며 config에 노출한다.
단, 구조와 pipeline만 확인하는 smoke는 별도 config에서 고정 5 epoch와 checkpoint
미보존을 사용하며 본 학습 성능으로 보고하지 않는다.

저장된 원본 행렬과 첫 paper-oriented smoke는 raw traffic을 사용했다. 후속 제한
pilot은 독립 config와 결과 경로에서 Train 20일 전체에만 적합한 셀별 Min-Max를
시험했고, 사전 등록한 20% Validation MAE 개선 문턱을 통과했다. 따라서 본 학습은
이 scaling을 후보 계약으로 사용하되 raw 결과를 대체하지 않는다. 근거는
[LSTM Train-only 셀별 Min-Max scaling pilot](12-lstm-train-only-scaling-pilot.md)에
기록한다.

### 10.2 비교 모델

| 모델 | 목적 |
|---|---|
| Persistence | 직전 10분 값을 사용하는 최저 기준선 |
| Daily seasonal naive | 144시점 전 값을 사용하는 일간 계절성 기준선 |
| LSTM | 일반 순환 모델 기준선 |
| RCTL without UPC | RCTL 구조 자체 평가 |
| LSTM + UPC | clustering의 모델 독립적 효과 확인 |
| RCTL + UPC | 핵심 GECOS 구성 |
| RCTL without RCC | residual connection 최소 ablation |

### 10.3 RCTL 구현 원칙

- channel 수는 `[16, 32, 64, 64, 32, 16]`을 사용한다.
- 논문 Fig. 2의 TCN-LSTM, RCC-1, RCC-2 연결을 주 구현으로 삼는다.
- dilated causal convolution이 미래 값을 보지 않는지 테스트한다.
- 논문의 parameter count `173,633`을 검증 목표로 사용한다.
- 공개 `main.py` 동작은 `public_reference` 변형으로 분리한다.
- parameter count가 맞지 않으면 수치에 맞춰 임의로 층을 바꾸지 않고 차이를 기록한다.

첫 구조 감사에서 논문 Fig. 2 해석형은 kernel 4, `2^i` dilation과
`Concatenate`를 사용해 `236,657`개, 공개 코드형은 kernel 3, `2*i` dilation과
`Add`를 사용해 `173,665`개 parameter로 확인됐다. 두 모델은 출력, residual shape,
causality와 gradient 검사를 모두 통과했지만 논문 값 `173,633`과는 일치하지 않았다.
결과를 본 뒤 구조를 바꾸지 않고 상세 근거와 Colab smoke 결과를
[RCTL 아키텍처 계약과 Colab T4 과적합 smoke](08-rctl-architecture-smoke.md)에 기록한다.

UPC를 사용할 때는 cluster마다 독립 모델을 학습하고, 해당 cluster의 셀 window를
하나의 표본 집합으로 합친다. 입력에 cell ID나 좌표를 추가하지 않는다.

### 10.4 LSTM 재구성 후보와 첫 pipeline smoke

논문 Table III의 LSTM `165,185` parameter를 만족하는
`64 → 128 → 64 → Dense(1)` stacked-LSTM을
`paper_parameter_reconstruction`으로 등록한다. 이는 원저자 구조로 확인된 구현이
아니며 layer별 unit과 출력 activation이 공개될 때까지 `GAP-LSTM-01`과
`GAP-LSTM-02`를 유지한다.

첫 pipeline smoke는 중앙 900셀 전체와 분할별 64개 결정적 target을 사용한다.
UPC 미적용 모델 하나와 `train_only` cluster별 모델 둘을 각각 고정 5 epoch 학습하고,
성능 우열이 아니라 정확한 parameter 수, 유한한 학습, Train MAE 감소와 900셀 예측
재결합을 필수 gate로 사용한다. 구현과 결과는
[중앙 900셀 LSTM·UPC Colab T4 pipeline smoke](11-lstm-upc-smoke.md)에 기록한다.

### 10.5 본 학습용 LSTM scaling 후보

첫 smoke의 raw-scale 과소학습 원인을 제한적으로 확인하기 위해 같은 900셀, 선택
target, 모델, seed와 5 epoch를 유지하고 입력·target scaling만 바꾼다. 셀별
최솟값과 범위는 Train 인덱스 `[0, 2880)` 전체에서만 적합하며 Validation/Test를
사용하지 않는다. 변환과 역변환에는 clipping을 적용하지 않고 지표는 원래 traffic
단위에서 계산한다.

Test를 bundle에서도 제외한 Colab T4 pilot에서 UPC off Validation micro MAE는 raw
`236.2853`에서 scaled `30.4308`로 `87.12%` 감소했다. 결과 전에 고정한 실질 개선
문턱 `189.0282`를 통과해 셀별 Train-only Min-Max를 본 학습 후보로 채택했다. 이
판정에 Persistence, UPC on/off 차이와 Test는 사용하지 않았다. 세부 계약과 두 번의
결정적 실행은
[LSTM Train-only 셀별 Min-Max scaling pilot](12-lstm-train-only-scaling-pilot.md)에
기록한다.

## 11. 평가 계약

### 11.1 필수 지표

- MAE: 원래 traffic 단위의 micro average
- MAPE ratio: `y > 0`인 항목만 사용한 비율
- MAPE percent: MAPE ratio에 100을 곱한 값
- WAPE: 전체 절대오차 합을 전체 실제값 합으로 나눈 값
- 셀별 MAE/MAPE 분포와 cell-macro average

MAPE를 계산할 때 제외된 `y=0` 표본 수와 비율을 결과에 함께 기록한다. 논문 표의
`0.1000`을 0.1%와 10% 중 어느 것으로 해석했는지 결과 표에서 단위로 구분한다.

### 11.2 반복 실행

- 신경망 seed: `42`, `43`, `44`
- Python, NumPy, TensorFlow seed를 모두 설정한다.
- 각 seed의 원시 결과를 보존하고 평균과 표준편차를 보고한다.
- 반복 수와 계산법이 확인되지 않은 논문의 99% 신뢰구간을 재현했다고 주장하지 않는다.

### 11.3 실행 metadata

모든 실행 결과에는 다음 값을 저장한다.

- Git commit SHA
- 데이터 manifest checksum
- config 파일과 config checksum
- seed
- Python, TensorFlow, Keras 버전
- CPU/GPU 이름과 메모리
- 학습 시작 및 종료 시각
- epoch 수와 epoch별 소요 시간
- 최적 validation epoch

## 12. 실험 순서와 중단 조건

1. 원본 manifest와 `(10000, 4320)` 전처리 검증
2. 중앙 900셀 목록 생성 및 지도 확인
3. 24개 UPC 초기 그룹과 논문 fingerprint 비교
4. Persistence와 daily seasonal naive 계산
   ([구현 및 실제 결과](07-naive-baselines.md))
5. 16셀 RCTL shape, causality 및 overfit smoke test
   ([구현 및 실제 결과](08-rctl-architecture-smoke.md))
6. UPC 초기 그룹을 PCC로 최종 `N=2` cluster에 병합
   ([구현 및 실제 결과](09-upc-pcc-final-clusters.md))
7. PCC 순차 배정의 미공개 순서에 대한 제한 설계 검토
   ([정책 결정 및 자동 검증](10-upc-order-training-policy.md))
8. 중앙 900셀 LSTM 학습·평가 smoke와 UPC on/off 재결합 검증
   ([구현 및 실제 결과](11-lstm-upc-smoke.md))
9. Train-only 모델 scaling의 제한 pilot과 본 학습 전처리 계약 확정
   ([구현 및 실제 결과](12-lstm-train-only-scaling-pilot.md))
10. 중앙 900셀 LSTM/RCTL, UPC on/off 3-seed 비교
11. RCC 최소 ablation
12. 최적 구성의 10,000셀 단일 확장 실험

다음 조건에서는 뒤 단계로 넘어가지 않는다.

- 데이터 shape, timestamp 또는 checksum이 비결정적임
- window에 미래 값이 포함되는 테스트가 실패함
- UPC 초기 그룹 합계가 10,000이 아님
- 학습 config가 UPC 정책에서 허용하지 않은 프로토콜을 요청함
- 보호한 UPC membership 또는 정책 checksum이 변경됨
- 모델이 작은 표본을 의도적으로 overfit하지 못함
- 실행 metadata가 누락됨

RCTL parameter 수가 논문 표와 다르다는 사실만으로 smoke를 중단하지는 않는다.
shape, causality, gradient와 과적합 통과 여부는 구현 건전성 검사이고, parameter
불일치는 별도의 재현 gap이기 때문이다. 단, 모델 이름에서 `paper_interpretation`과
`public_reference`를 생략한 성능 실험은 허용하지 않는다.

## 13. 완료 정의

다음 조건을 만족하면 1차 핵심 재현이 완료된 것으로 본다.

- 새 환경에서 원본 데이터로부터 학습 행렬을 재생성할 수 있음
- 모든 데이터 계약 검사가 자동화되어 있음
- UPC의 paper-faithful 분포를 논문과 비교할 수 있음
- 공개 코드와 논문 기준 RCTL의 차이를 설명할 수 있음
- 중앙 900셀에서 모든 필수 기준선과 GECOS 결과가 동일 split으로 생성됨
- seed별 결과와 실행 환경이 보존됨
- 논문 수치와의 차이가 gap ID 또는 실험 기록으로 추적됨
- 원시 데이터, 논문 PDF와 모델 checkpoint가 Git에 포함되지 않음

성공의 핵심은 특정 MAPE를 맞추는 것이 아니라, 다른 사람이 같은 입력과 설정으로
같은 결과를 만들고 논문과 다른 이유를 확인할 수 있게 하는 것이다.
