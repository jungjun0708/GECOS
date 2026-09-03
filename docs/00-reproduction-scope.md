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
- 문서 상태: 구현 전 계약 v1

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
6. 원본 공란과 완전 누락을 구분할 수 있도록 missing mask와 통계를 남긴다.

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
- 결과 ID 파일: `central_900.csv`

논문이 정확한 ID 목록을 공개하지 않았으므로 이 범위의 결과는
`central-900-approximate`로 표시한다. 추후 원저자 또는 선행 연구의 정확한
목록을 확인하면 기존 목록을 덮어쓰지 않고 별도 프로토콜로 추가한다.

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

### 9.3 두 가지 UPC 프로토콜

| 이름 | clustering에 사용하는 기간 | 용도 |
|---|---|---|
| `upc_train_only` | Train 20일의 평일 | 주 결과, 정보 누수 방지 |
| `upc_paper_faithful` | 30일 전체 평일 | 논문 Fig. 4 비교 |

주 성능 결과에는 `upc_train_only`를 사용한다. 두 프로토콜의 초기 그룹 수,
최종 cluster 크기와 membership 일치율을 함께 기록한다.

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

저장된 원본 행렬과 첫 paper-oriented 학습은 raw traffic을 사용한다. 별도의
모델 입력 scaling을 시험할 경우 독립 config와 실험 이름을 사용하고, 원래 결과를
대체하지 않는다.

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

UPC를 사용할 때는 cluster마다 독립 모델을 학습하고, 해당 cluster의 셀 window를
하나의 표본 집합으로 합친다. 입력에 cell ID나 좌표를 추가하지 않는다.

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
5. 10~50셀 RCTL shape 및 overfit smoke test
6. 중앙 900셀 LSTM/RCTL, UPC on/off 비교
7. RCC 최소 ablation
8. 최적 구성의 10,000셀 단일 확장 실험

다음 조건에서는 뒤 단계로 넘어가지 않는다.

- 데이터 shape, timestamp 또는 checksum이 비결정적임
- window에 미래 값이 포함되는 테스트가 실패함
- UPC 초기 그룹 합계가 10,000이 아님
- 모델이 작은 표본을 의도적으로 overfit하지 못함
- 실행 metadata가 누락됨

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
