# 학습용 논문 재현 최종 정리

## 1. 최종 결론

이 프로젝트는 **논문의 완전한 수치 재현이 아니라, 재현 과정에서 필요한 판단과
실험 습관을 익히는 부분 재현**으로 마무리한다.

원본 데이터 검증부터 전처리, 중앙 900셀 근사 선택, Urbanflow Peak
Clustering(UPC), 공통 평가 계약, 기준선, RCTL 구조 감사와 LSTM 전체
Train·Validation 학습까지 수행했다. 이 범위만으로도 다음 학습 목표를 달성했다.

- 데이터 provenance와 checksum을 이용해 입력을 검증할 수 있다.
- target 시각 기준 split과 Train-only scaling으로 누수를 막을 수 있다.
- 논문·그림·표·공개 코드가 다를 때 확인된 사실과 구현 가정을 분리할 수 있다.
- smoke test와 성능 실험의 목적을 구분할 수 있다.
- UPC off/on을 같은 데이터·seed·평가 계약으로 비교할 수 있다.
- 결과가 기대와 달라도 사후에 조건을 맞추지 않고 gap으로 남길 수 있다.

반면 RCTL 전체 학습, RCC ablation, 10,000셀 확장과 최종 Test 평가는 수행하지
않았다. 따라서 “GECOS 전체 재현 완료” 또는 “논문 성능 재현”이라고 표현하지 않는다.

## 2. 현재 지점에서 종료한 이유

남은 작업은 계산량뿐 아니라 해석의 불확실성이 크다. 논문 Fig. 2 해석형 RCTL은
`236,657`개, 공개 코드형은 `173,665`개 parameter였고 둘 다 Table III의
`173,633`개와 일치하지 않았다. 이 상태에서 900셀 전체 RCTL을 여러 seed로 학습하면
논문 모델의 재현보다 임의로 선택한 구조의 성능 측정에 가까워진다.

최종 Test 평가는 학습 자체보다 싸지만, 앞선 raw pipeline smoke가 Test를 진단용으로
사용한 이력이 있다. 이후 scaling과 best epoch 선택에는 Test를 사용하지 않았어도
완전히 보지 않은 pristine holdout은 아니다. 현재 Validation 결과를 정직하게
보고하고 Test를 추가 선택의 근거로 사용하지 않는 편이 학습용 종료 목적에 맞는다.

| 후보 작업 | 추가 학습 가치 | 비용 또는 해석 문제 | 최종 결정 |
|---|---|---|---|
| 지금까지의 판단과 결과 문서화 | 높음 | 낮음 | 완료 |
| locked Test 평가 | 낮음~보통 | 과거 raw smoke의 Test 노출 | 미실행 |
| 중앙 900셀 RCTL 3-seed 학습 | 보통 | 구조 미확정, 높은 GPU 비용 | 미실행 |
| RCC ablation | 낮음 | 기준 RCTL이 확정되지 않음 | 미실행 |
| 10,000셀 확장 | 낮음 | 매우 큰 계산·저장 비용 | 미실행 |
| Fig. 4에 맞춘 추가 규칙 탐색 | 낮음 | 결과에 맞춘 사후 조정 위험 | 중단 |

“할 수 있는 실험을 모두 실행하는 것”보다 “어떤 주장을 할 수 있는지 구분하는 것”을
최종 학습 성과로 선택했다.

## 3. 단계별 진행 결과

| 단계 | 수행 내용 | 최종 상태 | 근거 문서 |
|---:|---|---|---|
| 1 | 원본 30개 파일의 크기·MD5·목록 검증 | 완료 | [원본 데이터 무결성](02-raw-data-integrity.md) |
| 2 | 10,000셀 × 4,320시점 Internet traffic와 mask 생성 | 완료 | [Internet traffic 전처리](03-internet-preprocessing.md) |
| 3 | 공식 grid에서 중앙 30×30, 900셀 선택 | 근사 재현 | [중앙 900셀 선택](04-central-900-selection.md) |
| 4 | 평일 peak hour 기반 UPC 초기 그룹 생성 | 구현 완료 | [UPC 초기 그룹](05-upc-initial-groups.md) |
| 5 | Fig. 4 불일치 원인의 제한 감사 | 내부 감사 완료 | [Fig. 4 제한 감사](06-upc-fig4-bounded-audit.md) |
| 6 | target 정렬, split, MAE·MAPE·WAPE와 두 기준선 | 완료 | [기준선](07-naive-baselines.md) |
| 7 | RCTL 두 해석의 shape·causality·gradient·overfit 검사 | 구조 감사 완료 | [RCTL smoke](08-rctl-architecture-smoke.md) |
| 8 | PCC 기반 UPC 2개 cluster와 순서 민감도 확인 | 완료 | [최종 cluster](09-upc-pcc-final-clusters.md), [학습 정책](10-upc-order-training-policy.md) |
| 9 | LSTM parameter·cluster 재결합 pipeline smoke | 완료 | [LSTM pipeline smoke](11-lstm-upc-smoke.md) |
| 10 | Train-only 셀별 Min-Max scaling 제한 실험 | 완료·채택 | [scaling pilot](12-lstm-train-only-scaling-pilot.md) |
| 11 | 중앙 900셀 LSTM UPC off/on 3-seed 학습 | Train·Validation 완료 | [LSTM 전체 학습](13-lstm-full-training.md) |

중앙 900셀은 논문의 정확한 셀 ID가 공개되지 않아 `central-900-approximate`로
표시한다. UPC는 논문 Algorithm 1을 우선해 구현했지만 같은 절차로 Fig. 4의 그룹
수를 만들지 못했다. 두 차이는 실패를 숨긴 것이 아니라 결과 해석의 경계다.

## 4. 최종 LSTM Validation 결과

공통 예측 문제는 각 셀의 과거 8개 10분 시점으로 바로 다음 10분을 예측하는 것이다.
Train 20일 `[0, 2880)`, Validation 5일 `[2880, 3600)`, Test 5일
`[3600, 4320)`로 나눴다. LSTM 전체 학습 bundle은 index 3,600 이후 값을 포함하지
않는다.

각 셀의 Min-Max scaler는 Train 20일에만 적합했다. LSTM 후보는 Table III의
`165,185` parameter를 정확히 만족하는
`LSTM(64) → LSTM(128) → LSTM(64) → Dense(1)`이다. 이는 원저자 구조로 확인된
구현이 아니라 parameter 표와 계산이 일치하는 재구성 후보다.

seed `42`, `43`, `44`에서 UPC off 모델 3개와 cluster별 UPC on 모델 6개, 총 9개
job을 Colab T4에서 학습했다. 다음 값은 720개 Validation target × 900셀의
`all_targets` micro 평균 ± 표본표준편차다.

| 모델 | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| Persistence | 31.9310 | 12.9357% | 0.115476 |
| LSTM UPC off | 28.3164 ± 0.0610 | 11.4516% ± 0.1371%p | 0.102404 ± 0.000221 |
| LSTM UPC on | 28.2098 ± 0.0407 | 10.7346% ± 0.0377%p | 0.102019 ± 0.000147 |

LSTM은 raw 5 epoch smoke의 심한 과소학습을 벗어나 Persistence보다 낮은
Validation 오차를 보였다. UPC on−off paired 차이는 MAE `-0.1066`, MAPE
`-0.7170%p`, WAPE `-0.000385`였다. UPC on의 상대 개선은 MAE·WAPE 약 `0.38%`,
MAPE 약 `6.26%`다.

세 seed에서 방향은 같았지만 seed 수가 적고 Validation을 early stopping에
사용했다. 따라서 “UPC가 일반화 성능을 확실히 높였다”고 주장하지 않는다. 특히
MAPE는 작은 양수 target의 상대오차에 민감하므로 MAE·WAPE와 함께 읽어야 한다.

## 5. 실험 과정에서 얻은 핵심 학습

### 5.1 논문, 표와 코드는 서로 다른 증거다

공개 `main.py`는 논문 UPC의 직접 구현이 아니며 입력으로 요구하는
`correlation_matrix_0721.csv`도 공개되지 않았다. RCTL도 그림의 `Concatenate`,
공개 코드의 `Add`, kernel과 dilation 규칙이 서로 달랐다. 출처의 우선순위를 정하고
서로 다른 변형을 같은 이름으로 섞지 않는 것이 먼저였다.

### 5.2 재현되지 않은 그림도 결과다

UPC의 peak-hour 정의, 시간대, 결측 제외, 합계·평균과 전체 월 사용 여부를 제한된
후보 안에서 감사했지만 Fig. 4의 그룹 수와 일치하지 않았다. 더 많은 조합을 결과에
맞춰 탐색하면 재현이 아니라 tuning이 된다. 불일치를 `GAP-UPC-06`으로 남긴 것이
이 단계의 올바른 결론이다.

### 5.3 scaling은 단순한 구현 세부사항이 아니다

raw 5 epoch LSTM은 과소학습했다. 다른 조건을 고정한 제한 실험에서 Train-only
셀별 Min-Max scaling을 적용하자 UPC off Validation MAE가 `236.2853`에서
`30.4308`로 감소했다. 이를 보고 임의로 여러 scaler를 탐색하지 않고, 미리 정한
20% 개선 문턱에 따라 하나의 본 학습 계약으로 채택했다.

### 5.4 metric 이름보다 단위와 집계가 중요하다

MAPE는 ratio와 percent를 구분하고 `y=0` 제외 수를 기록했다. 전체 원소를 합치는
micro와 셀별 비율을 평균하는 cell-macro도 함께 계산했다. 같은 예측이어도 분모와
평균 순서에 따라 WAPE와 MAPE의 해석이 달라진다.

### 5.5 재현성은 seed 고정보다 넓다

config, source commit, 입력 배열, cluster membership, job descriptor, checkpoint와
집계 결과에 SHA-256을 남겼다. best weights 복원과 cluster 예측의 원래 셀 순서
재결합도 배열 단위로 검증했다. 원시 데이터와 checkpoint를 Git에 넣지 않으면서도
어떤 입력과 코드로 결과가 만들어졌는지 추적할 수 있게 했다.

## 6. 자원 사용과 실행 환경

로컬 노트북의 32GB RAM 전체를 실험 전용으로 가정하지 않았다. 전체 window를
미리 복제하지 않는 compact 입력을 만들었고 LSTM 전체 학습 준비의 로컬 peak RSS는
약 144.7MiB였다. GPU 학습은 Colab Tesla T4로 분리했다.

- 로컬 데이터 환경: Python 3.12, NumPy 2.5.2 계열
- 모델 환경: Colab Tesla T4 15,360MiB, Python 3.13.15, NumPy 2.1.3,
  TensorFlow 2.20.0, Keras 3.13.2
- 9개 job 순수 fit 시간 합: 약 65.0분
- Colab 관측 peak RSS 최댓값: 약 2.17GiB

seed 44 cluster 0의 첫 실행은 학습 직후 Colab VM 소실로 결과를 회수하지 못했다.
설정을 바꾸지 않고 같은 commit·bundle·descriptor를 다시 실행했고 best epoch와
Validation loss가 같았다. 인프라 실패와 실험 조건 변경을 구분하는 기록도 재현성의
일부다.

## 7. 재현 가능성과 보존 범위

전체 LSTM Validation 집계의 다음 gate가 모두 통과했다.

- source commit과 입력·config·descriptor checksum 일치
- 9개 job, `165,185` parameter와 유한한 weights·prediction 확인
- callback best weights와 최종 저장 weights exact 일치
- 두 cluster 예측의 중앙 900셀 원래 순서 exact 재결합
- Test 배열·prediction·metric 부재
- 모델 성능을 pipeline 성공 조건으로 사용하지 않음

핵심 release manifest 파일 SHA-256은
`5cb81e6c20411369069f6b9821b1f22938bdf3c096f927d39a61fa07c839ee88`이다.
상태 이름은 `ready_for_locked_test_evaluation`이지만 이 프로젝트의 최종 결정은
**평가 준비 상태로 보존하고 Test를 실행하지 않는 것**이다. 미래에 이어서 작업할
사람이 같은 checkpoint를 사용할 수 있다는 뜻이지, Test 실행이 완료 조건이라는
뜻은 아니다.

원시 데이터, 논문 PDF, ZIP, 전처리 결과, Validation prediction과 checkpoint는
`.gitignore`로 보호하며 Git에는 넣지 않는다. Git에는 재생성 코드, 작은 config,
metadata, 테스트와 문서만 보존한다.

## 8. 완료하지 않은 항목과 주장할 수 없는 것

- 논문 저자가 사용한 정확한 중앙 900셀 ID
- Fig. 4와 같은 UPC 초기 그룹 분포
- 저자의 UPC 순차 배정 순서
- Table III와 내부 연결까지 모두 일치하는 RCTL
- RCTL UPC off/on 전체 Train·Validation 결과
- RCC 유무 ablation
- 10,000셀 전체 모델 비교
- 논문의 99% confidence interval 계산
- Table II와 Table IV의 수치 재현
- 완전히 보지 않은 Test에 대한 최종 일반화 성능

이 항목들은 버그를 숨기기 위해 제외한 것이 아니다. 공개 정보가 부족하거나 현재
학습 목표에 비해 비용이 큰 항목을 의도적으로 중단한 것이다. 자세한 근거는
[재현 가능성 차이 및 처리 방침](01-reproducibility-gaps.md)에 남아 있다.

## 9. 이 저장소를 학습에 활용하는 방법

처음부터 모든 모델을 다시 학습할 필요는 없다.

1. [핵심 재현 범위와 실험 계약](00-reproduction-scope.md)에서 데이터와 평가 경계를
   먼저 읽는다.
2. [재현 가능성 차이 및 처리 방침](01-reproducibility-gaps.md)에서 구현 가정이 생긴
   이유를 확인한다.
3. 데이터 학습이 목적이면 문서 02~07과 해당 테스트를 실행한다.
4. clustering 학습이 목적이면 문서 05, 06, 09, 10을 비교한다.
5. 모델 재구성이 목적이면 RCTL 문서 08과 LSTM 문서 11~13을 읽는다.
6. 새로운 실험을 추가할 때는 결과를 보기 전에 config와 성공·중단 조건을 먼저
   commit한다.

단계 문서의 “다음 단계”는 실험 당시의 시간 순서 기록이다. 현재 프로젝트의 종료
판정에는 이 문서를 우선한다.

## 10. 선택적으로 다시 시작한다면

다음 중 하나가 생길 때만 전체 재현을 다시 여는 것이 합리적이다.

- 원저자가 중앙 900셀 ID와 UPC의 세부 규칙을 공개함
- RCTL layer별 shape, merge 방식과 parameter 계산 근거가 확인됨
- Test 노출 이력이 없는 새로운 기간 또는 별도의 holdout을 확보함
- “학습용 부분 재현”이 아니라 RCTL 구조 비교 자체가 새로운 학습 목표가 됨

그 경우에도 현재 결과를 덮어쓰지 않고 새 config와 결과 경로를 사용한다. 현재
release는 비교 가능한 기준점으로 그대로 보존한다.

## 11. 최종 상태 선언

이 저장소는 다음 문장으로 마무리한다.

> GECOS의 데이터 처리, UPC, 평가 계약과 모델 재구성 과정을 학습했고, 중앙 900셀
> LSTM Train·Validation 비교까지 재현 가능한 형태로 완료했다. 공개 정보만으로
> 확정할 수 없는 RCTL 전체 성능과 논문 수치의 완전 재현은 주장하지 않는다.
