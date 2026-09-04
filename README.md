# GECOS 학습용 부분 재현

*Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*

> **프로젝트 상태: 학습용 부분 재현 완료**
>
> 이 저장소는 원저자의 공식 구현이나 논문 수치의 완전 재현본이 아니다. 원저자
> 저장소를 fork한 뒤, 공개 자료의 차이를 추적하고 데이터 누수 없는 실험 설계와
> 재현 가능한 기록 방법을 학습할 수 있도록 확장했다.
>
> 원본 데이터 검증부터 UPC 구성, RCTL 구조 감사와 중앙 900 cell의 LSTM
> Train·Validation 비교까지 수행했다. RCTL 전체 학습과 최종 Test 평가는 확인된
> 정보의 한계와 추가 학습 가치를 고려해 의도적으로 실행하지 않았다.

## 논문을 먼저 이해하기

### 논문이 해결하려는 문제

- 도시 mobile traffic은 상업·주거 등 지역 기능과 시간에 따라 달라, 기존 통합
  학습으로는 공간적 이질성과 장기 의존성을 충분히 반영하기 어렵다.
- K-means는 물리적 거리에 의존해 멀리 있지만 traffic이 유사한 지역을 놓치고,
  DTW는 이를 찾을 수 있지만 pairwise 비교 비용 때문에 대규모 적용이 어렵다.

### 논문의 핵심 아이디어 및 방법론

- GECOS는 peak traffic 시간과 PCC로 cell을 묶는 Urbanflow Peak
  Clustering(UPC)과 cluster별 traffic을 예측하는 Residual Convolutional
  TCN-LSTM(RCTL)으로 구성된다.
- 논문은 Telecom Italia Milan 10,000-cell 데이터의 최초 30일을 10분 간격으로
  사용했다. Keras·CUDA, i7 CPU, 32GB RAM과 RTX 3070 Ti 환경에서 최초 20일을
  Train으로 사용했으며, 나머지 10일의 Validation·Test 세부 경계는 공개하지 않았다.

### 논문이 보고한 주요 실험 결과

- 중앙 900 cell, cluster 수 2에서 GECOS는 MAPE **0.1000 ± 0.004**, MAE
  **29.8520 ± 0.348**을 보고해 Transformer, MAMBA와 GASTN보다 낮은 오차를 보였다.
- UPC 적용 시 MAPE는 LSTM 약 53%, MLP 28%, Transformer 14%, MAMBA 21%
  개선됐으며, cluster를 더 늘리는 것보다 2개로 나눌 때 가장 효과적이었다.
- 10,000 cell의 UPC on/off MAPE는 **0.1186/0.6782**, MAE는
  **7.0632/7.1708**이고 추론 시간은 900 cell의 0.163초에서 0.178초로 소폭
  증가했다. 본문의 “약 42% MAPE 개선”은 Table IV 수치와 일치하지 않는다.

> 위 내용은 **논문 원문이 보고한 결과**다. 아래부터는 이 저장소에서 직접 구현하고
> 확인한 범위이며, 논문의 Table II·IV를 재현했다고 해석하지 않는다.

## 논문 재현으로 이해한 핵심

### 1. 왜 예측 전에 지역을 clustering하는가?

상업·주거처럼 기능이 비슷한 지역은 서로 멀리 떨어져 있어도 유사한 시간대에 traffic이
증가한다. GECOS는 서로 다른 분포를 하나의 model에 섞는 대신 유사한 cell끼리 묶고,
각 cluster가 더 균질한 pattern을 학습하게 한다.

### 2. UPC는 왜 peak hour와 PCC를 사용하는가?

UPC는 cell마다 평일에 가장 반복적으로 나타난 peak hour를 찾아 24개 초기 group을
만든다. 이후 각 group의 24시간 traffic profile 사이 PCC를 계산해 유사한 group을
병합하므로 지리적 거리 없이 활동 시간대가 비슷한 지역을 찾을 수 있다.

재현 과정에서는 기간과 peak 집계 방식이 조금만 달라도 초기 group이 크게 바뀌며,
논문의 Algorithm 1을 그대로 해석한 결과가 Fig. 4와 일치하지 않는다는 한계도
확인했다. 자세한 과정은 [UPC 초기 그룹](docs/05-upc-initial-groups.md)과
[Fig. 4 제한 감사](docs/06-upc-fig4-bounded-audit.md)에 기록했다.

### 3. UPC와 RCTL은 어떻게 하나의 GECOS를 구성하는가?

UPC는 학습 대상을 유사한 cell 집합으로 나누고, 각 cluster에는 별도의 RCTL을
할당한다. 즉 GECOS는 clustering만을 뜻하지 않으며, **UPC와 cluster별 RCTL을
결합한 전체 framework**다. cluster 수가 늘면 더 세밀하게 학습할 수 있지만 model
수와 계산 비용도 함께 증가하며, 논문 실험에서는 2개 cluster가 가장 효과적이었다.

### 4. RCTL은 기존 TCN-LSTM을 무엇 때문에 확장했는가?

논문은 깊은 TCN-LSTM에서 feature와 gradient가 약해지는 문제를 두 residual
connection으로 보완한다. RCC-1은 block 사이의 temporal feature 전달을, RCC-2는
같은 channel 사이의 spatial·channel feature 보존을 담당하며 논문 Fig. 11은 둘을
함께 사용한 RCTL의 가장 낮은 MAPE를 보고한다.

다만 논문 그림의 Concatenate, 공개 코드의 Add, kernel과 dilation 규칙이 서로
달랐다. 이 저장소는 두 해석의 계산 건전성만 [RCTL 구조 감사](docs/08-rctl-architecture-smoke.md)로
확인했으며, 어느 구조가 저자의 최종 실험 model인지는 확정하지 않았다.

### 5. 실험 결과는 clustering 효과를 어떻게 보여주는가?

논문에서는 UPC가 LSTM·MLP·Transformer·MAMBA의 MAPE를 모두 낮췄다. 이 저장소의
통제된 LSTM Validation 비교에서도 UPC on이 세 seed 모두 off보다 낮았지만, 상대
개선은 MAPE 약 6.26%에 비해 MAE·WAPE가 약 0.38%로 작았다.

따라서 clustering 효과는 하나의 metric만으로 판단할 수 없다. 작은 양수 target에
민감한 MAPE와 원래 traffic 단위의 MAE, 전체 오차 비율인 WAPE를 함께 봐야 한다.
직접 확인한 수치와 조건은 [LSTM 전체 학습](docs/13-lstm-full-training.md)에 있다.

### 6. 재현 후에도 확정할 수 없는 것은 무엇인가?

저자의 정확한 중앙 900 cell, Fig. 4 생성 규칙과 RCTL 내부 연결은 공개 정보만으로
확정할 수 없었다. Table IV의 수치와 본문 개선율도 일치하지 않으므로, 이 저장소는
논문 수치를 맞추기보다 **어떤 주장을 근거 있게 할 수 있는지 구분하는 것**을
학습 결과로 삼았다. 전체 gap과 종료 판단은
[재현 가능성 차이 및 처리 방침](docs/01-reproducibility-gaps.md)과
[학습용 논문 재현 최종 정리](docs/14-study-reproduction-conclusion.md)에 남겼다.

## 실험을 따라간 흐름

각 단계의 출력과 판단을 다음 단계의 입력 계약으로 고정했다.

~~~text
공식 원본 30개 파일 검증
→ 10,000 cell × 4,320 시점 traffic·결측 mask 생성
→ 공식 grid에서 중앙 30 × 30, 900 cell 근사 선택
→ target 시각 기준 20일/5일/5일 분할과 Train-only scaling
→ Train 평일 peak hour와 PCC로 UPC 2개 cluster 구성
→ Persistence·daily seasonal naive 기준선 평가
→ RCTL 두 구조의 smoke 및 LSTM pipeline·scaling 검증
→ LSTM UPC off/on 9개 job의 Train·Validation 비교
→ 주장 가능한 결과와 미실행 범위를 문서화
~~~

Fig. 4에 가까워지는 규칙이나 Validation 성능을 확인한 뒤 입력 계약을 바꾸지 않았다.
주요 결정은 결과를 보기 전에 config에 기록하고, 실제 출력에는 source commit과
입력·설정 checksum을 남겼다.

## 이 저장소에서 직접 확인한 결과

공통 예측 문제는 각 cell의 과거 8개 10분 시점으로 바로 다음 10분을 예측하는
것이다. 중앙 900 cell에서 Train 20일로 적합한 cell별 Min-Max scaler를 사용하고,
Validation 5일의 720개 target으로 early stopping을 수행했다.

LSTM은 논문 Table III의 **165,185 parameter**와 일치하도록 재구성한 후보이며
저자의 정확한 LSTM 구조로 확인된 것은 아니다. seed 42/43/44에서 UPC off 모델 3개와
cluster별 UPC on 모델 6개, 총 9개 job을 Colab T4에서 학습했다. 다음 값은
720 target × 900 cell의 Validation micro 평균 ± 표본표준편차다.

| 모델 | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| Persistence | 31.9310 | 12.9357% | 0.115476 |
| LSTM UPC off | 28.3164 ± 0.0610 | 11.4516% ± 0.1371%p | 0.102404 ± 0.000221 |
| LSTM UPC on | 28.2098 ± 0.0407 | 10.7346% ± 0.0377%p | 0.102019 ± 0.000147 |

- UPC on은 세 seed에서 모두 off보다 낮았다. 상대 개선은 MAE·WAPE 약 **0.38%**,
  MAPE 약 **6.26%**였다.
- 작은 양수 target에 민감한 MAPE만 보면 clustering 효과가 크게 보일 수 있다.
  MAE와 WAPE의 변화가 작다는 점까지 함께 해석했다.
- 이 결과는 Validation 비교이며, UPC의 일반화 성능이나 논문 성능표 재현을
  증명하지 않는다.

## 어디까지 재현했는가

| 영역 | 상태 | 확인한 내용과 경계 |
|---|---|---|
| 원본 데이터·전처리 | **재현 완료** | 30개 공식 파일과 **10,000 × 4,320** 행렬, 결측 mask를 검증했다. |
| 중앙 900 cell | **근사 재현** | 저자의 ID 목록이 없어 공식 grid의 기하학적 중앙 30 × 30을 사용하고 **central-900-approximate**로 표시했다. |
| UPC | **구현·진단 완료** | Train-only 초기 그룹과 PCC 기반 2개 cluster를 구성했다. |
| Fig. 4 | **불일치 기록** | 사전 등록한 10개 변형을 제한 감사했지만 같은 그룹 수를 만들지 못했고 원인을 확정하지 않았다. |
| 기준선 | **재현 완료** | 같은 split과 평가 계약에서 Persistence와 daily seasonal naive를 계산했다. |
| RCTL | **구조 진단 완료** | 논문형은 236,657개, 공개 코드형은 173,665개 parameter였다. 둘 다 Table III의 173,633개와 다르지만 shape·causality·gradient와 작은 표본 overfit 검사는 통과했다. |
| LSTM UPC off/on | **Train·Validation 완료** | 중앙 900 cell, seed 3개와 동일 split·scaling으로 비교했다. |
| 최종 Test·RCTL 전체 학습·RCC ablation·10,000 cell 확장 | **의도적으로 미실행** | Test 노출 이력, RCTL 구조 불확실성과 학습 가치 대비 비용을 고려해 중단했다. |

따라서 이 저장소는 **데이터 처리, UPC, 평가 계약과 모델 재구성을 학습한 부분
재현**이다. 논문의 전체 GECOS 성능, 99% confidence interval 또는 Table II·IV의
수치를 재현했다고 주장하지 않는다.

## 재현 과정에서 지킨 원칙

- 구현 근거는 출판된 논문 → 공식 데이터 설명과 실제 구조 → 원저자 공개 코드 →
  명시적으로 검증한 구현 가정 순으로 판단했다. 충돌을 조용히 섞지 않고 서로 다른
  변형과 gap으로 기록했다.
- clustering, scaling과 model 선택은 각각 허용된 Train·Validation 범위에서만
  수행했다. target 시각 기준 split과 고정 membership으로 미래 정보 누수를 막았다.
- smoke는 shape·gradient·pipeline을 검사하는 진단으로만 사용하고, 성능 주장은
  전체 Validation 학습 결과로 제한했다.
- 원시 데이터와 checkpoint를 Git에 넣지 않는 대신 source commit, config, 입력,
  cluster membership, job descriptor와 산출물 checksum을 연결했다.

## 자원 사용과 안전한 공유

로컬 노트북의 32GB RAM 전체를 실험 전용으로 가정하지 않았다. 전체 window를 미리
복제하지 않는 compact 입력을 사용해 LSTM 전체 학습 준비의 로컬 peak RSS를 약
144.7MiB로 제한했고, GPU 학습은 별도의 Colab Tesla T4 환경으로 분리했다.

- 로컬 데이터 환경: Python 3.12, NumPy 2.5.2 계열
- 모델 환경: Colab Tesla T4 15,360MiB, Python 3.13.15, NumPy 2.1.3,
  TensorFlow 2.20.0, Keras 3.13.2
- 9개 job 순수 fit 시간 합: 약 65.0분
- Colab에서 관측한 peak RSS 최댓값: 약 2.17GiB

Git에는 config, metadata, code, tests와 문서만 보존한다. 원시 데이터, ZIP, 논문
PDF, 전처리 결과, prediction과 model checkpoint는 [.gitignore](.gitignore)로
제외하고 checksum으로 provenance를 남긴다. 데이터 배치와 산출물 경로는
[데이터 안내](data/README.md)에서 확인할 수 있다.

## 이 저장소로 학습하는 방법

처음부터 모든 모델을 다시 학습할 필요는 없다. 관심 있는 학습 주제에 따라 문서를
선택한다.

| 학습 목적 | 먼저 읽을 문서 |
|---|---|
| 전체 결론과 판단 기준 | [학습용 논문 재현 최종 정리](docs/14-study-reproduction-conclusion.md), [핵심 재현 범위와 실험 계약](docs/00-reproduction-scope.md), [재현 가능성 차이 및 처리 방침](docs/01-reproducibility-gaps.md) |
| 원본 데이터와 누수 방지 | [원본 데이터 무결성](docs/02-raw-data-integrity.md), [Internet traffic 전처리](docs/03-internet-preprocessing.md), [중앙 900 cell](docs/04-central-900-selection.md), [예측·평가 계약](docs/07-naive-baselines.md) |
| UPC와 Fig. 4 불일치 | [UPC 초기 그룹](docs/05-upc-initial-groups.md), [Fig. 4 제한 감사](docs/06-upc-fig4-bounded-audit.md), [PCC 최종 cluster](docs/09-upc-pcc-final-clusters.md), [학습 정책](docs/10-upc-order-training-policy.md) |
| 모델 구조와 학습 실험 | [RCTL 구조 감사](docs/08-rctl-architecture-smoke.md), [LSTM pipeline smoke](docs/11-lstm-upc-smoke.md), [scaling pilot](docs/12-lstm-train-only-scaling-pilot.md), [LSTM 전체 학습](docs/13-lstm-full-training.md) |

저장소의 주요 역할은 다음과 같다.

~~~text
configs/       데이터·clustering·모델 실험 계약
metadata/      공식 파일 크기·checksum과 출처
scripts/       검증, 전처리, UPC, 기준선과 모델 pipeline
tests/         계약 변경과 데이터 누수를 막는 회귀 테스트
docs/          단계별 의사결정, 실행 결과와 최종 회고
data/README.md Git에 없는 데이터와 산출물의 배치 안내
main.py        원저자 공개 코드 비교용 파일
~~~

로컬 데이터 파이프라인과 회귀 테스트는 다음과 같이 준비할 수 있다.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/preprocess.txt \
  -r requirements/spatial.txt \
  -r requirements/upc.txt
python -m unittest discover -s tests -v
~~~

TensorFlow가 없는 로컬 환경에서는 Colab용 모델 구조 테스트 2개가 skip된다. Colab
입력 준비와 실행 명령은 각 모델 실험 문서에 기록했다. 새 실험을 추가할 때는 기존
결과를 덮어쓰지 말고 결과를 보기 전에 별도 config와 성공·중단 조건을 먼저
commit한다.

## 선택적으로 다시 시작한다면

다음 중 하나가 생길 때 전체 재현을 다시 여는 것이 합리적이다.

- 원저자가 중앙 900 cell ID와 UPC 세부 규칙을 공개함
- RCTL layer별 shape, merge 방식과 parameter 계산 근거가 확인됨
- Test 노출 이력이 없는 새로운 기간 또는 별도 holdout을 확보함
- RCTL 구조 비교 자체가 새로운 학습 목표가 됨

그 경우에도 현재 결과를 덮어쓰지 않고 새 config와 결과 경로를 사용한다. 현재
release는 학습 과정과 후속 실험을 비교할 기준점으로 보존한다.

## 출처와 주의사항

- 논문: *Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*
  <https://doi.org/10.1109/TNSM.2025.3599168>
- 원저자 공개 코드: <https://github.com/Superint-Lab/GECOS>
- Telecom Italia Big Data Challenge 데이터: <https://doi.org/10.7910/DVN/EGZHFV>

데이터는 원출처의 ODbL 1.0 조건과 출처 표기를 따른다. 원저자 공개 저장소에는
소프트웨어 라이선스가 명시돼 있지 않으므로, 이 저장소가 원저자 코드 전체의
재배포 조건을 새로 부여한다고 해석해서는 안 된다.
