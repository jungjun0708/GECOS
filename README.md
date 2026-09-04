# GECOS 학습용 부분 재현

*Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*

> **상태: 학습용 부분 재현 완료**
>
> 이 저장소는 논문 수치를 끝까지 맞춘 완전 재현본이 아니다. 논문의 문제와 GECOS의
> 동작 원리를 이해하고, 공개 자료로 확인 가능한 범위에서 UPC의 효과를 통제된
> LSTM 실험으로 검토한 학습 기록이다. RCTL 전체 학습과 최종 Test 평가는 수행하지
> 않았으며, 확인하지 않은 결과를 완료로 표시하지 않는다.

## 논문을 이해하는 데 필요한 핵심

### 논문이 해결하려는 문제

도시의 mobile traffic은 상업·주거 등 지역 기능과 시간대에 따라 서로 다른 형태를
보인다. 모든 cell을 하나의 model로 학습하면 큰 traffic pattern이 평균을 지배하고,
서로 멀리 있지만 생활 pattern이 비슷한 지역의 공통점을 놓칠 수 있다.

K-means는 주로 물리적 거리에 의존하고, DTW는 시계열 유사도를 비교할 수 있지만
cell 쌍이 많아질수록 계산량이 빠르게 증가한다. 논문은 **예측 전에 traffic pattern이
비슷한 cell을 효율적으로 묶고, 각 집합에 특화된 model을 학습하는 문제로** 접근한다.

### GECOS의 핵심 구조

~~~text
cell별 mobile traffic
→ UPC: peak hour로 초기 group 생성
→ PCC: 하루 traffic profile이 비슷한 group 병합
→ cluster별 RCTL 학습
→ cell별 예측을 원래 공간 순서로 결합
~~~

**Urbanflow Peak Clustering(UPC)은** 각 cell에서 반복적으로 traffic이 가장 높은
시간을 활동 pattern의 요약값으로 사용한다. 먼저 peak hour가 같은 cell을 묶고,
24시간 profile 사이의 Pearson Correlation Coefficient(PCC)가 높은 group을
합친다. 지리적으로 떨어진 지역도 사용 시간대가 비슷하면 같은 cluster에 들어갈 수
있다.

**Residual Convolutional TCN-LSTM(RCTL)은** cluster 내부의 시공간 pattern을
예측한다. 논문은 TCN-LSTM에 RCC-1과 RCC-2를 추가해 깊은 network에서 temporal
feature와 spatial·channel feature가 약해지는 문제를 보완하려 한다.

따라서 GECOS는 UPC만을 뜻하지 않는다. **UPC가 서로 다른 분포를 분리하고, 각
cluster의 RCTL이 해당 분포를 학습하는 전체 framework다.** 이 저장소의 LSTM
UPC off/on 실험은 이 중 clustering의 추가 효과만 분리해 본 것이며, GECOS 전체
성능을 재현한 실험은 아니다.

### 논문이 보고한 결과

논문은 Telecom Italia Milan 10,000-cell 데이터의 최초 30일을 10분 간격으로
사용했다. 최초 20일을 Train으로 사용했지만, 나머지 10일의 Validation·Test 경계는
공개하지 않았다. 실행 환경은 Keras·CUDA, i7 CPU, 32GB RAM과 RTX 3070 Ti로
기술돼 있다.

| 논문에서 확인하려는 질문 | 보고된 결과 | 해석 |
|---|---:|---|
| 중앙 900 cell에서 GECOS가 비교 model보다 좋은가? | MAPE **0.1000 ± 0.004**, MAE **29.8520 ± 0.348** | Transformer, MAMBA, GASTN보다 낮은 오차를 보고했다. |
| UPC가 model 종류와 무관하게 도움이 되는가? | LSTM **53%**, MLP **28%**, Transformer **14%**, MAMBA **21%** MAPE 개선 | 논문은 UPC를 특정 예측 model에 종속되지 않는 전처리 framework로 해석한다. |
| cluster를 더 많이 나누면 계속 좋아지는가? | 2개 cluster에서 가장 좋은 결과 | 세분화의 이득과 model 수·표본 분산 사이에 trade-off가 있음을 보여준다. |
| 10,000 cell로 확장 가능한가? | UPC on/off MAPE **0.1186/0.6782**, MAE **7.0632/7.1708**, 추론 **0.178초** | 대규모에서도 오차 감소와 짧은 추론 시간을 보고했다. |

논문 본문의 “10,000 cell에서 약 42% MAPE 개선”이라는 설명은 Table IV의
**0.6782 → 0.1186과** 일치하지 않는다. 이 저장소는 서로 충돌하는 서술을 임의로
맞추지 않고 표의 수치와 본문을 따로 기록한다.

## 이 저장소에서 검증한 질문

완전한 RCTL 구조를 확정할 수 없는 상태에서 고비용 학습을 진행하는 대신 다음 질문에
집중했다.

> **미래 정보를 사용하지 않고 만든 UPC cluster가, 동일한 LSTM·split·scaling·seed
> 조건에서 Validation 예측을 실제로 개선하는가?**

| 실험 조건 | 적용한 계약 |
|---|---|
| 데이터 | 2013년 11월 Milan Internet traffic, **10,000 × 4,320** |
| 공간 범위 | 공식 grid의 기하학적 중앙 30 × 30, **근사 900 cell** |
| 예측 문제 | 과거 8개 10분 시점으로 다음 10분 한 시점 예측 |
| 시간 경계 | Train 20일 / Validation 5일 / 봉인한 Test 5일 |
| scaling | Train 20일에만 적합한 cell별 Min-Max |
| UPC | Train 기간의 평일만 사용, 중앙 900 cell을 611/289 cell로 분리 |
| 비교 | 같은 LSTM을 UPC off 1개와 UPC on cluster별 2개로 학습 |
| 반복 | seed 42·43·44, 총 9개 Colab T4 job |
| 모델 | Table III의 165,185 parameter와 맞춘 LSTM 재구성 후보 |

Validation은 early stopping과 model 선택에 사용했으며 Test는 최종 평가하지 않았다.
따라서 아래 결과는 일반화 성능이 아니라 **동일 조건에서 clustering의 추가 효과를
살펴본 Validation 비교다.** 자세한 입력·학습 계약은
[LSTM 전체 학습 문서](docs/13-lstm-full-training.md)에 있다.

## 직접 재현한 결과와 해석

다음 값은 720 target × 900 cell의 Validation **all_targets micro** 평균이며,
LSTM은 seed 3개의 평균 ± 표본표준편차다.

| 모델 | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| Persistence | 31.9310 | 12.9357% | 0.115476 |
| LSTM UPC off | 28.3164 ± 0.0610 | 11.4516% ± 0.1371%p | 0.102404 ± 0.000221 |
| LSTM UPC on | 28.2098 ± 0.0407 | 10.7346% ± 0.0377%p | 0.102019 ± 0.000147 |

### 1. 복잡한 model을 쓰기 전에 기준선을 넘었는가?

UPC off LSTM만으로도 Persistence보다 MAE와 WAPE가 약 **11.32%** 낮았고,
UPC on은 약 **11.65%** 낮았다. 이는 학습된 LSTM pipeline이 “직전 10분 값을
그대로 복사”하는 강한 단기 기준선보다 낮은 Validation 오차를 만들었다는 뜻이다.

동시에 전체 절대오차 개선의 대부분은 UPC를 적용하기 전부터 발생했다. 따라서
Persistence 대비 향상을 전부 clustering의 효과로 해석할 수 없다.

### 2. UPC가 추가로 만든 차이는 얼마나 큰가?

UPC on−off 차이는 MAE **-0.1066**, MAPE **-0.7170%p**, WAPE **-0.000385였다.**
상대 개선은 MAE·WAPE 약 **0.38%**, MAPE 약 **6.26%다.** 세 seed의 전체 집계에서
방향은 모두 같았지만, 절대 traffic 오차에 대한 추가 이득은 작고 비율 오차에서 더
크게 나타났다.

이 차이는 “전체 traffic 예측이 6.26% 좋아졌다”는 뜻이 아니다. MAPE는 실제값이
작은 양수 target의 오차를 크게 반영하므로 MAE와 WAPE를 함께 봐야 한다.

### 3. 어떤 cell에서 효과가 나타났을 가능성이 큰가?

모든 cell-target을 합친 micro WAPE 개선은 약 **0.38%였지만**, cell마다 같은
가중치를 주는 cell-macro WAPE는 **0.115408 → 0.111791**, 약 **3.13%** 개선됐다.
traffic 규모가 작은 cell도 동일한 비중을 가질 때 효과가 더 커졌다는 뜻이다.

이는 UPC의 이득이 traffic이 큰 cell 전체에 균일하게 나타났다기보다, 상대적으로
작거나 예측하기 어려운 일부 cell에서 더 컸을 가능성을 시사한다. 다만 원인을
확정하는 별도 ablation을 하지 않았으므로 “낮은 traffic이 원인”이라고 단정하지
않는다.

### 4. 논문의 UPC 효과를 재현했다고 볼 수 있는가?

방향은 논문의 주장과 일치하지만 효과 크기는 논문의 LSTM MAPE 약 53%보다 훨씬
작다. 이것을 곧바로 논문의 실패나 성공으로 판정할 수 없는 이유는 다음과 같다.

- 저자가 사용한 중앙 900 cell ID가 없어 기하학적 중앙을 근사했다.
- 논문의 Algorithm 1 해석으로 만든 peak-hour group이 Fig. 4와 일치하지 않았다.
- 누수를 막기 위해 전체 30일이 아니라 Train 20일만으로 UPC membership을 고정했다.
- LSTM은 parameter 표와 맞춘 재구성 후보이며 저자 구현으로 확인되지 않았다.
- 결과는 seed 3개의 Validation 비교이고 최종 Test 결과가 아니다.

따라서 이번 실험은 **누수 없이 구성한 UPC 변형이 LSTM에 작은 추가 이득을 줄 수
있다는 근거는** 제공하지만, 논문의 UPC 수치나 GECOS 전체 성능을 재현하지는 않는다.

### 결과에서 얻은 최종 학습

> 이번 재현에서 UPC는 전체 절대오차를 크게 줄이지는 않았지만, 서로 다른 traffic
> pattern을 나눈 뒤 별도 model을 학습하는 방식이 일부 cell의 상대오차를 줄일
> 가능성을 보여줬다. 핵심은 “UPC on의 숫자가 더 작다”가 아니라, **어떤 지표에서
> 얼마나 좋아졌으며 그 차이를 어디까지 일반화할 수 있는지 구분하는 것이다.**

## 불일치에서 배운 내용

| 확인한 불일치 | 재현 결과에 미치는 의미 |
|---|---|
| UPC 구현 결과와 논문 Fig. 4가 다름 | 이 저장소의 cluster는 논문의 정확한 cluster가 아니라 검증 가능한 **train_only** 변형이다. |
| 논문 그림은 Concatenate, 공개 코드는 Add를 사용 | 동일한 이름 아래 서로 다른 RCTL 구조가 존재하므로 한쪽을 저자 최종 model로 단정할 수 없다. |
| RCTL parameter가 논문형 236,657, 공개 코드형 173,665, Table III 173,633으로 다름 | shape·causality·gradient smoke만 수행하고 전체 성능 학습은 중단했다. |
| Table IV 수치와 본문의 개선율이 다름 | 하위 근거의 차이를 숨기지 않고 직접 계산값과 저자 서술을 분리한다. |

이 불일치는 단순한 구현 실패가 아니다. 논문, 표, 그림과 공개 코드가 다를 때
**확인된 사실과 구현 가정을 분리해야 결과를 과장하지 않을 수 있다는** 것이 이번
재현의 중요한 학습이다.

## 재현 범위

| 영역 | 상태 | 해석 경계 |
|---|---|---|
| 원본 데이터·전처리 | 완료 | 공식 30개 파일과 **10,000 × 4,320** traffic·결측 mask 검증 |
| 중앙 900 cell | 근사 재현 | 저자 ID가 없어 **central-900-approximate** 사용 |
| UPC | 구현·진단 완료 | Train-only 2개 cluster 사용, Fig. 4 불일치 유지 |
| LSTM UPC off/on | Train·Validation 완료 | 동일 조건의 clustering 효과만 비교 |
| RCTL | 구조 진단 완료 | 두 해석의 smoke만 수행, 저자 최종 구조 미확정 |
| 최종 Test·RCTL 전체 학습·RCC ablation·10,000 cell 학습 | 미실행 | 완료나 논문 성능 재현으로 표시하지 않음 |

## 세부 문서 안내

README는 논문 이해와 결과 해석만 요약한다. 실행 계약, 수치 산출 과정과 재현성
근거는 다음 문서에서 확인할 수 있다.

| 더 확인할 내용 | 문서 |
|---|---|
| 전체 결론·재현 범위·공개 자료 차이 | [최종 정리](docs/14-study-reproduction-conclusion.md), [재현 범위](docs/00-reproduction-scope.md), [재현 가능성 차이](docs/01-reproducibility-gaps.md) |
| 원본 데이터·전처리·중앙 900 cell·평가 계약 | [원본 무결성](docs/02-raw-data-integrity.md), [전처리](docs/03-internet-preprocessing.md), [중앙 900 cell](docs/04-central-900-selection.md), [기준선](docs/07-naive-baselines.md) |
| UPC 구성·Fig. 4 감사·PCC·학습 정책 | [초기 group](docs/05-upc-initial-groups.md), [Fig. 4 감사](docs/06-upc-fig4-bounded-audit.md), [최종 cluster](docs/09-upc-pcc-final-clusters.md), [학습 정책](docs/10-upc-order-training-policy.md) |
| RCTL·LSTM 구조와 학습 결과 | [RCTL 구조 감사](docs/08-rctl-architecture-smoke.md), [LSTM smoke](docs/11-lstm-upc-smoke.md), [scaling pilot](docs/12-lstm-train-only-scaling-pilot.md), [LSTM 전체 학습](docs/13-lstm-full-training.md) |

원시 데이터와 파생 산출물의 위치 및 재생성 방법은
[데이터 안내](data/README.md)에 있다. 원시 데이터, ZIP, 논문 PDF, prediction과
checkpoint는 Git에 포함하지 않고 config, metadata, code, tests와 checksum만
공유한다.

로컬에서 데이터 pipeline과 회귀 테스트를 확인하려면 다음 명령을 사용한다.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/preprocess.txt \
  -r requirements/spatial.txt \
  -r requirements/upc.txt
python -m unittest discover -s tests -v
~~~

## 출처와 주의사항

- 논문: *Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*
  <https://doi.org/10.1109/TNSM.2025.3599168>
- 원저자 공개 코드: <https://github.com/Superint-Lab/GECOS>
- Telecom Italia Big Data Challenge 데이터: <https://doi.org/10.7910/DVN/EGZHFV>

데이터는 원출처의 ODbL 1.0 조건과 출처 표기를 따른다. 원저자 공개 저장소에는
소프트웨어 라이선스가 명시돼 있지 않으므로, 이 저장소가 원저자 코드 전체의
재배포 조건을 새로 부여한다고 해석해서는 안 된다.
