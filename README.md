# GECOS

*Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*

> **프로젝트 상태: 학습용 부분 재현 완료**
>
> 이 저장소는 원저자의 공식 구현이나 논문 수치의 완전 재현본이 아니다. 공개 자료의
> 차이를 추적하고, 데이터 누수 없는 실험 절차와 재현 가능한 기록을 학습하기 위해
> 원저자 저장소를 fork해 확장한 프로젝트다.

## 무엇을 공부했는가

이 프로젝트는 공개 코드를 단순 실행하는 대신 다음 질문을 따라 진행했다.

- 논문, 공개 코드와 실제 데이터가 서로 다를 때 무엇을 근거로 구현해야 하는가?
- 원본 데이터의 checksum, 시간축과 결측을 어떻게 검증해야 하는가?
- Urbanflow Peak Clustering(UPC)을 미래 정보 없이 구성할 수 있는가?
- MAE, MAPE와 WAPE를 어떤 단위와 집계 방식으로 비교해야 하는가?
- 모델 구조가 완전히 공개되지 않았을 때 확인된 사실과 구현 가정을 어떻게 분리하는가?
- Validation으로 선택한 모델과 Test 평가를 어떻게 분리하고 기록하는가?

결론과 학습 내용은
[학습용 논문 재현 최종 정리](docs/14-study-reproduction-conclusion.md)에 모았다.

## 최종 재현 범위

| 영역 | 상태 | 최종 범위 |
|---|---|---|
| 원본 데이터 검증 | 완료 | 2013년 11월 Milan 30개 파일의 크기와 MD5 검증 |
| Internet traffic 전처리 | 완료 | 10,000셀 × 4,320시점 행렬과 결측 mask 생성 |
| 중앙 900셀 | 근사 재현 | 공개되지 않은 셀 목록 대신 공간적으로 중앙인 30×30 격자 사용 |
| UPC | 구현·감사 완료 | Train-only 초기 그룹과 PCC 기반 2개 cluster 생성 |
| Fig. 4 | 불일치 기록 | 제한 감사를 마쳤지만 공개 정보만으로 동일한 그룹 수를 만들지 못함 |
| 기준선 | 완료 | Persistence와 daily seasonal naive를 공통 split에서 평가 |
| RCTL | 구조 감사 완료 | 두 해석의 shape, causality, gradient와 작은 표본 overfit 검증 |
| LSTM | 전체 Train·Validation 완료 | 중앙 900셀, seed 3개, UPC off/on 9개 job 실행 |
| 최종 Test·RCTL 전체 학습 | 의도적으로 미실행 | 학습 대비 추가 가치와 공개 정보의 한계를 고려해 종료 |

따라서 이 저장소의 결과는 **방법론과 실험 과정의 부분 재현**이다. 논문의 전체
GECOS 성능이나 Table II·IV의 수치를 재현했다고 주장하지 않는다.

## 핵심 Validation 결과

과거 8개 10분 시점으로 다음 10분을 예측했다. 각 셀의 Min-Max scaler는 Train
20일에만 적합했고, Validation 5일로 early stopping을 수행했다. 아래 값은 중앙
900셀의 720개 Validation target에 대한 seed `42/43/44` 평균 ± 표본표준편차다.

| 모델 | MAE | MAPE | WAPE |
|---|---:|---:|---:|
| Persistence | 31.9310 | 12.9357% | 0.115476 |
| LSTM UPC off | 28.3164 ± 0.0610 | 11.4516% ± 0.1371%p | 0.102404 ± 0.000221 |
| LSTM UPC on | 28.2098 ± 0.0407 | 10.7346% ± 0.0377%p | 0.102019 ± 0.000147 |

UPC on은 세 seed에서 모두 off보다 낮았지만 MAE·WAPE 상대 개선은 약 `0.38%`로
작았다. MAPE 상대 개선 약 `6.26%`만으로 clustering 효과를 과장하지 않는다.
이 수치는 **Validation 결과**이며 최종 Test 성능이 아니다.

## 가장 중요한 학습

1. **재현은 수치를 맞추는 작업보다 가정을 추적하는 작업에 가깝다.**
   UPC의 명시 알고리즘과 Fig. 4, RCTL 그림·표와 공개 코드는 각각 완전히 일치하지
   않았다. 결과에 맞춰 숨은 규칙을 조정하지 않고 gap으로 남겼다.

2. **전처리와 평가 계약이 모델보다 먼저다.**
   target 시각 기준 split, Train-only scaling, 결측 target 정책과 micro/cell-macro
   집계를 먼저 고정해 데이터 누수와 사후 선택을 방지했다.

3. **smoke test와 성능 실험의 목적은 다르다.**
   RCTL과 LSTM의 작은 smoke는 shape, gradient와 pipeline을 검사하는 데 사용했고,
   성능 주장은 전체 Validation 학습 결과에만 제한했다.

4. **여러 지표를 함께 봐야 한다.**
   작은 양수 target에 민감한 MAPE만 보면 UPC 효과가 크게 보이지만 MAE와 WAPE의
   변화는 작았다.

## 저장소 구성

```text
configs/       고정된 데이터·clustering·모델 실험 계약
data/          Git에 포함하지 않는 원본·중간·결과 경로와 사용 안내
docs/          단계별 의사결정, 실행 결과와 최종 회고
metadata/      공식 데이터 파일 크기·checksum과 출처
requirements/  로컬 전처리 환경과 Colab 모델 환경의 분리된 lock
scripts/       검증, 전처리, UPC, 기준선과 모델 pipeline
tests/         계약 변경과 데이터 누수를 막는 회귀 테스트
main.py        원저자 공개 코드 비교용 파일
```

원시 데이터, ZIP, 논문 PDF, 전처리 결과와 모델 checkpoint는 `.gitignore`로
제외한다. 데이터 위치와 생성 산출물은 [데이터 안내](data/README.md)를 참고한다.

## 실행 환경과 확인 방법

로컬 데이터 작업은 Python 3.12와 NumPy 2.5.2 계열 환경을 사용한다. TensorFlow
모델 작업은 로컬 노트북 자원을 점유하지 않도록 별도의 Google Colab T4 환경에서
Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2로 실행했다. 두 환경의 dependency를
하나의 가상환경에 섞지 않는다.

로컬 회귀 테스트 예시는 다음과 같다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/preprocess.txt \
  -r requirements/spatial.txt \
  -r requirements/upc.txt
python -m unittest discover -s tests -v
```

TensorFlow가 없는 로컬 환경에서는 Colab에서 수행하도록 등록된 모델 구조 테스트
2개가 skip된다. 단계별 데이터 준비와 Colab 실행 명령은 각 실험 문서에 기록했다.

## 문서 읽기 순서

빠르게 결과를 이해하려면 다음 세 문서면 충분하다.

1. [학습용 논문 재현 최종 정리](docs/14-study-reproduction-conclusion.md)
2. [핵심 재현 범위와 실험 계약](docs/00-reproduction-scope.md)
3. [재현 가능성 차이 및 처리 방침](docs/01-reproducibility-gaps.md)

실제 진행 과정을 순서대로 확인하려면 아래 기록을 읽는다.

1. [원본 데이터 manifest와 무결성 검사](docs/02-raw-data-integrity.md)
2. [Internet traffic 전처리](docs/03-internet-preprocessing.md)
3. [중앙 900셀 공간 선택과 검증](docs/04-central-900-selection.md)
4. [UPC 24개 초기 그룹 생성](docs/05-upc-initial-groups.md)
5. [UPC Fig. 4 불일치 제한 감사](docs/06-upc-fig4-bounded-audit.md)
6. [예측 표본 계약과 학습 없는 기준선](docs/07-naive-baselines.md)
7. [RCTL 아키텍처 계약과 Colab T4 smoke](docs/08-rctl-architecture-smoke.md)
8. [UPC PCC 기반 최종 2개 cluster](docs/09-upc-pcc-final-clusters.md)
9. [UPC 순서 민감도와 학습 정책](docs/10-upc-order-training-policy.md)
10. [중앙 900셀 LSTM·UPC pipeline smoke](docs/11-lstm-upc-smoke.md)
11. [LSTM Train-only 셀별 Min-Max scaling pilot](docs/12-lstm-train-only-scaling-pilot.md)
12. [중앙 900셀 LSTM 전체 Train·Validation 학습](docs/13-lstm-full-training.md)

각 문서의 “다음 단계”는 당시의 순차적인 판단 기록이다. 프로젝트의 현재 종료
상태와 선택적 후속 작업은 최종 정리 문서를 우선한다.

## 출처와 주의사항

- 논문: *Urban Mobile Data Prediction With Geospatial Clustering and Dual Residual Learning*
  <https://doi.org/10.1109/TNSM.2025.3599168>
- 원저자 공개 코드: <https://github.com/Superint-Lab/GECOS>
- Telecom Italia Big Data Challenge 데이터: <https://doi.org/10.7910/DVN/EGZHFV>

데이터는 원출처의 ODbL 1.0 조건과 출처 표기를 따른다. 원저자 공개 저장소에는
소프트웨어 라이선스가 명시돼 있지 않으므로, 이 저장소가 원저자 코드 전체의
재배포 조건을 새로 부여한다고 해석해서는 안 된다.
