# GECOS
Urban Mobile Data Prediction with Geospatial Clustering and Dual Residual Learning

- Project Overview: This repository provides implementations for correlation-based clustering and the Residual Convolutional Temporal Learning (RCTL) model, which are tailored for sequential data prediction tasks such as mobile network traffic forecasting.

- Dataset Description: The dataset used in clustering (correlation_matrix_0721.csv) contains a correlation matrix representing the pairwise similarities among variables (e.g., mobile network cells or sensors). Each cell value denotes the correlation strength between two variables, guiding the grouping into clusters with high internal similarity.

- Clustering Methodology: The clustering algorithm groups variables based on maximizing the average correlation within each cluster. The optimization iteratively reallocates variables to different clusters to enhance intra-cluster similarity, ultimately converging to an optimized distribution of correlated variables.

- RCTL Model Explanation: The RCTL architecture integrates convolutional, recurrent (LSTM), and residual connections, effectively capturing temporal dependencies and spatial features of time-series data. Residual connections improve gradient flow and model performance, making RCTL suitable for complex sequential prediction tasks.

## 한국어 재현 프로젝트 문서

이 포크는 공개 코드의 단순 실행보다 논문과 코드의 차이를 확인하고, 재현 가능한
데이터 처리 및 실험 과정을 학습하는 것을 목표로 한다.

- [핵심 재현 범위와 실험 계약](docs/00-reproduction-scope.md)
- [재현 가능성 차이 및 처리 방침](docs/01-reproducibility-gaps.md)
- [원본 데이터 manifest와 무결성 검사](docs/02-raw-data-integrity.md)
- [Internet traffic 전처리](docs/03-internet-preprocessing.md)
- [중앙 900셀 공간 선택과 검증](docs/04-central-900-selection.md)
- [UPC 24개 초기 그룹 생성과 논문 지문 검증](docs/05-upc-initial-groups.md)
- [UPC Fig. 4 불일치 제한 감사와 후속 프로토콜 결정](docs/06-upc-fig4-bounded-audit.md)
- [예측 표본 계약과 학습 없는 기준선](docs/07-naive-baselines.md)
- [RCTL 아키텍처 계약과 Colab T4 과적합 smoke](docs/08-rctl-architecture-smoke.md)
- [UPC PCC 기반 최종 2개 클러스터](docs/09-upc-pcc-final-clusters.md)
- [데이터 디렉터리와 출처](data/README.md)

현재 원본 무결성 검사, 메모리 제한 전처리, 중앙 900셀 공간 선택, UPC 초기 그룹과
Fig. 4 제한 감사, 공통 예측 계약과 학습 없는 두 기준선, RCTL 구조 감사와 실제
Train 부분집합의 Colab T4 과적합 smoke, UPC PCC 기반 최종 2개 클러스터까지
구현했다. UPC의 명시
알고리즘과 Fig. 4 그룹 수가 정확히 일치하지 않는 문제는 결과에 맞춰 숨은 규칙을
조정하지 않고 `GAP-UPC-06`으로 추적한다. 제한 감사 후 주 실험은 `train_only`,
Algorithm 1 민감도 실험은 `algorithm1_full_month`로 고정했으며 Fig. 4 probe는 모델
입력으로 사용하지 않는다. PCC 병합 결과 `train_only`은 남은 그룹 순서를 뒤집어도
membership이 같았지만 전체 월 경로는 50.48%만 일치해 `GAP-UPC-07`과 고비용 학습
전 검토 조건으로 남겼다. 기준선은 엄격한 20/5/5 시간 분할에서 Persistence가
일간 계절성 naive보다 강함을 확인했다. 각 단계는 입력·출력 checksum과 합성 테스트로
검증하며, 논문에서 공개하지 않은 중앙 셀 목록은 `central-900-approximate`
프로토콜로 명시한다. 논문 그림 해석형 RCTL의 parameter 수는 논문 표와 일치하지
않아 공개 코드형과 분리하고 `GAP-RCTL-01`부터 `GAP-RCTL-05`로 추적한다.
