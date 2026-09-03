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
- [데이터 디렉터리와 출처](data/README.md)

현재 단계에서는 실험 전에 적용할 재현 규칙과 알려진 불확실성을 문서로 고정했다.
구현 및 결과는 이후 단계별 PR에서 추가한다.
