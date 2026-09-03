# 데이터 디렉터리

이 디렉터리는 GECOS 재현에 사용하는 데이터의 저장 위치를 정의한다.
원시 데이터와 전처리 결과는 용량 및 라이선스 문제로 Git에 포함하지 않는다.

## 디렉터리

- `raw/`: 다운로드한 원시 데이터
- `interim/`: 다시 생성할 수 있는 중간 전처리 결과
- `processed/`: 학습에 직접 사용하는 최종 데이터

각 디렉터리의 실제 데이터는 Git에서 제외하며, 디렉터리 구조를 유지하기 위한
`.gitkeep` 파일만 추적한다.

## 데이터 출처

Telecom Italia, *Telecommunications - SMS, Call, Internet - MI*

- DOI: <https://doi.org/10.7910/DVN/EGZHFV>
- 라이선스: Open Database License (ODbL) 1.0
- 원출처: [from BigDataChallenge contest](http://www.telecomitalia.com/tit/en/bigdatachallenge.html)

데이터를 내려받거나 재배포할 때 ODbL 1.0 조건과 출처 표기를 확인한다.
향후 제공할 전처리 명령으로 모든 중간 및 처리 데이터를 다시 생성할 수 있어야 한다.

공간 선택에는 Telecom Italia의 *Milano Grid*도 사용한다.

- DOI: <https://doi.org/10.7910/DVN/QJWLFU>
- 파일: `raw/milano-grid.geojson`
- 공식 MD5: `e64ed3858e97347c14cd049efde8c7d3`
- 기준 metadata: [`metadata/milano_grid.json`](../metadata/milano_grid.json)

공식 checksum과 같은 공개 미러 사본은 다음 명령으로 준비한다. 직접 Dataverse에서
받은 정상 파일이 이미 있으면 다시 다운로드하지 않는다.

```bash
.venv/bin/python -m scripts.fetch_milano_grid
```

전처리를 시작하기 전에
[원본 데이터 manifest와 무결성 검사](../docs/02-raw-data-integrity.md)를 실행한다.
검증 보고서는 `interim/raw_integrity_report.json`에 생성되며 Git에는 포함하지 않는다.

## 전처리

원본 검증을 통과한 뒤
[Internet traffic 전처리](../docs/03-internet-preprocessing.md)를 실행한다. 전처리는
국가코드별 Internet 값을 셀과 10분 시점 단위로 합산하고 `processed/`에 학습용
행렬과 결측 mask를 생성한다. 모든 산출물은 Git에서 제외되며 `manifest.json`으로
입력, 설정, 통계와 SHA-256을 추적한다.

전처리 완료 후
[중앙 900셀 공간 선택과 검증](../docs/04-central-900-selection.md)을 실행한다. 이
단계는 공식 Grid 좌표로 중앙 30×30을 선택하고 `(900, 4320)` traffic 부분집합과
두 결측 mask를 생성한다. 결과는 저자의 정확한 ID가 공개되지 않았다는 한계를
반영해 `central-900-approximate`로 표시한다.

## UPC 초기 그룹

전처리와 중앙 900셀 선택이 끝나면
[UPC 24개 초기 그룹 생성과 논문 지문 검증](../docs/05-upc-initial-groups.md)을
실행한다. 결과는 `processed/upc/`에 생성되며 전체 10,000셀, 누수 방지 Train 기간,
중앙 900셀 membership을 분리해 저장한다. 이 디렉터리도 Git에서 제외되고 출력
SHA-256과 논문 Fig. 4 비교값은 `processed/upc/manifest.json`에 기록된다.

Fig. 4 불일치에 대한 사전 등록 감사는
[UPC Fig. 4 불일치 제한 감사](../docs/06-upc-fig4-bounded-audit.md)의 절차로
실행한다. 결과 JSON·CSV와 manifest는 `processed/upc/audits/`에 저장되며 역시
Git에서 제외된다. 이 감사는 진단 membership 파일을 만들지 않는다.

## UPC PCC 최종 클러스터

초기 그룹과 제한 감사가 끝나면
[UPC PCC 기반 최종 2개 클러스터](../docs/09-upc-pcc-final-clusters.md)를 실행한다.
두 프로토콜의 길이 24 group profile, PCC, 초기 group→최종 cluster 대응과 전체
10,000셀·중앙 900셀 membership은 `processed/upc/final_clusters/`에 저장한다.
모두 파생 데이터라 Git에서 제외하며 `summary.json`과 `manifest.json`에 배정 순서
민감도, 논문 Fig. 5 진단, 입력·출력 checksum을 보존한다. Fig. 4 probe는 이 단계의
입력으로 사용하지 않는다.

PCC 순서 민감도 검토 후에는
[프로토콜별 UPC 학습 정책](../docs/10-upc-order-training-policy.md)을 생성한다.
`processed/upc/training_policy.json`은 `train_only`만 모델 학습에 허용하고,
`algorithm1_full_month`와 Fig. 4 probe를 차단한다. 함께 생성되는
`training_policy_manifest.json`은 정책 config와 보호한 네 PCC 산출물의 checksum을
기록한다. 후속 Colab 입력을 만들기 전에 이 정책 검증을 통과해야 한다.

## 학습 없는 기준선

공통 예측 표본과 시간 분할을 검증한 뒤
[예측 표본 계약과 학습 없는 기준선](../docs/07-naive-baselines.md)을 실행한다.
Persistence와 daily seasonal naive의 요약, Test 셀별 지표와 manifest는
`processed/baselines/`에 저장된다. 이 결과도 파생 데이터이므로 Git에서 제외하며,
결정성은 manifest의 출력 SHA-256과 반복 실행으로 확인한다.

## RCTL smoke 입력과 결과

[RCTL 아키텍처 계약과 Colab T4 과적합 smoke](../docs/08-rctl-architecture-smoke.md)는
전체 배열을 Colab에 올리지 않는다. 로컬에서 전처리 checksum을 먼저 검증하고 중앙
30×30 격자에 등간격으로 흩어진 16셀, 셀당 Train target 64개만 추출한다.

- `interim/rctl_smoke/input.npz`: `(1024, 8, 1)` 입력과 target, Persistence 값, ID 및 mask
- `interim/rctl_smoke/input_manifest.json`: 선택 규칙과 입력·배열 checksum
- `processed/rctl_smoke/architecture_report.json`: 논문형/공개형 구조 감사
- `processed/rctl_smoke/overfit_report.json`: 일회성 Train 과적합 진단
- `processed/rctl_smoke/manifest.json`: Colab 환경과 출력 checksum

이 파일들은 모두 재생성 가능한 파생 데이터라 Git에서 제외한다. smoke 결과는
Validation/Test 성능이 아니며, 일회성 모델 checkpoint도 보존하지 않는다.

## LSTM·UPC pipeline smoke 입력과 결과

[중앙 900셀 LSTM·UPC Colab T4 pipeline smoke](../docs/11-lstm-upc-smoke.md)는
중앙 900셀을 모두 유지하고 각 Train/Validation/Test 분할에서 셀당 64개 target을
등간격으로 선택한다. 로컬에서는 필요한 window만 추출하고, UPC 미적용 모델 하나와
`train_only` cluster별 모델 둘의 학습은 Colab T4에서 수행한다.

- `interim/lstm_upc_smoke/input.npz`: 900셀의 선택 window·target·Persistence·mask
- `interim/lstm_upc_smoke/input_manifest.json`: 선택 규칙, clean Git commit과 입력·배열 checksum
- `interim/lstm_upc_smoke/colab_bundle.zip`: 최소 Colab 업로드 bundle
- `processed/lstm_upc_smoke/architecture_report.json`: `165,185` parameter 구조 감사
- `processed/lstm_upc_smoke/evaluation_report.json`: 세 모델의 학습·재결합·지표 결과
- `processed/lstm_upc_smoke/predictions.npz`: UPC off/on 및 Persistence 예측
- `processed/lstm_upc_smoke/per_cell_metrics.csv`: Test 셀별 MAE/MAPE/WAPE
- `processed/lstm_upc_smoke/manifest.json`: Git provenance, Colab 환경과 출력 checksum

위 파일과 Colab 결과 ZIP은 모두 재생성 가능한 파생 산출물이므로 Git에서 제외한다.
고정 5 epoch smoke는 checkpoint를 저장하지 않으며, 결과는 논문 성능표가 아니라
pipeline의 구조·학습·재결합 검증으로만 사용한다.
