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

데이터를 내려받거나 재배포할 때 ODbL 1.0 조건과 출처 표기를 확인한다.
향후 제공할 전처리 명령으로 모든 중간 및 처리 데이터를 다시 생성할 수 있어야 한다.
