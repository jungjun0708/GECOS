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
