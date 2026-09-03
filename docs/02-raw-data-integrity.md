# 원본 데이터 manifest와 무결성 검사

## 1. 목적

전처리 전에 로컬의 Telecom Italia 원본 30개가 공식 Harvard Dataverse 파일과
바이트 단위로 같은지 확인한다. 파일명이나 크기가 같다는 사실만으로는 다운로드
중 손상 또는 내용 변경을 발견할 수 없으므로, 공식 파일별 MD5까지 비교한다.

이 검사를 통과하지 않은 데이터로 UPC 또는 모델 실험을 진행하지 않는다.

## 2. 기준 manifest

추적 파일:
[`metadata/telecom_italia_mi_2013_11.json`](../metadata/telecom_italia_mi_2013_11.json)

기준값은 Harvard Dataverse API가 제공하는 dataset version 1.3의 메타데이터에서
가져왔다.

- 데이터: *Telecommunications - SMS, Call, Internet - MI*
- DOI: <https://doi.org/10.7910/DVN/EGZHFV>
- 기간: 2013년 11월 1일부터 30일까지
- 파일 수: 30개
- 전체 크기: 10,351,643,789 bytes, 약 9.64GiB
- checksum: 공식 파일별 MD5

MD5는 새로운 보안 보장을 위해 사용하지 않는다. 원출처가 공개한 값과 로컬
바이트가 같은지 확인하는 용도로만 사용한다.

## 3. 전체 무결성 검사

저장소 루트에서 실행한다.

```bash
python3 scripts/verify_raw_data.py
```

기본 경로는 다음과 같다.

| 구분 | 경로 |
|---|---|
| 원본 데이터 | `dataverse_files/` |
| 기준 manifest | `metadata/telecom_italia_mi_2013_11.json` |
| 실행 보고서 | `data/interim/raw_integrity_report.json` |

다른 위치에 원본을 보관했다면 명시적으로 지정한다.

```bash
python3 scripts/verify_raw_data.py --data-dir /path/to/raw-data
```

검증기는 기본 8MiB 버퍼로 파일을 순서대로 읽는다. 전체 9.64GiB를 메모리에
올리지 않으므로 데이터 크기에 비례해 메모리 사용량이 증가하지 않는다.

성공한 전체 검사의 핵심 결과는 다음과 같아야 한다.

```text
status=passed
files=30/30
bytes=10351643789/10351643789
integrity_verified=true
```

## 4. 빠른 검사

파일명과 크기만 먼저 확인하려면 다음 명령을 사용한다.

```bash
python3 scripts/verify_raw_data.py --quick
```

빠른 검사는 `status=passed_size_only`, `integrity_verified=false`를 기록한다. 실행
성공으로 종료되더라도 전처리 시작 조건을 충족하지 않는다. 최초 전처리 전에는
반드시 기본 전체 검사를 한 번 통과해야 한다.

## 5. 종료 코드와 실패 대응

| 종료 코드 | 의미 |
|---:|---|
| 0 | 요청한 검사 항목 통과 |
| 1 | 누락, 예상 외 파일, 크기 또는 checksum 불일치 |
| 2 | 잘못된 manifest, 인자 또는 파일 시스템 오류 |

실패하면 보고서의 `missing_files`, `unexpected_files`, `files[].status`를 확인한다.
원본 내용을 직접 수정해 checksum을 맞추지 않는다. 공식 출처에서 실패한 파일만
다시 내려받은 뒤 전체 검사를 반복한다.

## 6. 보고서와 Git 보호

실행 보고서는 검사 시각, 기준 출처, 모드, 파일별 결과와 manifest SHA-256을
기록한다. `data/interim/`은 `.gitignore`로 보호되므로 보고서와 원시 데이터는
커밋하지 않는다. 기준 manifest만 작고 추적 가능한 provenance 자료로 Git에
포함한다.

## 7. 자동 테스트

Python 3.10 이상에서 추가 패키지 없이 표준 라이브러리만으로 실행한다. 최초 구현은
Python 3.12.3에서 검증했다.

```bash
python3 -m unittest discover -s tests -v
```

테스트는 작은 임시 파일로 정상, 누락, 예상 외 파일, 크기 불일치, checksum 불일치,
빠른 검사와 잘못된 manifest를 확인한다. 실제 9.64GiB 데이터는 단위 테스트에
포함하지 않는다.

## 8. 다음 단계

전체 검사 통과 후 한 파일 또는 제한된 chunk만 메모리에 올리는 전처리를 구현한다.
그 단계에서 8개 열 구조, 행 수, 공란 통계, 셀 및 timestamp 범위를 검사하고
최종 `(10000, 4320)` traffic 행렬을 만든다.
