# Internet traffic 전처리

## 1. 목적

공식 checksum 검증을 통과한 Telecom Italia Milano 원본 30개를 읽어 GECOS 학습에
사용할 `10,000셀 × 4,320시점` Internet traffic 행렬을 만든다. 전체 원본은
9.64GiB, 160,108,003행이므로 한 번에 메모리에 적재하지 않는다.

전처리 설정은
[`configs/preprocess_milan_nov2013.json`](../configs/preprocess_milan_nov2013.json)에
고정한다.

## 2. 전처리 전용 환경

전처리는 Python 3.12에서 다음 버전을 사용한다.

| 패키지 | 버전 | 용도 |
|---|---:|---|
| NumPy | 2.5.2 | 일별 집계와 `.npy` 기록 |
| PyArrow | 25.0.1 | 엄격한 TSV parsing과 Parquet 기록 |
| tzdata | 2026.3 | `Europe/Rome` 시간대 데이터 |

일반적인 Python 환경에서는 다음과 같이 설치한다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --requirement requirements/preprocess.txt
```

현재 WSL처럼 `ensurepip`가 설치되지 않은 환경에서는 시스템 Python을 변경하지 않고
다음 방식으로 같은 환경을 만들 수 있다.

```bash
python3 -m venv --without-pip .venv
python3 -m pip --python .venv/bin/python install --requirement requirements/preprocess.txt
```

이 환경에는 TensorFlow를 설치하지 않는다. Google Colab 학습 환경은 모델 구현
단계에서 별도로 고정한다.

## 3. 실행 전 테스트

```bash
.venv/bin/python -m unittest discover -s tests -v
```

합성 테스트는 다음 조건을 확인한다.

- 여러 국가코드 행의 Internet 합계
- 원본 공란과 완전히 없는 셀-시점의 구분
- 입력 행 순서가 바뀌어도 같은 집계 결과가 생성되는지
- 잘못된 열 수, cell ID, timestamp와 음수 차단
- 실패할 때 완성 경로에 부분 산출물을 공개하지 않는지

## 4. 전체 전처리 실행

저장소 루트에서 실행한다.

```bash
.venv/bin/python -m scripts.preprocess_internet
```

원본 경로가 다르면 다음과 같이 지정한다.

```bash
.venv/bin/python -m scripts.preprocess_internet --data-dir /path/to/raw-data
```

전처리는 시작할 때 30개 원본의 크기와 MD5를 다시 계산한다. 한 파일이라도 공식
manifest와 다르면 TSV를 읽기 전에 중단한다.

## 5. 처리 과정

1. 공식 기준 manifest와 원본 30개를 MD5까지 비교한다.
2. PyArrow CSV streaming reader가 각 행을 정확히 8열로 해석한다.
3. cell ID, country code, timestamp, NaN, 무한대와 음수를 검사한다.
4. 하루 파일을 64MiB block으로 읽는다.
5. `(cell_id, timestamp_ms)`가 같은 국가코드 행의 Internet을 float64로 합산한다.
6. 원본 Internet 공란은 합산할 때 0으로 처리한다.
7. 하루 집계가 끝난 뒤 한 번만 float32로 변환한다.
8. 전체 행렬은 `.npy` memory map에 날짜별 slice로 기록한다.
9. Parquet은 날짜 순서로 쓰고, 각 날짜 안에서 `cell_id`, `timestamp_ms` 오름차순으로
   기록한다.
10. 모든 검사가 끝난 뒤에만 숨김 partial 파일을 완성 경로로 교체한다.

일별 float64 누적 배열은 약 11MiB이며, 43.2백만 셀-시점 전체를 float64로 메모리에
유지하지 않는다.

## 6. 결측값 계약

traffic 값은 두 결측 유형 모두 0이지만 원인을 별도 mask로 보존한다.

| 파일 | `True`의 의미 |
|---|---|
| `missing_mask.npy` | 해당 셀-시점의 원본 행이 하나도 없음 |
| `internet_null_mask.npy` | 원본 행은 있지만 모든 Internet 값이 공란임 |

한 위치에서 두 mask가 동시에 `True`일 수 없다. 일부 국가코드 행이 공란이어도 다른
행에 Internet 값이 있으면 `internet_null_mask`는 `False`다. 활동 열별 공란 **행**
수는 `manifest.json`에 별도로 기록한다.

## 7. 산출물

| 파일 | 내용 | shape/dtype 또는 행 수 |
|---|---|---|
| `data/interim/internet_10min.parquet` | 셀-시점별 집계 및 두 결측 열 | 43,200,000행, 5열 |
| `data/processed/traffic.npy` | Internet traffic 행렬 | `(10000, 4320)`, float32 |
| `data/processed/cell_ids.npy` | 행에 대응하는 cell ID | `(10000,)`, int32 |
| `data/processed/timestamps_ms.npy` | 열에 대응하는 epoch millisecond | `(4320,)`, int64 |
| `data/processed/missing_mask.npy` | 완전 누락 위치 | `(10000, 4320)`, bool |
| `data/processed/internet_null_mask.npy` | Internet 전체 공란 위치 | `(10000, 4320)`, bool |
| `data/processed/manifest.json` | 입력·설정·통계·환경·출력 SHA-256 | JSON |

NumPy 행렬은 cell ID가 행, 전체 timestamp가 열이다. Parquet은 날짜 오름차순이며
각 날짜 블록 안에서 cell ID, 해당 날짜 timestamp 오름차순이다. 모든 산출물은
`.gitignore`로 제외된다.

## 8. 30일 실행 결과

동일한 원본과 설정으로 전체 전처리를 두 번 실행했다.

| 항목 | 결과 |
|---|---:|
| 원본 행 | 160,108,003 |
| 고유 cell | 10,000 |
| 고유 timestamp | 4,320 |
| timestamp 간격 | 600,000ms |
| traffic 범위 | 0.0 ~ 8,044.0708 |
| 완전 누락 셀-시점 | 2,761 |
| 행은 있으나 Internet 전체 공란 | 4,380 |
| Parquet 행 | 43,200,000 |
| 전체 실행 시간 | 1분 56초 ~ 2분 1초 |
| 최대 RSS | 약 1.13GiB |

활동 열별 공란 행 수는 다음과 같다.

| 열 | 공란 행 수 |
|---|---:|
| `sms_in` | 74,836,997 |
| `sms_out` | 111,002,033 |
| `call_in` | 111,310,566 |
| `call_out` | 92,060,252 |
| `internet` | 78,998,226 |

두 실행에서 다음 SHA-256이 모두 일치했다.

| 산출물 | SHA-256 |
|---|---|
| Parquet | `4e8ca4d8311aa9b092fff37ff4e024a6c6921d0472bf74ef32b6d05c2b21b78e` |
| `traffic.npy` | `3b2919761e0791987f2131582301b9cfa501a6e69229aade881a4242c8fff885` |
| `cell_ids.npy` | `e3d762a4fa5573cf02197827f8a028a26a8c50b2605396e814b84df415fed347` |
| `timestamps_ms.npy` | `0005bf43a0c2c3dad033bfcfca8466ad23f28b83869c5403efae256454a31e4d` |
| `missing_mask.npy` | `e3aff0ba9461d7fe69b663575d4b74f9e573a0dee158614c57349a112a3b8838` |
| `internet_null_mask.npy` | `efe9661888ea7fd747ed34f58c1fbb1b563a50cc4e18b5b240ea671a466463d1` |

성능 수치는 현재 WSL과 저장장치에서 측정한 참고값이다. 재현 여부는 실행 시간이
아니라 각 산출물 checksum으로 판단한다.

## 9. 다음 단계

이 데이터 계약을 통과한 `traffic.npy`, `cell_ids.npy`, `timestamps_ms.npy`를 사용해
Milano Grid의 기하학적 중앙 30×30에 해당하는 900개 cell ID를 생성하고 지도에서
선택 영역을 확인한다.
