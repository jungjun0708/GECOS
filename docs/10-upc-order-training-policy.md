# UPC 순서 민감도 검토와 프로토콜별 학습 정책

## 1. 결론

PCC 군집 단계에서 발견된 순서 민감도를 검토한 결과, 후속 신경망 학습의 허용 범위를
다음과 같이 결정했다.

| 프로토콜 | 순서 일치율 | 역할 | 모델 학습 |
|---|---:|---|---|
| `train_only` | 100.00% | 누수 없는 주 결과 | 허용 |
| `algorithm1_full_month` | 50.48% | PCC 군집 민감도 | 보류 |
| `figure4_probe_complete_weeks_mean_profile` | 해당 없음 | Fig. 4 원인 진단 | 금지 |

기존 PCC manifest의 전역
`ready_for_expensive_model_training=false`는 계산 당시의 보수적 중단선이므로
변경하지 않았다. 대신 별도의 machine-readable 정책에서
`ready_for_primary_model_training=true`와
`ready_for_all_preregistered_protocol_training=false`를 동시에 기록한다.

이 결정은 cluster 균형이나 모델 성능을 보고 membership을 고른 것이 아니다.
`train_only`이 처음부터 주 프로토콜이었고 미래 정보를 사용하지 않으며, 사전 등록한
오름차순·내림차순 검사에서 10,000셀 모두 같았다는 근거로 학습 범위만 나눈 것이다.

## 2. 왜 별도 결정이 필요한가

[`UPC PCC 기반 최종 2개 클러스터`](09-upc-pcc-final-clusters.md)는 두 프로토콜 중
하나라도 순서 안정성 임계값 95%를 통과하지 못하면 전역 gate를 닫도록 보수적으로
구현했다. 실제로 전체 월 경로가 50.48%에 그쳐 gate가 닫혔다.

그러나 두 프로토콜의 목적은 같지 않다.

- `train_only`은 첫 20일 안의 평일만 사용한 주 성능용 membership이다.
- `algorithm1_full_month`는 논문 Algorithm 1 해석을 비교하기 위한 민감도다.
- Fig. 4 probe는 논문에 명시되지 않은 근접 가설이며 처음부터 모델 입력이 아니다.

민감도 경로의 불확실성을 무시하면 안 되지만, 그것이 안정적인 주 프로토콜까지
영구적으로 막는 것도 실험 목적과 맞지 않는다. 따라서 clustering 사실과 그 사실을
후속 학습에서 사용하는 정책을 서로 다른 산출물로 분리했다.

## 3. 변경하지 않고 고정한 근거

정책 설정은 다음 네 PCC 산출물의 SHA-256을 직접 고정한다.

| 산출물 | SHA-256 |
|---|---|
| `group_assignments.csv` | `9f5839a499ba81f2b69ae3f0819b6a3a3e91de20f204dbdd12d0b3165a88440d` |
| `all_cell_memberships.csv` | `d5a1bdd49d8b40f8f5d6e85754e67aea9a580aad80e77e52d2c0ac972cc2b0bb` |
| `central_900_memberships.csv` | `9cb60b83013ec953256a9e182e51f52ffddf9ea2aa5ae473740f7f413b5086fa` |
| `summary.json` | `09fbddbd2bf3f8a533578e9a7c7f518c14807b3074cfce53615520bfac946ab9` |

PCC config의 SHA-256
`60fdc27b0b5d486001b7259121e5b5e3e13b59a4f6210645bf95902fb1c9b20d`도
함께 고정한다. 이 값 중 하나라도 달라지면 기존 결정을 새 clustering 결과에 자동으로
적용하지 않고 실행을 중단한다.

이 설정은 결과를 보기 전의 사전 등록으로 위장하지 않는다. 명칭과 파일 안에
`post_clustering_pre_model`을 기록해, PCC 결과를 확인한 뒤 모델 학습 전에 내린
범위 결정임을 명시한다. 다만 모델 지표는 아직 계산하지 않았으므로 membership을
성능에 맞춰 선택하는 누수는 없다.

## 4. 제한된 순서 민감도 분석

### 4.1 바뀐 범위

전체 월 프로토콜에서 오름차순과 내림차순 사이에 배정이 바뀐 초기 그룹은 다음
10개다.

```text
[7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
```

이 그룹들의 셀 수 합계는 `4,952`다. label을 서로 바꿔 가장 잘 맞춘 뒤에도 전체
10,000셀 중 `5,048`개만 같아 일치율은 `50.48%`다. 중앙 900셀 일치율은
`60.78%`다. 단순한 cluster 0/1 이름 차이가 아니라 중간 시간대 그룹의 실제 이동이다.

`train_only`에서는 변경 group과 변경 cell이 모두 0이다.

### 4.2 어떻게 경로가 강화되는가

전체 월의 seed는 group 11과 21로 동일하다. 하지만 남은 group을 어떤 방향에서 먼저
추가하는지에 따라 현재 cluster의 구성과 다음 평균 PCC가 달라진다.

| Group | 오름차순 PCC 점수 `[C0, C1]` | 선택 | 내림차순 PCC 점수 `[C0, C1]` | 선택 |
|---:|---|---:|---|---:|
| 7 | `[0.380419, 0.716304]` | C1 | `[0.855428, 0.842886]` | C0 |
| 8 | `[0.485754, 0.702663]` | C1 | `[0.883480, 0.733856]` | C0 |
| 13 | `[0.441459, 0.805246]` | C1 | `[0.981670, 0.666169]` | C0 |
| 17 | `[0.411226, 0.893393]` | C1 | `[0.929040, 0.788018]` | C0 |

오름차순에서는 7부터 시작한 중간 시간대 그룹이 차례로 C1 쪽 평균을 강화한다.
내림차순에서는 17부터 반대 방향으로 들어오며 C0 쪽 평균을 강화한다. 내림차순의
group 7은 점수 차가 약 `0.0125`로 작지만, 나머지 여러 그룹은 더 큰 차이로 같은
방향을 이어 간다. 이는 개별 PCC 동률 하나가 아니라 순차 평균의 누적 경로 의존성이다.

새로운 순열이나 결과에 유리한 순서는 추가로 탐색하지 않았다. Fig. 5에 조금 더
가까운 내림차순을 선택하지도 않았다.

## 5. 최종 학습 정책

### 5.1 `train_only`

- 역할: 주 UPC 성능 프로토콜
- scaling 및 clustering 적합 범위: Train 기간만
- 오름차순·내림차순 membership 일치: 10,000 / 10,000
- 모델 학습: 허용
- 후속 UPC on/off 비교에서 사용할 순서: 사전 등록된 오름차순

### 5.2 `algorithm1_full_month`

- 역할: 논문 Algorithm 1의 clustering 민감도
- 전체 월 정보를 사용하므로 주 성능 입력으로 사용할 수 없음
- 순서 일치: 5,048 / 10,000
- 모델 학습: 현재 보류
- 기존 오름차순·내림차순 군집 산출물: 진단 근거로 보존
- 다시 사용하려면 독립 config와 명시적인 추가 결정이 필요

### 5.3 Fig. 4 probe와 순서 불변 대안

- Fig. 4 probe는 계속 모델 입력을 금지한다.
- seed에만 고정해 한 번에 배정하는 방법 등 순서 불변 대안은 논문 Algorithm 1이
  아니다.
- 순서 불변 방법은 필요할 때 `non_reproduction_extension`이라는 별도 실험으로만
  설계한다.
- 현재 핵심 재현 범위에는 추가하지 않는다.

## 6. 구현

정책 config:

```text
configs/upc_training_policy_milan_nov2013.json
```

검증 도구:

```text
scripts/validate_upc_training_policy.py
```

도구는 다음 순서로 작동한다.

1. PCC config checksum을 확인한다.
2. PCC manifest가 Fig. 4 불일치와 순서 검토 상태를 유지하는지 확인한다.
3. 고정한 네 산출물의 manifest checksum과 실제 파일 checksum을 모두 확인한다.
4. `summary.json` 파일과 manifest 안에 복제된 summary가 같은지 확인한다.
5. group assignment CSV에 두 프로토콜의 0~23 group이 정확히 한 번씩 있는지 확인한다.
6. label mapping을 적용해 변경 group과 4,952셀을 다시 계산한다.
7. 95% 임계값으로 프로토콜별 gate를 평가한다.
8. 결정론적 정책과 동적 실행 manifest를 원자적으로 게시한다.

정책 생성과 주 프로토콜 검사는 다음 명령으로 실행한다.

```bash
.venv/bin/python -m scripts.validate_upc_training_policy \
  --config configs/upc_training_policy_milan_nov2013.json \
  --check-protocol train_only
```

전체 월을 요청하면 명시적인 사유와 함께 실패한다.

```bash
.venv/bin/python -m scripts.validate_upc_training_policy \
  --config configs/upc_training_policy_milan_nov2013.json \
  --check-protocol algorithm1_full_month
```

후속 모델 학습기는 실행 전에 `require_training_allowed()`를 호출하거나 같은 정책
파일의 `training_gates`를 검증해야 한다.

## 7. 산출물

두 산출물은 재생성 가능한 데이터 정책이므로 Git에서 제외한다.

| 파일 | 역할 |
|---|---|
| `data/processed/upc/training_policy.json` | 결정론적 근거, 프로토콜별 gate와 허용 목록 |
| `data/processed/upc/training_policy_manifest.json` | config·입력·출력 checksum과 실행 환경 |

핵심 gate는 다음과 같다.

```text
ready_for_primary_model_training = true
ready_for_all_preregistered_protocol_training = false
allowed_model_training_protocols = [train_only]
blocked_model_training_protocols = [algorithm1_full_month, figure4_probe]
```

## 8. 테스트와 중단 조건

합성 테스트는 다음을 확인한다.

- 정책이 `post_clustering_pre_model` 결정으로 표시됨
- `train_only`만 학습 허용
- 전체 월, Fig. 4 probe와 알 수 없는 protocol 차단
- label swap 후에도 같은 10개 변경 group과 4,952셀 검출
- group cell 합과 membership 비교값 불일치 차단
- CSV 48개 protocol/group 행의 완전성과 고유성
- 같은 입력으로 정책 파일을 두 번 만들 때 byte 단위 일치
- 보호 파일 변조 시 기존 정책을 덮어쓰지 않음
- PCC clustering의 전역 `false` gate 값 보존

다음 상황에서는 주 모델 학습으로 넘어가지 않는다.

- 보호한 membership 또는 summary checksum이 변경됨
- `train_only` 순서 일치율이 95% 미만으로 바뀜
- 학습 config가 `train_only` 이외의 UPC protocol을 요청함
- Fig. 4 probe가 feature, label 또는 셀 선택에 포함됨
- 정책 검증 없이 Colab 학습 입력을 생성함

정책 검증 자체는 수십 MiB 수준의 메모리와 수 밀리초만 필요해 로컬 WSL CPU에서
실행한다. Colab은 사용하지 않는다.

실제 데이터에서 연속 두 번 실행한 결과 `training_policy.json`의 byte와 SHA-256이
같았다.

| 항목 | 결과 |
|---|---:|
| 전체 테스트 | 73개 통과, 기존 로컬 TensorFlow 검사 1개 skip |
| 정책 파일 크기 | 4,576 bytes |
| 정책 SHA-256 | `bff2a8aa4206ad292f6b10c2b4ada55fd69eedfc88a0af2befb23a04cf12b6d2` |
| 실제 실행 시간 | 약 0.006초 |
| 최대 RSS | 약 32.6MiB |

보호한 기존 PCC 산출물 네 개의 SHA-256도 정책 실행 전후 모두 설정값과 같았다.

## 9. 다음 단계

이 정책으로 주 학습 경로가 열렸다. 다음 작업은 중앙 900셀에서 공통 학습·평가
파이프라인과 LSTM 기준선을 먼저 구현하는 것이다.

첫 Colab 실행은 전체 3-seed 본 실험이 아니라 다음을 확인하는 제한 smoke로 시작한다.

1. `training_policy.json`의 `train_only` 허용 여부와 checksum 검증
2. 기존 20/5/5 target timestamp split 재사용
3. UPC off 단일 LSTM의 입력·target·지표 저장 검증
4. UPC on에서 cluster별 두 모델 학습과 900셀 예측 재결합 검증
5. Persistence보다 학습 파이프라인이 구조적으로 정상인지 확인

이 smoke가 통과한 뒤에만 seed `42`, `43`, `44`의 LSTM과 RCTL 비교로 확장한다.
