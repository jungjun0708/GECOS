# UPC PCC 기반 최종 2개 클러스터

## 1. 결론

24개 peak-hour 초기 그룹의 길이 24 traffic profile과 Pearson correlation
coefficient(PCC)를 계산하고, 논문 Algorithm 1을 따라 최종 `N=2` 클러스터로
병합했다. 계산, 입력 무결성, 전체 10,000셀 및 중앙 900셀 완전 배정, 반복 실행
결정성은 모두 통과했다.

그러나 두 프로토콜의 안정성은 달랐다.

| 프로토콜 | Seed 그룹 | Seed PCC | 오름차순 셀 수 | 순서 역전 일치율 | 판정 |
|---|---|---:|---:|---:|---|
| `train_only` | 12, 20 | 0.533283 | 5,409 / 4,591 | 100.00% | 주 분석 membership 안정 |
| `algorithm1_full_month` | 11, 21 | 0.492594 | 282 / 9,718 | 50.48% | 배정 순서 검토 필요 |

`train_only` 결과는 남은 그룹을 hour 오름차순과 내림차순으로 처리해도 완전히
같았다. 반면 논문 비교용 `algorithm1_full_month`는 순서를 뒤집으면 5,234 / 4,766으로
크게 달라졌다. 논문은 남은 그룹의 순서를 공개하지 않았으므로 이 차이는 결과를 보고
더 그럴듯한 쪽을 고르는 근거가 아니다. 사전 등록한 오름차순을 주 구현으로 유지하고,
일치율 95% 미만 규칙에 따라 고비용 모델 학습 전 검토가 필요한 상태로 표시했다.

## 2. 이 단계가 필요한 이유

UPC는 단순히 셀마다 peak hour label을 붙이는 데서 끝나지 않는다. 서로 비슷한
시간 흐름을 가진 초기 그룹을 PCC로 합쳐 최종 두 개의 셀 집합을 만든 뒤, 각 집합에
독립 모델을 학습하는 것이 핵심이다. 이 membership이 없으면 다음 비교를 같은 조건으로
정의할 수 없다.

- LSTM/RCTL의 UPC 적용 전후 비교
- cluster별 독립 학습과 전체 예측 재결합
- `train_only`와 논문 Algorithm 1 해석의 민감도 비교
- 중앙 900셀에서 cluster별 표본 수와 계산량 산정

Fig. 4 불일치는 [`GAP-UPC-06`](01-reproducibility-gaps.md)으로 남아 있지만, 제한
감사에서 PCC 구현을 막지 않기로 이미 결정했다. 따라서 Fig. 4에 가까운 진단 probe는
입력에서 배제하고, 검증된 `train_only`과 `algorithm1_full_month` membership만
사용한다.

## 3. 사전 등록한 계산 계약

설정은
[`configs/upc_pcc_milan_nov2013.json`](../configs/upc_pcc_milan_nov2013.json)에
고정했다.

### 3.1 Group profile

각 프로토콜이 허용하는 평일에 대해 셀별 min-max scaling을 먼저 적합한다. 10분 값
6개를 더해 시간 합계를 만들고, 같은 초기 그룹의 셀과 평일을 평균해 길이 24 profile을
만든다.

```text
scaled(c,t) = (traffic(c,t) - min_c) / (max_c - min_c)

profile(g,h) = mean(
  sum(scaled(c,t) for t in hour h),
  over cell c in group g and selected weekday
)
```

논문 수식은 평일 방향을 합으로 적었지만 모든 그룹에 같은 평일 수를 사용하므로 평균은
각 profile에 같은 양의 상수를 곱하는 차이뿐이다. PCC는 이 변환에 불변이다. 그래도
해석을 숨기지 않기 위해 구현과 manifest에는 평균이라고 명시했다.

### 3.2 PCC와 seed

비어 있지 않은 그룹 profile 두 개의 PCC를 계산한다.

```text
PCC(x, y) = covariance(x, y) / (std(x) * std(y))
```

- 대칭, 대각선 1, 값 범위 `[-1, 1]`을 검사한다.
- 분산 0, NaN 또는 무한대 profile은 즉시 실패한다.
- 크기가 `theta=10`보다 **큰** 그룹만 seed 후보로 사용한다.
- 후보 중 PCC가 가장 낮은 두 그룹을 seed로 선택한다.
- 같은 최솟값이면 hour 쌍이 사전식으로 가장 작은 것을 선택한다.
- seed hour가 작은 쪽부터 cluster 0, cluster 1로 이름을 붙인다.

`size=10`은 포함이 아니라 제외다. 실제 `train_only`의 group 1이 정확히 10개라 이
경계가 테스트와 결과에서 확인된다.

### 3.3 남은 그룹 배정

남은 각 그룹은 현재 cluster에 들어 있는 **초기 그룹들**과의 PCC를 단순 평균한다.
평균이 더 큰 cluster에 넣고 동률이면 cluster ID가 작은 쪽을 선택한다.

```text
score(group x, cluster C) = mean(PCC(x, member) for member in C)
```

논문 의사코드는 `x in H`라고만 써 순서를 고정하지 않았고, 앞서 배정된 그룹이 다음
평균에 포함되므로 이 연산은 순서에 따라 달라질 수 있다. 이를
[`GAP-UPC-07`](01-reproducibility-gaps.md)로 등록하고 다음 두 경로를 결과 확인 전에
고정했다.

- 주 구현: group ID 0→23 오름차순
- 순서 민감도: group ID 23→0 내림차순

크기 10 이하인 비어 있지 않은 그룹은 seed 후보에서만 제외하고 profile과 PCC를
계산해 최종 cluster에 배정한다. 빈 그룹은 계산과 배정에서 제외한다. 이는
[`GAP-UPC-03`](01-reproducibility-gaps.md)의 구현 가정이다.

## 4. 실행 방법

선행 초기 그룹을 검증·재생성한다.

```bash
.venv/bin/python -m scripts.build_upc_initial_groups \
  --config configs/upc_milan_nov2013.json
```

PCC와 최종 cluster를 생성한다.

```bash
.venv/bin/python -m scripts.build_upc_final_clusters \
  --config configs/upc_pcc_milan_nov2013.json
```

정상 실행은 초기 manifest의 `diagnostic_mismatch`를 숨기지 않은 다음 상태를
출력한다.

```text
complete_with_upstream_diagnostic_mismatch_and_order_sensitivity_review
```

다음 조건은 실행을 즉시 실패시킨다.

- 초기 UPC config 또는 membership checksum 불일치
- Fig. 4 probe를 모델 프로토콜로 사용
- group 수 합계가 10,000이 아님
- 비어 있지 않은 profile의 NaN, 무한대 또는 분산 0
- PCC 비대칭, 대각선 오류 또는 범위 이탈
- seed 후보가 두 개 미만
- 전체 10,000셀 또는 중앙 900셀의 미배정·중복 배정

순서 민감도 95% 미만은 계산 실패와 구분한다. 산출물은 보존하되 manifest의
`ready_for_expensive_model_training`을 `false`로 만들어 다음 단계에서 놓치지 않게
한다.

## 5. 실제 결과

### 5.1 `train_only`: 주 분석

첫 20일 안의 평일 14일을 사용했다.

- 비어 있지 않은 그룹: 22개
- 빈 그룹: 2, 4
- seed 후보 `size > 10`: 0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
  18, 19, 20, 21, 22, 23
- seed에서만 제외된 작은 그룹: 1, 3, 5, 6
- seed: 12와 20
- seed PCC: `0.5332832372`

오름차순 주 결과는 다음과 같다.

| Cluster | 포함된 초기 group | 전체 셀 | 중앙 900셀 |
|---:|---|---:|---:|
| 0 | 3, 5, 6, 8–17 | 5,409 | 611 |
| 1 | 0, 1, 7, 18–23 | 4,591 | 289 |

내림차순으로 다시 배정해도 group 구성과 10,000셀 membership이 모두 같았다. 즉 이
데이터와 프로토콜에서는 논문의 미공개 순서가 주 분석 결과를 바꾸지 않았다.

### 5.2 `algorithm1_full_month`: 논문 민감도

11월 전체의 평일 21일을 사용했다.

- 비어 있지 않은 그룹: 22개
- 빈 그룹: 2, 4
- seed 후보 `size > 10`: 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
  19, 20, 21, 22, 23
- seed에서만 제외된 작은 그룹: 1, 3, 5, 6, 7
- seed: 11과 21
- seed PCC: `0.4925938335`

| 순서 | Cluster 0 초기 group | Cluster 1 초기 group | 전체 셀 수 | 중앙 900셀 수 |
|---|---|---|---:|---:|
| 오름차순 주 구현 | 3, 5, 6, 11 | 0, 1, 7–10, 12–23 | 282 / 9,718 | 77 / 823 |
| 내림차순 민감도 | 3, 5–17 | 0, 1, 18–23 | 5,234 / 4,766 | 624 / 276 |

두 membership의 label-swap 불변 일치율은 전체 셀 `50.48%`, 중앙 900셀
`60.78%`다. 오름차순의 282 / 9,718이라는 불균형만 보고 알고리즘이 틀렸다고 말할
수는 없다. 같은 seed와 PCC에서도 순차 평균의 경로 의존성이 이렇게 큰 결과를 만들 수
있다는 것을 확인한 것이다.

### 5.3 논문 Fig. 5와의 진단 비교

논문 Fig. 5는 group 8–18이 한 cluster, 나머지가 다른 cluster라고 제시한다. cluster
번호 자체는 임의이므로 label을 바꿔 가장 많이 일치하는 경우로 비교했다.

| 프로토콜·순서 | 비어 있지 않은 group 일치 | 셀 가중 일치 |
|---|---:|---:|
| `train_only` 오름/내림 | 81.82% | 79.50% |
| 전체 월 오름차순 | 59.09% | 70.89% |
| 전체 월 내림차순 | 77.27% | 78.47% |

내림차순 전체 월이 Fig. 5에 더 가깝다는 사실은 진단 결과일 뿐, 그 순서를 사후
선택하는 근거가 아니다. Fig. 4 초기 그룹부터 정확히 일치하지 않으며 Fig. 5 생성
순서도 공개되지 않았기 때문이다.

### 5.4 두 프로토콜 비교

오름차순 주 구현끼리 비교한 label-swap 불변 membership 일치율은 전체 10,000셀
`51.53%`, 중앙 900셀 `59.56%`였다. 두 경로는 scaling 적합 기간뿐 아니라 각 셀의
초기 peak 그룹, seed와 순차 병합 경로까지 달라진다. 따라서 전체 월 UPC를 미리 만든
뒤 Train/Validation/Test에 공통 적용하는 것은 단순한 편의가 아니라 상당한 미래 정보
영향을 포함할 수 있다.

## 6. 산출물과 결정성

모든 산출물은 재생성 가능한 파생 데이터라
`data/processed/upc/final_clusters/`에 저장하고 Git에서는 제외한다.

| 파일 | 내용 |
|---|---|
| `{protocol}_group_profiles.npy` | 빈 group은 0행으로 둔 `(24, 24)` float64 profile |
| `{protocol}_profile_valid.npy` | 실제 profile이 있는 24개 group mask |
| `{protocol}_nonempty_group_ids.npy` | PCC 행·열에 대응하는 group ID |
| `{protocol}_pcc.npy` | 비어 있지 않은 22개 group의 `(22, 22)` PCC |
| `group_assignments.csv` | group 크기, seed 자격, 오름/내림 cluster |
| `all_cell_memberships.csv` | 10,000셀의 두 프로토콜 초기·최종 membership |
| `central_900_memberships.csv` | 중앙 900셀의 좌표와 초기·최종 membership |
| `summary.json` | 결정론적 계약, 결과, 비교와 중단선 |
| `manifest.json` | 입력·출력 checksum, 환경, 실행시간과 최대 RSS |

manifest를 제외한 12개 파일을 연속 두 번 생성해 SHA-256이 모두 같음을 확인했다.
manifest는 실행 시각과 소요 시간을 포함하므로 매번 달라진다. 저장된 PCC는 별도의
`numpy.corrcoef` 계산과 절대오차 `1e-12` 안에서 일치했고, CSV에는 전체 10,000행과
중앙 900행이 정확히 한 번씩 들어 있다.

## 7. 테스트와 자원 사용

추가한 합성 테스트는 다음을 검증한다.

- 셀별 scaling 후 셀·평일 평균 profile 계산
- cell chunk 크기와 C/Fortran 메모리 배치에 대한 결정성
- PCC의 +1, -1, 대칭성, 대각선과 범위
- 분산 0 profile 거부
- `size=10` 제외, `size=11` 포함
- 최소 PCC seed 및 사전식 동률 처리
- 작은 비어 있지 않은 group 배정과 빈 group 제외
- cluster 점수 동률 시 작은 cluster ID 선택
- 오름차순/내림차순이 실제로 다른 결과를 만들 수 있음
- label swap 불변 일치율
- 초기 membership checksum 변조 차단

실제 WSL CPU 실행은 약 `0.29초`, 최대 RSS 약 `224MiB`, 계산 chunk 추정치는 약
`10.8MiB`였다. `traffic.npy`는 memory map으로 읽고 셀 256개씩만 float64 작업
배열로 변환한다. 32GB RAM 중 Codex와 다른 앱이 절반가량을 쓰는 상황에서도 충분히
작으며, GPU 전송이 더 큰 비용이라 이 단계에는 Colab을 사용하지 않는다. Colab은
후속 신경망 학습에만 사용한다.

## 8. 학습 관점의 해석과 다음 중단선

이번 단계의 가장 중요한 학습은 PCC 공식보다 **순차 clustering에서 생략된 반복
순서도 모델 입력을 바꿀 수 있는 실험 변수**라는 점이다. `train_only`에서는 우연히
경로가 수렴했지만 전체 월에서는 절반 가까운 셀이 달라졌다. 따라서 논문 그림에 더
가깝거나 cluster 크기가 균형적이라는 이유로 결과를 사후 선택하면 재현이 아니라
튜닝이 된다.

현재 최우선 후속 작업은 고비용 LSTM/RCTL 실행이 아니다. 먼저 제한된 설계 검토에서
다음을 결정해야 한다.

1. 논문에 명시된 것으로 볼 수 있는 오름차순만 재현 경로로 유지할지,
2. 순서 불변 대안은 별도 확장 실험으로만 둘지,
3. 전체 월 민감도 모델을 보류하고 순서에 안정적인 `train_only` 주 분석만 먼저
   학습할지.

이 결정 전에는 결과를 보고 순서를 더 탐색하거나 모델 성능으로 membership을 선택하지
않는다. 현재 manifest는 이 중단선을 `ready_for_expensive_model_training=false`로
기록한다.
