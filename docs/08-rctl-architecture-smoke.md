# RCTL 아키텍처 계약과 Colab T4 과적합 smoke

## 1. 결론

논문 그림을 해석한 `paper_interpretation`과 원저자 `main.py`를 전사한
`public_reference`를 별도 모델로 구현하고, 실제 Milan Train 데이터의 작은
부분집합으로 구조와 학습 경로를 검증했다.

- 두 변형 모두 출력 shape, residual shape, 유한한 forward/backward, causal
  convolution 검사를 통과했다.
- 논문 해석형의 실제 parameter 수는 `236,657`로 논문 Table III의 `173,633`과
  `63,024`개 다르다.
- 공개 코드형의 실제 parameter 수는 `173,665`로 논문 값과 `32`개 차이다.
- 논문 해석형은 1,024개 Train 표본에서 평가 모드 MAE를 `199.9471`에서
  `16.2307`로 `91.88%` 낮췄고, 같은 표본의 Persistence MAE `18.3883`보다 낮아
  사전 등록한 과적합 smoke 기준을 통과했다.
- 독립된 Colab T4 세션에서 같은 실행을 두 번 수행했을 때 7개 핵심 학습 수치가
  부동소수점 값까지 같았다.

이 결과는 **구현이 학습 가능한지 확인한 진단 결과**이지 Validation/Test 성능이
아니다. 논문 수치 재현이나 일반화 성능의 근거로 사용하지 않는다.

## 2. 이 검사를 전체 학습보다 먼저 한 이유

RCTL은 convolution, LSTM과 두 종류의 residual connection이 겹쳐 있다. shape가
우연히 broadcast되거나 미래 시점이 convolution에 들어가거나 gradient가 일부 경로에
도달하지 않는다면, 중앙 900셀을 오래 학습해도 결과를 해석할 수 없다. 작은 표본을
의도적으로 외우게 하는 smoke는 다음 질문을 약 1분의 GPU 학습으로 먼저 답한다.

1. 모델 입력 `(batch, 8, 1)`이 출력 `(batch, 1)`로 연결되는가?
2. 모든 `Add` 입력 shape가 정확히 같아 암묵적 broadcast가 없는가?
3. 뒤쪽 입력을 바꿔도 앞쪽 causal convolution 출력이 변하지 않는가?
4. 모든 trainable variable에 유한하고 0이 아닌 gradient가 도달하는가?
5. 실제 데이터의 window와 target 정렬을 모델이 학습할 수 있는가?

이 단계가 실패하면 대규모 학습이나 하이퍼파라미터 탐색으로 넘어가지 않는 것이
실패 원인을 가장 싸게 분리하는 방법이다.

## 3. 결과를 보기 전에 고정한 두 아키텍처

공통 channel은 `[16, 32, 64, 64, 32, 16]`, 입력 길이는 8이다. block 번호와
RCC route는 코드에서 0부터 센다. 아래 표에는 사람이 읽기 쉽도록 1부터 센 block도
함께 설명한다.

| 항목 | `paper_interpretation` | `public_reference` |
|---|---|---|
| 근거 | 논문 Fig. 2, Table I, Appendix Fig. 12 | 공개 `main.py` 동작 |
| Conv1D kernel | 4 | 3 |
| dilation | `1, 2, 4, 8, 16, 32` | `1, 2, 4, 6, 8, 10` |
| TCN branch와 shortcut 결합 | `Concatenate` | `Add` |
| RCC-1 | block 입력을 1×1 projection한 뒤 LSTM 출력과 `Add` | 동일 |
| RCC-2 | B3→B4, B2→B5, B1→B6 | B3 자기 projection 추가 후 B3→B4, B2→B5, B1→B6 |
| 마지막 입력 shortcut | 입력을 16 channel 1×1 projection 후 `Add` | 동일 |
| 학습 대상 | 최초 smoke의 주 구현 | 구조 비교만 수행 |

논문 그림의 RCC 연산 순서는 완전히 공개되지 않았다. 따라서 논문형은 “정답 구현”이
아니라 그림을 일관된 tensor shape로 옮긴 **명시적 해석**이다. 공개 코드형과 같은
함수 안에서 조건을 숨기지 않고 config에 kernel, dilation, 결합 방식과 route를 모두
기록했다.

## 4. 실제 데이터 smoke 표본

`central-900-approximate`의 30×30 격자에서 행과 열을 각각 등간격으로 네 개씩
선택했다. 결과를 보고 고른 셀이 아니며, 고정된 4×4 교차점이다.

| grid row | 선택한 cell ID (grid column 35, 45, 54, 64 순) |
|---:|---|
| 35 | 3536, 3546, 3555, 3565 |
| 45 | 4536, 4546, 4555, 4565 |
| 54 | 5436, 5446, 5455, 5465 |
| 64 | 6436, 6446, 6455, 6465 |

Train target 2,872개에서도 첫 index 8과 마지막 index 2,879를 포함해 64개를
등간격으로 선택했다. 각 셀에 같은 target index를 사용하므로 총 표본은
`16 × 64 = 1,024`개다.

```text
X[cell, t-8:t] -> y[cell, t]
X shape: (1024, 8, 1)
y shape: (1024, 1)
scaling: 없음, float32 raw traffic
```

선택된 target과 입력 window에는 `missing_mask` 또는 `internet_null_mask`가 표시된
값이 하나도 없었다. 입력 NPZ는 `39,521 bytes`, SHA-256은
`89b70bee8e0780f2e9f339254a8ba795f65f06dc952b0108c58d7cd3494292d1`이다.
전체 `(10000, 4320)` 배열은 memory map으로 읽고 선택값만 복사했으며 로컬 준비
과정의 최대 RSS는 `50,667,520 bytes`, 약 48.3MiB였다. 따라서 32GB 노트북에서
Codex와 다른 앱이 메모리를 사용 중인 상황도 고려한 작은 로컬 작업이다.

## 5. 구조 감사 결과

| 검사 | 논문 해석형 | 공개 코드형 |
|---|---:|---:|
| 실제 출력 shape `(3, 1)` | 통과 | 통과 |
| 모든 residual 입력 shape 정확히 일치 | 통과 | 통과 |
| forward 값 NaN/Inf 없음 | 통과 | 통과 |
| 12개 causal Conv1D의 앞쪽 출력 최대 변화 | `0.0` | `0.0` |
| gradient가 없는 trainable variable | 0 / 100 | 0 / 100 |
| 0이 아닌 gradient가 있는 trainable variable | 100 / 100 | 100 / 100 |
| 실제 parameter 수 | 236,657 | 173,665 |

### 5.1 논문 parameter 수와의 불일치

논문형의 `Concatenate`는 각 LSTM에 들어가는 feature 수를 두 배로 만든다. kernel
4의 convolution parameter도 공개 코드의 kernel 3보다 많다. 그 결과 논문 그림을
따른 현재 해석형은 논문 표의 `173,633`보다 `63,024`개 많았다. 이는
`GAP-RCTL-01`부터 `GAP-RCTL-05`가 실제로 독립 문제가 아니라 서로 얽혀 있음을
보여준다.

추가로 다음 산술 관계를 발견했다.

```text
public_reference 실제 수                    173,665
- final_input_projection의 parameter 수          32
= 논문 Table III 수                         173,633
```

정확한 일치는 중요한 단서지만, 논문이 해당 layer를 세지 않았다는 증거는 아니다.
그림의 미공개 연산, parameter 집계 방식 또는 공개 코드와 최종 실험 코드의 차이도
가능하다. 특히 이 관계는 결과를 본 뒤 발견한 **사후 진단**이므로, 숫자를 맞추기 위해
주 구현에서 layer를 제거하지 않는다. 원저자의 최종 학습 코드나 model summary가
확인되기 전까지 parameter gap은 해결되지 않은 상태로 유지한다.

## 6. Colab T4 과적합 진단

### 6.1 실행 환경

| 항목 | 값 |
|---|---|
| GPU | Tesla T4, 15,360MiB, compute capability 7.5 |
| Python | 3.13.15 |
| NumPy | 2.1.3 |
| TensorFlow | 2.20.0 |
| tf.keras | 3.13.2 |
| CUDA / cuDNN | 12.5.1 / 9 |
| seed | 42 |
| optimizer / learning rate | Adam / 0.001 |
| loss / batch | MAE / 512 |
| dropout / shuffle | 0.05 / `false` |
| epoch | 200 |

요청한 T4 이름과 TensorFlow GPU device가 모두 확인되지 않으면 학습 전에 중단한다.
런타임에서 이미 위 버전이 제공되어 재설치하지 않았고, 확인한 조합을
`requirements/model.txt`에 기록했다.

### 6.2 사전 등록 기준과 실제 결과

| 판정 항목 | 기준 | 결과 | 판정 |
|---|---:|---:|---:|
| prefit 대비 final 평가 모드 MAE 감소 | 80% 이상 | 91.8825% | 통과 |
| final Train MAE | Persistence보다 작음 | 16.2307 < 18.3883 | 통과 |
| 학습·추론 값 | 모두 유한 | NaN/Inf 없음 | 통과 |

세부 수치는 다음과 같다.

| 수치 | 값 |
|---|---:|
| prefit Train MAE | 199.947148 |
| 첫 epoch 학습 loss | 160.004166 |
| 마지막 epoch 학습 loss | 16.257734 |
| final Train MAE (`training=false`) | 16.230687 |
| Persistence Train MAE | 18.388345 |
| final / Persistence | 0.882662 |
| 순수 `model.fit` 시간 | 약 60.27초 |
| 구조 감사 포함 전체 시간 | 약 74.68초 |
| Colab 프로세스 최대 RSS | 약 1.86GiB |

첫 epoch에는 TensorFlow graph compile 비용이 포함되어 약 15.60초가 걸렸고, 200개
epoch 각각의 시간은 결과 JSON에 보존했다. 첫 실행과 metadata 보강 후 최종 실행의
위 7개 핵심 학습 수치는 모두 정확히 같았다.

같은 Train 표본으로 학습하고 평가했으므로 이 수치는 모델의 일반화 능력을 말하지
않는다. Persistence를 이긴 것도 “RCTL이 실제 예측에서 더 좋다”는 뜻이 아니라,
현재 계산 그래프와 데이터 정렬이 작은 실제 표본을 학습할 수 있다는 뜻뿐이다.

## 7. 실행 방법

### 7.1 로컬 입력 준비와 테스트

```bash
.venv/bin/python -m scripts.prepare_rctl_smoke \
  --config configs/rctl_smoke_milan_nov2013.json
.venv/bin/python -m unittest discover -s tests -v
```

입력 준비는 전처리 manifest와 다섯 입력 배열, 중앙 900셀 manifest와 CSV checksum을
먼저 검증한다. 실패하면 NPZ와 manifest를 게시하지 않는다.

### 7.2 최소 Colab bundle

아래 파일만 ZIP에 넣는다. 원본 데이터, 전체 전처리 배열과 논문 PDF는 업로드하지
않는다.

```bash
zip -FS data/interim/rctl_smoke/colab_bundle.zip \
  scripts/__init__.py \
  scripts/build_upc_initial_groups.py \
  scripts/rctl_contract.py \
  scripts/rctl_model.py \
  scripts/run_rctl_smoke.py \
  configs/rctl_smoke_milan_nov2013.json \
  configs/naive_baselines_milan_nov2013.json \
  configs/upc_milan_nov2013.json \
  requirements/model.txt \
  data/interim/rctl_smoke/input.npz \
  data/interim/rctl_smoke/input_manifest.json
```

### 7.3 `google-colab-cli` 실행과 회수

```bash
colab new --session gecos-rctl-smoke --gpu T4
colab exec --session gecos-rctl-smoke \
  --file scripts/probe_colab_runtime.py --timeout 120
colab upload --session gecos-rctl-smoke \
  data/interim/rctl_smoke/colab_bundle.zip \
  /content/gecos_rctl_smoke_bundle.zip
colab exec --session gecos-rctl-smoke \
  --file scripts/colab_rctl_entry.py --timeout 1800
colab download --session gecos-rctl-smoke \
  /content/gecos_rctl_smoke_outputs.zip \
  data/interim/rctl_smoke/colab_outputs.zip
colab stop --session gecos-rctl-smoke
```

성공 여부와 관계없이 마지막 `stop`을 실행해야 한다. 다운로드한 ZIP에는 세 JSON만
있어야 하며, 경로를 확인한 뒤 푼다.

```bash
unzip -l data/interim/rctl_smoke/colab_outputs.zip
unzip -o data/interim/rctl_smoke/colab_outputs.zip -d .
```

## 8. 산출물

모든 실행 산출물은 파생 데이터이므로 Git에서 제외한다.

| 파일 | 내용 | 최종 SHA-256 |
|---|---|---|
| `data/interim/rctl_smoke/input.npz` | 1,024개 실제 Train window | `89b70bee8e0780f2e9f339254a8ba795f65f06dc952b0108c58d7cd3494292d1` |
| `data/interim/rctl_smoke/input_manifest.json` | 셀·target 선택, 원본 및 배열 checksum | 실행 metadata 포함 |
| `data/processed/rctl_smoke/architecture_report.json` | 두 변형의 layer shape, parameter, causal, gradient 감사 | `7ffe6054971f6ee1167547a7f77c3870851c36b7dc2503e97ff53e9afbd6afb4` |
| `data/processed/rctl_smoke/overfit_report.json` | 200 epoch history, 기준선과 통과 판정 | `530c56fb72dc71a232fc30bab31ac076140dff0f4d2c849680271239e14662df` |
| `data/processed/rctl_smoke/manifest.json` | 입력·config·환경·출력 checksum | 실행 metadata 포함 |

smoke 모델은 일회성 진단 도구이므로 checkpoint를 보존하지 않는다. 본 학습에서는
Validation MAE 기준 최적 checkpoint와 early stopping 상태를 별도 manifest에 남긴다.

## 9. 여기서 얻은 학습과 당시 다음 단계

이번 단계의 핵심 학습은 논문 그림, 하이퍼파라미터 표, parameter 표와 공개 코드가
동시에 하나의 구조를 가리키지 않는다는 점이다. `Concatenate`와 `Add` 하나의 차이가
LSTM parameter 수를 크게 바꾸며, parameter 표만 보면 오히려 공개 코드형이 훨씬
가깝다. 따라서 이후 성능표에서도 변형 이름을 생략하면 안 된다.

당시 다음 단계는 모델을 바로 900셀에 확대하는 것이 아니라 UPC의 24개 초기 그룹을
PCC로 최종 `N=2` cluster에 결정론적으로 병합하는 것이다. 그 membership을 고정한
뒤 같은 Train/Validation/Test 계약에서 LSTM과 RCTL의 UPC on/off 비교를 수행한다.

### 9.1 학습용 프로젝트의 최종 판정

이후 UPC membership과 학습 정책, LSTM UPC off/on 전체 Train·Validation 비교까지는
완료했다. 그러나 RCTL은 논문형 `236,657`, 공개 코드형 `173,665` parameter로 모두
논문 Table III의 `173,633`과 일치하지 않았다. 어느 하나를 저자 모델로 간주해
900셀 전체 학습을 실행하면 구조 선택의 임의성이 성능 해석보다 커진다.

따라서 이 프로젝트는 RCTL에서 shape, causality, gradient와 작은 실제 표본 overfit을
확인한 구조 감사까지를 최종 범위로 삼는다. 전체 학습과 RCC ablation은 실패로
숨기지 않고 **의도적으로 미실행**한 항목으로 남긴다. 전체 종료 근거는
[학습용 논문 재현 최종 정리](14-study-reproduction-conclusion.md)에 기록한다.
