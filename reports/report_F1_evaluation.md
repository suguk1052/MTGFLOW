# 작업 F-1 정밀 재평가 — Vy-rmsnorm(shape) vs raw-vib (Paderborn no-meta, 무학습 재추론)

seed 2026, window 2048, no-meta LOSO 4 split × 6 LONO. 학습된 체크포인트 재추론만(새 학습 없음). 점수 = 순수 shape flow_NLL(rms_lambda=0). rmsnorm=작업 D(amp_normalize=True) / raw=작업 B3(amp_normalize=False), 동일 fold paired 비교.

**F-1 배경(계획 B안):** F-1의 표현·채널·split·seed·window는 작업 D와 동일해 D 체크포인트를 재사용한다. F-0이 권한 window 16384/32768은 **적용하지 않는다** — (i) MTGFlow attention이 Linear(c,c), c=window_size라 파라미터가 window에 제곱으로 커져(16384→~805M, 32768→~3.2B) V100-16GB 학습 불가, (ii) F-0의 큰-window 근거는 envelope-spectrum **descriptor**의 주파수 해상도용이며 raw time-window를 그대로 LSTM+flow에 넣는 MTGFlow 학습엔 자동 이식되지 않는다. patchify 구조 변경은 F-1 범위 밖.

## 핵심 요약 (F-1 검증 목표별)

- **분리 AUROC (raw 대비)**: zero-support rmsnorm 0.662 vs raw 0.648 (Δ 0.015); compositional rmsnorm 0.639 vs raw 0.814 (Δ -0.175).
- **저진폭 target 역전**: 저진폭 target fold(LONO2/4/5) raw AUROC 0.835(역전 fold 0/12개) → rmsnorm 0.547(역전 5/12개). 고진폭 target(LONO1/3/6): raw 0.544 / rmsnorm 0.766.
- **ρ(RMS, flow_NLL) 하락 유지**: zero-support raw 0.966→rmsnorm -0.359; compositional raw 0.957→rmsnorm -0.270 (작업 B 기준선 ρ≈0.96).
- **KA15·KA22 per-fault AUROC**: KA15 raw 0.357→rmsnorm 0.606; KA22 raw 0.231→rmsnorm 0.586 (0.5 근방=불가시).
- **정상 FPR 악화 여부**: 고진폭 raw 0.532/0.507(zero/comp)→rmsnorm 0.113/0.070; 저진폭 raw 0.099/0.083→rmsnorm 0.464/0.376.

## 1) fold 유형별 집계 — rmsnorm vs raw

| fold 유형 | n | AUROC(rmsnorm) | AUROC(raw) | Δ | ρ(rmsnorm) | ρ(raw) | 고진폭FPR(rms/raw) | 저진폭FPR(rms/raw) |
|---|---|---|---|---|---|---|---|---|
| zero-support | 18 | **0.662** | 0.648 | 0.015 | -0.359 | 0.966 | 0.113/0.532 | 0.464/0.099 |
| compositional | 6 | **0.639** | 0.814 | -0.175 | -0.270 | 0.957 | 0.070/0.507 | 0.376/0.083 |

## 2) target-normal 진폭군별 분리 AUROC — 저진폭 역전 진단

| target 진폭군 | n | AUROC(rmsnorm) | AUROC(raw) | raw 역전(<0.5) | rmsnorm 역전(<0.5) |
|---|---|---|---|---|---|
| 고진폭(K001/K003/K006) | 12 | 0.766 | 0.544 | 3 | 2 |
| 저진폭(K002/K004/K005) | 12 | 0.547 | 0.835 | 0 | 5 |

## 3) LONO-1(고진폭 K001 target) · LONO-2(저진폭 K002 target) 분해

| split | LONO | target | AUROC(rmsnorm) | AUROC(raw) | Δ | ρ(rms) | ρ(raw) |
|---|---|---|---|---|---|---|---|
| 123to0 | 1 | K001 | **0.933** | 0.504 | 0.429 | 0.397 | 0.970 |
| 123to0 | 2 | K002 | **0.905** | 0.707 | 0.198 | -0.557 | 0.977 |
| 023to1 | 1 | K001 | **0.887** | 0.088 | 0.799 | -0.704 | 0.990 |
| 023to1 | 2 | K002 | **0.575** | 0.581 | -0.005 | -0.508 | 0.991 |
| 013to2 | 1 | K001 | **0.983** | 0.593 | 0.389 | -0.489 | 0.973 |
| 013to2 | 2 | K002 | **0.935** | 0.884 | 0.051 | -0.496 | 0.968 |
| 012to3 | 1 | K001 | **0.153** | 0.511 | -0.358 | 0.346 | 0.978 |
| 012to3 | 2 | K002 | **0.949** | 0.945 | 0.004 | -0.574 | 0.936 |

## 4) per-fault AUROC 집계 (fault id별 fold 평균) — KA15·KA22 강조

| fault id | fold수 | AUROC(rmsnorm) | AUROC(raw) | Δ |
|---|---|---|---|---|
| KA04 | 24 | 0.646 | 0.926 | -0.279 |
| KA15 ⚠ | 24 | 0.606 | 0.357 | 0.248 |
| KA16 | 24 | 0.725 | 0.952 | -0.227 |
| KA22 ⚠ | 24 | 0.586 | 0.231 | 0.355 |
| KA30 | 24 | 0.638 | 0.848 | -0.210 |
| KB23 | 24 | 0.703 | 0.943 | -0.240 |
| KB24 | 24 | 0.659 | 0.994 | -0.335 |
| KB27 | 24 | 0.691 | 0.399 | 0.292 |
| KI04 | 24 | 0.692 | 0.853 | -0.161 |
| KI14 | 24 | 0.659 | 0.355 | 0.304 |
| KI16 | 24 | 0.687 | 0.854 | -0.167 |
| KI17 | 24 | 0.687 | 0.525 | 0.162 |
| KI18 | 24 | 0.496 | 0.806 | -0.310 |
| KI21 | 24 | 0.716 | 0.608 | 0.108 |

> ⚠ = 작업 B/F-0에서 불가시(AUROC≈0.5) 이력이 있는 KA15·KA22. 0.5 근방이면 학습 후에도 불가시.

## 5) fold 전체 상세 (rmsnorm / raw AUROC)

| split | LONO | 유형 | target | AUROC(rms) | AUROC(raw) | Δ |
|---|---|---|---|---|---|---|
| 123to0 | 1 | compositional | K001 | 0.933 | 0.504 | 0.429 |
| 123to0 | 2 | compositional | K002 | 0.905 | 0.707 | 0.198 |
| 123to0 | 3 | compositional | K003 | 0.799 | 0.990 | -0.191 |
| 123to0 | 4 | compositional | K004 | 0.073 | 0.995 | -0.922 |
| 123to0 | 5 | compositional | K005 | 0.437 | 0.988 | -0.551 |
| 123to0 | 6 | compositional | K006 | 0.686 | 0.702 | -0.016 |
| 023to1 | 1 | zero-support | K001 | 0.887 | 0.088 | 0.799 |
| 023to1 | 2 | zero-support | K002 | 0.575 | 0.581 | -0.005 |
| 023to1 | 3 | zero-support | K003 | 0.000 | 0.361 | -0.361 |
| 023to1 | 4 | zero-support | K004 | 0.285 | 0.696 | -0.411 |
| 023to1 | 5 | zero-support | K005 | 0.820 | 0.548 | 0.272 |
| 023to1 | 6 | zero-support | K006 | 0.975 | 0.194 | 0.781 |
| 013to2 | 1 | zero-support | K001 | 0.983 | 0.593 | 0.389 |
| 013to2 | 2 | zero-support | K002 | 0.935 | 0.884 | 0.051 |
| 013to2 | 3 | zero-support | K003 | 0.968 | 0.678 | 0.290 |
| 013to2 | 4 | zero-support | K004 | 0.229 | 1.000 | -0.770 |
| 013to2 | 5 | zero-support | K005 | 0.014 | 0.855 | -0.840 |
| 013to2 | 6 | zero-support | K006 | 0.824 | 0.640 | 0.184 |
| 012to3 | 1 | zero-support | K001 | 0.153 | 0.511 | -0.358 |
| 012to3 | 2 | zero-support | K002 | 0.949 | 0.945 | 0.004 |
| 012to3 | 3 | zero-support | K003 | 0.989 | 0.688 | 0.301 |
| 012to3 | 4 | zero-support | K004 | 0.561 | 1.000 | -0.439 |
| 012to3 | 5 | zero-support | K005 | 0.774 | 0.816 | -0.042 |
| 012to3 | 6 | zero-support | K006 | 0.999 | 0.578 | 0.422 |

## 주의 / 한계 (F-1 위험 반영)

- 무학습 재추론, seed 2026 단일. rmsnorm AUROC는 D 리포트 flow_only와 동일 정의(정합 확인용).
- **target-normal이 setting+bearing 이중 held-out** → bearing 개체차 confound 잔존. 높은 AUROC 일부는 fault가 아니라 진폭군·개체차 분리일 수 있음(F-0·작업 B와 정합).
- F-0의 window 16384/32768은 envelope-spectrum descriptor용이라 MTGFlow raw window 학습에 직접 적용하지 않음(§배경). patchify 구조 변경은 F-1 범위 밖(F-2 이후 별도 검토 가능).
- KA15·KA22 per-fault AUROC가 0.5 근방이면 표현·정규화와 무관하게 밀도모델이 못 가르는 결함(§4).

