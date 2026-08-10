# 작업 F-2 평가 — Vy-rmsnorm 기준 C1+C2(전류) 다변량 확장 (채널-as-노드)

seed 2026, window 2048, no-meta, 채널셋 C1C2Vy(=3노드), 대표 gate fold. 점수=순수 shape flow_NLL. 기준선 Vy(단일채널 rmsnorm)=F-1 결과 재사용(재학습 없음).

- **all**: 전 채널 rmsnorm(shape-only) — 전류 파형 shape가 독립 fault 정보를 주는지 순수 검증.
- **mixed**: Vy-rmsnorm + 전류 raw(진폭 보존) — 전류 절대 진폭의 조건/진폭 confound 재주입 노출.

## 핵심 요약 (F-2 검증 목표별)

- **분리 AUROC (Vy 기준선 대비)**: zero-support Vy 0.747 → all 0.631(Δ-0.116), mixed 0.321(Δ-0.426); compositional Vy 0.919 → all 0.381(Δ-0.538), mixed 0.154(Δ-0.765).
- **ρ(RMS_current, flow) [confound 재주입 지표]**: all zero -0.024/comp -0.069; mixed zero 0.532/comp 0.849 (0 근방=전류 진폭 무상관, 큰 양수=confound 재주입).
- **ρ(RMS_Vy, flow)**: all zero 0.084/comp 0.213; mixed zero 0.015/comp -0.019 (F-1 Vy 기준선 ≈ -0.36/-0.27).
- **저진폭 target(LONO2/4/5)**: Vy 0.841(역전 0/4) → all 0.632(역전 1/4), mixed 0.208(역전 4/4).
- **정상 FPR**: 고진폭 all zero 0.141/comp 0.288; mixed zero 0.049/comp 0.107 / 저진폭 all zero 0.134/comp 0.266; mixed zero 0.078/comp 0.284.

## 1) fold 유형별 집계 (Vy 기준선 vs 구성)

| fold 유형 | n | AUROC(Vy) | AUROC(all)/Δ | AUROC(mixed)/Δ | ρ_cur(all) | ρ_cur(mixed) | ρ_vib(all) | ρ_vib(mixed) |
|---|---|---|---|---|---|---|---|---|
| zero-support | 6 | 0.747 | 0.631/-0.116 | 0.321/-0.426 | -0.024 | 0.532 | 0.084 | 0.015 |
| compositional | 2 | 0.919 | 0.381/-0.538 | 0.154/-0.765 | -0.069 | 0.849 | 0.213 | -0.019 |

## 2) target-normal 진폭군별 분리 AUROC (저진폭 역전 진단)

| target 진폭군 | n | AUROC(Vy) | 역전(Vy) | AUROC(all)/역전 | AUROC(mixed)/역전 |
|---|---|---|---|---|---|
| 고진폭(K001/K003/K006) | 4 | 0.739 | 1 | 0.505/2 | 0.350/3 |
| 저진폭(K002/K004/K005) | 4 | 0.841 | 0 | 0.632/1 | 0.208/4 |

## 3) per-fault AUROC 집계 (fault id별 fold 평균) — KA15·KA22 강조

| fault id | AUROC(Vy) | AUROC(all) | AUROC(mixed) |
|---|---|---|---|
| KA04 | 0.822 | 0.641 | 0.119 |
| KA15 ⚠ | 0.745 | 0.505 | 0.384 |
| KA16 | 0.855 | 0.661 | 0.160 |
| KA22 ⚠ | 0.741 | 0.526 | 0.138 |
| KA30 | 0.760 | 0.561 | 0.242 |
| KB23 | 0.910 | 0.662 | 0.602 |
| KB24 | 0.859 | 0.678 | 0.328 |
| KB27 | 0.765 | 0.496 | 0.218 |
| KI04 | 0.828 | 0.616 | 0.374 |
| KI14 | 0.759 | 0.496 | 0.192 |
| KI16 | 0.818 | 0.538 | 0.196 |
| KI17 | 0.782 | 0.527 | 0.299 |
| KI18 | 0.629 | 0.589 | 0.124 |
| KI21 | 0.785 | 0.463 | 0.530 |

> ⚠ = 작업 B/F-0/F-1에서 불가시(AUROC≈0.5) 이력. 전류 추가로 상승하면 독립 정보 보완 신호.

## 4) fold 상세 (AUROC + mean adjacency A[Vy행: 전류→진동 기여])

| split | LONO | 유형 | target | AUROC(Vy) | AUROC(all) | AUROC(mixed) | A[Vy←C1,C2](all) | A[Vy←C1,C2](mixed) |
|---|---|---|---|---|---|---|---|---|
| 123to0 | 1 | compositional | K001 | 0.933 | 0.215 | 0.056 | 0.36,0.46 | 0.33,0.33 |
| 123to0 | 2 | compositional | K002 | 0.905 | 0.547 | 0.252 | 0.33,0.33 | 0.41,0.00 |
| 023to1 | 1 | zero-support | K001 | 0.887 | 0.990 | 0.441 | 0.33,0.33 | 0.11,0.24 |
| 023to1 | 2 | zero-support | K002 | 0.575 | 0.444 | 0.039 | 0.17,0.22 | 0.33,0.33 |
| 013to2 | 1 | zero-support | K001 | 0.983 | 0.565 | 0.775 | 0.00,0.27 | 0.00,0.11 |
| 013to2 | 2 | zero-support | K002 | 0.935 | 0.597 | 0.348 | 0.33,0.33 | 0.33,0.33 |
| 012to3 | 1 | zero-support | K001 | 0.153 | 0.252 | 0.126 | 0.33,0.33 | 0.06,0.25 |
| 012to3 | 2 | zero-support | K002 | 0.949 | 0.940 | 0.194 | 0.33,0.33 | 0.33,0.33 |

## 판정 가이드 (go/no-go)

- **독립 fault 정보 보완**: 분리 AUROC↑(Δ>0) & 저진폭 target 개선·역전 감소 & KA15/KA22 상승 & ρ(RMS_current) 무상관 & adjacency가 전류→진동 비자명 기여.
- **confound 재주입**: ρ(RMS_current)↑(양수) & 저진폭 target 악화 & 고진폭 정상 FPR↑ (특히 mixed·zero-support 023to1/013to2).
- all은 개선·mixed는 악화면 → 전류 shape=독립 정보, 절대 진폭=confound로 분리 결론.

## 주의 / 한계

- 무학습 재추론(학습된 F-2 체크포인트), seed 2026 단일. target-normal이 setting+bearing 이중 held-out → 개체차 confound 잔존(F-0/B/F-1과 정합).
- adjacency는 softmax(dim=1) 참조 가중; 절대적 인과 아님(정보 흐름 대리 지표).

