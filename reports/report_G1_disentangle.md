# 작업 G-1 — Dual-branch Shape/Amplitude Disentangle (Paderborn no-meta, 무학습 재추론)

seed 2024, window 2048, no-meta LOSO 4 split × 6 LONO (24 fold). G-1 = amp_normalize(shape-only window) + amp_branch(조건부 진폭 head, Joint 학습). S_shape=flow NLL, S_amp=-log N(a|h_shape), S_total=z_val(S_shape)+z_val(S_amp)(val-normal 표준화, label-free). raw baseline = 동일 seed B3 checkpoint flow_NLL, 동일 fold paired.

## 핵심 요약 (go/no-go ①~⑤)

- **전체 AUROC** (24 fold 평균): S_total 0.683 / S_shape 0.733 / S_amp 0.460 vs **raw 0.715** (가드레일 no-meta LOSO 0.696).
- **① 고진폭 정상 S_amp 과도 상승 여부**: 정상 FPR(S_amp) 고진폭 0.131 vs 저진폭 0.149; 정상 mean(S_amp) 고 1.452 / 저 1.862. (고진폭에서만 FPR·mean이 튀면 ① 실패 = 조건부 진폭이 진폭 크기를 그대로 벌함.)
- **② amplitude-sensitive fault 회복(S_amp)**: raw 0.897 → S_shape 0.701 (rmsnorm에서 잃음) → **S_amp 0.595** → S_total 0.715. (S_amp가 raw 수준으로 회복하면 ② 성공.)
- **③ shape-sensitive fault 유지(S_shape)**: raw 0.473 → **S_shape 0.776** → S_amp 0.280 → S_total 0.641. (S_shape가 rmsnorm 회복 수준을 유지하면 ③ 성공.)
- **④ 결합 희생 여부(S_total)**: amp-sensitive S_total 0.715 / shape-sensitive S_total 0.641 — 두 군 모두 각 branch 최고치에 근접하면 ④ 성공(한쪽 희생 없음).
- **⑤ raw 대비 paired**: S_total−raw Δ = zero-support 0.008 / compositional -0.151. 정상 FPR(S_total) 고 0.122/저 0.274 vs raw 고 0.515/저 0.089.
- **ρ(RMS, score) 완화**: zero-support S_shape -0.411 / S_amp -0.037 / S_total -0.360 vs raw 0.964 (작업 B raw ρ≈0.96).

## 1) fold 유형별 집계

| fold 유형 | n | S_total | S_shape | S_amp | raw | Δ(total−raw) | ρ_total | ρ_amp | ρ_raw |
|---|---|---|---|---|---|---|---|---|---|
| zero-support | 18 | **0.702** | 0.806 | 0.450 | 0.694 | 0.008 | -0.360 | -0.037 | 0.964 |
| compositional | 6 | **0.627** | 0.514 | 0.490 | 0.778 | -0.151 | -0.158 | -0.017 | 0.975 |

## 2) fault군별 branch AUROC — ②(amp-sensitive) / ③(shape-sensitive) / ④(결합)

| fault군 | raw | S_shape | S_amp | S_total |
|---|---|---|---|---|
| amp-sensitive (KA04/16/30·KB23/24·KI04/16/18) | 0.897 | 0.701 | **0.595** | 0.715 |
| shape-sensitive (KA15/22·KB27·KI14/17/21) | 0.473 | **0.776** | 0.280 | 0.641 |

## 3) per-fault AUROC (fault id별 fold 평균)

| fault id | 분류 | raw | S_shape | S_amp | S_total |
|---|---|---|---|---|---|
| KA04 | A | 0.910 | 0.737 | 0.664 | 0.805 |
| KA15 | S | 0.421 | 0.771 | 0.267 | 0.630 |
| KA16 | A | 0.940 | 0.754 | 0.696 | 0.811 |
| KA22 | S | 0.235 | 0.763 | 0.470 | 0.681 |
| KA30 | A | 0.851 | 0.656 | 0.385 | 0.539 |
| KB23 | A | 0.935 | 0.682 | 0.704 | 0.741 |
| KB24 | A | 0.992 | 0.653 | 0.888 | 0.858 |
| KB27 | S | 0.495 | 0.784 | 0.220 | 0.634 |
| KI04 | A | 0.853 | 0.682 | 0.390 | 0.546 |
| KI14 | S | 0.467 | 0.773 | 0.222 | 0.627 |
| KI16 | A | 0.869 | 0.761 | 0.510 | 0.740 |
| KI17 | S | 0.585 | 0.779 | 0.260 | 0.639 |
| KI18 | A | 0.825 | 0.686 | 0.519 | 0.681 |
| KI21 | S | 0.634 | 0.784 | 0.242 | 0.635 |

> 분류 A=amplitude-sensitive(raw가 잡고 rmsnorm이 잃음, S_amp로 회복 기대), S=shape-sensitive(raw가 못 잡고 rmsnorm이 회복, S_shape에서 유지 기대).

## 4) ① 정상에서 S_amp 진폭군별 (고진폭 정상 과탐 여부)

| 지표 | 고진폭(K001/K003/K006) | 저진폭(K002/K004/K005) |
|---|---|---|
| 정상 FPR(S_amp @val95p) | 0.131 | 0.149 |
| 정상 mean S_amp | 1.452 | 1.862 |
| 정상 FPR(S_total) | 0.122 | 0.274 |
| 정상 FPR(raw) | 0.515 | 0.089 |

## 5) fold 전체 상세

| split | LONO | 유형 | target | S_total | S_shape | S_amp | raw | Δ(total−raw) |
|---|---|---|---|---|---|---|---|---|
| 123to0 | 1 | compositional | K001 | 0.353 | 0.118 | 0.367 | 0.596 | -0.243 |
| 123to0 | 2 | compositional | K002 | 0.979 | 0.847 | 0.858 | 0.926 | 0.053 |
| 123to0 | 3 | compositional | K003 | 0.408 | 0.327 | 0.507 | 0.577 | -0.168 |
| 123to0 | 4 | compositional | K004 | 0.487 | 0.646 | 0.168 | 0.967 | -0.480 |
| 123to0 | 5 | compositional | K005 | 0.542 | 0.151 | 0.625 | 0.999 | -0.457 |
| 123to0 | 6 | compositional | K006 | 0.996 | 0.996 | 0.412 | 0.604 | 0.391 |
| 023to1 | 1 | zero-support | K001 | 0.829 | 0.996 | 0.643 | 0.126 | 0.704 |
| 023to1 | 2 | zero-support | K002 | 0.817 | 0.900 | 0.564 | 0.857 | -0.040 |
| 023to1 | 3 | zero-support | K003 | 0.996 | 0.996 | 0.871 | 0.313 | 0.684 |
| 023to1 | 4 | zero-support | K004 | 0.248 | 0.928 | 0.045 | 0.885 | -0.637 |
| 023to1 | 5 | zero-support | K005 | 0.648 | 0.925 | 0.108 | 1.000 | -0.352 |
| 023to1 | 6 | zero-support | K006 | 0.050 | 0.001 | 0.413 | 0.095 | -0.046 |
| 013to2 | 1 | zero-support | K001 | 0.451 | 0.969 | 0.386 | 0.598 | -0.147 |
| 013to2 | 2 | zero-support | K002 | 0.724 | 0.497 | 0.857 | 0.879 | -0.155 |
| 013to2 | 3 | zero-support | K003 | 1.000 | 0.998 | 0.517 | 0.697 | 0.303 |
| 013to2 | 4 | zero-support | K004 | 0.865 | 0.916 | 0.186 | 1.000 | -0.135 |
| 013to2 | 5 | zero-support | K005 | 0.983 | 0.893 | 0.632 | 0.900 | 0.084 |
| 013to2 | 6 | zero-support | K006 | 0.947 | 0.997 | 0.401 | 0.624 | 0.322 |
| 012to3 | 1 | zero-support | K001 | 0.745 | 0.999 | 0.336 | 0.541 | 0.204 |
| 012to3 | 2 | zero-support | K002 | 0.891 | 0.647 | 0.732 | 0.922 | -0.031 |
| 012to3 | 3 | zero-support | K003 | 1.000 | 0.999 | 0.495 | 0.647 | 0.352 |
| 012to3 | 4 | zero-support | K004 | 0.257 | 0.613 | 0.072 | 0.976 | -0.720 |
| 012to3 | 5 | zero-support | K005 | 0.427 | 0.283 | 0.469 | 0.909 | -0.482 |
| 012to3 | 6 | zero-support | K006 | 0.757 | 0.957 | 0.370 | 0.524 | 0.232 |

## 주의 / 한계

- 무학습 재추론, seed 2026 단일(가능성 확인용). 판정은 다지표(①~⑤) — AUROC 단일 금지.
- S_total = val-normal 표준화 등가중 합(test 라벨 튜닝 없음). 원시 합은 S_shape가 수천 스케일이라 S_amp가 묻히므로 표준화가 primary.
- **target-normal이 setting+bearing 이중 held-out** → bearing 개체차 confound 잔존(작업 B/F-1과 정합).
- 실패 조건(TODO): fault morphology가 고진폭스러우면 S_amp residual 안 뜸(특히 023→1). 그 경우 shape branch에 latent 거리 병행(F-0 근거)이 후속 옵션.

