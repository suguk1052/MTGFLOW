# 작업 G-3a — Multi-band Amplitude Branch (Paderborn no-meta, 무학습 재추론)

seed 2028, amp_n_bands=6, window 2048, no-meta LOSO 4 split × 6 LONO (24 fold). G-3a = amp_normalize(shape-only window) + amp_branch(K-band 조건부 진폭 head, Joint 학습). S_shape=flow NLL, S_amp=-Σ_k log N(a_k|h_shape)/K, S_total=z_val(S_shape)+z_val(S_amp)(val-normal 표준화, label-free). raw baseline = 동일 seed B3 checkpoint flow_NLL, 동일 fold paired.

## 핵심 요약 (go/no-go ①~⑤)

- **전체 AUROC** (24 fold 평균): S_total 0.711 / S_shape 0.753 / S_amp 0.650 vs **raw 0.690** (가드레일 no-meta LOSO 0.696).
- **① 고진폭 정상 S_amp 과탐 여부**: 정상 FPR(S_amp) 고진폭 0.128 vs 저진폭 0.176; 정상 mean(S_amp) 고 1.436 / 저 2.022.
- **② amp-sensitive fault**: raw 0.884 → S_shape 0.731 → **S_amp 0.659** → S_total 0.694.
- **③ shape-sensitive fault**: raw 0.432 → **S_shape 0.783** → S_amp 0.638 → S_total 0.735.
- **④ 결합 희생 여부(S_total)**: amp-sensitive S_total 0.694 / shape-sensitive S_total 0.735.
- **⑤ raw 대비 paired**: S_total−raw Δ = zero-support -0.026 / compositional 0.163. 정상 FPR(S_total) 고 0.207/저 0.310 vs raw 고 0.545/저 0.102.
- **ρ(RMS, score)**: zero-support S_shape -0.161 / S_amp -0.099 / S_total -0.187 vs raw 0.966 (작업 B raw ρ≈0.96).

## 1) fold 유형별 집계

| fold 유형 | n | S_total | S_shape | S_amp | raw | Δ(total−raw) | ρ_total | ρ_amp | ρ_raw |
|---|---|---|---|---|---|---|---|---|---|
| zero-support | 18 | **0.643** | 0.685 | 0.628 | 0.669 | -0.026 | -0.187 | -0.099 | 0.966 |
| compositional | 6 | **0.917** | 0.956 | 0.714 | 0.754 | 0.163 | -0.219 | -0.073 | 0.982 |

## 2) fault군별 branch AUROC — ②(amp-sensitive) / ③(shape-sensitive) / ④(결합)

| fault군 | raw | S_shape | S_amp | S_total |
|---|---|---|---|---|
| amp-sensitive (KA04/16/30·KB23/24·KI04/16/18) | 0.884 | 0.731 | **0.659** | 0.694 |
| shape-sensitive (KA15/22·KB27·KI14/17/21) | 0.432 | **0.783** | 0.638 | 0.735 |

## 3) per-fault AUROC (fault id별 fold 평균)

| fault id | 분류 | raw | S_shape | S_amp | S_total |
|---|---|---|---|---|---|
| KA04 | A | 0.913 | 0.750 | 0.891 | 0.825 |
| KA15 | S | 0.356 | 0.779 | 0.566 | 0.699 |
| KA16 | A | 0.932 | 0.778 | 0.864 | 0.812 |
| KA22 | S | 0.225 | 0.776 | 0.704 | 0.758 |
| KA30 | A | 0.841 | 0.692 | 0.351 | 0.478 |
| KB23 | A | 0.912 | 0.758 | 0.678 | 0.708 |
| KB24 | A | 0.985 | 0.709 | 0.848 | 0.843 |
| KB27 | S | 0.451 | 0.786 | 0.623 | 0.732 |
| KI04 | A | 0.837 | 0.743 | 0.330 | 0.481 |
| KI14 | S | 0.415 | 0.786 | 0.599 | 0.719 |
| KI16 | A | 0.850 | 0.775 | 0.656 | 0.725 |
| KI17 | S | 0.547 | 0.782 | 0.770 | 0.794 |
| KI18 | A | 0.801 | 0.641 | 0.654 | 0.677 |
| KI21 | S | 0.596 | 0.786 | 0.564 | 0.707 |

## 4) ① 정상에서 S_amp 진폭군별 (고진폭 정상 과탐 여부)

| 지표 | 고진폭(K001/K003/K006) | 저진폭(K002/K004/K005) |
|---|---|---|
| 정상 FPR(S_amp @val95p) | 0.128 | 0.176 |
| 정상 mean S_amp | 1.436 | 2.022 |
| 정상 FPR(S_total) | 0.207 | 0.310 |
| 정상 FPR(raw) | 0.545 | 0.102 |

## 5) fold 전체 상세

| split | LONO | 유형 | target | S_total | S_shape | S_amp | raw | Δ(total−raw) |
|---|---|---|---|---|---|---|---|---|
| 123to0 | 1 | compositional | K001 | 0.888 | 0.996 | 0.606 | 0.515 | 0.373 |
| 123to0 | 2 | compositional | K002 | 1.000 | 0.933 | 1.000 | 0.849 | 0.151 |
| 123to0 | 3 | compositional | K003 | 1.000 | 0.997 | 0.963 | 0.621 | 0.379 |
| 123to0 | 4 | compositional | K004 | 0.736 | 0.856 | 0.508 | 1.000 | -0.264 |
| 123to0 | 5 | compositional | K005 | 0.916 | 0.962 | 0.811 | 0.927 | -0.011 |
| 123to0 | 6 | compositional | K006 | 0.964 | 0.991 | 0.397 | 0.612 | 0.352 |
| 023to1 | 1 | zero-support | K001 | 0.329 | 0.014 | 0.824 | 0.082 | 0.247 |
| 023to1 | 2 | zero-support | K002 | 0.837 | 0.995 | 0.532 | 0.749 | 0.088 |
| 023to1 | 3 | zero-support | K003 | 0.999 | 1.000 | 0.923 | 0.310 | 0.690 |
| 023to1 | 4 | zero-support | K004 | 0.187 | 0.882 | 0.008 | 0.480 | -0.293 |
| 023to1 | 5 | zero-support | K005 | 0.045 | 0.225 | 0.183 | 1.000 | -0.955 |
| 023to1 | 6 | zero-support | K006 | 0.039 | 0.004 | 0.511 | 0.304 | -0.265 |
| 013to2 | 1 | zero-support | K001 | 0.760 | 0.965 | 0.618 | 0.605 | 0.155 |
| 013to2 | 2 | zero-support | K002 | 1.000 | 0.909 | 1.000 | 0.870 | 0.129 |
| 013to2 | 3 | zero-support | K003 | 1.000 | 0.994 | 0.967 | 0.649 | 0.351 |
| 013to2 | 4 | zero-support | K004 | 0.744 | 0.846 | 0.521 | 0.972 | -0.227 |
| 013to2 | 5 | zero-support | K005 | 0.876 | 0.683 | 0.784 | 0.865 | 0.011 |
| 013to2 | 6 | zero-support | K006 | 0.946 | 0.991 | 0.378 | 0.624 | 0.322 |
| 012to3 | 1 | zero-support | K001 | 0.874 | 0.999 | 0.545 | 0.508 | 0.366 |
| 012to3 | 2 | zero-support | K002 | 0.998 | 0.981 | 0.995 | 0.922 | 0.076 |
| 012to3 | 3 | zero-support | K003 | 0.123 | 0.016 | 0.942 | 0.637 | -0.514 |
| 012to3 | 4 | zero-support | K004 | 0.191 | 0.219 | 0.398 | 1.000 | -0.809 |
| 012to3 | 5 | zero-support | K005 | 0.839 | 0.652 | 0.822 | 0.936 | -0.097 |
| 012to3 | 6 | zero-support | K006 | 0.780 | 0.962 | 0.362 | 0.525 | 0.255 |

## 주의 / 한계

- 무학습 재추론, 단일 seed. 판정은 다지표(①~⑤) — AUROC 단일 금지. 5-seed 취합은 report_G3a_5seeds.py.
- S_total = val-normal 표준화 등가중 합(test 라벨 튜닝 없음).
- **target-normal이 setting+bearing 이중 held-out** → bearing 개체차 confound 잔존(작업 B/F-1과 정합).

