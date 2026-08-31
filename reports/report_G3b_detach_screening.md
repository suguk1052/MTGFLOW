# G-3b detach ablation — seed 2026 paired 비교 (S_shape 중심)

- paired fold 수: **24** (g3a-only [], g3b-only [])
- raw baseline(동일 checkpoint): g3a 0.689 / g3b 0.689 (동일해야 정상 — 파이프라인 검증)
- 자동 휴리스틱: GO(overall Δ≤−0.02 & 하락 fold≥60%) — **단, 아래 최종 판정이 이를 기각(fold 일관성 미충족).**

## 🎯 최종 판정 (사용자 승인, 2026-08-26): **NO-GO — G-3b 1-seed 종료**
overall S_shape는 하락 방향(−0.038)이나 **auxiliary-gradient 가설을 명확히 입증하지 못함.** 근거:
1. **크기가 1-seed 노이즈대 이내**: −0.038은 G-3a shape deepdive가 관측한 seed 분산(std ~0.05–0.07)보다 작다.
2. **가설 핵심 예측 미성립**: 가설이 맞다면 gain의 원천이던 **고진폭-test fold의 S_shape가 detach로 붕괴**해야
   하는데, 고진폭 fold는 **6/12만 하락(정확히 반반)** 이고 일부는 오히려 급등(023to1 L3 0.025→0.964,
   013to2 L6 0.060→0.985). 즉 "gradient 제거 → gain 소멸"의 깨끗한 서명이 없다.
3. **per-fold Δ가 극단적 양방향**(+0.94 ↔ −0.89) = 구조적 하락이 아니라 고분산 재배치. 단일 seed로는
   gradient 원인과 seed 노이즈를 분리 불가.
→ **5-seed 확장 안 함.** auxiliary-task 가설은 1-seed 스크리닝에서 입증 실패로 기록하고 G-3b 종료.

## Overall AUROC (g3a → g3b)
- **S_shape**: 0.715 → 0.676 (Δ -0.038)  ← 최우선 지표
- S_total: 0.714 → 0.713 (Δ -0.002)
- S_amp:   0.635 → 0.651 (Δ +0.016)

## S_shape by fold type
- zero-support: 0.687 → 0.666 (Δ -0.021)
- compositional: 0.798 → 0.706 (Δ -0.092)

## S_shape by fault group
- amp_sensitive: 0.710 → 0.699 (Δ -0.011)
- shape_sensitive: 0.721 → 0.646 (Δ -0.075)

## 정상 FPR (S_shape block)
- 고진폭: 0.121 → 0.060 (Δ -0.061)
- 저진폭: 0.296 → 0.354 (Δ +0.058)

## per-fold S_shape (g3a → g3b)
| split | LONO | fold_type | amp_grp | g3a | g3b | Δ |
|---|---|---|---|---|---|---|
| 012to3 | 1 | zero-support | high | 0.998 | 0.995 | -0.003 |
| 012to3 | 2 | zero-support | low | 0.984 | 0.957 | -0.027 |
| 012to3 | 3 | zero-support | high | 0.690 | 0.674 | -0.016 |
| 012to3 | 4 | zero-support | low | 0.794 | 0.162 | -0.632 |
| 012to3 | 5 | zero-support | low | 0.200 | 0.158 | -0.043 |
| 012to3 | 6 | zero-support | high | 0.998 | 0.999 | +0.002 |
| 013to2 | 1 | zero-support | high | 0.224 | 0.990 | +0.766 |
| 013to2 | 2 | zero-support | low | 0.892 | 0.884 | -0.008 |
| 013to2 | 3 | zero-support | high | 0.976 | 0.977 | +0.000 |
| 013to2 | 4 | zero-support | low | 0.945 | 0.052 | -0.894 |
| 013to2 | 5 | zero-support | low | 0.396 | 0.103 | -0.294 |
| 013to2 | 6 | zero-support | high | 0.060 | 0.985 | +0.924 |
| 023to1 | 1 | zero-support | high | 0.999 | 0.511 | -0.487 |
| 023to1 | 2 | zero-support | low | 0.979 | 0.995 | +0.015 |
| 023to1 | 3 | zero-support | high | 0.025 | 0.964 | +0.939 |
| 023to1 | 4 | zero-support | low | 0.915 | 0.587 | -0.328 |
| 023to1 | 5 | zero-support | low | 0.298 | 0.859 | +0.561 |
| 023to1 | 6 | zero-support | high | 0.986 | 0.136 | -0.849 |
| 123to0 | 1 | compositional | high | 0.984 | 0.978 | -0.006 |
| 123to0 | 2 | compositional | low | 0.967 | 0.971 | +0.004 |
| 123to0 | 3 | compositional | high | 0.969 | 0.991 | +0.022 |
| 123to0 | 4 | compositional | low | 0.336 | 0.062 | -0.274 |
| 123to0 | 5 | compositional | low | 0.542 | 0.275 | -0.267 |
| 123to0 | 6 | compositional | high | 0.993 | 0.962 | -0.030 |

> 판정 규칙(TODO): detach에서 S_shape가 **명확히 하락**하면 auxiliary-task 가설 입증 → 5-seed 확장. 차이가 작으면 가설 미지지 → 1-seed에서 G-3b 종료. 최종 판단은 사용자 승인.
