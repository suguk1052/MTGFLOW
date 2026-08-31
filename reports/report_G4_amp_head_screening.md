# G-4 Amp Head ablation — seed 2026 paired 비교 (S_total 중심)

- 후보: B(n=24), C(n=24) · baseline = G-3a (diag_G3a_amp_bands, seed 2026)
- **B 판정 휴리스틱: GO(5-seed 확장 후보)** (ΔS_total=+0.028, Δamp=+0.015, Δshape=+0.046, fault 비붕괴=True)
- **C 판정 휴리스틱: NO-GO(1-seed 종료 후보)** (ΔS_total=+0.010, Δamp=-0.026, Δshape=+0.058, fault 비붕괴=False)

> 판정 규칙(TODO §G-4): S_total이 seed 분산(~0.02) 초과 상승 + 두 fault군 비붕괴면 해당 후보만 5-seed 확장. 최종 판단은 사용자 승인. cherry-pick 금지.

## 지표 비교 (G-3a → 후보, Δ)
| 지표 | G-3a | B | C |
|---|---|---|---|
| **S_total** (primary) | 0.714 | 0.743 (Δ+0.028) | 0.724 (Δ+0.010) |
| amp-sensitive fault (total) | 0.739 | 0.754 (Δ+0.015) | 0.714 (Δ-0.026) |
| shape-sensitive fault (total) | 0.680 | 0.726 (Δ+0.046) | 0.738 (Δ+0.058) |
| S_shape | 0.715 | 0.706 (Δ-0.009) | 0.742 (Δ+0.027) |
| S_amp | 0.635 | 0.650 (Δ+0.015) | 0.633 (Δ-0.002) |
| zero-support (total) | 0.667 | 0.741 (Δ+0.074) | 0.683 (Δ+0.016) |
| compositional (total) | 0.855 | 0.748 (Δ-0.108) | 0.847 (Δ-0.008) |
| 정상 FPR 고진폭 (total) | 0.134 | 0.154 (Δ+0.020) | 0.123 (Δ-0.011) |
| 정상 FPR 저진폭 (total) | 0.311 | 0.380 (Δ+0.069) | 0.356 (Δ+0.045) |
| ρ(RMS, S_total) | -0.230 | -0.377 (Δ-0.146) | -0.420 (Δ-0.189) |
| raw AUROC (파이프라인 검증) | 0.689 | 0.689 (Δ+0.000) | 0.689 (Δ+0.000) |

> raw AUROC 행은 세 열이 동일해야 정상(같은 B3 checkpoint 재추론). G-3a 열은 각 후보와의 공통 fold로 재집계하므로 후보간 fold 수가 다르면 미세 차이 가능.

## per-fold S_total (G-3a → B)
| split | LONO | fold_type | amp_grp | G-3a | B | Δ |
|---|---|---|---|---|---|---|
| 012to3 | 1 | zero-support | high | 0.852 | 0.753 | -0.099 |
| 012to3 | 2 | zero-support | low | 0.998 | 0.982 | -0.016 |
| 012to3 | 3 | zero-support | high | 0.999 | 0.995 | -0.005 |
| 012to3 | 4 | zero-support | low | 0.427 | 0.340 | -0.086 |
| 012to3 | 5 | zero-support | low | 0.970 | 0.880 | -0.091 |
| 012to3 | 6 | zero-support | high | 0.941 | 0.880 | -0.060 |
| 013to2 | 1 | zero-support | high | 0.560 | 0.688 | +0.128 |
| 013to2 | 2 | zero-support | low | 0.903 | 0.966 | +0.063 |
| 013to2 | 3 | zero-support | high | 0.998 | 0.926 | -0.072 |
| 013to2 | 4 | zero-support | low | 0.867 | 0.258 | -0.609 |
| 013to2 | 5 | zero-support | low | 0.775 | 0.646 | -0.129 |
| 013to2 | 6 | zero-support | high | 0.292 | 0.738 | +0.446 |
| 023to1 | 1 | zero-support | high | 0.999 | 0.965 | -0.034 |
| 023to1 | 2 | zero-support | low | 0.176 | 0.720 | +0.544 |
| 023to1 | 3 | zero-support | high | 0.078 | 0.988 | +0.910 |
| 023to1 | 4 | zero-support | low | 0.136 | 0.115 | -0.021 |
| 023to1 | 5 | zero-support | low | 0.071 | 0.516 | +0.445 |
| 023to1 | 6 | zero-support | high | 0.969 | 0.978 | +0.008 |
| 123to0 | 1 | compositional | high | 0.822 | 0.438 | -0.384 |
| 123to0 | 2 | compositional | low | 0.999 | 0.995 | -0.005 |
| 123to0 | 3 | compositional | high | 0.976 | 1.000 | +0.024 |
| 123to0 | 4 | compositional | low | 0.509 | 0.531 | +0.022 |
| 123to0 | 5 | compositional | low | 0.829 | 0.523 | -0.306 |
| 123to0 | 6 | compositional | high | 0.996 | 0.999 | +0.003 |

## per-fold S_total (G-3a → C)
| split | LONO | fold_type | amp_grp | G-3a | C | Δ |
|---|---|---|---|---|---|---|
| 012to3 | 1 | zero-support | high | 0.852 | 0.890 | +0.037 |
| 012to3 | 2 | zero-support | low | 0.998 | 0.982 | -0.016 |
| 012to3 | 3 | zero-support | high | 0.999 | 1.000 | +0.001 |
| 012to3 | 4 | zero-support | low | 0.427 | 0.668 | +0.241 |
| 012to3 | 5 | zero-support | low | 0.970 | 0.759 | -0.211 |
| 012to3 | 6 | zero-support | high | 0.941 | 0.970 | +0.030 |
| 013to2 | 1 | zero-support | high | 0.560 | 0.887 | +0.327 |
| 013to2 | 2 | zero-support | low | 0.903 | 1.000 | +0.097 |
| 013to2 | 3 | zero-support | high | 0.998 | 1.000 | +0.002 |
| 013to2 | 4 | zero-support | low | 0.867 | 0.603 | -0.263 |
| 013to2 | 5 | zero-support | low | 0.775 | 0.569 | -0.207 |
| 013to2 | 6 | zero-support | high | 0.292 | 0.974 | +0.682 |
| 023to1 | 1 | zero-support | high | 0.999 | 0.881 | -0.118 |
| 023to1 | 2 | zero-support | low | 0.176 | 0.113 | -0.062 |
| 023to1 | 3 | zero-support | high | 0.078 | 0.055 | -0.023 |
| 023to1 | 4 | zero-support | low | 0.136 | 0.013 | -0.123 |
| 023to1 | 5 | zero-support | low | 0.071 | 0.004 | -0.066 |
| 023to1 | 6 | zero-support | high | 0.969 | 0.928 | -0.041 |
| 123to0 | 1 | compositional | high | 0.822 | 0.827 | +0.005 |
| 123to0 | 2 | compositional | low | 0.999 | 1.000 | +0.000 |
| 123to0 | 3 | compositional | high | 0.976 | 1.000 | +0.024 |
| 123to0 | 4 | compositional | low | 0.509 | 0.429 | -0.080 |
| 123to0 | 5 | compositional | low | 0.829 | 0.968 | +0.139 |
| 123to0 | 6 | compositional | high | 0.996 | 0.859 | -0.137 |

---

## 해석·판정 (가드레일 적용)

자동 휴리스틱은 S_total 상승 임계를 **0.02**로 잡아 B를 "GO"로 표기하나, TODO §G-4 가드레일은
**1-seed Δ가 seed 분산(G-3a deepdive std ~0.05–0.07) 이내면 "미개선"**으로 판정한다. 이 기준을 적용하면:

- **C (LayerNorm head) — NO-GO(명확).** S_total Δ+0.010은 seed 분산 미만이고, **amp-sensitive가 −0.026
  붕괴**(shape-sensitive만 +0.058 얻고 amp를 잃음). G-2a scalar gate가 못 깬 **amp↔shape trade-off가 재발**.
  S_shape는 +0.027 개선하나 primary(S_total)·fault 균형 기준 탈락. 정상 FPR·ρ는 소폭 개선이나 판정 뒤집기엔 부족.

- **B (wider head, d=64) — 미개선(가드레일 미달).** S_total Δ+0.028은 두 fault군 모두 보존(+0.015/+0.046)해
  C보다 낫지만 **seed 분산 밴드(0.05–0.07) 이내**다. 결정적으로 per-fold Δ가 **극단 양방향
  (+0.910 023to1·L3 / +0.544 L2 ↔ −0.609 013to2·L4 / −0.384 123to0·L1)** = 구조적 이득이 아니라
  **고분산 재배치**(G-3b detach에서 관측된 것과 동일 패턴). 동반 악화도 뚜렷: **compositional −0.108,
  정상 FPR 저진폭 +0.069(악화)**. S_total 순증은 주로 zero-support fold 구제에서 나오고 compositional을 깎아 상쇄.

**결론.** 두 후보 모두 **1-seed에서 가드레일을 넘는 명확한 개선 없음** → **G-4 종료, 5-seed 확장 안 함.**
head architecture 개입(wider/LayerNorm)으로는 G-3a 대비 순개선이 확인되지 않았다. **G-3a의 조건부 GO 지위는
불변**이며, G 트랙의 현행 최고 구조로 유지된다. (raw AUROC 0.689가 G-3a/B/C 3열 동일 → 재추론 파이프라인 정합 검증됨.)

> 판정 근거는 다지표(S_total primary + amp/shape 균형 + per-fold 분산 구조)이며 단일 AUROC cherry-pick 아님.
> B의 약한 순증(+0.028)을 신호로 보려면 5-seed로 분산을 좁혀야 하나, 동반 악화(compositional·FPR)와
> 고분산 재배치 성격을 감안하면 우선순위 낮음(사용자 판단).
