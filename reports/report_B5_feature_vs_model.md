# B-5. 특징 vs 모델 진단 — 같은 6-log-band 특징 위 밀도/OCC 모델 비교

analysis-only(기존 캐시 재사용, 학습·모델·하이퍼 변경 없음). S_amp=제안 amp 분기(조건부 Gaussian head on 6-band), OC-SVM·band/IF·band=동일 6-band 특징의 generic OCC. Fisher/S_shape=참고.

## ① PU 24 fold별 AUROC (seed 평균)

| fold | 유형 | 진폭군 | OC-SVM·band | IF·band | 제안 Fisher | 제안 S_amp | 제안 S_shape |
|---|---|---|---|---|---|---|---|
| 123to0_L1 | comp | high | 0.999 | 0.984 | 0.965 | 0.891 | 0.978 |
| 123to0_L2 | comp | low | 0.992 | 0.946 | 0.990 | 0.999 | 0.823 |
| 123to0_L3 | comp | high | 1.000 | 0.999 | 0.997 | 0.971 | 0.990 |
| 123to0_L4 | comp | low | 0.927 | 0.666 | 0.800 | 0.521 | 0.686 |
| 123to0_L5 | comp | low | 0.993 | 0.965 | 0.905 | 0.946 | 0.483 |
| 123to0_L6 | comp | high | 0.999 | 0.922 | 0.998 | 0.847 | 0.989 |
| 023to1_L1 | zero | high | 0.972 | 0.981 | 0.850 | 0.926 | 0.627 |
| 023to1_L2 | zero | low | 0.826 | 0.711 | 0.807 | 0.778 | 0.821 |
| 023to1_L3 | zero | high | 0.963 | 0.963 | 0.740 | 0.970 | 0.487 |
| 023to1_L4 | zero | low | 0.136 | 0.442 | 0.503 | 0.189 | 0.727 |
| 023to1_L5 | zero | low | 0.809 | 0.752 | 0.651 | 0.542 | 0.741 |
| 023to1_L6 | zero | high | 0.953 | 0.910 | 0.956 | 0.865 | 0.946 |
| 013to2_L1 | zero | high | 0.999 | 0.979 | 0.918 | 0.887 | 0.814 |
| 013to2_L2 | zero | low | 0.997 | 0.964 | 0.999 | 0.999 | 0.812 |
| 013to2_L3 | zero | high | 1.000 | 0.999 | 0.999 | 0.974 | 0.991 |
| 013to2_L4 | zero | low | 0.936 | 0.730 | 0.848 | 0.685 | 0.593 |
| 013to2_L5 | zero | low | 0.996 | 0.981 | 0.929 | 0.948 | 0.170 |
| 013to2_L6 | zero | high | 0.999 | 0.892 | 0.939 | 0.801 | 0.794 |
| 012to3_L1 | zero | high | 0.996 | 0.972 | 0.942 | 0.859 | 0.941 |
| 012to3_L2 | zero | low | 0.970 | 0.817 | 0.996 | 0.995 | 0.939 |
| 012to3_L3 | zero | high | 0.998 | 0.993 | 0.874 | 0.944 | 0.825 |
| 012to3_L4 | zero | low | 0.930 | 0.715 | 0.509 | 0.525 | 0.418 |
| 012to3_L5 | zero | low | 0.997 | 0.989 | 0.923 | 0.929 | 0.705 |
| 012to3_L6 | zero | high | 0.998 | 0.893 | 1.000 | 0.859 | 0.922 |

## ② 밀도 모델 비교 — 같은 6-band 특징, S_amp(조건부 Gaussian) vs generic OCC

overall(24 fold 평균): S_amp 0.827±0.196 · OC-SVM·band 0.933±0.174 · IF·band 0.882±0.139

### PU paired (24 fold, Wilcoxon+Holm) — 기준 = 제안 S_amp
| 비교 | Δ(대상−S_amp) | p(Holm) | 대상 우세/24 |
|---|---|---|---|
| OC-SVM·band vs S_amp | +0.106 | 0.000 | 19/24 |
| IF·band vs S_amp | +0.055 | 0.005 | 19/24 |

### UODS paired (100 split, Wilcoxon) — 기준 = 제안 S_amp
overall: S_amp 0.878±0.072 · OC-SVM·band 0.882±0.062 · IF·band 0.875±0.059
| 비교 | Δ(대상−S_amp) | p | 대상 우세/100 |
|---|---|---|---|
| OC-SVM·band vs S_amp | +0.004 | 0.525 | 51/100 |
| IF·band vs S_amp | -0.003 | 0.419 | 49/100 |

## ③ 저진폭 정상 FPR · ρ(RMS,score) (PU 24 fold 평균)

| 모델 | 저진폭 정상 FPR | 고진폭 정상 FPR | ρ(RMS,score) |
|---|---|---|---|
| S_amp | 0.150 | 0.118 | -0.080 |
| OC-SVM·band | 0.103 | 0.022 | -0.133 |
| IF·band | 0.119 | 0.024 | -0.285 |
| 제안 Fisher (참고) | 0.458 | 0.132 | -0.392 |

## 해석
- **[확정]** 같은 6-band 특징에서 **OC-SVM·band가 제안 S_amp를 유의하게 상회**(Δ+0.106, Holm p=0.000, 19/24) → 제안의 조건부 Gaussian amp head는 **generic OCC 대비 순이득 없음**. 이득의 원천은 특징(log-band)이지 amp 밀도 모델이 아님.
- **[확정]** raw 입력 대비 6-band 입력이 amp·shape 결함 동시 보존(B-1 표) — 표현이 판별력의 핵심.
- **[가설]** 제안 Fisher가 S_amp 단독·OCC 대비 갖는 이점은 shape branch 결합(dual)에서 오며, amp 밀도 모델링 자체의 기여는 제한적. (S_shape·Fisher 열/③ FPR로 뒷받침 정도 판단.)

> 모델·하이퍼 변경 없음. 전 수치는 동결 캐시(제안 p2_log·UODS proposed, B-1 band)에서 재계산.