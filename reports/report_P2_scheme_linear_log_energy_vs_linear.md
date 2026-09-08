# P-2 Band partition — Fisher-tail(B)/equal-z(A) scheme별 집계 & paired 비교

비교 scheme=['linear', 'log', 'energy'], paired 기준=linear. seed [2024, 2025, 2026, 2027, 2028]. analysis-only(캐시 재사용, 학습·추론 없음).
linear=g3a(N=6) 재사용. paired = 각 fold seed평균 AUROC로 24-fold pairing → Wilcoxon.

## 완료 상태
- linear: seed aggregate 5개 [2024, 2025, 2026, 2027, 2028], per-fold(seed평균) 24/24
- log: seed aggregate 5개 [2024, 2025, 2026, 2027, 2028], per-fold(seed평균) 24/24
- energy: seed aggregate 5개 [2024, 2025, 2026, 2027, 2028], per-fold(seed평균) 24/24

## 전체 AUROC

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.800±0.049 | 0.762±0.058 |
| log | 0.877±0.013 | 0.857±0.014 |
| energy | 0.789±0.048 | 0.742±0.058 |

## zero-support 외삽

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.778±0.072 | 0.736±0.079 |
| log | 0.855±0.016 | 0.832±0.019 |
| energy | 0.765±0.054 | 0.714±0.060 |

## compositional

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.866±0.048 | 0.840±0.059 |
| log | 0.943±0.011 | 0.930±0.007 |
| energy | 0.864±0.094 | 0.826±0.110 |

## amp-sensitive fault

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.779±0.039 | 0.752±0.040 |
| log | 0.842±0.020 | 0.825±0.023 |
| energy | 0.775±0.038 | 0.743±0.043 |

## shape-sensitive fault

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.828±0.071 | 0.775±0.087 |
| log | 0.923±0.012 | 0.899±0.004 |
| energy | 0.808±0.078 | 0.741±0.095 |

## 정상 FPR 고진폭

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.155±0.028 | 0.151±0.029 |
| log | 0.132±0.021 | 0.129±0.018 |
| energy | 0.148±0.036 | 0.145±0.029 |

## 정상 FPR 저진폭

| scheme | Fisher-tail(B) | equal-z(A) |
|---|---|---|
| linear (ref) | 0.418±0.035 | 0.323±0.028 |
| log | 0.458±0.035 | 0.349±0.049 |
| energy | 0.428±0.040 | 0.341±0.041 |

## seed별 전체 AUROC (Fisher B / equal-z A)

| scheme | s2024 | s2025 | s2026 | s2027 | s2028 |
|---|---|---|---|---|---|
| linear | B 0.844/A 0.833 | B 0.865/A 0.833 | B 0.784/A 0.714 | B 0.778/A 0.720 | B 0.730/A 0.711 |
| log | B 0.876/A 0.844 | B 0.860/A 0.854 | B 0.897/A 0.878 | B 0.866/A 0.841 | B 0.883/A 0.866 |
| energy | B 0.856/A 0.831 | B 0.814/A 0.764 | B 0.805/A 0.751 | B 0.724/A 0.658 | B 0.748/A 0.708 |

## 24-fold paired Wilcoxon signed-rank test (scheme vs ref=linear)

> 각 fold의 seed평균 Fisher(B)/equal-z(A) AUROC로 24-fold pairing 후 Wilcoxon(양측). mean_diff>0 이면 scheme이 ref보다 높음. Holm-Bonferroni 보정 p도 병기. 채택 판정은 P-G2.

### Fisher-tail(B)
| scheme | n_fold | mean_diff(scheme−ref) | pos/neg | p(raw) | p(Holm) |
|---|---|---|---|---|---|
| log | 24 | +0.0761 | 21/3 | 0.0002 | 0.0004 |
| energy | 24 | -0.0109 | 10/14 | 0.7469 | 0.7469 |

### equal-z(A)
| scheme | n_fold | mean_diff(scheme−ref) | pos/neg | p(raw) | p(Holm) |
|---|---|---|---|---|---|
| log | 24 | +0.0943 | 22/2 | 0.0001 | 0.0001 |
| energy | 24 | -0.0201 | 12/12 | 0.7898 | 0.7898 |

