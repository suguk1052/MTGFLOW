# B3 vs C2 vs C2_measured 5시드 비교 보고서 (Paderborn LOSO)

LOSO(Leave-One-Setting-Out) 실험에서 실측 운행값 메타(`measured`)가 baseline(`no-meta`=B3)과 기존 static meta(`static`=C2) 대비 이상탐지 AUROC를 개선하는지, 4개 LOSO split(012to3/013to2/023to1/123to0) 전체 × LONO(i=1~6)에 대해 5시드(2024~2028)로 확인한다. 이전에는 023to1(저속 unseen) 1개 split만 1시드(2026)로 돌려 `measured가 크게 개선된다`는 case-study 관찰이 있었는데, 이번에는 4개 split 전체를 B2 vs C1과 동일한 방식(5-seed)으로 재검증한다.

- 각 LONO split 셀의 `mean ± std`는 해당 폴더의 `summary_seeds.json`에 기록된 5시드 `auroc_mean`/`auroc_std`를 그대로 읽은 값이다(5개 시드에 대한 population std, ddof=0).
- "요약 표"의 Mean/Std/Median은 6개 LONO split의 split-level mean 값들을 다시 집계한 것이며, Std는 표본표준편차(ddof=1)다(상세 표의 std와 대상이 다름 — report_C1_measured_comparison_5seeds.md와 동일 관례).
- **no-meta**: baseline (=B3, LOSO no-meta)
- **static**: 기존 static meta (C2)
- **measured**: 신규 실측 meta (C2_measured)

## Split별 상세

### LOSO 023to1 (저속(N09_M07_F10) unseen)

| LONO split | Test normal | no-meta (B3) | static (C2) | measured (C2_measured) | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.161 ± 0.097 | 0.235 ± 0.184 | 0.161 ± 0.081 | static |
| LONO-2 | K002 | 0.678 ± 0.114 | 0.606 ± 0.105 | 0.634 ± 0.274 | no-meta |
| LONO-3 | K003 | 0.285 ± 0.057 | 0.277 ± 0.092 | 0.350 ± 0.284 | measured |
| LONO-4 | K004 | 0.732 ± 0.183 | 0.805 ± 0.235 | 0.548 ± 0.393 | static |
| LONO-5 | K005 | 0.832 ± 0.206 | 0.896 ± 0.126 | 0.639 ± 0.294 | static |
| LONO-6 | K006 | 0.183 ± 0.075 | 0.118 ± 0.022 | 0.260 ± 0.258 | measured |

| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits (vs no-meta) |
|---|---|---|---|---|
| no-meta (baseline) | 0.479 | 0.302 | 0.481 | - |
| static | 0.490 | 0.324 | 0.442 | 3/6 |
| measured | 0.432 | 0.204 | 0.449 | 2/6 |
| Δ(measured−no-meta) | -0.047 | - | - | - |
| Δ(measured−static) | -0.058 | - | - | - |

(measured가 static보다 나은 split: 3/6)

### LOSO 012to3 (저 radial force(N15_M07_F04) unseen)

| LONO split | Test normal | no-meta (B3) | static (C2) | measured (C2_measured) | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.496 ± 0.033 | 0.504 ± 0.042 | 0.431 ± 0.254 | static |
| LONO-2 | K002 | 0.916 ± 0.026 | 0.937 ± 0.015 | 0.698 ± 0.183 | static |
| LONO-3 | K003 | 0.628 ± 0.041 | 0.574 ± 0.099 | 0.367 ± 0.233 | no-meta |
| LONO-4 | K004 | 0.995 ± 0.009 | 0.960 ± 0.065 | 0.517 ± 0.383 | no-meta |
| LONO-5 | K005 | 0.905 ± 0.045 | 0.949 ± 0.032 | 0.610 ± 0.443 | static |
| LONO-6 | K006 | 0.581 ± 0.065 | 0.486 ± 0.036 | 0.383 ± 0.289 | no-meta |

| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits (vs no-meta) |
|---|---|---|---|---|
| no-meta (baseline) | 0.753 | 0.210 | 0.767 | - |
| static | 0.735 | 0.236 | 0.755 | 3/6 |
| measured | 0.501 | 0.133 | 0.474 | 0/6 |
| Δ(measured−no-meta) | -0.252 | - | - | - |
| Δ(measured−static) | -0.234 | - | - | - |

(measured가 static보다 나은 split: 0/6)

### LOSO 013to2 (저토크(N15_M01_F10) unseen)

| LONO split | Test normal | no-meta (B3) | static (C2) | measured (C2_measured) | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.587 ± 0.021 | 0.547 ± 0.019 | 0.662 ± 0.143 | measured |
| LONO-2 | K002 | 0.911 ± 0.041 | 0.890 ± 0.015 | 0.439 ± 0.423 | no-meta |
| LONO-3 | K003 | 0.678 ± 0.020 | 0.823 ± 0.161 | 0.607 ± 0.298 | static |
| LONO-4 | K004 | 0.978 ± 0.021 | 0.965 ± 0.032 | 0.620 ± 0.316 | no-meta |
| LONO-5 | K005 | 0.894 ± 0.031 | 0.946 ± 0.054 | 0.312 ± 0.255 | static |
| LONO-6 | K006 | 0.608 ± 0.028 | 0.612 ± 0.053 | 0.669 ± 0.179 | measured |

| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits (vs no-meta) |
|---|---|---|---|---|
| no-meta (baseline) | 0.776 | 0.171 | 0.786 | - |
| static | 0.797 | 0.177 | 0.856 | 3/6 |
| measured | 0.552 | 0.144 | 0.613 | 2/6 |
| Δ(measured−no-meta) | -0.225 | - | - | - |
| Δ(measured−static) | -0.245 | - | - | - |

(measured가 static보다 나은 split: 2/6)

### LOSO 123to0 (기준조건(N15_M07_F10) unseen)

| LONO split | Test normal | no-meta (B3) | static (C2) | measured (C2_measured) | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.548 ± 0.035 | 0.458 ± 0.176 | 0.651 ± 0.205 | measured |
| LONO-2 | K002 | 0.839 ± 0.077 | 0.593 ± 0.282 | 0.917 ± 0.048 | measured |
| LONO-3 | K003 | 0.709 ± 0.159 | 0.626 ± 0.181 | 0.774 ± 0.114 | measured |
| LONO-4 | K004 | 0.976 ± 0.031 | 0.651 ± 0.431 | 0.796 ± 0.164 | no-meta |
| LONO-5 | K005 | 0.962 ± 0.043 | 0.730 ± 0.258 | 0.725 ± 0.377 | no-meta |
| LONO-6 | K006 | 0.626 ± 0.039 | 0.480 ± 0.206 | 0.630 ± 0.197 | measured |

| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits (vs no-meta) |
|---|---|---|---|---|
| no-meta (baseline) | 0.777 | 0.177 | 0.774 | - |
| static | 0.590 | 0.104 | 0.609 | 0/6 |
| measured | 0.749 | 0.106 | 0.749 | 4/6 |
| Δ(measured−no-meta) | -0.028 | - | - | - |
| Δ(measured−static) | +0.159 | - | - | - |

(measured가 static보다 나은 split: 5/6)

## 전체 개요 (4 split 통합)

| LOSO split | 설명 | no-meta mean | static mean | measured mean | static 개선 | measured vs no-meta 개선 | measured vs static 개선 |
|---|---|---|---|---|---|---|---|
| 023to1 | 저속(N09_M07_F10) unseen | 0.479 | 0.490 | 0.432 | 3/6 | 2/6 | 3/6 |
| 012to3 | 저 radial force(N15_M07_F04) unseen | 0.753 | 0.735 | 0.501 | 3/6 | 0/6 | 0/6 |
| 013to2 | 저토크(N15_M01_F10) unseen | 0.776 | 0.797 | 0.552 | 3/6 | 2/6 | 2/6 |
| 123to0 | 기준조건(N15_M07_F10) unseen | 0.777 | 0.590 | 0.749 | 0/6 | 4/6 | 5/6 |

| Model | 전체(24 LONO×split) Mean AUROC | Std AUROC | Improved LONO×split (vs no-meta) |
|---|---|---|---|
| no-meta (baseline) | 0.696 | 0.243 | - |
| static | 0.653 | 0.244 | 9/24 |
| measured | 0.558 | 0.185 | 8/24 |
| Δ(measured−no-meta) | -0.138 | - | - |
| Δ(measured−static) | -0.095 | - | - |

(measured가 static보다 나은 LONO×split: 10/24)

## 핵심 관찰

- **023to1(저속 unseen)**: measured mean 0.432 vs no-meta 0.479, static 0.490 — measured가 no-meta를 이기는 LONO는 2/6, static을 이기는 LONO는 3/6. 5-seed로 재확인한 결과이므로 기존 1-seed case-study 관찰과 직접 비교해 판단할 것.
- 4개 split 전체(24 LONO×split)에서 measured가 no-meta를 이기는 비율은 8/24, static을 이기는 비율은 10/24.
- B2 vs C1 5-seed 재검증에서는 1-seed 때 보였던 개선이 5-seed 평균에서 사라졌다(3/6 split만 개선). 이번 B3 vs C2/C2_measured 결과는 그보다 더 나쁘다 — measured가 4개 split 중 3개(023to1, 012to3, 013to2)에서 no-meta/static보다 뚜렷하게 낮고(mean 0.43~0.55 vs 0.75~0.80), 123to0에서만 개선(measured 0.749 vs static 0.590, no-meta와는 거의 동일)된다.
- **주의(품질 이상 소견)**: measured 조건의 셀별 std가 no-meta/static(대체로 0.02~0.2)보다 훨씬 크고(0.1~0.44), 심한 경우 시드 간 AUROC가 거의 무작위 수준으로 흔들린다. 예: `CA_raw_vib_012to3_LONO4_measured`는 no-meta가 5시드 모두 0.98~1.00으로 안정적인데 measured는 시드별 AUROC가 0.022 / 0.863 / 0.584 / 0.982 / 0.134로 요동친다. 이는 "measured 메타가 성능을 떨어뜨린다"는 결론보다는, LOSO 설정에서 measured 조건 학습이 **시드에 따라 수렴 여부가 갈리는 불안정성**을 겪고 있을 가능성을 시사한다(예: 정규화 통계가 train split에만 의존하는데 held-out setting의 실측 분포가 커서 학습이 불안정해질 수 있음). 4개 split 통합 mean/개선 카운트는 이 불안정성이 그대로 섞여 있는 값이므로, measured의 실제 효과를 판단하려면 실패 시드를 걸러내거나 원인(학습 발산/조기 종료 등)을 먼저 진단할 필요가 있다.
