# MTGFLOW conditioning 비교표 — seed 2026 단독 (Paderborn)

no-meta / static / measured-concat / FiLM 4종 conditioning의 window-level AUROC를 C1(pooled, in-distribution) 6 LONO와 C2(LOSO 4 split: 012to3/013to2/023to1/123to0) × LONO(1~6)에서 **seed 2026 한 시드**로 비교한다. 기존 5시드 리포트(report_*_5seeds.md)와 축은 같고 값만 단일 시드다.

- 각 셀 값 = 해당 run의 `_s2026/paderborn_per_bearing_metrics.json`의 `overall_auroc` (seed 2026 단일값, ±std 없음).
- "요약 표"의 Mean/Median은 6개 LONO의 값들을 집계한 것이다.
- 이 표는 TODO ⭐"다음 무대"(단일시드 2026에서 C2 개선)의 출발 기준선이다.

## C1 (pooled, in-distribution)

| LONO split | Test normal | no-meta | static | measured-concat | FiLM | best |
|---|---|---|---|---|---|---|
| LONO-1 | K001 | 0.449 | 0.454 | **0.575** | 0.463 | measured-concat |
| LONO-2 | K002 | 0.742 | 0.740 | 0.825 | **0.831** | FiLM |
| LONO-3 | K003 | 0.494 | 0.492 | **0.641** | 0.484 | measured-concat |
| LONO-4 | K004 | **0.896** | 0.781 | 0.892 | 0.770 | no-meta |
| LONO-5 | K005 | 0.809 | **0.873** | 0.808 | 0.863 | static |
| LONO-6 | K006 | 0.442 | **0.533** | 0.512 | 0.415 | static |

| Model | Mean AUROC | Median AUROC |
|---|---|---|
| no-meta | 0.639 | 0.618 |
| static | 0.646 | 0.637 |
| measured-concat | **0.709** | 0.724 |
| FiLM | 0.638 | 0.627 |
| Δ(FiLM−no-meta) | -0.001 | - |
| Δ(FiLM−measured-concat) | -0.071 | - |

(FiLM이 이기는 LONO: no-meta 대비 3/6, static 대비 2/6, measured-concat 대비 2/6)

## C2 (LOSO, cross-setting) — split별 상세

### LOSO 023to1 (저속(N09_M07_F10) unseen)

| LONO split | Test normal | no-meta | static | measured-concat | FiLM | best |
|---|---|---|---|---|---|---|
| LONO-1 | K001 | 0.088 | **0.285** | 0.068 | 0.182 | static |
| LONO-2 | K002 | 0.581 | 0.616 | **0.978** | 0.777 | measured-concat |
| LONO-3 | K003 | 0.361 | 0.394 | **0.881** | 0.834 | measured-concat |
| LONO-4 | K004 | 0.696 | 0.996 | **1.000** | 0.928 | measured-concat |
| LONO-5 | K005 | 0.548 | **1.000** | 0.872 | 0.999 | static |
| LONO-6 | K006 | 0.194 | 0.124 | 0.139 | **0.826** | FiLM |

| Model | Mean AUROC | Median AUROC |
|---|---|---|
| no-meta | 0.411 | 0.455 |
| static | 0.569 | 0.505 |
| measured-concat | 0.656 | 0.876 |
| FiLM | **0.758** | 0.830 |
| Δ(FiLM−no-meta) | +0.347 | - |
| Δ(FiLM−measured-concat) | +0.101 | - |

(FiLM이 이기는 LONO: no-meta 대비 6/6, static 대비 3/6, measured-concat 대비 3/6)

### LOSO 012to3 (저 radial force(N15_M07_F04) unseen)

| LONO split | Test normal | no-meta | static | measured-concat | FiLM | best |
|---|---|---|---|---|---|---|
| LONO-1 | K001 | 0.511 | 0.539 | **0.803** | 0.315 | measured-concat |
| LONO-2 | K002 | **0.945** | 0.934 | 0.896 | 0.938 | no-meta |
| LONO-3 | K003 | 0.688 | 0.582 | **0.703** | 0.666 | measured-concat |
| LONO-4 | K004 | **1.000** | 0.970 | 0.584 | 0.004 | no-meta |
| LONO-5 | K005 | 0.816 | 0.904 | **0.987** | 0.006 | measured-concat |
| LONO-6 | K006 | **0.578** | 0.495 | 0.093 | 0.026 | no-meta |

| Model | Mean AUROC | Median AUROC |
|---|---|---|
| no-meta | **0.756** | 0.752 |
| static | 0.737 | 0.743 |
| measured-concat | 0.678 | 0.753 |
| FiLM | 0.326 | 0.171 |
| Δ(FiLM−no-meta) | -0.430 | - |
| Δ(FiLM−measured-concat) | -0.352 | - |

(FiLM이 이기는 LONO: no-meta 대비 0/6, static 대비 2/6, measured-concat 대비 1/6)

### LOSO 013to2 (저토크(N15_M01_F10) unseen)

| LONO split | Test normal | no-meta | static | measured-concat | FiLM | best |
|---|---|---|---|---|---|---|
| LONO-1 | K001 | 0.593 | 0.571 | **0.663** | 0.263 | measured-concat |
| LONO-2 | K002 | **0.884** | 0.879 | 0.300 | 0.493 | no-meta |
| LONO-3 | K003 | 0.678 | **0.972** | 0.473 | 0.232 | static |
| LONO-4 | K004 | **1.000** | 0.999 | 0.950 | 0.868 | no-meta |
| LONO-5 | K005 | 0.855 | **0.940** | 0.289 | 0.034 | static |
| LONO-6 | K006 | 0.640 | **0.655** | 0.454 | 0.486 | static |

| Model | Mean AUROC | Median AUROC |
|---|---|---|
| no-meta | 0.775 | 0.766 |
| static | **0.836** | 0.909 |
| measured-concat | 0.522 | 0.464 |
| FiLM | 0.396 | 0.374 |
| Δ(FiLM−no-meta) | -0.379 | - |
| Δ(FiLM−measured-concat) | -0.125 | - |

(FiLM이 이기는 LONO: no-meta 대비 0/6, static 대비 0/6, measured-concat 대비 2/6)

### LOSO 123to0 (기준조건(N15_M07_F10) unseen)

| LONO split | Test normal | no-meta | static | measured-concat | FiLM | best |
|---|---|---|---|---|---|---|
| LONO-1 | K001 | 0.504 | 0.685 | **0.763** | 0.522 | measured-concat |
| LONO-2 | K002 | 0.707 | 0.957 | **0.971** | 0.886 | measured-concat |
| LONO-3 | K003 | **0.990** | 0.584 | 0.906 | 0.696 | no-meta |
| LONO-4 | K004 | 0.995 | **1.000** | 0.844 | 0.146 | static |
| LONO-5 | K005 | 0.988 | **0.998** | 0.984 | 0.873 | static |
| LONO-6 | K006 | **0.702** | 0.167 | 0.302 | 0.626 | no-meta |

| Model | Mean AUROC | Median AUROC |
|---|---|---|
| no-meta | **0.814** | 0.848 |
| static | 0.732 | 0.821 |
| measured-concat | 0.795 | 0.875 |
| FiLM | 0.625 | 0.661 |
| Δ(FiLM−no-meta) | -0.190 | - |
| Δ(FiLM−measured-concat) | -0.170 | - |

(FiLM이 이기는 LONO: no-meta 대비 2/6, static 대비 2/6, measured-concat 대비 1/6)

## C2(LOSO) 전체 개요 (4 split × 6 LONO = 24 셀)

| LOSO split | 설명 | no-meta | static | measured-concat | FiLM | FiLM win vs no-meta |
|---|---|---|---|---|---|---|
| 023to1 | 저속(N09_M07_F10) unseen | 0.411 | 0.569 | 0.656 | **0.758** | 6/6 |
| 012to3 | 저 radial force(N15_M07_F04) unseen | **0.756** | 0.737 | 0.678 | 0.326 | 0/6 |
| 013to2 | 저토크(N15_M01_F10) unseen | 0.775 | **0.836** | 0.522 | 0.396 | 0/6 |
| 123to0 | 기준조건(N15_M07_F10) unseen | **0.814** | 0.732 | 0.795 | 0.625 | 2/6 |

| Model | 전체(24 셀) Mean AUROC |
|---|---|
| no-meta | 0.689 |
| static | **0.719** |
| measured-concat | 0.663 |
| FiLM | 0.526 |
| Δ(FiLM−no-meta) | -0.163 |
| Δ(FiLM−measured-concat) | -0.136 |

(FiLM이 이기는 LONO×split: no-meta 대비 8/24, static 대비 7/24, measured-concat 대비 7/24)

