# C1 measured 비교 보고서 (Paderborn pooled LONO)

pooled C1 실험(`0123C` = 4개 setting 통합)에서 measured 운행값 메타(`measured`)가 baseline(`no-meta`=B2)과 기존 static meta(`static`=C1) 대비 이상탐지 AUROC를 개선하는지 비교한다. LOSO fold 구분은 없고 LONO(i=1~6)만 변함. 수치는 각 실험 폴더 `paderborn_per_bearing_metrics.json`의 `overall_auroc`를 원본 float 그대로 읽어 계산했고, 표기 시에만 소수점 셋째자리로 반올림했다. Std는 표본 표준편차(ddof=1).

- **no-meta**: baseline (=B2)
- **static**: 기존 static meta (C1)
- **measured**: 신규 실측 meta

## 상세 표 (LONO별)

| LONO split | Test normal | no-meta | static | measured | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.507 | 0.459 | 0.575 | measured |
| LONO-2 | K002 | 0.757 | 0.811 | 0.825 | measured |
| LONO-3 | K003 | 0.497 | 0.641 | 0.641 | static |
| LONO-4 | K004 | 0.794 | 0.914 | 0.892 | static |
| LONO-5 | K005 | 0.829 | 0.834 | 0.702 | static |
| LONO-6 | K006 | 0.445 | 0.462 | 0.542 | measured |
| **Mean** | - | **0.638** | **0.687** | **0.696** | - |
| Std | - | 0.173 | 0.197 | 0.139 | - |
| Median | - | 0.632 | 0.726 | 0.671 | - |

## 요약 표

| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits |
|---|---|---|---|---|
| no-meta (baseline) | 0.638 | 0.173 | 0.632 | - |
| static | 0.687 | 0.197 | 0.726 | 5/6 |
| measured | 0.696 | 0.139 | 0.671 | 5/6 |
| Δ(measured−no-meta) | +0.058 | 0.095 | +0.082 | - |
| Δ(measured−static) | +0.009 | 0.087 | +0.007 | - |

