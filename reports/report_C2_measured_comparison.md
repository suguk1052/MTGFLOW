# measured meta 비교 보고서 (Paderborn LONO / LOSO)

실측 메타데이터(`measured meta`) 주입이 baseline(`no-meta`)과 기존 static meta(`static`) 대비 이상탐지 AUROC를 개선하는지 비교한다. 수치는 각 실험 폴더의 `paderborn_per_bearing_metrics.json`의 `overall_auroc`를 원본 float 그대로 읽어 계산했고, 표기 시에만 소수점 셋째자리로 반올림했다. Std는 표본 표준편차(ddof=1).

- **no-meta**: baseline (=B3)
- **static**: 기존 static meta (C2)
- **measured**: 신규 실측 meta

## 상세 표 (LOSO별)

### LOSO-0 / 123→0  (target normal: N15_M07_F10)

| LONO split | Test normal | no-meta | static | measured | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.504 | 0.685 | 0.830 | measured |
| LONO-2 | K002 | 0.707 | 0.957 | 0.726 | static |
| LONO-3 | K003 | 0.990 | 0.584 | 0.906 | no-meta |
| LONO-4 | K004 | 0.995 | 1.000 | 0.844 | static |
| LONO-5 | K005 | 0.988 | 0.998 | 0.276 | static |
| LONO-6 | K006 | 0.702 | 0.167 | 0.231 | no-meta |
| **Mean** | - | **0.814** | **0.732** | **0.636** | - |
| Std | - | 0.207 | 0.328 | 0.302 | - |
| Median | - | 0.848 | 0.821 | 0.778 | - |

### LOSO-1 / 023→1  (target normal: N09_M07_F10)

| LONO split | Test normal | no-meta | static | measured | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.108 | 0.168 | 0.892 | measured |
| LONO-2 | K002 | 0.994 | 0.574 | 0.457 | no-meta |
| LONO-3 | K003 | 0.061 | 0.359 | 0.881 | measured |
| LONO-4 | K004 | 0.771 | 0.586 | 1.000 | measured |
| LONO-5 | K005 | 0.628 | 0.991 | 1.000 | measured |
| LONO-6 | K006 | 0.086 | 0.203 | 0.832 | measured |
| **Mean** | - | **0.441** | **0.480** | **0.844** | - |
| Std | - | 0.408 | 0.306 | 0.201 | - |
| Median | - | 0.368 | 0.467 | 0.887 | - |

### LOSO-2 / 013→2  (target normal: N15_M01_F10)

| LONO split | Test normal | no-meta | static | measured | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.560 | 0.440 | 0.106 | no-meta |
| LONO-2 | K002 | 0.957 | 0.850 | 0.548 | no-meta |
| LONO-3 | K003 | 0.580 | 0.619 | 0.473 | static |
| LONO-4 | K004 | 1.000 | 1.000 | 0.950 | static |
| LONO-5 | K005 | 0.944 | 0.994 | 0.204 | static |
| LONO-6 | K006 | 0.637 | 0.608 | 0.006 | no-meta |
| **Mean** | - | **0.780** | **0.752** | **0.381** | - |
| Std | - | 0.208 | 0.230 | 0.349 | - |
| Median | - | 0.791 | 0.734 | 0.339 | - |

### LOSO-3 / 012→3  (target normal: N15_M07_F04)

| LONO split | Test normal | no-meta | static | measured | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.562 | 0.546 | 0.334 | no-meta |
| LONO-2 | K002 | 0.829 | 0.948 | 0.662 | static |
| LONO-3 | K003 | 0.587 | 0.554 | 0.703 | measured |
| LONO-4 | K004 | 0.998 | 1.000 | 0.584 | static |
| LONO-5 | K005 | 0.904 | 0.993 | 0.291 | static |
| LONO-6 | K006 | 0.531 | 0.531 | 0.845 | measured |
| **Mean** | - | **0.735** | **0.762** | **0.570** | - |
| Std | - | 0.200 | 0.240 | 0.217 | - |
| Median | - | 0.708 | 0.751 | 0.623 | - |

## 요약 표 (setting split별 Mean)

| Setting split | Target setting | no-meta Mean | static Mean | measured Mean | Δ(measured−no-meta) | Δ(measured−static) | best method |
|---|---|---|---|---|---|---|---|
| 123→0 | N15_M07_F10 | 0.814 | 0.732 | 0.636 | -0.179 | -0.096 | no-meta |
| 023→1 | N09_M07_F10 | 0.441 | 0.480 | 0.844 | +0.402 | +0.364 | measured |
| 013→2 | N15_M01_F10 | 0.780 | 0.752 | 0.381 | -0.399 | -0.371 | no-meta |
| 012→3 | N15_M07_F04 | 0.735 | 0.762 | 0.570 | -0.165 | -0.192 | static |
| **Overall** | - | **0.693** | **0.681** | **0.608** | **-0.085** | **-0.074** | **no-meta** |

