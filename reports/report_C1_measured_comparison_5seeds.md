# C1 measured 비교 보고서 (5시드 평균, Paderborn pooled LONO)

pooled C1 실험(`0123C` = 4개 setting 통합)에서 measured 운행값 메타(`measured`)가 baseline(`no-meta`=B2)과 기존 static meta(`static`=C1) 대비 이상탐지 AUROC를 개선하는지 비교한다. LOSO fold 구분은 없고 LONO(i=1~6)만 변한다. 이전 리포트(`report_C1_measured_comparison.md`)는 시드 2026 하나만 사용했지만, 이번에는 세 실험 모두 시드를 5개(2024~2028)로 확장해 재실행한 결과를 사용한다.

- 각 LONO split 셀의 `mean ± std`는 해당 폴더의 `summary_seeds.json`에 이미 기록된 5시드 `auroc_mean`/`auroc_std`를 그대로 읽은 값이다. 이 std는 **5개 시드에 대한 모집단 표준편차(ddof=0)**로, 기존 파이프라인이 산출한 값을 그대로 가져왔다.
- "요약 표"의 Mean/Std/Median은 6개 LONO split의 split-level mean 값들을 다시 집계한 것이며, 여기 Std는 **표본 표준편차(ddof=1)**다(이전 리포트와 동일한 집계 방식). 즉 상세 표의 std(시드 변동성)와 요약 표의 std(split 간 변동성)는 서로 다른 대상을 측정한다.
- **no-meta**: baseline (=B2)
- **static**: 기존 static meta (C1)
- **measured**: 신규 실측 meta

## 상세 표 (LONO별, 5시드 mean ± std)

| LONO split | Test normal | no-meta | static | measured | best |
|---|---|---|---|---|---|
| LONO-1 | K001 | 0.484 ± 0.052 | 0.466 ± 0.044 | 0.527 ± 0.060 | measured |
| LONO-2 | K002 | 0.803 ± 0.042 | 0.751 ± 0.014 | 0.766 ± 0.042 | no-meta |
| LONO-3 | K003 | 0.477 ± 0.037 | 0.551 ± 0.058 | 0.525 ± 0.066 | static |
| LONO-4 | K004 | 0.893 ± 0.018 | 0.832 ± 0.046 | 0.810 ± 0.101 | no-meta |
| LONO-5 | K005 | 0.859 ± 0.042 | 0.850 ± 0.062 | 0.777 ± 0.067 | no-meta |
| LONO-6 | K006 | 0.456 ± 0.047 | 0.480 ± 0.048 | 0.539 ± 0.093 | measured |

`best`는 std 겹침을 고려하지 않고 5시드 평균값만 비교해 결정했다.

## 요약 표 (split 간 집계)

| Model | Mean AUROC | Std AUROC | Median AUROC | Improved splits (vs no-meta) |
|---|---|---|---|---|
| no-meta (baseline) | 0.662 | 0.210 | 0.643 | - |
| static | 0.655 | 0.177 | 0.651 | 2/6 |
| measured | 0.657 | 0.140 | 0.653 | 3/6 |
| Δ(measured−no-meta) | -0.005 | 0.072 | +0.003 | - |
| Δ(measured−static) | +0.002 | 0.053 | -0.003 | - |

(measured가 static보다 나은 split도 3/6.)

## 핵심 관찰

이전 1시드 리포트에서는 `measured`가 6개 LONO split 중 5개에서 우세했다. 하지만 5시드 평균으로 다시 보면:

- `measured`가 `no-meta`를 이기는 split은 3/6, `static`을 이기는 split도 3/6으로 줄어든다.
- split별 평균 AUROC의 차이(Δ mean, -0.005 ~ +0.002)가 같은 split 내 시드 간 표준편차(0.04~0.10)보다 작은 경우가 대부분이다.
- 즉 이전 리포트에서 관찰된 `measured`의 개선폭은 상당 부분이 시드 하나에 의한 노이즈였을 가능성이 크고, 5시드 평균 기준으로는 `measured`가 `no-meta`/`static` 대비 뚜렷한 우위를 보이지 않는다(LONO-1, LONO-6에서는 여전히 우세하지만 LONO-2, 4, 5에서는 오히려 열세).
