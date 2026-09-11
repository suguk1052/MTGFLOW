# Q-1 — 결함별 검출률 분해, log 최종안 기준 재산출 (analysis-only)

> **analysis-only — 캐시 재사용(학습·추론 없음).** seeds=[2024,2025,2026,2027,2028], threshold=**val 정상 score 95pct 고정**(test 라벨 미사용).
> 재사용 캐시 3종(모두 `results/Paderborn/`):
> - raw = `b3_raw_window_scores/*.npz` (B3 flow NLL, `te_f`) — 5-seed
> - 제안 log-Fisher = `p2_log_window_scores/*.npz` (`te_sh`/`te_am` → G-5 `fisher_tail`, **log 밴드 최종안**) — 5-seed
> - OC-SVM·band = `b1_ocsvm_band_window_scores/*.npz` (`te_f` = anomaly −decision_function, 이미 이상지향) — **s2024 단일 run(결정론, seed 무관)**
>
> 목적: G-7(`report_G7_detection_breakdown.md`)의 결함별 표는 **linear 밴드(G-5)** 기준 → **log 최종안(P-2 승격)** 기준으로 재집계.
> 코드 `analysis/diagnose_Q1_detection_breakdown_log.py` (G-7의 `score_block`·`fisher_tail`·상수 재사용, G-7 원본 불변).
>
> **sanity 재현 (5-seed mean±std, |Δ|<0.02 전부 OK):** raw **0.696±0.014**(기존 0.696) · log-Fisher **0.877±0.013**(기존 0.877) · OC-SVM·band **0.933±0.000**(기존 B-1 0.933).

---

## 1. 결함별 AUROC (14종, Δ(log−raw) 내림차순)

| fault | family | 사후분류 | raw | 제안 log-Fisher | OC-SVM·band† | Δ(log−raw) |
|---|---|---|---|---|---|---|
| KA22 | KA | shape-sensitive | 0.229±0.013 | 0.923±0.016 | 0.988 | **+0.694** |
| KA15 | KA | shape-sensitive | 0.377±0.027 | 0.913±0.018 | 0.975 | **+0.536** |
| KI14 | KI | shape-sensitive | 0.412±0.049 | 0.915±0.016 | 0.974 | **+0.503** |
| KB27 | KB | shape-sensitive | 0.447±0.051 | 0.925±0.011 | 0.960 | **+0.478** |
| KI17 | KI | shape-sensitive | 0.545±0.041 | 0.929±0.011 | 0.964 | +0.384 |
| KI21 | KI | shape-sensitive | 0.603±0.028 | 0.931±0.007 | 0.967 | +0.328 |
| KI18 | KI | amp-sensitive | 0.809±0.009 | 0.868±0.018 | 0.958 | +0.059 |
| KI16 | KI | amp-sensitive | 0.856±0.013 | 0.906±0.014 | 0.957 | +0.050 |
| KA04 | KA | amp-sensitive | 0.918±0.009 | 0.946±0.017 | 0.958 | +0.028 |
| KA16 | KA | amp-sensitive | 0.950±0.012 | 0.954±0.023 | 0.958 | +0.004 |
| KB23 | KB | amp-sensitive | 0.937±0.014 | 0.897±0.036 | 0.943 | −0.040 |
| KB24 | KB | amp-sensitive | 0.993±0.004 | 0.922±0.014 | 0.908 | −0.071 |
| **KI04** ⭑ | KI | amp-sensitive | 0.836±0.017 | 0.634±0.048 | 0.770 | −0.202 |
| **KA30** ⭑ | KA | amp-sensitive | 0.836±0.013 | 0.608±0.047 | 0.777 | −0.229 |

> † OC-SVM·band = s2024 단일 run(결정론, seed 무관)이라 5-seed 평균 불필요 → 단일값(std=0).
> ⭑ = log 최종안에서도 raw 대비 회복 실패가 지속되는 amp-sensitive 결함(KA30·KI04). §4·주의 참조.

## 2. val95 검출률(recall) (Δ(log−raw) 내림차순)

| fault | 사후분류 | raw recall | log-Fisher recall | OC-SVM·band† | Δ(log−raw) |
|---|---|---|---|---|---|
| KA22 | shape-sensitive | 0.124±0.019 | 0.858±0.030 | 1.000 | **+0.734** |
| KA15 | shape-sensitive | 0.253±0.015 | 0.856±0.029 | 1.000 | **+0.603** |
| KI14 | shape-sensitive | 0.283±0.015 | 0.864±0.024 | 1.000 | **+0.581** |
| KB27 | shape-sensitive | 0.326±0.015 | 0.880±0.027 | 1.000 | +0.553 |
| KI17 | shape-sensitive | 0.387±0.014 | 0.886±0.023 | 1.000 | +0.499 |
| KI21 | shape-sensitive | 0.400±0.015 | 0.886±0.031 | 1.000 | +0.487 |
| KI18 | amp-sensitive | 0.625±0.015 | 0.798±0.042 | 1.000 | +0.173 |
| KI16 | amp-sensitive | 0.696±0.022 | 0.850±0.045 | 1.000 | +0.154 |
| KA04 | amp-sensitive | 0.813±0.004 | 0.889±0.024 | 1.000 | +0.077 |
| KA16 | amp-sensitive | 0.856±0.008 | 0.924±0.024 | 1.000 | +0.068 |
| KB23 | amp-sensitive | 0.849±0.014 | 0.819±0.049 | 0.992 | −0.030 |
| KB24 | amp-sensitive | 0.956±0.001 | 0.925±0.015 | 0.930 | −0.031 |
| KI04 | amp-sensitive | 0.626±0.035 | 0.538±0.056 | 0.602 | −0.089 |
| KA30 | amp-sensitive | 0.622±0.039 | 0.520±0.035 | 0.610 | −0.102 |
| **shape-sensitive 평균** | — | 0.296 | 0.872 | 1.000 | +0.576 |
| **amp-sensitive 평균** | — | 0.755 | 0.783 | 0.892 | +0.028 |

> † OC-SVM·band 단일 run. 진폭군 정상 FPR 고=K001/K003/K006, 저=K002/K004/K005; threshold=val 정상 95pct.

## 3. 전체·fold-type AUROC / 진폭군 정상 FPR (val95)

| 지표 | raw | 제안 log-Fisher | OC-SVM·band† |
|---|---|---|---|
| overall AUROC | 0.696±0.014 | 0.877±0.013 | 0.933 |
| zero-support AUROC | 0.669±0.020 | 0.855±0.016 | 0.915 |
| compositional AUROC | 0.777±0.021 | 0.943±0.011 | 0.985 |
| 정상 FPR 고진폭 | 0.528±0.010 | 0.132±0.021 | 0.022 |
| 정상 FPR 저진폭 | 0.095±0.004 | 0.458±0.035 | 0.103 |

## 4. 필수 기록 — KA30·KI04·KB23의 linear→log 변화

linear = G-5 Fisher-tail(`report_G7_detection_breakdown.md`, `diag_G7_detection_breakdown/combined_5seed.json`의 B열).

| fault | 지표 | raw | linear(G-5) | **log(최종)** | Δ(log−linear) |
|---|---|---|---|---|---|
| KB23 | AUROC | 0.937 | 0.769 | **0.897** | **+0.127** |
| KB23 | recall95 | 0.849 | 0.616 | **0.819** | +0.203 |
| KI04 | AUROC | 0.836 | 0.604 | **0.634** | +0.030 |
| KI04 | recall95 | 0.626 | 0.478 | **0.538** | +0.060 |
| KA30 | AUROC | 0.836 | 0.588 | **0.608** | +0.019 |
| KA30 | recall95 | 0.622 | 0.459 | **0.520** | +0.061 |

- **KB23는 log에서 크게 회복**(AUROC +0.127, recall +0.203) → raw(0.937)에 근접(0.897). linear에서 잃었던 amp 결함 중 KB23은 log 밴드가 실질 복구.
- **KA30·KI04는 log에서도 소폭 개선에 그침**(AUROC +0.019/+0.030). raw(0.836) 대비 여전히 큰 미탐(log 0.608/0.634) 지속 → 확정 limitation("KA30/KI04 amp-sensitive 회복 실패")과 일치.

## 해석

- **log 밴드는 linear 대비 전 결함에서 비열위**(14종 모두 Δ(log−linear)≥+0.019). 특히 shape-sensitive 6종은 linear에서도 이미 회복됐으나 log이 추가 상향(예: KA22 0.836→0.923, KA15 0.789→0.913).
- **그룹 recall(val95): log이 shape 0.781→0.872, amp 0.672→0.783 양쪽 상향**(linear→log). raw는 shape 0.296으로 붕괴, amp 0.755 유지 → log은 raw의 amp 강점을 크게 잃지 않으면서 shape 미탐을 대폭 복구.
- **OC-SVM·band가 전 열에서 최상위**(overall 0.933, shape recall 1.000, amp recall 0.892, 정상 FPR도 우수). 이는 B-5 결론("이득 원천 = log-band 표현이며 제안 조건부 진폭 밀도는 그 위에서 순이득 없음")을 결함 단위에서 재확인. 특히 KA30/KI04도 OC-SVM·band(0.777/0.770)가 제안 log(0.608/0.634)보다 우수.
- **저진폭 정상 FPR은 제안 log-Fisher가 열위**(0.458, raw 0.095·OC-SVM 0.103 대비). 확정 limitation(저진폭 FPR 악화, 임계값으로 해결 불가)과 일치.

## 판정

- Q-1은 analysis-only 재집계 — **모델 재선택 없음**(P-G2 미적용). 결과는 Table 5 정합성 확보용.
- log 최종안은 결함 단위에서도 linear 대비 clean win(전 결함 비열위 + shape/amp 그룹 recall 양쪽 상향)임을 확인. **최종안(N=6+log+Fisher-tail) 유지.**
- KA30·KI04는 log·OC-SVM 어느 쪽으로도 raw 수준 회복 불가 → **잔존 limitation으로 유지·종료**(추가 파기 없음).

## 주의 / 한계

- **[사실]** OC-SVM·band는 s2024 단일 run(결정론). 5-seed 분산 없음. raw·log-Fisher만 5-seed mean±std.
- **[사실]** 표1은 Δ(log−raw) 내림차순, 표2도 표2 자체 Δ(log−raw) 내림차순 정렬(G-7 표1/표2 관례 계승) → 두 표의 결함 행 순서는 서로 다를 수 있음.
- **[사실]** OC-SVM·band recall이 상위 결함에서 1.000인 것은 val95 임계값(OC-SVM 점수 분포)에서 해당 결함 window가 전부 임계 이상이기 때문 — recall 지표 특성, AUROC(0.96~0.99)와 정합.
- **[한계]** 저진폭 정상 FPR(제안 log 0.458) 악화는 임계값으로 해결 불가(확정 limitation). 개체차(K004·K005) 주도.

## 산출물

- 리포트: `reports/report_Q1_detection_breakdown_log.md`
- 코드: `analysis/diagnose_Q1_detection_breakdown_log.py` (G-7 원본 불변, 헬퍼 재사용)
- 캐시(JSON): `results/Paderborn/diag_Q1_detection_breakdown_log/{<fold>_s<seed>,aggregate_s<seed>,combined_5seed}.json`
- 브랜치: `claude/q1-detection-breakdown-log` (g-final 분기)
