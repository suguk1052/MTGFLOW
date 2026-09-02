# 작업 G-7 — 검출/오탐 단위 진단 분해 (raw vs A=equal-z vs B=Fisher-tail)

seed [2024, 2025, 2026, 2027, 2028] (5-seed) mean±std. **analysis-only** — G-3a per-window 캐시(g3a_window_scores/*.npz) + raw(B3) 재추론 캐시(b3_raw_window_scores/*.npz, forward-only, 학습 없음)만 사용. 각 seed 24 fold(4 split × 6 LONO). 비교 3종: **raw**=B3 flow NLL / **A**=equal-z S_total(G-3a) / **B**=Fisher-tail(G-5 최종). threshold = **val 정상 score percentile 고정**(test 라벨 튜닝 없음).

> **sanity**: 재집계 전체 AUROC raw 0.696 / A 0.762 / B 0.800 — 기존 리포트(0.696 / 0.762 / 0.800) 재현. raw·A per-fold AUROC vs diag_G3a 최대 |Δ| = 0e+00 / 0e+00 (<1e-6, 캐시·재추론 정합).

## 핵심 발견

1. **AUROC 상승(raw 0.696 → B 0.800)의 정체 = shape-sensitive 결함의 미탐 급감.** 고정 임계값(val95) recall이 shape-sensitive군 평균 **0.30 → 0.78**(KA22 0.12→0.79, KA15 0.25→0.76 등)로 뛴다. raw에서 정상보다 낮게 깔려 '역전'돼 있던 결함들이 B에서 정상 위로 올라온 것.
2. **대가는 amp-sensitive 결함 recall 소폭 하락(0.76 → 0.67)** + **저진폭 정상 FPR 악화**(val95 raw 0.095 → B 0.418). 고진폭 정상 FPR은 반대로 개선(raw 0.528 → B 0.155).
3. **저진폭 FPR 악화는 임계값 문제가 아니라 순위 문제(중증).** percentile을 95→99로 올려도 B의 저진폭 FPR은 0.418 → 0.360 로만 내려가 raw 수준(0.065)에 한참 못 미치고, 그 사이 recall은 0.719 → 0.654 손실.
4. **저진폭 악화의 주범은 K004·K005 두 개체.** val95 FPR raw→B: K004 0.008→0.539, K005 0.106→0.513 인 반면 K002는 0.171→0.204 로 거의 불변. val↔test 진폭 분포 이동 가설의 직접 증거(가설).
5. **KA30·KI04는 회복 실패 확정.** per-fault AUROC raw 0.836 → B 0.588(KA30) / 0.604(KI04), recall도 raw 0.62 → B 0.46~0.48로 하락. amp-sensitive라 shape 정규화가 오히려 신호를 지운다.

## 방법

- **score(클수록 이상)**: raw=`-log p_x`(B3 flow NLL). A=val정상 z-score 후 S_shape+S_amp 등가중 합. B=각 branch를 val정상 상단 tail 확률 p로 변환 후 Fisher χ² 결합 `-2[ln p_shape+ln p_amp]`. 셋 다 test 라벨 무사용.
- **threshold**: fold별 **val 정상 score의 percentile**만으로 결정(90/95/97.5/99). test 라벨은 어떤 임계값에도 미사용.
- **recall** = fault window 중 score ≥ threshold 비율(=1−미탐률). **정상 FPR** = 정상 pool(train+val+test-normal) per-bearing FPR을 고/저진폭군 평균. **per-fault/per-bearing** 은 fold 평균 후 5-seed 평균±std.
- **fold 분리**: zero-support(023→1·013→2·012→3, 순수 외삽) vs compositional(123→0). 24 fold 평균으로 뭉뚱그리지 않음.
- 진폭군: 고진폭 K001·K003·K006 / 저진폭 K002·K004·K005. 결함 사후분류 amp-sensitive(8) / shape-sensitive(6).

## 1) per-fault AUROC (raw vs A vs B, Δ=B−raw 내림차순)

| fault | family | 사후분류 | raw | A: equal-z | B: Fisher-tail | Δ(B−raw) |
|---|---|---|---|---|---|---|
| **KA22** ⭑ | KA | shape-sensitive | 0.229±0.013 | 0.796±0.077 | 0.836±0.074 | +0.607 |
| **KA15** ⭑ | KA | shape-sensitive | 0.377±0.027 | 0.720±0.084 | 0.789±0.076 | +0.412 |
| **KI14** ⭑ | KI | shape-sensitive | 0.412±0.049 | 0.751±0.092 | 0.816±0.076 | +0.404 |
| **KB27** ⭑ | KB | shape-sensitive | 0.447±0.051 | 0.774±0.094 | 0.831±0.073 | +0.384 |
| KI17 | KI | shape-sensitive | 0.545±0.041 | 0.851±0.090 | 0.881±0.069 | +0.335 |
| KI21 | KI | shape-sensitive | 0.603±0.028 | 0.759±0.091 | 0.817±0.069 | +0.214 |
| KA04 | KA | amp-sensitive | 0.918±0.009 | 0.903±0.064 | 0.922±0.052 | +0.004 |
| KI16 | KI | amp-sensitive | 0.856±0.013 | 0.793±0.072 | 0.825±0.058 | -0.031 |
| KI18 | KI | amp-sensitive | 0.809±0.009 | 0.732±0.062 | 0.766±0.048 | -0.043 |
| KA16 | KA | amp-sensitive | 0.950±0.012 | 0.894±0.058 | 0.906±0.051 | -0.043 |
| KB24 | KB | amp-sensitive | 0.993±0.004 | 0.855±0.015 | 0.855±0.025 | -0.138 |
| KB23 | KB | amp-sensitive | 0.937±0.014 | 0.772±0.034 | 0.769±0.038 | -0.167 |
| **KI04** ⭑ | KI | amp-sensitive | 0.836±0.017 | 0.541±0.039 | 0.604±0.040 | -0.232 |
| **KA30** ⭑ | KA | amp-sensitive | 0.836±0.013 | 0.531±0.043 | 0.588±0.040 | -0.248 |

> ⭑ = raw에서 0.5 미만이었던 역전 결함군(KA30·KI04는 raw 0.836이나 amp-sensitive라 A/B에서 회복 실패 — 별 범주). shape-sensitive 6종이 Δ 상위를 독점, amp-sensitive는 전부 Δ≤0.

## 2) 고정 임계값(val95) 검출률 recall — raw vs B

| fault | 사후분류 | raw recall | B recall | Δ(B−raw) |
|---|---|---|---|---|
| KA22 | shape-sensitive | 0.124±0.019 | 0.793±0.055 | +0.669 |
| KA15 | shape-sensitive | 0.253±0.015 | 0.760±0.061 | +0.506 |
| KI14 | shape-sensitive | 0.283±0.015 | 0.768±0.072 | +0.485 |
| KB27 | shape-sensitive | 0.326±0.015 | 0.783±0.065 | +0.457 |
| KI17 | shape-sensitive | 0.387±0.014 | 0.825±0.052 | +0.438 |
| KI21 | shape-sensitive | 0.400±0.015 | 0.758±0.075 | +0.358 |
| KA04 | amp-sensitive | 0.813±0.004 | 0.871±0.045 | +0.059 |
| KI16 | amp-sensitive | 0.696±0.022 | 0.713±0.074 | +0.017 |
| KI18 | amp-sensitive | 0.625±0.015 | 0.636±0.064 | +0.011 |
| KA16 | amp-sensitive | 0.856±0.008 | 0.836±0.038 | -0.020 |
| KI04 | amp-sensitive | 0.626±0.035 | 0.478±0.054 | -0.148 |
| KA30 | amp-sensitive | 0.622±0.039 | 0.459±0.047 | -0.162 |
| KB24 | amp-sensitive | 0.956±0.001 | 0.770±0.031 | -0.185 |
| KB23 | amp-sensitive | 0.849±0.014 | 0.616±0.029 | -0.233 |
| **shape-sensitive 평균** | — | 0.296 | 0.781 | +0.485 |
| **amp-sensitive 평균** | — | 0.755 | 0.672 | -0.083 |

**진폭군별 정상 FPR (val95, raw / A / B):**

| 진폭군 | raw | A: equal-z | B: Fisher-tail |
|---|---|---|---|
| 고진폭(K001·K003·K006) | 0.528±0.010 | 0.151±0.029 | 0.155±0.028 |
| 저진폭(K002·K004·K005) | 0.095±0.004 | 0.323±0.028 | 0.418±0.035 |

> AUROC 상승분은 shape-sensitive 결함 recall(+~0.49)에서 온다. 대가는 amp-sensitive recall 소폭 하락과 저진폭 정상 FPR 악화.

## 3) threshold sweep — FPR–recall 트레이드오프 (진폭군별, raw vs B)

**고진폭 정상 패널** (그림: `figs_g7/fpr_recall_high.png`)

| val pct | raw 정상FPR | raw recall | B 정상FPR | B recall |
|---|---|---|---|---|
| 90 | 0.558 | 0.576 | 0.204 | 0.751 |
| 95 | 0.528 | 0.558 | 0.155 | 0.719 |
| 97.5 | 0.507 | 0.542 | 0.120 | 0.689 |
| 99 | 0.488 | 0.524 | 0.088 | 0.654 |

**저진폭 정상 패널** (그림: `figs_g7/fpr_recall_low.png`)

| val pct | raw 정상FPR | raw recall | B 정상FPR | B recall |
|---|---|---|---|---|
| 90 | 0.114 | 0.576 | 0.455 | 0.751 |
| 95 | 0.095 | 0.558 | 0.418 | 0.719 |
| 97.5 | 0.081 | 0.542 | 0.389 | 0.689 |
| 99 | 0.065 | 0.524 | 0.360 | 0.654 |

> 저진폭 패널: pct를 95→99로 올려도 B FPR 0.418→0.360 (raw 0.065에 한참 못 미침), recall 손실 0.065. → 임계값 보정으로 저진폭 FPR을 raw 수준으로 되돌릴 수 없다(순위 악화, 중증).

## 4) 정상 bearing 6개별 FPR·mean_score (val95, raw / A / B)

| bearing | 진폭군 | raw FPR | A FPR | B FPR | raw mean_score | B mean_score |
|---|---|---|---|---|---|---|
| K001 | high | 0.579±0.018 | 0.214±0.034 | 0.211±0.028 | -7.29 | 5.30 |
| K002 | low | 0.171±0.012 | 0.085±0.022 | 0.204±0.025 | -7.66 | 4.97 |
| K003 | high | 0.463±0.024 | 0.082±0.023 | 0.094±0.027 | -7.41 | 3.54 |
| K004 | low | 0.008±0.000 | 0.467±0.048 | 0.539±0.058 | -7.78 | 12.58 |
| K005 | low | 0.106±0.003 | 0.416±0.038 | 0.513±0.045 | -7.71 | 11.89 |
| K006 | high | 0.541±0.006 | 0.156±0.029 | 0.159±0.034 | -7.33 | 4.63 |

> 저진폭 3개 중 **K004·K005** 가 악화를 주도(FPR raw≈0.01·0.11 → B≈0.54·0.51, B mean_score 12.6·11.9로 최고), K002는 거의 불변(0.17→0.20). 고진폭 3개는 모두 raw 대비 대폭 개선. raw는 mean_score가 모든 개체에서 −7.x대로 진폭에 무관, B는 K004/K005만 튐.

## 5) fold 유형 분리 (zero-support vs compositional)

| 지표 | zero-support raw | zero-support B | compositional raw | compositional B |
|---|---|---|---|---|
| 전체 AUROC | 0.669±0.020 | 0.778±0.072 | 0.777±0.021 | 0.866±0.048 |
| shape recall(val95) | 0.266 | 0.795 | 0.383 | 0.738 |
| amp recall(val95) | 0.695 | 0.654 | 0.935 | 0.728 |
| 정상 FPR 고진폭 | 0.529 | 0.168 | 0.524 | 0.116 |
| 정상 FPR 저진폭 | 0.100 | 0.435 | 0.082 | 0.369 |

> 저진폭 FPR 악화는 zero-support(0.10→0.44)·compositional(0.08→0.37) 모두에서 나타나 zero-support 전용 artifact가 아니다.

## 판정

**(a) AUROC 상승의 귀속 — 확인됨(데이터).** recall Δ 합은 전적으로 shape-sensitive 결함에 몰려 있다(shape 평균 0.30→0.78, amp 평균 0.76→0.67). 표 1의 per-fault AUROC Δ 상위 6종(KA22·KA15·KI14·KB27·KI17·KI21)이 모두 shape-sensitive로 정확히 일치. → '역전돼 있던 결함이 정상 위로 올라온 것'이 AUROC 상승의 주원인.

**(b) 저진폭 FPR 악화의 성격 — 중증(순위 문제, 데이터).** percentile을 99까지 올려도 B의 저진폭 FPR은 0.360로 raw(0.065)의 5배 이상에 머물고, 그 대가로 recall이 0.065 깎인다. 임계값 보정만으로는 완화되지 않는다 → **pseudo-LOSO 등 분포정렬의 필요성 강화**.

**(c) 개체 특정 — K004·K005 주도(데이터) → 진폭 분포 이동 가설(가설).** 저진폭 FPR 0.418은 K004(0.54)·K005(0.51)가 끌어올린 값이고 K002(0.20)는 거의 정상이다. K004/K005의 B mean_score가 12.6·11.9로 최고인 것은, shape 정규화 후에도 이 두 held-out 저진폭 정상이 val 분포보다 이상하게 보인다는 뜻 — val↔test 진폭 분포 이동의 직접 증거로 해석된다(인과 확정은 추가 실험 필요, **가설**).

## 주의 / 한계

**데이터로 확인된 사실**
- 전체·per-fault AUROC, per-fault recall, per-bearing FPR 수치는 5-seed 평균±std이며, raw·A는 diag_G3a와 per-fold |Δ|=0으로 정합. sanity(0.696/0.762/0.800) 재현.
- shape-sensitive recall 급증 / amp-sensitive recall 소폭 하락 / 저진폭 FPR 악화(K004·K005 주도) / 고진폭 FPR 개선은 모두 관측된 사실.

**가설·추측(추가 확인 필요)**
- K004·K005의 FPR 급등 원인을 'val↔test 진폭 분포 이동'으로 본 것은 mean_score 정황에 기반한 **해석**이다. 인과 확정에는 val/test 진폭 분포 직접 비교·pseudo-LOSO 검증이 필요.
- pseudo-LOSO가 저진폭 FPR을 실제로 완화할지는 미검증. PU는 축당 2조건뿐인 순수 외삽(zero-support)이라 효과 제한 가능.

**방법상 주의**
- 진폭군 정상 FPR은 정상 pool에 **train bearing을 포함**(diag_G3a `_score_block`과 동일 규약)하므로, 저진폭 FPR은 held-out test-normal 단독 FPR보다 완만할 수 있다. 개체 특정(표 4)은 per-bearing으로 분리해 이 영향 배제.
- recall/FPR은 **window-level**. bearing-level 판정(다수결·집계)과는 다를 수 있다.
- raw per-window는 B3 checkpoint forward-only 재추론(학습 없음) 산출. A/B는 G-3a 캐시 재조합.

