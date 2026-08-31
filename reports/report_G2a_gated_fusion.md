# 작업 G-2a — Conditional Amplitude-Gated Fusion (재학습 없음 / checkpoint 재추론)

seed [2024, 2025, 2026, 2027, 2028] (5 seed) mean±std. no-meta LOSO 4 split × 6 LONO = 24 fold/seed, window 2048. 재학습 없이 G-1(amp_branch+amp_normalize)·raw B3 checkpoint를 재추론해 per-window S_shape/S_amp/raw_NLL 산출. **convex gate F=(1−g)·z(shape)+g·z(raw), g=σ((S_amp−τ)/s), τ=val-normal S_amp p95, s=val-normal S_amp std.** z·τ·s 전부 val-normal에서만 계산(target-normal은 FPR·ρ 평가에만). FPR threshold=val-normal p95. test-label 튜닝 없음. **primary=p95, p90는 sensitivity 전용.**

## 핵심 요약 (5-seed mean±std, primary p95)

- **전체 AUROC**: gate 0.706±0.064 vs raw 0.696±0.014 / S_shape 0.704±0.069 / S_total 0.631±0.050 (가드레일 0.696).
- **(b) amp-sensitive fault**: gate 0.801±0.047 vs raw 0.892±0.006 / S_shape 0.704±0.042.
- **(c) shape-sensitive fault**: gate 0.580±0.090 vs raw 0.436±0.033 / S_shape 0.704±0.109.
- **(d) 고진폭 target-normal FPR**: gate 0.359±0.081 vs raw 0.623±0.029 (저진폭: gate 0.370±0.056 vs raw 0.000±0.000).
- **ρ(RMS, score)**: gate 0.060±0.132 vs raw 0.966±0.005.

## 1) fold 유형별 (5-seed mean±std, p95)

| fold 유형 | gate | raw | S_shape | S_total | ρ_gate | ρ_raw |
|---|---|---|---|---|---|---|
| zero-support | **0.692±0.080** | 0.669±0.020 | 0.715±0.095 | 0.612±0.070 | 0.045±0.147 | 0.965±0.006 |
| compositional | **0.750±0.078** | 0.777±0.021 | 0.671±0.124 | 0.688±0.084 | 0.105±0.135 | 0.969±0.009 |

## 2) fault군별 branch AUROC (5-seed mean±std, p95)

| fault군 | raw | S_shape | **gate** | S_total |
|---|---|---|---|---|
| amp-sensitive | 0.892±0.006 | 0.704±0.042 | **0.801±0.047** | 0.688±0.035 |
| shape-sensitive | 0.436±0.033 | 0.704±0.109 | **0.580±0.090** | 0.554±0.074 |

## 3) per-fault AUROC (5-seed mean±std, p95)

| fault id | 분류 | raw | S_shape | gate | S_total |
|---|---|---|---|---|---|
| KA04 | A | 0.918±0.009 | 0.715±0.057 | 0.858±0.069 | 0.740±0.057 |
| KA15 | S | 0.377±0.027 | 0.683±0.109 | 0.564±0.090 | 0.533±0.068 |
| KA16 | A | 0.950±0.012 | 0.752±0.051 | 0.924±0.040 | 0.771±0.043 |
| KA22 | S | 0.229±0.013 | 0.664±0.104 | 0.533±0.079 | 0.583±0.076 |
| KA30 | A | 0.836±0.013 | 0.690±0.049 | 0.675±0.042 | 0.550±0.042 |
| KB23 | A | 0.937±0.014 | 0.736±0.035 | 0.923±0.034 | 0.749±0.020 |
| KB24 | A | 0.993±0.004 | 0.690±0.031 | 0.963±0.018 | 0.864±0.017 |
| KB27 | S | 0.447±0.051 | 0.716±0.113 | 0.587±0.094 | 0.549±0.077 |
| KI04 | A | 0.836±0.017 | 0.727±0.047 | 0.705±0.045 | 0.567±0.040 |
| KI14 | S | 0.412±0.049 | 0.699±0.111 | 0.575±0.097 | 0.538±0.074 |
| KI16 | A | 0.856±0.013 | 0.725±0.083 | 0.733±0.084 | 0.665±0.071 |
| KI17 | S | 0.545±0.041 | 0.726±0.105 | 0.593±0.089 | 0.562±0.074 |
| KI18 | A | 0.809±0.009 | 0.597±0.079 | 0.629±0.081 | 0.602±0.057 |
| KI21 | S | 0.603±0.028 | 0.733±0.116 | 0.629±0.096 | 0.558±0.079 |

## 4) 진폭군별 target-normal FPR (5-seed mean±std, p95)

| 지표 | 고진폭(K001/K003/K006) | 저진폭(K002/K004/K005) |
|---|---|---|
| FPR(gate) | 0.359±0.081 | 0.370±0.056 |
| FPR(raw)  | 0.623±0.029 | 0.000±0.000 |
| FPR(S_shape) | 0.095±0.087 | 0.409±0.063 |

## 5) sensitivity — gate percentile p90 (판정 미사용, 참고)

| 지표 | p95 (primary) | p90 (sensitivity) |
|---|---|---|
| 전체 AUROC(gate) | 0.706±0.064 | 0.709±0.063 |
| amp-sensitive(gate) | 0.801±0.047 | 0.812±0.045 |
| shape-sensitive(gate) | 0.580±0.090 | 0.571±0.091 |
| 고진폭 FPR(gate) | 0.359±0.081 | 0.389±0.081 |

## 6) 정합성 체크 (재추론 raw/S_shape ≈ G-1 리포트)

- 재추론 raw 전체 AUROC 0.696±0.014 (G-1 리포트 raw 0.696±0.014).
- 재추론 S_shape 전체 AUROC 0.704±0.069 (G-1 리포트 S_shape 0.704±0.069).
- 두 값이 G-1과 일치하면 재추론 파이프라인 정상(=fusion 비교의 baseline 신뢰 가능).

## 종합 판정 (가드레일 다지표, primary p95)

- [GO] (a) 전체 AUROC≥~0.70/raw·shape 이상 — gate 0.706 vs raw 0.696/shape 0.704
- [GO] (b) amp-sensitive가 S_shape보다 유의 회복 — gate 0.801 vs shape 0.704
- [NO] (c) shape-sensitive 유지(≳0.65) — gate 0.580
- [GO] (d) 고진폭 정상 FPR ≪ raw — gate 0.359 vs raw 0.623

**자동 판정: NO-GO** (조건 (a)~(d) 동시 충족 + 5-seed 분산 확인 필요). GO면 G-2b(end-to-end dual-view+gate) 제안, NO-GO면 multi-band/multi-scale amplitude branch 전환. 최종 판정은 표의 mean±std(분산<gap)까지 사람이 확인.

> 주의: convex gate·표준화·threshold 전부 val-normal only(test 라벨 미사용). 재추론만 있고 재학습 없음. target-normal은 FPR·ρ 평가에만 사용.

