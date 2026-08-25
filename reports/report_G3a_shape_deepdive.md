# 작업 G-3a S_shape 심층 진단 (G-3b 전제조건)

seed [2024, 2025, 2026, 2027, 2028] × 24 fold = **120 per-fold** 재취합 (학습·재추론 없음, analysis-only). G-3a 최우선 지표 **S_shape**(shape branch flow_NLL AUROC)만 단독 해부. 원천 = `diag_G3a_amp_bands/<split>_LONO<n>_s<seed>.json`의 `g1.blocks.shape` + `raw.auroc`(paired B3). 가드레일 no-meta LOSO raw=0.696.

## 핵심 요약 (S_shape, 5-seed × 24 fold mean±std)

- **전체 S_shape AUROC**: **0.776±0.301** (raw 0.696 대비). ρ(RMS,S_shape) -0.269±0.430 (raw≈0.96 → 진폭 confound 제거).
- **fold 유형 분리**: zero-support 외삽 0.767±0.308 / compositional 0.804±0.277.
- **정상 FPR(S_shape)**: 고진폭 0.087±0.194 / 저진폭 0.344±0.329 — 저진폭>고진폭이면 저진폭 정상 threshold 붕괴(val↔test 진폭분포 shift).
- **fault군별**: amp-sensitive 0.756±0.291 / shape-sensitive 0.803±0.361.

## 1) 정상 FPR(S_shape) — 진폭군별 · fold유형별

| 구간 | 고진폭 FPR | 저진폭 FPR | 고진폭 mean(S_shape) | 저진폭 mean |
|---|---|---|---|---|
| 전체 | 0.087±0.194 | 0.344±0.329 | -7.706±0.286 | -7.648±0.290 |
| zero-support | 0.106±0.219 | 0.351±0.337 | -7.685±0.305 | -7.624±0.307 |
| compositional | 0.030±0.032 | 0.324±0.301 | -7.772±0.207 | -7.719±0.220 |

> 고진폭=K001/K003/K006, 저진폭=K002/K004/K005. FPR은 해당 진폭군 정상 window 풀 기준.

## 2) fold 유형별 S_shape AUROC + ρ(RMS,S_shape)

| fold 유형 | S_shape AUROC | ρ(RMS,S_shape) |
|---|---|---|
| 전체 | **0.776±0.301** | -0.269±0.430 |
| zero-support | 0.767±0.308 | -0.267±0.459 |
| compositional | 0.804±0.277 | -0.274±0.328 |

## 3) fault군별 S_shape AUROC

| fault군 | S_shape AUROC |
|---|---|
| amp-sensitive | 0.756±0.291 |
| shape-sensitive | 0.803±0.361 |

### per-fault S_shape AUROC (진단용)

| fault id | 분류 | S_shape AUROC |
|---|---|---|
| KA04 | A | 0.787±0.340 |
| KA15 | S | 0.792±0.370 |
| KA16 | A | 0.815±0.326 |
| KA22 | S | 0.788±0.371 |
| KA30 | A | 0.710±0.319 |
| KB23 | A | 0.780±0.316 |
| KB24 | A | 0.731±0.339 |
| KB27 | S | 0.808±0.363 |
| KI04 | A | 0.761±0.301 |
| KI14 | S | 0.804±0.366 |
| KI16 | A | 0.795±0.342 |
| KI17 | S | 0.813±0.351 |
| KI18 | A | 0.667±0.355 |
| KI21 | S | 0.813±0.359 |

> A=amplitude-sensitive, S=shape-sensitive.

## 4) fold별 Δ(S_shape − raw) — 24 fold (5-seed mean±std, Δ 오름차순)

| fold | 유형 | target진폭 | Δ(shape−raw) | S_shape | raw |
|---|---|---|---|---|---|
| 123to0_LONO5 | compositional | low | **-0.585±0.354** | 0.377±0.338 | 0.962±0.043 |
| 012to3_LONO5 | zero-support | low | **-0.411±0.248** | 0.494±0.268 | 0.905±0.045 |
| 013to2_LONO5 | zero-support | low | **-0.332±0.109** | 0.562±0.116 | 0.894±0.031 |
| 013to2_LONO4 | zero-support | low | **-0.310±0.329** | 0.668±0.321 | 0.978±0.021 |
| 012to3_LONO4 | zero-support | low | **-0.283±0.252** | 0.712±0.248 | 0.995±0.009 |
| 023to1_LONO5 | zero-support | low | **-0.274±0.264** | 0.558±0.255 | 0.832±0.206 |
| 123to0_LONO4 | compositional | low | **-0.261±0.202** | 0.715±0.192 | 0.976±0.031 |
| 012to3_LONO2 | zero-support | low | **-0.035±0.201** | 0.881±0.198 | 0.916±0.026 |
| 013to2_LONO2 | zero-support | low | **-0.018±0.083** | 0.893±0.047 | 0.911±0.041 |
| 023to1_LONO4 | zero-support | low | **0.029±0.287** | 0.762±0.297 | 0.732±0.183 |
| 123to0_LONO2 | compositional | low | **0.043±0.127** | 0.883±0.095 | 0.839±0.077 |
| 012to3_LONO3 | zero-support | high | **0.073±0.381** | 0.701±0.362 | 0.628±0.041 |
| 023to1_LONO2 | zero-support | low | **0.169±0.200** | 0.847±0.228 | 0.678±0.114 |
| 013to2_LONO6 | zero-support | high | **0.197±0.389** | 0.805±0.372 | 0.608±0.028 |
| 013to2_LONO1 | zero-support | high | **0.223±0.297** | 0.810±0.296 | 0.587±0.021 |
| 123to0_LONO6 | compositional | high | **0.267±0.187** | 0.893±0.187 | 0.626±0.039 |
| 123to0_LONO3 | compositional | high | **0.276±0.164** | 0.985±0.011 | 0.709±0.159 |
| 013to2_LONO3 | zero-support | high | **0.294±0.035** | 0.972±0.037 | 0.678±0.020 |
| 023to1_LONO3 | zero-support | high | **0.344±0.434** | 0.628±0.426 | 0.285±0.057 |
| 012to3_LONO6 | zero-support | high | **0.408±0.057** | 0.989±0.014 | 0.581±0.065 |
| 123to0_LONO1 | compositional | high | **0.422±0.064** | 0.970±0.041 | 0.548±0.035 |
| 012to3_LONO1 | zero-support | high | **0.447±0.104** | 0.943±0.111 | 0.496±0.033 |
| 023to1_LONO6 | zero-support | high | **0.613±0.459** | 0.796±0.396 | 0.183±0.075 |
| 023to1_LONO1 | zero-support | high | **0.618±0.354** | 0.779±0.384 | 0.161±0.097 |

> Δ<0 = 해당 fold에서 S_shape가 raw보다 나쁨(손해 fold). Δ>0 = 이득 fold.

## 5) threshold 붕괴 지점 — bearing이 **held-out test 정상**일 때의 FPR(S_shape)

| bearing | 진폭군 | test fold수 | test FPR | mean(S_shape) | mean_score−threshold | mean_rms |
|---|---|---|---|---|---|---|
| K001 | high | 20 | 0.058±0.217 | -7.640±0.278 | -0.072±0.050 | 1.373±0.080 |
| K002 | low | 20 | 0.471±0.390 | -7.806±0.298 | -0.006±0.062 | 0.722±0.093 |
| K003 | high | 20 | 0.123±0.265 | -7.771±0.253 | -0.112±0.099 | 1.145±0.064 |
| K004 | low | 20 | 0.021±0.042 | -7.586±0.330 | -0.047±0.036 | 0.438±0.085 |
| K005 | low | 20 | 0.581±0.445 | -7.615±0.254 | 0.046±0.091 | 0.592±0.127 |
| K006 | high | 20 | 0.067±0.181 | -7.625±0.253 | -0.064±0.061 | 1.588±0.087 |

> 각 bearing이 held-out test 정상으로 등장한 fold(4 split × 5 seed = 20)만 집계 — train(학습에서 봄)·val(정의상 FPR≈0.05 고정)을 제외해 threshold 붕괴를 정직하게 본다. `mean_score−threshold`>0 이면 정상 window 평균 score가 threshold보다 이상(anomaly)쪽 = 붕괴. 저진폭(K004/K005)이 test일 때 FPR이 치솟으면 val↔test 진폭분포 shift로 threshold가 무너진 것.

