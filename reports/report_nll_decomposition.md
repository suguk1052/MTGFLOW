# 작업 B §1 진단 — NLL 분해 + 진폭 상관 (Paderborn no-meta, 무학습)

seed 2026, no-meta LOSO 기준선, 4 LOSO split × 6 LONO = 24 fold. 기존 체크포인트(`LONO_B3_5seeds`) 재추론만, 새 학습 없음.

## 방법

anomaly score = `-log p_x`, flow에서 `log p_x = log p_Z + log|det J|`로 분해:
- **log p_Z(base density)** = `Σ(-0.5(u-μ)²) + C·ln(1/√2π)` (μ=`base_dist_mean`, mode='rand'라 μ≠0인 flow Gaussian base·단위분산).
- **log|det J|** = flow(BatchNorm+MADE)의 log-det 누적.
- NLL 항 = 각 log 항의 부호 반전(값이 클수록 이상).

**그룹**: source-normal(train+val, source setting·K001~K004) / target-normal(held-out setting **+** 다른 bearing = 이중 held-out) / fault(held-out setting).

> ⚠️ **캐비앗**: 진폭 지표(RMS·peak-to-peak·spectral energy)는 저장된 **스케일된 window**로 계산(원 raw 미저장, 전역 단일 affine 스케일). peak-to-peak는 affine 불변, Spearman은 스케일 불변이라 **단조/상관 판정엔 충분**. base μ≠0이라 log p_Z는 표준정규가 아닌 flow의 학습된 Gaussian base 하 밀도로 해석(분해 정확성엔 영향 없음, 합=total).

## 1) 무결성 게이트

(a) 손수 분해 total == `model.test` (diff<1e-4). (b) total 기반 AUROC == 저장 `overall_auroc` (diff<1e-3). 둘 다 통과해야 분해가 유효.

| split | LONO | (a) diff | (a) pass | (b) repro AUROC | (b) saved | (b) diff | pass |
|---|---|---|---|---|---|---|---|
| 123to0 | 1 | 1.9e-06 | ✓ | 0.5041 | 0.5041 | 0.00 | ✅ |
| 123to0 | 2 | 1.9e-06 | ✓ | 0.7072 | 0.7072 | 0.00 | ✅ |
| 123to0 | 3 | 1.9e-06 | ✓ | 0.9899 | 0.9899 | 0.00 | ✅ |
| 123to0 | 4 | 1.9e-06 | ✓ | 0.9953 | 0.9953 | 0.00 | ✅ |
| 123to0 | 5 | 1.9e-06 | ✓ | 0.9884 | 0.9884 | 0.00 | ✅ |
| 123to0 | 6 | 1.9e-06 | ✓ | 0.7018 | 0.7018 | 0.00 | ✅ |
| 023to1 | 1 | 2.9e-06 | ✓ | 0.0876 | 0.0876 | 0.00 | ✅ |
| 023to1 | 2 | 1.9e-06 | ✓ | 0.5808 | 0.5808 | 0.00 | ✅ |
| 023to1 | 3 | 2.4e-06 | ✓ | 0.3613 | 0.3613 | 0.00 | ✅ |
| 023to1 | 4 | 1.9e-06 | ✓ | 0.6956 | 0.6956 | 0.00 | ✅ |
| 023to1 | 5 | 1.9e-06 | ✓ | 0.5481 | 0.5481 | 0.00 | ✅ |
| 023to1 | 6 | 1.9e-06 | ✓ | 0.1939 | 0.1939 | 0.00 | ✅ |
| 013to2 | 1 | 1.9e-06 | ✓ | 0.5935 | 0.5935 | 0.00 | ✅ |
| 013to2 | 2 | 1.9e-06 | ✓ | 0.8843 | 0.8843 | 0.00 | ✅ |
| 013to2 | 3 | 1.9e-06 | ✓ | 0.6781 | 0.6781 | 0.00 | ✅ |
| 013to2 | 4 | 2.9e-06 | ✓ | 1.0000 | 1.0000 | 0.00 | ✅ |
| 013to2 | 5 | 1.9e-06 | ✓ | 0.8545 | 0.8545 | 0.00 | ✅ |
| 013to2 | 6 | 1.9e-06 | ✓ | 0.6402 | 0.6402 | 0.00 | ✅ |
| 012to3 | 1 | 1.9e-06 | ✓ | 0.5107 | 0.5107 | 0.00 | ✅ |
| 012to3 | 2 | 1.9e-06 | ✓ | 0.9446 | 0.9446 | 0.00 | ✅ |
| 012to3 | 3 | 1.9e-06 | ✓ | 0.6881 | 0.6881 | 0.00 | ✅ |
| 012to3 | 4 | 1.9e-06 | ✓ | 1.0000 | 1.0000 | 0.00 | ✅ |
| 012to3 | 5 | 1.9e-06 | ✓ | 0.8161 | 0.8161 | 0.00 | ✅ |
| 012to3 | 6 | 1.9e-06 | ✓ | 0.5778 | 0.5778 | 0.00 | ✅ |

통과: 24/24. (이하 분석은 통과 fold만.)

## 2) fold 유형별 집계 (핵심 판정, 뭉뚱그리지 않음)

분리 AUROC = target-normal(0) vs fault(1)를 각 항으로 스코어. 0.5에서 멀수록 분리력 (<0.5=역전). domain_shift = source→target normal 평균 이동(source std 표준화). confound = 정상 pooled에서 Spearman(RMS, NLL_total).

| fold 유형 | n | AUROC_total | AUROC_base | AUROC_logdet | shift_base(σ) | shift_logdet(σ) | confound ρ |
|---|---|---|---|---|---|---|---|
| zero-support | 18 | 0.648 | 0.500 | 0.664 | 0.462 | -0.154 | 0.966 |
| compositional | 6 | 0.814 | 0.703 | 0.799 | 0.200 | 0.292 | 0.957 |

**플래그 카운트(통과 fold 중):**

| fold 유형 | domain-shift | likelihood-cancellation | fault-민감도부족 | 진폭-confound |
|---|---|---|---|---|
| zero-support (18) | 5 | 4 | 0 | 18 |
| compositional (6) | 2 | 1 | 0 | 6 |

## 3) 판정표 해석

TODO §1 판정 규약 대응(플래그가 우세한 유형으로 읽음):
- **두 항 모두 이동** → domain shift
- **base 분리·total 안 분리** → likelihood cancellation
- **전 항 겹침** → 표현의 fault 민감도 부족
- **RMS-NLL 단조** → 진폭 confound (발견 ③)

## 4) 통과 fold 상세

| split | LONO | target-norm | AUROC_tot | AUROC_base | AUROC_logdet | shift_b | shift_l | conf ρ | flags |
|---|---|---|---|---|---|---|---|---|---|
| 123to0 | 1 | K001 | 0.504 | 0.705 | 0.502 | 1.355 | 1.828 | 0.970 | domain-shift, likelihood-cancellation, amplitude-confound |
| 123to0 | 2 | K002 | 0.707 | 0.054 | 0.815 | -0.497 | -0.484 | 0.977 | amplitude-confound |
| 123to0 | 3 | K003 | 0.990 | 0.575 | 0.986 | 0.452 | 0.847 | 0.889 | amplitude-confound |
| 123to0 | 4 | K004 | 0.995 | 1.000 | 0.893 | -0.454 | -1.041 | 0.955 | amplitude-confound |
| 123to0 | 5 | K005 | 0.988 | 0.992 | 0.987 | -1.112 | -0.815 | 0.986 | amplitude-confound |
| 123to0 | 6 | K006 | 0.702 | 0.894 | 0.611 | 1.457 | 1.416 | 0.967 | domain-shift, amplitude-confound |
| 023to1 | 1 | K001 | 0.088 | 0.066 | 0.144 | 1.020 | 0.830 | 0.990 | amplitude-confound |
| 023to1 | 2 | K002 | 0.581 | 0.226 | 0.657 | 4.044 | -1.626 | 0.991 | domain-shift, likelihood-cancellation, amplitude-confound |
| 023to1 | 3 | K003 | 0.361 | 0.364 | 0.362 | -1.576 | -0.877 | 0.859 | amplitude-confound |
| 023to1 | 4 | K004 | 0.696 | 0.992 | 0.586 | -2.872 | -1.877 | 0.978 | domain-shift, amplitude-confound |
| 023to1 | 5 | K005 | 0.548 | 0.001 | 0.980 | 4.083 | -1.753 | 0.990 | domain-shift, likelihood-cancellation, amplitude-confound |
| 023to1 | 6 | K006 | 0.194 | 0.053 | 0.217 | 1.023 | 0.023 | 0.970 | amplitude-confound |
| 013to2 | 1 | K001 | 0.593 | 0.718 | 0.586 | 1.038 | 1.824 | 0.973 | domain-shift, amplitude-confound |
| 013to2 | 2 | K002 | 0.884 | 1.000 | 0.767 | 0.787 | -0.487 | 0.968 | amplitude-confound |
| 013to2 | 3 | K003 | 0.678 | 0.998 | 0.552 | 0.372 | 0.920 | 0.982 | amplitude-confound |
| 013to2 | 4 | K004 | 1.000 | 1.000 | 0.998 | -0.351 | -1.202 | 0.970 | amplitude-confound |
| 013to2 | 5 | K005 | 0.855 | 0.063 | 0.891 | 0.762 | -0.434 | 0.952 | amplitude-confound |
| 013to2 | 6 | K006 | 0.640 | 0.610 | 0.644 | 0.273 | 1.470 | 0.979 | amplitude-confound |
| 012to3 | 1 | K001 | 0.511 | 0.845 | 0.488 | 1.117 | 1.498 | 0.978 | domain-shift, likelihood-cancellation, amplitude-confound |
| 012to3 | 2 | K002 | 0.945 | 0.367 | 0.945 | -0.902 | -0.410 | 0.936 | amplitude-confound |
| 012to3 | 3 | K003 | 0.688 | 0.061 | 0.717 | 0.501 | 0.666 | 0.953 | amplitude-confound |
| 012to3 | 4 | K004 | 1.000 | 0.721 | 1.000 | -0.311 | -1.464 | 0.968 | amplitude-confound |
| 012to3 | 5 | K005 | 0.816 | 0.782 | 0.826 | -0.150 | -1.079 | 0.979 | amplitude-confound |
| 012to3 | 6 | K006 | 0.578 | 0.139 | 0.590 | -0.550 | 1.199 | 0.978 | likelihood-cancellation, amplitude-confound |

## 5) 진폭 confound 근거 — 정상 bearing별 (통과 fold pooled)

발견 ③: 고진폭 정상 bearing(K001/K003/K006)이 저진폭(K002/K004/K005)보다 높은 mean NLL_total을 받으면 진폭이 스코어를 지배한다는 직접 증거.

| bearing | 진폭군 | fold수 | mean RMS | mean NLL_total |
|---|---|---|---|---|
| K001 | 고진폭 | 24 | 1.343 | -7.458 |
| K002 | 저진폭 | 24 | 0.737 | -7.823 |
| K003 | 고진폭 | 24 | 1.208 | -7.587 |
| K004 | 저진폭 | 24 | 0.515 | -7.942 |
| K005 | 저진폭 | 24 | 0.599 | -7.863 |
| K006 | 고진폭 | 24 | 1.299 | -7.505 |

## 6) fault 유형(bearing id)별 mean NLL (통과 fold pooled)

| fault id | family | fold수 | mean NLL_total | mean NLL_base | mean NLL_logdet |
|---|---|---|---|---|---|
| KA04 | KA | 24 | -7.069 | 1.163 | -8.232 |
| KA15 | KA | 24 | -7.774 | 1.129 | -8.903 |
| KA16 | KA | 24 | -6.804 | 1.177 | -7.980 |
| KA22 | KA | 24 | -7.864 | 1.124 | -8.988 |
| KA30 | KA | 24 | -7.422 | 1.149 | -8.571 |
| KB23 | KB | 24 | -6.777 | 1.177 | -7.954 |
| KB24 | KB | 24 | -5.935 | 1.371 | -7.306 |
| KB27 | KB | 24 | -7.729 | 1.135 | -8.864 |
| KI04 | KI | 24 | -7.418 | 1.147 | -8.565 |
| KI14 | KI | 24 | -7.762 | 1.132 | -8.893 |
| KI16 | KI | 24 | -7.336 | 1.153 | -8.490 |
| KI17 | KI | 24 | -7.651 | 1.139 | -8.790 |
| KI18 | KI | 24 | -7.424 | 1.150 | -8.573 |
| KI21 | KI | 24 | -7.612 | 1.142 | -8.753 |

## 주의 / 한계

- 무학습 진단(재추론). seed 2026 단일 — 표현/밀도 수준 진단이라 시드 변동은 대상 아님.
- 진폭 지표는 스케일된 window 기준(위 캐비앗). 단조/상관 판정엔 유효, 절대 진폭 비교는 근사.
- target-normal은 setting·bearing 이중 held-out이라 domain-shift와 bearing 개체차가 섞임.
- 판정 임계(AUROC dev 0.15/0.10, shift 1.0σ, confound ρ 0.5)는 서술 편의용 — 표의 원수치를 함께 볼 것.

