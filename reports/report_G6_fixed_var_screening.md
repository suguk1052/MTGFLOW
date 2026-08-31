# G-6 Fixed-variance Multi-band Amplitude Loss — seed 2026 × 24 fold screening

> 브랜치 `claude/g6-fixed-var-amp`(g3a-amp-bands 분기). seed 2026 단일 screening, G-3a 동일 seed paired.
> equal-z(A)·Fisher-tail(B) 두 fusion 동일 checkpoint 적용. **결론: NO-GO(종료 권고) — 5-seed 미확장.**

## 개입 (G-3a 대비 유일한 변경)
Amp head를 heteroscedastic Gaussian NLL(band별 `μ_k, logσ_k`, 출력 2K)에서 **fixed-variance**
(μ만, 출력 K, σ=1 고정 → `L_amp=(1/K)Σ_k 0.5(a_k−μ_k)²`, MSE형)로 교체(`--amp_fixed_var`).
학습 구조(shared encoder joint 역전파, 6-band z-score target, 등가중 `L=L_shape+L_amp`)·checkpoint
선택·추론 집계·fusion·raw baseline 전부 G-3a와 동일. 재학습 24 fold 전부 성공(loss NaN 0, 실패 0).

## 가설과 결과 (가설 = fixed-var가 amp 검출력↑, 특히 KA30/KI04 회복)
**가설 refuted.** 가설이 지목한 amp-sensitive fault가 오히려 악화:
- KA30 S_amp 0.446→**0.351(−0.095)**, KI04 0.442→**0.329(−0.113)**. (Fisher도 각 −0.049/−0.070.)
- amp-sensitive 군 전체 S_amp −0.031(0.690→0.658), Fisher −0.027. σ 제거가 조건부 진폭 검출을 강화하지 못함.
- S_amp가 오른 곳은 대부분 **shape-sensitive** fault(KA22 +0.106·KI14 +0.095·KA15 +0.091·KI17 +0.086) =
  MSE S_amp가 조건부 진폭이 아니라 일반 shape deviation과 더 상관된 부작용으로 해석.

## 전체 AUROC (24 fold, seed 2026) — G-3a paired
| method | G-3a | G-6 | Δ |
|---|---|---|---|
| shape | 0.715 | 0.736 | +0.021 |
| amp | 0.635 | 0.650 | +0.015 |
| **fusion_A (equal-z=S_total)** | 0.714 | **0.774** | **+0.060** |
| **fusion_B (Fisher-tail, 채택 inference)** | 0.784 | **0.787** | **+0.003** |
| raw (sanity, 모델 독립) | 0.689 | 0.689 | +0.000 |

- **채택 inference(Fisher-tail B) 기준 overall 사실상 flat(+0.003).** G-5 GO로 확정된 최종 추론에서 개선 없음.
- equal-z(A) +0.060은 크나, ① G-3a의 seed-2026 A가 이례적으로 낮음(0.714 vs 5-seed 0.762)이라 과대평가 소지,
  ② 아래 fold 유형에서 **재분배(zero-support↑ / compositional↓)**일 뿐 clean win 아님.
- sanity: fusion_A가 diag_G6 auroc_total을 |Δ|=0으로 재현(캐시·재구현 정합). raw는 G-3a와 완전 동일.

## 재분배 구조 (개선이 균일하지 않음)
| | fusion_A Δ | fusion_B Δ | shape Δ |
|---|---|---|---|
| zero-support (18 fold) | **+0.120** | +0.033 | +0.098 |
| compositional (6 fold) | **−0.121** | **−0.090** | −0.209 |
| amp-sensitive fault | +0.018 | **−0.027** | +0.014 |
| shape-sensitive fault | +0.115 | +0.042 | +0.030 |

G-6은 zero-support 외삽·shape-sensitive는 올리고 **compositional(123to0 기준조건)·amp-sensitive는 떨어뜨림**.
G-3a가 지녔던 amp↔shape 동시보존(G-3/5의 핵심 성취)이 fixed-var에서 부분적으로 무너짐(Fisher amp-sensitive −0.027).

## 정상 FPR·ρ
- 저진폭 정상 FPR 소폭 악화: fusion_A 0.311→0.353, fusion_B 0.378→0.441(잔존 약점 축이 더 나빠짐).
  고진폭 FPR은 fusion_A +0.018·fusion_B −0.031로 혼재. ρ는 여전히 음(진폭 confound 제거 유지, fusion_B −0.38).

## 판정
프로토콜 = "명확한 개선 시에만 5-seed 확장, 아니면 종료". 세 근거로 **NO-GO(종료)**:
1. **채택 inference(Fisher-tail) overall 개선 없음(+0.003).** 실제 쓰는 추론에서 이득 부재.
2. **핵심 가설 refuted** — 겨냥한 KA30/KI04·amp-sensitive 군이 오히려 악화.
3. equal-z +0.060은 fold-type **재분배(compositional −0.121)** + G-3a seed-2026 A 저점의 합작 = clean improvement 아님.

→ **fixed-variance는 채택하지 않음. G-3a(heteroscedastic) + Fisher-tail 현행 기준 유지.** 5-seed 미확장.

## 산출물
- checkpoint `results/Paderborn/g6_<split>_LONO<n>_s2026/`(24), 학습 로그 `runners/slurm_logs/run_Paderborn_g6_*`.
- 재추론 캐시 `results/Paderborn/g6_window_scores/*.npz`(24), 진단 `diag_G6_amp_bands/*.json`(24),
  fusion `diag_G6_tail_fusion/{*.json, aggregate_G5_s2026.json}`.
- 코드: `--amp_fixed_var`(main/test/models/MTGFLOW), `diagnose_G3a_amp_bands.py --g3a_prefix`,
  runner `runners/Paderborn/LONO_G6_s2026/`(gen_g6_runners.py·submit, git 미추적).
