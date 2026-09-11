# B-3. CATCH (재구성형 TSAD, ICLR'25) — 표 편입 + per-window 메타 복원

B-0 확인: checkpoint=val-normal 재구성 loss(누수 0) → 재추론 불필요. metrics.json `auroc` 집계 + **CATCH 로더 재실행(모델 없음)으로 per-window id·rms 복원**해 subgroup 열을 채움.
**순서 일치 검증**: (count + test_labels array_equal) — 통과 120 / 실패 0 (fold×seed). patch 64/64, seq_len 2048, K=1 채널.

## PU (24 LOSO fold) — Table 4 열 (seed 평균)

| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |
|---|---|---|---|---|---|---|---|---|
| CATCH | 0.539±0.311 | 0.521±0.316 | 0.591±0.290 | 0.712 | 0.308 | 0.428±0.467 | 0.025±0.022 | 0.963±0.038 |
| raw MTGFlow (참조) | 0.696 | | | | | | | |
| 제안 N6+log+Fisher (참조) | 0.877 | | | | | | | |

### paired (24 fold overall AUROC, Wilcoxon+Holm)
| 대상 | Δ(CATCH−대상) | p(Holm) | CATCH 우세/24 |
|---|---|---|---|
| vs raw MTGFlow | -0.158 | 0.000 | 2/24 |
| vs 제안(Fisher) | -0.338 | 0.001 | 6/24 |

## 각주 (B-G1/B-G2)
- **정상 pool 차이**: CATCH scores.npz에 train score 없음 → 진폭군 정상 FPR·ρ의 정상 pool = **val + test-normal**(제안/B-1은 train+val+test-normal). amp/shape-sensitive는 test-normal을 음성으로 쓰므로 영향 없음.
- **정정**: CATCH LOSO scores.npz는 window-level(= n_test)로 확인 → id/rms 복원·정렬 검증 후 열 채움 완료.
- **프레이밍(B-G2)**: CATCH는 표준 재구성형 TSAD 참고 행. ① 시점 단위 탐지용 설계 ② K=1 채널이라 채널 융합 비활성 ③ seq_len 2048 제약. '우세' 주장 대상 아님. 표값 = LOSO 0.539(B2 pooled 0.524는 표 미사용).

> analysis-only. 재추론·모델 변경 없음. 복원은 CATCH 로더(동일 인자) 재실행으로 id/rms만 산출.