# B-3. CATCH (재구성형 TSAD, ICLR'25) — 표 편입 (재사용 검증)

B-0에서 checkpoint 규칙 = val-normal 재구성 loss 선택 확인(누수 0) → **재추론 불필요, metrics.json 재사용**.
LOSO 24 fold × 5 seed = 120개 metrics.json의 `auroc` 집계. patch 64/64, seq_len 2048, K=1 채널.

## PU (24 LOSO fold) — Table 4 열

| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |
|---|---|---|---|---|---|---|---|---|
| CATCH | 0.539±0.311 | 0.521±0.316 | 0.591±0.290 | —(CATCH per-window id·rms 없음) | —(CATCH per-window id·rms 없음) | —(CATCH per-window id·rms 없음) | —(CATCH per-window id·rms 없음) | —(CATCH per-window id·rms 없음) |
| raw MTGFlow (참조) | 0.696 | | | | | | | |
| 제안 N6+log+Fisher (참조) | 0.877 | | | | | | | |

### paired (24 fold overall AUROC, Wilcoxon+Holm)
| 대상 | Δ(CATCH−대상) | p(Holm) | CATCH 우세/24 |
|---|---|---|---|
| vs raw MTGFlow | -0.158 | 0.000 | 2/24 |
| vs 제안(Fisher) | -0.338 | 0.001 | 6/24 |

## 한계·각주 (B-G1/B-G2)
- **—(CATCH per-window id·rms 없음)**: CATCH `scores.npz`는 point-level(299,521 등)이고 window별 bearing id·RMS를 저장하지 않아 subgroup(amp/shape-sensitive)·진폭군 정상 FPR·ρ(RMS,score) 열은 산출 불가. 행은 유지하고 사유 표기(삭제 금지).
- **프레이밍(B-G2)**: CATCH는 표준 재구성형 TSAD 설계의 참고 행. ① 시점 단위 탐지용 설계 ② K=1 채널이라 채널 간 융합 비활성 ③ seq_len 2048 제약. '우세' 주장 대상 아님.
- 표값 = LOSO 0.539(24셀×5seed). B2 pooled(0.524)는 표 미사용.

> analysis-only. 재추론·모델 변경 없음.