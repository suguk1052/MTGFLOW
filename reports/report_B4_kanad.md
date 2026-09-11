# B-4. KAN-AD (재구성형 TSAD, 참고 행) — PU Table 4 행

입력 raw 2048 window(train-normal StandardScaler, amp_normalize=False). score=mean_L(recon−x)². threshold=val-normal 95pct. 24 fold × seed 2024~2028 5-seed 평균.
어댑터: TSLib exp 래퍼 미사용(train-normal 학습·val-normal loss checkpoint·anomaly_ratio/PA 미사용). 하이퍼(동결): order(d_model)=4, lr0.01, batch128, epoch100, Adam, MSE 재구성.

## PU (24 fold) — Table 4 열

| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |
|---|---|---|---|---|---|---|---|---|
| KAN-AD (raw, 참고) | 0.870±0.124 | 0.862±0.130 | 0.896±0.097 | 0.846 | 0.902 | 0.453±0.420 | 0.067±0.090 | 0.854±0.040 |
| raw MTGFlow (참조) | 0.696 | | | | | | | |
| 제안 N6+log+Fisher (참조) | 0.877 | | | | | | | |

### paired (24 fold overall AUROC, Wilcoxon+Holm)
| 대상 | Δ(KAN-AD−대상) | p(Holm) | KAN-AD 우세/24 |
|---|---|---|---|
| vs raw MTGFlow | +0.174 | 0.000 | 24/24 |
| vs 제안(Fisher) | -0.006 | 0.527 | 9/24 |

## 실측·주의
- peak GPU(`torch.cuda.max_memory_allocated`): max 0.174 GB / mean 0.174 GB. n_params 4.20M.
- **파일럿 비대표성**: 파일럿 단일 fold(012to3_LONO1_s2024)=0.699는 고진폭 K001 hard fold. 24 fold 평균 0.870로 훨씬 높음 — 파일럿 AUROC로 하이퍼 미변경(B-G6).
- **cudnn conv 비결정성**: 동일 seed GPU 재실행 미세차 가능(`use_deterministic_algorithms` 미적용) → 5-seed 평균 보고.
- **프레이밍(B-G2)**: KAN-AD는 최신 TSAD 참고 행. ① 시점 단위 탐지용 설계 ② K=1 채널이라 채널 융합 비활성 ③ seq_len 2048 제약. '우세' 주장 대상 아님(우세 주장은 OCC 계열 한정).