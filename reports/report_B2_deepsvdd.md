# B-2. Deep SVDD (One-Class, 1D-CNN) — PU Table 4 행

입력 raw 2048 window(train-normal StandardScaler, amp_normalize=False). score=‖φ(x)−c‖²(거리=이상). threshold=val-normal 95pct. 24 fold × seed 2024~2028 5-seed 평균.
하이퍼(동결): 1D-CNN(전 층 bias 없음·BN affine=False·LeakyReLU), center=init forward 평균+eps(0.1) 고정, Adam lr1e-3 wd1e-6, MultiStepLR[50], epoch=100, batch=128.

## PU (24 fold) — Table 4 열

| 모델 | overall | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR고 | 정상FPR저 | ρ(RMS,score) |
|---|---|---|---|---|---|---|---|---|
| Deep SVDD (raw) | 0.645±0.099 | 0.635±0.104 | 0.676±0.073 | 0.702 | 0.569 | 0.078±0.054 | 0.041±0.022 | 0.177±0.071 |
| raw MTGFlow (참조) | 0.696 | | | | | | | |
| 제안 N6+log+Fisher (참조) | 0.877 | | | | | | | |

### paired (24 fold overall AUROC, Wilcoxon+Holm)

| 대상 | Δ | p(Holm) | 개선/24 |
|---|---|---|---|
| vs raw MTGFlow | -0.051 | 0.169 | 10/24 |
| vs 제안(Fisher) | -0.231 | 0.000 | 1/24 |

## collapse 진단 (요건 ②, label-free, 하이퍼 불변)

- **기준 미달(collapse=true) fold: 1/120.** 기준: emb_std<1e-3 / score_cv<1e-2 / val_rank_ratio<1.05.
- emb_std 분포: min 9.88e-04 / median 2.07e-03 / max 3.03e-03.
- 미달 fold:
  - 013to2_LONO5_s2025: emb_std=9.88e-04<0.001 (emb_std 9.88e-04, cv 7.03e-01, rank 3.23)
- **판정**: 경계 하회 소수(주로 emb_std 임계 1e-3 근방) — 학습은 정상 수렴, AUROC로 판별 유지. 하이퍼 변경 없음(B-G6).

## GPU peak · 재현성 (요건 ③)

- peak GPU(allocator, `torch.cuda.max_memory_allocated`): max 0.048 GB / mean 0.048 GB (V100-16 대비 무시 가능).
- **cudnn conv 비결정성**: 동일 seed GPU 재실행 시 미세차 가능(`use_deterministic_algorithms` 미적용, 속도 우선). **5-seed 평균으로 보고**.

## 프레이밍 (B-G2)
- Deep SVDD는 deep OCC 기준점. 본문 우세 주장 대상은 OCC 계열 한정. raw 입력 deep OCC가 6-band 특징 없이 얻는 상한을 보여주는 참고 행.