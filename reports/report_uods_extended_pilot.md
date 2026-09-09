# UODS 외부 검증 — 1-split × 5-seed extended pilot

> split=**uods_split_tuning_run_0.json**(Vieira tuning/run_0, 2/1/2), seeds=[2024, 2025, 2026, 2027, 2028]. band=N=6 **log** edges=[1, 3, 10, 32, 102, 323, 1025], threshold=95pct(val-normal).

> 파이프라인 sanity·비용 측정 전용(모델 설정 변경·데이터셋 폐기 근거 아님). scaler·band=train-fit / Fisher·threshold=val, test bearing 미사용.


## 1) method × subgroup AUROC (mean±std, 5 seed)

| method | overall | developing | faulty | nonball_developing | nonball_faulty | ball | nonball | fpr95 |
|---|---|---|---|---|---|---|---|---|
| raw | 0.808±0.012 | 0.797±0.015 | 0.819±0.010 | 0.785±0.019 | 0.801±0.012 | 0.854±0.007 | 0.793±0.016 | 0.130±0.012 |
| shape | 0.874±0.015 | 0.865±0.016 | 0.882±0.014 | 0.875±0.023 | 0.884±0.022 | 0.856±0.074 | 0.880±0.023 | 0.184±0.034 |
| amp | 0.867±0.013 | 0.885±0.010 | 0.849±0.020 | 0.924±0.009 | 0.865±0.023 | 0.784±0.020 | 0.894±0.012 | 0.309±0.094 |
| equal_z | 0.907±0.013 | 0.907±0.011 | 0.907±0.015 | 0.928±0.014 | 0.912±0.016 | 0.867±0.044 | 0.920±0.014 | 0.296±0.092 |
| fisher | 0.868±0.013 | 0.869±0.015 | 0.867±0.012 | 0.887±0.012 | 0.872±0.009 | 0.833±0.052 | 0.880±0.009 | 0.314±0.105 |

## 2) family별 AUROC (mean±std)

| method | inner | outer | ball | cage |
|---|---|---|---|---|
| raw | 0.844±0.027 | 0.629±0.036 | 0.854±0.007 | 0.906±0.007 |
| shape | 0.939±0.038 | 0.797±0.059 | 0.856±0.074 | 0.903±0.035 |
| amp | 0.927±0.040 | 0.765±0.044 | 0.784±0.020 | 0.991±0.011 |
| equal_z | 0.960±0.006 | 0.836±0.051 | 0.867±0.044 | 0.965±0.020 |
| fisher | 0.922±0.022 | 0.806±0.014 | 0.833±0.052 | 0.911±0.025 |

## 3) seed별 overall AUROC

| method | 2024 | 2025 | 2026 | 2027 | 2028 |
|---|---|---|---|---|---|
| raw | 0.794 | 0.807 | 0.811 | 0.830 | 0.799 |
| shape | 0.874 | 0.868 | 0.859 | 0.866 | 0.902 |
| amp | 0.860 | 0.882 | 0.858 | 0.882 | 0.853 |
| equal_z | 0.921 | 0.907 | 0.886 | 0.902 | 0.919 |
| fisher | 0.870 | 0.892 | 0.856 | 0.860 | 0.861 |

## 4) sanity & 실측 시간

- Fisher ≡ product-of-p: seed 전체 max|ΔAUROC| = **5.37e-07** (0 근처 = 정상).
- test 구성: normal 3272 / fault 6544 window. 전 score finite.
- **proposed 학습 seed당**: mean **108.7s** (min 106.4/max 113.3), n=5.
- **raw 학습 seed당**: mean **105.4s** (min 104.4/max 107.8), n=5.
- 잡 전체(학습 5 seed + dump 5): proposed **11:45**(705s) · raw **10:58**(658s), n17 2 GPU 병렬. checkpoint 49MB/개, dump ~130KB/개.

## 5) 전량(eval 100 split) 예상 규모

- **잡 수**: eval 100 split × {proposed, raw} = **200 잡**(각 5-seed). 학습 런 = 100×5×2 = **1000**.
- **GPU 시간(compute)**: 100 × (705+658)s = 136,300s ≈ **37.9 GPU-h**.
  - wall-clock: n17+n16 6 GPU 병렬 ≈ **6.3 h**, n17 3 GPU만 ≈ 12.6 h.
- **디스크**: checkpoint 1000×49MB ≈ **47.9 GB**(보존 시). dump 1000×~130KB ≈ **130 MB**.
  → **dump 후 checkpoint 삭제** 시 dump·fusion JSON만 남겨 ~0.2 GB로 축소 가능(권장).
- split manifest: `make_uods_split.py --source eval:run_5 … run_104`로 100개 사전 생성(seed 무관 고정).

## 6) 해석 & 가드레일 (판정 아님, sanity·비용만)

- **파이프라인 sanity 통과**: Fisher≡product-of-p max|Δ|=5.4e-7(≈0), 전 score finite, seed std 낮음(≤0.015).
  누수 가드(train-fit-only 통계 불변·bearing disjoint)는 Step 4 unit test에서 통과.
- **전이 관찰(참고용)**: proposed(fisher) overall 0.868 > raw 0.808(+0.06)로 외부 bearing에 전이됨. developing/faulty 모두 raw 상회.
- **ball no-load confound 분리 확인**: ball/non-ball·nonball-developing/faulty 별도 산출됨(예: fisher nonball 0.880 vs ball 0.833). ball 결과 단독 일반화 주장 배제 가능.
- ⚠️ **UODS에서 equal_z(0.907) > fisher(0.868)** 관찰. **그러나 동결 규칙상 이 결과로 fusion을 재선택하지 않는다**(fisher 유지).
  UODS는 confirmatory dataset이며 fusion/N/band 재선택 금지. 이 관찰은 전량 단계에서 함께 보고만 한다.
- 전량 실행·외부 baseline·g-final 병합/push는 **별도 승인 대기**.
