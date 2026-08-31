# G 트랙 최종 inference 경로 — 재현 매뉴얼 (`claude/g-final`)

> **최종 채택안 = G-3a 학습 구조 + G-5 Fisher-tail fusion.** 이 문서는 raw 신호에서
> 최종 window-level 이상점수(`S_fisher`)까지의 재현 가능한 코드 경로를 고정한다.
> 상세 수치·판정 근거는 `report_G_final_summary.md`(비교표)·`report_G5_tail_fusion_5seeds.md`(5-seed).

---

## 0. 한눈에 보는 파이프라인

```
raw window x  ──[G-3a checkpoint, forward-only]──▶  (S_shape, S_amp)  per window
   (학습: main.py --amp_branch)      (dump: dump_G3a_window_scores.py → .npz)
                                                        │
                                          [G-5 Fisher-tail fusion, CPU]
                                          diagnose_G5_tail_fusion.py
                                                        ▼
                                   S_fisher = -2[ln p_shape + ln p_amp]   ← 최종 점수
                                          (val-normal 경험적 tail 확률)
```

- **학습 구조는 G-3a**(dual-branch disentangle, 6-band heteroscedastic amplitude). 학습 시점에 확정된 checkpoint를 그대로 쓴다.
- **fusion만 G-5**(추론단 analysis-only). 재학습·checkpoint 변경 없음. equal-z(A)에서 Fisher-tail(B)로 결합식만 교체.
- 채택 이후 산출된 checkpoint(`results/Paderborn/g3a_*`)·per-window 캐시(`g3a_window_scores/*.npz`)는 이미 디스크에 존재 →
  **Step 3~4(CPU)만으로 최종 수치를 완전 재현**할 수 있다(Step 1~2는 재학습/재추론이 필요할 때만).

---

## 1. 채택 구조 정의 (G-3a)

**모델**: MTGFlow(window-level normalizing flow) + amplitude branch. 코드 `models/MTGFLOW.py`.

- 입력 window `x`를 `x_shape = x / RMS`(형상)·진폭으로 분리. flow는 `x_shape`의 밀도를 학습 → **shape density**.
- shape encoder의 hidden `h_shape`에서 **amplitude head**가 조건부 진폭을 예측:
  - `amp_head: Linear→ReLU→Linear(hidden=32, 2·K)`, **K=6 band** (`--amp_n_bands 6`).
  - 출력 = band별 `(μ_k, logσ_k)` — **heteroscedastic Gaussian**(`models/MTGFLOW.py:252-257`, `log_sigma` clamp [-7,7]).
  - target `a` = per-window **6-band z-scored log-RMS**(train 정규화, `--amp_normalize`).
  - `amp_logprob = mean_k[ -0.5((a_k-μ_k)/σ_k)² - logσ_k - 0.5·ln2π ]` (band축 평균).
- **loss = shape flow NLL + amp NLL** 를 **joint 역전파**(amp gradient가 shape encoder까지 흐름).
- per-window score(둘 다 "클수록 이상", `analysis/diagnose_G3a_amp_bands.py:119` `branch_scores`):
  - `S_shape = -log p_shape(x_shape)`
  - `S_amp   = -amp_logprob(a | h_shape)`

**학습 명령**(예, 24 fold × 5 seed):

```bash
# runners/Paderborn/LONO_G3a_5seeds/run_Paderborn_g3a_<split>_LONO<n>_5seeds.sh
conda run -n mtgflow python main.py --name=paderborn --run_name="g3a_<split>_LONO<n>" \
    --n_blocks=2 --batch_size=256 --window_size=2048 --stride_size=1024 --sampling_rate=64000 \
    --train_load_setting <...> --test_load_setting <...> \
    --train_ids K003 K004 K005 K006 --val_ids K002 --test_norm_ids K001 \
    --seeds 2024 2025 2026 2027 2028 \
    --amp_branch --amp_n_bands 6 --amp_normalize
# → results/Paderborn/g3a_<split>_LONO<n>_s<seed>/model.pth   (완료: 4 split × 6 LONO × 5 seed = 120)
```

24 fold = **4 split**(012to3·013to2·023to1 = zero-support 외삽 / 123to0 = compositional) × **6 LONO**.

---

## 2. Fisher-tail fusion 정의 (G-5, 최종 채택)

코드 `analysis/diagnose_G5_tail_fusion.py`. 두 branch를 각각 **val-normal 경험적 상단 tail 확률**로 변환 후 결합.

- tail 확률(label-free, val-normal only): 정렬된 val-normal score `v`(N개)에 대해
  `p(s) = (#{v_i ≥ s} + 1) / (N + 1)` — rank 기반, `+1` floor로 `ln 0` 방지. score↑(이상) → `p→0`.
- **최종 결합(B = Fisher-tail, 채택)**: `S_fisher = -2·[ ln p_shape + ln p_amp ]` (Fisher χ² 결합, 클수록 이상).
- 비교 기준(A = equal-z, 종전 G-3a S_total): `z_val(S_shape) + z_val(S_amp)` (val-normal 평균/표준편차 z-score 등가중 합).
- **test score/label 무사용**, weight search 없음, threshold = val-normal score 95pct 고정.

> A→B 교체 근거: branch raw scale이 아니라 "각 branch 정상분포에서 얼마나 드문가"를 공통 척도로 통일해 결합 →
> 전체 AUROC 0.762→0.800, amp/shape-sensitive 동시보존 강화(상세 `report_G_final_summary.md`).

---

## 3. 재현 절차 (Step 1~4)

> Step 1~2는 checkpoint/캐시가 없을 때만. 현재 디스크에 둘 다 존재하므로 **Step 3~4(CPU)만으로 최종 리포트 재현 가능.**
> 모든 파이썬 실행은 `conda run -n mtgflow` 접두사. 분석 스크립트는 `analysis/`에서 실행(형제 모듈 import).

| Step | 명령 | 입력 → 출력 | 자원 |
|---|---|---|---|
| 1. 학습 | `runners/Paderborn/LONO_G3a_5seeds/*.sh` | raw → `g3a_*_s<seed>/model.pth` (120) | GPU (완료) |
| 2. per-window dump | `python analysis/dump_G3a_window_scores.py` | checkpoint → `results/Paderborn/g3a_window_scores/*.npz` (120) | GPU forward-only (완료) |
| 3. Fisher-tail fusion | `python analysis/diagnose_G5_tail_fusion.py --seed <s>` (seed별) | `.npz` + diag_G3a JSON → `diag_G5_tail_fusion/*.json` + `aggregate_G5_s<s>.json` | **CPU** |
| 4. 리포트 집계 | `python analysis/report_G5_5seeds.py --seeds 2024 2025 2026 2027 2028` | aggregate JSON → `reports/report_G5_tail_fusion_5seeds.md` | CPU |

**Step 3~4 최소 재현 예**(캐시만 사용, 학습·GPU 불필요):

```bash
cd MTGFLOW/analysis
for s in 2024 2025 2026 2027 2028; do
    conda run -n mtgflow python diagnose_G5_tail_fusion.py --seed $s
done
conda run -n mtgflow python report_G5_5seeds.py --seeds 2024 2025 2026 2027 2028
```

---

## 4. 재현성 sanity (검증됨)

`diagnose_G5_tail_fusion.py`는 fold마다 **fusion_A(equal-z)가 기존 diag_G3a `auroc_total`을 재현**하는지 대조(`repro Δ`)한다.
`claude/g-final`에서 캐시만으로 Step 3~4를 재실행한 결과:

- **전 fold `repro Δ = 0.00e+00`** (120셀 = 5 seed × 24 fold) — 캐시·재구현이 G-3a 원본과 정합.
- 재생성한 `report_G5_tail_fusion_5seeds.md`가 커밋본과 **byte-identical**(`git diff` 무차이).
- overall: raw 0.696±0.014 / S_shape 0.776±0.047 / S_amp 0.652±0.011 / A(equal-z) 0.762±0.058 / **B(Fisher-tail) 0.800±0.049**.

> 재현 상세 표는 `report_G_final_summary.md` §재현성.
