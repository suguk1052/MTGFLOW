# 진단 리포트 — 5-seed 학습 불안정성의 정체: val_loss 기반 checkpoint 선택과 test AUROC의 괴리 (TODO.md §0)

## 정의 (이 문서만 읽어도 되도록)

- **corr(val_loss, test_auroc)**: 한 run(seed 1개)의 전체 학습 epoch(0~39, 총 40개)에 대해 `val_loss_mean`과 `test_auroc_log_only`의 Pearson 상관계수. 음수이고 클수록(예: -0.7) "val loss가 낮은 epoch일수록 test AUROC도 높다"는 바람직한 관계. 0에 가깝거나 양수면 val loss가 AUROC와 무관하거나 반대 방향.
- **selected_auroc**: 실제 production 파이프라인(`main.py`)이 checkpoint로 저장하는 epoch(= `val_loss_mean`이 처음으로 global minimum을 찍는 epoch, `if loss_best > mean_val_loss:` 조건과 동일)의 `test_auroc_log_only` 값. 이게 5-seed 결과표에 실제로 찍히는 AUROC다.
- **best_auroc**: 그 run이 40 epoch 동안 도달했던 `test_auroc_log_only`의 최댓값(어느 epoch이든).
- **gap = best_auroc − selected_auroc**: 모델이 학습 중 도달했던 최고 성능과, val loss 기준으로 실제 선택된 checkpoint 성능의 격차. **gap이 크다 = 모델은 좋은 해를 지나갔지만 val-loss 기준 selection이 그걸 놓쳤다**는 뜻(학습 자체의 발산이 아니라 선택 프로토콜의 문제).
- 모든 수치는 `train_log.jsonl`(이번 진단을 위해 `main.py`에 추가한 관찰용 로그, 아래 "산출물" 참고)에서 그대로 계산했다. GPU 재학습 결과이며 사후 가공 없음.

## 배경

TODO.md §0(학습 안정성 확보)에서 C2(LOSO) 5-seed 재검증 시 measured 조건의 시드 간 AUROC 분산이 매우 크다는 게 확인됐다(std 0.1~0.44). 이번 진단은 그 원인을 좁히기 위해 세 단계로 진행했다:

1. **기각된 가설 (참고용, 상세 근거는 `analysis/diagnose_meta_origin_embedding.py`, `analysis/diagnose_meta_scale_vs_auroc.py` 실행 결과)**:
   - *가설: held-out(원점) 조건이 unseen이라 meta_encoder가 불안정하다* — 같은 시드 안에서 held-out 설정의 임베딩이 seen 설정들과 방향·크기가 거의 동일해 기각.
   - *가설: meta_encoder 출력 스케일(norm)이 크면 성능이 나쁘다* — 120개 (checkpoint, AUROC) 포인트에서 상관계수 -0.046(사실상 무상관)로 기각.
2. **본 진단**: `main.py`에 epoch별 `train_loss_mean`/`val_loss_mean`/`test_auroc_log_only`를 기록하는 관찰용 로깅(`--log_test_auroc`, checkpoint 선택·학습 결과에는 전혀 영향 없음)을 추가하고, 123to0(C2 static, 기존 24 run) + no-meta/pooled 대조군(4 run) + 나머지 3개 LOSO split의 static/measured(LONO 1,4,6 × 3seed, 108 run 중 완료분)를 재실행해 `corr`/`gap`을 계산했다.
3. 위 결과로 "meta+unseen(LOSO)에서만 문제인가, 아니면 no-meta·pooled도 겪는 일반적 문제인가"를 대조군으로 갈랐다.

---

## 2×2 요약 — no-meta/meta × pooled/LOSO

각 run(seed) 단위 corr/gap을 그 칸에 속하는 모든 run에 대해 그대로 모아 집계했다(그룹 평균의 평균이 아니라 개별 run 값 전체의 평균/중앙값/최댓값).

| | **pooled** | **LOSO (unseen)** |
|---|---|---|
| **no-meta** | n=3 (B2), corr=-0.507, gap=0.084 (max 0.119), gap>0.3: 0/3 | n=6 (B3, 123to0+012to3), corr=-0.625, gap=0.024 (max 0.067), gap>0.3: 0/6 |
| **meta** | n=3 (C1_measured만), corr=-0.711, gap=0.212 (max 0.267), gap>0.3: 0/3 | **n=78 (static+measured, 4 split 전체), corr=-0.166, gap=0.359 (max 0.986), gap>0.3: 44/78** |

**읽는 법**: 문제(gap이 크고 0.3을 넘는 run이 절반 이상)는 **meta × LOSO** 칸에만 있다. 나머지 세 칸(no-meta×pooled, no-meta×LOSO, meta×pooled)은 전부 gap>0.3인 run이 0건이다.

> ⚠️ 커버리지 비대칭: `meta×pooled` 칸은 measured만(static pooled 대조군은 안 돌림) n=3으로 표본이 작다. `no-meta×LOSO`는 4개 split 중 123to0/012to3 두 개만 확인했다(023to1/013to2 no-meta 대조군은 없음). 결론의 방향성은 아래 상세표(특히 no-meta가 확인된 두 split이 "심한 split"과 "덜 심한 split" 하나씩이라는 점)로 보강되지만, 완전한 4×2 대조는 아니다.

---

## 상세 표 (11개 조건, 19+24=43개 run 집계)

| 조건 | n | corr 평균 | corr 중앙값 | gap 평균 | gap 중앙값 | gap 최대 | gap>0.3 | gap>0.5 |
|---|---|---|---|---|---|---|---|---|
| **[대조군] no-meta, pooled (B2, LONO4)** | 3 | -0.507 | -0.551 | 0.084 | 0.077 | 0.119 | 0/3 | 0/3 |
| **[대조군] no-meta, LOSO 123to0 (B3, LONO4)** | 3 | -0.573 | -0.642 | 0.029 | 0.019 | 0.067 | 0/3 | 0/3 |
| **[대조군] no-meta, LOSO 012to3 (B3, LONO4)** | 3 | -0.676 | -0.702 | 0.019 | 0.001 | 0.055 | 0/3 | 0/3 |
| **[대조군] meta(measured), pooled (C1_measured, LONO4)** | 3 | -0.711 | -0.658 | 0.212 | 0.225 | 0.267 | 0/3 | 0/3 |
| meta(static), LOSO 123to0 (전체 6 LONO, 기존) | 24 | -0.201 | -0.185 | 0.329 | 0.337 | 0.927 | 13/24 | 4/24 |
| meta(static), LOSO 023to1 (LONO 1,4,6) | 9 | 0.051 | 0.112 | 0.501 | 0.608 | 0.849 | 7/9 | 6/9 |
| meta(measured), LOSO 023to1 (LONO 1,4,6) | 9 | -0.054 | -0.090 | 0.482 | 0.457 | 0.945 | 6/9 | 4/9 |
| meta(static), LOSO 013to2 (LONO 1,4,6) | 9 | -0.494 | -0.506 | 0.214 | 0.209 | 0.403 | 4/9 | 0/9 |
| meta(measured), LOSO 013to2 (LONO 1,4,6) | 9 | 0.107 | 0.077 | 0.491 | 0.463 | 0.955 | 7/9 | 4/9 |
| meta(static), LOSO 012to3 (LONO 1,4,6) | 9 | -0.487 | -0.546 | 0.269 | 0.289 | 0.546 | 4/9 | 1/9 |
| meta(measured), LOSO 012to3 (LONO 1,4,6) | 9 | -0.028 | -0.037 | 0.274 | 0.099 | 0.986 | 3/9 | 2/9 |

핵심 극단 사례: `123to0_LONO3_s2024` — 선택된 checkpoint AUROC 0.073(사실상 랜덤)인데 같은 run이 학습 중 AUROC 1.000에 도달했었다(gap=0.927).

---

## 결론

### [현상 확정] meta + unseen(LOSO) 조합에서만 val_loss 기준 checkpoint 선택이 사실상 복권이다

- **no-meta는 pooled든 LOSO든 안전하다**: corr -0.51~-0.68(강한 음의 상관), gap 0.02~0.08, gap>0.3인 run 0건(총 9건 중 0건).
- **pooled는 meta(measured)가 있어도 안전하다**: corr -0.71, gap 0.21(0.3 문턱 밑), gap>0.3 0건.
- **LOSO + meta는 static이든 measured든, 4개 split 전부에서 깨진다**: corr이 대체로 -0.2~+0.11로 no-meta 대비 뚜렷이 약하고, gap은 평균 0.21~0.50, 최대 0.85~0.99까지 나오며 44/78 run이 gap>0.3이다. 123to0만의 현상이 아니라 **023to1/013to2/012to3에서도 동일한 패턴**이 재현됐다(오히려 023to1/013to2는 measured 조건에서 123to0보다 더 심함).
- 이 네 가지 대조가 교차하므로, 문제는 "학습이 발산한다"거나 "meta가 나쁘다"가 아니라 **"unseen(LOSO) 조건을 meta로 조건화했을 때, val loss(정상 밀도 적합)와 test AUROC(판별력)가 서로 정보를 안 주는 상황이 되고, 그 상태에서 val loss로 checkpoint를 고르면 무작위에 가까운 성능이 선택된다"**는 프로토콜 수준 문제로 확정한다.

### [유력한 해석, 메커니즘 미확정] density 적합 목적과 판별 목적의 불일치

corr이 무너지는 **이유**(왜 unseen+meta 조합에서 정상 밀도 적합과 이상 판별이 분리되는지 — 예: meta 조건화가 unseen 지점에서 정상 분포를 과적합에 가깝게 좁혀 val loss는 계속 낮아지는데 실제 판별 경계는 나빠지는 것인지, 다른 메커니즘인지)는 **이번 진단(로그 상관분석)으로는 확인되지 않았다**. 이 해석은 정황상 유력하지만, 실제로 확인하려면 epoch별 정상/이상 score 분포를 직접 들여다보거나 개입 실험이 추가로 필요하다 — 이번 리포트의 스코프 밖이다.

### 이전에 기각된 가설 (기록용)

같은 조사 세션에서 먼저 검토했다가 근거 부족으로 기각한 가설 두 개:
1. held-out(원점) 조건이 unseen이라 meta_encoder 임베딩이 불안정하다 — 같은 시드 내 seen/unseen 임베딩이 거의 동일해 기각.
2. meta_encoder 출력 스케일(norm)이 클수록 성능이 나쁘다 — 120 포인트 상관계수 -0.046으로 기각.

---

## 산출물 / 재현 경로

- **코드 변경**: `main.py`에 `--log_test_auroc` 플래그 + `train_log.jsonl` 로깅 추가(checkpoint 선택 로직은 미변경, 관찰 전용). 브랜치 `claude/loso-c2-stability-diagnosis`.
- **분석 스크립트**: `analysis/diagnose_meta_origin_embedding.py`, `analysis/diagnose_meta_scale_vs_auroc.py`(기각된 가설 2개), `analysis/diagnose_val_loss_vs_auroc_gap.py`(본 진단의 corr/gap 계산, 이 리포트의 표 산출).
- **재실행 데이터**:
  - 기존: `results/Paderborn/LONO_C2_5seeds/CA_raw_vib_123to0_LONO*_s*` (5-seed, 6 LONO, static) → `train_log.jsonl` 확보용으로 `_lossobs` 접미사로 재실행: `results/Paderborn/CA_raw_vib_123to0_LONO*_lossobs_s*/train_log.jsonl`.
  - 대조군(3seed, `_lossobs3s` 접미사): `results/Paderborn/raw_vib_{123to0,012to3}_LONO4_lossobs3s_s*` (no-meta LOSO), `results/Paderborn/raw_vib_0123C_LONO4_lossobs3s_s*` (no-meta pooled), `results/Paderborn/CA_raw_vib_0123C_LONO4_measured_lossobs3s_s*` (measured pooled).
  - 나머지 3 split(3seed, LONO 1/4/6, static+measured): `results/Paderborn/CA_raw_vib_{023to1,013to2,012to3}_LONO{1,4,6}[_measured]_lossobs3s_s*`.
  - 실행 스크립트: `runners/contrast_lossobs/`, `runners/LONO_C2_123to0_lossobs/`, `runners/LOSO_static_measured_lossobs/`.
- **원본 결과는 전혀 덮어쓰지 않음** — 전부 새 이름(`_lossobs`/`_lossobs3s` 접미사)으로 저장.

## 한계 요약

- `meta×pooled` 칸은 measured만(n=3) 확인, static pooled 대조군 없음.
- `no-meta×LOSO`는 4 split 중 2개(123to0, 012to3)만 확인, 023to1/013to2의 no-meta 대조는 없음(다만 023to1은 기존에 "threshold collapse" 이슈가 별도로 알려져 있어 해석에 참고가 필요 — TODO.md 및 메모리 `mtgflow-023to1-threshold-collapse` 참고).
- LONO은 6개 중 1,4,6 세 개만 확인(나머지 3개 split 기준). 전수(6 LONO × 4 split × 2 조건 × 5 seed)는 GPU 시간상 하지 않았다.
- corr/gap은 **checkpoint 선택 프로토콜의 문제**를 확정할 뿐, 그 프로토콜을 무엇으로 바꿔야 하는지(다음 트랙)는 이 리포트의 범위 밖이다.
