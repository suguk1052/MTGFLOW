# B-0. 외부 baseline 비교군 동결 + 코드 audit

> **B 트랙 1단계 (2026-09-09).** 브랜치 `claude/b-external-baselines`(g-final 분기).
> **학습·GPU 실행 없음.** 동결 목록 7행이 B-G3(누수 0)·B-G7(window 점수) 규칙을 만족하는지를
> **코드 구조로만** 판정한다. 결과는 B-1~B-4 착수 게이트이며, 최신 TSAD 참고 행은
> KAN-AD/DCdetector 중 **1개만** B-4 후보로 통과시킨다.

## 0. 규칙과 표기

체크리스트 8항목(TODO §B-0):

| # | 항목 |
|---|---|
| 1 | 정상 train만 학습에 사용(test를 학습 루프에 섞지 않음) |
| 2 | checkpoint를 val-normal loss(또는 고정 epoch)로 선택 — best-test-F1 로직 제거 가능 |
| 3 | threshold·하이퍼 선택에 test 라벨 미사용(anomaly_ratio·PA 우회 가능) |
| 4 | 독립 2048-window 입력 → **window당 스칼라 점수**(시점 점수면 B-G7 mean 집계) |
| 5 | AUROC용 raw anomaly score 직접 출력 |
| 6 | **seq_len 2048 · V100-16GB 학습 메모리 실측**(1 fold·1 epoch **forward+backward+optimizer step**까지) |
| 7 | scaler를 train-normal에만 fit하도록 PU/UODS 로더 그대로 사용 |
| 8 | 재현성: 동일 seed 재실행 시 AUROC 일치 |

표기 규약:
- ✅ = 통과, ❌ = 위반. ❌에는 수정 가능 여부를 본문에 명기(B-G1: 탈락 사유는 표에서 삭제 금지, `—(사유)`).
- **항목 6은 GPU 미사용 원칙상 B-0에서 실측하지 않는다.** 미실행 모델(Deep SVDD·KAN-AD·DCdetector)은
  **구조상 위험도만** 판정하고, 실제 fwd+bwd+opt 실측은 각 파일럿(Deep SVDD=B-2, TSAD=B-4)에서 확인한다.
  - `✅*` = 구조상 통과 예상(실측 대기), `❌`(항목 6) = 구조상 고위험(실측 없이도 OOM 우려 근거 있음).
  - CATCH·raw MTGFlow·제안은 **이미 V100에서 학습 실측 완료**된 재사용 행이라 실측 수치를 인용한다.

---

## 1. audit 매트릭스 (7행 × 8항목)

| 행 | 모델 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | Isolation Forest | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(CPU) | ✅ | ✅ |
| 2 | OC-SVM(RBF) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅(CPU) | ✅ | ✅ |
| 3 | Deep SVDD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ |
| 4 | CATCH (재사용) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅** | ✅ |
| 5a | **KAN-AD** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅* | ✅ | ✅ |
| 5b | DCdetector | ✅ | ❌→수정가능 | ❌→수정가능 | ✅ | ✅ | ❌ 고위험 | ✅ | ✅ |
| 6 | raw MTGFlow (재사용) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 7 | 제안 N=6+log+Fisher-tail (재사용) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

- `*` 항목 6 = 구조상 통과 예상, 실측은 파일럿(3행=B-2, 5a행=B-4)에서 확정.
- `**` CATCH 항목 7 = MTGFLOW 로더가 아닌 자체 `pu/paderborn_dataset.py`이나 train/val/test id 분리·train-fit → 누수 0(§4 각주).

**결론: KAN-AD가 유일하게 8/8 통과한 TSAD 참고 후보 → B-4 후보로 추천. DCdetector 제외(§5b).**

---

## 2. 고전 baseline — Isolation Forest, OC-SVM (행 1·2)

sklearn 무학습(iterative epoch 없음)·결정론 모델이라 항목 2·6은 구조상 자명 통과.

- **항목 1·7**: 입력 (a) raw 2048 / (b) 6-log-band z-vector 모두 MTGFLOW 로더(`Dataset/paderborn.py`,
  `uods.py`)를 재사용한다. `paderborn.py:356-357` — "StandardScaler는 항상 train setting의 train normal
  파일에만 fit되고, val/test에는 동일 scaler의 transform만 적용" → **train-normal fit 보장**. band z-score도
  train-normal 통계만 사용(`uods.py:5-12`). 학습 루프에 test 미포함. → ✅
- **항목 2**: 반복 학습이 없어 checkpoint 개념 없음(단일 fit). best-test-F1 로직 자체가 존재하지 않음. → ✅
- **항목 3**: IF `n_estimators=100`, OC-SVM `nu=0.05, gamma='scale'` 원 라이브러리 기본값 고정(TODO B-1).
  임계값은 val-normal 95pct(B-G3)로 별도 산출, AUROC는 raw score로 계산 → test 라벨 미개입. → ✅
- **항목 4**: window 1개당 sample 1개 → 점수 1개(정의상 window 스칼라). → ✅
- **항목 5**: IF `score_samples`/`decision_function`, OC-SVM `decision_function`이 raw score 직접 출력. → ✅
- **항목 6**: CPU 모델, GPU 메모리 무관. → ✅(CPU)
- **항목 8**: IF는 `random_state` 고정 시 결정론, OC-SVM은 본래 결정론. 동일 seed → 동일 AUROC. → ✅

**부호 규약(필수).** IF `score_samples`/`decision_function`과 OC-SVM `decision_function`은 모두 **클수록 정상**이다.
따라서 **anomaly score = 부호 반전(−decision_function)**으로 통일하고, AUROC·threshold 산출 모두 이 반전 점수를 쓴다.
(반전을 빼먹으면 AUROC가 1−AUROC로 뒤집히므로 B-1 구현 시 부호 sanity를 반드시 확인.)

**val-normal FPR sanity (B-1 필수 기록).** threshold를 **val-normal anomaly score의 95pct**로 잡으므로,
정의상 val-normal 자체 FPR ≈ 0.05가 되어야 한다(val n ≈ 15,000). B-1의 각 fold에서 이 **val FPR≈0.05 sanity를
산출·리포트**한다 — 크게 벗어나면 부호·스케일·split 파이프라인 결함 신호다.

**OC-SVM 계산량 + train window 서브샘플 규칙 (실험 전 고정).**
OC-SVM(RBF)의 SMO는 비용이 **표본수 n에 지배적**(n×n 커널; 60k²≈3.6e9 항)이고, raw 입력은 2048차원이라 커널 평가마다
2048곱까지 겹친다. PU train-normal은 **fold당 ≈ 59,844 window**
(`results/Paderborn/b3_raw_window_scores/012to3_LONO1_s2024.npz`의 `tr_f.npy` shape `(59844,)`, val 15,000·test 74,895 참고)이라
**raw 전량 OC-SVM은 비현실적**. → **raw 입력 서브샘플 규칙 고정**: fold마다 `min(N, n_train_fold)` 개를
**seed=2024 고정 균등 샘플(비복원)** — OC-SVM은 결정론 유지. **제안 N = 10,000**(≈17% 서브샘플, 커널 10k²×8B≈0.8GB, CPU 수 분).
B-1 착수 시 **24 fold 각각의 train window 수를 먼저 전량 집계해 표로 기록**(silent cap 금지, TODO 규칙)한 뒤 N을 확정한다.
- **band 입력(6차원)은 전량** 사용. 단 OC-SVM 비용은 차원이 아니라 n에 지배되므로 band 60k도 느릴 수 있다 →
  전량이 비현실적이면 **동일 N 규칙으로 폴백**하고 리포트에 그 사실을 명기한다.
- **IF는 raw·band 모두 전량**(O(n log n)이라 60k도 부담 없음). IF만 24 fold × 5 seed, OC-SVM은 1 run(결정론·고정 샘플 seed).

**판정: 8/8 통과(양쪽 입력 (a)(b)). B-1 즉시 착수 가능(CPU).**

---

## 3. Deep SVDD (행 3) — 신규 구현 예정

TODO B-2 사양: 1D-CNN encoder(bias 없음·BN affine 없음, collapse 방지 표준), one-class 목적,
center c = 초기 forward 평균(train-normal), 고정 epoch, score = ‖φ(x)−c‖².

- **항목 1**: 학습·center 모두 train-normal만. test 미참여. → ✅
- **항목 2**: 고정 epoch(원논문 관행) → test 참조 checkpoint 로직 없음. → ✅
- **항목 3**: 점수가 center 거리라 anomaly_ratio·PA 등 임계 하이퍼 불필요. AUROC는 raw 거리로 산출. → ✅
- **항목 4**: window 1개 → 거리 스칼라 1개. → ✅
- **항목 5**: ‖φ(x)−c‖² 자체가 raw anomaly score. → ✅
- **항목 6**: 경량 1D-CNN encoder(attention 없음) → seq_len 2048에서 저위험. **단, B-0 실측 없음 →
  B-2 파일럿(1 fold·1 seed)에서 fwd+bwd+opt 메모리·collapse 확인이 착수 게이트.** → ✅*
- **항목 7**: MTGFLOW 로더 재사용(raw window, train-normal fit). → ✅
- **항목 8**: seed 고정 시 재현. → ✅

**판정: 8/8(항목 6은 B-2 파일럿 실측 대기). B-2에서 진행.**

---

## 4. CATCH (행 4) — 재사용 검증 (B-3)

근거: `CATCH/result_pu/B2_LONO1_s2024/{config.json, metrics.json, scores.npz}` 및
`CATCH/pu/runners/slurm_logs/` 학습 로그 직접 확인.

- **항목 1**: `config.json` — `train_ids=[K003,K004,K005,K006]`, `val_ids=[K002]`, `test_norm_ids=[K001]`.
  train/val/test id가 분리되고 학습 로그가 train_loss/val_recon만 감시 → test 학습 루프 혼입 없음. → ✅
- **항목 2**: 학습 로그 `model, best_val_recon = train(model, train_loader, val_loader, ...)` 및
  `metrics.json.best_val_recon` — **checkpoint를 val(=K002 정상) 재구성 loss로 선택**. best-test-F1 아님. → ✅
- **항목 3**: 표값은 raw score AUROC(`metrics.json.auroc`)로 산출. anomaly_ratio·PA 임계는 AUROC에 미개입. → ✅
- **항목 4**: window-level 점수. `scores.npz` = `{test_scores.npy, test_labels.npy, val_scores.npy}`. → ✅
- **항목 5**: `test_scores.npy`가 window별 raw anomaly score → **per-window score 이미 존재. 재추론 불필요.** → ✅
- **항목 6**: `metrics.json.peak_gpu_mem_gb = 10.153`, `trainable_params = 461,785,152`.
  **학습(fwd+bwd+opt) 실측이 16GB 이내로 이미 완료됨**(patch_size=64). 초기 forward-only OOM 전례는 해소. → ✅
- **항목 7**: MTGFLOW 로더가 아닌 자체 `CATCH/pu/paderborn_dataset.py`. 단 id 분리 + train-normal fit 구조로
  누수는 없음(입력·정규화가 test를 참조하지 않음). → ✅**(자체 로더, 무누수)
- **항목 8**: `seed`가 config에 고정(2024~2028), 재사용 값이라 재실행 불필요. → ✅

**B-3 판정: checkpoint 규칙 = val-normal recon → 재사용 적합. per-window score 존재 → forward-only 재추론 불필요.
Table 4 값 = LOSO 0.539(24셀×5seed). B2 pooled 0.524는 표 미사용(TODO §B-3).**

---

## 5. 최신 TSAD 참고 1개 (행 5) — KAN-AD vs DCdetector

두 후보의 **항목 4(window 스칼라)·6(seq_len 2048 학습 메모리)**을 공개 코드 구조로 집중 검증했다.

### 5a. KAN-AD  ← **B-4 후보 추천**

근거: `Time-Series-Library/models/KANAD.py`(공개 구현).

- **구조**: 재구성형. `Model.anomaly_detection`(74-81행)이 window를 `[B,L,D]`로 재구성해 반환 →
  시점별 재구성오차 → **window 내 mean 집계(B-G7)로 스칼라 점수**. → **항목 4 ✅**
- **구성 요소**: `KANADModel`(7-59행)은 `Conv1d`(order 채널)·`GELU`·`BatchNorm1d`·`Linear(window,window)`뿐 —
  **self-attention 없음**. 최대층은 `final_conv = nn.Linear(2048,2048)` ≈ 4.2M params(≈16MB fp32).
  활성 메모리는 B×(2·order+1)×2048 수준으로 선형. → seq_len 2048에서 O(L²) attention 병목 없음.
  → **항목 6 ✅\*** (구조상 저위험, 실측은 B-4 파일럿 1 fold·1 seed에서 fwd+bwd+opt까지 확인).
- 항목 1(재구성 대상=train-normal), 2(고정 epoch/val loss)†, 3(AUROC는 raw 재구성오차 → anomaly_ratio·PA 불필요)†,
  5(재구성오차 raw 출력), 7(로더 재사용), 8(seed) 모두 통과.

**† 항목 2·3 통과 전제 — TSLib 래퍼 미사용.** KAN-AD를 `Time-Series-Library/exp/exp_anomaly_detection.py`
래퍼로 그대로 돌리면 항목 2·3을 위반한다. 그 래퍼의 `test()`는
`combined_energy = np.concatenate([train_energy, test_energy]); threshold = np.percentile(combined_energy, 100 - anomaly_ratio)`(172-173행)로
**train+test 합산 energy percentile**에 anomaly_ratio를 걸고, `gt, pred = adjustment(gt, pred)`(186행)로
**GT 기반 point-adjustment**를 적용한다 → **DCdetector(solver.py)와 동일한 test 참조 누수**. 또한 학습부 early-stopping도
동일 코드베이스 관행상 test loss를 감시한다. 따라서 KAN-AD는 이 래퍼를 쓰지 않고, **CATCH처럼 MTGFLOW 로더 위
별도 어댑터**(train-normal만 학습·**val-normal loss checkpoint**·**AUROC는 raw 재구성오차**로 산출·anomaly_ratio/PA 미사용)로
실행하는 것을 **B-4 착수 조건**으로 명시한다. (매트릭스 5a행 2·3 = ✅는 이 어댑터 전제하의 값.)

**‡ 정규화 노트 (진폭 정보 보존).** `models/KANAD.py`는 **per-window mean/std(instance/RevIN류) 정규화를 하지 않는다.**
`KANADModel`의 정규화는 `BatchNorm1d`(배치·running-stat, per-window 아님)뿐이고 나머지는 cosine 피처확장·`Conv1d`·`Linear`다
→ **window 진폭 정보가 제거되지 않는다.** 우리의 진폭 민감(amp-sensitive) 과제에서 raw MTGFlow처럼 amplitude confound를
그대로 노출하므로, 스케일 통제는 **MTGFLOW 로더의 train-normal StandardScaler(전역 affine)로만** 이뤄진다(해석·각주용).

**판정: 8/8(항목 2·3은 † 어댑터 전제). TODO §293 가설("경량 구조라 seq_len 2048 메모리 통과 가능성 높음") 구조 확인. B-4 후보로 채택.**

### 5b. DCdetector  ← **B-4 제외**

근거: 공개 repo `DAMO-DI-ML/KDD2023-DCdetector` `solver.py` `test()`/학습 루프 확인.

- **항목 4 ✅**: point energy(`attens_energy = np.concatenate(...).reshape(-1)`) → window 내 mean 집계 가능.
- **항목 5 ✅**: `attens_energy` raw 출력 가능.
- **항목 2 ❌ → 수정 가능**: early-stopping이 `vali_loss1, vali_loss2 = self.vali(self.test_loader)`로
  **test_loader의 loss를 감시**해 checkpoint를 고른다(test 참조). → val-normal 로더로 교체해야 함(코드 수술 필요).
- **항목 3 ❌ → 수정 가능**: threshold = `np.percentile(np.concatenate([train_energy, test_energy]), 100-anomaly_ratio)`
  + **point-adjustment(GT 라벨 사용)**. → AUROC는 raw energy로 우회 가능하나 원 파이프라인은 test 참조.
- **항목 6 ❌ 고위험**: dual **multi-head self-attention**(patch-wise + in-patch) → attention 메모리 **O(L²)**.
  seq_len 2048에서 CATCH의 초기 forward-only OOM 전례와 **동일 위험 계열**. B-0 실측 불가이나
  구조상 학습(fwd+bwd+opt) 메모리 OOM 우려가 KAN-AD보다 명백히 높음.

**판정: 항목 2·3은 우회·수정 가능하나 코드 수술 필요, 항목 6은 O(L²) 고위험.
KAN-AD 대비 열위 → B-4 후보에서 제외.** (B-G1에 따라 목록에서 삭제하지 않고 제외 사유를 여기 보존.
KAN-AD 파일럿이 메모리·점수 산출에서 실패할 경우에만 fallback 재검토.)

---

## 6. raw MTGFlow · 제안 (행 6·7) — 재사용

두 행 모두 그룹 자체 프로토콜(B-G3 누수 0)로 이미 산출된 재사용 행(raw 0.696/0.780, 제안 0.877/0.885).

- 항목 1·7: `Dataset/paderborn.py`·`uods.py` 로더 — train-normal fit(§2 근거 동일). → ✅
- 항목 2: val-normal loss checkpoint(동결 규칙). → ✅
- 항목 3: threshold = val-normal 95pct 고정, AUROC는 raw NLL score. → ✅
- 항목 4·5: window별 NLL 스칼라 raw 출력(per-window .npz 덤프, P-G6). → ✅
- 항목 6: V100에서 학습 실측 완료(기존 실행). → ✅
- 항목 8: seed 2024~2028 재현. → ✅

**판정: 8/8. 표값 재사용.**

---

## 7. 종합

| 항목 | 결과 |
|---|---|
| 8/8 통과 | IF, OC-SVM, Deep SVDD(6은 B-2 실측 대기), CATCH, **KAN-AD**(6은 B-4 실측 대기), raw MTGFlow, 제안 |
| 조건부/탈락 | **DCdetector** — 항목 2·3 수정 필요 + 항목 6 O(L²) 고위험 → B-4 제외(사유 §5b 보존) |
| B-4 후보(1개) | **KAN-AD** |
| B-3(CATCH) | per-window score 존재 → 재추론 불필요, checkpoint=val-recon 재사용 적합, 표값 LOSO 0.539 |
| 다음 게이트 | 항목 6 실측: Deep SVDD=B-2 파일럿, KAN-AD=B-4 파일럿(각 1 fold·1 seed, fwd+bwd+opt) |

**후속(TODO §B 순서)**: B-1(고전, CPU, PU+UODS) → B-2(Deep SVDD, GPU) → B-3(CATCH 재추론 없이 표 편입)
→ B-4(KAN-AD, GPU, audit 통과). GPU 사용은 B-2·B-4만.
