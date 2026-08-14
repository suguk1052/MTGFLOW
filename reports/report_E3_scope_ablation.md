# 작업 E 3단계 — Order tracking 적용범위 ablation (4 split × 6 LONO × 5 seed)

> 상태: **완료(부정 결론).** raw vs raw+OT를 4 LOSO split 전체·6 LONO·5 seed(=24 fold×5seed)로 확장.
> 재현 게이트(023→1 raw arm s2026 = 아카이브 B3) 통과.
> **최종 판정: OT의 이득은 전역적이지 않다. 24-fold 순효과는 ΔOT −0.011(사실상 null)이고,
> 유일한 순양수 split은 test-time OT를 쓴 023→1(+0.021)뿐이며 그마저 절대 AUROC 0.479→0.500으로
> 역전 detector를 chance 쪽으로 미는 것이지 실검출 이득이 아니다. train-side로 정렬을 학습시킨
> 나머지 3 split은 전부 음수. → 작업 E(order tracking) 트랙 종결(부정 결론) 유지.**
> 관련: TODO §E, 1단계 `report_E1_order_tracking.md`, 2단계 `report_E2_rmsnorm_ot.md`,
> 계획 `~/.claude/plans/vscode-adaptive-allen.md`.

## 목적

1·2단계는 저속 zero-support **023→1** 한 split에 국한됐다(1단계 raw vs raw+OT 5-seed / 2단계 OT×rmsnorm
2×2). 결론은 "OT = 저진폭 target 국소 효과, rmsnorm과 non-stack". 3단계 핵심 질문:

**OT를 023→1 한 split이 아니라 4 LOSO split(123→0 compositional · 023→1 저속 · 013→2 저토크 ·
012→3 저하중) 전체에 적용하면, 저진폭 국소 이득이 어디까지 일반화되고 어디서 뒤집히는가.**

**개입 차이(결론 해석의 핵심).** OT 적용 방식이 split별로 다르다:
- **023→1 = test-time OT.** source가 전부 1500rpm 단일 → OT(상수-SPR 각도영역 리샘플)는 train에서
  정확한 identity(`Dataset/paderborn.py` `if up == down: return sig`). 따라서 **raw checkpoint(B3 재사용)를
  order_track on loader로 재추론**만 하면 raw+OT가 된다. 학습 개입 없음, test-time 표현 변형만.
- **123→0 · 013→2 · 012→3 = train-side representation alignment.** source에 여러 rpm이 섞여 있어 OT가
  non-identity → **flow를 order-resampled window로 학습**해야 한다(`--order_track`, 신규 90잡).
  즉 여기서의 raw+OT는 "정렬된 표현으로 밀도모델을 새로 학습"한 것.

→ 리포트는 **단순 24-fold 평균으로 결론 내리지 않는다.** split(개입 유형)·진폭군·LONO 축으로 분해한다.

## 설계 — 신규 학습은 90잡(train-side 3 split만)

- **raw arm(전 split 공통)**: 기존 B3 5-seed checkpoint **재추론**(학습 0). 24 fold × 5 seed.
- **raw+OT arm**:
  - **023→1**: raw checkpoint를 order_track on으로 재추론(test-time OT, 학습 0). 6 LONO × 5 seed.
  - **123→0·013→2·012→3**: `--order_track` 전용 신규 학습 = **3 split × 6 LONO × 5 seed = 90잡**
    (no-meta, window 2048, nominal rpm 각도 리샘플). 체크포인트 `E3_rawOT_{split}_{LONO}_s{seed}`.
- seed 2024/2025/2026/2027/2028. 지표 = flow-only NLL AUROC(primary) · val95 정상 FPR ·
  seed별 Δ · fault-family(KA/KB/KI) · per-fault paired.
- 학습 90/90 COMPLETED(실패 0), eval 1잡(job 3325) COMPLETED(error/nan 0), 누락 fold 0.

### 재현 게이트 — 통과
- 023→1 raw arm s2026 = 아카이브 B3: LONO1 **0.0876**(OK) / LONO2 **0.5808**(OK).
  (023→1 LONO3~6은 1단계에서 미측정이라 아카이브 대조값 없음 — raw 재추론값만 신규 산출.)

---

## 결과 (5-seed 집계, flow-only AUROC)

### overall 24-fold — 순효과 null

| | raw | +OT | ΔOT | fold+ |
|---|---|---|---|---|
| **24 fold 평균** | 0.696 | 0.685 | **−0.011±0.045** | 13/24 |

24-fold 순효과는 사실상 0(coin-flip). **이 평균만으로는 결론이 안 나오며**, 아래 split·진폭 분해가 본론.

### split별 — 개입 유형이 부호를 가른다

| split | 유형 | 개입 | raw | +OT | ΔOT | fold+ |
|---|---|---|---|---|---|---|
| 123→0 | compositional | train-side align | 0.777 | 0.759 | −0.017±0.054 | 3/6 |
| **023→1** | 저속 zero-support | **test-time OT** | 0.479 | 0.500 | **+0.021±0.030** | 5/6 |
| 013→2 | 저토크 zero-support | train-side align | 0.776 | 0.752 | −0.024±0.028 | 2/6 |
| 012→3 | 저하중 zero-support | train-side align | 0.753 | 0.729 | −0.025±0.046 | 3/6 |

- **순양수는 023→1(test-time OT)뿐**(+0.021, 5/6 fold+). 그러나 절대 AUROC **0.479→0.500** — 이 split은
  고진폭 target fold(LONO1/3/6)가 D의 진폭 confound로 역전(AUROC 0.16~0.29)돼 평균이 chance 아래다.
  OT의 "+0.021"은 **역전된 detector를 chance 쪽으로 조금 미는 것**이지 실검출 이득이 아니다.
  1단계의 저속 LONO2 저진폭 이득(+0.073)은 여기서도 재현되나(아래 LONO2 참조) 여전히 국소적.
- **train-side로 정렬을 학습시킨 3 split은 전부 음수**(−0.017/−0.024/−0.025). 정렬된 표현으로 밀도모델을
  새로 학습해도 이득이 없고 오히려 소폭 악화 → order 정렬 자체가 fault 분리를 새로 만들지 못함.

### 진폭군별 — 고진폭 악화 / 저진폭 미세 개선 (E-1·E-2·D 정합)

| 진폭군 | target | raw | +OT | ΔOT | fold+ |
|---|---|---|---|---|---|
| **high** | K001/K003/K006 | 0.508 | 0.479 | **−0.028±0.045** | 4/12 |
| **low** | K002/K004/K005 | 0.885 | 0.891 | **+0.006±0.039** | 9/12 |

- **고진폭에서 OT가 악화**(−0.028, 4/12만 양수). 2단계에서 rmsnorm+OT가 고진폭 LONO1을 악화(−0.092)시킨
  것과 같은 방향 — order 리샘플이 고진폭 정상↔fault 순위를 되레 흔든다.
- **저진폭은 미세 개선**(+0.006, 9/12 양수)이나 크기가 작고 절대 AUROC가 이미 높다(0.885). 이득 여지가
  큰 곳이 아니라 이미 잘 되는 곳에서의 소폭 변동.

### LONO별

| LONO | target | 진폭 | raw | +OT | ΔOT | fold+ |
|---|---|---|---|---|---|---|
| LONO1 | K001 | high | 0.448 | 0.445 | −0.003±0.028 | 3/4 |
| LONO2 | K002 | low | 0.836 | 0.842 | +0.006±0.062 | 2/4 |
| LONO3 | K003 | high | 0.575 | 0.509 | **−0.066±0.037** | 0/4 |
| LONO4 | K004 | low | 0.920 | 0.928 | +0.007±0.005 | 4/4 |
| LONO5 | K005 | low | 0.898 | 0.903 | +0.005±0.025 | 3/4 |
| LONO6 | K006 | high | 0.500 | 0.484 | −0.016±0.040 | 1/4 |

- LONO3(고진폭 K003) 최악(−0.066, **0/4 fold+**), LONO4(저진폭 K004) 최선(+0.007, 4/4)이나 이미 0.92.
  고진폭 LONO(1/3/6) 3개 중 2개가 음수 — 진폭군 분해와 정합.

### seed별 Δ와 positive seeds/5 (fold별)

전 24 fold에서 seed 부호가 크게 엇갈린다(대표):
- 023→1/LONO2(1단계 이득 fold): +0.076/+0.135/+0.115/+0.053/−0.012 → **+4/5**(1단계 재현).
- 013→2/LONO1: −0.025/−0.072/−0.053/−0.011/−0.082 → **0/5**(일관 악화).
- 013→2/LONO6: 전 seed 음수 → **0/5**.
- 012→3/LONO2: +0.012/−0.038/−0.147/−0.187/−0.004 → **1/5**(seed 흩어짐, 큰 음수 존재).
- 123→0/LONO3: +0.002/−0.208/−0.424/+0.073/−0.042 → **2/5**(고진폭, seed 편차 극심 std 0.178).

positive_seeds가 4~5/5로 안정적으로 양수인 fold는 저진폭·저속(023→1 LONO2/LONO5, 012→3 LONO1/LONO5,
013→2 LONO4) 일부뿐이고, 고진폭 fold는 대부분 0~1/5. **일관된 전역 개선 없음.**

### val95 정상 FPR

- 저진폭 target fold(LONO2/4/5)는 raw·OT 모두 FPR 0.000 유지(OT가 정상 분리를 깨지 않음).
- 고진폭 target fold(LONO1/3, K001/K003)는 raw·OT 모두 FPR≈1.0 — D의 진폭 confound로 val(저진폭)↔
  test(고진폭) 진폭 이동 때문이며 OT로 해소되지 않음(rawFPR≈otFPR).
- 예외: K006(LONO6)·저토크/저하중 일부에서 FPR 0.01~0.08로 낮고 OT가 미세 변동만.

### fault-family (KA/KB/KI) — 진폭군별 ΔOT

| 진폭군 | KA | KB | KI |
|---|---|---|---|
| high | −0.012 | −0.008 | **−0.053** |
| low | +0.010 | −0.001 | +0.006 |

고진폭에서 **KI가 가장 크게 악화**(−0.053) — 2단계에서 KI가 rmsnorm으로 이미 올라가 OT 여지 없던 것과 달리,
여기 train-side align에선 OT가 고진폭 KI를 되레 떨어뜨린다. 저진폭 개선은 KA 중심이나 +0.01로 미미.

### per-fault (역전 fault 중심) — 전 24 fold 집계

| fault | family | raw | +OT | ΔOT(24fold 평균) |
|---|---|---|---|---|
| KA15 | KA | 0.377 | 0.359 | −0.018 |
| KA22 | KA | 0.229 | 0.244 | +0.016 |
| KB27 | KB | 0.447 | 0.425 | −0.022 |

- 1단계에서 023→1/LONO2 국소적으로 크게 올랐던 KA15(+0.084)·KA22(+0.125)·KB27(+0.095)이,
  **전 24 fold로 평균하면 ±0.02 이내로 소멸**하고 절대 AUROC도 여전히 0.5 미만.
  → 역전 fault 회복은 저속·저진폭 특정 fold의 국소 현상이지 OT의 일반 효과가 아님을 재확인.

---

## 최종 판정

- **핵심 질문 답: OT는 전역 해법이 아니다.** 24-fold 순효과 ΔOT −0.011(13/24 fold+)로 null.
- **개입 유형이 부호를 가른다(중요).** 순양수 split은 **test-time OT**를 쓴 023→1(+0.021)뿐이고,
  이는 절대 AUROC 0.479→0.500 — **역전 detector를 chance로 미는 것**이지 실검출 이득이 아니다.
  반면 **train-side representation alignment**로 OT를 학습시킨 3 split(123→0·013→2·012→3)은 전부 음수
  (−0.017/−0.024/−0.025). 즉 order 정렬을 표현·밀도모델 수준에서 학습시켜도 fault 분리는 생기지 않는다.
- **진폭축이 방향을 설명한다(E-1·E-2·D 정합).** 고진폭(K001/3/6) OT 악화(−0.028, KI −0.053 주도),
  저진폭(K002/4/5) 미세 개선(+0.006)이나 이미 잘 되는 곳의 소폭 변동. 저속·저진폭 특정 fold의
  1단계 이득(023→1 LONO2 +0.073)은 재현되나 국소적이고, 역전 fault(KA15/KA22/KB27) 회복도
  전 fold 평균에서는 소멸한다.
- **결론**: OT 적용범위를 4 split로 넓혀도 **무일반화**. raw-vib 밀도모델의 근본 fault 민감도 한계
  (B·D·F, 2단계)를 order 정렬은 test-time·train-side 어느 개입으로도 넘지 못한다.
  **작업 E(order tracking) 트랙 종결(부정 결론) 유지.**

### 산출물
- 결과 JSON: `results/Paderborn/E_order_tracking/phaseE3_scope_4split.json`(gitignore).
- eval 스크립트: `analysis/diagnose_E3_scope_eval.py`.
- 학습 러너: `runners/Paderborn/E_order_tracking/submit_E3_rawOT_train.sh`.
- raw+OT 신규 체크포인트(90개): `results/Paderborn/E3_rawOT_{123to0,013to2,012to3}_{LONO1..6}_s{2024..2028}`(gitignore).
