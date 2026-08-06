# 작업 A §0 진단 — conditioning speed-sweep NLL 곡선 (Paderborn 023→1, 무학습)

> 무학습·기존 체크포인트 재사용. 동일 정상 window의 **신호(x)는 고정**한 채 flow에 주입되는 **speed 조건값만 격자로 스캔**해 forward NLL 곡선을 뽑고, conditioning 함수의 형태(무시/외삽/폭주)를 실측한다. 성능평가가 아니라 conditioning 함수 형태 진단.

- **무대**: LOSO 023→1(저속 N09 unseen) × 6 LONO(bearing fold) × seed 2026.
- **대상 방식 3종**: static(파일명 고정값 concat) / measured-concat(실측 z-score 6D) / FiLM(실측 mean 3D로 조건 C 변조). no-meta는 speed 입력이 없어 sweep 대상 아님(평평한 기준선).
- **판정 척도**: 원본 meta NLL의 window 간 자연 산포 σ(baseline). speed sweep의 평균 NLL 이동을 이 σ로 정규화(ΔNLL/σ)해, 산포 내(무시)/산포 수준(선형)/산포 수배 이상 급변(폭주)으로 분류.
- 스크립트: `analysis/diagnose_conditioning_speed_sweep.py` (JSON), `analysis/report_conditioning_speed_sweep.py` (본 리포트).

## 0) 게이트0 (sanity) — 통과

sweep 이전 필수 관문. 정상 window에 **원본 meta**를 넣어 forward한 NLL로 AUROC를 재현해 checkpoint 저장값과 대조. **18개(3방식×6 fold) 전부 diff=0.0으로 완전 일치** → 진단 스크립트의 checkpoint 로드·meta 차원·정규화 규약·window 정렬이 `test.py`와 동일함이 확인됨. 따라서 아래 sweep 곡선은 학습된 모델의 실제 거동을 반영한다.

| 방식 | fold별 재현 AUROC = 저장값 (diff=0.0) |
|---|---|
| static | 0.285 / 0.616 / 0.394 / 0.996 / 1.000 / 0.124 |
| measured-concat | 0.068 / 0.978 / 0.881 / 1.000 / 0.872 / 0.139 |
| FiLM | 0.182 / 0.777 / 0.834 / 0.928 / 0.999 / 0.826 |

## 1) 판정 요약

| 방식 | 6 fold 판정 | 대표 |
|---|---|---|
| static | ignore ignore ignore ignore ignore ignore | **무시(불변)** |
| measured-concat | blowup blowup blowup blowup blowup blowup | **폭주(OOD)** |
| FiLM | blowup blowup blowup blowup blowup blowup | **폭주(OOD)** |

- **static → 전 fold 정확히 0.00σ(완전 무시)**: speed를 어떤 값으로 바꿔도 NLL이 소수점까지 불변.
- **measured-concat / FiLM → 전 fold 폭주(OOD)**: train 근처로 speed를 옮기면 NLL이 baseline σ의 수십~수만 배로 급변.

## 2) fold별 상세

`target_z` = unseen 저속(N09) 정상 window의 speed 입력이 train 기준으로 몇 σ인가(measured만). `max/edge/target Δσ` = baseline σ 대비 최대/격자끝/target지점 NLL 이탈.

| 방식 | fold | 판정 | target_z | base σ | max Δσ | edge Δσ | target Δσ |
|---|---|---|---|---|---|---|---|
| static | 1 | ignore | -0.40(norm) | 0.082 | 0.0 | 0.0 | 0.00 |
| static | 2 | ignore | -0.40(norm) | 0.040 | 0.0 | 0.0 | 0.00 |
| static | 3 | ignore | -0.40(norm) | 0.064 | 0.0 | 0.0 | 0.00 |
| static | 4 | ignore | -0.40(norm) | 0.016 | 0.0 | 0.0 | 0.00 |
| static | 5 | ignore | -0.40(norm) | 0.015 | 0.0 | 0.0 | 0.00 |
| static | 6 | ignore | -0.40(norm) | 0.124 | 0.0 | 0.0 | 0.00 |
| measured-concat | 1 | blowup | -710.73 | 3.714 | 18.4 | 18.2 | 0.20 |
| measured-concat | 2 | blowup | -785.80 | 0.072 | 157.4 | 79.6 | 0.28 |
| measured-concat | 3 | blowup | -754.91 | 7.583 | 84.7 | 62.4 | 0.22 |
| measured-concat | 4 | blowup | -673.89 | 8.143 | 136.4 | 6.7 | 0.36 |
| measured-concat | 5 | blowup | -589.36 | 0.029 | 972.6 | 259.8 | 0.31 |
| measured-concat | 6 | blowup | -592.42 | 0.729 | 84.7 | 11.2 | 0.07 |
| FiLM | 1 | blowup | -710.73 | 12.339 | 52.8 | 52.8 | 0.26 |
| FiLM | 2 | blowup | -785.80 | 0.108 | 3036.5 | 617.6 | 1.90 |
| FiLM | 3 | blowup | -754.91 | 0.081 | 1967.3 | 156.9 | 0.29 |
| FiLM | 4 | blowup | -673.89 | 0.036 | 4703.5 | 188.9 | 6.75 |
| FiLM | 5 | blowup | -589.36 | 479.619 | 28.2 | 28.2 | 0.03 |
| FiLM | 6 | blowup | -592.42 | 0.019 | 44269.0 | 582.2 | 2.49 |

## 3) 대표 곡선 (LONO-2)

**static** (x=정규화 speed; train=0, target N09=−0.4):

| speed(norm) | mean NLL | Δσ |
|---|---|---|
| -0.800 | -8.3461 | 0.00 |
| -0.392 | -8.3461 | 0.00 |
| +0.017 | -8.3461 | 0.00 |
| +0.600 | -8.3461 | 0.00 |

→ 전 구간 동일값. **speed 완전 무시.**

**measured-concat** (x=train z-score; train=0, target N09 z≈-786):

| speed z | mean NLL | Δσ |
|---|---|---|
| -790.0 | -0.810 | 0.3 |
| -392.0 | 5.568 | 88.5 |
| -93.5 | -2.516 | -23.3 |
| 6.0 | -6.587 | -79.6 |

→ target(z≈-786)의 원본 위치에선 정상(Δσ≈0)이나, 거기서 speed를 train 쪽으로 옮기면 NLL이 급변·발산. **OOD conditioning 폭주.**

**FiLM** (x=train z-score; train=0, target N09 z≈-786):

| speed z | mean NLL | Δσ |
|---|---|---|
| -790.0 | 60.253 | -1.9 |
| -392.0 | 183.023 | 1134.9 |
| -93.5 | 24.167 | -336.0 |
| 6.0 | -6.241 | -617.6 |

→ target(z≈-786)의 원본 위치에선 정상(Δσ≈0)이나, 거기서 speed를 train 쪽으로 옮기면 NLL이 급변·발산. **OOD conditioning 폭주.**

## 4) 핵심 발견

**① static은 unseen speed를 구조적으로 무시한다 [해석].** 023 train의 3개 세팅은 모두 N15(1500rpm)이라 **정규화 speed가 항상 정확히 0(분산 0)**. train에서 speed 입력이 상수면 meta_encoder의 speed 가중치는 grad=0이고 **weight_decay(5e-4)만 작용해 0으로 감쇠**한다. 그래서 test에서 speed를 아무리 바꿔도 반응이 정확히 0. → 'static 단순 주입은 unseen에서 개선 없음'(TODO 핵심발견 ①)의 **미시적 원인**. (확증은 후속 'static torque축 대조'로.)

**② measured/FiLM은 OOD 외삽으로 폭주한다 → §1′ 정량 증거.** unseen 저속(N09)의 measured speed는 train 기준 **z ≈ −590 ~ −785**(수백 σ 밖). train speed가 거의 상수라 std가 극소 → z가 폭발한다. 이 극단 입력 근방에서 주입기(concat 임베딩 / FiLM γ,β)가 외삽하며 NLL이 baseline σ의 수십~수만 배로 요동. TODO §1′ 'held-out 운행값이 학습분포 밖(OOD)이라 주입기가 외삽하며 예측을 붕괴시킨다'는 가설을 **정량적으로 지지**한다.

**③ 원본 target 위치에서도 conditioning은 불안정하다.** 여러 fold에서 baseline σ 자체가 크다(예 measured LONO4 σ≈8, FiLM LONO5 σ≈480) — speed를 건드리지 않아도 원본 meta만으로 정상 window 간 NLL이 크게 퍼진다. 주입기가 이미 target 근방에서 예민·불안정함을 시사.

## 5) 격자 해상도 한계 (곡선 해석 주의)

measured/FiLM 곡선의 x축은 **z ≈ −785 같은 극단 target에 지배**당한다. 격자를 [target−1, +6] 구간에 25점 균등으로 잡아, 점 간격이 z로 ~30씩이다. 결과적으로 **train 근처(|z|≲6) 해상도를 거의 잃어**, train 부근의 미세 거동이 곡선에 거의 담기지 않는다. **판정(폭주 여부)에는 영향이 없으나**(극단·중간 대역에서 이미 수십~수만 σ 급변이 확인됨), 곡선의 train 근처 모양을 논하려면 격자 재설계(train 근처 조밀 + target 대역 분리)가 필요하다. → 후속 보강 항목.

## 6) 일반화 및 후속 [해석]

- **[해석] 이 패턴은 다른 hold-out 축에도 일반화될 것으로 본다** — 단, 아직 023→1(speed 축)만 봤으므로 **확정이 아니다**. static의 '분산 0 축 → weight-decay로 가중치 사망'과 measured의 'held-out 축 z 폭발 → 외삽 폭주'는 축에 무관한 메커니즘이라, 저토크(013→2, torque 축)·저하중(012→3, force 축) fold에서도 재현될 것으로 예상. **검증 예정**(같은 스크립트 `--axis torque|force`, 해당 split 체크포인트).
- **후속 보강(예정, 이번 미실행)**: (a) measured 격자 재설계(train 근처 조밀화), (b) static torque축 대조(‘speed는 0σ인데 torque는 반응’으로 ① 확증).

## 7) TODO 판정 기준 매핑

| TODO 판정 기준 | 관측 | 해당 방식 |
|---|---|---|
| NLL 거의 불변 → speed 무시 | 전 fold 0.00σ | **static** |
| 완만·선형 → 제한적 외삽 | 해당 없음 | - |
| 범위 밖 급변 → OOD conditioning 폭주 | 전 fold 수십~수만 σ | **measured-concat, FiLM** |

