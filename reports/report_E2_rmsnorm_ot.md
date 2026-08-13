# 작업 E 2단계 — Order tracking × RMS-normalization 2×2 (023→1 저속 LOSO)

> 상태: **완료(부정 결론).** 5-seed 2×2. 재현 게이트 4/4 통과.
> **최종 판정: raw+OT의 저진폭 이득은 rmsnorm 표현에서 유지·추가되지 않는다.
> OT와 rmsnorm은 같은 저진폭/order 채널을 부분적으로 공유해 stack 되지 않음
> (LONO2 추가 이득 0, LONO1 오히려 악화).**
> 관련: TODO §E, 1단계 리포트 `report_E1_order_tracking.md`, 계획 `~/.claude/plans/soft-gliding-stallman.md`.

## 목적

1단계에서 저속 zero-support(023→1)의 raw vs raw+OT를 5-seed로 비교해, **OT는 저진폭 target
(LONO2, K002)에서만 나타나는 국소 개선**(ΔAUROC +0.073±0.051, KI 계열 주도)으로 판정됐다.
1단계는 raw 표현에 한정됐다.

2단계 핵심 질문: **저진폭 target에서 robust했던 OT 개선이, 진폭 confound를 제거한 RMS-normalized
shape 표현(작업 D의 amp_normalize)에서도 유지되거나 추가 이득을 주는가.** 즉 raw+OT의 이득이
"진폭 단서와 겹쳐 생긴 것"인지 "표현과 무관한 shape/order 정렬 자체의 이득"인지 분리한다.

**2×2 arm**: (1) raw · (2) raw+OT · (3) rmsnorm · (4) rmsnorm+OT.
primary = 023→1 **LONO2**(저진폭 K002). LONO1(고진폭 K001)은 5-seed 대조.
seed 2024/2025/2026/2027/2028. 지표 = flow-only NLL AUROC(primary) · val95 정상 FPR · seed별 Δ ·
fault-family · per-fault paired.

---

## 설계 — 재사용 극대화 (신규 학습 8잡뿐)

- **OT identity 트릭**: 023→1은 source가 전부 1500rpm 단일 → OT(상수-SPR 각도영역 리샘플)는
  train에서 정확한 identity(`Dataset/paderborn.py` `if up == down: return sig`, 배율 1.0). 따라서 한
  checkpoint를 `order_track` off/on 두 loader로 재추론하면 OT 유/무 두 arm이 나온다. `amp_normalize`는
  per-window 연산이라 OT와 직교(loader에서 OT 리샘플 → RMS 정규화 순서).
- **arm1 raw / arm2 raw+OT**: E-1 raw 체크포인트 10개(5seed×2fold) 재추론 — 학습 0.
- **arm3 rmsnorm**: seed 2026은 작업 D 체크포인트(`ampnorm_023to1_{fold}_s2026`) 재사용, 나머지
  seed 2024/2025/2027/2028 × {LONO1,LONO2} = **8잡만 신규 학습**(`--amp_normalize`, no-meta, window 2048,
  E-1/D와 동일 split·인자, `E2_rmsnorm_023to1_{fold}`).
- **arm4 rmsnorm+OT**: rmsnorm 체크포인트 재추론 — 학습 0(OT identity 트릭).
- eval `analysis/diagnose_E2_rmsnorm_ot_eval.py`(arm별 val95 threshold는 각 arm 자신의 val로 계산).
  결과 `results/Paderborn/E_order_tracking/phaseE2_2x2_rmsnorm_ot.json`.

### 재현 게이트 — 4/4 통과
- raw arm(order_track off) s2026 = 아카이브 B3: LONO1 **0.088**(OK) / LONO2 **0.581**(OK).
- rmsnorm arm(order_track off) s2026 = 작업 D flow_only: LONO1 **0.887**(OK) / LONO2 **0.575**(D 0.576, OK).

---

## 결과 (5-seed 집계)

### LONO2 — primary (저진폭 K002)

| 표현 | raw | +OT | Δ(OT) |
|---|---|---|---|
| **raw**     | 0.678±0.114 | 0.752±0.111 | **+0.073±0.051** (4/5 seed +) |
| **rmsnorm** | 0.740±0.283 | 0.744±0.264 | **+0.003±0.112** (null, seed 흩어짐) |

- **핵심: raw에서 robust했던 OT 이득이 rmsnorm에서 소멸.** seed별 OT|rmsnorm =
  +0.178 / −0.036 / −0.169 / +0.028 / +0.015 (부호 엇갈림, 평균 ≈0). raw에서 이득을 주도하던 KI 계열도
  rmsnorm에선 Δ≈0 (KI17 −0.03, KI18 −0.07, KI21 −0.01).
- **비가산성의 원인**: rmsnorm이 이미 저진폭 confound를 처리해 KI를 끌어올려(KI17 raw 0.41→rms 0.68,
  KI18 0.76→0.77, KI21 0.58→0.67) OT가 더 얹을 여지가 없음. interaction −0.070±0.139.
- **rmsnorm 자체도 LONO2 순이득 불안정**: rmsnorm−raw = +0.062지만 **std 0.356**(한 seed만 0.575,
  나머지 0.93~0.97) — F-1의 "rmsnorm은 저진폭 target에서 seed별 새 역전을 만든다"와 정합.
- **정상 FPR 악화**: raw/raw+OT 0.000 → rmsnorm 0.079 → rmsnorm+OT **0.284**.

### LONO1 — 대조 (고진폭 K001)

| 표현 | raw | +OT | Δ(OT) |
|---|---|---|---|
| **raw**     | 0.161±0.097 | 0.164±0.071 | +0.003±0.030 (null, E-1 재확인) |
| **rmsnorm** | 0.972±0.043 | 0.880±0.078 | **−0.092±0.086 (OT가 오히려 악화)** |

- rms=0.972는 **진폭 confound 제거로 고진폭 정상 역전이 풀린 효과**(raw 0.161→0.972, D 정합)이지
  새 fault 민감도가 아니다. rmsnorm−raw = +0.811.
- 그 위에 OT를 얹으면 **악화**(−0.092, seed 5개 중 5개 음수), 특히 저진폭 fault **KA15 −0.23·KA22 −0.28**.
  order 리샘플이 rmsnorm이 잘 세운 정상↔fault 순위를 되레 흔드는 것으로 보임.

### fault-family (seed평균, raw/raw+OT/rms/rms+OT | ΔOT|raw ΔOT|rms)

| fold | KA | KB | KI |
|---|---|---|---|
| LONO1 | 0.15/0.16/0.96/0.82 (+0.02 / **−0.14**) | 0.44/0.46/0.97/0.88 (+0.02 / −0.10) | 0.04/0.02/0.98/0.93 (−0.02 / −0.05) |
| LONO2 | 0.68/0.74/0.73/0.74 (+0.06 / +0.01) | 0.79/0.83/0.80/0.80 (+0.03 / +0.01) | 0.62/0.73/0.72/0.72 (**+0.11** / −0.01) |

LONO2 KI에서 raw의 OT 이득(+0.11)이 rmsnorm에선 −0.01로 사라지는 것이 2×2의 핵심 대비.

---

## 최종 판정

- **핵심 질문 답: NO.** raw+OT의 저진폭 이득(LONO2 +0.073)은 rmsnorm 표현에서 **유지·추가되지 않는다**
  (OT|rmsnorm +0.003±0.112, KI Δ≈0). 고진폭 LONO1에서는 OT가 rmsnorm을 **악화**(−0.092)시킨다.
- **해석**: raw+OT의 이득은 **raw 표현·저진폭 target에 국한된 국소 효과**다. 진폭을 정규화해버리면
  (rmsnorm) OT는 얹을 게 없거나(LONO2) 되레 해가 된다(LONO1). OT와 rmsnorm은 같은 저진폭/order-locked
  채널을 부분적으로 공유해 **stack 되지 않음**(interaction 음수). E-1의 "OT=저진폭 국소 효과" 결론이
  표현 축에서도 재확인된다.
- **결론**: 2×2 조합에서 추가 이득 없음(부정 결론). **작업 E(order tracking) 트랙 종결.** rmsnorm은
  진폭 편향은 정직하게 제거하나(D/F-1), 저진폭 target에서 seed 불안정·정상 FPR 악화를 동반하며
  raw-vib 밀도모델의 근본 fault 민감도 한계(B·D·F)를 넘지 못한다는 기존 결론과 일관.

### 산출물
- 결과 JSON: `results/Paderborn/E_order_tracking/phaseE2_2x2_rmsnorm_ot.json`(gitignore).
- 학습 러너 `runners/Paderborn/E_order_tracking/submit_E2_rmsnorm_train.sh`,
  eval 러너 `submit_E2_eval.sh`, eval 스크립트 `analysis/diagnose_E2_rmsnorm_ot_eval.py`.
- rmsnorm 신규 체크포인트 `results/Paderborn/E2_rmsnorm_023to1_{LONO1,LONO2}_s{2024,2025,2027,2028}`(gitignore).
