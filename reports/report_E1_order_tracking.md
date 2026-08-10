# 작업 E 1단계 — Order tracking 자체 효과 격리 (023→1 저속 LOSO)

> 상태: **Phase A(무학습 정렬 진단) 완료 · 게이트 PASS.** Phase B(학습 비교)는 GPU 승인 대기.
> 관련: TODO §E, 계획 `~/.claude/plans/todo-md-swift-church.md`.

## 목적

D/F 트랙에서 raw-vibration 밀도모델의 fault 민감도 한계가 확정됐다. 남은 국소 가설:
**저속(023→1) unseen 세팅에서 fault 주파수가 Hz축에서 어긋나 정상/이상이 안 갈리는 것**이라면,
order tracking(각도영역 리샘플 → 회전차수 축 정렬)이 이 shift를 보정할 수 있는가.

1단계는 **order tracking 단독 효과**만 격리한다(RMS-normalization 미혼합). Phase A는 무학습으로
"실제 speed shift가 OT로 정렬되는가"를 게이트로 확인하고, 통과 시에만 Phase B(학습)로 넘어간다.

---

## Phase A — 무학습 speed-shift 정렬 진단 (완료)

### 방법
- 스크립트: `analysis/diagnose_order_tracking.py` (GPU 불필요).
- **speed 효과 격리**: 같은 bearing을 `N15_M07_F10`(1500rpm)과 `N09_M07_F10`(900rpm)에서 비교
  (두 세팅은 speed만 다르고 torque 0.7Nm·force 1000N 동일).
- **진단 스펙트럼은 긴 segment로**: 모델 window(2048=<1rev)와 분리해 Welch(nperseg 20480 Hz축 /
  각도영역 SPR·8=20480) → 저차 order 해상도 확보(order 해상도 0.125).
- **OT(등속 nominal)**: 신호를 SPR=2560 samples/rev로 시간영역 리샘플(Phase B loader와 동일 방식).
  **instantaneous(측정 speed 적분)**는 sanity로 병행.
- 지표: 매칭 peak의 이동/정렬 비율(Hz축 기대 0.60, order축 기대 1.00), 전역 log-PSD 상관.

### 사전 사실 (재확인)
| 세팅 | nominal | speed mean | speed std | rev / 2048-window |
|---|---|---|---|---|
| N15_M07_F10 | 1500 | 1499.59 | 0.121 | **0.800** |
| N09_M07_F10 | 900 | 899.71 | 0.182 | **0.480** |

→ 파일 내 speed 변동 <0.02% (사실상 상수) → nominal 근사 타당. window당 <1 회전(정렬 이득 상한 제약).

### 결과
| bearing | kind | Hz peak 이동비 (기대 0.60) | order 정렬비 (기대 1.00) | order 잔차 | corr_hz → corr_order | nominal↔inst (src/tgt) |
|---|---|---|---|---|---|---|
| K001 | 정상(고진폭) | 0.596 (n=7) | 1.002 (n=8) | 0.047 | 0.224 → 0.161 | 0.99 / 0.89 |
| K002 | 정상(저진폭) | 0.595 (n=5) | 0.995 (n=7) | 0.034 | 0.498 → 0.085 | 0.99 / 0.83 |
| KA04 | fault(고진폭) | 0.602 (n=6) | **1.000** (n=7) | **0.000** | 0.376 → **0.710** | 1.00 / 1.00 |
| KA15 | fault(저진폭) | 0.559 (n=4) | 0.999 (n=6) | 0.016 | 0.507 → 0.509 | 1.00 / 0.97 |

**정상 게이트 요약**: Hz 이동비 median **0.595**, order 정렬비 median **0.998**.

### 해석
1. **speed-proportional Hz shift 실재 확인**: 매칭 peak가 전 bearing에서 0.56~0.60배로 이동
   (=900/1500). 저속 unseen 세팅에서 주파수가 실제로 어긋난다는 가설의 전제 성립.
2. **OT가 order-locked 성분을 정렬**: 같은 peak들이 OT 후 order축에서 비율 ~1.0(잔차 ~0.03)로 정렬.
   특히 고진폭 fault **KA04는 bearing 결함 harmonic(order 6.1·9.1·12.25·15.25·18.25)이 완벽 겹침**,
   전역 order 상관이 0.376→**0.710**으로 크게 상승 → 결함 신호는 회전차수에 고정(order-locked)돼 OT로 잘 정렬.
3. **nominal ≈ instantaneous** (order 스펙트럼 상관 0.83~1.00) → 실시간 speed noise·interpolation
   부수변수 없이 **nominal(파일평균) 기반 OT를 primary로** 써도 됨(sanity 통과).
4. **⚠️ 정상 bearing 전역 상관은 오히려 하락**(corr_hz 0.22~0.50 → corr_order 0.09~0.16):
   정상 진동 에너지는 **회전차수에 고정되지 않은 성분(구조 resonance·broadband)**이 지배하는데,
   이 고정-Hz 성분은 두 speed에서 order축으로 서로 다른 위치로 매핑돼 **정렬이 오히려 나빠진다**(물리적으로 정상).
   ⇒ OT는 fault 관련 order-locked 성분은 정렬하지만, **정상의 지배 에너지는 정렬하지 못한다.**
   이는 정렬 이득이 국소적이라는 기대치 하향(TODO §E)과 정합.

### 게이트 판정 — **PASS (단, 기대치 하향 유지)**
- (a) speed-proportional Hz shift 실재: **충족** (이동비 median 0.595).
- (b) OT 후 order-locked 성분 mismatch 유의 감소: **충족** (정렬비 median 0.998, 잔차 ~0.03;
  고진폭 fault 전역 상관 0.38→0.71).
- 정상 전역 상관 하락은 게이트 실패가 아니라 "정상은 비-order-locked 성분 지배"라는 물리적 관찰이며,
  Phase B에서 **OCC 학습된 밀도모델의 순 AUROC 효과**로 최종 판단해야 할 대상이다.

⇒ **Phase B 진행 근거 성립.** 단 <1rev/window·정상 비-order-locked 지배로 인해 국소·제한적 효과 예상.

---

## Phase B — 학습·평가 (GPU 승인 대기)

- `order_track` 옵션을 loader/main/test에 배관(`amp_normalize`와 동일 패턴, source/target 대칭 규칙,
  nominal primary). 023→1 **LONO1(K001 고진폭)/LONO2(K002 저진폭)**에서 **seed 2026** 단일로
  raw vs raw+OT 비교(AUROC primary, 정상 FPR secondary). 효과 확인 시에만 5-seed 확장. compositional 제외.
- (결과 표·결론은 Phase B 실행 후 이 절에 추가.)
