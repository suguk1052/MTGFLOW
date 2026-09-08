# P-2 — Band partition scheme ("왜 균등분할(linear)인가")

> 최종안 G-3a의 amplitude branch **밴드 경계 산출 방식**을 방어하는 ablation.
> **N*=6 고정**(P-1 결정), 분할 **방식만** 비교: linear(현행) vs log vs energy.
> 판정 지표 = Fisher-tail(B, 채택 inference), equal-z(A) 병기. 전 설정 5-seed × 24-fold(P-G1). 완료: 2026-09.
> cond·physical은 이번 제외(사용자 결정): cond=P-G3 정의 논란 회피, physical=Hz 경계 미지정.

## 방법
- **linear**(현행 G-3a): rfft bin(0..1024)을 np.array_split로 6등분. **P-1 N=6 캐시(=g-final=G-5) 재사용**(재학습 없음).
- **log**(신규 학습): DC 제외, bin 1..1024를 로그 상대경계로 6분할 → edges [1,3,10,32,102,323,1025](bin), 데이터 무의존(고정).
  band 폭 [3,7,22,70,221,702] bin → 저주파에 밴드 집중(31~1000 Hz에 3밴드), 고주파는 넓게.
- **energy**(신규 학습): fold별 train-normal 평균 PSD 누적 에너지 1/6 분위 경계(DC 제외, 최소폭 가드). **P-G3 준수**(train-normal만, val/test·fault·label 무사용).
- 파이프라인: 학습 → per-window score dump(forward-only) → diagnose_G5_tail_fusion(CPU, equal-z/Fisher) → 5-seed 집계. --amp_band_scheme 외 구조·추론 무변경.
- **선행 EDA**: report_P2_band_eda.md(경계·band폭·PSD·가드 24/24 미발동·로더 정합성 OK).
- **파일럿 6조건 게이트**(023to1/LONO2): 전량 전 검증. 게이트가 잡은 결함 2건 수정
  (dump 로더의 scheme 미복원 → linear로 target 재구성 / log edges 직렬화 관례 불일치). analysis/verify_P2_pilot.py.

## 결과 (5-seed × 24-fold)

| scheme | **Fisher-tail(B)** | equal-z(A) | zero-support | compositional | amp-sensitive | shape-sensitive | 정상FPR 고 | 정상FPR 저 |
|---|---|---|---|---|---|---|---|---|
| linear (ref) | 0.800±0.049 | 0.762±0.058 | 0.778 | 0.866 | 0.779 | 0.828 | 0.155 | 0.418 |
| **log** (별) | **0.877±0.013** | 0.857±0.014 | 0.855 | 0.943 | 0.842 | 0.923 | 0.132 | 0.458 |
| energy | 0.789±0.048 | 0.742±0.058 | 0.765 | 0.864 | 0.775 | 0.808 | 0.148 | 0.428 |

> raw baseline 0.696. 정상FPR 고=K001/K003/K006, 저=K002/K004/K005. threshold=val-normal 95pct.

### 24-fold paired Wilcoxon signed-rank test (scheme vs linear)

| scheme | 지표 | mean_diff | pos/neg | p(raw) | p(Holm) |
|---|---|---|---|---|---|
| log | Fisher(B) | **+0.076** | **21/3** | 0.0002 | **0.0004** |
| log | equal-z(A) | +0.094 | 22/2 | 0.0001 | **0.0001** |
| energy | Fisher(B) | -0.011 | 10/14 | 0.75 | 0.75 |
| energy | equal-z(A) | -0.020 | 12/12 | 0.79 | 0.79 |

## 해석
- **log가 linear를 큰 폭·고유의로 상회**: Fisher B 0.877 vs 0.800(+0.077), paired p=0.0002(Holm 0.0004), 24 fold 중 21 fold에서 우세.
  게다가 **seed std가 0.049→0.013으로 급감**(seed별 0.860~0.897 범위 vs linear 0.730~0.865) — 안정성도 크게 개선.
- **모든 사전 지정 AUROC 하위군에서 개선(단 저진폭 정상 FPR은 악화)**: zero-support 0.778→0.855, compositional 0.866→0.943,
  amp-sensitive 0.779→0.842, shape-sensitive 0.828→0.923. 고진폭 정상 FPR도 0.155→0.132 개선. **단 저진폭 정상 FPR은 0.418→0.458로 악화**(확정 limitation 축, AUROC엔 부차).
- **[가설] 메커니즘**: 로그 밴드는 저주파(31~1000 Hz)에 3밴드를 배치해, EDA에서 train-normal PSD 에너지의
  ~47~58%가 몰려 있던 저대역의 **조건부 진폭 해상도**를 높인다. 고주파는 넓게 묶어 잡음 밴드의 SNR 저하를 피함.
  (log는 **데이터 무의존 고정 경계**라 누수·과적합 여지 없음 — 단순 재가중.)
- **energy는 linear와 사실상 동률**(-0.011, p=0.75 비유의): fold별 적응 경계가 여기선 이득 없음. adaptive라고 나은 게 아님.
- **잔존 약점**: 저진폭 정상 FPR log 0.458 > linear 0.418(+0.04, 확정 limitation 축). 단 AUROC가 primary이고
  고진폭 FPR은 오히려 개선 → 순효과는 명확히 log 우위. (저진폭 FPR은 G-7에서 임계값 무관 구조적 한계로 종료된 축.)

### 저진폭 정상 FPR paired 일관성 (기존 캐시, 추가 학습 없음)
24-fold seed평균 저진폭 정상 FPR paired(log vs linear): Fisher(B) log 0.458 vs linear 0.418, mean_diff +0.039, 15/9 fold, **Wilcoxon p=0.14(비유의)**; equal-z(A) +0.026, 13/11, p=0.39. → 저진폭 FPR 악화는 **평균 이동 수준이며 fold 일관성은 약함**(AUROC 이득 p=0.0002와 대조). 확정 limitation 축(G-7)이라 승격 게이트로 삼지 않음.

## scheme 결정 (사전 고정 P-G2)
채택 조건 = (1) 전체 AUROC 우위 > seed std · (2) zero-support/compositional 양쪽 비열위 · (3) amp/shape-sensitive 양쪽 비열위 — 3개 동시.
- **log**: (1) 충족(+0.077 > std, paired p=0.0004) · (2) 충족(zero 0.855>0.778, comp 0.943>0.866) · (3) 충족(amp 0.842>0.779, shape 0.923>0.828).
  → **3조건 동시 충족 → P-G2상 채택 자격(clean win).**
- **energy**: (1) 미충족(-0.011, 비유의) → 미채택.

> **결론: P-2는 log 분할이 linear를 clean-win으로 대체(P-G2 충족).** 기여점은 특정 PU 경계가 아니라
> **"저주파 집중 로그 밴드"라는 데이터 무의존 분할 규칙**(전이 시 고정 Hz가 아닌 로그 상대경계 그대로 재사용 가능).
> energy(적응 경계)는 이 데이터에선 이득 없음. **최종안의 밴드 분할을 linear→log로 승격할지는 g-final 동결 해제가 필요한
> 모델 변경이므로 사용자 확인 후 반영**(P 트랙 원칙: ablation 결과의 최종안 승격은 명시적 결정).

## 산출물
- 리포트: 본 파일 · report_P2_band_eda.md · report_P2_scheme_linear_log_energy_vs_linear.md(세부 표·seed별).
- 코드: --amp_band_scheme(main/test/paderborn), analysis/{eda_P2_band_partition,verify_P2_pilot,report_P2_scheme}.py,
  러너 runners/Paderborn/P2_band_partition/(브랜치 claude/p2-band-partition).
- 캐시: results/Paderborn/p2_{log,energy}_window_scores/*.npz(각 120) · diag_P2_{log,energy}_tail_fusion/. linear=diag_G5_tail_fusion 재사용.
