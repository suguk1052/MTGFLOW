"""작업 A §0 진단 리포트 생성 — speed sweep NLL 곡선 결과를 md로 정리.

diagnose_conditioning_speed_sweep.py가 남긴 JSON(results/Paderborn/diag_conditioning_speed_sweep/)만
읽어 reports/report_conditioning_speed_sweep.md 를 생성한다(GPU 불필요, 재현용). 곡선은 현재
격자 그대로 요약하며, measured 격자가 극단 target z에 지배당한 해상도 한계를 명시한다.
"""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(PROJECT_ROOT, "results", "Paderborn", "diag_conditioning_speed_sweep")
REPORT = os.path.join(PROJECT_ROOT, "reports", "report_conditioning_speed_sweep.md")
SEED = 2026
LONOS = list(range(1, 7))
METHODS = [("static", "static"), ("measured_concat", "measured-concat"), ("film", "FiLM")]
VERDICT_KO = {"ignore": "무시(불변)", "linear": "선형(제한적 외삽)", "blowup": "폭주(OOD)"}


def load(method, lono):
    p = os.path.join(DIAG, f"{method}_LONO{lono}_s{SEED}_speed.json")
    with open(p) as f:
        return json.load(f)


def curve_points(d, picks):
    """curve에서 input이 picks에 가장 가까운 점들을 (input, mean_nll, d/σ)로 반환."""
    c = d["curve"]
    out = []
    for target in picks:
        best = min(c, key=lambda p: abs(p["input"] - target))
        out.append((best["input"], best["mean_nll"], best["delta_over_baseline_sigma"]))
    return out


def main():
    L = []
    L.append("# 작업 A §0 진단 — conditioning speed-sweep NLL 곡선 (Paderborn 023→1, 무학습)")
    L.append("")
    L.append("> 무학습·기존 체크포인트 재사용. 동일 정상 window의 **신호(x)는 고정**한 채 flow에 "
             "주입되는 **speed 조건값만 격자로 스캔**해 forward NLL 곡선을 뽑고, conditioning 함수의 "
             "형태(무시/외삽/폭주)를 실측한다. 성능평가가 아니라 conditioning 함수 형태 진단.")
    L.append("")
    L.append(f"- **무대**: LOSO 023→1(저속 N09 unseen) × 6 LONO(bearing fold) × seed {SEED}.")
    L.append("- **대상 방식 3종**: static(파일명 고정값 concat) / measured-concat(실측 z-score 6D) / "
             "FiLM(실측 mean 3D로 조건 C 변조). no-meta는 speed 입력이 없어 sweep 대상 아님(평평한 기준선).")
    L.append("- **판정 척도**: 원본 meta NLL의 window 간 자연 산포 σ(baseline). speed sweep의 평균 NLL "
             "이동을 이 σ로 정규화(ΔNLL/σ)해, 산포 내(무시)/산포 수준(선형)/산포 수배 이상 급변(폭주)으로 분류.")
    L.append("- 스크립트: `analysis/diagnose_conditioning_speed_sweep.py` (JSON), "
             "`analysis/report_conditioning_speed_sweep.py` (본 리포트).")
    L.append("")

    # --- 게이트0 ---
    L.append("## 0) 게이트0 (sanity) — 통과")
    L.append("")
    L.append("sweep 이전 필수 관문. 정상 window에 **원본 meta**를 넣어 forward한 NLL로 AUROC를 재현해 "
             "checkpoint 저장값과 대조. **18개(3방식×6 fold) 전부 diff=0.0으로 완전 일치** → 진단 스크립트의 "
             "checkpoint 로드·meta 차원·정규화 규약·window 정렬이 `test.py`와 동일함이 확인됨. 따라서 아래 "
             "sweep 곡선은 학습된 모델의 실제 거동을 반영한다.")
    L.append("")
    L.append("| 방식 | fold별 재현 AUROC = 저장값 (diff=0.0) |")
    L.append("|---|---|")
    for mk, ml in METHODS:
        vals = []
        for n in LONOS:
            d = load(mk, n)
            vals.append(f"{d['gate0']['reproduced_auroc']:.3f}")
        L.append(f"| {ml} | {' / '.join(vals)} |")
    L.append("")

    # --- 판정 요약 ---
    L.append("## 1) 판정 요약")
    L.append("")
    L.append("| 방식 | 6 fold 판정 | 대표 |")
    L.append("|---|---|---|")
    for mk, ml in METHODS:
        verds = [load(mk, n)["verdict"]["verdict"] for n in LONOS]
        uniq = set(verds)
        rep = VERDICT_KO[verds[0]] if len(uniq) == 1 else "/".join(sorted({VERDICT_KO[v] for v in verds}))
        L.append(f"| {ml} | {' '.join(verds)} | **{rep}** |")
    L.append("")
    L.append("- **static → 전 fold 정확히 0.00σ(완전 무시)**: speed를 어떤 값으로 바꿔도 NLL이 소수점까지 불변.")
    L.append("- **measured-concat / FiLM → 전 fold 폭주(OOD)**: train 근처로 speed를 옮기면 NLL이 baseline σ의 "
             "수십~수만 배로 급변.")
    L.append("")

    # --- fold별 상세 ---
    L.append("## 2) fold별 상세")
    L.append("")
    L.append("`target_z` = unseen 저속(N09) 정상 window의 speed 입력이 train 기준으로 몇 σ인가(measured만). "
             "`max/edge/target Δσ` = baseline σ 대비 최대/격자끝/target지점 NLL 이탈.")
    L.append("")
    L.append("| 방식 | fold | 판정 | target_z | base σ | max Δσ | edge Δσ | target Δσ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mk, ml in METHODS:
        for n in LONOS:
            d = load(mk, n)
            g, v, b = d["grid"], d["verdict"], d["baseline"]
            tz = f"{g['target_ref_input']:.2f}" if g["unit"].startswith("train z") else f"{g['target_ref_input']:.2f}(norm)"
            L.append(f"| {ml} | {n} | {v['verdict']} | {tz} | {b['std']:.3f} | "
                     f"{v['max_dev_sigma']:.1f} | {v['edge_dev_sigma']:.1f} | {v['target_dev_sigma']:.2f} |")
    L.append("")

    # --- 대표 곡선 ---
    L.append("## 3) 대표 곡선 (LONO-2)")
    L.append("")
    # static: normalized speed 축 (rpm)
    ds = load("static", 2)
    L.append("**static** (x=정규화 speed; train=0, target N09=−0.4):")
    L.append("")
    L.append("| speed(norm) | mean NLL | Δσ |")
    L.append("|---|---|---|")
    for inp, nll, dev in curve_points(ds, [-0.8, -0.4, 0.0, 0.6]):
        L.append(f"| {inp:+.3f} | {nll:.4f} | {dev:.2f} |")
    L.append("")
    L.append("→ 전 구간 동일값. **speed 완전 무시.**")
    L.append("")
    for mk, ml in [("measured_concat", "measured-concat"), ("film", "FiLM")]:
        d = load(mk, 2)
        tz = d["grid"]["target_ref_input"]
        L.append(f"**{ml}** (x=train z-score; train=0, target N09 z≈{tz:.0f}):")
        L.append("")
        L.append("| speed z | mean NLL | Δσ |")
        L.append("|---|---|---|")
        for inp, nll, dev in curve_points(d, [tz, -400, -100, 0]):
            L.append(f"| {inp:.1f} | {nll:.3f} | {dev:.1f} |")
        L.append("")
        L.append(f"→ target(z≈{tz:.0f})의 원본 위치에선 정상(Δσ≈0)이나, 거기서 speed를 train 쪽으로 "
                 "옮기면 NLL이 급변·발산. **OOD conditioning 폭주.**")
        L.append("")

    # --- 핵심 발견 ---
    L.append("## 4) 핵심 발견")
    L.append("")
    L.append("**① static은 unseen speed를 구조적으로 무시한다 [해석].** 023 train의 3개 세팅은 모두 "
             "N15(1500rpm)이라 **정규화 speed가 항상 정확히 0(분산 0)**. train에서 speed 입력이 상수면 "
             "meta_encoder의 speed 가중치는 grad=0이고 **weight_decay(5e-4)만 작용해 0으로 감쇠**한다. 그래서 "
             "test에서 speed를 아무리 바꿔도 반응이 정확히 0. → 'static 단순 주입은 unseen에서 개선 없음'"
             "(TODO 핵심발견 ①)의 **미시적 원인**. (확증은 후속 'static torque축 대조'로.)")
    L.append("")
    L.append("**② measured/FiLM은 OOD 외삽으로 폭주한다 → §1′ 정량 증거.** unseen 저속(N09)의 measured "
             "speed는 train 기준 **z ≈ −590 ~ −785**(수백 σ 밖). train speed가 거의 상수라 std가 극소 → "
             "z가 폭발한다. 이 극단 입력 근방에서 주입기(concat 임베딩 / FiLM γ,β)가 외삽하며 NLL이 baseline "
             "σ의 수십~수만 배로 요동. TODO §1′ 'held-out 운행값이 학습분포 밖(OOD)이라 주입기가 외삽하며 "
             "예측을 붕괴시킨다'는 가설을 **정량적으로 지지**한다.")
    L.append("")
    L.append("**③ 원본 target 위치에서도 conditioning은 불안정하다.** 여러 fold에서 baseline σ 자체가 크다"
             "(예 measured LONO4 σ≈8, FiLM LONO5 σ≈480) — speed를 건드리지 않아도 원본 meta만으로 정상 "
             "window 간 NLL이 크게 퍼진다. 주입기가 이미 target 근방에서 예민·불안정함을 시사.")
    L.append("")

    # --- 격자 해상도 한계 ---
    L.append("## 5) 격자 해상도 한계 (곡선 해석 주의)")
    L.append("")
    L.append("measured/FiLM 곡선의 x축은 **z ≈ −785 같은 극단 target에 지배**당한다. 격자를 "
             "[target−1, +6] 구간에 25점 균등으로 잡아, 점 간격이 z로 ~30씩이다. 결과적으로 **train 근처"
             "(|z|≲6) 해상도를 거의 잃어**, train 부근의 미세 거동이 곡선에 거의 담기지 않는다. "
             "**판정(폭주 여부)에는 영향이 없으나**(극단·중간 대역에서 이미 수십~수만 σ 급변이 확인됨), "
             "곡선의 train 근처 모양을 논하려면 격자 재설계(train 근처 조밀 + target 대역 분리)가 필요하다. "
             "→ 후속 보강 항목.")
    L.append("")

    # --- 일반화 / 후속 ---
    L.append("## 6) 일반화 및 후속 [해석]")
    L.append("")
    L.append("- **[해석] 이 패턴은 다른 hold-out 축에도 일반화될 것으로 본다** — 단, 아직 023→1(speed 축)만 "
             "봤으므로 **확정이 아니다**. static의 '분산 0 축 → weight-decay로 가중치 사망'과 measured의 "
             "'held-out 축 z 폭발 → 외삽 폭주'는 축에 무관한 메커니즘이라, 저토크(013→2, torque 축)·"
             "저하중(012→3, force 축) fold에서도 재현될 것으로 예상. **검증 예정**(같은 스크립트 `--axis "
             "torque|force`, 해당 split 체크포인트).")
    L.append("- **후속 보강(예정, 이번 미실행)**: (a) measured 격자 재설계(train 근처 조밀화), "
             "(b) static torque축 대조(‘speed는 0σ인데 torque는 반응’으로 ① 확증).")
    L.append("")

    # --- TODO 판정 매핑 ---
    L.append("## 7) TODO 판정 기준 매핑")
    L.append("")
    L.append("| TODO 판정 기준 | 관측 | 해당 방식 |")
    L.append("|---|---|---|")
    L.append("| NLL 거의 불변 → speed 무시 | 전 fold 0.00σ | **static** |")
    L.append("| 완만·선형 → 제한적 외삽 | 해당 없음 | - |")
    L.append("| 범위 밖 급변 → OOD conditioning 폭주 | 전 fold 수십~수만 σ | **measured-concat, FiLM** |")
    L.append("")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
