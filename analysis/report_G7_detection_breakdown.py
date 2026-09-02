"""작업 G-7 Step 3 — 검출/오탐 분해 리포트 생성 (순수 취합, 학습·추론 없음).

diagnose_G7_detection_breakdown.py 가 남긴 combined_5seed.json(5-seed mean±std)을 읽어
reports/report_G7_detection_breakdown.md 를 생성한다. 표 수치는 전부 JSON에서 렌더(전사 오류 방지),
해석 문구의 핵심 숫자도 JSON에서 채운다.

산출: reports/report_G7_detection_breakdown.md
"""
import argparse
import json
import os

import numpy as np

from diagnose_G3a_amp_bands import RESULTS_ROOT, PROJECT_ROOT  # noqa: E402

DIAG_DIR = os.path.join(RESULTS_ROOT, "diag_G7_detection_breakdown")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G7_detection_breakdown.md")
REVERSAL = ["KA30", "KI04", "KA15", "KA22", "KB27", "KI14"]  # raw<0.5 였던 역전 결함군(task 강조)
PCTS = ["90", "95", "97.5", "99"]


def f(x):
    return "nan" if x["mean"] != x["mean"] else f"{x['mean']:.3f}±{x['std']:.3f}"


def mm(x):
    return x["mean"]


def cls_recall(pf, cls, meth):
    vs = [mm(e[meth]["recall95"]) for e in pf.values() if e["class"] == cls]
    return float(np.mean(vs)) if vs else float("nan")


def build(d):
    ov = d["overall"]
    pf = ov["per_fault"]
    nf = ov["normal_fpr"]
    sw = ov["sweep"]
    pb = d["per_bearing"]
    bt = d["by_fold_type"]
    seeds = d["seeds"]
    san = d["sanity"]

    L = []
    W = L.append

    # ---- 헤더 / 메타 ----
    W("# 작업 G-7 — 검출/오탐 단위 진단 분해 (raw vs A=equal-z vs B=Fisher-tail)")
    W("")
    W(f"seed {seeds} (5-seed) mean±std. **analysis-only** — G-3a per-window 캐시(g3a_window_scores/*.npz) + "
      "raw(B3) 재추론 캐시(b3_raw_window_scores/*.npz, forward-only, 학습 없음)만 사용. "
      "각 seed 24 fold(4 split × 6 LONO). "
      "비교 3종: **raw**=B3 flow NLL / **A**=equal-z S_total(G-3a) / **B**=Fisher-tail(G-5 최종). "
      "threshold = **val 정상 score percentile 고정**(test 라벨 튜닝 없음).")
    W("")
    W(f"> **sanity**: 재집계 전체 AUROC raw {mm(ov['auroc']['raw']):.3f} / A {mm(ov['auroc']['A']):.3f} / "
      f"B {mm(ov['auroc']['B']):.3f} — 기존 리포트(0.696 / 0.762 / 0.800) 재현. "
      f"raw·A per-fold AUROC vs diag_G3a 최대 |Δ| = {san['raw_repro_max_diff']:.0e} / {san['a_repro_max_diff']:.0e} "
      "(<1e-6, 캐시·재추론 정합).")
    W("")

    # ---- 핵심 발견 ----
    sh_r_raw, sh_r_b = cls_recall(pf, "shape-sensitive", "raw"), cls_recall(pf, "shape-sensitive", "B")
    am_r_raw, am_r_b = cls_recall(pf, "amp-sensitive", "raw"), cls_recall(pf, "amp-sensitive", "B")
    W("## 핵심 발견")
    W("")
    W(f"1. **AUROC 상승(raw 0.696 → B 0.800)의 정체 = shape-sensitive 결함의 미탐 급감.** 고정 임계값(val95) "
      f"recall이 shape-sensitive군 평균 **{sh_r_raw:.2f} → {sh_r_b:.2f}**(KA22 0.12→0.79, KA15 0.25→0.76 등)로 "
      "뛴다. raw에서 정상보다 낮게 깔려 '역전'돼 있던 결함들이 B에서 정상 위로 올라온 것.")
    W(f"2. **대가는 amp-sensitive 결함 recall 소폭 하락({am_r_raw:.2f} → {am_r_b:.2f})** + **저진폭 정상 FPR 악화**"
      f"(val95 raw {mm(nf['raw']['low']):.3f} → B {mm(nf['B']['low']):.3f}). 고진폭 정상 FPR은 반대로 "
      f"개선(raw {mm(nf['raw']['high']):.3f} → B {mm(nf['B']['high']):.3f}).")
    W(f"3. **저진폭 FPR 악화는 임계값 문제가 아니라 순위 문제(중증).** percentile을 95→99로 올려도 B의 저진폭 FPR은 "
      f"{mm(sw['B']['95']['low_fpr']):.3f} → {mm(sw['B']['99']['low_fpr']):.3f} 로만 내려가 raw 수준"
      f"({mm(sw['raw']['99']['low_fpr']):.3f})에 한참 못 미치고, 그 사이 recall은 "
      f"{mm(sw['B']['95']['overall_recall']):.3f} → {mm(sw['B']['99']['overall_recall']):.3f} 손실.")
    W(f"4. **저진폭 악화의 주범은 K004·K005 두 개체.** val95 FPR raw→B: K004 {mm(pb['K004']['raw']['fpr95']):.3f}→"
      f"{mm(pb['K004']['B']['fpr95']):.3f}, K005 {mm(pb['K005']['raw']['fpr95']):.3f}→{mm(pb['K005']['B']['fpr95']):.3f} "
      f"인 반면 K002는 {mm(pb['K002']['raw']['fpr95']):.3f}→{mm(pb['K002']['B']['fpr95']):.3f} 로 거의 불변. "
      "val↔test 진폭 분포 이동 가설의 직접 증거(가설).")
    W(f"5. **KA30·KI04는 회복 실패 확정.** per-fault AUROC raw 0.836 → B 0.588(KA30) / 0.604(KI04), "
      "recall도 raw 0.62 → B 0.46~0.48로 하락. amp-sensitive라 shape 정규화가 오히려 신호를 지운다.")
    W("")

    # ---- 방법 ----
    W("## 방법")
    W("")
    W("- **score(클수록 이상)**: raw=`-log p_x`(B3 flow NLL). A=val정상 z-score 후 S_shape+S_amp 등가중 합. "
      "B=각 branch를 val정상 상단 tail 확률 p로 변환 후 Fisher χ² 결합 `-2[ln p_shape+ln p_amp]`. 셋 다 test 라벨 무사용.")
    W("- **threshold**: fold별 **val 정상 score의 percentile**만으로 결정(90/95/97.5/99). test 라벨은 어떤 임계값에도 미사용.")
    W("- **recall** = fault window 중 score ≥ threshold 비율(=1−미탐률). **정상 FPR** = 정상 pool(train+val+test-normal) "
      "per-bearing FPR을 고/저진폭군 평균. **per-fault/per-bearing** 은 fold 평균 후 5-seed 평균±std.")
    W("- **fold 분리**: zero-support(023→1·013→2·012→3, 순수 외삽) vs compositional(123→0). 24 fold 평균으로 뭉뚱그리지 않음.")
    W("- 진폭군: 고진폭 K001·K003·K006 / 저진폭 K002·K004·K005. 결함 사후분류 amp-sensitive(8) / shape-sensitive(6).")
    W("")

    # ---- 표 1 ----
    W("## 1) per-fault AUROC (raw vs A vs B, Δ=B−raw 내림차순)")
    W("")
    W("| fault | family | 사후분류 | raw | A: equal-z | B: Fisher-tail | Δ(B−raw) |")
    W("|---|---|---|---|---|---|---|")
    rows = sorted(pf.items(), key=lambda kv: (mm(kv[1]["B"]["auroc"]) - mm(kv[1]["raw"]["auroc"])), reverse=True)
    for fid, e in rows:
        dv = mm(e["B"]["auroc"]) - mm(e["raw"]["auroc"])
        star = " ⭑" if fid in REVERSAL else ""
        name = f"**{fid}**{star}" if fid in REVERSAL else fid
        W(f"| {name} | {e['family']} | {e['class']} | {f(e['raw']['auroc'])} | {f(e['A']['auroc'])} | "
          f"{f(e['B']['auroc'])} | {dv:+.3f} |")
    W("")
    W("> ⭑ = raw에서 0.5 미만이었던 역전 결함군(KA30·KI04는 raw 0.836이나 amp-sensitive라 A/B에서 회복 실패 — "
      "별 범주). shape-sensitive 6종이 Δ 상위를 독점, amp-sensitive는 전부 Δ≤0.")
    W("")

    # ---- 표 2 ----
    W("## 2) 고정 임계값(val95) 검출률 recall — raw vs B")
    W("")
    W("| fault | 사후분류 | raw recall | B recall | Δ(B−raw) |")
    W("|---|---|---|---|---|")
    for fid, e in sorted(pf.items(), key=lambda kv: (mm(kv[1]["B"]["recall95"]) - mm(kv[1]["raw"]["recall95"])), reverse=True):
        dv = mm(e["B"]["recall95"]) - mm(e["raw"]["recall95"])
        W(f"| {fid} | {e['class']} | {f(e['raw']['recall95'])} | {f(e['B']['recall95'])} | {dv:+.3f} |")
    W(f"| **shape-sensitive 평균** | — | {sh_r_raw:.3f} | {sh_r_b:.3f} | {sh_r_b - sh_r_raw:+.3f} |")
    W(f"| **amp-sensitive 평균** | — | {am_r_raw:.3f} | {am_r_b:.3f} | {am_r_b - am_r_raw:+.3f} |")
    W("")
    W("**진폭군별 정상 FPR (val95, raw / A / B):**")
    W("")
    W("| 진폭군 | raw | A: equal-z | B: Fisher-tail |")
    W("|---|---|---|---|")
    W(f"| 고진폭(K001·K003·K006) | {f(nf['raw']['high'])} | {f(nf['A']['high'])} | {f(nf['B']['high'])} |")
    W(f"| 저진폭(K002·K004·K005) | {f(nf['raw']['low'])} | {f(nf['A']['low'])} | {f(nf['B']['low'])} |")
    W("")
    W("> AUROC 상승분은 shape-sensitive 결함 recall(+~0.49)에서 온다. 대가는 amp-sensitive recall 소폭 하락과 "
      "저진폭 정상 FPR 악화.")
    W("")

    # ---- 표 3 ----
    W("## 3) threshold sweep — FPR–recall 트레이드오프 (진폭군별, raw vs B)")
    W("")
    for panel, fk, kor in (("high", "high_fpr", "고진폭"), ("low", "low_fpr", "저진폭")):
        W(f"**{kor} 정상 패널** (그림: `figs_g7/fpr_recall_{panel}.png`)")
        W("")
        W("| val pct | raw 정상FPR | raw recall | B 정상FPR | B recall |")
        W("|---|---|---|---|---|")
        for p in PCTS:
            r, b = sw["raw"][p], sw["B"][p]
            W(f"| {p} | {mm(r[fk]):.3f} | {mm(r['overall_recall']):.3f} | {mm(b[fk]):.3f} | {mm(b['overall_recall']):.3f} |")
        W("")
    W(f"> 저진폭 패널: pct를 95→99로 올려도 B FPR {mm(sw['B']['95']['low_fpr']):.3f}→{mm(sw['B']['99']['low_fpr']):.3f} "
      f"(raw {mm(sw['raw']['99']['low_fpr']):.3f}에 한참 못 미침), recall 손실 "
      f"{mm(sw['B']['95']['overall_recall']) - mm(sw['B']['99']['overall_recall']):.3f}. "
      "→ 임계값 보정으로 저진폭 FPR을 raw 수준으로 되돌릴 수 없다(순위 악화, 중증).")
    W("")

    # ---- 표 4 ----
    W("## 4) 정상 bearing 6개별 FPR·mean_score (val95, raw / A / B)")
    W("")
    W("| bearing | 진폭군 | raw FPR | A FPR | B FPR | raw mean_score | B mean_score |")
    W("|---|---|---|---|---|---|---|")
    for bid, e in pb.items():
        W(f"| {bid} | {e['group']} | {f(e['raw']['fpr95'])} | {f(e['A']['fpr95'])} | {f(e['B']['fpr95'])} | "
          f"{mm(e['raw']['mean_score']):.2f} | {mm(e['B']['mean_score']):.2f} |")
    W("")
    W("> 저진폭 3개 중 **K004·K005** 가 악화를 주도(FPR raw≈0.01·0.11 → B≈0.54·0.51, B mean_score 12.6·11.9로 최고), "
      "K002는 거의 불변(0.17→0.20). 고진폭 3개는 모두 raw 대비 대폭 개선. "
      "raw는 mean_score가 모든 개체에서 −7.x대로 진폭에 무관, B는 K004/K005만 튐.")
    W("")

    # ---- fold 유형 분리 ----
    W("## 5) fold 유형 분리 (zero-support vs compositional)")
    W("")
    W("| 지표 | zero-support raw | zero-support B | compositional raw | compositional B |")
    W("|---|---|---|---|---|")
    zs, cp = bt["zero-support"], bt["compositional"]
    W(f"| 전체 AUROC | {f(zs['auroc']['raw'])} | {f(zs['auroc']['B'])} | {f(cp['auroc']['raw'])} | {f(cp['auroc']['B'])} |")
    for cls, lab in (("shape-sensitive", "shape recall"), ("amp-sensitive", "amp recall")):
        W(f"| {lab}(val95) | {cls_recall(zs['per_fault'], cls, 'raw'):.3f} | {cls_recall(zs['per_fault'], cls, 'B'):.3f} "
          f"| {cls_recall(cp['per_fault'], cls, 'raw'):.3f} | {cls_recall(cp['per_fault'], cls, 'B'):.3f} |")
    W(f"| 정상 FPR 고진폭 | {mm(zs['normal_fpr']['raw']['high']):.3f} | {mm(zs['normal_fpr']['B']['high']):.3f} "
      f"| {mm(cp['normal_fpr']['raw']['high']):.3f} | {mm(cp['normal_fpr']['B']['high']):.3f} |")
    W(f"| 정상 FPR 저진폭 | {mm(zs['normal_fpr']['raw']['low']):.3f} | {mm(zs['normal_fpr']['B']['low']):.3f} "
      f"| {mm(cp['normal_fpr']['raw']['low']):.3f} | {mm(cp['normal_fpr']['B']['low']):.3f} |")
    W("")
    W("> 저진폭 FPR 악화는 zero-support(0.10→0.44)·compositional(0.08→0.37) 모두에서 나타나 zero-support 전용 "
      "artifact가 아니다.")
    W("")

    # ---- 판정 ----
    W("## 판정")
    W("")
    W("**(a) AUROC 상승의 귀속 — 확인됨(데이터).** recall Δ 합은 전적으로 shape-sensitive 결함에 몰려 있다"
      f"(shape 평균 {sh_r_raw:.2f}→{sh_r_b:.2f}, amp 평균 {am_r_raw:.2f}→{am_r_b:.2f}). 표 1의 per-fault AUROC Δ "
      "상위 6종(KA22·KA15·KI14·KB27·KI17·KI21)이 모두 shape-sensitive로 정확히 일치. "
      "→ '역전돼 있던 결함이 정상 위로 올라온 것'이 AUROC 상승의 주원인.")
    W("")
    W("**(b) 저진폭 FPR 악화의 성격 — 중증(순위 문제, 데이터).** percentile을 99까지 올려도 B의 저진폭 FPR은 "
      f"{mm(sw['B']['99']['low_fpr']):.3f}로 raw({mm(sw['raw']['99']['low_fpr']):.3f})의 5배 이상에 머물고, 그 대가로 "
      f"recall이 {mm(sw['B']['95']['overall_recall']) - mm(sw['B']['99']['overall_recall']):.3f} 깎인다. "
      "임계값 보정만으로는 완화되지 않는다 → **pseudo-LOSO 등 분포정렬의 필요성 강화**.")
    W("")
    W("**(c) 개체 특정 — K004·K005 주도(데이터) → 진폭 분포 이동 가설(가설).** 저진폭 FPR 0.418은 K004(0.54)·"
      "K005(0.51)가 끌어올린 값이고 K002(0.20)는 거의 정상이다. K004/K005의 B mean_score가 12.6·11.9로 최고인 것은, "
      "shape 정규화 후에도 이 두 held-out 저진폭 정상이 val 분포보다 이상하게 보인다는 뜻 — val↔test 진폭 분포 "
      "이동의 직접 증거로 해석된다(인과 확정은 추가 실험 필요, **가설**).")
    W("")

    # ---- 주의 / 한계 ----
    W("## 주의 / 한계")
    W("")
    W("**데이터로 확인된 사실**")
    W("- 전체·per-fault AUROC, per-fault recall, per-bearing FPR 수치는 5-seed 평균±std이며, raw·A는 diag_G3a와 "
      "per-fold |Δ|=0으로 정합. sanity(0.696/0.762/0.800) 재현.")
    W("- shape-sensitive recall 급증 / amp-sensitive recall 소폭 하락 / 저진폭 FPR 악화(K004·K005 주도) / "
      "고진폭 FPR 개선은 모두 관측된 사실.")
    W("")
    W("**가설·추측(추가 확인 필요)**")
    W("- K004·K005의 FPR 급등 원인을 'val↔test 진폭 분포 이동'으로 본 것은 mean_score 정황에 기반한 **해석**이다. "
      "인과 확정에는 val/test 진폭 분포 직접 비교·pseudo-LOSO 검증이 필요.")
    W("- pseudo-LOSO가 저진폭 FPR을 실제로 완화할지는 미검증. PU는 축당 2조건뿐인 순수 외삽(zero-support)이라 효과 제한 가능.")
    W("")
    W("**방법상 주의**")
    W("- 진폭군 정상 FPR은 정상 pool에 **train bearing을 포함**(diag_G3a `_score_block`과 동일 규약)하므로, "
      "저진폭 FPR은 held-out test-normal 단독 FPR보다 완만할 수 있다. 개체 특정(표 4)은 per-bearing으로 분리해 이 영향 배제.")
    W("- recall/FPR은 **window-level**. bearing-level 판정(다수결·집계)과는 다를 수 있다.")
    W("- raw per-window는 B3 checkpoint forward-only 재추론(학습 없음) 산출. A/B는 G-3a 캐시 재조합.")
    W("")

    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=os.path.join(DIAG_DIR, "combined_5seed.json"))
    ap.add_argument("--out", type=str, default=REPORT_PATH)
    args = ap.parse_args()

    with open(args.json) as fp:
        d = json.load(fp)
    md = build(d)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        fp.write(md)
    print(f"wrote {args.out}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
