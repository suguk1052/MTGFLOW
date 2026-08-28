"""작업 G-5 Step 3 — fusion 규칙 비교 리포트 (equal-z A vs Fisher-tail B).

diagnose_G5_tail_fusion.py 가 seed별로 남긴 aggregate_G5_s<seed>.json 을 읽어 mean±std로 재집계한다.
1-seed(스크리닝)에서도 동작하며 std는 0(단일 seed 표기). 학습·추론 없음(순수 취합).

판정 축(TODO G-5): primary = 전체 AUROC + amp/shape-sensitive **동시보존**,
  secondary = zero-support/compositional 분리 · 정상 고/저진폭 FPR · ρ(RMS,score).
현 S_total(=fusion_A) 0.762±0.058(5-seed) 대비 B(Fisher-tail)가 명확한 순개선인지 확인.

산출: reports/report_G5_tail_fusion_5seeds.md
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_DIR = os.path.join(PROJECT_ROOT, "results", "Paderborn", "diag_G5_tail_fusion")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G5_tail_fusion_5seeds.md")

METHODS = ["raw", "shape", "amp", "fusion_A", "fusion_B"]
LABEL = {"raw": "raw", "shape": "S_shape", "amp": "S_amp",
         "fusion_A": "A: equal-z", "fusion_B": "B: Fisher-tail"}
AMP_SENSITIVE = {"KA04", "KA16", "KA30", "KB23", "KB24", "KI04", "KI16", "KI18"}
SHAPE_SENSITIVE = {"KA15", "KA22", "KB27", "KI14", "KI17", "KI21"}


def ms(vals):
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and x == x]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def fmt(m, s, p=3):
    return "nan" if m != m else f"{m:.{p}f}±{s:.{p}f}"


def gpath(a, *keys):
    for k in keys:
        a = a.get(k) if isinstance(a, dict) else None
        if a is None:
            return None
    return a


def row_methods(aggs, path_fn):
    """method별 (seed 리스트 → mean±std) 반환. path_fn(agg, method) → 값."""
    return {m: fmt(*ms([path_fn(a, m) for a in aggs])) for m in METHODS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--diag_dir", type=str, default=DIAG_DIR)
    args = ap.parse_args()

    aggs, used, missing = [], [], []
    for s in args.seeds:
        p = os.path.join(args.diag_dir, f"aggregate_G5_s{s}.json")
        if not os.path.exists(p):
            missing.append(s)
            continue
        with open(p) as f:
            aggs.append(json.load(f))
        used.append(s)
    if not aggs:
        # seeds 기본값에 없더라도 폴더에 있는 aggregate 전부 사용
        for p in sorted(glob.glob(os.path.join(args.diag_dir, "aggregate_G5_s*.json"))):
            with open(p) as f:
                aggs.append(json.load(f))
            used.append(int(re.search(r"_s(\d+)\.json", p).group(1)))
    if not aggs:
        raise SystemExit(f"aggregate_G5_s*.json 없음: {args.diag_dir} (먼저 diagnose_G5 실행)")

    n_folds = ms([a.get("n_folds") for a in aggs])[0]
    max_repro = max([a.get("reproduce_total_max_diff") or 0.0 for a in aggs])

    overall = row_methods(aggs, lambda a, m: gpath(a, "overall", m))
    ztype = row_methods(aggs, lambda a, m: gpath(a, "by_fold_type", "zero-support", "auroc", m))
    ctype = row_methods(aggs, lambda a, m: gpath(a, "by_fold_type", "compositional", "auroc", m))
    amp_grp = row_methods(aggs, lambda a, m: gpath(a, "fault_group", "amp_sensitive", m))
    shp_grp = row_methods(aggs, lambda a, m: gpath(a, "fault_group", "shape_sensitive", m))
    fpr_hi = row_methods(aggs, lambda a, m: gpath(a, "normal_fpr", m, "high"))
    fpr_lo = row_methods(aggs, lambda a, m: gpath(a, "normal_fpr", m, "low"))
    rho = row_methods(aggs, lambda a, m: gpath(a, "rho", m))

    fault_ids = sorted({fid for a in aggs for fid in a.get("per_fault", {})})

    L = []
    L.append("# 작업 G-5 — Branch Score Fusion 비교 (equal-z A vs Fisher-tail B)")
    L.append("")
    tag = "스크리닝" if len(used) == 1 else f"{len(used)}-seed"
    L.append(f"seed {used} ({tag}) mean±std. analysis-only — G-3a checkpoint per-window score 캐시(.npz) 재사용, "
             "학습·재추론 없음. 두 fusion을 동일 지표(_score_block)로 평가. "
             "A=equal-z(현행 S_total 재현) / B=Fisher-tail(val-normal 상단 tail 확률 -2[ln p_shape+ln p_amp]). "
             f"raw=B3 flow_NLL. 각 seed {n_folds:.0f} fold(4 split × 6 LONO). test 라벨 튜닝 없음.")
    if missing:
        L.append("")
        L.append(f"> ⚠️ 누락 seed(집계 제외): {missing}")
    L.append("")
    L.append(f"> **sanity**: fusion_A 가 기존 diag_G3a `auroc_total` 을 재현하는지 — 전 seed 최대 |Δ| = "
             f"{max_repro:.2e} " + ("(<1e-6 OK, 캐시·재구현 정합)." if max_repro < 1e-6 else "(⚠️ >1e-6, 재현 실패)."))
    L.append("")

    def mrow(name, d):
        return f"| {name} | " + " | ".join(d[m] for m in METHODS) + " |"

    hdr = "| 지표 | " + " | ".join(LABEL[m] for m in METHODS) + " |"
    sep = "|---|" + "---|" * len(METHODS)

    L.append("## 1) 핵심 비교 (primary) — 전체 AUROC + fault군 동시보존")
    L.append("")
    L.append(hdr)
    L.append(sep)
    L.append(mrow("전체 AUROC", overall))
    L.append(mrow("amp-sensitive fault", amp_grp))
    L.append(mrow("shape-sensitive fault", shp_grp))
    L.append("")
    L.append("> 동시보존 = A/B 각각에서 amp-sensitive·shape-sensitive가 함께 높아야 트레이드오프 극복. "
             "B가 A보다 전체 AUROC↑ + 두 fault군 모두 비열세여야 GO 후보.")
    L.append("")

    L.append("## 2) fold 유형 분리 (secondary)")
    L.append("")
    L.append(hdr)
    L.append(sep)
    L.append(mrow("zero-support 외삽", ztype))
    L.append(mrow("compositional", ctype))
    L.append("")

    L.append("## 3) 정상 FPR(진폭군) · ρ(RMS,score) (secondary)")
    L.append("")
    L.append(hdr)
    L.append(sep)
    L.append(mrow("정상 FPR 고진폭", fpr_hi))
    L.append(mrow("정상 FPR 저진폭", fpr_lo))
    L.append(mrow("ρ(RMS,score)", rho))
    L.append("")
    L.append("> 고진폭=K001/K003/K006, 저진폭=K002/K004/K005. FPR threshold=val-normal score 95pct. "
             "ρ↓(0 근처)이면 진폭 confound 제거 유지.")
    L.append("")

    L.append("## 4) per-fault AUROC")
    L.append("")
    L.append("| fault id | 분류 | " + " | ".join(LABEL[m] for m in METHODS) + " |")
    L.append("|---|---|" + "---|" * len(METHODS))
    for fid in fault_ids:
        cls = "A" if fid in AMP_SENSITIVE else "S" if fid in SHAPE_SENSITIVE else ""
        cells = {m: fmt(*ms([gpath(a, "per_fault", fid, m) for a in aggs])) for m in METHODS}
        L.append(f"| {fid} | {cls} | " + " | ".join(cells[m] for m in METHODS) + " |")
    L.append("")
    L.append("> A=amplitude-sensitive, S=shape-sensitive. 판정은 다지표를 함께 — cherry-pick 금지. "
             "1-seed 스크리닝은 방향성만, 확정은 5-seed에서.")
    L.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT_PATH}")
    print(f"seeds={used}  n_folds≈{n_folds:.0f}  repro_max|Δ|={max_repro:.2e}")
    print("overall: " + "  ".join(f"{LABEL[m]}={overall[m]}" for m in METHODS))


if __name__ == "__main__":
    main()
