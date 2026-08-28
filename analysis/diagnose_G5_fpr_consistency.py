"""작업 G-5 보조 — Fisher-tail(B) vs equal-z(A) 정상 FPR 악화의 일관성 진단.

핵심 질문: B가 A보다 정상 FPR이 높은(악화) 현상이 seed/fold 전반에 **일관**되는가,
아니면 특정 seed·fold·진폭군에 몰려서 평균만 끌어올린 것인가.

원천: results/Paderborn/diag_G5_tail_fusion/<split>_LONO<n>_s<seed>.json 의
  blocks.fusion_A / blocks.fusion_B 의 high_amp_fpr / low_amp_fpr (fold별, val 95pct threshold 기준).

집계(진폭군 high/low 각각):
  - 전체 120셀(24 fold × 5 seed) 중 B>A(악화) 비율, mean/median Δ(=B−A)
  - seed별 평균 Δ (seed 일관성)
  - fold별 평균 Δ (fold 편중 여부, Δ 큰 순)
  - fold유형(zero-support/compositional)별 평균 Δ
출력: stdout (+ --md 로 reports/report_G5_fpr_consistency.md 저장 가능)
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_DIR = os.path.join(PROJECT_ROOT, "results", "Paderborn", "diag_G5_tail_fusion")
MD_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G5_fpr_consistency.md")


def load_folds(diag_dir, seeds):
    folds = []
    for p in sorted(glob.glob(os.path.join(diag_dir, "*_LONO*_s*.json"))):
        m = re.search(r"(\w+)_LONO(\d+)_s(\d+)\.json$", os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(3))
        if seeds and seed not in seeds:
            continue
        with open(p) as f:
            d = json.load(f)
        folds.append(d)
    return folds


def cell_rows(folds, grp_key):
    """(B−A) FPR 셀 목록. grp_key ∈ {high_amp_fpr, low_amp_fpr}."""
    rows = []
    for d in folds:
        a = d["blocks"]["fusion_A"][grp_key]
        b = d["blocks"]["fusion_B"][grp_key]
        if a != a or b != b:  # nan 방어
            continue
        rows.append({
            "split": d["split"], "lono": d["lono"], "seed": d["seed"],
            "fold_type": d["fold_type"],
            "fold": f"{d['split']}_LONO{d['lono']}",
            "a": float(a), "b": float(b), "delta": float(b) - float(a),
        })
    return rows


def _fmt(m, s=None, p=3):
    return f"{m:.{p}f}" + (f"±{s:.{p}f}" if s is not None else "")


def summarize(rows, label, L):
    if not rows:
        L.append(f"### {label}: 데이터 없음")
        return
    d = np.array([r["delta"] for r in rows])
    n = len(d)
    worse = int((d > 0).sum())
    L.append(f"### {label} (셀 {n}개 = fold×seed)")
    L.append("")
    L.append(f"- **B가 A보다 악화(Δ>0) 비율: {worse}/{n} = {worse/n:.0%}** "
             f"(개선 Δ<0: {(d<0).sum()}, 동일: {(d==0).sum()})")
    L.append(f"- Δ(B−A) mean {_fmt(d.mean())} / median {_fmt(np.median(d))} / "
             f"std {_fmt(d.std())} / [min {_fmt(d.min())}, max {_fmt(d.max())}]")
    L.append(f"- A mean {_fmt(np.mean([r['a'] for r in rows]))} → B mean {_fmt(np.mean([r['b'] for r in rows]))}")
    L.append("")

    # seed별 평균 Δ
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r["delta"])
    seed_line = " / ".join(f"s{s}={_fmt(np.mean(v))}(악화 {int(np.sum(np.array(v)>0))}/{len(v)})"
                           for s, v in sorted(by_seed.items()))
    L.append(f"- **seed별 평균 Δ**: {seed_line}")

    # fold유형별
    by_ft = defaultdict(list)
    for r in rows:
        by_ft[r["fold_type"]].append(r["delta"])
    ft_line = " / ".join(f"{ft}={_fmt(np.mean(v))}(악화 {int(np.sum(np.array(v)>0))}/{len(v)})"
                         for ft, v in sorted(by_ft.items()))
    L.append(f"- **fold유형별 평균 Δ**: {ft_line}")
    L.append("")

    # fold별 평균 Δ (seed 평균) — Δ 큰(악화) 순 상위, 작은(개선) 순 하위
    by_fold = defaultdict(list)
    fold_type = {}
    for r in rows:
        by_fold[r["fold"]].append(r["delta"])
        fold_type[r["fold"]] = r["fold_type"]
    fold_mean = sorted(((f, np.mean(v), len(v)) for f, v in by_fold.items()),
                       key=lambda x: -x[1])
    L.append(f"- **fold별 평균 Δ (악화 상위 5 / 개선 상위 5, seed 평균)**:")
    for f, mv, k in fold_mean[:5]:
        L.append(f"    - ↑악화 {f} ({fold_type[f]}): Δ={_fmt(mv)} (n={k})")
    for f, mv, k in fold_mean[-5:][::-1]:
        L.append(f"    - ↓개선 {f} ({fold_type[f]}): Δ={_fmt(mv)} (n={k})")
    L.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="비우면 폴더 내 전 seed")
    ap.add_argument("--diag_dir", type=str, default=DIAG_DIR)
    ap.add_argument("--md", action="store_true", help="reports/report_G5_fpr_consistency.md 저장")
    args = ap.parse_args()

    folds = load_folds(args.diag_dir, set(args.seeds) if args.seeds else None)
    if not folds:
        raise SystemExit(f"diag_G5 per-fold JSON 없음: {args.diag_dir}")
    seeds = sorted({d["seed"] for d in folds})

    L = []
    L.append("# 작업 G-5 보조 — Fisher-tail(B) 정상 FPR 악화 일관성 진단")
    L.append("")
    L.append(f"seed {seeds}, fold {len({(d['split'],d['lono']) for d in folds})}종 × seed = {len(folds)} 셀. "
             "Δ=FPR(B)−FPR(A), Δ>0이면 B가 정상을 더 많이 오탐(악화). threshold=val-normal 95pct.")
    L.append("")
    summarize(cell_rows(folds, "high_amp_fpr"), "고진폭 정상 FPR (K001/K003/K006)", L)
    summarize(cell_rows(folds, "low_amp_fpr"), "저진폭 정상 FPR (K002/K004/K005)", L)
    L.append("> 해석 가이드: 악화 비율이 ~50%에 가깝고 Δ가 작으면 '일관되지만 경미', "
             "특정 seed/fold에 몰리면 '편중'. 저진폭 편중이면 val↔test 진폭 shift(G-3a 잔존 약점)와 동일 축.")

    text = "\n".join(L) + "\n"
    print(text)
    if args.md:
        with open(MD_PATH, "w") as f:
            f.write(text)
        print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    main()
