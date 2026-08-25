"""작업 G-3a S_shape 심층 진단 — G-3b(auxiliary-task 구조 검증)의 전제조건.

G-3a의 최우선 지표 S_shape(0.776±0.047 > raw 0.696)를 **단독 해부**해 어디서 이득/붕괴하는지 확정한다.
학습·재추론 없음 — diagnose_G3a_amp_bands.py가 이미 남긴 per-fold JSON 120개만 재취합(analysis-only).

입력: results/Paderborn/diag_G3a_amp_bands/<split>_LONO<n>_s<seed>.json (120 = split4 × LONO6 × seed5).
  (aggregate_s*.json·summary_5seeds.json은 사용 안 함 — fold별 Δ·bearing별 threshold 붕괴는 per-fold 입도 필요.)
  각 fold의 g1.blocks.shape 블록과 raw.auroc만 추출한다.

산출: reports/report_G3a_shape_deepdive.md + results/Paderborn/diag_G3a_amp_bands/summary_shape_deepdive.json

분석 항목(TODO 116–120):
  1) 고/저진폭 정상 FPR(S_shape) — fold_type 분리 포함
  2) zero-support vs compositional 분리 S_shape AUROC + ρ(RMS,S_shape)
  3) fault군별(amp/shape-sensitive) S_shape AUROC + per-fault
  4) fold별 Δ(S_shape − raw) 분포 (24 fold × 5 seed)
  5) threshold 붕괴 지점 — per_bearing_normal의 bearing별 FPR·mean_score·threshold 간극
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_DIR = os.path.join(PROJECT_ROOT, "results", "Paderborn", "diag_G3a_amp_bands")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G3a_shape_deepdive.md")
SUMMARY_PATH = os.path.join(DIAG_DIR, "summary_shape_deepdive.json")

# report_G3a_5seeds.py:22–23와 동일한 fault 분류 (일관성 유지).
AMP_SENSITIVE = {"KA04", "KA16", "KA30", "KB23", "KB24", "KI04", "KI16", "KI18"}
SHAPE_SENSITIVE = {"KA15", "KA22", "KB27", "KI14", "KI17", "KI21"}

SPLITS = ("012to3", "013to2", "023to1", "123to0")
SEEDS = (2024, 2025, 2026, 2027, 2028)
BEARINGS = ("K001", "K002", "K003", "K004", "K005", "K006")
# per-fold 파일명 패턴: <split>_LONO<n>_s<seed>.json (aggregate_*/summary_* 제외)
FOLD_RE = re.compile(r"^(?P<split>\w+?)_LONO(?P<lono>\d+)_s(?P<seed>\d+)\.json$")


def ms(vals):
    """mean±std (None/nan 제외). 반환 (mean, std, n)."""
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and x == x]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v)), len(v)


def fmt_ms(m, s, p=3):
    if m != m:
        return "nan"
    return f"{m:.{p}f}±{s:.{p}f}"


def group_fault_mean(per_fault_auroc, ids):
    """per_fault_auroc{fault:{n,auroc}}에서 ids에 속한 fault들의 AUROC 평균(해당 fold 내)."""
    vals = [v.get("auroc") for k, v in per_fault_auroc.items()
            if k in ids and isinstance(v, dict) and v.get("auroc") is not None]
    return float(np.mean(vals)) if vals else None


def load_folds(diag_dir):
    """per-fold JSON 120개 로드. (records, missing) 반환."""
    records = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(diag_dir, "*_LONO*_s*.json"))):
        name = os.path.basename(path)
        if name.startswith("aggregate") or name.startswith("summary"):
            continue
        m = FOLD_RE.match(name)
        if not m:
            continue
        with open(path) as f:
            j = json.load(f)
        shape = j.get("g1", {}).get("blocks", {}).get("shape")
        if shape is None:
            continue
        rec = {
            "split": j.get("split"),
            "lono": j.get("lono"),
            "seed": j.get("seed"),
            "fold_type": j.get("fold_type"),
            "target_amp_group": j.get("target_amp_group"),
            "target_norm_ids": j.get("target_norm_ids", []),
            "val_ids": j.get("val_ids"),  # 있으면 사용, 없으면 per_bearing_normal에서 유추
            "shape": shape,
            "raw_auroc": j.get("raw", {}).get("auroc"),
        }
        records.append(rec)
        seen.add((rec["split"], rec["lono"], rec["seed"]))
    # 누락 점검
    missing = []
    for sp in SPLITS:
        for n in range(1, 7):
            for sd in SEEDS:
                if (sp, n, sd) not in seen:
                    missing.append(f"{sp}_LONO{n}_s{sd}")
    return records, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_dir", type=str, default=DIAG_DIR)
    args = ap.parse_args()

    records, missing = load_folds(args.diag_dir)
    n = len(records)
    print(f"loaded {n} per-fold JSON (기대 120)")
    if missing:
        print(f"⚠️ 누락 fold {len(missing)}개: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if not records:
        raise SystemExit(f"per-fold JSON 없음: {args.diag_dir} (먼저 diagnose_G3a_amp_bands.py 실행)")

    # ---- overall / fold_type 분리용 인덱싱 ----
    by_type = {"zero-support": [], "compositional": []}
    for r in records:
        by_type.get(r["fold_type"], by_type.setdefault(r["fold_type"], [])).append(r)

    # ===== 1) 고/저진폭 정상 FPR(S_shape) =====
    def fpr_block(recs):
        return {
            "high_fpr": ms([r["shape"].get("high_amp_fpr") for r in recs]),
            "low_fpr": ms([r["shape"].get("low_amp_fpr") for r in recs]),
            "high_mean": ms([r["shape"].get("high_amp_mean_score") for r in recs]),
            "low_mean": ms([r["shape"].get("low_amp_mean_score") for r in recs]),
        }
    fpr_overall = fpr_block(records)
    fpr_by_type = {ft: fpr_block(recs) for ft, recs in by_type.items() if recs}

    # ===== 2) fold_type별 S_shape AUROC + ρ(RMS,S_shape) =====
    auroc_overall = ms([r["shape"].get("auroc") for r in records])
    rho_overall = ms([r["shape"].get("rho_rms_score") for r in records])
    auroc_by_type, rho_by_type = {}, {}
    for ft, recs in by_type.items():
        if not recs:
            continue
        auroc_by_type[ft] = ms([r["shape"].get("auroc") for r in recs])
        rho_by_type[ft] = ms([r["shape"].get("rho_rms_score") for r in recs])

    # ===== 3) fault군별 S_shape AUROC + per-fault =====
    fault_group = {}
    for grp, ids in (("amp_sensitive", AMP_SENSITIVE), ("shape_sensitive", SHAPE_SENSITIVE)):
        fault_group[grp] = ms([group_fault_mean(r["shape"].get("per_fault_auroc", {}), ids)
                               for r in records])
    fault_ids = sorted({fid for r in records for fid in r["shape"].get("per_fault_auroc", {})})
    per_fault = {}
    for fid in fault_ids:
        per_fault[fid] = ms([r["shape"].get("per_fault_auroc", {}).get(fid, {}).get("auroc")
                             for r in records])

    # ===== 4) fold별 Δ(S_shape − raw) — (split,lono) 24 fold × 5 seed =====
    fold_delta = {}  # (split,lono) -> {"delta": ms, "shape": ms, "raw": ms, "fold_type", "target_amp_group"}
    for sp in SPLITS:
        for lono in range(1, 7):
            recs = [r for r in records if r["split"] == sp and r["lono"] == lono]
            if not recs:
                continue
            deltas = [r["shape"].get("auroc") - r["raw_auroc"]
                      for r in recs
                      if r["shape"].get("auroc") is not None and r["raw_auroc"] is not None]
            fold_delta[f"{sp}_LONO{lono}"] = {
                "delta": ms(deltas),
                "shape": ms([r["shape"].get("auroc") for r in recs]),
                "raw": ms([r["raw_auroc"] for r in recs]),
                "fold_type": recs[0]["fold_type"],
                "target_amp_group": recs[0]["target_amp_group"],
            }

    # ===== 5) threshold 붕괴 지점 — bearing별 FPR·mean_score(S_shape) =====
    # threshold는 val 정상 p95로 고정되므로, "붕괴"는 held-out **test 정상**일 때만 정직하게 드러난다.
    # (train bearing은 학습에서 봐서 FPR 낮고, val bearing은 정의상 FPR≈0.05로 고정 → 섞으면 신호 희석.)
    # 따라서 bearing별로 그 bearing이 test 정상으로 등장한 fold(각 20 = 4 split × 5 seed)만 집계한다.
    # gap = mean_score − threshold_val: >0이면 정상 window 평균 score가 threshold보다 이상쪽 = 붕괴.
    bearing_acc = {b: {"fpr": [], "mean_score": [], "mean_rms": [], "gap": [],
                       "group": None, "n_test": 0} for b in BEARINGS}
    for r in records:
        pbn = r["shape"].get("per_bearing_normal", {})
        thr = r["shape"].get("threshold_val")
        for b in (r.get("target_norm_ids") or []):
            d = pbn.get(b)
            if d is None:
                continue
            acc = bearing_acc.setdefault(b, {"fpr": [], "mean_score": [], "mean_rms": [],
                                             "gap": [], "group": None, "n_test": 0})
            acc["group"] = d.get("group")
            acc["n_test"] += 1
            if d.get("fpr") is not None:
                acc["fpr"].append(d["fpr"])
            if d.get("mean_score") is not None:
                acc["mean_score"].append(d["mean_score"])
                if thr is not None:
                    acc["gap"].append(d["mean_score"] - thr)
            if d.get("mean_rms") is not None:
                acc["mean_rms"].append(d["mean_rms"])
    bearing_summary = {}
    for b, acc in bearing_acc.items():
        if acc["n_test"] == 0:
            continue
        bearing_summary[b] = {
            "group": acc["group"],
            "n_test_folds": acc["n_test"],
            "fpr": ms(acc["fpr"]),
            "mean_score": ms(acc["mean_score"]),
            "mean_rms": ms(acc["mean_rms"]),
            "gap_score_minus_thr": ms(acc["gap"]),
        }

    # ---- summary JSON ----
    summary = {
        "n_folds": n, "missing": missing, "seeds": list(SEEDS),
        "auroc_overall": auroc_overall, "rho_overall": rho_overall,
        "auroc_by_type": auroc_by_type, "rho_by_type": rho_by_type,
        "fpr_overall": fpr_overall, "fpr_by_type": fpr_by_type,
        "fault_group": fault_group, "per_fault": per_fault,
        "fold_delta": fold_delta, "bearing_summary": bearing_summary,
    }
    os.makedirs(args.diag_dir, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 리포트 ----
    L = []
    L.append("# 작업 G-3a S_shape 심층 진단 (G-3b 전제조건)")
    L.append("")
    L.append(f"seed {list(SEEDS)} × 24 fold = **{n} per-fold** 재취합 (학습·재추론 없음, analysis-only). "
             "G-3a 최우선 지표 **S_shape**(shape branch flow_NLL AUROC)만 단독 해부. "
             "원천 = `diag_G3a_amp_bands/<split>_LONO<n>_s<seed>.json`의 `g1.blocks.shape` + `raw.auroc`(paired B3). "
             "가드레일 no-meta LOSO raw=0.696.")
    if missing:
        L.append("")
        L.append(f"> ⚠️ 누락 fold {len(missing)}개(집계 제외): {', '.join(missing[:12])}"
                 f"{' ...' if len(missing) > 12 else ''}")
    L.append("")

    # 핵심 요약
    zt = auroc_by_type.get("zero-support", (float('nan'),) * 3)
    ct = auroc_by_type.get("compositional", (float('nan'),) * 3)
    L.append("## 핵심 요약 (S_shape, 5-seed × 24 fold mean±std)")
    L.append("")
    L.append(f"- **전체 S_shape AUROC**: **{fmt_ms(*auroc_overall[:2])}** (raw 0.696 대비). "
             f"ρ(RMS,S_shape) {fmt_ms(*rho_overall[:2])} (raw≈0.96 → 진폭 confound 제거).")
    L.append(f"- **fold 유형 분리**: zero-support 외삽 {fmt_ms(*zt[:2])} / compositional {fmt_ms(*ct[:2])}.")
    L.append(f"- **정상 FPR(S_shape)**: 고진폭 {fmt_ms(*fpr_overall['high_fpr'][:2])} / "
             f"저진폭 {fmt_ms(*fpr_overall['low_fpr'][:2])} "
             "— 저진폭>고진폭이면 저진폭 정상 threshold 붕괴(val↔test 진폭분포 shift).")
    L.append(f"- **fault군별**: amp-sensitive {fmt_ms(*fault_group['amp_sensitive'][:2])} / "
             f"shape-sensitive {fmt_ms(*fault_group['shape_sensitive'][:2])}.")
    L.append("")

    # 1) 정상 FPR
    L.append("## 1) 정상 FPR(S_shape) — 진폭군별 · fold유형별")
    L.append("")
    L.append("| 구간 | 고진폭 FPR | 저진폭 FPR | 고진폭 mean(S_shape) | 저진폭 mean |")
    L.append("|---|---|---|---|---|")
    L.append(f"| 전체 | {fmt_ms(*fpr_overall['high_fpr'][:2])} | {fmt_ms(*fpr_overall['low_fpr'][:2])} | "
             f"{fmt_ms(*fpr_overall['high_mean'][:2])} | {fmt_ms(*fpr_overall['low_mean'][:2])} |")
    for ft in ("zero-support", "compositional"):
        if ft in fpr_by_type:
            b = fpr_by_type[ft]
            L.append(f"| {ft} | {fmt_ms(*b['high_fpr'][:2])} | {fmt_ms(*b['low_fpr'][:2])} | "
                     f"{fmt_ms(*b['high_mean'][:2])} | {fmt_ms(*b['low_mean'][:2])} |")
    L.append("")
    L.append("> 고진폭=K001/K003/K006, 저진폭=K002/K004/K005. FPR은 해당 진폭군 정상 window 풀 기준.")
    L.append("")

    # 2) fold유형별 AUROC + rho
    L.append("## 2) fold 유형별 S_shape AUROC + ρ(RMS,S_shape)")
    L.append("")
    L.append("| fold 유형 | S_shape AUROC | ρ(RMS,S_shape) |")
    L.append("|---|---|---|")
    L.append(f"| 전체 | **{fmt_ms(*auroc_overall[:2])}** | {fmt_ms(*rho_overall[:2])} |")
    for ft in ("zero-support", "compositional"):
        if ft in auroc_by_type:
            L.append(f"| {ft} | {fmt_ms(*auroc_by_type[ft][:2])} | {fmt_ms(*rho_by_type[ft][:2])} |")
    L.append("")

    # 3) fault군별 + per-fault
    L.append("## 3) fault군별 S_shape AUROC")
    L.append("")
    L.append("| fault군 | S_shape AUROC |")
    L.append("|---|---|")
    L.append(f"| amp-sensitive | {fmt_ms(*fault_group['amp_sensitive'][:2])} |")
    L.append(f"| shape-sensitive | {fmt_ms(*fault_group['shape_sensitive'][:2])} |")
    L.append("")
    L.append("### per-fault S_shape AUROC (진단용)")
    L.append("")
    L.append("| fault id | 분류 | S_shape AUROC |")
    L.append("|---|---|---|")
    for fid in fault_ids:
        cls = "A" if fid in AMP_SENSITIVE else "S" if fid in SHAPE_SENSITIVE else ""
        L.append(f"| {fid} | {cls} | {fmt_ms(*per_fault[fid][:2])} |")
    L.append("")
    L.append("> A=amplitude-sensitive, S=shape-sensitive.")
    L.append("")

    # 4) fold별 Δ
    L.append("## 4) fold별 Δ(S_shape − raw) — 24 fold (5-seed mean±std, Δ 오름차순)")
    L.append("")
    L.append("| fold | 유형 | target진폭 | Δ(shape−raw) | S_shape | raw |")
    L.append("|---|---|---|---|---|---|")
    for fold, d in sorted(fold_delta.items(), key=lambda kv: (kv[1]["delta"][0]
                          if kv[1]["delta"][0] == kv[1]["delta"][0] else 1e9)):
        L.append(f"| {fold} | {d['fold_type']} | {d['target_amp_group']} | "
                 f"**{fmt_ms(*d['delta'][:2])}** | {fmt_ms(*d['shape'][:2])} | {fmt_ms(*d['raw'][:2])} |")
    L.append("")
    L.append("> Δ<0 = 해당 fold에서 S_shape가 raw보다 나쁨(손해 fold). Δ>0 = 이득 fold.")
    L.append("")

    # 5) threshold 붕괴
    L.append("## 5) threshold 붕괴 지점 — bearing이 **held-out test 정상**일 때의 FPR(S_shape)")
    L.append("")
    L.append("| bearing | 진폭군 | test fold수 | test FPR | mean(S_shape) | mean_score−threshold | mean_rms |")
    L.append("|---|---|---|---|---|---|---|")
    for b in sorted(bearing_summary):
        d = bearing_summary[b]
        L.append(f"| {b} | {d['group']} | {d['n_test_folds']} | {fmt_ms(*d['fpr'][:2])} | "
                 f"{fmt_ms(*d['mean_score'][:2])} | {fmt_ms(*d['gap_score_minus_thr'][:2], p=3)} | "
                 f"{fmt_ms(*d['mean_rms'][:2])} |")
    L.append("")
    L.append("> 각 bearing이 held-out test 정상으로 등장한 fold(4 split × 5 seed = 20)만 집계 — "
             "train(학습에서 봄)·val(정의상 FPR≈0.05 고정)을 제외해 threshold 붕괴를 정직하게 본다. "
             "`mean_score−threshold`>0 이면 정상 window 평균 score가 threshold보다 이상(anomaly)쪽 = 붕괴. "
             "저진폭(K004/K005)이 test일 때 FPR이 치솟으면 val↔test 진폭분포 shift로 threshold가 무너진 것.")
    L.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print(f"S_shape overall AUROC = {fmt_ms(*auroc_overall[:2])} "
          f"(기존 리포트 0.776±0.047과 대조) | zero {fmt_ms(*zt[:2])} / comp {fmt_ms(*ct[:2])}")
    print(f"정상 FPR(S_shape) 고 {fmt_ms(*fpr_overall['high_fpr'][:2])} / 저 {fmt_ms(*fpr_overall['low_fpr'][:2])}")


if __name__ == "__main__":
    main()
