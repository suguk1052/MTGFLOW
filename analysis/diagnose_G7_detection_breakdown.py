"""작업 G-7 — 최종 모델(G-3a 구조 + G-5 Fisher-tail)이 raw 대비 무엇을 얻고 잃었는지를
AUROC 한 숫자가 아니라 **검출률(recall)·오탐(FPR) 단위로 분해**하는 진단 (analysis-only, GPU 없음).

입력(모두 per-window, 재추론 캐시만 사용):
  A/B = results/Paderborn/g3a_window_scores/<fold>.npz   (S_shape/S_amp → equal_z / fisher_tail)
  raw = results/Paderborn/b3_raw_window_scores/<fold>.npz (B3 flow NLL, dump_B3_raw_window_scores.py 산출)
  raw 집계 대조 = results/Paderborn/diag_G3a_amp_bands/<fold>.json 의 raw.block (sanity)

비교 3종:  raw(B3 flow NLL) / A(equal-z S_total, G-3a) / B(Fisher-tail, G-5 최종)
threshold: 전부 **val 정상 score percentile 고정**(test 라벨 미사용).
fold 집계: zero-support(023→1·013→2·012→3) vs compositional(123→0) 분리 + 전체.
seed: --seeds 로 5-seed(2024~2028) 평균±표준편차.

산출:
  results/Paderborn/diag_G7_detection_breakdown/<fold>_s<seed>.json (fold별)
  results/Paderborn/diag_G7_detection_breakdown/aggregate_s<seed>.json (seed별)
  results/Paderborn/diag_G7_detection_breakdown/combined_5seed.json (5-seed mean±std)
  reports/figs_g7/fpr_recall_{high,low}.png (표 3 라인플롯)
  reports/report_G7_detection_breakdown.md 는 별도로 작성(이 스크립트는 표 원자료·그림 생성).
"""
import argparse
import json
import os

import numpy as np

from diagnose_G3a_amp_bands import (  # noqa: E402  (검증된 상수·유틸 재사용)
    AMP_SENSITIVE,
    SHAPE_SENSITIVE,
    HIGH_AMP,
    RESULTS_ROOT,
    PROJECT_ROOT,
    SPLIT_INFO,
    safe_auroc,
    _mean,
)
from diagnose_G5_tail_fusion import equal_z, fisher_tail  # noqa: E402  (A/B fusion 규칙 재사용)

G3A_CACHE = os.path.join(RESULTS_ROOT, "g3a_window_scores")
RAW_CACHE = os.path.join(RESULTS_ROOT, "b3_raw_window_scores")
DIAG_G3A_DIR = os.path.join(RESULTS_ROOT, "diag_G3a_amp_bands")
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_G7_detection_breakdown")
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figs_g7")

METHODS = ["raw", "A", "B"]
PCTS = [90.0, 95.0, 97.5, 99.0]
PRIMARY_PCT = 95.0


def _pk(p):
    """percentile → 문자열 키('90','95','97.5','99')."""
    return f"{p:g}"


def _std(vals):
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and x == x]
    return float(np.std(v)) if v else float("nan")


# ---------------------------------------------------------------------------
# 한 method 점수 S 에 대한 fold-단위 지표 (AUROC + per-fault recall + 진폭군 FPR sweep)
# ---------------------------------------------------------------------------
def score_block(S_te, S_va, S_tr, te_lab, te_ids, tr_ids, va_ids, tgt_te_ids):
    """
    S_te/S_va/S_tr : test/val/train per-window score (클수록 이상)
    te_lab         : test 라벨(0 정상, 1 결함)
    te_ids/tr_ids/va_ids : bearing id
    tgt_te_ids     : test-normal(target) bearing id (te_ids[te_lab==0])
    반환: auroc, per_fault{fid:{n,auroc,recall{pct}}}, per_bearing{bid:{group,mean_score,fpr{pct}}},
          sweep{pct:{high_fpr,low_fpr,overall_recall,amp_recall,shape_recall}}
    """
    tgt = te_lab == 0
    flt = te_lab == 1
    tgt_S = S_te[tgt]
    auroc = safe_auroc(te_lab, S_te)

    thr = {p: (float(np.percentile(S_va, p)) if len(S_va) else float("inf")) for p in PCTS}

    # per-fault: AUROC(정상 target vs 해당 fault) + 고정 임계값 recall
    per_fault = {}
    for fid in sorted(set(te_ids[flt].tolist())):
        m = flt & (te_ids == fid)
        labels = np.concatenate([np.zeros(int(tgt.sum()), int), np.ones(int(m.sum()), int)])
        scores = np.concatenate([tgt_S, S_te[m]])
        per_fault[fid] = {
            "n": int(m.sum()),
            "auroc": safe_auroc(labels, scores),
            "recall": {_pk(p): float(np.mean(S_te[m] >= thr[p])) for p in PCTS},
        }

    # 정상 pool = train + val + test-normal (진폭군 FPR은 _score_block(diag_G3a)과 동일 규약)
    norm_S = np.concatenate([S_tr, S_va, tgt_S])
    norm_ids = np.concatenate([tr_ids, va_ids, tgt_te_ids])
    per_bearing = {}
    for bid in sorted(set(norm_ids.tolist())):
        mm = norm_ids == bid
        per_bearing[bid] = {
            "group": "high" if bid in HIGH_AMP else "low",
            "mean_score": float(np.mean(norm_S[mm])),
            "fpr": {_pk(p): float(np.mean(norm_S[mm] >= thr[p])) for p in PCTS},
        }

    def grp_fpr(grp, p):
        vals = [v["fpr"][_pk(p)] for v in per_bearing.values() if v["group"] == grp]
        return float(np.mean(vals)) if vals else float("nan")

    def cls_recall(fault_set, p):
        vals = [per_fault[fid]["recall"][_pk(p)] for fid in fault_set if fid in per_fault]
        return float(np.mean(vals)) if vals else float("nan")

    sweep = {}
    for p in PCTS:
        sweep[_pk(p)] = {
            "high_fpr": grp_fpr("high", p),
            "low_fpr": grp_fpr("low", p),
            "overall_recall": float(np.mean(S_te[flt] >= thr[p])),
            "amp_recall": cls_recall(AMP_SENSITIVE, p),
            "shape_recall": cls_recall(SHAPE_SENSITIVE, p),
        }
    return {"auroc": auroc, "per_fault": per_fault, "per_bearing": per_bearing, "sweep": sweep}


# ---------------------------------------------------------------------------
# fold 하나 = raw/A/B 세 method 블록 (+ diag_G3a raw.block 과 sanity 대조)
# ---------------------------------------------------------------------------
def process_fold(split, lono, seed):
    g3a_p = os.path.join(G3A_CACHE, f"{split}_LONO{lono}_s{seed}.npz")
    raw_p = os.path.join(RAW_CACHE, f"{split}_LONO{lono}_s{seed}.npz")
    if not os.path.exists(g3a_p) or not os.path.exists(raw_p):
        print(f"    [skip] 캐시 없음: {'g3a' if not os.path.exists(g3a_p) else 'raw'} — {split} LONO{lono} s{seed}")
        return None

    g = np.load(g3a_p, allow_pickle=False)
    r = np.load(raw_p, allow_pickle=False)
    meta = json.loads(str(g["meta_json"]))

    te_lab, te_ids = g["te_lab"], g["te_ids"]
    tr_ids, va_ids = g["tr_ids"], g["va_ids"]
    tgt = te_lab == 0
    tgt_te_ids = te_ids[tgt]

    # raw 캐시와 g3a 캐시가 동일 fold(같은 window/label)인지 검증
    assert len(r["te_lab"]) == len(te_lab) and bool(np.array_equal(r["te_lab"], te_lab)), \
        f"raw/g3a te_lab 불일치: {split} LONO{lono} s{seed}"

    # A/B per-window (val 정상 통계 기반, test 라벨 무사용)
    A_te = equal_z(g["te_sh"], g["te_am"], g["va_sh"], g["va_am"])
    A_va = equal_z(g["va_sh"], g["va_am"], g["va_sh"], g["va_am"])
    A_tr = equal_z(g["tr_sh"], g["tr_am"], g["va_sh"], g["va_am"])
    B_te = fisher_tail(g["te_sh"], g["te_am"], g["va_sh"], g["va_am"])
    B_va = fisher_tail(g["va_sh"], g["va_am"], g["va_sh"], g["va_am"])
    B_tr = fisher_tail(g["tr_sh"], g["tr_am"], g["va_sh"], g["va_am"])

    blocks = {
        "raw": score_block(r["te_f"], r["va_f"], r["tr_f"], te_lab, te_ids, tr_ids, va_ids, tgt_te_ids),
        "A": score_block(A_te, A_va, A_tr, te_lab, te_ids, tr_ids, va_ids, tgt_te_ids),
        "B": score_block(B_te, B_va, B_tr, te_lab, te_ids, tr_ids, va_ids, tgt_te_ids),
    }

    # sanity: 재추론 raw AUROC 및 A(=fusion_A) AUROC 를 diag_G3a JSON 과 대조
    raw_ref_auroc = None
    a_ref_auroc = None
    g3a_json = os.path.join(DIAG_G3A_DIR, f"{split}_LONO{lono}_s{seed}.json")
    if os.path.exists(g3a_json):
        with open(g3a_json) as f:
            d = json.load(f)
        raw_ref_auroc = (d.get("raw") or {}).get("auroc")
        a_ref_auroc = (d.get("g1") or {}).get("auroc_total")

    return {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": meta["fold_type"], "target_amp_group": meta["target_amp_group"],
        "target_norm_ids": meta["target_norm_ids"],
        "blocks": blocks,
        "raw_ref_auroc": raw_ref_auroc,
        "raw_repro_diff": (abs(blocks["raw"]["auroc"] - raw_ref_auroc)
                           if raw_ref_auroc is not None else None),
        "a_repro_diff": (abs(blocks["A"]["auroc"] - a_ref_auroc)
                         if a_ref_auroc is not None else None),
    }


# ---------------------------------------------------------------------------
# seed 하나 집계 (24 fold → 전체/fold_type 분리)
# ---------------------------------------------------------------------------
def _mean_over(folds, getter):
    return _mean([getter(f) for f in folds])


def aggregate_seed(folds):
    ok = [f for f in folds if f]

    def auroc_mean(rows, m):
        return _mean([f["blocks"][m]["auroc"] for f in rows])

    def subset(rows):
        # per-fault AUROC / recall(val95)
        fault = {}
        for f in rows:
            for m in METHODS:
                for fid, d in f["blocks"][m]["per_fault"].items():
                    e = fault.setdefault(fid, {mm: {"auroc": [], "recall95": []} for mm in METHODS})
                    e[m]["auroc"].append(d["auroc"])
                    e[m]["recall95"].append(d["recall"][_pk(PRIMARY_PCT)])
        per_fault = {}
        for fid, e in sorted(fault.items()):
            per_fault[fid] = {
                "family": fid[:2],
                "class": ("amp-sensitive" if fid in AMP_SENSITIVE else
                          "shape-sensitive" if fid in SHAPE_SENSITIVE else "other"),
                **{m: {"auroc": _mean(e[m]["auroc"]), "recall95": _mean(e[m]["recall95"])}
                   for m in METHODS},
            }
        # 진폭군 정상 FPR(val95) + sweep(pct별)
        normal_fpr = {m: {"high": _mean_over(rows, lambda f, m=m: f["blocks"][m]["sweep"][_pk(PRIMARY_PCT)]["high_fpr"]),
                          "low": _mean_over(rows, lambda f, m=m: f["blocks"][m]["sweep"][_pk(PRIMARY_PCT)]["low_fpr"])}
                      for m in METHODS}
        sweep = {m: {_pk(p): {k: _mean_over(rows, lambda f, m=m, p=p, k=k: f["blocks"][m]["sweep"][_pk(p)][k])
                              for k in ("high_fpr", "low_fpr", "overall_recall", "amp_recall", "shape_recall")}
                     for p in PCTS}
                 for m in METHODS}
        return {"n_folds": len(rows),
                "auroc": {m: auroc_mean(rows, m) for m in METHODS},
                "per_fault": per_fault, "normal_fpr": normal_fpr, "sweep": sweep}

    out = {"overall": subset(ok), "by_fold_type": {}}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ftype]
        if rows:
            out["by_fold_type"][ftype] = subset(rows)

    # per-bearing 정상 FPR/mean_score (val95) — bearing이 등장한 fold 평균
    bear = {}
    for f in ok:
        for m in METHODS:
            for bid, d in f["blocks"][m]["per_bearing"].items():
                e = bear.setdefault(bid, {"group": d["group"], **{mm: {"fpr95": [], "mean_score": []} for mm in METHODS}})
                e[m]["fpr95"].append(d["fpr"][_pk(PRIMARY_PCT)])
                e[m]["mean_score"].append(d["mean_score"])
    per_bearing = {bid: {"group": e["group"],
                         **{m: {"fpr95": _mean(e[m]["fpr95"]), "mean_score": _mean(e[m]["mean_score"])}
                            for m in METHODS}}
                   for bid, e in sorted(bear.items())}
    out["per_bearing"] = per_bearing

    # sanity 대조
    rr = [f["raw_repro_diff"] for f in ok if f.get("raw_repro_diff") is not None]
    ar = [f["a_repro_diff"] for f in ok if f.get("a_repro_diff") is not None]
    out["sanity"] = {
        "raw_repro_max_diff": (max(rr) if rr else None),
        "a_repro_max_diff": (max(ar) if ar else None),
        "n_folds": len(ok),
    }
    return out


# ---------------------------------------------------------------------------
# 5-seed 결합 (mean±std) — seed별 aggregate 리스트에서 leaf별로 취합
# ---------------------------------------------------------------------------
def combine_seeds(aggs):
    def ms(getter):
        vals = []
        for a in aggs:
            try:
                v = getter(a)
            except (KeyError, TypeError):
                v = None
            if v is not None and v == v:
                vals.append(v)
        return {"mean": (_mean(vals) if vals else float("nan")),
                "std": (_std(vals) if vals else float("nan"))}

    def subset_ms(path):
        # path(a) -> subset dict of one seed
        s0 = path(aggs[0])
        methods = METHODS
        res = {"auroc": {m: ms(lambda a, m=m: path(a)["auroc"][m]) for m in methods}}
        # per-fault
        fids = sorted(s0["per_fault"].keys())
        res["per_fault"] = {}
        for fid in fids:
            res["per_fault"][fid] = {
                "family": s0["per_fault"][fid]["family"],
                "class": s0["per_fault"][fid]["class"],
                **{m: {"auroc": ms(lambda a, fid=fid, m=m: path(a)["per_fault"][fid][m]["auroc"]),
                       "recall95": ms(lambda a, fid=fid, m=m: path(a)["per_fault"][fid][m]["recall95"])}
                   for m in methods},
            }
        res["normal_fpr"] = {m: {g: ms(lambda a, m=m, g=g: path(a)["normal_fpr"][m][g])
                                 for g in ("high", "low")} for m in methods}
        res["sweep"] = {m: {_pk(p): {k: ms(lambda a, m=m, p=p, k=k: path(a)["sweep"][m][_pk(p)][k])
                                     for k in ("high_fpr", "low_fpr", "overall_recall", "amp_recall", "shape_recall")}
                            for p in PCTS} for m in methods}
        return res

    combined = {"seeds": [a["_seed"] for a in aggs],
                "overall": subset_ms(lambda a: a["overall"]),
                "by_fold_type": {}}
    for ftype in ("zero-support", "compositional"):
        if all(ftype in a["by_fold_type"] for a in aggs):
            combined["by_fold_type"][ftype] = subset_ms(lambda a, ftype=ftype: a["by_fold_type"][ftype])

    bids = sorted(aggs[0]["per_bearing"].keys())
    combined["per_bearing"] = {}
    for bid in bids:
        combined["per_bearing"][bid] = {
            "group": aggs[0]["per_bearing"][bid]["group"],
            **{m: {"fpr95": ms(lambda a, bid=bid, m=m: a["per_bearing"][bid][m]["fpr95"]),
                   "mean_score": ms(lambda a, bid=bid, m=m: a["per_bearing"][bid][m]["mean_score"])}
               for m in METHODS},
        }
    combined["sanity"] = {
        "raw_repro_max_diff": max([a["sanity"]["raw_repro_max_diff"] or 0.0 for a in aggs]),
        "a_repro_max_diff": max([a["sanity"]["a_repro_max_diff"] or 0.0 for a in aggs]),
    }
    return combined


# ---------------------------------------------------------------------------
# 그림: 표 3 FPR–recall 트레이드오프 (진폭군별, raw vs B)
# ---------------------------------------------------------------------------
def make_figures(combined):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] matplotlib 없음 — 그림 생략: {e}")
        return []

    os.makedirs(FIG_DIR, exist_ok=True)
    sw = combined["overall"]["sweep"]
    paths = []
    for grp, fpr_key in (("high", "high_fpr"), ("low", "low_fpr")):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for m, color, mk in (("raw", "#888888", "o"), ("B", "#c0392b", "s")):
            xs = [sw[m][_pk(p)][fpr_key]["mean"] for p in PCTS]
            ys = [sw[m][_pk(p)]["overall_recall"]["mean"] for p in PCTS]
            ax.plot(xs, ys, "-", color=color, marker=mk, label=("raw" if m == "raw" else "B: Fisher-tail"))
            for p, x, y in zip(PCTS, xs, ys):
                ax.annotate(f"{_pk(p)}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)
        ax.set_xlabel(f"{grp}-amp normal FPR")
        ax.set_ylabel("fault recall (overall)")
        ax.set_title(f"FPR–recall trade-off ({grp}-amp normal)\nval percentile ∈ {{90,95,97.5,99}}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        p = os.path.join(FIG_DIR, f"fpr_recall_{grp}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
        print(f"    wrote {p}")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--no_figures", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_aggs = []
    for seed in args.seeds:
        folds = []
        for split in args.splits:
            for lono in args.lonos:
                res = process_fold(split, lono, seed)
                if res is None:
                    continue
                folds.append(res)
                with open(os.path.join(args.out_dir, f"{split}_LONO{lono}_s{seed}.json"), "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
        if not folds:
            print(f"  [skip seed {seed}] 평가된 fold 없음")
            continue
        agg = aggregate_seed(folds)
        agg["_seed"] = seed
        seed_aggs.append(agg)
        with open(os.path.join(args.out_dir, f"aggregate_s{seed}.json"), "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        o = agg["overall"]["auroc"]
        sn = agg["sanity"]
        print(f"[seed {seed}] {agg['overall']['n_folds']} fold  "
              f"AUROC raw={o['raw']:.3f} A={o['A']:.3f} B={o['B']:.3f}  "
              f"[raw repro Δmax={sn['raw_repro_max_diff']}, A repro Δmax={sn['a_repro_max_diff']}]")

    if not seed_aggs:
        raise SystemExit(f"평가된 seed 없음 — 캐시 확인: {G3A_CACHE}, {RAW_CACHE}")

    combined = combine_seeds(seed_aggs)
    with open(os.path.join(args.out_dir, "combined_5seed.json"), "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # ---- sanity 요약 (전체 AUROC 재현) ----
    ov = combined["overall"]["auroc"]
    print("\n=== sanity: 전체 AUROC (5-seed mean±std) ===")
    for m, ref in (("raw", 0.696), ("A", 0.762), ("B", 0.800)):
        c = ov[m]
        flag = "OK" if abs(c["mean"] - ref) < 0.02 else "⚠️ 불일치"
        print(f"  {m:>3}: {c['mean']:.3f}±{c['std']:.3f}  (기존 {ref:.3f})  {flag}")
    san = combined["sanity"]
    print(f"  raw per-fold AUROC vs diag_G3a 최대 |Δ| = {san['raw_repro_max_diff']:.2e}"
          + ("  OK" if san['raw_repro_max_diff'] < 1e-6 else "  ⚠️ (>1e-6)"))
    print(f"  A   per-fold AUROC vs diag_G3a 최대 |Δ| = {san['a_repro_max_diff']:.2e}"
          + ("  OK" if san['a_repro_max_diff'] < 1e-6 else "  ⚠️ (>1e-6)"))

    if not args.no_figures:
        make_figures(combined)

    print(f"\n산출: {os.path.join(args.out_dir, 'combined_5seed.json')}")


if __name__ == "__main__":
    main()
