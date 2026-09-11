"""작업 Q-1 — 결함별 검출률 표를 **log 최종안 기준**으로 재산출 (analysis-only, GPU 없음).

G-7(report_G7_detection_breakdown.md)의 14종 결함별 AUROC·val95 검출률 표는 linear(G-5) 기준이다.
최종 채택안이 P-2에서 log 밴드로 승격(전체 AUROC 0.800→0.877)됐으므로, 같은 분해 표를 log 기준으로
다시 뽑는다. diagnose_G7_detection_breakdown.py 는 METHODS·sanity 가 linear 에 동결돼 있어 **원본을
건드리지 않고**, 그 method-무관 헬퍼(score_block, _pk, _std, _mean_over)와 fisher_tail·상수만 재사용한다.
학습·모델·하이퍼 변경 없음(per-window 캐시만 사용).

비교 3열:
  raw       = results/Paderborn/b3_raw_window_scores/<fold>.npz   te_f(B3 flow NLL, 클수록 이상)        5-seed
  logfisher = results/Paderborn/p2_log_window_scores/<fold>.npz   te_sh/te_am → fisher_tail(log 최종)   5-seed
  ocsvm     = results/Paderborn/b1_ocsvm_band_window_scores/<fold>.npz  -te_f(OC-SVM decision 부호반전)  **s2024 단일 run(결정론)**

  · logfisher 는 g3a 캐시와 동일 스키마(te_sh/te_am)에 G-5 fisher_tail 을 그대로 적용 → g-final log 추론 재현.
  · ocsvm 캐시는 s2024 1개만 존재. OC-SVM 은 결정론(seed 무관)이라 seed 루프마다 s2024 를 재사용 →
    5-seed 결합 시 std≈0. 리포트에서 "단일 run·결정론"으로 명시.

threshold: 전부 val 정상 score percentile 고정(test 라벨 미사용).
fold 집계: zero-support(023→1·013→2·012→3) vs compositional(123→0) 분리 + 전체. seed: --seeds 5-seed 평균±표준편차.

산출:
  results/Paderborn/diag_Q1_detection_breakdown_log/<fold>_s<seed>.json (fold별)
  results/Paderborn/diag_Q1_detection_breakdown_log/aggregate_s<seed>.json (seed별)
  results/Paderborn/diag_Q1_detection_breakdown_log/combined_5seed.json (5-seed mean±std)
  reports/report_Q1_detection_breakdown_log.md 는 별도로 작성(이 스크립트는 표 원자료 생성).
"""
import argparse
import json
import os

import numpy as np

from diagnose_G3a_amp_bands import (  # noqa: E402  (검증된 상수·유틸 재사용)
    AMP_SENSITIVE,
    SHAPE_SENSITIVE,
    RESULTS_ROOT,
    SPLIT_INFO,
    _mean,
)
from diagnose_G5_tail_fusion import fisher_tail  # noqa: E402  (log 최종 fusion 규칙 재사용)
from diagnose_G7_detection_breakdown import (  # noqa: E402  (method-무관 지표/집계 헬퍼 재사용)
    score_block,
    _pk,
    _std,
    _mean_over,
)

RAW_CACHE = os.path.join(RESULTS_ROOT, "b3_raw_window_scores")
LOG_CACHE = os.path.join(RESULTS_ROOT, "p2_log_window_scores")
OCSVM_CACHE = os.path.join(RESULTS_ROOT, "b1_ocsvm_band_window_scores")
OCSVM_SEED = 2024  # b1_ocsvm 은 s2024 1개만 존재(결정론). seed 루프와 무관하게 항상 이 파일 사용.
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_Q1_detection_breakdown_log")

METHODS = ["raw", "logfisher", "ocsvm"]
PCTS = [90.0, 95.0, 97.5, 99.0]
PRIMARY_PCT = 95.0

# sanity: 기존 리포트의 전체 AUROC (raw=B3 0.696 / log Fisher=P-2 0.877 / OC-SVM·band=B-1 0.933)
SANITY_REF = {"raw": 0.696, "logfisher": 0.877, "ocsvm": 0.933}


# ---------------------------------------------------------------------------
# fold 하나 = raw/logfisher/ocsvm 세 method 블록
# ---------------------------------------------------------------------------
def process_fold(split, lono, seed):
    raw_p = os.path.join(RAW_CACHE, f"{split}_LONO{lono}_s{seed}.npz")
    log_p = os.path.join(LOG_CACHE, f"{split}_LONO{lono}_s{seed}.npz")
    ocsvm_p = os.path.join(OCSVM_CACHE, f"{split}_LONO{lono}_s{OCSVM_SEED}.npz")
    for tag, p in (("raw", raw_p), ("log", log_p), ("ocsvm", ocsvm_p)):
        if not os.path.exists(p):
            print(f"    [skip] 캐시 없음: {tag} — {split} LONO{lono} s{seed}")
            return None

    r = np.load(raw_p, allow_pickle=False)
    g = np.load(log_p, allow_pickle=False)
    o = np.load(ocsvm_p, allow_pickle=False)
    meta = json.loads(str(g["meta_json"]))  # fold_type/target_* 는 p2_log meta 사용

    te_lab, te_ids = g["te_lab"], g["te_ids"]
    tr_ids, va_ids = g["tr_ids"], g["va_ids"]
    tgt = te_lab == 0
    tgt_te_ids = te_ids[tgt]

    # 세 캐시가 동일 fold(같은 window/label 정렬)인지 검증 — 누수·정렬 sanity (G-7 패턴 계승)
    assert bool(np.array_equal(r["te_lab"], te_lab)), f"raw/log te_lab 불일치: {split} LONO{lono} s{seed}"
    assert bool(np.array_equal(o["te_lab"], te_lab)), f"ocsvm/log te_lab 불일치: {split} LONO{lono}"
    assert bool(np.array_equal(r["te_ids"], te_ids)), f"raw/log te_ids 불일치: {split} LONO{lono} s{seed}"
    assert bool(np.array_equal(o["te_ids"], te_ids)), f"ocsvm/log te_ids 불일치: {split} LONO{lono}"

    # log Fisher per-window (val 정상 tail 확률 기반, test 라벨 무사용) — g-final log 추론과 동일
    L_te = fisher_tail(g["te_sh"], g["te_am"], g["va_sh"], g["va_am"])
    L_va = fisher_tail(g["va_sh"], g["va_am"], g["va_sh"], g["va_am"])
    L_tr = fisher_tail(g["tr_sh"], g["tr_am"], g["va_sh"], g["va_am"])

    blocks = {
        "raw": score_block(r["te_f"], r["va_f"], r["tr_f"], te_lab, te_ids, tr_ids, va_ids, tgt_te_ids),
        "logfisher": score_block(L_te, L_va, L_tr, te_lab, te_ids, tr_ids, va_ids, tgt_te_ids),
        # b1_ocsvm 캐시의 te_f 는 이미 anomaly=−decision_function 으로 저장됨(if_ocsvm.py:62-64) → 그대로 사용(클수록 이상).
        "ocsvm": score_block(o["te_f"], o["va_f"], o["tr_f"], te_lab, te_ids, tr_ids, va_ids, tgt_te_ids),
    }

    return {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": meta["fold_type"], "target_amp_group": meta["target_amp_group"],
        "target_norm_ids": meta.get("target_norm_ids"),
        "blocks": blocks,
    }


# ---------------------------------------------------------------------------
# seed 하나 집계 (24 fold → 전체/fold_type 분리) — diagnose_G7.aggregate_seed 를 3-method로 이식
# ---------------------------------------------------------------------------
def aggregate_seed(folds):
    ok = [f for f in folds if f]

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
                "auroc": {m: _mean([f["blocks"][m]["auroc"] for f in rows]) for m in METHODS},
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
    out["n_folds"] = len(ok)
    return out


# ---------------------------------------------------------------------------
# 5-seed 결합 (mean±std) — diagnose_G7.combine_seeds 를 3-method로 이식
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
        s0 = path(aggs[0])
        res = {"auroc": {m: ms(lambda a, m=m: path(a)["auroc"][m]) for m in METHODS}}
        fids = sorted(s0["per_fault"].keys())
        res["per_fault"] = {}
        for fid in fids:
            res["per_fault"][fid] = {
                "family": s0["per_fault"][fid]["family"],
                "class": s0["per_fault"][fid]["class"],
                **{m: {"auroc": ms(lambda a, fid=fid, m=m: path(a)["per_fault"][fid][m]["auroc"]),
                       "recall95": ms(lambda a, fid=fid, m=m: path(a)["per_fault"][fid][m]["recall95"])}
                   for m in METHODS},
            }
        res["normal_fpr"] = {m: {gp: ms(lambda a, m=m, gp=gp: path(a)["normal_fpr"][m][gp])
                                 for gp in ("high", "low")} for m in METHODS}
        res["sweep"] = {m: {_pk(p): {k: ms(lambda a, m=m, p=p, k=k: path(a)["sweep"][m][_pk(p)][k])
                                     for k in ("high_fpr", "low_fpr", "overall_recall", "amp_recall", "shape_recall")}
                            for p in PCTS} for m in METHODS}
        return res

    combined = {"seeds": [a["_seed"] for a in aggs],
                "ocsvm_note": f"OC-SVM·band = s{OCSVM_SEED} 단일 run(결정론, seed 무관) → std≈0",
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
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--no_figures", action="store_true")  # G-7 인터페이스 호환(이 스크립트는 그림 없음)
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
        print(f"[seed {seed}] {agg['overall']['n_folds']} fold  "
              f"AUROC raw={o['raw']:.3f} logfisher={o['logfisher']:.3f} ocsvm={o['ocsvm']:.3f}")

    if not seed_aggs:
        raise SystemExit(f"평가된 seed 없음 — 캐시 확인: {RAW_CACHE}, {LOG_CACHE}, {OCSVM_CACHE}")

    combined = combine_seeds(seed_aggs)
    with open(os.path.join(args.out_dir, "combined_5seed.json"), "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    # ---- sanity 요약 (전체 AUROC 재현) ----
    ov = combined["overall"]["auroc"]
    print("\n=== sanity: 전체 AUROC (5-seed mean±std) ===")
    for m in METHODS:
        c = ov[m]
        ref = SANITY_REF[m]
        flag = "OK" if abs(c["mean"] - ref) < 0.02 else "⚠️ 불일치"
        print(f"  {m:>9}: {c['mean']:.3f}±{c['std']:.3f}  (기존 {ref:.3f})  {flag}")

    # ---- KA30/KI04/KB23 요약 (필수 기록) ----
    pf = combined["overall"]["per_fault"]
    print("\n=== amp-sensitive 재점검 (log 최종안) ===")
    for fid in ("KA30", "KI04", "KB23"):
        if fid in pf:
            e = pf[fid]
            print(f"  {fid}: AUROC raw={e['raw']['auroc']['mean']:.3f} log={e['logfisher']['auroc']['mean']:.3f} "
                  f"ocsvm={e['ocsvm']['auroc']['mean']:.3f} | "
                  f"recall95 raw={e['raw']['recall95']['mean']:.3f} log={e['logfisher']['recall95']['mean']:.3f}")

    print(f"\n산출: {os.path.join(args.out_dir, 'combined_5seed.json')}")


if __name__ == "__main__":
    main()
