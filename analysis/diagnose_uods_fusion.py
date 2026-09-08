"""UODS 외부 검증 — Fisher-tail 재구성 + subgroup AUROC (analysis-only, CPU).

동결된 fusion 함수(diagnose_G5_tail_fusion: tail_prob·fisher_tail·equal_z)를 그대로 import해
proposed dump(.npz)에서 최종 window-level 이상점수를 재구성한다. tail 확률·threshold는 PU와 동일하게
**val-normal only**로 보정한다(test-side 미사용). raw dump(.npz, --raw_cache)가 있으면 raw vs proposed 1차 비교.

subgroup(문서 §5/§6): overall · developing-only(state1) · faulty-only(state2) · family(I/O/B/C) · ball/non-ball.
sanity: fisher ≡ product-of-p(단조 동치) AUROC |Δ|.

산출: results/UODS/diag_uods_fusion/<run>.json  + 표 출력
사용: conda run -n mtgflow python analysis/diagnose_uods_fusion.py --proposed_cache <p.npz> [--raw_cache <r.npz>]
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_G3a_amp_bands import safe_auroc  # noqa: E402
from diagnose_G5_tail_fusion import tail_prob, fisher_tail, equal_z  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "UODS", "diag_uods_fusion")


def subgroup_aurocs(score, state, fam, isball):
    """score(test-window) → subgroup AUROC 딕셔너리. 각 subgroup은 test-normal(state0) + 해당 fault."""
    state = np.asarray(state, int); fam = np.asarray(fam); isball = np.asarray(isball, bool)
    score = np.asarray(score, float)
    norm = state == 0
    out = {}
    out["overall"] = safe_auroc((state >= 1).astype(int), score)
    for key, fmask in [("developing", state == 1), ("faulty", state == 2)]:
        m = norm | fmask
        out[key] = safe_auroc(fmask[m].astype(int), score[m])
    for fname in ["inner", "outer", "ball", "cage"]:
        fmask = (state >= 1) & (fam == fname)
        m = norm | fmask
        out["fam_" + fname] = safe_auroc(fmask[m].astype(int), score[m])
    for key, fmask in [("ball", (state >= 1) & isball), ("nonball", (state >= 1) & (~isball))]:
        m = norm | fmask
        out[key] = safe_auroc(fmask[m].astype(int), score[m])
    # non-ball developing / faulty (ball no-load confound 배제 핵심 지표)
    for key, fmask in [("nonball_developing", (state == 1) & (~isball)),
                       ("nonball_faulty", (state == 2) & (~isball))]:
        m = norm | fmask
        out[key] = safe_auroc(fmask[m].astype(int), score[m])
    return out


def fpr_at_val95(test_norm_score, val_score, pct=95.0):
    thr = float(np.percentile(val_score, pct)) if len(val_score) else float("inf")
    return float(np.mean(np.asarray(test_norm_score) >= thr)), thr


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    return d, meta


def evaluate_run(proposed_cache, raw_cache=None, thr_pct=95.0):
    """proposed(+옵션 raw) dump에서 method×subgroup AUROC·val-95 FPR·sanity를 계산해 dict 반환.

    보정(tail 확률·equal-z·threshold)은 모두 val-normal only. report_uods_pilot이 seed별로 재사용.
    """
    d, meta = load_npz(proposed_cache)
    assert meta.get("kind") == "proposed", "proposed_cache must be a proposed dump"
    te_sh, te_am = d["te_sh"], d["te_am"]
    va_sh, va_am = d["va_sh"], d["va_am"]
    state = d["te_state"]; fam = d["te_fam"]; isball = d["te_isball"]
    norm = np.asarray(state, int) == 0

    # 최종 결합(동결): 각 branch를 val-normal tail 확률로 변환 → Fisher χ². equal-z는 비교기준.
    fisher_te = fisher_tail(te_sh, te_am, va_sh, va_am)
    fisher_va = fisher_tail(va_sh, va_am, va_sh, va_am)
    eqz_te = equal_z(te_sh, te_am, va_sh, va_am)
    eqz_va = equal_z(va_sh, va_am, va_sh, va_am)

    methods = {
        "shape": (te_sh, va_sh),
        "amp": (te_am, va_am),
        "equal_z": (eqz_te, eqz_va),
        "fisher": (fisher_te, fisher_va),
    }
    if raw_cache:
        dr, mr = load_npz(raw_cache)
        assert mr.get("kind") == "raw", "raw_cache must be a raw dump"
        # raw dump이 동일 split·순서인지 확인(누수·정렬 sanity)
        assert np.array_equal(dr["te_state"], state) and np.array_equal(dr["te_ids"], d["te_ids"]), \
            "raw/proposed dump의 test 정렬 불일치 — 동일 split·seed인지 확인"
        methods = {"raw": (dr["te_raw"], dr["va_raw"]), **methods}

    results = {}
    for name, (te_s, va_s) in methods.items():
        sub = subgroup_aurocs(te_s, state, fam, isball)
        fpr, thr = fpr_at_val95(te_s[norm], va_s, thr_pct)
        results[name] = dict(subgroup=sub, fpr_val95=fpr, threshold=thr)

    # sanity: fisher ≡ product-of-p (단조 동치) → AUROC 동일해야 함
    prodp_te = tail_prob(te_sh, va_sh) * tail_prob(te_am, va_am)   # 작을수록 이상
    auroc_fisher = results["fisher"]["subgroup"]["overall"]
    auroc_prodp = safe_auroc((np.asarray(state, int) >= 1).astype(int), -prodp_te)
    sanity = dict(fisher_vs_prodp_absdiff=abs(auroc_fisher - auroc_prodp),
                  n_test=int(len(state)), n_norm=int(norm.sum()), n_fault=int((~norm).sum()),
                  all_finite=bool(np.isfinite(fisher_te).all() and np.isfinite(eqz_te).all()))
    return dict(run_name=meta.get("run_name", "uods"), source=meta.get("source"),
                band_scheme=meta.get("amp_band_scheme"), band_edges=meta.get("amp_band_edges"),
                threshold_percentile=thr_pct, methods=results, sanity=sanity,
                proposed_cache=proposed_cache, raw_cache=raw_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed_cache", required=True)
    ap.add_argument("--raw_cache", default=None)
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    out = evaluate_run(args.proposed_cache, args.raw_cache, args.threshold_percentile)
    results = out["methods"]; sanity = out["sanity"]; methods = results
    os.makedirs(args.out_dir, exist_ok=True)
    run = out["run_name"]
    out_path = os.path.join(args.out_dir, f"{run}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # --- 표 출력 ---
    cols = ["overall", "developing", "faulty", "nonball_developing", "nonball_faulty",
            "fam_inner", "fam_outer", "fam_ball", "fam_cage", "ball", "nonball"]
    print(f"\n=== UODS Fisher-tail 재구성: {run} (val-normal 보정, {args.threshold_percentile:.0f}pct thr) ===")
    header = "method    " + " ".join(f"{c[:9]:>9}" for c in cols) + "   fpr95"
    print(header)
    for name in methods:
        sub = results[name]["subgroup"]
        row = " ".join(f"{sub.get(c, float('nan')):>9.3f}" for c in cols)
        print(f"{name:<9} {row}  {results[name]['fpr_val95']:>6.3f}")
    print(f"\nsanity: fisher≡product-of-p |ΔAUROC|={sanity['fisher_vs_prodp_absdiff']:.2e}  "
          f"(0에 가까워야 정상) | finite={sanity['all_finite']} | "
          f"test norm/fault = {sanity['n_norm']}/{sanity['n_fault']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
