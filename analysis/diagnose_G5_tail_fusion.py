"""작업 G-5 Step 2 — branch score fusion 규칙 비교 (analysis-only, GPU 없음).

dump_G3a_window_scores.py 가 남긴 per-window score 캐시(.npz)만 읽어, 동일 per-window
S_shape/S_amp에 **두 fusion 규칙**을 적용하고 G-3a와 같은 지표 파이프라인(_score_block)으로 평가한다.
학습·재추론 없음(캐시만 사용).

  A(현행 재현) = equal-z: val-normal 표준화 등가중 합  z_val(S_shape)+z_val(S_amp).
                → 기존 diag_G3a JSON 의 auroc_total 을 fold별로 재현해야 캐시·재구현이 옳다는 sanity.
  B(후보) = Fisher-tail: 각 branch를 val-normal 경험적 상단 tail 확률 p 로 변환 후 결합.
      p(s) = (#{v_i >= s} + 1)/(N+1)   (v = val-normal score, N개; rank 기반 +1 floor, log0 방지)
             → 큰 score(이상)일수록 정상에서 드묾 → p→0. test score/label 무사용(label-free).
      S_fisher = -2[ln p_shape + ln p_amp]   (Fisher χ² 결합, 클수록 이상. -2 상수는 AUROC와 무관·해석용.)

raw baseline(paired 비교용)은 재추론하지 않고 기존 diag_G3a JSON의 raw 블록을 그대로 병합한다.

산출:
  results/Paderborn/diag_G5_tail_fusion/<split>_LONO<n>_s<seed>.json  (fold별)
  results/Paderborn/diag_G5_tail_fusion/aggregate_G5_s<seed>.json     (seed별 집계)
"""
import argparse
import json
import os

import numpy as np

from diagnose_G3a_amp_bands import (  # noqa: E402  (검증된 지표·상수 재사용)
    AMP_SENSITIVE,
    SHAPE_SENSITIVE,
    RESULTS_ROOT,
    SPLIT_INFO,
    _mean,
    _score_block,
)

CACHE_DIR = os.path.join(RESULTS_ROOT, "g3a_window_scores")
DIAG_G3A_DIR = os.path.join(RESULTS_ROOT, "diag_G3a_amp_bands")
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_G5_tail_fusion")

METHODS = ["shape", "amp", "fusion_A", "fusion_B", "raw"]


# ---------------------------------------------------------------------------
# fusion 규칙 (모두 val-normal only, label-free)
# ---------------------------------------------------------------------------
def equal_z(sh, am, va_sh, va_am):
    """A: val-normal 평균/표준편차로 z-score 후 등가중 합 (eval_g3a.ztot 와 동일 정의)."""
    mu_s, sd_s = float(va_sh.mean()), float(va_sh.std() + 1e-8)
    mu_a, sd_a = float(va_am.mean()), float(va_am.std() + 1e-8)
    return (sh - mu_s) / sd_s + (am - mu_a) / sd_a


def tail_prob(scores, val_scores):
    """val-normal 경험적 상단 tail 확률 p(s)=(#{v>=s}+1)/(N+1). score↑ → p↓, p∈(0,1]."""
    v = np.sort(np.asarray(val_scores, dtype=np.float64))
    n = len(v)
    if n == 0:
        return np.ones_like(np.asarray(scores, dtype=np.float64))
    ge = n - np.searchsorted(v, np.asarray(scores, dtype=np.float64), side="left")  # #{v_i >= s}
    return (ge + 1.0) / (n + 1.0)


def fisher_tail(sh, am, va_sh, va_am):
    """B: 각 branch tail 확률의 Fisher χ² 결합 -2[ln p_shape + ln p_amp] (클수록 이상)."""
    p_sh = tail_prob(sh, va_sh)
    p_am = tail_prob(am, va_am)
    return -2.0 * (np.log(p_sh) + np.log(p_am))


# ---------------------------------------------------------------------------
# fold 하나 평가
# ---------------------------------------------------------------------------
def process_fold(split, lono, seed, cache_dir, diag_dir, thr_pct):
    cache_path = os.path.join(cache_dir, f"{split}_LONO{lono}_s{seed}.npz")
    if not os.path.exists(cache_path):
        print(f"    [skip] 캐시 없음: {cache_path}")
        return None
    z = np.load(cache_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))

    te_sh, te_am = z["te_sh"], z["te_am"]
    va_sh, va_am = z["va_sh"], z["va_am"]
    tr_sh, tr_am = z["tr_sh"], z["tr_am"]
    te_lab, te_ids, te_rms = z["te_lab"], z["te_ids"], z["te_rms"]
    va_ids, va_rms = z["va_ids"], z["va_rms"]
    tr_ids, tr_rms = z["tr_ids"], z["tr_rms"]

    # 각 method의 (test, val, train) score — eval_g3a 와 동일 순서로 구성
    A_te, A_va, A_tr = equal_z(te_sh, te_am, va_sh, va_am), equal_z(va_sh, va_am, va_sh, va_am), equal_z(tr_sh, tr_am, va_sh, va_am)
    B_te = fisher_tail(te_sh, te_am, va_sh, va_am)
    B_va = fisher_tail(va_sh, va_am, va_sh, va_am)
    B_tr = fisher_tail(tr_sh, tr_am, va_sh, va_am)

    tgt = te_lab == 0
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])

    def blk(name, S_te, S_va, S_tr):
        return _score_block(name, S_te, S_va, te_lab, te_ids, te_rms, S_tr, S_va,
                            te_rms[tgt], norm_rms, norm_ids, thr_pct)

    blocks = {
        "shape": blk("shape", te_sh, va_sh, tr_sh),
        "amp": blk("amp", te_am, va_am, tr_am),
        "fusion_A": blk("fusion_A", A_te, A_va, A_tr),
        "fusion_B": blk("fusion_B", B_te, B_va, B_tr),
    }

    # raw baseline: 기존 diag_G3a JSON에서 그대로 병합(재추론 없음)
    raw = None
    g3a_path = os.path.join(diag_dir, f"{split}_LONO{lono}_s{seed}.json")
    if os.path.exists(g3a_path):
        with open(g3a_path) as f:
            g3a = json.load(f)
        raw = g3a.get("raw")

    out = {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": meta["fold_type"], "desc": meta["desc"],
        "target_amp_group": meta["target_amp_group"],
        "target_norm_ids": meta["target_norm_ids"], "amp_n_bands": meta["amp_n_bands"],
        "blocks": blocks, "raw": raw,
        "auroc": {m: blocks[m]["auroc"] for m in ("shape", "amp", "fusion_A", "fusion_B")},
    }
    # sanity: fusion_A 는 기존 auroc_total 재현이어야 함
    if os.path.exists(g3a_path):
        ref = g3a.get("g1", {}).get("auroc_total")
        if ref is not None:
            out["reproduce_total_diff"] = abs(blocks["fusion_A"]["auroc"] - ref)
    return out


# ---------------------------------------------------------------------------
# 집계 (method-generic)
# ---------------------------------------------------------------------------
def _blk(f, method):
    if method == "raw":
        return f["raw"]["block"] if f.get("raw") else None
    return f["blocks"].get(method)


def aggregate(folds):
    ok = [f for f in folds if f]

    def auroc_mean(rows, m):
        return _mean([_blk(f, m)["auroc"] for f in rows if _blk(f, m)])

    overall = {m: auroc_mean(ok, m) for m in METHODS}

    by_fold_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ftype]
        if not rows:
            continue
        by_fold_type[ftype] = {
            "n_folds": len(rows),
            "auroc": {m: auroc_mean(rows, m) for m in METHODS},
            "rho": {m: _mean([_blk(f, m)["rho_rms_score"] for f in rows if _blk(f, m)]) for m in METHODS},
        }

    normal_fpr = {
        m: {"high": _mean([_blk(f, m)["high_amp_fpr"] for f in ok if _blk(f, m)]),
            "low": _mean([_blk(f, m)["low_amp_fpr"] for f in ok if _blk(f, m)])}
        for m in METHODS
    }
    rho = {m: _mean([_blk(f, m)["rho_rms_score"] for f in ok if _blk(f, m)]) for m in METHODS}

    # per-fault AUROC(fault id별 fold 평균) → fault군 평균
    fault_agg = {}
    for f in ok:
        for m in METHODS:
            b = _blk(f, m)
            if not b:
                continue
            for fid, d in b["per_fault_auroc"].items():
                fault_agg.setdefault(fid, {mm: [] for mm in METHODS})[m].append(d["auroc"])
    per_fault = {
        fid: {**{m: _mean(e[m]) for m in METHODS},
              "class": ("amp-sensitive" if fid in AMP_SENSITIVE else
                        "shape-sensitive" if fid in SHAPE_SENSITIVE else "other")}
        for fid, e in sorted(fault_agg.items())
    }

    def grp_mean(fault_set, m):
        return _mean([per_fault[fid][m] for fid in per_fault if fid in fault_set])

    fault_group = {
        "amp_sensitive": {m: grp_mean(AMP_SENSITIVE, m) for m in METHODS},
        "shape_sensitive": {m: grp_mean(SHAPE_SENSITIVE, m) for m in METHODS},
    }

    repro = [f["reproduce_total_diff"] for f in ok if "reproduce_total_diff" in f]
    return {
        "n_folds": len(ok), "methods": METHODS,
        "overall": overall, "by_fold_type": by_fold_type,
        "normal_fpr": normal_fpr, "rho": rho,
        "fault_group": fault_group, "per_fault": per_fault,
        "reproduce_total_max_diff": (max(repro) if repro else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    ap.add_argument("--diag_dir", type=str, default=DIAG_G3A_DIR)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    folds = []
    for split in args.splits:
        for lono in args.lonos:
            print(f"[{split} LONO{lono} s{args.seed}]")
            res = process_fold(split, lono, args.seed, args.cache_dir, args.diag_dir,
                               args.threshold_percentile)
            if res is None:
                continue
            folds.append(res)
            with open(os.path.join(args.out_dir, f"{split}_LONO{lono}_s{args.seed}.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            a = res["auroc"]
            rp = res.get("reproduce_total_diff")
            print(f"    shape={a['shape']:.3f} amp={a['amp']:.3f} "
                  f"A(equal-z)={a['fusion_A']:.3f} B(fisher)={a['fusion_B']:.3f}"
                  + (f"  [repro Δ={rp:.2e}]" if rp is not None else ""))

    if not folds:
        raise SystemExit(f"평가된 fold 없음 — 캐시 확인: {args.cache_dir}")

    agg = aggregate(folds)
    with open(os.path.join(args.out_dir, f"aggregate_G5_s{args.seed}.json"), "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    o = agg["overall"]
    print(f"\n=== 전체 ({agg['n_folds']} fold, seed {args.seed}) ===")
    print(f"  shape={o['shape']:.3f}  amp={o['amp']:.3f}  "
          f"A(equal-z)={o['fusion_A']:.3f}  B(fisher)={o['fusion_B']:.3f}  raw={o['raw']:.3f}")
    md = agg["reproduce_total_max_diff"]
    if md is not None:
        print(f"  [sanity] fusion_A vs 기존 auroc_total 최대 |Δ| = {md:.2e}  "
              + ("OK" if md < 1e-6 else "⚠️ 재현 실패(>1e-6)"))


if __name__ == "__main__":
    main()
