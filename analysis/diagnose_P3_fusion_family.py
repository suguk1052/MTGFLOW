#!/usr/bin/env python3
"""P-3 Fusion family sweep — "기여는 브랜치이지 퓨전 트릭이 아니다" (CPU, 학습·추론 없음).

N=6 + log(P-2 확정 최종 구조)의 per-window score 캐시(results/Paderborn/p2_log_window_scores/*.npz)만 읽어,
S_shape·S_amp에 **여러 결합 규칙**을 적용하고 G-3a와 동일 지표 파이프라인(_score_block)으로 평가한다.
재학습·재추론 없음. 모든 규칙은 val-normal만 사용(test score/label 무사용). 하이퍼파라미터는 결과 확인 전 고정(아래 상수).

규칙(전부 "클수록 이상"):
  단일(하한): shape=S_shape · amp=S_amp · raw=B3 flow NLL(diag_G3a JSON에서 병합).
  결합: equal_z(=weighted-z w=0.5 등가) · weighted-z w∈{0.0..1.0}(0/1=단일=기준점, 비퇴화 0<w<1) ·
        fisher(현행) · stouffer · tippett(min-p) · wilkinson(max-p) · hmp(harmonic-mean-p) ·
        ranksum(경험적 CDF rank 합) · maxz · mahalanobis(val-normal 2D 공분산) · gmm2(val-normal 2D GMM −log q).
  sanity: fisher 는 product-of-p 의 단조변환 → AUROC 동일해야(다르면 버그).

출력: results/Paderborn/diag_P3_fusion/<split>_LONO<n>_s<seed>.json + aggregate_P3_s<seed>.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import norm as _norm
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from diagnose_G3a_amp_bands import (  # noqa: E402
    SPLIT_INFO, AMP_SENSITIVE, SHAPE_SENSITIVE, _score_block, spearman, safe_auroc,
)

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
CACHE_DIR = os.path.join(RESULTS_ROOT, "p2_log_window_scores")   # P-2 확정 최종 구조(N=6+log)
DIAG_G3A_DIR = os.path.join(RESULTS_ROOT, "diag_G3a_amp_bands")  # raw baseline 병합용(scheme 무관 B3)
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_P3_fusion")

# ---- 결과 확인 전 고정 하이퍼파라미터 (val-normal only) ----
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]   # weighted-z: 0.0..1.0 (0/1=단일 기준점)
GMM_N_COMPONENTS = 2
GMM_REG_COVAR = 1e-6
GMM_SEED = 0
GMM_N_INIT = 5
MAHA_RIDGE = 1e-6            # z-space 공분산 대각 릿지
P_CLIP = 1e-12              # stouffer 등 극단 p 보호


# ---- 결합 규칙 유틸 (전부 val-normal 참조, 클수록 이상) ----
def _z(x, ref):
    return (np.asarray(x, float) - float(np.mean(ref))) / (float(np.std(ref)) + 1e-8)


def _tail_p(scores, val_scores):
    v = np.sort(np.asarray(val_scores, float)); n = len(v)
    if n == 0:
        return np.ones_like(np.asarray(scores, float))
    ge = n - np.searchsorted(v, np.asarray(scores, float), side="left")
    return (ge + 1.0) / (n + 1.0)


def make_rules():
    """name -> f(sh, am, va_sh, va_am) -> score. va_*는 항상 val-normal 참조."""
    rules = {}
    rules["equal_z"] = lambda sh, am, vs, va: _z(sh, vs) + _z(am, va)
    for w in WEIGHTS:
        rules[f"wz_{w:.1f}"] = (lambda w: lambda sh, am, vs, va: w * _z(sh, vs) + (1 - w) * _z(am, va))(w)

    def fisher(sh, am, vs, va):
        return -2.0 * (np.log(_tail_p(sh, vs)) + np.log(_tail_p(am, va)))
    rules["fisher"] = fisher

    def stouffer(sh, am, vs, va):
        ps, pa = _tail_p(sh, vs), _tail_p(am, va)
        zs = _norm.ppf(np.clip(1 - ps, P_CLIP, 1 - P_CLIP))
        za = _norm.ppf(np.clip(1 - pa, P_CLIP, 1 - P_CLIP))
        return (zs + za) / np.sqrt(2.0)
    rules["stouffer"] = stouffer

    rules["tippett"] = lambda sh, am, vs, va: -np.minimum(_tail_p(sh, vs), _tail_p(am, va))
    rules["wilkinson"] = lambda sh, am, vs, va: -np.maximum(_tail_p(sh, vs), _tail_p(am, va))

    def hmp(sh, am, vs, va):
        ps, pa = _tail_p(sh, vs), _tail_p(am, va)
        return -2.0 / (1.0 / ps + 1.0 / pa)
    rules["hmp"] = hmp

    rules["ranksum"] = lambda sh, am, vs, va: (1 - _tail_p(sh, vs)) + (1 - _tail_p(am, va))
    rules["maxz"] = lambda sh, am, vs, va: np.maximum(_z(sh, vs), _z(am, va))

    def mahalanobis(sh, am, vs, va):
        # val-normal (z_sh, z_am) 2D 공분산으로 브랜치 상관 보정한 제곱 거리
        Zv = np.stack([_z(vs, vs), _z(va, va)], axis=1)     # (Nval,2), val 기준 z (평균0)
        cov = np.cov(Zv, rowvar=False) + MAHA_RIDGE * np.eye(2)
        inv = np.linalg.inv(cov)
        Z = np.stack([_z(sh, vs), _z(am, va)], axis=1)      # (N,2)
        d2 = np.einsum("ij,jk,ik->i", Z, inv, Z)            # (x-0)^T inv (x-0)
        return d2
    rules["mahalanobis"] = mahalanobis

    def gmm2(sh, am, vs, va):
        Zv = np.stack([_z(vs, vs), _z(va, va)], axis=1)
        g = GaussianMixture(n_components=GMM_N_COMPONENTS, covariance_type="full",
                            reg_covar=GMM_REG_COVAR, random_state=GMM_SEED, n_init=GMM_N_INIT)
        g.fit(Zv)
        Z = np.stack([_z(sh, vs), _z(am, va)], axis=1)
        return -g.score_samples(Z)                          # -log q (밀도 낮을수록 이상)
    rules["gmm2"] = gmm2

    def prodp(sh, am, vs, va):   # sanity: fisher 와 AUROC 동일해야
        return -(_tail_p(sh, vs) * _tail_p(am, va))
    rules["_prodp_sanity"] = prodp
    return rules


RULES = make_rules()
SINGLES = ["shape", "amp"]   # dual 판정용 단일 브랜치(raw는 외부 하한)
DEGENERATE_WZ = {"wz_0.0", "wz_1.0"}  # 단일과 동일 → 판정 제외, 기준점 표시만


def process_fold(split, lono, seed, cache_dir, diag_dir, thr_pct):
    cache_path = os.path.join(cache_dir, f"{split}_LONO{lono}_s{seed}.npz")
    if not os.path.exists(cache_path):
        print(f"    [skip] 캐시 없음: {cache_path}")
        return None
    z = np.load(cache_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    te_sh, te_am = z["te_sh"], z["te_am"]; va_sh, va_am = z["va_sh"], z["va_am"]; tr_sh, tr_am = z["tr_sh"], z["tr_am"]
    te_lab, te_ids, te_rms = z["te_lab"], z["te_ids"], z["te_rms"]
    va_ids, va_rms = z["va_ids"], z["va_rms"]; tr_ids, tr_rms = z["tr_ids"], z["tr_rms"]
    tgt = te_lab == 0
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])

    def blk(name, S_te, S_va, S_tr):
        return _score_block(name, S_te, S_va, te_lab, te_ids, te_rms, S_tr, S_va,
                            te_rms[tgt], norm_rms, norm_ids, thr_pct)

    blocks = {}
    blocks["shape"] = blk("shape", te_sh, va_sh, tr_sh)
    blocks["amp"] = blk("amp", te_am, va_am, tr_am)
    for name, f in RULES.items():
        S_te = f(te_sh, te_am, va_sh, va_am)
        S_va = f(va_sh, va_am, va_sh, va_am)
        S_tr = f(tr_sh, tr_am, va_sh, va_am)
        blocks[name] = blk(name, S_te, S_va, S_tr)

    # sanity: fisher AUROC == product-of-p AUROC
    repro_fisher_prodp = abs(blocks["fisher"]["auroc"] - blocks["_prodp_sanity"]["auroc"])

    # raw baseline 병합(diag_G3a JSON, scheme 무관 B3)
    raw = None
    g3a_path = os.path.join(diag_dir, f"{split}_LONO{lono}_s{seed}.json")
    if os.path.exists(g3a_path):
        raw = json.load(open(g3a_path)).get("raw")

    return {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": meta["fold_type"], "amp_n_bands": meta["amp_n_bands"],
        "amp_band_scheme": meta.get("amp_band_scheme", "?"),
        "blocks": blocks, "raw": raw,
        "auroc": {m: blocks[m]["auroc"] for m in blocks},
        "repro_fisher_prodp": repro_fisher_prodp,
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

    folds, max_repro = [], 0.0
    for split in args.splits:
        for lono in args.lonos:
            r = process_fold(split, lono, args.seed, args.cache_dir, args.diag_dir, args.threshold_percentile)
            if r is None:
                continue
            max_repro = max(max_repro, r["repro_fisher_prodp"])
            json.dump(r, open(os.path.join(args.out_dir, f"{split}_LONO{lono}_s{args.seed}.json"), "w"))
            folds.append(r)

    # seed 집계(간단: 규칙별 overall/fold유형/fault군/FPR)
    ok = [f for f in folds if f]
    RULE_NAMES = list(RULES.keys()) + ["shape", "amp"]

    def amean(rows, m, key="auroc"):
        vs = [f["blocks"][m][key] for f in rows if m in f["blocks"] and f["blocks"][m].get(key) is not None]
        return float(np.mean(vs)) if vs else None

    agg = {"n_folds": len(ok), "seed": args.seed, "repro_fisher_prodp_max": max_repro,
           "overall": {m: amean(ok, m) for m in RULE_NAMES},
           "raw_overall": float(np.mean([f["raw"]["block"]["auroc"] for f in ok if f.get("raw")]))
                          if any(f.get("raw") for f in ok) else None,
           "by_fold_type": {}, "normal_fpr": {}}
    for ft in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ft]
        if rows:
            agg["by_fold_type"][ft] = {m: amean(rows, m) for m in RULE_NAMES}
    agg["normal_fpr"] = {m: {"high": amean(ok, m, "high_amp_fpr"), "low": amean(ok, m, "low_amp_fpr")} for m in RULE_NAMES}
    # fault군
    fault_agg = {}
    for f in ok:
        for m in RULE_NAMES:
            for fid, d in f["blocks"][m]["per_fault_auroc"].items():
                fault_agg.setdefault(m, {}).setdefault(fid, []).append(d["auroc"])
    def grp(m, S):
        vals = [np.mean(v) for fid, v in fault_agg.get(m, {}).items() if fid in S]
        return float(np.mean(vals)) if vals else None
    agg["fault_group"] = {m: {"amp_sensitive": grp(m, AMP_SENSITIVE), "shape_sensitive": grp(m, SHAPE_SENSITIVE)} for m in RULE_NAMES}
    json.dump(agg, open(os.path.join(args.out_dir, f"aggregate_P3_s{args.seed}.json"), "w"))
    print(f"seed {args.seed}: folds={len(ok)}  fisher==prodp max|Δ|={max_repro:.2e}")
    top = sorted([(m, agg['overall'][m]) for m in RULE_NAMES if agg['overall'][m] is not None], key=lambda x: -x[1])[:6]
    print("  overall top:", ", ".join(f"{m}={v:.3f}" for m, v in top), f"| raw={agg['raw_overall']}")


if __name__ == "__main__":
    main()
