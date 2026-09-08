"""P-2 Band partition sweep — scheme별 Fusion(equal-z A / Fisher-tail B) 집계 및 ref 대비 paired 비교.

`report_P1_fisher.py`와 동일 구조. N 대신 **scheme(linear/log/energy)**을 축으로 집계한다.
diagnose_G5_tail_fusion.py 가 각 scheme에 대해 남긴 seed별 aggregate_G5_s<seed>.json + per-fold JSON을
읽어(학습·추론 없음, analysis-only) ref(기본 linear=g3a) 대비 paired 비교를 만든다.

diag 경로 규약:
  linear → results/Paderborn/diag_G5_tail_fusion (기존 G-5=g3a=N6 linear 재사용)
  그 외  → results/Paderborn/diag_P2_<scheme>_tail_fusion

산출: reports/report_P2_scheme_<tag>.md  (+ stdout 요약)
"""
import argparse
import json
import os

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT_ROOT, "results", "Paderborn")
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
LONOS = [1, 2, 3, 4, 5, 6]


def diag_dir(scheme):
    return (os.path.join(RES, "diag_G5_tail_fusion") if scheme == "linear"
            else os.path.join(RES, f"diag_P2_{scheme}_tail_fusion"))


def ms(vals):
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and x == x]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def fmt(m, s, p=3):
    return "nan" if m != m else f"{m:.{p}f}±{s:.{p}f}"


def gpath(a, *keys):
    for k in keys:
        a = a.get(k) if isinstance(a, dict) else None
        if a is None:
            return None
    return a


def load_seed_aggs(scheme, seeds):
    d = diag_dir(scheme)
    aggs, used = [], []
    for s in seeds:
        p = os.path.join(d, f"aggregate_G5_s{s}.json")
        if os.path.exists(p):
            aggs.append(json.load(open(p)))
            used.append(s)
    return aggs, used


def per_fold_metric(scheme, seeds, method):
    d = diag_dir(scheme)
    fold_vals = {}
    for split in SPLITS:
        for lono in LONOS:
            vs = []
            for s in seeds:
                p = os.path.join(d, f"{split}_LONO{lono}_s{s}.json")
                if not os.path.exists(p):
                    continue
                j = json.load(open(p))
                v = gpath(j, "auroc", method)
                if v is not None:
                    vs.append(v)
            if vs:
                fold_vals[(split, lono)] = float(np.mean(vs))
    return fold_vals


def seed_overall(scheme, seeds, method):
    aggs, used = load_seed_aggs(scheme, seeds)
    return {s: gpath(a, "overall", method) for s, a in zip(used, aggs)}


def agg_row(scheme, seeds, path_keys):
    aggs, _ = load_seed_aggs(scheme, seeds)
    out = {}
    for method in ("fusion_A", "fusion_B", "raw"):
        vals = [gpath(a, *[k if k != "<m>" else method for k in path_keys]) for a in aggs]
        out[method] = ms(vals)
    return out


def paired(scheme, ref, seeds, method):
    a = per_fold_metric(scheme, seeds, method)
    b = per_fold_metric(ref, seeds, method)
    common = sorted(set(a) & set(b))
    xa = np.array([a[k] for k in common])
    xb = np.array([b[k] for k in common])
    d = xa - xb
    res = {"n": len(common), "mean_diff": float(d.mean()) if len(d) else float("nan"),
           "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "p": None}
    if _HAVE_SCIPY and len(common) >= 1 and np.any(d != 0):
        try:
            res["p"] = float(wilcoxon(xa, xb).pvalue)
        except Exception:
            res["p"] = None
    return res


def holm(pvals):
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adj = [None] * len(pvals)
    prev = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvals[j])):
        a = min(1.0, pvals[i] * (m - rank))
        a = max(a, prev)
        prev = a
        adj[i] = a
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=["linear", "log", "energy"])
    ap.add_argument("--ref", type=str, default="linear", help="paired 기준 scheme (기본 linear=g3a)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    schemes = args.schemes
    ref = args.ref
    tag = args.tag or ("_".join(schemes) + f"_vs_{ref}")
    report_path = os.path.join(PROJECT_ROOT, "reports", f"report_P2_scheme_{tag}.md")

    L = ["# P-2 Band partition — Fisher-tail(B)/equal-z(A) scheme별 집계 & paired 비교", ""]
    L.append(f"비교 scheme={schemes}, paired 기준={ref}. seed {args.seeds}. analysis-only(캐시 재사용, 학습·추론 없음).")
    L.append(f"linear=g3a(N=6) 재사용. paired = 각 fold seed평균 AUROC로 24-fold pairing → Wilcoxon."
             + ("" if _HAVE_SCIPY else "  ⚠️ scipy 없음: p-value 생략."))
    L.append("")

    L.append("## 완료 상태")
    for sc in schemes + ([ref] if ref not in schemes else []):
        aggs, used = load_seed_aggs(sc, args.seeds)
        nfold = per_fold_metric(sc, args.seeds, "fusion_B")
        L.append(f"- {sc}: seed aggregate {len(used)}개 {used}, per-fold(seed평균) {len(nfold)}/24")
    L.append("")

    METHS = [("fusion_B", "Fisher-tail(B)"), ("fusion_A", "equal-z(A)")]

    def block(title, path_keys):
        L.append(f"## {title}")
        L.append("")
        L.append("| scheme | " + " | ".join(lbl for _, lbl in METHS) + " |")
        L.append("|---|" + "---|" * len(METHS))
        for sc in schemes:
            row = agg_row(sc, args.seeds, path_keys)
            cells = [fmt(*row[m]) for m, _ in METHS]
            mark = " (ref)" if sc == ref else ""
            L.append(f"| {sc}{mark} | " + " | ".join(cells) + " |")
        L.append("")

    block("전체 AUROC", ["overall", "<m>"])
    block("zero-support 외삽", ["by_fold_type", "zero-support", "auroc", "<m>"])
    block("compositional", ["by_fold_type", "compositional", "auroc", "<m>"])
    block("amp-sensitive fault", ["fault_group", "amp_sensitive", "<m>"])
    block("shape-sensitive fault", ["fault_group", "shape_sensitive", "<m>"])
    block("정상 FPR 고진폭", ["normal_fpr", "<m>", "high"])
    block("정상 FPR 저진폭", ["normal_fpr", "<m>", "low"])

    L.append("## seed별 전체 AUROC (Fisher B / equal-z A)")
    L.append("")
    L.append("| scheme | " + " | ".join(f"s{s}" for s in args.seeds) + " |")
    L.append("|---|" + "---|" * len(args.seeds))
    for sc in schemes:
        ob = seed_overall(sc, args.seeds, "fusion_B")
        oa = seed_overall(sc, args.seeds, "fusion_A")
        cells = [f"B {ob.get(s):.3f}/A {oa.get(s):.3f}" if ob.get(s) is not None else "-" for s in args.seeds]
        L.append(f"| {sc} | " + " | ".join(cells) + " |")
    L.append("")

    L.append(f"## 24-fold paired Wilcoxon signed-rank test (scheme vs ref={ref})")
    L.append("")
    L.append(f"> 각 fold의 seed평균 Fisher(B)/equal-z(A) AUROC로 24-fold pairing 후 Wilcoxon(양측). "
             f"mean_diff>0 이면 scheme이 ref보다 높음. Holm-Bonferroni 보정 p도 병기. 채택 판정은 P-G2.")
    L.append("")
    for m, lbl in METHS:
        cand = [sc for sc in schemes if sc != ref]
        raws = [paired(sc, ref, args.seeds, m) for sc in cand]
        holm_p = holm([r["p"] for r in raws])
        L.append(f"### {lbl}")
        L.append("| scheme | n_fold | mean_diff(scheme−ref) | pos/neg | p(raw) | p(Holm) |")
        L.append("|---|---|---|---|---|---|")
        for sc, r, hp in zip(cand, raws, holm_p):
            pstr = "nan" if r["p"] is None else f"{r['p']:.4f}"
            hstr = "nan" if hp is None else f"{hp:.4f}"
            L.append(f"| {sc} | {r['n']} | {r['mean_diff']:+.4f} | {r['n_pos']}/{r['n_neg']} | {pstr} | {hstr} |")
        L.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {report_path}")
    for sc in schemes:
        ob = agg_row(sc, args.seeds, ["overall", "<m>"])
        print(f"  {sc}: B={fmt(*ob['fusion_B'])}  A={fmt(*ob['fusion_A'])}")


if __name__ == "__main__":
    main()
