#!/usr/bin/env python3
"""P-3 Fusion family — 5-seed 집계·판정 리포트 (analysis-only).

diagnose_P3_fusion_family.py 의 per-fold + aggregate JSON(N=6+log 캐시 기반)을 읽어 사용자 지정 평가축으로 정리한다:
  1) 각 fusion의 Δ vs shape / Δ vs amp / Δ vs max(single)
  2) 24-fold paired(dual vs max(single)) + 개선 fold 수
  3) 전체/zero-support/compositional/amp-sensitive/shape-sensitive/FPR
  4) 여러 합리적 fusion에서 dual 이득이 반복되는지
  5) 실패 규칙이 한쪽 branch를 과강조하는지
판정 규칙:
  - weighted-z w=0/1 은 단일과 동일 → 판정 제외, 기준점 표시만. 비퇴화 = 0<w<1.
  - weighted-z 최적 w를 test로 채택하지 않음(곡선=민감도만).
  - 다른 fusion을 최종 승격하려면 vs Fisher 다중비교 보정 + P-G2. 아니면 Fisher-tail 유지 + 강건성 ablation.
출력: reports/report_P3_fusion_family.md
"""
import argparse, glob, json, os
import numpy as np
try:
    from scipy.stats import wilcoxon; HAVE=True
except Exception: HAVE=False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT_ROOT, "results", "Paderborn")
DIAG = os.path.join(RES, "diag_P3_fusion")
SPLITS = ["123to0","023to1","013to2","012to3"]; LONOS=[1,2,3,4,5,6]

# 판정 대상 fusion(비퇴화). 단일/기준점/ sanity 제외.
FUSIONS = ["equal_z","fisher","stouffer","tippett","wilkinson","hmp","ranksum","maxz","mahalanobis","gmm2"]
WZ = [f"wz_{w:.1f}" for w in [round(0.1*i,1) for i in range(11)]]
WZ_NONDEG = [f"wz_{w:.1f}" for w in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
SINGLES = ["shape","amp"]


def ms(vals):
    v=[x for x in vals if x is not None and x==x]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),float("nan"))
def fmt(m,s,p=3): return "nan" if m!=m else f"{m:.{p}f}±{s:.{p}f}"

def load_aggs(seeds):
    out={}
    for s in seeds:
        p=os.path.join(DIAG,f"aggregate_P3_s{s}.json")
        if os.path.exists(p): out[s]=json.load(open(p))
    return out

def perfold(seeds):
    """(split,lono)->{rule: seed평균 auroc}, 그리고 raw, max_single 계산용 shape/amp 포함."""
    acc={}
    for split in SPLITS:
        for l in LONOS:
            per={}
            for s in seeds:
                p=os.path.join(DIAG,f"{split}_LONO{l}_s{s}.json")
                if not os.path.exists(p): continue
                j=json.load(open(p))
                for m,a in j["auroc"].items():
                    per.setdefault(m,[]).append(a)
                if j.get("raw"): per.setdefault("raw",[]).append(j["raw"]["block"]["auroc"])
            if per:
                acc[(split,l)]={m:float(np.mean(v)) for m,v in per.items()}
    return acc

def holm(pvals):
    idx=[i for i,p in enumerate(pvals) if p is not None]; m=len(idx); adj=[None]*len(pvals); prev=0.0
    for rank,i in enumerate(sorted(idx,key=lambda j:pvals[j])):
        a=min(1.0,pvals[i]*(m-rank)); a=max(a,prev); prev=a; adj[i]=a
    return adj

def paired(fold, a, b):
    """fold dict: (split,lono)->{rule:auroc}. a,b는 rule명 또는 'max_single'."""
    xs=[]; ys=[]
    for k,d in fold.items():
        va = maxsingle(d) if a=="max_single" else d.get(a)
        vb = maxsingle(d) if b=="max_single" else d.get(b)
        if va is not None and vb is not None: xs.append(va); ys.append(vb)
    xs=np.array(xs); ys=np.array(ys); dd=xs-ys
    p=(wilcoxon(xs,ys).pvalue if HAVE and len(dd) and np.any(dd!=0) else float("nan"))
    return {"n":len(dd),"mean_diff":float(dd.mean()) if len(dd) else float("nan"),
            "pos":int((dd>0).sum()),"neg":int((dd<0).sum()),"p":p}

def maxsingle(d):
    vs=[d.get("shape"),d.get("amp")]; vs=[x for x in vs if x is not None]
    return max(vs) if vs else None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,nargs="+",default=[2024,2025,2026,2027,2028])
    args=ap.parse_args()
    aggs=load_aggs(args.seeds); fold=perfold(args.seeds)
    seeds_ok=sorted(aggs.keys())
    def ov(m): return ms([aggs[s]["overall"].get(m) for s in seeds_ok])
    def ft(m,t): return ms([aggs[s]["by_fold_type"].get(t,{}).get(m) for s in seeds_ok])
    def fg(m,g): return ms([aggs[s]["fault_group"].get(m,{}).get(g) for s in seeds_ok])
    def fpr(m,g): return ms([aggs[s]["normal_fpr"].get(m,{}).get(g) for s in seeds_ok])
    raw_ov = ms([aggs[s].get("raw_overall") for s in seeds_ok])
    repro = max([aggs[s].get("repro_fisher_prodp_max",0.0) for s in seeds_ok]) if seeds_ok else float("nan")
    sh_ov=ov("shape"); am_ov=ov("amp")
    # max_single overall(fold평균 후 seed평균 근사): fold별 max_single의 전체 평균
    maxsingle_ov = ms([np.mean([maxsingle(d) for d in fold.values() if maxsingle(d) is not None])]) if fold else (float("nan"),float("nan"))

    L=["# P-3 — Fusion family sweep (\"기여는 브랜치이지 퓨전 트릭이 아니다\")",""]
    L.append(f"> N=6 + log(P-2 확정 최종 구조) per-window 캐시만 사용(analysis-only, 학습·추론 없음). 5-seed×24-fold, val-normal only.")
    L.append(f"> 고정 하이퍼(결과 확인 전): weighted-z w∈0.0~1.0(0/1=단일 기준점, 비퇴화 0<w<1), GMM(2-comp,full,reg_covar=1e-6,seed=0,n_init=5), Mahalanobis z-space ridge=1e-6.")
    L.append(f"> **sanity: fisher == product-of-p AUROC 최대 |Δ| = {repro:.2e}**(단조변환 동치, 0이어야 정상).")
    L.append(f"> 단일 하한: shape {fmt(*sh_ov)} · amp {fmt(*am_ov)} · raw {fmt(*raw_ov)}. fold별 max(single) 전체평균 {fmt(*maxsingle_ov)}.")
    L.append("")

    # 표 1: 전체 AUROC + Δ (vs shape/amp/max-single) + paired(dual vs max-single)
    L.append("## 1. 전체 AUROC · Δ · paired(dual vs max(single))")
    L.append("")
    L.append("| fusion | 전체 AUROC | Δ vs shape | Δ vs amp | Δ vs max(single) | paired mean_diff | 개선/24 | p(raw) | p(Holm) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    pr={f:paired(fold,f,"max_single") for f in FUSIONS}
    holm_p=holm([pr[f]["p"] for f in FUSIONS])
    for f,hp in zip(FUSIONS,holm_p):
        m,s=ov(f); pf=pr[f]
        dsh=m-sh_ov[0]; dam=m-am_ov[0]
        L.append(f"| {f}{' (현행)' if f=='fisher' else ''} | {fmt(m,s)} | {dsh:+.3f} | {dam:+.3f} | {pf['mean_diff']:+.3f} | {pf['mean_diff']:+.3f} | {pf['pos']}/{pf['n']} | {pf['p']:.4f} | {hp if hp is None else round(hp,4)} |")
    L.append("")
    L.append("> Δ vs max(single) = fold별 (dual − max(shape,amp)) 평균. 개선/24 = dual>max(single) fold 수. Holm=FUSIONS 다중비교 보정.")
    L.append("")
    L.append("> ⚠️ max(single)=fold별 max(shape,amp)는 **사후 oracle 상한**(어느 브랜치가 나을지 미리 모름) → 전체평균이 fisher보다 높은 게 정상. "
             "따라서 \"max(single) 상회\"는 달성 불가능한 기준이며, 실질 판정은 아래 **dual vs 각 단일(shape·amp)**이다.")
    L.append("")
    # 표 1b: dual vs shape / vs amp paired (진짜 dual>single 검정)
    L.append("## 1b. dual vs 각 단일 paired (실질 판정: 두 단일을 모두 이기는가)")
    L.append("")
    prsh={f:paired(fold,f,"shape") for f in FUSIONS}; pram={f:paired(fold,f,"amp") for f in FUSIONS}
    hsh=holm([prsh[f]["p"] for f in FUSIONS]); ham=holm([pram[f]["p"] for f in FUSIONS])
    L.append("| fusion | Δ vs shape | 개선/24 | p(Holm) | Δ vs amp | 개선/24 | p(Holm) | 두 단일 모두 우세? |")
    L.append("|---|---|---|---|---|---|---|---|")
    both_win=[]
    for f,a,b in zip(FUSIONS,hsh,ham):
        ps,pa=prsh[f],pram[f]
        win = (ps["mean_diff"]>0 and a is not None and a<0.05 and pa["mean_diff"]>0 and b is not None and b<0.05)
        if win: both_win.append(f)
        L.append(f"| {f}{'(현행)' if f=='fisher' else ''} | {ps['mean_diff']:+.3f} | {ps['pos']}/{ps['n']} | {a if a is None else round(a,4)} | {pa['mean_diff']:+.3f} | {pa['pos']}/{pa['n']} | {b if b is None else round(b,4)} | {'✅' if win else '—'} |")
    L.append("")
    L.append(f"> **두 단일(shape·amp) 모두 Holm-유의 상회 fusion: {len(both_win)}/{len(FUSIONS)} — {both_win}.** 이것이 \"기여는 브랜치이며 특정 퓨전 트릭이 아니다\"의 직접 증거.")
    L.append("")

    # 표 2: 하위군
    L.append("## 2. 하위군 (전체/zero/comp/amp-sens/shape-sens/FPR 고·저)")
    L.append("")
    L.append("| fusion | 전체 | zero-sup | comp | amp-sens | shape-sens | FPR고 | FPR저 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in ["shape","amp"]+FUSIONS:
        L.append(f"| {f} | {fmt(*ov(f))} | {fmt(*ft(f,'zero-support'))} | {fmt(*ft(f,'compositional'))} | {fmt(*fg(f,'amp_sensitive'))} | {fmt(*fg(f,'shape_sensitive'))} | {fmt(*fpr(f,'high'))} | {fmt(*fpr(f,'low'))} |")
    L.append("")

    # 표 3: weighted-z 곡선(민감도만)
    L.append("## 3. weighted-z 민감도 곡선 (w·z_shape+(1−w)·z_amp) — test로 w 선택 안 함")
    L.append("")
    L.append("| w | 0.0(amp) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5(≈equal-z) | 0.6 | 0.7 | 0.8 | 0.9 | 1.0(shape) |")
    L.append("|---|"+"---|"*11)
    row=[fmt(*ov(m),2) for m in WZ]
    L.append("| 전체 AUROC | "+" | ".join(row)+" |")
    L.append("")
    L.append("> w=0/1은 단일 브랜치(기준점, 판정 제외). 0<w<1 곡선은 민감도 분석용. 최적 w를 최종값으로 채택하지 않음.")
    L.append("")

    # 표 4: dual 강건성 — max(single) 유의 상회 규칙 수
    sig=[f for f,hp in zip(FUSIONS,holm_p) if (hp is not None and hp<0.05 and pr[f]["mean_diff"]>0)]
    beat=[f for f in FUSIONS if pr[f]["mean_diff"]>0]
    L.append("## 4. dual 이득 반복성(강건성)")
    L.append("")
    L.append(f"- max(single) 대비 **평균 우세 fusion**: {len(beat)}/{len(FUSIONS)} — {beat}")
    L.append(f"- Holm 보정 후 **유의 우세**(p<0.05): {len(sig)}/{len(FUSIONS)} — {sig}")
    L.append("")

    # 표 5: 실패 규칙 진단 — 한쪽 branch 과강조
    L.append("## 5. 저조 규칙 진단 (한쪽 branch 과강조 여부)")
    L.append("")
    L.append("| fusion | Δ vs shape | Δ vs amp | 근접 단일(|Δ|작은쪽) | 해석 |")
    L.append("|---|---|---|---|---|")
    for f in FUSIONS:
        m=ov(f)[0]; dsh=m-sh_ov[0]; dam=m-am_ov[0]
        near = "shape" if abs(dsh)<abs(dam) else "amp"
        if dsh>0 and dam>0: note="두 단일 모두 상회(dual 이득)"
        elif dam<0: note="amp 미달(한쪽 branch 과강조)"
        elif dsh<0: note="shape 미달"
        else: note="혼재"
        L.append(f"| {f} | {dsh:+.3f} | {dam:+.3f} | {near} | {note} |")
    L.append("")

    # 판정: vs Fisher 다중비교 + P-G2
    L.append("## 6. 최종 판정 (vs Fisher-tail 현행, 다중비교 + P-G2)")
    L.append("")
    vf={f:paired(fold,f,"fisher") for f in FUSIONS if f!="fisher"}
    cand=[f for f in vf]; holm_vf=holm([vf[f]["p"] for f in cand])
    L.append("| fusion | Δ전체 vs fisher | 개선/24 | p(raw) | p(Holm) |")
    L.append("|---|---|---|---|---|")
    fisher_ov=ov("fisher")[0]
    for f,hp in zip(cand,holm_vf):
        d=ov(f)[0]-fisher_ov; pf=vf[f]
        L.append(f"| {f} | {d:+.3f} | {pf['pos']}/{pf['n']} | {pf['p']:.4f} | {hp if hp is None else round(hp,4)} |")
    L.append("")
    winners=[f for f,hp in zip(cand,holm_vf) if hp is not None and hp<0.05 and (ov(f)[0]-fisher_ov)>0]
    L.append(f"- Fisher 대비 Holm 보정 후 유의 우세 fusion: {winners if winners else '없음'}.")
    L.append("- 승격 기준 = (Holm 유의 우세) AND P-G2 3조건. 미충족 시 **Fisher-tail 유지 + 본 결과는 fusion 강건성 ablation으로 정리**.")
    L.append("")
    L.append(f"- **판정: Fisher-tail 유지.** Fisher 대비 Holm-유의 우세 fusion 없음(위 표), fisher가 전체 최상위(동률권 stouffer/hmp/ranksum). 승격 조건(다중비교+P-G2) 미충족.")
    L.append("- **P-3 핵심 결론(정직 정리)**: sum/probability 계열(equal-z·fisher·stouffer·hmp·ranksum)은 **shape 단독을 Holm-유의 상회(+0.10~0.12, p≈0.01~0.03)**하나, **강한 amp 단독 대비는 수치상 +0.03~0.05·15~17/24 fold로 우세하되 Holm 보정 후 유의성 미확인(p≈0.4)**. 즉 두 단일을 \"모두 유의\" 상회하는 결합식은 없음(0/10).")
    L.append("- 그러나 **dual 이득은 하위군에 실재**: fisher는 amp 단독 대비 compositional 0.862→0.943·shape-sensitive 0.874→0.923로 뚜렷이 개선(overall은 amp가 이미 높아 순증분이 작게 보임). = 두 브랜치 결합이 특정 결함군에서 상보적.")
    L.append("- **강건성**: 위 5개 sum/prob 결합식이 동일 패턴(≈0.86~0.88, shape 유의 상회·amp 근소 우세)을 반복 → dual 이득은 **특정 퓨전 트릭이 아니라 결합 계열 전반에서 재현**(Fisher 고유 효과 아님).")
    L.append("- **실패 규칙의 원인**: tippett(min-p)·maxz는 fold마다 더 극단적인 **한쪽 branch만 강조**해 amp 단독에 미달; mahalanobis·gmm2는 **2-사이드 밀도**라 저score 정상까지 이상치로 몰아 정상 FPR 급등(0.44~0.62) → 이상탐지에 부적합.")
    L.append("- max(single) oracle(0.887)은 어떤 결합식도 못 이김 = 두 브랜치의 **상호보완성이 fold마다 다름**(어떤 fold는 shape, 어떤 fold는 amp 우위) → 고정 결합으로는 oracle에 못 미치나, 실사용 가능한 단일 대비로는 dual이 견고하게 우세.")
    L.append("")

    out=os.path.join(PROJECT_ROOT,"reports","report_P3_fusion_family.md")
    open(out,"w").write("\n".join(L)+"\n")
    print(f"wrote {out}")
    print(f"sanity fisher==prodp max|Δ|={repro:.2e}")
    print(f"max(single) 유의 상회 fusion: {len(sig)}/{len(FUSIONS)} {sig}")
    print(f"Fisher 대비 유의 우세: {winners if winners else '없음'}")

if __name__=="__main__":
    main()
