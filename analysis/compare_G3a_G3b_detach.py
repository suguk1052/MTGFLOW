#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""작업 G-3b: detach ablation의 paired 비교 (analysis-only, 재추론 없음).

G-3a(joint, amp gradient가 shape 인코더로 흐름)와 G-3b(detach, amp gradient 차단)의
per-fold 진단 JSON을 같은 seed에서 맞대어 S_shape가 detach로 **명확히 하락**하는지 검증한다.
(하락 = "amp auxiliary-task의 gradient가 S_shape gain의 원인"이라는 가설 입증 → 5-seed 확장 근거.)

입력: diagnose_G3a_amp_bands.py가 만든 per-fold JSON
  - G-3a: results/Paderborn/diag_G3a_amp_bands/<split>_LONO<n>_s<seed>.json (기존)
  - G-3b: results/Paderborn/diag_G3b_amp_bands/<split>_LONO<n>_s<seed>.json (--g3a_prefix g3b로 생성)
두 진단은 동일 스키마(키 "g1")를 쓰므로 기존 aggregate()를 그대로 재사용한다.

출력:
  - reports/report_G3b_detach_screening.md
  - results/Paderborn/diag_G3b_amp_bands/summary_g3b_vs_g3a_s<seed>.json

사용:
  python analysis/compare_G3a_G3b_detach.py --seed 2026
"""
import os
import sys
import json
import glob
import argparse

# 같은 폴더의 diagnose 모듈에서 집계 로직·상수를 재사용(중복 구현 금지).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_G3a_amp_bands import aggregate, RESULTS_ROOT, PROJECT_ROOT  # noqa: E402


def load_folds(diag_dir, seed):
    """diag_dir에서 seed에 해당하는 per-fold JSON들을 로드(aggregate/summary 파일 제외)."""
    folds = []
    for path in sorted(glob.glob(os.path.join(diag_dir, f"*_LONO*_s{seed}.json"))):
        base = os.path.basename(path)
        if base.startswith("aggregate") or base.startswith("summary"):
            continue
        with open(path) as f:
            folds.append(json.load(f))
    return folds


def _fold_key(f):
    return (f["split"], f["lono"])


def _fmt(a, b):
    """a(g3a) → b(g3b) Δ 문자열."""
    if a != a or b != b:  # nan
        return f"{a:.3f} → {b:.3f} (Δ nan)"
    return f"{a:.3f} → {b:.3f} (Δ {b - a:+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--g3a_dir", type=str,
                    default=os.path.join(RESULTS_ROOT, "diag_G3a_amp_bands"))
    ap.add_argument("--g3b_dir", type=str,
                    default=os.path.join(RESULTS_ROOT, "diag_G3b_amp_bands"))
    ap.add_argument("--report", type=str,
                    default=os.path.join(PROJECT_ROOT, "reports", "report_G3b_detach_screening.md"))
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    fa = load_folds(args.g3a_dir, args.seed)
    fb = load_folds(args.g3b_dir, args.seed)
    if not fa:
        raise SystemExit(f"[에러] G-3a per-fold JSON 없음: {args.g3a_dir} (seed {args.seed})")
    if not fb:
        raise SystemExit(f"[에러] G-3b per-fold JSON 없음: {args.g3b_dir} (seed {args.seed})")

    # 같은 fold만 paired로 맞댄다(교집합).
    ka = {_fold_key(f): f for f in fa}
    kb = {_fold_key(f): f for f in fb}
    common = sorted(set(ka) & set(kb))
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))

    fa_c = [ka[k] for k in common]
    fb_c = [kb[k] for k in common]
    agg_a = aggregate(fa_c)
    agg_b = aggregate(fb_c)

    # ---- per-fold paired Δ(S_shape) ----
    per_fold = []
    n_drop = n_drop_high = n_high = 0
    for k in common:
        a, b = ka[k], kb[k]
        sa = a["g1"]["auroc_shape"]
        sb = b["g1"]["auroc_shape"]
        grp = a.get("target_amp_group", "?")
        d = sb - sa
        if grp == "high":
            n_high += 1
            if d < 0:
                n_drop_high += 1
        if d < 0:
            n_drop += 1
        per_fold.append({
            "split": k[0], "lono": k[1], "fold_type": a["fold_type"],
            "target_amp_group": grp,
            "shape_g3a": sa, "shape_g3b": sb, "delta_shape": d,
        })

    # ---- 요약 지표(S_shape 중심; total/amp/raw는 참고) ----
    def dd(da, db, path):
        cur_a, cur_b = da, db
        for p in path:
            cur_a = cur_a.get(p, float("nan")) if isinstance(cur_a, dict) else float("nan")
            cur_b = cur_b.get(p, float("nan")) if isinstance(cur_b, dict) else float("nan")
        return cur_a, cur_b

    summary = {
        "seed": args.seed,
        "n_folds_common": len(common),
        "folds_only_g3a": only_a, "folds_only_g3b": only_b,
        "overall": {
            "shape": {"g3a": agg_a["overall"]["auroc_shape"], "g3b": agg_b["overall"]["auroc_shape"]},
            "total": {"g3a": agg_a["overall"]["auroc_total"], "g3b": agg_b["overall"]["auroc_total"]},
            "amp":   {"g3a": agg_a["overall"]["auroc_amp"],   "g3b": agg_b["overall"]["auroc_amp"]},
            "raw":   {"g3a": agg_a["overall"]["auroc_raw"],   "g3b": agg_b["overall"]["auroc_raw"]},
        },
        "by_fold_type_shape": {
            ft: {"g3a": agg_a["by_fold_type"].get(ft, {}).get("auroc_shape", float("nan")),
                 "g3b": agg_b["by_fold_type"].get(ft, {}).get("auroc_shape", float("nan"))}
            for ft in ("zero-support", "compositional")
        },
        "fault_group_shape": {
            grp: {"g3a": agg_a["fault_group"][grp]["auroc_shape"],
                  "g3b": agg_b["fault_group"][grp]["auroc_shape"]}
            for grp in ("amp_sensitive", "shape_sensitive")
        },
        "normal_fpr_shape": {},  # 아래에서 blocks.shape FPR로 채움
        "per_fold": per_fold,
        "drop_counts": {
            "n_folds": len(common), "n_shape_drop": n_drop,
            "n_high_amp": n_high, "n_high_amp_drop": n_drop_high,
        },
    }

    # 정상 FPR(S_shape 블록)은 aggregate에 blocks.shape 단위로 있으나 overall 요약엔 없음 → 직접 평균.
    def _shape_fpr(folds, key):
        vals = [f["g1"]["blocks"]["shape"][key] for f in folds
                if f["g1"]["blocks"]["shape"].get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary["normal_fpr_shape"] = {
        "high": {"g3a": _shape_fpr(fa_c, "high_amp_fpr"), "g3b": _shape_fpr(fb_c, "high_amp_fpr")},
        "low":  {"g3a": _shape_fpr(fa_c, "low_amp_fpr"),  "g3b": _shape_fpr(fb_c, "low_amp_fpr")},
    }

    # ---- GO/NO-GO 휴리스틱(참고용; 최종 판단은 사용자) ----
    ov_a = agg_a["overall"]["auroc_shape"]
    ov_b = agg_b["overall"]["auroc_shape"]
    delta_overall = ov_b - ov_a
    # 명확한 하락 = overall S_shape가 눈에 띄게 내려가고(예: -0.02 이하), fold 방향이 다수 하락.
    clear_drop = (delta_overall <= -0.02) and (n_drop >= len(common) * 0.6)
    verdict = "GO(5-seed 확장 후보)" if clear_drop else "NO-GO(1-seed 종료 후보)"
    summary["verdict_heuristic"] = {
        "delta_overall_shape": delta_overall,
        "clear_drop": clear_drop,
        "verdict": verdict,
        "note": "휴리스틱일 뿐 — cherry-pick 금지, 최종 판단은 리포트 검토 후 사용자 승인.",
    }

    os.makedirs(args.g3b_dir, exist_ok=True)
    out_json = os.path.join(args.g3b_dir, f"summary_g3b_vs_g3a_s{args.seed}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[저장] {out_json}")

    # ---- 리포트 ----
    if not args.no_report:
        L = []
        L.append(f"# G-3b detach ablation — seed {args.seed} paired 비교 (S_shape 중심)\n")
        L.append(f"- paired fold 수: **{len(common)}** (g3a-only {only_a}, g3b-only {only_b})")
        L.append(f"- 판정 휴리스틱: **{verdict}** "
                 f"(overall S_shape Δ={delta_overall:+.3f}, 하락 fold {n_drop}/{len(common)}, "
                 f"고진폭 fold 하락 {n_drop_high}/{n_high})")
        L.append(f"- raw baseline(동일 checkpoint): g3a {agg_a['overall']['auroc_raw']:.3f} / "
                 f"g3b {agg_b['overall']['auroc_raw']:.3f} (동일해야 정상 — 파이프라인 검증)\n")

        L.append("## Overall AUROC (g3a → g3b)")
        L.append(f"- **S_shape**: {_fmt(ov_a, ov_b)}  ← 최우선 지표")
        L.append(f"- S_total: {_fmt(agg_a['overall']['auroc_total'], agg_b['overall']['auroc_total'])}")
        L.append(f"- S_amp:   {_fmt(agg_a['overall']['auroc_amp'], agg_b['overall']['auroc_amp'])}\n")

        L.append("## S_shape by fold type")
        for ft in ("zero-support", "compositional"):
            a = agg_a["by_fold_type"].get(ft, {}).get("auroc_shape", float("nan"))
            b = agg_b["by_fold_type"].get(ft, {}).get("auroc_shape", float("nan"))
            L.append(f"- {ft}: {_fmt(a, b)}")
        L.append("")

        L.append("## S_shape by fault group")
        for grp in ("amp_sensitive", "shape_sensitive"):
            a = agg_a["fault_group"][grp]["auroc_shape"]
            b = agg_b["fault_group"][grp]["auroc_shape"]
            L.append(f"- {grp}: {_fmt(a, b)}")
        L.append("")

        L.append("## 정상 FPR (S_shape block)")
        nf = summary["normal_fpr_shape"]
        L.append(f"- 고진폭: {_fmt(nf['high']['g3a'], nf['high']['g3b'])}")
        L.append(f"- 저진폭: {_fmt(nf['low']['g3a'], nf['low']['g3b'])}\n")

        L.append("## per-fold S_shape (g3a → g3b)")
        L.append("| split | LONO | fold_type | amp_grp | g3a | g3b | Δ |")
        L.append("|---|---|---|---|---|---|---|")
        for r in per_fold:
            L.append(f"| {r['split']} | {r['lono']} | {r['fold_type']} | {r['target_amp_group']} | "
                     f"{r['shape_g3a']:.3f} | {r['shape_g3b']:.3f} | {r['delta_shape']:+.3f} |")
        L.append("")
        L.append("> 판정 규칙(TODO): detach에서 S_shape가 **명확히 하락**하면 auxiliary-task 가설 입증 → 5-seed 확장. "
                 "차이가 작으면 가설 미지지 → 1-seed에서 G-3b 종료. 최종 판단은 사용자 승인.")

        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"[저장] {args.report}")

    # 콘솔 요약
    print(f"\n=== G-3b vs G-3a (seed {args.seed}) ===")
    print(f"overall S_shape: {_fmt(ov_a, ov_b)}")
    print(f"하락 fold: {n_drop}/{len(common)} (고진폭 {n_drop_high}/{n_high})")
    print(f"판정 휴리스틱: {verdict}")


if __name__ == "__main__":
    main()
