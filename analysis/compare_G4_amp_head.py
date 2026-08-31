#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""작업 G-4: Amp Head architecture ablation의 paired 비교 (analysis-only, 재추론 없음).

G-3a(baseline head: Linear→ReLU→Linear, d=32)와 두 후보를 같은 seed에서 3열로 맞댄다:
  - B(wider head): hidden d 32→64 (--amp_branch_hidden 64, 코드변경 없음)
  - C(normalized head): amp head 맨 앞 LayerNorm(hidden_size) 추가 (--amp_head_norm)

G-3b compare는 S_shape 중심이었으나 G-4 primary는 **S_total overall + amp/shape-sensitive 균형**이다
(TODO §G-4). secondary = S_shape · zero-support/compositional · 정상 FPR(고/저진폭) · ρ(RMS,S_total).
raw baseline은 세 열 모두 동일해야 한다(동일 B3 checkpoint 재추론 → 파이프라인 검증).

입력: diagnose_G3a_amp_bands.py가 만든 per-fold JSON(동일 스키마, 키 "g1")
  - G-3a: results/Paderborn/diag_G3a_amp_bands/<split>_LONO<n>_s<seed>.json (기존 재사용)
  - B:    results/Paderborn/diag_G4b_amp_bands/<split>_LONO<n>_s<seed>.json (--g3a_prefix g4b 로 생성)
  - C:    results/Paderborn/diag_G4c_amp_bands/<split>_LONO<n>_s<seed>.json (--g3a_prefix g4c 로 생성)

출력:
  - reports/report_G4_amp_head_screening.md
  - results/Paderborn/diag_G4_amp_head/summary_g4_vs_g3a_s<seed>.json

사용:
  python analysis/compare_G4_amp_head.py --seed 2026
"""
import os
import sys
import json
import glob
import argparse

# 같은 폴더의 diagnose 모듈에서 집계 로직·상수를 재사용(중복 구현 금지).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_G3a_amp_bands import aggregate, RESULTS_ROOT, PROJECT_ROOT  # noqa: E402

# 판정 임계: 1-seed screening이므로 seed 분산(deepdive std ~0.05–0.07) 이내 변화는 "미개선".
SEED_NOISE = 0.02      # S_total 개선이 이 이상이어야 신호로 간주(가드레일: cherry-pick 금지)
COLLAPSE_TOL = 0.02    # fault군이 이 이상 하락하면 "붕괴"(한쪽 희생 = trade-off 재발)


def load_folds(diag_dir, seed):
    """diag_dir에서 seed에 해당하는 per-fold JSON들을 로드(aggregate/summary 파일 제외)."""
    folds = []
    if not diag_dir or not os.path.isdir(diag_dir):
        return folds
    for path in sorted(glob.glob(os.path.join(diag_dir, f"*_LONO*_s{seed}.json"))):
        base = os.path.basename(path)
        if base.startswith("aggregate") or base.startswith("summary"):
            continue
        with open(path) as f:
            folds.append(json.load(f))
    return folds


def _fold_key(f):
    return (f["split"], f["lono"])


def _overall_rho_total(folds):
    """overall ρ(RMS, S_total) = per-fold blocks.total.rho_rms_score 평균."""
    vals = [f["g1"]["blocks"]["total"]["rho_rms_score"] for f in folds
            if f["g1"]["blocks"]["total"].get("rho_rms_score") is not None]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _fmt(base, cand):
    """base → cand Δ 문자열. cand가 None이면 '—'."""
    if cand is None or base != base or cand != cand:  # None/nan
        b = f"{base:.3f}" if base == base else "nan"
        return f"{b} → — "
    return f"{base:.3f} → {cand:.3f} (Δ {cand - base:+.3f})"


def _metric_rows(agg, folds):
    """리포트/summary에 쓸 지표를 한 dict로 평탄화."""
    ov = agg["overall"]
    fg = agg["fault_group"]
    bt = agg["by_fold_type"]
    tn = agg["total_normal"]
    return {
        # primary
        "S_total": ov["auroc_total"],
        # fault군 동시보존(총점 기준)
        "amp_sensitive_total": fg["amp_sensitive"]["auroc_total"],
        "shape_sensitive_total": fg["shape_sensitive"]["auroc_total"],
        # secondary
        "S_shape": ov["auroc_shape"],
        "S_amp": ov["auroc_amp"],
        "zero_support_total": bt.get("zero-support", {}).get("auroc_total", float("nan")),
        "compositional_total": bt.get("compositional", {}).get("auroc_total", float("nan")),
        "normal_fpr_total_high": tn["high_fpr"],
        "normal_fpr_total_low": tn["low_fpr"],
        "rho_rms_total": _overall_rho_total(folds),
        # 파이프라인 검증(세 열 동일해야)
        "raw": ov["auroc_raw"],
    }


def _verdict(base_m, cand_m):
    """후보 GO/NO-GO 휴리스틱(참고용, 최종 판단은 사용자).
    GO 조건: S_total이 seed noise 초과 상승 AND 두 fault군 모두 비붕괴."""
    d_total = cand_m["S_total"] - base_m["S_total"]
    d_amp = cand_m["amp_sensitive_total"] - base_m["amp_sensitive_total"]
    d_shape = cand_m["shape_sensitive_total"] - base_m["shape_sensitive_total"]
    no_collapse = (d_amp >= -COLLAPSE_TOL) and (d_shape >= -COLLAPSE_TOL)
    go = (d_total >= SEED_NOISE) and no_collapse
    return {
        "delta_S_total": d_total,
        "delta_amp_sensitive": d_amp,
        "delta_shape_sensitive": d_shape,
        "no_fault_collapse": no_collapse,
        "verdict": "GO(5-seed 확장 후보)" if go else "NO-GO(1-seed 종료 후보)",
    }


ROW_LABELS = [
    ("S_total", "**S_total** (primary)"),
    ("amp_sensitive_total", "amp-sensitive fault (total)"),
    ("shape_sensitive_total", "shape-sensitive fault (total)"),
    ("S_shape", "S_shape"),
    ("S_amp", "S_amp"),
    ("zero_support_total", "zero-support (total)"),
    ("compositional_total", "compositional (total)"),
    ("normal_fpr_total_high", "정상 FPR 고진폭 (total)"),
    ("normal_fpr_total_low", "정상 FPR 저진폭 (total)"),
    ("rho_rms_total", "ρ(RMS, S_total)"),
    ("raw", "raw AUROC (파이프라인 검증)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--g3a_dir", type=str,
                    default=os.path.join(RESULTS_ROOT, "diag_G3a_amp_bands"))
    ap.add_argument("--g4b_dir", type=str,
                    default=os.path.join(RESULTS_ROOT, "diag_G4b_amp_bands"))
    ap.add_argument("--g4c_dir", type=str,
                    default=os.path.join(RESULTS_ROOT, "diag_G4c_amp_bands"))
    ap.add_argument("--report", type=str,
                    default=os.path.join(PROJECT_ROOT, "reports", "report_G4_amp_head_screening.md"))
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(RESULTS_ROOT, "diag_G4_amp_head"))
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    base = load_folds(args.g3a_dir, args.seed)
    if not base:
        raise SystemExit(f"[에러] G-3a per-fold JSON 없음: {args.g3a_dir} (seed {args.seed})")

    cands = {}
    for name, d in (("B", args.g4b_dir), ("C", args.g4c_dir)):
        folds = load_folds(d, args.seed)
        if folds:
            cands[name] = folds
        else:
            print(f"[알림] {name} diag 없음(스킵): {d}")
    if not cands:
        raise SystemExit("[에러] 비교할 후보(B/C) diag JSON이 하나도 없다.")

    kbase = {_fold_key(f): f for f in base}

    # 각 후보를 G-3a와 공통 fold(교집합)에서 paired 비교. G-3a 열은 후보별 공통집합으로 재집계.
    results = {}
    for name, folds in cands.items():
        kc = {_fold_key(f): f for f in folds}
        common = sorted(set(kbase) & set(kc))
        only_base = sorted(set(kbase) - set(kc))
        only_cand = sorted(set(kc) - set(kbase))
        base_c = [kbase[k] for k in common]
        cand_c = [kc[k] for k in common]
        base_m = _metric_rows(aggregate(base_c), base_c)
        cand_m = _metric_rows(aggregate(cand_c), cand_c)
        # per-fold S_total Δ
        per_fold = []
        for k in common:
            a, b = kbase[k], kc[k]
            per_fold.append({
                "split": k[0], "lono": k[1], "fold_type": a["fold_type"],
                "target_amp_group": a.get("target_amp_group", "?"),
                "g3a": a["g1"]["auroc_total"], "cand": b["g1"]["auroc_total"],
                "delta": b["g1"]["auroc_total"] - a["g1"]["auroc_total"],
            })
        results[name] = {
            "n_common": len(common), "only_g3a": only_base, "only_cand": only_cand,
            "base_metrics": base_m, "cand_metrics": cand_m,
            "verdict": _verdict(base_m, cand_m), "per_fold": per_fold,
        }

    summary = {"seed": args.seed, "seed_noise_floor": SEED_NOISE,
               "collapse_tol": COLLAPSE_TOL, "candidates": results}

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"summary_g4_vs_g3a_s{args.seed}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[저장] {out_json}")

    # ---- 리포트 ----
    if not args.no_report:
        L = []
        L.append(f"# G-4 Amp Head ablation — seed {args.seed} paired 비교 (S_total 중심)\n")
        L.append("- 후보: " + ", ".join(
            f"{n}(n={results[n]['n_common']})" for n in results) +
            f" · baseline = G-3a (diag_G3a_amp_bands, seed {args.seed})")
        for n in results:
            v = results[n]["verdict"]
            L.append(f"- **{n} 판정 휴리스틱: {v['verdict']}** "
                     f"(ΔS_total={v['delta_S_total']:+.3f}, "
                     f"Δamp={v['delta_amp_sensitive']:+.3f}, Δshape={v['delta_shape_sensitive']:+.3f}, "
                     f"fault 비붕괴={v['no_fault_collapse']})")
            if results[n]["only_g3a"] or results[n]["only_cand"]:
                L.append(f"  - ⚠️ fold 불일치: g3a-only {results[n]['only_g3a']}, "
                         f"{n}-only {results[n]['only_cand']}")
        L.append(f"\n> 판정 규칙(TODO §G-4): S_total이 seed 분산(~{SEED_NOISE}) 초과 상승 + 두 fault군 비붕괴면 "
                 f"해당 후보만 5-seed 확장. 최종 판단은 사용자 승인. cherry-pick 금지.\n")

        # 지표 표(열 = G-3a | B | C)
        names = list(results.keys())
        L.append("## 지표 비교 (G-3a → 후보, Δ)")
        header = "| 지표 | G-3a | " + " | ".join(names) + " |"
        L.append(header)
        L.append("|" + "---|" * (2 + len(names)))
        for key, label in ROW_LABELS:
            base_val = results[names[0]]["base_metrics"][key]  # G-3a는 후보간 공통집합 차이로 미세 다를 수 있음 → 첫 후보 기준 표기
            cells = [label, f"{base_val:.3f}"]
            for n in names:
                bm = results[n]["base_metrics"][key]
                cm = results[n]["cand_metrics"][key]
                cells.append(f"{cm:.3f} (Δ{cm - bm:+.3f})" if cm == cm else "—")
            L.append("| " + " | ".join(cells) + " |")
        L.append("\n> raw AUROC 행은 세 열이 동일해야 정상(같은 B3 checkpoint 재추론). "
                 "G-3a 열은 각 후보와의 공통 fold로 재집계하므로 후보간 fold 수가 다르면 미세 차이 가능.\n")

        # per-fold S_total (후보별)
        for n in names:
            L.append(f"## per-fold S_total (G-3a → {n})")
            L.append("| split | LONO | fold_type | amp_grp | G-3a | " + n + " | Δ |")
            L.append("|---|---|---|---|---|---|---|")
            for r in results[n]["per_fold"]:
                L.append(f"| {r['split']} | {r['lono']} | {r['fold_type']} | {r['target_amp_group']} | "
                         f"{r['g3a']:.3f} | {r['cand']:.3f} | {r['delta']:+.3f} |")
            L.append("")

        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"[저장] {args.report}")

    # 콘솔 요약
    print(f"\n=== G-4 vs G-3a (seed {args.seed}) ===")
    for n in results:
        v = results[n]["verdict"]
        print(f"[{n}] ΔS_total={v['delta_S_total']:+.3f} "
              f"Δamp={v['delta_amp_sensitive']:+.3f} Δshape={v['delta_shape_sensitive']:+.3f} "
              f"→ {v['verdict']}")


if __name__ == "__main__":
    main()
