"""작업 G-3a 5-seed 재집계 — diagnose_G3a_amp_bands.py가 seed별로 남긴 aggregate_s<seed>.json을
읽어 5 seed(2024/2025/2026/2027/2028) 평균±표준편차로 재집계·판정한다(학습/추론 없음, 순수 취합).

전제: 각 seed에 대해 `python analysis/diagnose_G3a_amp_bands.py --seed <s>`를 이미 실행해
  results/Paderborn/diag_G3a_amp_bands/aggregate_s<s>.json 이 존재해야 한다.

산출: reports/report_G3a_amp_bands_5seeds.md + results/Paderborn/diag_G3a_amp_bands/summary_5seeds.json
재집계 항목(가드레일): S_shape/S_amp/S_total AUROC(전체 + 외삽/compositional 분리),
  ρ(RMS,score), 정상 FPR(진폭군별), fault군별 AUROC(amp/shape-sensitive 동시보존).
판정 기준: 전체 AUROC>0.696(no-meta LOSO) · ρ↓ · 고진폭 정상 FPR 억제 · 두 fault군 동시보존 · 5-seed 평균 · cherry-pick 금지.
"""
import argparse
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG_DIR = os.path.join(PROJECT_ROOT, "results", "Paderborn", "diag_G3a_amp_bands")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G3a_amp_bands_5seeds.md")

AMP_SENSITIVE = {"KA04", "KA16", "KA30", "KB23", "KB24", "KI04", "KI16", "KI18"}
SHAPE_SENSITIVE = {"KA15", "KA22", "KB27", "KI14", "KI17", "KI21"}


def ms(vals):
    """mean±std (nan 제외). 반환 (mean, std, n)."""
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and x == x]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), float(np.std(v)), len(v)


def fmt_ms(m, s, p=3):
    if m != m:
        return "nan"
    return f"{m:.{p}f}±{s:.{p}f}"


def get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def collect(aggs, path):
    """seed별 aggregate에서 path(키 튜플) 값을 모아 리스트로."""
    out = []
    for a in aggs:
        v = a
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                break
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    ap.add_argument("--diag_dir", type=str, default=DIAG_DIR)
    args = ap.parse_args()

    aggs, used_seeds, missing = [], [], []
    for s in args.seeds:
        p = os.path.join(args.diag_dir, f"aggregate_s{s}.json")
        if not os.path.exists(p):
            missing.append(s)
            continue
        with open(p) as f:
            aggs.append(json.load(f))
        used_seeds.append(s)
    if not aggs:
        raise SystemExit(f"aggregate_s*.json 없음: {args.diag_dir} (먼저 seed별 diagnose 실행)")
    if missing:
        print(f"⚠️ 누락 seed(집계 제외): {missing}")

    # ---- 헤드라인(overall) ----
    overall = {k: ms(collect(aggs, ("overall", k)))
               for k in ("auroc_shape", "auroc_amp", "auroc_total", "auroc_raw")}

    # ---- fold 유형별 ----
    by_type = {}
    for ft in ("zero-support", "compositional"):
        by_type[ft] = {k: ms(collect(aggs, ("by_fold_type", ft, k)))
                       for k in ("auroc_shape", "auroc_amp", "auroc_total", "auroc_raw",
                                 "delta_total_raw", "rho_shape", "rho_amp", "rho_total", "rho_raw")}

    # ---- 정상 FPR(진폭군별) ----
    amp_normal = {k: ms(collect(aggs, ("amp_normal", k)))
                  for k in ("high_fpr", "low_fpr", "high_mean", "low_mean")}
    total_normal = {k: ms(collect(aggs, ("total_normal", k))) for k in ("high_fpr", "low_fpr")}
    raw_normal = {k: ms(collect(aggs, ("raw_normal", k))) for k in ("high_fpr", "low_fpr")}

    # ---- fault군별 ----
    fault_group = {}
    for grp in ("amp_sensitive", "shape_sensitive"):
        fault_group[grp] = {k: ms(collect(aggs, ("fault_group", grp, k)))
                            for k in ("auroc_raw", "auroc_shape", "auroc_amp", "auroc_total")}

    # ---- per-fault ----
    fault_ids = sorted({fid for a in aggs for fid in a.get("per_fault", {})})
    per_fault = {}
    for fid in fault_ids:
        per_fault[fid] = {k: ms([get(a, "per_fault", fid, k) for a in aggs])
                          for k in ("auroc_raw", "auroc_shape", "auroc_amp", "auroc_total")}

    summary = {
        "seeds": used_seeds, "n_seeds": len(used_seeds),
        "overall": overall, "by_fold_type": by_type,
        "amp_normal": amp_normal, "total_normal": total_normal, "raw_normal": raw_normal,
        "fault_group": fault_group, "per_fault": per_fault,
    }
    os.makedirs(args.diag_dir, exist_ok=True)
    with open(os.path.join(args.diag_dir, "summary_5seeds.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- 리포트 ----
    o = overall
    zt, ct = by_type["zero-support"], by_type["compositional"]
    a_s, s_s = fault_group["amp_sensitive"], fault_group["shape_sensitive"]
    an, tn, rn = amp_normal, total_normal, raw_normal

    L = []
    L.append("# 작업 G-3a 5-seed 재집계 — Multi-band Amplitude Branch (6밴드)")
    L.append("")
    L.append(f"seed {used_seeds} ({len(used_seeds)} seed) mean±std. 무학습 재추론 seed별 aggregate 취합. "
             "G-3a = amp_normalize(shape-only window) + amp_branch(K-band 조건부 진폭 head, Joint 학습). "
             "window 2048, no-meta LOSO 4×6=24 fold/seed. S_total=val-normal 표준화 등가중 합, "
             "raw=B3 flow_NLL paired. (test 라벨 튜닝 없음.) "
             "G-1(scalar) shape 0.704 / G-2a(scalar gate) 0.706가 못 깬 amp↔shape 트레이드오프를 6밴드가 깨는지 검증.")
    if missing:
        L.append("")
        L.append(f"> ⚠️ 누락 seed: {missing} — 집계에서 제외됨.")
    L.append("")

    L.append("## 핵심 요약 (5-seed mean±std) — 가드레일 GO/NO-GO")
    L.append("")
    L.append(f"- **전체 AUROC** (24 fold): **S_total {fmt_ms(*o['auroc_total'][:2])}** / **S_shape {fmt_ms(*o['auroc_shape'][:2])}** / "
             f"S_amp {fmt_ms(*o['auroc_amp'][:2])} vs raw {fmt_ms(*o['auroc_raw'][:2])} (가드레일 no-meta LOSO 0.696).")
    L.append(f"- **fold 유형 분리**: zero-support 외삽 S_total {fmt_ms(*zt['auroc_total'][:2])} / "
             f"compositional S_total {fmt_ms(*ct['auroc_total'][:2])} (네 fold 뭉뚱그림 금지).")
    L.append(f"- **① 고진폭 정상 S_amp 과탐 여부**: 정상 FPR(S_amp) 고 {fmt_ms(*an['high_fpr'][:2])} vs 저 "
             f"{fmt_ms(*an['low_fpr'][:2])}; mean(S_amp) 고 {fmt_ms(*an['high_mean'][:2])} / 저 {fmt_ms(*an['low_mean'][:2])}. "
             f"(raw 고진폭 FPR {fmt_ms(*rn['high_fpr'][:2])}.)")
    L.append(f"- **② amp-sensitive fault**: raw {fmt_ms(*a_s['auroc_raw'][:2])} → S_shape {fmt_ms(*a_s['auroc_shape'][:2])} → "
             f"**S_amp {fmt_ms(*a_s['auroc_amp'][:2])}** → S_total {fmt_ms(*a_s['auroc_total'][:2])}.")
    L.append(f"- **③ shape-sensitive fault**: raw {fmt_ms(*s_s['auroc_raw'][:2])} → **S_shape {fmt_ms(*s_s['auroc_shape'][:2])}** → "
             f"S_amp {fmt_ms(*s_s['auroc_amp'][:2])} → S_total {fmt_ms(*s_s['auroc_total'][:2])}.")
    L.append(f"- **④ 동시보존(S_total)**: amp-sensitive {fmt_ms(*a_s['auroc_total'][:2])} / shape-sensitive {fmt_ms(*s_s['auroc_total'][:2])} "
             "— 두 군 모두 각 branch 최고치에 근접해야 트레이드오프 극복(G-1·G-2a 실패 지점).")
    L.append(f"- **⑤ raw 대비 paired**: Δ(total−raw) zero {fmt_ms(*zt['delta_total_raw'][:2])} / "
             f"comp {fmt_ms(*ct['delta_total_raw'][:2])}. 정상 FPR(S_total) 고 {fmt_ms(*tn['high_fpr'][:2])}/저 {fmt_ms(*tn['low_fpr'][:2])}.")
    L.append(f"- **ρ(RMS,·)**: zero S_shape {fmt_ms(*zt['rho_shape'][:2])} / S_amp {fmt_ms(*zt['rho_amp'][:2])} / "
             f"S_total {fmt_ms(*zt['rho_total'][:2])} vs raw {fmt_ms(*zt['rho_raw'][:2])} (작업 B raw≈0.96).")
    L.append("")

    L.append("## 1) fold 유형별 (5-seed mean±std)")
    L.append("")
    L.append("| fold 유형 | S_total | S_shape | S_amp | raw | Δ(total−raw) | ρ_total | ρ_raw |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ft, a in by_type.items():
        L.append(f"| {ft} | **{fmt_ms(*a['auroc_total'][:2])}** | {fmt_ms(*a['auroc_shape'][:2])} | "
                 f"{fmt_ms(*a['auroc_amp'][:2])} | {fmt_ms(*a['auroc_raw'][:2])} | {fmt_ms(*a['delta_total_raw'][:2])} | "
                 f"{fmt_ms(*a['rho_total'][:2])} | {fmt_ms(*a['rho_raw'][:2])} |")
    L.append("")

    L.append("## 2) fault군별 branch AUROC (5-seed mean±std) — ②③④ 동시보존")
    L.append("")
    L.append("| fault군 | raw | S_shape | S_amp | S_total |")
    L.append("|---|---|---|---|---|")
    L.append(f"| amp-sensitive | {fmt_ms(*a_s['auroc_raw'][:2])} | {fmt_ms(*a_s['auroc_shape'][:2])} | "
             f"**{fmt_ms(*a_s['auroc_amp'][:2])}** | {fmt_ms(*a_s['auroc_total'][:2])} |")
    L.append(f"| shape-sensitive | {fmt_ms(*s_s['auroc_raw'][:2])} | **{fmt_ms(*s_s['auroc_shape'][:2])}** | "
             f"{fmt_ms(*s_s['auroc_amp'][:2])} | {fmt_ms(*s_s['auroc_total'][:2])} |")
    L.append("")

    L.append("## 3) 정상 FPR 진폭군별 (5-seed mean±std)")
    L.append("")
    L.append("| 지표 | 고진폭(K001/K003/K006) | 저진폭(K002/K004/K005) |")
    L.append("|---|---|---|")
    L.append(f"| FPR(S_amp) | {fmt_ms(*an['high_fpr'][:2])} | {fmt_ms(*an['low_fpr'][:2])} |")
    L.append(f"| mean(S_amp) | {fmt_ms(*an['high_mean'][:2])} | {fmt_ms(*an['low_mean'][:2])} |")
    L.append(f"| FPR(S_total) | {fmt_ms(*tn['high_fpr'][:2])} | {fmt_ms(*tn['low_fpr'][:2])} |")
    L.append(f"| FPR(raw) | {fmt_ms(*rn['high_fpr'][:2])} | {fmt_ms(*rn['low_fpr'][:2])} |")
    L.append("")

    L.append("## 4) per-fault AUROC (5-seed mean±std)")
    L.append("")
    L.append("| fault id | 분류 | raw | S_shape | S_amp | S_total |")
    L.append("|---|---|---|---|---|---|")
    for fid in fault_ids:
        d = per_fault[fid]
        cls = "A" if fid in AMP_SENSITIVE else "S" if fid in SHAPE_SENSITIVE else ""
        L.append(f"| {fid} | {cls} | {fmt_ms(*d['auroc_raw'][:2])} | {fmt_ms(*d['auroc_shape'][:2])} | "
                 f"{fmt_ms(*d['auroc_amp'][:2])} | {fmt_ms(*d['auroc_total'][:2])} |")
    L.append("")
    L.append("> A=amplitude-sensitive, S=shape-sensitive. 판정은 다지표(①~⑤)를 5-seed 평균으로 확인 — cherry-pick 금지.")
    L.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {REPORT_PATH}")
    print(f"seeds={used_seeds}")
    print(f"overall: S_total={fmt_ms(*o['auroc_total'][:2])} S_shape={fmt_ms(*o['auroc_shape'][:2])} "
          f"S_amp={fmt_ms(*o['auroc_amp'][:2])} raw={fmt_ms(*o['auroc_raw'][:2])}")


if __name__ == "__main__":
    main()
