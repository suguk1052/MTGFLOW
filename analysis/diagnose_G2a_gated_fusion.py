"""작업 G-2a — conditional amplitude-gated fusion feasibility (재학습 없음 / checkpoint 재추론 있음).

착상(TODO G-2): G-1의 S_amp(=-log p(a|h_shape))는 독립 detector로는 약했지만 고진폭 정상의
amplitude confound를 안정적으로 제거했다(고진폭 정상 FPR 0.133 vs raw 0.528, ρ(RMS,S_amp)≈-0.05).
따라서 S_amp를 anomaly score에 직접 더하지 말고, **raw 진폭 evidence를 얼마나 믿을지 정하는
reliability gate**로 전용한다. 진폭이 형상 대비 implausible할 때(S_amp↑)만 raw를 신뢰하고,
정상 범위면(S_amp↓, 고진폭 정상 포함) shape에 의존해 raw의 고진폭 과탐을 억제한다.

이 스크립트는 **재학습 없이 기존 G-1·B3 checkpoint를 재추론**해 per-window score를 뽑고
(Phase A, GPU), **val-normal calibration만으로** convex gate를 구성한 gated fusion을
24 fold × 5 seed 다지표로 평가한다(Phase B, CPU). **test-label 기반 gate/weight tuning 없음.**

convex gate (val-normal ONLY):
    z_raw   = (raw_nll  - μ_raw)   / σ_raw       # μ,σ: val-normal
    z_shape = (S_shape  - μ_shape) / σ_shape     # μ,σ: val-normal
    g       = sigmoid((S_amp - τ) / s)           # τ = val-normal S_amp p95(primary), s = val-normal S_amp std
    F       = (1 - g) * z_shape + g * z_raw      # convex fusion

판정(가드레일 다지표, cherry-pick 금지, primary=p95):
    (a) overall AUROC(F) ≥ ~0.70 (raw 0.696 / S_shape 0.704 이상)
    (b) amp-sensitive fault가 S_shape(0.704)보다 유의 회복(raw 0.892 쪽)
    (c) shape-sensitive fault 유지(≳0.65, raw 0.436으로 붕괴 안 함)
    (d) 고진폭 target-normal FPR(F) ≪ raw 0.528
    (e) 5-seed 재현(분산 < gap)
GO면 G-2b(end-to-end) 제안, NO-GO면 multi-band/multi-scale amplitude branch 전환.

산출:
- results/Paderborn/diag_G2a_gated_fusion/cache/<split>_LONO<n>_s<seed>.npz  (Phase A, per-window)
- results/Paderborn/diag_G2a_gated_fusion/aggregate_s<seed>.json             (Phase B)
- results/Paderborn/diag_G2a_gated_fusion/summary_5seeds.json                (Phase B)
- reports/report_G2a_gated_fusion.md                                         (Phase B)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# G-1 진단 스크립트의 검증된 블록·상수 재사용(중복 구현 금지)
from analysis.diagnose_G1_disentangle import (  # noqa: E402
    build_model, build_loader, branch_scores, flow_nll, pull,
    safe_auroc, spearman, RESULTS_ROOT,
    HIGH_AMP, SPLIT_INFO, LONO_TARGET, AMP_SENSITIVE, SHAPE_SENSITIVE,
)

OUT_ROOT = os.path.join(RESULTS_ROOT, "diag_G2a_gated_fusion")
CACHE_DIR = os.path.join(OUT_ROOT, "cache")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G2a_gated_fusion.md")

SPLITS = list(SPLIT_INFO.keys())            # 123to0, 023to1, 013to2, 012to3
LONOS = [1, 2, 3, 4, 5, 6]
SEEDS = [2024, 2025, 2026, 2027, 2028]


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------
def _find_ckpt(batch_dir, run_name):
    p = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if os.path.exists(p):
        return p
    alt = os.path.join(RESULTS_ROOT, run_name, "model.pth")
    return alt if os.path.exists(alt) else None


def _cache_path(split, lono, seed):
    return os.path.join(CACHE_DIR, f"{split}_LONO{lono}_s{seed}.npz")


def fmt(v, p=3):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{p}f}"


def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.std(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Phase A — 재추론 → per-window npz 캐시 (GPU)
# ---------------------------------------------------------------------------
def _align_raw_to_g1(g1s, ids, lab, rms, raw, tag):
    """raw per-window 배열을 G-1 window 순서에 맞춘다. 순서 가정 금지 — 검증 후 필요 시 join."""
    aligned = (len(ids) == len(g1s["ids"])
               and np.array_equal(ids, g1s["ids"])
               and np.array_equal(lab, g1s["label"])
               and np.allclose(rms, g1s["rms_raw"], rtol=0, atol=1e-5))
    if aligned:
        return raw
    # fallback: (id, round(rms,6)) 복합키 join
    print(f"    [align] {tag}: 순서 불일치 → 복합키 join 시도")
    key_to_idx = {}
    for i, k in enumerate(zip(ids.tolist(), np.round(rms, 6).tolist())):
        key_to_idx.setdefault(k, []).append(i)
    order = []
    used = {}
    for k in zip(g1s["ids"].tolist(), np.round(g1s["rms_raw"], 6).tolist()):
        cand = key_to_idx.get(k)
        if not cand:
            raise RuntimeError(f"alignment join 실패({tag}): 대응 window 없음 key={k}")
        j = used.get(k, 0)
        if j >= len(cand):
            raise RuntimeError(f"alignment join 실패({tag}): 중복키 소진 key={k}")
        order.append(cand[j])
        used[k] = j + 1
    order = np.asarray(order)
    raw = raw[order]
    if not np.array_equal(lab[order], g1s["label"]):
        raise RuntimeError(f"alignment join 실패({tag}): label 불일치")
    return raw


def infer_fold(split, lono, seed, device, batch_size, g1_batch_dir, raw_batch_dir, raw_prefix, force):
    out = _cache_path(split, lono, seed)
    if os.path.exists(out) and not force:
        print(f"  [skip] cache 존재: {os.path.basename(out)}")
        return out
    g1_ckpt = _find_ckpt(g1_batch_dir, f"g1_{split}_LONO{lono}_s{seed}")
    raw_ckpt = _find_ckpt(raw_batch_dir, f"{raw_prefix}_{split}_LONO{lono}_s{seed}")
    if g1_ckpt is None or raw_ckpt is None:
        print(f"  [skip] checkpoint 없음 (g1={g1_ckpt}, raw={raw_ckpt})")
        return None

    # --- G-1 재추론: per-window S_shape / S_amp ---
    ck = torch.load(g1_ckpt, map_location=device)
    meta = ck["paderborn_metadata"]
    assert bool(meta.get("amp_branch", False)), f"amp_branch=False: {g1_ckpt}"
    assert bool(meta.get("amp_normalize", False)), f"amp_normalize=False: {g1_ckpt}"
    assert not bool(meta["use_meta"]), f"use_meta=True(no-meta 전용): {g1_ckpt}"
    trl, vl, tel, ns = build_loader(meta, batch_size)
    model = build_model(ck, ns, device, amp_branch=True)

    data = {}
    for name, loader in (("train", trl), ("val", vl), ("test", tel)):
        w, rms, rz, ids, lab = pull(loader.dataset)
        sh, am = branch_scores(model, w, rz, device, batch_size)
        data[name] = {"ids": ids, "label": lab.astype(np.int64),
                      "rms_raw": rms.astype(np.float64),
                      "s_shape": sh.astype(np.float64), "s_amp": am.astype(np.float64)}

    # --- raw B3 재추론: per-window flow NLL (동일 fold/seed, paired) ---
    ckr = torch.load(raw_ckpt, map_location=device)
    metar = ckr["paderborn_metadata"]
    trl2, vl2, tel2, ns2 = build_loader(metar, batch_size)
    modelr = build_model(ckr, ns2, device, amp_branch=False)
    for name, loader in (("train", trl2), ("val", vl2), ("test", tel2)):
        w, rms, rz, ids, lab = pull(loader.dataset)
        raw = flow_nll(modelr, w, device, batch_size).astype(np.float64)
        data[name]["raw_nll"] = _align_raw_to_g1(
            data[name], ids, lab.astype(np.int64), rms.astype(np.float64), raw,
            tag=f"{split}_LONO{lono}_s{seed}/{name}")

    tgt_bid = LONO_TARGET.get(lono)
    payload = {"split": np.array(split), "lono": np.array(lono), "seed": np.array(seed),
               "fold_type": np.array(SPLIT_INFO[split][1]),
               "target_amp_group": np.array("high" if tgt_bid in HIGH_AMP else "low")}
    for name in ("train", "val", "test"):
        for k, v in data[name].items():
            payload[f"{name}_{k}"] = v
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"  wrote {os.path.basename(out)} "
          f"(train {len(data['train']['ids'])}, val {len(data['val']['ids'])}, "
          f"test {len(data['test']['ids'])})")
    return out


def run_phase_a(args, device):
    print(f"[Phase A] 재추론 → per-window 캐시 (device={device})")
    n_ok = 0
    for seed in args.seeds:
        for split in args.splits:
            for lono in args.lonos:
                print(f"[{split} LONO{lono} s{seed}]")
                r = infer_fold(split, lono, seed, device, args.batch_size,
                               args.g1_batch_dir, args.raw_batch_dir, args.raw_prefix, args.force)
                if r:
                    n_ok += 1
    print(f"[Phase A] 완료 — {n_ok} fold-cache 확보")


# ---------------------------------------------------------------------------
# Phase B — convex gated fusion 평가 (CPU, label-free)
# ---------------------------------------------------------------------------
def _load_cache(split, lono, seed):
    p = _cache_path(split, lono, seed)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    d = {}
    for name in ("train", "val", "test"):
        d[name] = {k: z[f"{name}_{k}"] for k in ("ids", "label", "rms_raw", "s_shape", "s_amp", "raw_nll")}
    d["split"] = str(z["split"]); d["lono"] = int(z["lono"]); d["seed"] = int(z["seed"])
    d["fold_type"] = str(z["fold_type"]); d["target_amp_group"] = str(z["target_amp_group"])
    return d


def _zget(name, d, mu, sd):
    return (d[name] - mu) / (sd + 1e-8)


def _score_metrics(name, F_tr, F_va, F_te, d, thr_pct):
    """한 score(F/raw/shape/total)에 대한 지표. calibration은 이미 반영된 최종 score 입력."""
    te_lab = d["test"]["label"]
    te_ids = d["test"]["ids"]
    tgt = te_lab == 0
    auroc = safe_auroc(te_lab, F_te)

    # per-fault AUROC: target-normal(negatives) vs 각 fault id(positives)
    tgt_F = F_te[tgt]
    per_fault = {}
    for fid in sorted(set(te_ids[te_lab == 1].tolist())):
        m = (te_lab == 1) & (te_ids == fid)
        labels = np.concatenate([np.zeros(int(tgt.sum()), int), np.ones(int(m.sum()), int)])
        scores = np.concatenate([tgt_F, F_te[m]])
        per_fault[fid] = safe_auroc(labels, scores)

    # normal pool(train+val+target-normal) — ρ(RMS,F) 및 per-bearing FPR용 (target은 평가에만)
    norm_F = np.concatenate([F_tr, F_va, tgt_F])
    norm_rms = np.concatenate([d["train"]["rms_raw"], d["val"]["rms_raw"], d["test"]["rms_raw"][tgt]])
    norm_ids = np.concatenate([d["train"]["ids"], d["val"]["ids"], d["test"]["ids"][tgt]])
    rho = spearman(norm_rms, norm_F)

    # FPR threshold = val-normal F의 thr_pct (calibration split만)
    thr = float(np.percentile(F_va, thr_pct)) if len(F_va) else float("inf")

    # target-normal FPR 진폭군별 (per-bearing → 고/저 평균)
    per_bearing = {}
    for bid in sorted(set(te_ids[tgt].tolist())):
        m = tgt & (te_ids == bid)
        per_bearing[bid] = {"group": "high" if bid in HIGH_AMP else "low",
                            "fpr": float(np.mean(F_te[m] >= thr))}

    def _grp(grp):
        vals = [v["fpr"] for v in per_bearing.values() if v["group"] == grp]
        return float(np.mean(vals)) if vals else float("nan")

    return {"name": name, "auroc": auroc, "per_fault_auroc": per_fault,
            "rho_rms_score": rho, "threshold_val": thr,
            "tgt_high_fpr": _grp("high"), "tgt_low_fpr": _grp("low")}


def eval_fold(d, gate_pct, thr_pct):
    """convex gated fusion + 참조(raw/shape/total) 지표. 모든 calibration은 val-normal only."""
    va, tr, te = d["val"], d["train"], d["test"]

    # val-normal calibration (μ,σ,τ,s는 val에서만)
    mu_raw, sd_raw = float(va["raw_nll"].mean()), float(va["raw_nll"].std())
    mu_sh, sd_sh = float(va["s_shape"].mean()), float(va["s_shape"].std())
    mu_am, sd_am = float(va["s_amp"].mean()), float(va["s_amp"].std())
    tau = float(np.percentile(va["s_amp"], gate_pct))
    s_temp = float(va["s_amp"].std()) + 1e-8

    def z_raw(x): return _zget("raw_nll", x, mu_raw, sd_raw)
    def z_shape(x): return _zget("s_shape", x, mu_sh, sd_sh)
    def z_amp(x): return _zget("s_amp", x, mu_am, sd_am)

    def gate(x):  # g = sigmoid((S_amp - τ)/s) — 진폭이 형상 대비 val-normal 상단 초과 시 raw 신뢰
        return 1.0 / (1.0 + np.exp(-(x["s_amp"] - tau) / s_temp))

    def F_gate(x):
        g = gate(x)
        return (1.0 - g) * z_shape(x) + g * z_raw(x)

    def F_total(x):  # G-1 등가중 합(참조)
        return z_shape(x) + z_amp(x)

    blocks = {
        "gate": _score_metrics("gate", F_gate(tr), F_gate(va), F_gate(te), d, thr_pct),
        "raw": _score_metrics("raw", z_raw(tr), z_raw(va), z_raw(te), d, thr_pct),
        "shape": _score_metrics("shape", z_shape(tr), z_shape(va), z_shape(te), d, thr_pct),
        "total": _score_metrics("total", F_total(tr), F_total(va), F_total(te), d, thr_pct),
    }
    return {"split": d["split"], "lono": d["lono"], "seed": d["seed"],
            "fold_type": d["fold_type"], "target_amp_group": d["target_amp_group"],
            "calib": {"tau": tau, "s_temp": s_temp, "mu_raw": mu_raw, "sd_raw": sd_raw,
                      "mu_shape": mu_sh, "sd_shape": sd_sh},
            "blocks": blocks}


def _fault_group_auroc(folds, branch, fault_set):
    """fault id별 fold 평균 → 지정 fault군 평균 AUROC."""
    per_fault = {}
    for f in folds:
        for fid, au in f["blocks"][branch]["per_fault_auroc"].items():
            per_fault.setdefault(fid, []).append(au)
    vals = [_mean(v) for fid, v in per_fault.items() if fid in fault_set]
    return _mean(vals)


def aggregate_seed(folds):
    """seed 하나(24 fold) 집계."""
    def ov(branch):
        return _mean([f["blocks"][branch]["auroc"] for f in folds])

    by_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in folds if f["fold_type"] == ftype]
        if not rows:
            continue
        by_type[ftype] = {br: _mean([r["blocks"][br]["auroc"] for r in rows])
                          for br in ("gate", "raw", "shape", "total")}
        by_type[ftype]["rho_gate"] = _mean([r["blocks"]["gate"]["rho_rms_score"] for r in rows])
        by_type[ftype]["rho_raw"] = _mean([r["blocks"]["raw"]["rho_rms_score"] for r in rows])

    fault_group = {}
    for br in ("gate", "raw", "shape", "total"):
        fault_group[br] = {
            "amp_sensitive": _fault_group_auroc(folds, br, AMP_SENSITIVE),
            "shape_sensitive": _fault_group_auroc(folds, br, SHAPE_SENSITIVE),
        }

    normal_fpr = {}
    for br in ("gate", "raw", "shape", "total"):
        normal_fpr[br] = {"high": _mean([f["blocks"][br]["tgt_high_fpr"] for f in folds]),
                          "low": _mean([f["blocks"][br]["tgt_low_fpr"] for f in folds])}

    rho = {br: _mean([f["blocks"][br]["rho_rms_score"] for f in folds])
           for br in ("gate", "raw", "shape", "total")}

    # per-fault (fold 평균) — 리포트 3)용
    per_fault = {}
    for f in folds:
        for br in ("gate", "raw", "shape", "total"):
            for fid, au in f["blocks"][br]["per_fault_auroc"].items():
                per_fault.setdefault(fid, {}).setdefault(br, []).append(au)
    per_fault = {fid: {br: _mean(v.get(br, [])) for br in ("gate", "raw", "shape", "total")}
                 for fid, v in sorted(per_fault.items())}

    return {"n_folds": len(folds),
            "overall": {br: ov(br) for br in ("gate", "raw", "shape", "total")},
            "by_fold_type": by_type, "fault_group": fault_group,
            "normal_fpr": normal_fpr, "rho": rho, "per_fault": per_fault}


# ---------------------------------------------------------------------------
# 5-seed 요약 + 리포트
# ---------------------------------------------------------------------------
def _msd(aggs, getter):
    vals = [getter(a) for a in aggs]
    return _mean(vals), _std(vals)


def summarize(aggs):
    """seed별 aggregate 리스트 → 5-seed mean±std."""
    S = {}
    for br in ("gate", "raw", "shape", "total"):
        S.setdefault("overall", {})[br] = _msd(aggs, lambda a, b=br: a["overall"][b])
        S.setdefault("fault_amp", {})[br] = _msd(aggs, lambda a, b=br: a["fault_group"][b]["amp_sensitive"])
        S.setdefault("fault_shape", {})[br] = _msd(aggs, lambda a, b=br: a["fault_group"][b]["shape_sensitive"])
        S.setdefault("fpr_high", {})[br] = _msd(aggs, lambda a, b=br: a["normal_fpr"][b]["high"])
        S.setdefault("fpr_low", {})[br] = _msd(aggs, lambda a, b=br: a["normal_fpr"][b]["low"])
        S.setdefault("rho", {})[br] = _msd(aggs, lambda a, b=br: a["rho"][b])
    for ftype in ("zero-support", "compositional"):
        S.setdefault(ftype, {})
        for br in ("gate", "raw", "shape", "total"):
            S[ftype][br] = _msd(aggs, lambda a, ft=ftype, b=br: a["by_fold_type"].get(ft, {}).get(b, float("nan")))
        S[ftype]["rho_gate"] = _msd(aggs, lambda a, ft=ftype: a["by_fold_type"].get(ft, {}).get("rho_gate", float("nan")))
        S[ftype]["rho_raw"] = _msd(aggs, lambda a, ft=ftype: a["by_fold_type"].get(ft, {}).get("rho_raw", float("nan")))
    # per-fault 5-seed 평균 (aggregate_seed의 per_fault[fid][br]는 seed 내 fold 평균 float)
    fids = sorted({fid for a in aggs for fid in a["per_fault"]})
    S["per_fault"] = {}
    for fid in fids:
        S["per_fault"][fid] = {
            br: _msd(aggs, lambda a, f=fid, b=br: a["per_fault"].get(f, {}).get(b, float("nan")))
            for br in ("gate", "raw", "shape", "total")}
    return S


def _pm(msd, p=3):
    m, s = msd
    return f"{fmt(m, p)}±{fmt(s, p)}"


def build_report(prim, sens, seeds, gate_pct, sens_pct, thr_pct):
    """prim/sens: primary(p95)·sensitivity(p90) 5-seed summary dict."""
    L = []
    L.append("# 작업 G-2a — Conditional Amplitude-Gated Fusion (재학습 없음 / checkpoint 재추론)")
    L.append("")
    L.append(f"seed {seeds} ({len(seeds)} seed) mean±std. no-meta LOSO 4 split × 6 LONO = 24 fold/seed, window 2048. "
             "재학습 없이 G-1(amp_branch+amp_normalize)·raw B3 checkpoint를 재추론해 per-window "
             "S_shape/S_amp/raw_NLL 산출. **convex gate F=(1−g)·z(shape)+g·z(raw), "
             f"g=σ((S_amp−τ)/s), τ=val-normal S_amp p{int(gate_pct)}, s=val-normal S_amp std.** "
             "z·τ·s 전부 val-normal에서만 계산(target-normal은 FPR·ρ 평가에만). FPR threshold=val-normal "
             f"p{int(thr_pct)}. test-label 튜닝 없음. **primary=p{int(gate_pct)}, p{int(sens_pct)}는 sensitivity 전용.**")
    L.append("")

    ov = prim["overall"]
    fa, fs = prim["fault_amp"], prim["fault_shape"]
    fh, fl = prim["fpr_high"], prim["fpr_low"]
    rho = prim["rho"]
    L.append("## 핵심 요약 (5-seed mean±std, primary p95)")
    L.append("")
    L.append(f"- **전체 AUROC**: gate {_pm(ov['gate'])} vs raw {_pm(ov['raw'])} / S_shape {_pm(ov['shape'])} / "
             f"S_total {_pm(ov['total'])} (가드레일 0.696).")
    L.append(f"- **(b) amp-sensitive fault**: gate {_pm(fa['gate'])} vs raw {_pm(fa['raw'])} / S_shape {_pm(fa['shape'])}.")
    L.append(f"- **(c) shape-sensitive fault**: gate {_pm(fs['gate'])} vs raw {_pm(fs['raw'])} / S_shape {_pm(fs['shape'])}.")
    L.append(f"- **(d) 고진폭 target-normal FPR**: gate {_pm(fh['gate'])} vs raw {_pm(fh['raw'])} "
             f"(저진폭: gate {_pm(fl['gate'])} vs raw {_pm(fl['raw'])}).")
    L.append(f"- **ρ(RMS, score)**: gate {_pm(rho['gate'])} vs raw {_pm(rho['raw'])}.")
    L.append("")

    L.append("## 1) fold 유형별 (5-seed mean±std, p95)")
    L.append("")
    L.append("| fold 유형 | gate | raw | S_shape | S_total | ρ_gate | ρ_raw |")
    L.append("|---|---|---|---|---|---|---|")
    for ft in ("zero-support", "compositional"):
        a = prim[ft]
        L.append(f"| {ft} | **{_pm(a['gate'])}** | {_pm(a['raw'])} | {_pm(a['shape'])} | {_pm(a['total'])} | "
                 f"{_pm(a['rho_gate'])} | {_pm(a['rho_raw'])} |")
    L.append("")

    L.append("## 2) fault군별 branch AUROC (5-seed mean±std, p95)")
    L.append("")
    L.append("| fault군 | raw | S_shape | **gate** | S_total |")
    L.append("|---|---|---|---|---|")
    L.append(f"| amp-sensitive | {_pm(fa['raw'])} | {_pm(fa['shape'])} | **{_pm(fa['gate'])}** | {_pm(fa['total'])} |")
    L.append(f"| shape-sensitive | {_pm(fs['raw'])} | {_pm(fs['shape'])} | **{_pm(fs['gate'])}** | {_pm(fs['total'])} |")
    L.append("")

    L.append("## 3) per-fault AUROC (5-seed mean±std, p95)")
    L.append("")
    L.append("| fault id | 분류 | raw | S_shape | gate | S_total |")
    L.append("|---|---|---|---|---|---|")
    for fid, d in prim["per_fault"].items():
        tag = "A" if fid in AMP_SENSITIVE else "S" if fid in SHAPE_SENSITIVE else ""
        L.append(f"| {fid} | {tag} | {_pm(d['raw'])} | {_pm(d['shape'])} | {_pm(d['gate'])} | {_pm(d['total'])} |")
    L.append("")

    L.append("## 4) 진폭군별 target-normal FPR (5-seed mean±std, p95)")
    L.append("")
    L.append("| 지표 | 고진폭(K001/K003/K006) | 저진폭(K002/K004/K005) |")
    L.append("|---|---|---|")
    L.append(f"| FPR(gate) | {_pm(fh['gate'])} | {_pm(fl['gate'])} |")
    L.append(f"| FPR(raw)  | {_pm(fh['raw'])} | {_pm(fl['raw'])} |")
    L.append(f"| FPR(S_shape) | {_pm(fh['shape'])} | {_pm(fl['shape'])} |")
    L.append("")

    # sensitivity(p90) 부록
    so = sens["overall"]; sfa = sens["fault_amp"]; sfs = sens["fault_shape"]; sfh = sens["fpr_high"]
    L.append(f"## 5) sensitivity — gate percentile p{int(sens_pct)} (판정 미사용, 참고)")
    L.append("")
    L.append("| 지표 | p95 (primary) | p90 (sensitivity) |")
    L.append("|---|---|---|")
    L.append(f"| 전체 AUROC(gate) | {_pm(ov['gate'])} | {_pm(so['gate'])} |")
    L.append(f"| amp-sensitive(gate) | {_pm(fa['gate'])} | {_pm(sfa['gate'])} |")
    L.append(f"| shape-sensitive(gate) | {_pm(fs['gate'])} | {_pm(sfs['gate'])} |")
    L.append(f"| 고진폭 FPR(gate) | {_pm(fh['gate'])} | {_pm(sfh['gate'])} |")
    L.append("")

    # 정합성 체크(재추론이 G-1 재현?)
    L.append("## 6) 정합성 체크 (재추론 raw/S_shape ≈ G-1 리포트)")
    L.append("")
    L.append(f"- 재추론 raw 전체 AUROC {_pm(ov['raw'])} (G-1 리포트 raw 0.696±0.014).")
    L.append(f"- 재추론 S_shape 전체 AUROC {_pm(ov['shape'])} (G-1 리포트 S_shape 0.704±0.069).")
    L.append("- 두 값이 G-1과 일치하면 재추론 파이프라인 정상(=fusion 비교의 baseline 신뢰 가능).")
    L.append("")

    # GO/NO-GO 판정 (자동 요약, primary p95)
    L.append("## 종합 판정 (가드레일 다지표, primary p95)")
    L.append("")
    g_ov, r_ov, sh_ov = ov["gate"][0], ov["raw"][0], ov["shape"][0]
    cond_a = g_ov >= 0.70 or g_ov >= max(r_ov, sh_ov)
    cond_b = fa["gate"][0] > fa["shape"][0] + 0.02
    cond_c = fs["gate"][0] >= 0.65
    cond_d = fh["gate"][0] < fh["raw"][0] - 0.05
    conds = [("(a) 전체 AUROC≥~0.70/raw·shape 이상", cond_a, f"gate {fmt(g_ov)} vs raw {fmt(r_ov)}/shape {fmt(sh_ov)}"),
             ("(b) amp-sensitive가 S_shape보다 유의 회복", cond_b, f"gate {fmt(fa['gate'][0])} vs shape {fmt(fa['shape'][0])}"),
             ("(c) shape-sensitive 유지(≳0.65)", cond_c, f"gate {fmt(fs['gate'][0])}"),
             ("(d) 고진폭 정상 FPR ≪ raw", cond_d, f"gate {fmt(fh['gate'][0])} vs raw {fmt(fh['raw'][0])}")]
    for label, ok, note in conds:
        L.append(f"- [{'GO' if ok else 'NO'}] {label} — {note}")
    verdict = "GO" if all(c[1] for c in conds) else "NO-GO"
    L.append("")
    L.append(f"**자동 판정: {verdict}** (조건 (a)~(d) 동시 충족 + 5-seed 분산 확인 필요). "
             "GO면 G-2b(end-to-end dual-view+gate) 제안, NO-GO면 multi-band/multi-scale amplitude branch 전환. "
             "최종 판정은 표의 mean±std(분산<gap)까지 사람이 확인.")
    L.append("")
    L.append("> 주의: convex gate·표준화·threshold 전부 val-normal only(test 라벨 미사용). "
             "재추론만 있고 재학습 없음. target-normal은 FPR·ρ 평가에만 사용.")
    L.append("")
    return "\n".join(L) + "\n"


def run_phase_b(args):
    print("[Phase B] convex gated fusion 평가 (CPU, label-free)")
    prim_aggs, sens_aggs, used_seeds, missing = [], [], [], []
    for seed in args.seeds:
        folds = []
        for split in args.splits:
            for lono in args.lonos:
                d = _load_cache(split, lono, seed)
                if d is None:
                    missing.append(f"{split}_LONO{lono}_s{seed}")
                    continue
                folds.append(d)
        if not folds:
            print(f"  [seed {seed}] cache 없음 — 건너뜀")
            continue
        prim = [eval_fold(d, args.gate_percentile, args.threshold_percentile) for d in folds]
        sens = [eval_fold(d, args.sens_percentile, args.threshold_percentile) for d in folds]
        agg_p = aggregate_seed(prim)
        agg_s = aggregate_seed(sens)
        prim_aggs.append(agg_p); sens_aggs.append(agg_s); used_seeds.append(seed)
        with open(os.path.join(OUT_ROOT, f"aggregate_s{seed}.json"), "w") as f:
            json.dump({"primary_p95": agg_p, "sensitivity_p90": agg_s,
                       "gate_percentile": args.gate_percentile,
                       "sens_percentile": args.sens_percentile,
                       "threshold_percentile": args.threshold_percentile}, f, indent=2, ensure_ascii=False)
        ov = agg_p["overall"]
        print(f"  [seed {seed}] {agg_p['n_folds']} fold — gate {fmt(ov['gate'])} "
              f"raw {fmt(ov['raw'])} shape {fmt(ov['shape'])} total {fmt(ov['total'])}")
    if not prim_aggs:
        raise SystemExit("Phase B: cache가 없습니다. 먼저 Phase A(재추론)를 실행하세요.")
    if missing:
        print(f"  ⚠️ 누락 fold({len(missing)}): {missing[:6]}{' …' if len(missing) > 6 else ''}")

    prim = summarize(prim_aggs)
    sens = summarize(sens_aggs)
    with open(os.path.join(OUT_ROOT, "summary_5seeds.json"), "w") as f:
        json.dump({"seeds": used_seeds, "primary_p95": prim, "sensitivity_p90": sens}, f,
                  indent=2, ensure_ascii=False, default=list)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(build_report(prim, sens, used_seeds, args.gate_percentile,
                             args.sens_percentile, args.threshold_percentile))
    print(f"[Phase B] 완료 — {REPORT_PATH}")
    ov = prim["overall"]
    print(f"=== 5-seed({used_seeds}) 전체 AUROC === "
          f"gate {_pm(ov['gate'])} | raw {_pm(ov['raw'])} | shape {_pm(ov['shape'])} | total {_pm(ov['total'])}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--lonos", type=int, nargs="+", default=LONOS)
    ap.add_argument("--gate_percentile", type=float, default=95.0, help="primary gate τ percentile (val-normal S_amp)")
    ap.add_argument("--sens_percentile", type=float, default=90.0, help="sensitivity 전용 gate percentile")
    ap.add_argument("--threshold_percentile", type=float, default=95.0, help="FPR threshold (val-normal F)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--g1_batch_dir", type=str, default="LONO_G1_s2026")
    ap.add_argument("--raw_batch_dir", type=str, default="LONO_B3_5seeds")
    ap.add_argument("--raw_prefix", type=str, default="raw_vib")
    ap.add_argument("--force", action="store_true", help="Phase A: 기존 cache 무시하고 재추론")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    if args.phase in ("A", "all"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_phase_a(args, device)
    if args.phase in ("B", "all"):
        run_phase_b(args)


if __name__ == "__main__":
    main()
