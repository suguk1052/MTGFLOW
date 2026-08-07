"""작업 F-1 정밀 재평가 — Vy-rmsnorm(shape) vs raw-vib, 학습된 체크포인트 재추론(무학습).

전제(계획 B안): F-1의 표현·채널·split·seed·window(2048)는 작업 D와 완전히 동일하고, D의 24 fold
체크포인트가 이미 존재한다. F-0이 권한 window 16384/32768은 (1) MTGFlow attention Linear(c,c)이
c=window_size라 파라미터가 window에 제곱으로 커져 학습 불가이고, (2) 그 근거(envelope-spectrum
주파수 해상도)는 descriptor 파이프라인 얘기라 raw time-window를 그대로 넣는 MTGFlow 학습엔 자동
이식되지 않는다. 따라서 재학습하지 않고 기존 체크포인트를 재추론해 F-1 검증 목표를 정밀 평가·종결한다.

비교쌍(둘 다 no-meta, window 2048, seed 2026, 동일 4 LOSO × 6 LONO):
  - rmsnorm(작업 D): results/Paderborn/<rmsnorm_batch_dir>/<rmsnorm_prefix>_<split>_LONO<n>_s<seed>/model.pth
    (amp_normalize=True → per-window RMS로 shape-only 정규화한 window의 flow_NLL)
  - raw(작업 B3):    results/Paderborn/<raw_batch_dir>/<raw_prefix>_<split>_LONO<n>_s<seed>/model.pth
    (amp_normalize=False → 원 window의 flow_NLL)

점수 = **순수 shape flow_NLL(rms_lambda=0)**. F-1은 shape 표현 자체를 평가하므로 D의 penalty 결합
score는 쓰지 않는다(rmsnorm의 flow-only가 곧 F-1 점수 = D 리포트 flow_only와 동일 정의).

fold별 산출(rmsnorm/raw 각각 + Δ):
  1) 분리 AUROC: target-normal(0) vs fault 전체(1).
  2) per-fault AUROC: target-normal vs 각 fault id(KA/KI/KB). KA15·KA22 명시.
  3) ρ(원 RMS, flow_NLL): 정상 pooled(train+val+target-normal), Spearman.
  4) 정상 FPR: threshold=val-normal flow_NLL 95p, per-bearing → 고/저진폭 그룹 평균.
  5) 저진폭 target 역전: 저진폭 target fold(LONO2/4/5)에서 raw<0.5 → rmsnorm 회복 여부.

집계: zero-support/compositional 분리, LONO-1(고진폭)/LONO-2(저진폭) 명시 분해, per-fault(KA15/KA22),
raw 대비 AUROC Δ.

산출:
- results/Paderborn/diag_F1_evaluation/<split>_LONO<n>_s<seed>.json (fold별)
- results/Paderborn/diag_F1_evaluation/aggregate_s<seed>.json
- reports/report_F1_evaluation.md
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.MTGFLOW import MTGFLOW  # noqa: E402
from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402

DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "Paderborn"))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_F1_evaluation.md")

HIGH_AMP = {"K001", "K003", "K006"}  # 고진폭 정상(작업 B). 나머지 K00x = 저진폭.

# LOSO split ↔ (설명, fold 유형). 작업 B/D와 동일 규약(뭉뚱그리지 않음).
SPLIT_INFO = {
    "123to0": ("기준조건(N15_M07_F10) unseen", "compositional"),
    "023to1": ("저속(N09) unseen", "zero-support"),
    "013to2": ("저토크(M01) unseen", "zero-support"),
    "012to3": ("저 radial force(F04) unseen", "zero-support"),
}

# LONO idx → target-normal bearing (MTGFLOW/CLAUDE.md §4 표). 고/저진폭은 HIGH_AMP로 판정.
LONO_TARGET = {1: "K001", 2: "K002", 3: "K003", 4: "K004", 5: "K005", 6: "K006"}


# ---------------------------------------------------------------------------
# 모델·로더 복원 (diagnose_amp_correction.py 규약 준용, 자립형)
# ---------------------------------------------------------------------------
def build_model(ckpt, n_sensor, device):
    a = ckpt.get("args", {})
    meta = ckpt["paderborn_metadata"]
    model = MTGFLOW(
        n_blocks=int(a.get("n_blocks", 2)),
        input_size=int(a.get("input_size", 1)),
        hidden_size=int(a.get("hidden_size", 32)),
        n_hidden=int(a.get("n_hidden", 1)),
        window_size=int(meta["window_size"]),
        n_sensor=n_sensor,
        dropout=0.0,
        model=a.get("model", "MAF"),
        batch_norm=bool(a.get("batch_norm", False)),
        use_meta=bool(meta["use_meta"]),
        meta_input_dim=int(meta.get("meta_input_dim", 3)),
        meta_emb_dim=int(meta.get("meta_emb_dim", 8)),
        meta_inject=meta.get("meta_inject", "concat"),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def build_loader(meta, batch_size):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_Paderborn_OCC(
            root=DATA_ROOT,
            train_loads=meta["train_load_setting"],
            test_loads=meta["test_load_setting"],
            train_ids=meta["train_ids"],
            val_ids=meta["val_ids"],
            test_norm_ids=meta["test_norm_ids"],
            exclude_ids=meta.get("exclude_ids", []),
            window_size=int(meta["window_size"]),
            stride_size=int(meta["stride_size"]),
            sensor_mode=meta["sensor_mode"],
            meta_source=meta.get("meta_source", "static"),
            measured_meta_stats=meta.get("measured_meta_stats", "meanstd"),
            amp_normalize=bool(meta.get("amp_normalize", False)),
            rms_eps=float(meta.get("rms_eps", 1e-8)),
            batch_size=batch_size,
        )


@torch.no_grad()
def flow_nll(model, windows, device, batch_size=256):
    """windows: [N, win] → flow_NLL(=-log p_x) per window [N]. (단일 채널, K=1,D=1)"""
    n = len(windows)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    win = windows.shape[1]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.as_tensor(windows[s:e], dtype=torch.float32, device=device)
        xb = xb.reshape(e - s, win, 1, 1).transpose(1, 2).contiguous()  # [b,K=1,L,D=1]
        out[s:e] = (-model.test(xb, None)).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# 통계 유틸 (작업 B/D와 동일 정의)
# ---------------------------------------------------------------------------
def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def safe_auroc(labels, scores):
    labels = np.asarray(labels, int)
    if len(labels) == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fmt(v, p=3):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{p}f}"


# ---------------------------------------------------------------------------
# 한 variant(rmsnorm 또는 raw) 재추론 → 지표 dict
# ---------------------------------------------------------------------------
def eval_variant(batch_dir, prefix, split, lono, seed, device, batch_size, thr_pct):
    run_name = f"{prefix}_{split}_LONO{lono}_s{seed}"
    ckpt_path = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        print(f"    [skip] checkpoint 없음: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert not bool(meta["use_meta"]), f"no-meta 전용 평가인데 use_meta=True: {run_name}"

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device)

    def pull(ds):
        return (np.asarray(ds.windows, dtype=np.float32),
                np.asarray(ds.rms_raw, dtype=np.float64),
                np.asarray(ds.ids),
                np.asarray(ds.label, dtype=int))

    tr_w, tr_rms, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, te_ids, te_lab = pull(test_loader.dataset)

    tr_f = flow_nll(model, tr_w, device, batch_size)
    va_f = flow_nll(model, va_w, device, batch_size)
    te_f = flow_nll(model, te_w, device, batch_size)

    tgt = te_lab == 0
    flt = te_lab == 1

    # (1) 분리 AUROC: target-normal(0) vs fault 전체(1)
    auroc = safe_auroc(te_lab, te_f)

    # (2) per-fault AUROC: target-normal vs 각 fault id
    tgt_f = te_f[tgt]
    per_fault = {}
    for fid in sorted(set(te_ids[flt].tolist())):
        m = flt & (te_ids == fid)
        f_scores = te_f[m]
        labels = np.concatenate([np.zeros(len(tgt_f), int), np.ones(len(f_scores), int)])
        scores = np.concatenate([tgt_f, f_scores])
        per_fault[fid] = {"n": int(m.sum()), "auroc": safe_auroc(labels, scores)}

    # (3) ρ(원 RMS, flow_NLL): 정상 pooled(train+val+target-normal)
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_flow = np.concatenate([tr_f, va_f, te_f[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])
    rho = spearman(norm_rms, norm_flow)

    # (4) 정상 FPR: threshold=val-normal flow_NLL 95p, per-bearing → 고/저진폭 그룹 평균
    thr = float(np.percentile(va_f, thr_pct)) if len(va_f) else float("inf")
    per_bearing = {}
    for bid in sorted(set(norm_ids.tolist())):
        m = norm_ids == bid
        per_bearing[bid] = {
            "n": int(m.sum()),
            "group": "high" if bid in HIGH_AMP else "low",
            "mean_rms": float(np.mean(norm_rms[m])),
            "mean_flow_nll": float(np.mean(norm_flow[m])),
            "fpr": float(np.mean(norm_flow[m] >= thr)),
        }

    def _grp_fpr(grp):
        vals = [v["fpr"] for v in per_bearing.values() if v["group"] == grp]
        return float(np.mean(vals)) if vals else float("nan")

    tgt_bid = LONO_TARGET.get(lono)
    return {
        "run_name": run_name,
        "amp_normalize": bool(meta.get("amp_normalize", False)),
        "auroc": auroc,
        "per_fault_auroc": per_fault,
        "rho_rms_flow": rho,
        "threshold_val95": thr,
        "high_amp_fpr": _grp_fpr("high"),
        "low_amp_fpr": _grp_fpr("low"),
        "per_bearing_normal": per_bearing,
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "n": {"train": int(len(tr_w)), "val": int(len(va_w)),
              "target_norm": int(tgt.sum()), "fault": int(flt.sum())},
    }


# ---------------------------------------------------------------------------
# fold 하나 = rmsnorm + raw paired
# ---------------------------------------------------------------------------
def process_fold(cfg, split, lono, seed, device):
    rms = eval_variant(cfg["rmsnorm_batch_dir"], cfg["rmsnorm_prefix"], split, lono,
                       seed, device, cfg["batch_size"], cfg["thr_pct"])
    raw = eval_variant(cfg["raw_batch_dir"], cfg["raw_prefix"], split, lono,
                       seed, device, cfg["batch_size"], cfg["thr_pct"])
    if rms is None and raw is None:
        return None

    def _auroc(v):
        return v["auroc"] if v else float("nan")

    d_auroc = (_auroc(rms) - _auroc(raw)) if (rms and raw) else float("nan")
    return {
        "split": split, "lono": lono, "seed": seed,
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_norm_ids": (rms or raw)["target_norm_ids"],
        "target_amp_group": (rms or raw)["target_amp_group"],
        "rmsnorm": rms, "raw": raw,
        "delta_auroc_rmsnorm_minus_raw": d_auroc,
    }


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------
def _mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(folds):
    ok = [f for f in folds if f]
    by_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ftype]
        if not rows:
            continue
        by_type[ftype] = {
            "n_folds": len(rows),
            "auroc_rmsnorm": _mean([r["rmsnorm"]["auroc"] for r in rows if r["rmsnorm"]]),
            "auroc_raw": _mean([r["raw"]["auroc"] for r in rows if r["raw"]]),
            "delta_auroc": _mean([r["delta_auroc_rmsnorm_minus_raw"] for r in rows]),
            "rho_rmsnorm": _mean([r["rmsnorm"]["rho_rms_flow"] for r in rows if r["rmsnorm"]]),
            "rho_raw": _mean([r["raw"]["rho_rms_flow"] for r in rows if r["raw"]]),
            "high_fpr_rmsnorm": _mean([r["rmsnorm"]["high_amp_fpr"] for r in rows if r["rmsnorm"]]),
            "high_fpr_raw": _mean([r["raw"]["high_amp_fpr"] for r in rows if r["raw"]]),
            "low_fpr_rmsnorm": _mean([r["rmsnorm"]["low_amp_fpr"] for r in rows if r["rmsnorm"]]),
            "low_fpr_raw": _mean([r["raw"]["low_amp_fpr"] for r in rows if r["raw"]]),
        }

    # target 진폭군별(고진폭/저진폭) 분리 AUROC — 저진폭 target 역전 진단
    by_amp = {}
    for grp in ("high", "low"):
        rows = [f for f in ok if f["target_amp_group"] == grp]
        if not rows:
            continue
        by_amp[grp] = {
            "n_folds": len(rows),
            "auroc_rmsnorm": _mean([r["rmsnorm"]["auroc"] for r in rows if r["rmsnorm"]]),
            "auroc_raw": _mean([r["raw"]["auroc"] for r in rows if r["raw"]]),
            "n_reversed_raw": int(sum(1 for r in rows if r["raw"] and r["raw"]["auroc"] < 0.5)),
            "n_reversed_rmsnorm": int(sum(1 for r in rows if r["rmsnorm"] and r["rmsnorm"]["auroc"] < 0.5)),
        }

    # per-fault AUROC 집계(fault id별 fold 평균) — rmsnorm/raw
    fault_agg = {}
    for f in ok:
        for variant in ("rmsnorm", "raw"):
            v = f[variant]
            if not v:
                continue
            for fid, d in v["per_fault_auroc"].items():
                e = fault_agg.setdefault(fid, {"rmsnorm": [], "raw": []})
                e[variant].append(d["auroc"])
    fault_summary = {
        fid: {"auroc_rmsnorm": _mean(e["rmsnorm"]), "auroc_raw": _mean(e["raw"]),
              "n_folds_rmsnorm": len([x for x in e["rmsnorm"] if x == x])}
        for fid, e in sorted(fault_agg.items())
    }

    return {"n_folds": len(ok), "by_fold_type": by_type,
            "by_target_amp": by_amp, "per_fault": fault_summary}


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def build_report(folds, agg, seed):
    ok = [f for f in folds if f]
    L = []
    L.append("# 작업 F-1 정밀 재평가 — Vy-rmsnorm(shape) vs raw-vib (Paderborn no-meta, 무학습 재추론)")
    L.append("")
    L.append(f"seed {seed}, window 2048, no-meta LOSO 4 split × 6 LONO. 학습된 체크포인트 재추론만"
             "(새 학습 없음). 점수 = 순수 shape flow_NLL(rms_lambda=0). "
             "rmsnorm=작업 D(amp_normalize=True) / raw=작업 B3(amp_normalize=False), 동일 fold paired 비교.")
    L.append("")
    L.append("**F-1 배경(계획 B안):** F-1의 표현·채널·split·seed·window는 작업 D와 동일해 D 체크포인트를 재사용한다. "
             "F-0이 권한 window 16384/32768은 **적용하지 않는다** — (i) MTGFlow attention이 Linear(c,c), "
             "c=window_size라 파라미터가 window에 제곱으로 커져(16384→~805M, 32768→~3.2B) V100-16GB 학습 불가, "
             "(ii) F-0의 큰-window 근거는 envelope-spectrum **descriptor**의 주파수 해상도용이며 raw time-window를 "
             "그대로 LSTM+flow에 넣는 MTGFlow 학습엔 자동 이식되지 않는다. patchify 구조 변경은 F-1 범위 밖.")
    L.append("")

    # ---- 핵심 요약(자동) ----
    zt = agg["by_fold_type"].get("zero-support", {})
    ct = agg["by_fold_type"].get("compositional", {})
    hi = agg["by_target_amp"].get("high", {})
    lo = agg["by_target_amp"].get("low", {})
    ka15 = agg["per_fault"].get("KA15", {})
    ka22 = agg["per_fault"].get("KA22", {})
    L.append("## 핵심 요약 (F-1 검증 목표별)")
    L.append("")
    L.append(f"- **분리 AUROC (raw 대비)**: zero-support rmsnorm {fmt(zt.get('auroc_rmsnorm'))} vs raw "
             f"{fmt(zt.get('auroc_raw'))} (Δ {fmt(zt.get('delta_auroc'))}); compositional rmsnorm "
             f"{fmt(ct.get('auroc_rmsnorm'))} vs raw {fmt(ct.get('auroc_raw'))} (Δ {fmt(ct.get('delta_auroc'))}).")
    L.append(f"- **저진폭 target 역전**: 저진폭 target fold(LONO2/4/5) raw AUROC {fmt(lo.get('auroc_raw'))}"
             f"(역전 fold {lo.get('n_reversed_raw')}/{lo.get('n_folds')}개) → rmsnorm {fmt(lo.get('auroc_rmsnorm'))}"
             f"(역전 {lo.get('n_reversed_rmsnorm')}/{lo.get('n_folds')}개). "
             f"고진폭 target(LONO1/3/6): raw {fmt(hi.get('auroc_raw'))} / rmsnorm {fmt(hi.get('auroc_rmsnorm'))}.")
    L.append(f"- **ρ(RMS, flow_NLL) 하락 유지**: zero-support raw {fmt(zt.get('rho_raw'))}→rmsnorm "
             f"{fmt(zt.get('rho_rmsnorm'))}; compositional raw {fmt(ct.get('rho_raw'))}→rmsnorm "
             f"{fmt(ct.get('rho_rmsnorm'))} (작업 B 기준선 ρ≈0.96).")
    L.append(f"- **KA15·KA22 per-fault AUROC**: KA15 raw {fmt(ka15.get('auroc_raw'))}→rmsnorm "
             f"{fmt(ka15.get('auroc_rmsnorm'))}; KA22 raw {fmt(ka22.get('auroc_raw'))}→rmsnorm "
             f"{fmt(ka22.get('auroc_rmsnorm'))} (0.5 근방=불가시).")
    L.append(f"- **정상 FPR 악화 여부**: 고진폭 raw {fmt(zt.get('high_fpr_raw'))}/{fmt(ct.get('high_fpr_raw'))}"
             f"(zero/comp)→rmsnorm {fmt(zt.get('high_fpr_rmsnorm'))}/{fmt(ct.get('high_fpr_rmsnorm'))}; "
             f"저진폭 raw {fmt(zt.get('low_fpr_raw'))}/{fmt(ct.get('low_fpr_raw'))}→rmsnorm "
             f"{fmt(zt.get('low_fpr_rmsnorm'))}/{fmt(ct.get('low_fpr_rmsnorm'))}.")
    L.append("")

    # ---- 1) fold 유형별 집계 ----
    L.append("## 1) fold 유형별 집계 — rmsnorm vs raw")
    L.append("")
    L.append("| fold 유형 | n | AUROC(rmsnorm) | AUROC(raw) | Δ | ρ(rmsnorm) | ρ(raw) | "
             "고진폭FPR(rms/raw) | 저진폭FPR(rms/raw) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ftype, a in agg["by_fold_type"].items():
        L.append(f"| {ftype} | {a['n_folds']} | **{fmt(a['auroc_rmsnorm'])}** | {fmt(a['auroc_raw'])} | "
                 f"{fmt(a['delta_auroc'])} | {fmt(a['rho_rmsnorm'])} | {fmt(a['rho_raw'])} | "
                 f"{fmt(a['high_fpr_rmsnorm'])}/{fmt(a['high_fpr_raw'])} | "
                 f"{fmt(a['low_fpr_rmsnorm'])}/{fmt(a['low_fpr_raw'])} |")
    L.append("")

    # ---- 2) target 진폭군별(저진폭 역전) ----
    L.append("## 2) target-normal 진폭군별 분리 AUROC — 저진폭 역전 진단")
    L.append("")
    L.append("| target 진폭군 | n | AUROC(rmsnorm) | AUROC(raw) | raw 역전(<0.5) | rmsnorm 역전(<0.5) |")
    L.append("|---|---|---|---|---|---|")
    for grp, a in agg["by_target_amp"].items():
        gname = "고진폭(K001/K003/K006)" if grp == "high" else "저진폭(K002/K004/K005)"
        L.append(f"| {gname} | {a['n_folds']} | {fmt(a['auroc_rmsnorm'])} | {fmt(a['auroc_raw'])} | "
                 f"{a['n_reversed_raw']} | {a['n_reversed_rmsnorm']} |")
    L.append("")

    # ---- 3) LONO-1(고진폭)·LONO-2(저진폭) 명시 분해 ----
    L.append("## 3) LONO-1(고진폭 K001 target) · LONO-2(저진폭 K002 target) 분해")
    L.append("")
    L.append("| split | LONO | target | AUROC(rmsnorm) | AUROC(raw) | Δ | ρ(rms) | ρ(raw) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in ok:
        if f["lono"] not in (1, 2):
            continue
        rms, raw = f["rmsnorm"], f["raw"]
        L.append(f"| {f['split']} | {f['lono']} | {','.join(f['target_norm_ids'])} | "
                 f"**{fmt(rms['auroc']) if rms else '—'}** | {fmt(raw['auroc']) if raw else '—'} | "
                 f"{fmt(f['delta_auroc_rmsnorm_minus_raw'])} | "
                 f"{fmt(rms['rho_rms_flow']) if rms else '—'} | {fmt(raw['rho_rms_flow']) if raw else '—'} |")
    L.append("")

    # ---- 4) per-fault AUROC 집계 ----
    L.append("## 4) per-fault AUROC 집계 (fault id별 fold 평균) — KA15·KA22 강조")
    L.append("")
    L.append("| fault id | fold수 | AUROC(rmsnorm) | AUROC(raw) | Δ |")
    L.append("|---|---|---|---|---|")
    for fid, d in agg["per_fault"].items():
        star = " ⚠" if fid in ("KA15", "KA22") else ""
        dd = (d["auroc_rmsnorm"] - d["auroc_raw"]) if (d["auroc_rmsnorm"] == d["auroc_rmsnorm"]
                                                       and d["auroc_raw"] == d["auroc_raw"]) else float("nan")
        L.append(f"| {fid}{star} | {d['n_folds_rmsnorm']} | {fmt(d['auroc_rmsnorm'])} | "
                 f"{fmt(d['auroc_raw'])} | {fmt(dd)} |")
    L.append("")
    L.append("> ⚠ = 작업 B/F-0에서 불가시(AUROC≈0.5) 이력이 있는 KA15·KA22. 0.5 근방이면 학습 후에도 불가시.")
    L.append("")

    # ---- 5) fold 상세 ----
    L.append("## 5) fold 전체 상세 (rmsnorm / raw AUROC)")
    L.append("")
    L.append("| split | LONO | 유형 | target | AUROC(rms) | AUROC(raw) | Δ |")
    L.append("|---|---|---|---|---|---|---|")
    for f in ok:
        rms, raw = f["rmsnorm"], f["raw"]
        L.append(f"| {f['split']} | {f['lono']} | {f['fold_type']} | {','.join(f['target_norm_ids'])} | "
                 f"{fmt(rms['auroc']) if rms else '—'} | {fmt(raw['auroc']) if raw else '—'} | "
                 f"{fmt(f['delta_auroc_rmsnorm_minus_raw'])} |")
    L.append("")

    L.append("## 주의 / 한계 (F-1 위험 반영)")
    L.append("")
    L.append("- 무학습 재추론, seed 2026 단일. rmsnorm AUROC는 D 리포트 flow_only와 동일 정의(정합 확인용).")
    L.append("- **target-normal이 setting+bearing 이중 held-out** → bearing 개체차 confound 잔존. 높은 AUROC "
             "일부는 fault가 아니라 진폭군·개체차 분리일 수 있음(F-0·작업 B와 정합).")
    L.append("- F-0의 window 16384/32768은 envelope-spectrum descriptor용이라 MTGFlow raw window 학습에 직접 "
             "적용하지 않음(§배경). patchify 구조 변경은 F-1 범위 밖(F-2 이후 별도 검토 가능).")
    L.append("- KA15·KA22 per-fault AUROC가 0.5 근방이면 표현·정규화와 무관하게 밀도모델이 못 가르는 결함(§4).")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rmsnorm_batch_dir", type=str, default="LONO_D_ampnorm_s2026")
    ap.add_argument("--rmsnorm_prefix", type=str, default="ampnorm")
    ap.add_argument("--raw_batch_dir", type=str, default="LONO_B3_5seeds")
    ap.add_argument("--raw_prefix", type=str, default="raw_vib")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=os.path.join(RESULTS_ROOT, "diag_F1_evaluation"))
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = {
        "rmsnorm_batch_dir": args.rmsnorm_batch_dir, "rmsnorm_prefix": args.rmsnorm_prefix,
        "raw_batch_dir": args.raw_batch_dir, "raw_prefix": args.raw_prefix,
        "batch_size": args.batch_size, "thr_pct": args.threshold_percentile,
    }

    folds = []
    for split in args.splits:
        for lono in args.lonos:
            print(f"[{split} LONO{lono}]")
            res = process_fold(cfg, split, lono, args.seed, device)
            if res is None:
                continue
            folds.append(res)
            with open(os.path.join(args.out_dir, f"{split}_LONO{lono}_s{args.seed}.json"), "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
            rms, raw = res["rmsnorm"], res["raw"]
            print(f"    AUROC rms={fmt(rms['auroc']) if rms else '—'} "
                  f"raw={fmt(raw['auroc']) if raw else '—'} "
                  f"Δ={fmt(res['delta_auroc_rmsnorm_minus_raw'])} "
                  f"ρ(rms)={fmt(rms['rho_rms_flow']) if rms else '—'}")

    if args.no_report or not folds:
        print(f"\nfold JSON {len(folds)}개 기록. (집계/리포트 생략)")
        return

    agg = aggregate(folds)
    with open(os.path.join(args.out_dir, f"aggregate_s{args.seed}.json"), "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    with open(REPORT_PATH, "w") as f:
        f.write(build_report(folds, agg, args.seed))
    print(f"\nwrote {REPORT_PATH}")
    print("\n=== fold 유형별 집계 ===")
    for ftype, a in agg["by_fold_type"].items():
        print(f"  {ftype:>13} n={a['n_folds']} AUROC rms={fmt(a['auroc_rmsnorm'])} "
              f"raw={fmt(a['auroc_raw'])} Δ={fmt(a['delta_auroc'])} "
              f"ρ rms={fmt(a['rho_rmsnorm'])} raw={fmt(a['rho_raw'])}")


if __name__ == "__main__":
    main()
