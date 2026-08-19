"""작업 G-1 정밀 평가 — dual-branch shape/amplitude disentangle (Paderborn no-meta, 무학습 재추론).

G-1 체크포인트(g1_<split>_LONO<n>_s<seed>, amp_normalize=True + amp_branch=True)를 재추론해
per-window S_shape(=-shape_logprob), S_amp(=-amp_logprob)를 분리 계산하고,
val-normal 표준화 등가중 합 S_total = z_val(S_shape)+z_val(S_amp)(label-free)을 만든다.
동일 seed raw baseline(raw_vib_<split>_LONO<n>_s<seed>, amp_normalize=False)을 flow_NLL로 재추론해
paired 비교한다. (test 라벨을 이용한 가중치/threshold 튜닝 없음.)

go/no-go (TODO G-1 ①~⑤, 모델에서 직접 읽음):
 ① 고/저진폭 정상 모두 S_amp가 낮은가 (고진폭 정상에서 S_amp 과도 상승 없는지 — 정상 FPR·mean 비교)
 ② amplitude-sensitive fault(raw가 잡고 rmsnorm이 잃은 것)가 S_amp로 회복되는가
 ③ shape-sensitive fault(raw가 못 잡고 rmsnorm이 회복한 것)가 S_shape에서 유지되는가
 ④ 두 branch 결합(S_total)이 한 fault군을 희생하지 않는가
 ⑤ 동일 seed raw baseline 대비 S_total AUROC/FPR paired 개선

추가: ρ(원 RMS, S_shape/S_amp/S_total) Spearman(정상 pool), 정상 FPR 진폭군별, fault별 AUROC.

산출:
- results/Paderborn/diag_G1_disentangle/<split>_LONO<n>_s<seed>.json (fold별)
- results/Paderborn/diag_G1_disentangle/aggregate_s<seed>.json
- reports/report_G1_disentangle.md
"""
import argparse
import contextlib
import io
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
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_G1_disentangle.md")

HIGH_AMP = {"K001", "K003", "K006"}  # 고진폭 정상(작업 B). 나머지 K00x = 저진폭.

SPLIT_INFO = {
    "123to0": ("기준조건(N15_M07_F10) unseen", "compositional"),
    "023to1": ("저속(N09) unseen", "zero-support"),
    "013to2": ("저토크(M01) unseen", "zero-support"),
    "012to3": ("저 radial force(F04) unseen", "zero-support"),
}
LONO_TARGET = {1: "K001", 2: "K002", 3: "K003", 4: "K004", 5: "K005", 6: "K006"}

# F-1 §4 트레이드오프: raw가 잡고 rmsnorm이 잃는(amplitude-sensitive) / raw가 못 잡고 rmsnorm이 회복(shape-sensitive)
AMP_SENSITIVE = {"KA04", "KA16", "KA30", "KB23", "KB24", "KI04", "KI16", "KI18"}
SHAPE_SENSITIVE = {"KA15", "KA22", "KB27", "KI14", "KI17", "KI21"}


# ---------------------------------------------------------------------------
# 모델·로더 복원 (F-1 규약 준용)
# ---------------------------------------------------------------------------
def build_model(ckpt, n_sensor, device, amp_branch):
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
        amp_branch=amp_branch,
        amp_branch_hidden=int(meta.get("amp_branch_hidden", 32)),
    )
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def build_loader(meta, batch_size):
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


def _reshape_windows(windows, s, e, win, device):
    xb = torch.as_tensor(windows[s:e], dtype=torch.float32, device=device)
    return xb.reshape(e - s, win, 1, 1).transpose(1, 2).contiguous()  # [b,K=1,L,D=1]


@torch.no_grad()
def branch_scores(model, windows, rms_z, device, batch_size=256):
    """G-1: windows[N,win], rms_z[N,1] → (S_shape[N], S_amp[N]). S=-log p (클수록 이상)."""
    n = len(windows)
    if n == 0:
        return np.empty(0), np.empty(0)
    win = windows.shape[1]
    s_shape = np.empty(n, dtype=np.float64)
    s_amp = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = _reshape_windows(windows, s, e, win, device)
        ab = torch.as_tensor(rms_z[s:e], dtype=torch.float32, device=device).reshape(e - s, 1)
        sl, al = model.forward_disentangle(xb, None, ab)
        s_shape[s:e] = (-sl).cpu().numpy()
        s_amp[s:e] = (-al).cpu().numpy()
    return s_shape, s_amp


@torch.no_grad()
def flow_nll(model, windows, device, batch_size=256):
    """raw baseline: windows[N,win] → flow_NLL(=-log p_x)[N]."""
    n = len(windows)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    win = windows.shape[1]
    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = _reshape_windows(windows, s, e, win, device)
        out[s:e] = (-model.test(xb, None)).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# 통계 유틸 (F-1과 동일 정의)
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


def pull(ds):
    return (np.asarray(ds.windows, dtype=np.float32),
            np.asarray(ds.rms_raw, dtype=np.float64),
            np.asarray(ds.rms_z, dtype=np.float32),
            np.asarray(ds.ids),
            np.asarray(ds.label, dtype=int))


def _score_block(name, S_test, S_val, te_lab, te_ids, te_rms, tr_S, va_S, tgt_rms_ref,
                 norm_rms, norm_ids, thr_pct):
    """한 branch 점수 S에 대한 지표(분리 AUROC, per-fault, rho, 정상 FPR 진폭군별)."""
    tgt = te_lab == 0
    flt = te_lab == 1
    auroc = safe_auroc(te_lab, S_test)

    tgt_S = S_test[tgt]
    per_fault = {}
    for fid in sorted(set(te_ids[flt].tolist())):
        m = flt & (te_ids == fid)
        labels = np.concatenate([np.zeros(int(tgt.sum()), int), np.ones(int(m.sum()), int)])
        scores = np.concatenate([tgt_S, S_test[m]])
        per_fault[fid] = {"n": int(m.sum()), "auroc": safe_auroc(labels, scores)}

    # ρ(원 RMS, S): 정상 pool(train+val+target-normal) — rank 기반이라 표준화 무관
    norm_S = np.concatenate([tr_S, va_S, tgt_S])
    rho = spearman(norm_rms, norm_S)

    # 정상 FPR: threshold=val-normal S의 thr_pct, per-bearing → 고/저진폭 그룹 평균
    thr = float(np.percentile(S_val, thr_pct)) if len(S_val) else float("inf")
    per_bearing = {}
    for bid in sorted(set(norm_ids.tolist())):
        m = norm_ids == bid
        per_bearing[bid] = {
            "n": int(m.sum()),
            "group": "high" if bid in HIGH_AMP else "low",
            "mean_rms": float(np.mean(norm_rms[m])),
            "mean_score": float(np.mean(norm_S[m])),
            "fpr": float(np.mean(norm_S[m] >= thr)),
        }

    def _grp(key, grp):
        vals = [v[key] for v in per_bearing.values() if v["group"] == grp]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "name": name,
        "auroc": auroc,
        "per_fault_auroc": per_fault,
        "rho_rms_score": rho,
        "threshold_val": thr,
        "high_amp_fpr": _grp("fpr", "high"),
        "low_amp_fpr": _grp("fpr", "low"),
        "high_amp_mean_score": _grp("mean_score", "high"),
        "low_amp_mean_score": _grp("mean_score", "low"),
        "per_bearing_normal": per_bearing,
    }


# ---------------------------------------------------------------------------
# G-1 variant 재추론 → S_shape/S_amp/S_total 지표
# ---------------------------------------------------------------------------
def eval_g1(batch_dir, split, lono, seed, device, batch_size, thr_pct):
    run_name = f"g1_{split}_LONO{lono}_s{seed}"
    ckpt_path = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        # 배치 폴더 없이 바로 results/Paderborn/<run>/ 에 있을 수도 있음
        alt = os.path.join(RESULTS_ROOT, run_name, "model.pth")
        ckpt_path = alt if os.path.exists(alt) else ckpt_path
    if not os.path.exists(ckpt_path):
        print(f"    [skip] G-1 checkpoint 없음: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert bool(meta.get("amp_branch", False)), f"amp_branch=False 인 checkpoint: {run_name}"
    assert bool(meta.get("amp_normalize", False)), f"amp_normalize=False (shape window 아님): {run_name}"
    assert not bool(meta["use_meta"]), f"no-meta 전용인데 use_meta=True: {run_name}"

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device, amp_branch=True)

    tr_w, tr_rms, tr_z, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, va_z, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, te_z, te_ids, te_lab = pull(test_loader.dataset)

    tr_sh, tr_am = branch_scores(model, tr_w, tr_z, device, batch_size)
    va_sh, va_am = branch_scores(model, va_w, va_z, device, batch_size)
    te_sh, te_am = branch_scores(model, te_w, te_z, device, batch_size)

    # val-normal 표준화(label-free)
    mu_s, sd_s = float(va_sh.mean()), float(va_sh.std() + 1e-8)
    mu_a, sd_a = float(va_am.mean()), float(va_am.std() + 1e-8)

    def ztot(sh, am):
        return (sh - mu_s) / sd_s + (am - mu_a) / sd_a

    tr_to, va_to, te_to = ztot(tr_sh, tr_am), ztot(va_sh, va_am), ztot(te_sh, te_am)

    tgt = te_lab == 0
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])

    blocks = {
        "shape": _score_block("shape", te_sh, va_sh, te_lab, te_ids, te_rms, tr_sh, va_sh,
                              te_rms[tgt], norm_rms, norm_ids, thr_pct),
        "amp": _score_block("amp", te_am, va_am, te_lab, te_ids, te_rms, tr_am, va_am,
                            te_rms[tgt], norm_rms, norm_ids, thr_pct),
        "total": _score_block("total", te_to, va_to, te_lab, te_ids, te_rms, tr_to, va_to,
                              te_rms[tgt], norm_rms, norm_ids, thr_pct),
    }
    tgt_bid = LONO_TARGET.get(lono)
    return {
        "run_name": run_name,
        "val_norm_stats": {"mu_shape": mu_s, "sd_shape": sd_s, "mu_amp": mu_a, "sd_amp": sd_a},
        "auroc_shape": blocks["shape"]["auroc"],
        "auroc_amp": blocks["amp"]["auroc"],
        "auroc_total": blocks["total"]["auroc"],
        "blocks": blocks,
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "n": {"train": int(len(tr_w)), "val": int(len(va_w)),
              "target_norm": int(tgt.sum()), "fault": int((te_lab == 1).sum())},
    }


# ---------------------------------------------------------------------------
# raw baseline 재추론 (F-1 eval_variant와 동일 정의)
# ---------------------------------------------------------------------------
def eval_raw(batch_dir, prefix, split, lono, seed, device, batch_size, thr_pct):
    run_name = f"{prefix}_{split}_LONO{lono}_s{seed}"
    ckpt_path = os.path.join(RESULTS_ROOT, batch_dir, run_name, "model.pth")
    if not os.path.exists(ckpt_path):
        alt = os.path.join(RESULTS_ROOT, run_name, "model.pth")
        ckpt_path = alt if os.path.exists(alt) else ckpt_path
    if not os.path.exists(ckpt_path):
        print(f"    [skip] raw checkpoint 없음: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device, amp_branch=False)

    tr_w, tr_rms, _, tr_ids, _ = pull(train_loader.dataset)
    va_w, va_rms, _, va_ids, _ = pull(val_loader.dataset)
    te_w, te_rms, _, te_ids, te_lab = pull(test_loader.dataset)

    tr_f = flow_nll(model, tr_w, device, batch_size)
    va_f = flow_nll(model, va_w, device, batch_size)
    te_f = flow_nll(model, te_w, device, batch_size)

    tgt = te_lab == 0
    norm_rms = np.concatenate([tr_rms, va_rms, te_rms[tgt]])
    norm_ids = np.concatenate([tr_ids, va_ids, te_ids[tgt]])
    blk = _score_block("raw", te_f, va_f, te_lab, te_ids, te_rms, tr_f, va_f,
                       te_rms[tgt], norm_rms, norm_ids, thr_pct)
    return {"run_name": run_name, "auroc": blk["auroc"], "block": blk}


# ---------------------------------------------------------------------------
# fold 하나 = G-1(shape/amp/total) + raw paired
# ---------------------------------------------------------------------------
def process_fold(cfg, split, lono, seed, device):
    g1 = eval_g1(cfg["g1_batch_dir"], split, lono, seed, device, cfg["batch_size"], cfg["thr_pct"])
    raw = eval_raw(cfg["raw_batch_dir"], cfg["raw_prefix"], split, lono, seed, device,
                   cfg["batch_size"], cfg["thr_pct"])
    if g1 is None:
        return None
    d_total_raw = (g1["auroc_total"] - raw["auroc"]) if raw else float("nan")
    return {
        "split": split, "lono": lono, "seed": seed,
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_norm_ids": g1["target_norm_ids"],
        "target_amp_group": g1["target_amp_group"],
        "g1": g1, "raw": raw,
        "delta_total_minus_raw": d_total_raw,
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
            "auroc_shape": _mean([r["g1"]["auroc_shape"] for r in rows]),
            "auroc_amp": _mean([r["g1"]["auroc_amp"] for r in rows]),
            "auroc_total": _mean([r["g1"]["auroc_total"] for r in rows]),
            "auroc_raw": _mean([r["raw"]["auroc"] for r in rows if r["raw"]]),
            "delta_total_raw": _mean([r["delta_total_minus_raw"] for r in rows]),
            "rho_shape": _mean([r["g1"]["blocks"]["shape"]["rho_rms_score"] for r in rows]),
            "rho_amp": _mean([r["g1"]["blocks"]["amp"]["rho_rms_score"] for r in rows]),
            "rho_total": _mean([r["g1"]["blocks"]["total"]["rho_rms_score"] for r in rows]),
            "rho_raw": _mean([r["raw"]["block"]["rho_rms_score"] for r in rows if r["raw"]]),
        }

    # ① 정상 S_amp 진폭군별 FPR/mean (고진폭 정상에서 과도 상승 없는지)
    amp_normal = {
        "high_fpr": _mean([r["g1"]["blocks"]["amp"]["high_amp_fpr"] for r in ok]),
        "low_fpr": _mean([r["g1"]["blocks"]["amp"]["low_amp_fpr"] for r in ok]),
        "high_mean": _mean([r["g1"]["blocks"]["amp"]["high_amp_mean_score"] for r in ok]),
        "low_mean": _mean([r["g1"]["blocks"]["amp"]["low_amp_mean_score"] for r in ok]),
    }
    total_normal = {
        "high_fpr": _mean([r["g1"]["blocks"]["total"]["high_amp_fpr"] for r in ok]),
        "low_fpr": _mean([r["g1"]["blocks"]["total"]["low_amp_fpr"] for r in ok]),
    }
    raw_normal = {
        "high_fpr": _mean([r["raw"]["block"]["high_amp_fpr"] for r in ok if r["raw"]]),
        "low_fpr": _mean([r["raw"]["block"]["low_amp_fpr"] for r in ok if r["raw"]]),
    }

    # per-fault AUROC 집계(fault id별 fold 평균) — shape/amp/total/raw
    fault_agg = {}
    for f in ok:
        for br in ("shape", "amp", "total"):
            for fid, d in f["g1"]["blocks"][br]["per_fault_auroc"].items():
                fault_agg.setdefault(fid, {"shape": [], "amp": [], "total": [], "raw": []})[br].append(d["auroc"])
        if f["raw"]:
            for fid, d in f["raw"]["block"]["per_fault_auroc"].items():
                fault_agg.setdefault(fid, {"shape": [], "amp": [], "total": [], "raw": []})["raw"].append(d["auroc"])
    per_fault = {
        fid: {"auroc_shape": _mean(e["shape"]), "auroc_amp": _mean(e["amp"]),
              "auroc_total": _mean(e["total"]), "auroc_raw": _mean(e["raw"]),
              "class": ("amp-sensitive" if fid in AMP_SENSITIVE else
                        "shape-sensitive" if fid in SHAPE_SENSITIVE else "other")}
        for fid, e in sorted(fault_agg.items())
    }

    # ②③④ fault군별 branch 평균
    def _grp_mean(fault_set, key):
        return _mean([per_fault[fid][key] for fid in per_fault if fid in fault_set])

    fault_group = {
        "amp_sensitive": {
            "auroc_raw": _grp_mean(AMP_SENSITIVE, "auroc_raw"),
            "auroc_shape": _grp_mean(AMP_SENSITIVE, "auroc_shape"),
            "auroc_amp": _grp_mean(AMP_SENSITIVE, "auroc_amp"),
            "auroc_total": _grp_mean(AMP_SENSITIVE, "auroc_total"),
        },
        "shape_sensitive": {
            "auroc_raw": _grp_mean(SHAPE_SENSITIVE, "auroc_raw"),
            "auroc_shape": _grp_mean(SHAPE_SENSITIVE, "auroc_shape"),
            "auroc_amp": _grp_mean(SHAPE_SENSITIVE, "auroc_amp"),
            "auroc_total": _grp_mean(SHAPE_SENSITIVE, "auroc_total"),
        },
    }

    overall = {
        "auroc_shape": _mean([r["g1"]["auroc_shape"] for r in ok]),
        "auroc_amp": _mean([r["g1"]["auroc_amp"] for r in ok]),
        "auroc_total": _mean([r["g1"]["auroc_total"] for r in ok]),
        "auroc_raw": _mean([r["raw"]["auroc"] for r in ok if r["raw"]]),
    }
    return {"n_folds": len(ok), "overall": overall, "by_fold_type": by_type,
            "amp_normal": amp_normal, "total_normal": total_normal, "raw_normal": raw_normal,
            "fault_group": fault_group, "per_fault": per_fault}


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------
def build_report(folds, agg, seed):
    ok = [f for f in folds if f]
    L = []
    L.append("# 작업 G-1 — Dual-branch Shape/Amplitude Disentangle (Paderborn no-meta, 무학습 재추론)")
    L.append("")
    L.append(f"seed {seed}, window 2048, no-meta LOSO 4 split × 6 LONO ({len(ok)} fold). "
             "G-1 = amp_normalize(shape-only window) + amp_branch(조건부 진폭 head, Joint 학습). "
             "S_shape=flow NLL, S_amp=-log N(a|h_shape), S_total=z_val(S_shape)+z_val(S_amp)(val-normal 표준화, label-free). "
             "raw baseline = 동일 seed B3 checkpoint flow_NLL, 동일 fold paired.")
    L.append("")

    ov = agg["overall"]
    L.append("## 핵심 요약 (go/no-go ①~⑤)")
    L.append("")
    L.append(f"- **전체 AUROC** (24 fold 평균): S_total {fmt(ov['auroc_total'])} / S_shape {fmt(ov['auroc_shape'])} / "
             f"S_amp {fmt(ov['auroc_amp'])} vs **raw {fmt(ov['auroc_raw'])}** (가드레일 no-meta LOSO 0.696).")
    an, tn, rn = agg["amp_normal"], agg["total_normal"], agg["raw_normal"]
    L.append(f"- **① 고진폭 정상 S_amp 과도 상승 여부**: 정상 FPR(S_amp) 고진폭 {fmt(an['high_fpr'])} vs 저진폭 "
             f"{fmt(an['low_fpr'])}; 정상 mean(S_amp) 고 {fmt(an['high_mean'])} / 저 {fmt(an['low_mean'])}. "
             "(고진폭에서만 FPR·mean이 튀면 ① 실패 = 조건부 진폭이 진폭 크기를 그대로 벌함.)")
    fg = agg["fault_group"]
    a_s, s_s = fg["amp_sensitive"], fg["shape_sensitive"]
    L.append(f"- **② amplitude-sensitive fault 회복(S_amp)**: raw {fmt(a_s['auroc_raw'])} → S_shape {fmt(a_s['auroc_shape'])} "
             f"(rmsnorm에서 잃음) → **S_amp {fmt(a_s['auroc_amp'])}** → S_total {fmt(a_s['auroc_total'])}. "
             "(S_amp가 raw 수준으로 회복하면 ② 성공.)")
    L.append(f"- **③ shape-sensitive fault 유지(S_shape)**: raw {fmt(s_s['auroc_raw'])} → **S_shape {fmt(s_s['auroc_shape'])}** "
             f"→ S_amp {fmt(s_s['auroc_amp'])} → S_total {fmt(s_s['auroc_total'])}. "
             "(S_shape가 rmsnorm 회복 수준을 유지하면 ③ 성공.)")
    L.append(f"- **④ 결합 희생 여부(S_total)**: amp-sensitive S_total {fmt(a_s['auroc_total'])} / shape-sensitive "
             f"S_total {fmt(s_s['auroc_total'])} — 두 군 모두 각 branch 최고치에 근접하면 ④ 성공(한쪽 희생 없음).")
    zt = agg["by_fold_type"].get("zero-support", {})
    ct = agg["by_fold_type"].get("compositional", {})
    L.append(f"- **⑤ raw 대비 paired**: S_total−raw Δ = zero-support {fmt(zt.get('delta_total_raw'))} / "
             f"compositional {fmt(ct.get('delta_total_raw'))}. 정상 FPR(S_total) 고 {fmt(tn['high_fpr'])}/저 {fmt(tn['low_fpr'])} "
             f"vs raw 고 {fmt(rn['high_fpr'])}/저 {fmt(rn['low_fpr'])}.")
    L.append(f"- **ρ(RMS, score) 완화**: zero-support S_shape {fmt(zt.get('rho_shape'))} / S_amp {fmt(zt.get('rho_amp'))} / "
             f"S_total {fmt(zt.get('rho_total'))} vs raw {fmt(zt.get('rho_raw'))} (작업 B raw ρ≈0.96).")
    L.append("")

    # 1) fold 유형별
    L.append("## 1) fold 유형별 집계")
    L.append("")
    L.append("| fold 유형 | n | S_total | S_shape | S_amp | raw | Δ(total−raw) | ρ_total | ρ_amp | ρ_raw |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ftype, a in agg["by_fold_type"].items():
        L.append(f"| {ftype} | {a['n_folds']} | **{fmt(a['auroc_total'])}** | {fmt(a['auroc_shape'])} | "
                 f"{fmt(a['auroc_amp'])} | {fmt(a['auroc_raw'])} | {fmt(a['delta_total_raw'])} | "
                 f"{fmt(a['rho_total'])} | {fmt(a['rho_amp'])} | {fmt(a['rho_raw'])} |")
    L.append("")

    # 2) fault군별 branch AUROC (②③④)
    L.append("## 2) fault군별 branch AUROC — ②(amp-sensitive) / ③(shape-sensitive) / ④(결합)")
    L.append("")
    L.append("| fault군 | raw | S_shape | S_amp | S_total |")
    L.append("|---|---|---|---|---|")
    L.append(f"| amp-sensitive (KA04/16/30·KB23/24·KI04/16/18) | {fmt(a_s['auroc_raw'])} | {fmt(a_s['auroc_shape'])} | "
             f"**{fmt(a_s['auroc_amp'])}** | {fmt(a_s['auroc_total'])} |")
    L.append(f"| shape-sensitive (KA15/22·KB27·KI14/17/21) | {fmt(s_s['auroc_raw'])} | **{fmt(s_s['auroc_shape'])}** | "
             f"{fmt(s_s['auroc_amp'])} | {fmt(s_s['auroc_total'])} |")
    L.append("")

    # 3) per-fault
    L.append("## 3) per-fault AUROC (fault id별 fold 평균)")
    L.append("")
    L.append("| fault id | 분류 | raw | S_shape | S_amp | S_total |")
    L.append("|---|---|---|---|---|---|")
    for fid, d in agg["per_fault"].items():
        tag = {"amp-sensitive": "A", "shape-sensitive": "S", "other": ""}[d["class"]]
        L.append(f"| {fid} | {tag} | {fmt(d['auroc_raw'])} | {fmt(d['auroc_shape'])} | "
                 f"{fmt(d['auroc_amp'])} | {fmt(d['auroc_total'])} |")
    L.append("")
    L.append("> 분류 A=amplitude-sensitive(raw가 잡고 rmsnorm이 잃음, S_amp로 회복 기대), "
             "S=shape-sensitive(raw가 못 잡고 rmsnorm이 회복, S_shape에서 유지 기대).")
    L.append("")

    # 4) ① 정상 S_amp 진폭군별
    L.append("## 4) ① 정상에서 S_amp 진폭군별 (고진폭 정상 과탐 여부)")
    L.append("")
    L.append("| 지표 | 고진폭(K001/K003/K006) | 저진폭(K002/K004/K005) |")
    L.append("|---|---|---|")
    L.append(f"| 정상 FPR(S_amp @val95p) | {fmt(an['high_fpr'])} | {fmt(an['low_fpr'])} |")
    L.append(f"| 정상 mean S_amp | {fmt(an['high_mean'])} | {fmt(an['low_mean'])} |")
    L.append(f"| 정상 FPR(S_total) | {fmt(tn['high_fpr'])} | {fmt(tn['low_fpr'])} |")
    L.append(f"| 정상 FPR(raw) | {fmt(rn['high_fpr'])} | {fmt(rn['low_fpr'])} |")
    L.append("")

    # 5) fold 상세
    L.append("## 5) fold 전체 상세")
    L.append("")
    L.append("| split | LONO | 유형 | target | S_total | S_shape | S_amp | raw | Δ(total−raw) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for f in ok:
        g = f["g1"]; raw = f["raw"]
        L.append(f"| {f['split']} | {f['lono']} | {f['fold_type']} | {','.join(f['target_norm_ids'])} | "
                 f"{fmt(g['auroc_total'])} | {fmt(g['auroc_shape'])} | {fmt(g['auroc_amp'])} | "
                 f"{fmt(raw['auroc']) if raw else '—'} | {fmt(f['delta_total_minus_raw'])} |")
    L.append("")

    L.append("## 주의 / 한계")
    L.append("")
    L.append("- 무학습 재추론, seed 2026 단일(가능성 확인용). 판정은 다지표(①~⑤) — AUROC 단일 금지.")
    L.append("- S_total = val-normal 표준화 등가중 합(test 라벨 튜닝 없음). 원시 합은 S_shape가 수천 스케일이라 "
             "S_amp가 묻히므로 표준화가 primary.")
    L.append("- **target-normal이 setting+bearing 이중 held-out** → bearing 개체차 confound 잔존(작업 B/F-1과 정합).")
    L.append("- 실패 조건(TODO): fault morphology가 고진폭스러우면 S_amp residual 안 뜸(특히 023→1). "
             "그 경우 shape branch에 latent 거리 병행(F-0 근거)이 후속 옵션.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--g1_batch_dir", type=str, default="LONO_G1_s2026")
    ap.add_argument("--raw_batch_dir", type=str, default="LONO_B3_5seeds")
    ap.add_argument("--raw_prefix", type=str, default="raw_vib")
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()), choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--threshold_percentile", type=float, default=95.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=os.path.join(RESULTS_ROOT, "diag_G1_disentangle"))
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = {
        "g1_batch_dir": args.g1_batch_dir,
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
            g = res["g1"]; raw = res["raw"]
            print(f"    AUROC total={fmt(g['auroc_total'])} shape={fmt(g['auroc_shape'])} "
                  f"amp={fmt(g['auroc_amp'])} raw={fmt(raw['auroc']) if raw else '—'} "
                  f"Δ={fmt(res['delta_total_minus_raw'])}")

    if args.no_report or not folds:
        print(f"\nfold JSON {len(folds)}개 기록. (집계/리포트 생략)")
        return

    agg = aggregate(folds)
    with open(os.path.join(args.out_dir, f"aggregate_s{args.seed}.json"), "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    with open(REPORT_PATH, "w") as f:
        f.write(build_report(folds, agg, args.seed))
    print(f"\nwrote {REPORT_PATH}")
    ov = agg["overall"]
    print(f"\n=== 전체 ({agg['n_folds']} fold) === "
          f"S_total={fmt(ov['auroc_total'])} S_shape={fmt(ov['auroc_shape'])} "
          f"S_amp={fmt(ov['auroc_amp'])} raw={fmt(ov['auroc_raw'])}")


if __name__ == "__main__":
    main()
