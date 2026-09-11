"""B-2 Deep SVDD (One-Class) — PU raw window baseline (GPU 학습). B-0/TODO B-2 사양.

collapse 방지 표준 처방(Ruff et al. 2018): 전 층 bias 없음 + BatchNorm affine 없음 + 비포화 활성(LeakyReLU),
center c는 init forward 평균으로 고정(비학습). One-Class 목적 L=(1/n)Σ‖φ(x)-c‖²+(λ/2)‖W‖².
score s(x)=‖φ(x)-c‖² (거리=이상, 부호반전 불필요). threshold=val-normal 95pct(FPR sanity).

입력: raw 2048 window(train-normal StandardScaler, amp_normalize=False) — B-1 raw와 동일 로더 경로.
로더 인자는 B3 checkpoint metadata에서 복원(dump_B1_pu 재사용). **B-1 파일 수정 없음(읽기 재사용만).**

산출: results/Paderborn/b2_dsvdd_window_scores/<split>_LONO<n>_s<seed>.npz (b3 스키마 동형).
사용: python baselines/deep_svdd.py --split 012to3 --lono 1 --seed 2024
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "analysis"))

from diagnose_G3a_amp_bands import RESULTS_ROOT, SPLIT_INFO, LONO_TARGET, HIGH_AMP, pull  # noqa: E402
from dump_B1_pu_window_scores import load_fold_meta, DATA_ROOT  # noqa: E402 (B-1 헬퍼 읽기 재사용)
from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402

# --- 하이퍼(사전 고정, Table 3) ---
REP_DIM = 128
N_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-6
LR_MILESTONES = [50]
LR_GAMMA = 0.1
BATCH_SIZE = 128
CENTER_EPS = 0.1
LEAKY = 0.01
OUT_DIR = os.path.join(RESULTS_ROOT, "b2_dsvdd_window_scores")

# --- collapse 판정 임계(수치·label-free) ---
COLLAPSE_EMB_STD = 1e-3       # train 임베딩 per-dim std 평균
COLLAPSE_SCORE_MEAN = 1e-6    # train 점수 평균
COLLAPSE_SCORE_CV = 1e-2      # train 점수 std/mean
COLLAPSE_RANK_RATIO = 1.05    # val 점수 pct95/pct50


class DSVDDNet(nn.Module):
    """1D-CNN encoder. 전 층 bias 없음, BatchNorm1d affine 없음, LeakyReLU(비포화). → φ(x)∈R^rep_dim."""

    def __init__(self, window_size=2048, rep_dim=REP_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=4, padding=6, bias=False),
            nn.BatchNorm1d(16, affine=False), nn.LeakyReLU(LEAKY), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32, affine=False), nn.LeakyReLU(LEAKY), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64, affine=False), nn.LeakyReLU(LEAKY), nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 1, window_size)).flatten(1).shape[1]
        self.fc = nn.Linear(flat, rep_dim, bias=False)  # bias 없음

    def forward(self, x):  # x: [B,1,L]
        return self.fc(self.features(x).flatten(1))  # [B, rep_dim]


def _windows_tensor(windows):
    return torch.as_tensor(np.asarray(windows, dtype=np.float32)).unsqueeze(1)  # [N,1,L]


@torch.no_grad()
def _embed(net, X, device, bs=512):
    net.eval()
    out = []
    for s in range(0, len(X), bs):
        out.append(net(X[s:s + bs].to(device)).cpu())
    return torch.cat(out, 0) if out else torch.empty(0, REP_DIM)


def init_center_c(net, X, device, eps=CENTER_EPS):
    """center c = init forward 평균(train-normal). |c_j|<eps → ±eps (Ruff: zero unit trivial-match 방지)."""
    emb = _embed(net, X, device)
    c = emb.mean(0)
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c.to(device)


def train_dsvdd(net, Xtr, device, seed):
    torch.manual_seed(seed)
    c = init_center_c(net, Xtr, device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=LR_MILESTONES, gamma=LR_GAMMA)
    n = len(Xtr)
    g = torch.Generator().manual_seed(seed)
    for ep in range(N_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g)
        ep_loss = 0.0
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            xb = Xtr[idx].to(device)
            opt.zero_grad()
            emb = net(xb)
            loss = ((emb - c) ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        sch.step()
        if ep == 0 or (ep + 1) % 20 == 0 or ep == N_EPOCHS - 1:
            print(f"  epoch {ep+1}/{N_EPOCHS} loss={ep_loss/n:.6f}", flush=True)
    return c


@torch.no_grad()
def score(net, X, c, device, bs=512):
    net.eval()
    out = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), bs):
        emb = net(X[s:s + bs].to(device))
        out[s:s + len(emb)] = ((emb - c) ** 2).sum(dim=1).cpu().numpy()
    return out


def collapse_check(net, Xtr, tr_scores, va_scores, device):
    emb = _embed(net, Xtr, device).numpy()
    emb_std = float(np.mean(np.std(emb, axis=0)))
    s_mean = float(np.mean(tr_scores)); s_std = float(np.std(tr_scores))
    s_cv = s_std / s_mean if s_mean > 0 else 0.0
    p50 = float(np.percentile(va_scores, 50)); p95 = float(np.percentile(va_scores, 95))
    rank_ratio = (p95 / p50) if p50 > 0 else float("inf")
    fails = []
    if emb_std < COLLAPSE_EMB_STD: fails.append(f"emb_std={emb_std:.2e}<{COLLAPSE_EMB_STD}")
    if s_mean < COLLAPSE_SCORE_MEAN: fails.append(f"score_mean={s_mean:.2e}<{COLLAPSE_SCORE_MEAN}")
    if s_cv < COLLAPSE_SCORE_CV: fails.append(f"score_cv={s_cv:.2e}<{COLLAPSE_SCORE_CV}")
    if rank_ratio < COLLAPSE_RANK_RATIO: fails.append(f"rank_ratio={rank_ratio:.3f}<{COLLAPSE_RANK_RATIO}")
    return {"emb_std": emb_std, "score_mean": s_mean, "score_cv": s_cv,
            "val_rank_ratio": rank_ratio, "collapse": bool(fails), "fails": fails}


def build_loader(meta, batch_size=256):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return loader_Paderborn_OCC(
            root=DATA_ROOT, train_loads=meta["train_load_setting"],
            test_loads=meta["test_load_setting"], train_ids=meta["train_ids"],
            val_ids=meta["val_ids"], test_norm_ids=meta["test_norm_ids"],
            exclude_ids=meta.get("exclude_ids", []),
            window_size=int(meta["window_size"]), stride_size=int(meta["stride_size"]),
            sensor_mode=meta["sensor_mode"], meta_source=meta.get("meta_source", "static"),
            measured_meta_stats=meta.get("measured_meta_stats", "meanstd"),
            amp_normalize=False, amp_n_bands=1, batch_size=batch_size)


def run_fold(split, lono, seed, out_dir, overwrite):
    out_path = os.path.join(out_dir, f"{split}_LONO{lono}_s{seed}.npz")
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] 이미 존재: {out_path}"); return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    meta = load_fold_meta(split, lono)
    tr, va, te, _ = build_loader(meta)
    win = int(meta["window_size"])
    tr_w, tr_rms, _, tr_ids, _ = pull(tr.dataset)
    va_w, va_rms, _, va_ids, _ = pull(va.dataset)
    te_w, te_rms, _, te_ids, te_lab = pull(te.dataset)
    Xtr, Xva, Xte = _windows_tensor(tr_w), _windows_tensor(va_w), _windows_tensor(te_w)
    print(f"[{split} LONO{lono} s{seed}] device={device} n_tr={len(Xtr)} n_va={len(Xva)} n_te={len(Xte)} load={time.time()-t0:.1f}s", flush=True)

    net = DSVDDNet(window_size=win).to(device)
    t1 = time.time()
    c = train_dsvdd(net, Xtr, device, seed)
    t_train = time.time() - t1

    tr_s = score(net, Xtr, c, device)
    va_s = score(net, Xva, c, device)
    te_s = score(net, Xte, c, device)

    thr = float(np.percentile(va_s, 95))
    val_fpr = float(np.mean(va_s >= thr))
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(te_lab, te_s)) if len(set(te_lab.tolist())) > 1 else float("nan")
    col = collapse_check(net, Xtr, tr_s, va_s, device)
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None

    tgt = te_lab == 0
    tgt_bid = LONO_TARGET.get(lono)
    meta_json = {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "run_name": f"b2_dsvdd_{split}_LONO{lono}_s{seed}",
        "model": "deep_svdd", "rep_dim": REP_DIM, "n_epochs": N_EPOCHS, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "center_eps": CENTER_EPS,
        "val_thr95": thr, "val_fpr": val_fpr, "collapse": col,
        "peak_gpu_gb": peak_gb, "t_train_s": round(t_train, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path, te_f=te_s, va_f=va_s, tr_f=tr_s,
        te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
        va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids, tr_rms=tr_rms,
        meta_json=json.dumps(meta_json, ensure_ascii=False))
    summary = {"fold": f"{split}_LONO{lono}_s{seed}", "auroc": round(auroc, 4),
               "val_fpr": round(val_fpr, 4), "t_train_s": round(t_train, 1),
               "peak_gpu_gb": (round(peak_gb, 2) if peak_gb else None),
               "collapse": col}
    print("RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lono", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run_fold(args.split, args.lono, args.seed, args.out_dir, args.overwrite)


if __name__ == "__main__":
    main()
