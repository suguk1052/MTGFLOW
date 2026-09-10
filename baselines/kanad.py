"""B-4 KAN-AD (재구성형 TSAD 참고 모델) — MTGFLOW 로더 위 별도 어댑터 (GPU 학습).

B-0 §5a 규칙: Time-Series-Library의 exp_anomaly_detection 래퍼 **미사용**(그 래퍼는 train+test 합산
threshold·point-adjustment·test loss early-stop = 누수). 여기서는 KANAD 재구성망만 가져와
**train-normal 학습·val-normal 최소 recon MSE checkpoint·window 재구성오차=score·anomaly_ratio/PA 미사용**.

모델(KANADModel)은 Time-Series-Library/models/KANAD.py 에서 복사(출처 명시). 주기 cosine 확장 +
Conv1d + final Linear(window,window). **self-attention·instance norm 없음 → window 진폭 보존**.
입력 raw 2048 window(train-normal StandardScaler, amp_normalize=False) — B-1 raw와 동일 로더 경로.

하이퍼(TSLib 기본 고정, 실험 전 확정·이후 불변, Table 3):
  order(d_model)=4, seq_len=2048, enc_in=c_out=1, lr=0.01, batch_size=128, train_epochs=100, Adam,
  loss=MSE 재구성, checkpoint=val-normal 최소 recon MSE. anomaly_ratio·point-adjustment 미사용.
score s(x)=mean_L (recon−x)²(큼=이상). threshold=val-normal 95pct.

산출: results/Paderborn/b4_kanad_window_scores/<split>_LONO<n>_s<seed>.npz (b3 스키마).
사용: python baselines/kanad.py --split 012to3 --lono 1 --seed 2024
"""
import argparse
import copy
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
from dump_B1_pu_window_scores import load_fold_meta, DATA_ROOT  # noqa: E402
from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402

# --- 하이퍼(고정) ---
ORDER = 4
SEQ_LEN = 2048
LR = 0.01
BATCH_SIZE = 128
N_EPOCHS = 100
OUT_DIR = os.path.join(RESULTS_ROOT, "b4_kanad_window_scores")


# =========================================================================
# KANADModel — Time-Series-Library/models/KANAD.py 에서 복사 (출처: TSLib, 공개 구현)
# =========================================================================
class KANADModel(nn.Module):
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.order = order
        self.window = window
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window, self.order).unsqueeze(0),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(window, window)

    def forward(self, x: torch.Tensor, *args, **kwargs):  # x: [B, window]
        res = []
        res.append(x.unsqueeze(1))
        ff = torch.cat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # [B, channels, window]
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        return ff.squeeze(1)  # [B, window] 재구성

    def _create_custom_periodic_cosine(self, window: int, period) -> torch.Tensor:
        d = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(pl):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i, :] = torch.cos(t)
        return result
# =========================================================================


def _windows_tensor(windows):
    return torch.as_tensor(np.asarray(windows, dtype=np.float32))  # [N, L]


@torch.no_grad()
def recon_error(net, X, device, bs=256):
    """per-window score = mean_L (recon−x)² (큼=이상)."""
    net.eval()
    out = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), bs):
        xb = X[s:s + bs].to(device)
        rec = net(xb)
        out[s:s + len(xb)] = ((rec - xb) ** 2).mean(dim=1).cpu().numpy()
    return out


@torch.no_grad()
def val_mse(net, Xva, device, bs=256):
    net.eval()
    tot, n = 0.0, 0
    for s in range(0, len(Xva), bs):
        xb = Xva[s:s + bs].to(device)
        rec = net(xb)
        tot += ((rec - xb) ** 2).mean(dim=1).sum().item()
        n += len(xb)
    return tot / max(n, 1)


def train_kanad(net, Xtr, Xva, device, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    best_val, best_state = float("inf"), None
    for ep in range(N_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g)
        ep_loss = 0.0
        for s in range(0, n, BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            xb = Xtr[idx].to(device)
            opt.zero_grad()
            rec = net(xb)
            loss = ((rec - xb) ** 2).mean()  # MSE 재구성
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        vm = val_mse(net, Xva, device)  # val-normal loss checkpoint (test 미참조)
        if vm < best_val:
            best_val, best_state = vm, copy.deepcopy(net.state_dict())
        if ep == 0 or (ep + 1) % 20 == 0 or ep == N_EPOCHS - 1:
            print(f"  epoch {ep+1}/{N_EPOCHS} train_mse={ep_loss/n:.6f} val_mse={vm:.6f} best={best_val:.6f}", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    return best_val


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
    print(f"[{split} LONO{lono} s{seed}] device={device} n_tr={len(Xtr)} n_va={len(Xva)} n_te={len(Xte)} win={win} load={time.time()-t0:.1f}s", flush=True)

    net = KANADModel(window=win, order=ORDER).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    t1 = time.time()
    best_val = train_kanad(net, Xtr, Xva, device, seed)
    t_train = time.time() - t1

    tr_s = recon_error(net, Xtr, device)
    va_s = recon_error(net, Xva, device)
    te_s = recon_error(net, Xte, device)

    thr = float(np.percentile(va_s, 95))
    val_fpr = float(np.mean(va_s >= thr))
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(te_lab, te_s)) if len(set(te_lab.tolist())) > 1 else float("nan")
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None

    tgt = te_lab == 0
    tgt_bid = LONO_TARGET.get(lono)
    meta_json = {
        "split": split, "lono": int(lono), "seed": int(seed),
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_amp_group": "high" if (tgt_bid in HIGH_AMP) else "low",
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "run_name": f"b4_kanad_{split}_LONO{lono}_s{seed}",
        "model": "kanad", "order": ORDER, "lr": LR, "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS, "n_params": int(n_params), "best_val_mse": best_val,
        "val_thr95": thr, "val_fpr": val_fpr, "peak_gpu_gb": peak_gb, "t_train_s": round(t_train, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path, te_f=te_s, va_f=va_s, tr_f=tr_s,
        te_lab=te_lab, te_ids=te_ids, te_rms=te_rms,
        va_ids=va_ids, va_rms=va_rms, tr_ids=tr_ids, tr_rms=tr_rms,
        meta_json=json.dumps(meta_json, ensure_ascii=False))
    summary = {"fold": f"{split}_LONO{lono}_s{seed}", "auroc": round(auroc, 4),
               "val_fpr": round(val_fpr, 4), "best_val_mse": round(best_val, 6),
               "t_train_s": round(t_train, 1), "n_params": int(n_params),
               "peak_gpu_gb": (round(peak_gb, 3) if peak_gb else None)}
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
