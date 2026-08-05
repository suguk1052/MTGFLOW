"""작업 B §1 — NLL 분해 + 진폭 상관 진단. 무학습·기존 no-meta 체크포인트 재추론.

목적: PU raw-vibration window-level OCC의 실패를 **정성 결론 → 정량 분해**로 끌어올린다.
MTGFlow anomaly score = -log p_x 이고, normalizing flow에서
    log p_x = log p_Z(base density) + log|det J|(flow log-det)
로 정확히 분해된다(합 = 기존 score). 두 항을 정상(source/target) vs fault별로 분리해 보고,
같은 window의 진폭 지표(RMS·peak-to-peak·spectral energy)와의 상관을 재서 실패 유형을 판정한다.

판정표(TODO §1 규약, 상호배타 아님 — 각각 플래그):
- source→target normal에서 base·logdet 둘 다 이동  → domain-shift
- base는 분리(AUROC_base 큼)·total 안 분리(AUROC_total≈0.5) → likelihood-cancellation
- base·logdet·total 분리 AUROC 모두 ≈0.5           → 표현의 fault 민감도 부족
- |Spearman(RMS, NLL)| 큼(단조)                     → 진폭 confound

핵심 원칙(작업 A 준용):
- 무결성 게이트 2종을 먼저 통과: (a) 손수 재현한 분해 total == model.test (diff<1e-4),
  (b) total 기반 AUROC == 저장 overall_auroc (diff<1e-3). 실패 fold는 skip.
- 진폭 지표는 저장된 **스케일된 window**로 계산(원 raw 미저장, 전역 단일 affine 스케일).
  peak-to-peak는 affine 불변, Spearman은 스케일 불변이라 단조/상관 판정엔 충분(리포트에 캐비앗).
- base Gaussian 평균 μ≠0(mode='rand', 체크포인트 buffer). 분해는 여전히 정확.

산출:
- results/Paderborn/diag_nll_decomposition/<split>_LONO<n>_s<seed>.json (fold별)
- results/Paderborn/diag_nll_decomposition/aggregate_s<seed>.json (집계)
- reports/report_nll_decomposition.md
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

from models.NF import _GCONST_  # noqa: E402  per-dim 정규화 상수 ln(1/sqrt(2π))
from models.MTGFLOW import MTGFLOW  # noqa: E402
from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402

DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "Paderborn"))


# ---------------------------------------------------------------------------
# 모델·로더 복원 (diagnose_conditioning_speed_sweep.py와 동일 규약을 자립형으로 인라인)
# ---------------------------------------------------------------------------
def build_model(ckpt, n_sensor, device):
    """checkpoint의 args(모델 hparam)+paderborn_metadata(meta 설정)로 모델 복원."""
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
        meta_input_dim=int(meta["meta_input_dim"]),
        meta_emb_dim=int(meta["meta_emb_dim"]),
        meta_inject=meta.get("meta_inject", "concat"),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model


def build_loader(meta, batch_size):
    """checkpoint metadata로 loader 재구성 (test.py::build_loaders 와 동일 규약)."""
    with contextlib.redirect_stdout(io.StringIO()):
        train_loader, val_loader, test_loader, n_sensor = loader_Paderborn_OCC(
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
            meta_source=meta["meta_source"],
            measured_meta_stats=meta["measured_meta_stats"],
            batch_size=batch_size,
        )
    return train_loader, val_loader, test_loader, n_sensor

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
B3_BATCH = "LONO_B3_5seeds"           # no-meta LOSO 체크포인트 배치 폴더
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_nll_decomposition")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_nll_decomposition.md")
METRICS_FNAME = "paderborn_per_bearing_metrics.json"

# LOSO split ↔ (설명, fold 유형). fold 유형은 뭉뚱그리지 말고 분리 집계(TODO 확정).
SPLIT_INFO = {
    "123to0": ("기준조건(N15_M07_F10) unseen", "compositional"),
    "023to1": ("저속(N09) unseen", "zero-support"),
    "013to2": ("저토크(M01) unseen", "zero-support"),
    "012to3": ("저 radial force(F04) unseen", "zero-support"),
}


# ---------------------------------------------------------------------------
# 경로/로드
# ---------------------------------------------------------------------------
def run_name(split, lono, seed):
    return f"raw_vib_{split}_LONO{lono}_s{seed}"


def ckpt_path(split, lono, seed):
    return os.path.join(RESULTS_ROOT, B3_BATCH, run_name(split, lono, seed), "model.pth")


def saved_auroc(split, lono, seed):
    p = os.path.join(RESULTS_ROOT, B3_BATCH, run_name(split, lono, seed), METRICS_FNAME)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return float(json.load(f)["overall_auroc"])


# ---------------------------------------------------------------------------
# NLL 분해 — MTGFLOW.test(models/MTGFLOW.py:198-213) 파이프라인을 손수 재현한 뒤
# MAF.log_prob(models/NF.py:372-376)의 두 항을 분리한다. 모델 파일은 수정하지 않는다.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _decompose_batch(model, xb):
    """xb: [N, K=1, L, D=1] → window별 (log p_Z, log|det J|, total) 각 [N]."""
    full_shape = xb.shape
    graph, _ = model.attention(xb)
    x = xb.reshape((full_shape[0] * full_shape[1], full_shape[2], full_shape[3]))
    h, _ = model.rnn(x)
    h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))
    h = model.gcn(h, graph)
    # no-meta: meta=None이면 _append_meta_condition은 [N*K*L, H]로 reshape만 수행
    h = model._append_meta_condition(h, None, full_shape)
    x = x.reshape((-1, full_shape[3]))
    # flow forward → 잠재 u, per-dim log|det J| (BatchNorm+MADE 누적)
    u, sum_logdet = model.nf.forward(x, h)
    C = u.shape[1]
    # log p_Z = Σ_dim(-0.5(u-μ)²) + C·ln(1/√2π)  (μ=base_dist_mean, 체크포인트 buffer)
    base_row = torch.sum(model.nf.base_dist(u, full_shape[1], full_shape[2]), dim=1) + C * _GCONST_
    logdet_row = torch.sum(sum_logdet, dim=1)
    # window당 = 센서·타임스텝(K*L) 평균 (MTGFLOW.test와 동일)
    base_w = base_row.reshape([full_shape[0], -1]).mean(dim=1)
    logdet_w = logdet_row.reshape([full_shape[0], -1]).mean(dim=1)
    total_w = base_w + logdet_w
    return base_w, logdet_w, total_w


@torch.no_grad()
def decompose_dataset(model, windows, device, batch_size=256):
    """windows: [N, win](스케일된 신호) → NLL 3배열(base, logdet, total). NLL = -log p."""
    n = len(windows)
    win = windows.shape[1]
    nll_base = np.empty(n, dtype=np.float64)
    nll_logdet = np.empty(n, dtype=np.float64)
    nll_total = np.empty(n, dtype=np.float64)
    model_total = np.empty(n, dtype=np.float64)  # 게이트(a)용: model.test 직접 호출
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.as_tensor(windows[s:e], dtype=torch.float32, device=device)
        xb = xb.reshape(e - s, win, 1, 1).transpose(1, 2).contiguous()  # [b,K=1,L,D=1]
        b, l, t = _decompose_batch(model, xb)
        nll_base[s:e] = (-b).cpu().numpy()
        nll_logdet[s:e] = (-l).cpu().numpy()
        nll_total[s:e] = (-t).cpu().numpy()
        model_total[s:e] = (-model.test(xb, None)).cpu().numpy()
    return nll_base, nll_logdet, nll_total, model_total


# ---------------------------------------------------------------------------
# 진폭 지표 (스케일된 window)
# ---------------------------------------------------------------------------
def amplitude_features(windows):
    x = np.asarray(windows, dtype=np.float64)  # [N, win]
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    p2p = x.max(axis=1) - x.min(axis=1)
    X = np.fft.rfft(x, axis=1)
    power = np.abs(X) ** 2
    # DC(0번) 제외 spectral energy, window 길이로 정규화
    spec_energy = power[:, 1:].sum(axis=1) / x.shape[1]
    return rms, p2p, spec_energy


# ---------------------------------------------------------------------------
# 통계 유틸
# ---------------------------------------------------------------------------
def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def dist_stats(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }


def safe_auroc(labels, scores):
    labels = np.asarray(labels, int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# fold 하나 처리
# ---------------------------------------------------------------------------
def process_fold(split, lono, seed, device, batch_size, gate_tol_a=1e-4, gate_tol_b=1e-3):
    path = ckpt_path(split, lono, seed)
    if not os.path.exists(path):
        print(f"  [skip] checkpoint 없음: {path}")
        return None
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    assert not bool(meta["use_meta"]), "no-meta 전용 진단인데 use_meta=True 체크포인트"
    ref_auroc = saved_auroc(split, lono, seed)

    train_loader, val_loader, test_loader, n_sensor = build_loader(meta, batch_size)
    model = build_model(ckpt, n_sensor, device)

    # ---- 그룹 window 수집 (index 정렬) ----
    tr_w = np.asarray(train_loader.dataset.windows, dtype=np.float32)
    va_w = np.asarray(val_loader.dataset.windows, dtype=np.float32)
    tr_ids = np.asarray(train_loader.dataset.ids)
    va_ids = np.asarray(val_loader.dataset.ids)
    src_w = np.concatenate([tr_w, va_w], axis=0)          # source-normal (train+val, 전부 label0)
    src_ids = np.concatenate([tr_ids, va_ids], axis=0)

    te_w = np.asarray(test_loader.dataset.windows, dtype=np.float32)
    te_lab = np.asarray(test_loader.dataset.label, dtype=int)
    te_ids = np.asarray(test_loader.dataset.ids)
    tgt_mask = te_lab == 0                                  # target-normal (held-out setting+bearing)
    flt_mask = te_lab == 1                                  # fault

    # ---- 분해 재추론 ----
    s_b, s_l, s_t, _ = decompose_dataset(model, src_w, device, batch_size)
    t_b, t_l, t_t, t_model = decompose_dataset(model, te_w, device, batch_size)

    # ---- 게이트(a): 손수 분해 total == model.test ----
    max_diff_a = float(np.max(np.abs(t_t - t_model)))
    # ---- 게이트(b): total 기반 AUROC == 저장 overall_auroc ----
    repro_auroc = safe_auroc(te_lab, t_t)
    diff_b = None if ref_auroc is None else abs(repro_auroc - ref_auroc)
    passed_a = max_diff_a < gate_tol_a
    passed_b = (ref_auroc is None) or (diff_b < gate_tol_b)
    gate = {
        "max_abs_diff_total_vs_modeltest": max_diff_a,
        "reproduced_auroc": repro_auroc,
        "saved_auroc": ref_auroc,
        "auroc_abs_diff": diff_b,
        "passed_a": bool(passed_a),
        "passed_b": bool(passed_b),
        "passed": bool(passed_a and passed_b),
    }
    print(f"  [gate] {split} LONO{lono}: (a)diff={max_diff_a:.2e} pass={passed_a} | "
          f"(b)repro={repro_auroc:.4f} saved={ref_auroc} diff={diff_b} pass={passed_b}")
    if not gate["passed"]:
        print(f"  [STOP-fold] 게이트 실패 — 이 fold 분해 무의미.")
        return {"split": split, "lono": lono, "seed": seed, "gate": gate,
                "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0]}

    # ---- 진폭 지표 ----
    s_rms, s_p2p, s_spec = amplitude_features(src_w)
    t_rms, t_p2p, t_spec = amplitude_features(te_w)

    # target-normal / fault 슬라이스
    tn = dict(base=t_b[tgt_mask], logdet=t_l[tgt_mask], total=t_t[tgt_mask],
              rms=t_rms[tgt_mask], p2p=t_p2p[tgt_mask], spec=t_spec[tgt_mask], ids=te_ids[tgt_mask])
    fl = dict(base=t_b[flt_mask], logdet=t_l[flt_mask], total=t_t[flt_mask],
              rms=t_rms[flt_mask], p2p=t_p2p[flt_mask], spec=t_spec[flt_mask], ids=te_ids[flt_mask])
    sn = dict(base=s_b, logdet=s_l, total=s_t, rms=s_rms, p2p=s_p2p, spec=s_spec, ids=src_ids)

    # ---- 분포 통계 ----
    def group_dist(g):
        return {term: dist_stats(g[term]) for term in ("base", "logdet", "total")}
    distributions = {"source_normal": group_dist(sn), "target_normal": group_dist(tn),
                     "fault": group_dist(fl)}

    # ---- 분리 AUROC (target-normal=0 vs fault=1), 각 항을 score로 ----
    sep_auroc = {term: safe_auroc(te_lab, t) for term, t in
                 (("total", t_t), ("base", t_b), ("logdet", t_l))}

    # ---- 도메인 shift (source-normal → target-normal, source std로 표준화) ----
    def std_shift(term):
        s = sn[term]
        t = tn[term]
        sd = float(np.std(s)) + 1e-12
        return float((np.mean(t) - np.mean(s)) / sd)
    domain_shift = {term: std_shift(term) for term in ("base", "logdet", "total")}

    # ---- 진폭 상관 (그룹별, RMS·p2p·spec vs 각 NLL 항) ----
    def corr_group(g):
        out = {}
        for amp in ("rms", "p2p", "spec"):
            for term in ("total", "base", "logdet"):
                out[f"{amp}_vs_{term}"] = {
                    "pearson": pearson(g[amp], g[term]),
                    "spearman": spearman(g[amp], g[term]),
                }
        return out
    # 정상 pooled (source+target normal) = 진폭 confound 핵심 관찰 대상
    normal_pool = {k: np.concatenate([sn[k], tn[k]]) for k in ("base", "logdet", "total", "rms", "p2p", "spec")}
    amp_corr = {
        "normal_pool": corr_group(normal_pool),
        "all_test": corr_group(dict(
            base=t_b, logdet=t_l, total=t_t, rms=t_rms, p2p=t_p2p, spec=t_spec)),
        "source_normal": corr_group(sn),
        "target_normal": corr_group(tn),
        "fault": corr_group(fl),
    }

    # ---- 정상 bearing별 평균 (진폭 confound 시각: 고진폭 정상이 고 NLL) ----
    per_bearing_normal = {}
    all_norm_ids = np.concatenate([sn["ids"], tn["ids"]])
    all_norm_rms = normal_pool["rms"]
    all_norm_total = normal_pool["total"]
    for bid in sorted(set(all_norm_ids.tolist())):
        m = all_norm_ids == bid
        per_bearing_normal[bid] = {
            "n": int(m.sum()),
            "mean_rms": float(np.mean(all_norm_rms[m])),
            "mean_nll_total": float(np.mean(all_norm_total[m])),
            "group": "target" if bid in set(tn["ids"].tolist()) else "source",
        }

    # ---- fault 유형(bearing id)별 평균 ----
    per_fault_type = {}
    for bid in sorted(set(fl["ids"].tolist())):
        m = fl["ids"] == bid
        per_fault_type[bid] = {
            "n": int(m.sum()),
            "family": bid[:2] if len(bid) >= 2 else bid,  # KA/KI/KB
            "mean_nll_total": float(np.mean(fl["total"][m])),
            "mean_nll_base": float(np.mean(fl["base"][m])),
            "mean_nll_logdet": float(np.mean(fl["logdet"][m])),
            "mean_rms": float(np.mean(fl["rms"][m])),
        }

    # ---- 자동 판정 플래그 (상호배타 아님) ----
    flags = compute_flags(sep_auroc, domain_shift, amp_corr)

    return {
        "split": split, "lono": lono, "seed": seed,
        "fold_type": SPLIT_INFO[split][1], "desc": SPLIT_INFO[split][0],
        "target_norm_ids": sorted(set(tn["ids"].tolist())),
        "gate": gate,
        "distributions": distributions,
        "separation_auroc": sep_auroc,
        "domain_shift_std": domain_shift,
        "amplitude_corr": amp_corr,
        "per_bearing_normal": per_bearing_normal,
        "per_fault_type": per_fault_type,
        "flags": flags,
    }


def compute_flags(sep_auroc, domain_shift, amp_corr,
                  sep_thr=0.15, flat_thr=0.10, shift_thr=1.0, conf_thr=0.5):
    """판정표 4종 플래그. |AUROC-0.5|=분리력, 표준화 shift, |Spearman(RMS,NLL_total)|."""
    a_tot = abs(sep_auroc["total"] - 0.5) if sep_auroc["total"] == sep_auroc["total"] else 0.0
    a_base = abs(sep_auroc["base"] - 0.5) if sep_auroc["base"] == sep_auroc["base"] else 0.0
    a_log = abs(sep_auroc["logdet"] - 0.5) if sep_auroc["logdet"] == sep_auroc["logdet"] else 0.0
    # 진폭 confound: 정상 pooled에서 RMS↔NLL_total 단조성
    conf_sp = amp_corr["normal_pool"]["rms_vs_total"]["spearman"]
    conf_sp = 0.0 if conf_sp != conf_sp else conf_sp
    return {
        "domain_shift": bool(abs(domain_shift["base"]) > shift_thr and
                             abs(domain_shift["logdet"]) > shift_thr),
        "likelihood_cancellation": bool(a_base > sep_thr and a_tot < flat_thr and
                                        (a_base - a_tot) > sep_thr),
        "fault_sensitivity_lack": bool(a_tot < flat_thr and a_base < flat_thr and a_log < flat_thr),
        "amplitude_confound": bool(abs(conf_sp) > conf_thr),
        "_metrics": {"auroc_dev_total": a_tot, "auroc_dev_base": a_base,
                     "auroc_dev_logdet": a_log, "confound_spearman_normalpool": conf_sp},
    }


# ---------------------------------------------------------------------------
# 집계 + 리포트
# ---------------------------------------------------------------------------
def aggregate(folds):
    """fold 유형별로 분리 집계 (뭉뚱그리지 않음)."""
    ok = [f for f in folds if f.get("gate", {}).get("passed")]
    by_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in ok if f["fold_type"] == ftype]
        if not rows:
            continue
        def mean_of(path):
            vals = []
            for r in rows:
                v = r
                for p in path:
                    v = v[p]
                if v == v:  # nan 제외
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")
        flag_counts = {k: int(sum(1 for r in rows if r["flags"][k]))
                       for k in ("domain_shift", "likelihood_cancellation",
                                 "fault_sensitivity_lack", "amplitude_confound")}
        by_type[ftype] = {
            "n_folds": len(rows),
            "sep_auroc_total": mean_of(["separation_auroc", "total"]),
            "sep_auroc_base": mean_of(["separation_auroc", "base"]),
            "sep_auroc_logdet": mean_of(["separation_auroc", "logdet"]),
            "domain_shift_base": mean_of(["domain_shift_std", "base"]),
            "domain_shift_logdet": mean_of(["domain_shift_std", "logdet"]),
            "confound_spearman_normalpool": mean_of(
                ["amplitude_corr", "normal_pool", "rms_vs_total", "spearman"]),
            "flag_counts": flag_counts,
        }
    return {"n_folds_total": len(folds), "n_folds_passed": len(ok), "by_fold_type": by_type}


def fmt(v, p=3):
    if v is None:
        return "—"
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v:.{p}f}"


def build_report(folds, agg, seed):
    L = []
    L.append("# 작업 B §1 진단 — NLL 분해 + 진폭 상관 (Paderborn no-meta, 무학습)")
    L.append("")
    L.append(f"seed {seed}, no-meta LOSO 기준선, 4 LOSO split × 6 LONO = {len(folds)} fold. "
             "기존 체크포인트(`LONO_B3_5seeds`) 재추론만, 새 학습 없음.")
    L.append("")
    L.append("## 방법")
    L.append("")
    L.append("anomaly score = `-log p_x`, flow에서 `log p_x = log p_Z + log|det J|`로 분해:")
    L.append("- **log p_Z(base density)** = `Σ(-0.5(u-μ)²) + C·ln(1/√2π)` "
             "(μ=`base_dist_mean`, mode='rand'라 μ≠0인 flow Gaussian base·단위분산).")
    L.append("- **log|det J|** = flow(BatchNorm+MADE)의 log-det 누적.")
    L.append("- NLL 항 = 각 log 항의 부호 반전(값이 클수록 이상).")
    L.append("")
    L.append("**그룹**: source-normal(train+val, source setting·K001~K004) / "
             "target-normal(held-out setting **+** 다른 bearing = 이중 held-out) / fault(held-out setting).")
    L.append("")
    L.append("> ⚠️ **캐비앗**: 진폭 지표(RMS·peak-to-peak·spectral energy)는 저장된 **스케일된 "
             "window**로 계산(원 raw 미저장, 전역 단일 affine 스케일). peak-to-peak는 affine 불변, "
             "Spearman은 스케일 불변이라 **단조/상관 판정엔 충분**. base μ≠0이라 log p_Z는 표준정규가 "
             "아닌 flow의 학습된 Gaussian base 하 밀도로 해석(분해 정확성엔 영향 없음, 합=total).")
    L.append("")

    # 게이트 요약
    L.append("## 1) 무결성 게이트")
    L.append("")
    L.append("(a) 손수 분해 total == `model.test` (diff<1e-4). (b) total 기반 AUROC == 저장 "
             "`overall_auroc` (diff<1e-3). 둘 다 통과해야 분해가 유효.")
    L.append("")
    L.append("| split | LONO | (a) diff | (a) pass | (b) repro AUROC | (b) saved | (b) diff | pass |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in folds:
        g = f["gate"]
        L.append(f"| {f['split']} | {f['lono']} | {g['max_abs_diff_total_vs_modeltest']:.1e} | "
                 f"{'✓' if g['passed_a'] else '✗'} | {fmt(g['reproduced_auroc'],4)} | "
                 f"{fmt(g['saved_auroc'],4)} | {fmt(g['auroc_abs_diff'],2) if g['auroc_abs_diff'] is not None else '—'} | "
                 f"{'✅' if g['passed'] else '❌'} |")
    n_pass = sum(1 for f in folds if f["gate"]["passed"])
    L.append("")
    L.append(f"통과: {n_pass}/{len(folds)}. (이하 분석은 통과 fold만.)")
    L.append("")

    # fold 유형별 집계 (핵심 판정)
    L.append("## 2) fold 유형별 집계 (핵심 판정, 뭉뚱그리지 않음)")
    L.append("")
    L.append("분리 AUROC = target-normal(0) vs fault(1)를 각 항으로 스코어. 0.5에서 멀수록 분리력 "
             "(<0.5=역전). domain_shift = source→target normal 평균 이동(source std 표준화). "
             "confound = 정상 pooled에서 Spearman(RMS, NLL_total).")
    L.append("")
    L.append("| fold 유형 | n | AUROC_total | AUROC_base | AUROC_logdet | shift_base(σ) | shift_logdet(σ) | confound ρ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for ftype, a in agg["by_fold_type"].items():
        L.append(f"| {ftype} | {a['n_folds']} | {fmt(a['sep_auroc_total'])} | {fmt(a['sep_auroc_base'])} | "
                 f"{fmt(a['sep_auroc_logdet'])} | {fmt(a['domain_shift_base'])} | "
                 f"{fmt(a['domain_shift_logdet'])} | {fmt(a['confound_spearman_normalpool'])} |")
    L.append("")
    L.append("**플래그 카운트(통과 fold 중):**")
    L.append("")
    L.append("| fold 유형 | domain-shift | likelihood-cancellation | fault-민감도부족 | 진폭-confound |")
    L.append("|---|---|---|---|---|")
    for ftype, a in agg["by_fold_type"].items():
        fc = a["flag_counts"]
        L.append(f"| {ftype} ({a['n_folds']}) | {fc['domain_shift']} | {fc['likelihood_cancellation']} | "
                 f"{fc['fault_sensitivity_lack']} | {fc['amplitude_confound']} |")
    L.append("")

    # 판정표 해석
    L.append("## 3) 판정표 해석")
    L.append("")
    L.append("TODO §1 판정 규약 대응(플래그가 우세한 유형으로 읽음):")
    L.append("- **두 항 모두 이동** → domain shift")
    L.append("- **base 분리·total 안 분리** → likelihood cancellation")
    L.append("- **전 항 겹침** → 표현의 fault 민감도 부족")
    L.append("- **RMS-NLL 단조** → 진폭 confound (발견 ③)")
    L.append("")

    # 통과 fold 상세
    L.append("## 4) 통과 fold 상세")
    L.append("")
    L.append("| split | LONO | target-norm | AUROC_tot | AUROC_base | AUROC_logdet | shift_b | shift_l | conf ρ | flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in folds:
        if not f["gate"]["passed"]:
            continue
        sa = f["separation_auroc"]
        ds = f["domain_shift_std"]
        conf = f["amplitude_corr"]["normal_pool"]["rms_vs_total"]["spearman"]
        active = [k.replace("_", "-") for k, v in f["flags"].items() if k != "_metrics" and v]
        L.append(f"| {f['split']} | {f['lono']} | {','.join(f['target_norm_ids'])} | "
                 f"{fmt(sa['total'])} | {fmt(sa['base'])} | {fmt(sa['logdet'])} | "
                 f"{fmt(ds['base'])} | {fmt(ds['logdet'])} | {fmt(conf)} | {', '.join(active) or '—'} |")
    L.append("")

    # 진폭 confound 시각 — 정상 bearing별
    L.append("## 5) 진폭 confound 근거 — 정상 bearing별 (통과 fold pooled)")
    L.append("")
    L.append("발견 ③: 고진폭 정상 bearing(K001/K003/K006)이 저진폭(K002/K004/K005)보다 높은 "
             "mean NLL_total을 받으면 진폭이 스코어를 지배한다는 직접 증거.")
    L.append("")
    bagg = {}
    for f in folds:
        if not f["gate"]["passed"]:
            continue
        for bid, d in f["per_bearing_normal"].items():
            e = bagg.setdefault(bid, {"rms": [], "nll": [], "n": 0})
            e["rms"].append(d["mean_rms"])
            e["nll"].append(d["mean_nll_total"])
            e["n"] += d["n"]
    L.append("| bearing | 진폭군 | fold수 | mean RMS | mean NLL_total |")
    L.append("|---|---|---|---|---|")
    HIGH = {"K001", "K003", "K006"}
    for bid in sorted(bagg):
        e = bagg[bid]
        grp = "고진폭" if bid in HIGH else ("저진폭" if bid.startswith("K0") else "?")
        L.append(f"| {bid} | {grp} | {len(e['rms'])} | {fmt(np.mean(e['rms']),3)} | "
                 f"{fmt(np.mean(e['nll']),3)} |")
    L.append("")

    # fault 유형별
    L.append("## 6) fault 유형(bearing id)별 mean NLL (통과 fold pooled)")
    L.append("")
    fagg = {}
    for f in folds:
        if not f["gate"]["passed"]:
            continue
        for bid, d in f["per_fault_type"].items():
            e = fagg.setdefault(bid, {"family": d["family"], "tot": [], "base": [], "log": [], "n": 0})
            e["tot"].append(d["mean_nll_total"])
            e["base"].append(d["mean_nll_base"])
            e["log"].append(d["mean_nll_logdet"])
            e["n"] += d["n"]
    L.append("| fault id | family | fold수 | mean NLL_total | mean NLL_base | mean NLL_logdet |")
    L.append("|---|---|---|---|---|---|")
    for bid in sorted(fagg):
        e = fagg[bid]
        L.append(f"| {bid} | {e['family']} | {len(e['tot'])} | {fmt(np.mean(e['tot']))} | "
                 f"{fmt(np.mean(e['base']))} | {fmt(np.mean(e['log']))} |")
    L.append("")

    L.append("## 주의 / 한계")
    L.append("")
    L.append("- 무학습 진단(재추론). seed 2026 단일 — 표현/밀도 수준 진단이라 시드 변동은 대상 아님.")
    L.append("- 진폭 지표는 스케일된 window 기준(위 캐비앗). 단조/상관 판정엔 유효, 절대 진폭 비교는 근사.")
    L.append("- target-normal은 setting·bearing 이중 held-out이라 domain-shift와 bearing 개체차가 섞임.")
    L.append("- 판정 임계(AUROC dev 0.15/0.10, shift 1.0σ, confound ρ 0.5)는 서술 편의용 — "
             "표의 원수치를 함께 볼 것.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_INFO.keys()),
                    choices=list(SPLIT_INFO.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--no_report", action="store_true", help="fold JSON만 쓰고 집계/리포트 생략")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(args.out_dir, exist_ok=True)

    folds = []
    for split in args.splits:
        for lono in args.lonos:
            res = process_fold(split, lono, args.seed, device, args.batch_size)
            if res is None:
                continue
            folds.append(res)
            out = os.path.join(args.out_dir, f"{split}_LONO{lono}_s{args.seed}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)

    if args.no_report or not folds:
        print(f"\nfold JSON {len(folds)}개 기록. (집계/리포트 생략)")
        return

    agg = aggregate(folds)
    agg_path = os.path.join(args.out_dir, f"aggregate_s{args.seed}.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    report = build_report(folds, agg, args.seed)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nwrote {agg_path}")
    print(f"wrote {REPORT_PATH}")
    print("\n=== fold 유형별 집계 ===")
    for ftype, a in agg["by_fold_type"].items():
        fc = a["flag_counts"]
        print(f"  {ftype:>13} n={a['n_folds']} AUROC(tot/base/log)="
              f"{a['sep_auroc_total']:.3f}/{a['sep_auroc_base']:.3f}/{a['sep_auroc_logdet']:.3f} "
              f"confρ={a['confound_spearman_normalpool']:.3f} flags={fc}")


if __name__ == "__main__":
    main()
