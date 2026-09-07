#!/usr/bin/env python3
"""P-2(밴드 분할 방식) 선행 EDA — CPU·라벨 무사용.

목적(TODO P-2 "선행 EDA"): GPU 전량 학습 전에, 각 fold의 train-normal 통계만으로
- linear / log / energy 세 scheme의 산출 경계(rfft bin)와 band 폭(bin/Hz)
- fold별 train-normal 평균 PSD 요약(각 linear band의 에너지 비중, 지배 대역)
- energy 경계의 fold 간 변동·최소폭 가드 발동 여부
- scheme 간 경계 근접도(energy vs linear/log)
- band별 log-RMS std(=학습 target 분산; 유효 신호량 대리)
를 산출해 `reports/report_P2_band_eda.md`로 남긴다.

누수 방지(P-G3): 경계는 **fold별 train-normal ch0**만으로 산출. val/test·fault·label 미사용.
fold 스펙(train_load_setting·train_ids 등)은 기존 G-3a 체크포인트 metadata에서 읽어 전사 오류를 없앤다.

정합성 보증: --verify_loader 지정 시, 대표 fold에서 실제 loader_Paderborn_OCC(amp_band_scheme='energy')를
호출해 dataset.amp_band_edges 가 본 EDA가 재현한 edges와 정확히 일치함을 assert(코드 경로 divergence 방지).

실행: conda run -n mtgflow python analysis/eda_P2_band_partition.py [--verify_loader]
"""
import argparse
import io
import contextlib
import json
import os
import sys

import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from Dataset.paderborn import (  # noqa: E402
    compute_band_ms_per_bin, compute_band_boundaries, band_rms_from_ms,
    resolve_sensor_names, load_stacked_signals, loader_Paderborn_OCC,
)

DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "Paderborn"))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_P2_band_eda.md")

SPLITS = ["123to0", "023to1", "013to2", "012to3"]
SPLIT_TYPE = {"123to0": "compositional", "023to1": "zero-support",
              "013to2": "zero-support", "012to3": "zero-support"}
LONOS = [1, 2, 3, 4, 5, 6]
N_BANDS = 6
MIN_WIDTH = 4
SCHEMES = ["linear", "log", "energy"]
FS_VIB = 64000  # Hz (G-3a 러너 --sampling_rate)


def g3a_ckpt_meta(split, lono, seed=2026):
    """G-3a 체크포인트 metadata(=fold 스펙). 경계는 seed 무관이라 대표 seed 하나만 읽는다."""
    import torch
    run = f"g3a_{split}_LONO{lono}_s{seed}"
    p = os.path.join(RESULTS_ROOT, run, "model.pth")
    if not os.path.exists(p):
        return None, run
    ckpt = torch.load(p, map_location="cpu")
    return ckpt.get("paderborn_metadata"), run


def train_normal_mean_psd(meta):
    """fold의 train-normal ch0만으로 loader와 동일 경로(scaler fit→transform→window→ms_per_bin)를
    재현해 평균 per-bin mean-square(F,)와 창 수를 반환한다. (order_track 미사용 전제 = G-3a 동일)"""
    assert not bool(meta.get("order_track", False)), "G-3a는 order_track 미사용 전제인데 meta에 켜져 있음"
    sensor_names = resolve_sensor_names(meta["sensor_mode"])
    train_ids = set(str(meta["train_ids"]).split()) if isinstance(meta["train_ids"], str) else set(meta["train_ids"])
    window_size = int(meta["window_size"])
    stride_size = int(meta["stride_size"])

    def bearing_of(fname):
        import re
        m = re.search(r'(K[A-Z]?\d{2,3})(?:_|\.)', fname)
        return m.group(1) if m else fname.replace('.mat', '')

    # train setting들에서 train_ids 파일만 수집
    file_tuples = []
    for setting in meta["train_load_setting"]:
        sp = os.path.join(DATA_ROOT, setting)
        if not os.path.isdir(sp):
            continue
        for f in os.listdir(sp):
            if f.endswith('.mat') and bearing_of(f) in train_ids:
                file_tuples.append((sp, f))
    assert file_tuples, f"train-normal 파일 없음: {meta['train_load_setting']} / {train_ids}"

    # scaler fit (train-normal 전체 ch0..C)
    raw = []
    loaded = []
    for sp, f in file_tuples:
        mat = scipy.io.loadmat(os.path.join(sp, f))
        sig = load_stacked_signals(mat, f.replace('.mat', ''), sensor_names)  # (T,C)
        if sig is None:
            continue
        raw.append(sig)
        loaded.append(sig)
    scaler = StandardScaler().fit(np.concatenate(raw, axis=0))

    # window → ms_per_bin 누적(정규화 전 = amp_normalize로 나누기 전, ch0)
    F = window_size // 2 + 1
    psd_sum = np.zeros(F, dtype=np.float64)
    n_win = 0
    for sig in loaded:
        scaled = scaler.transform(sig)
        start = 0
        while start + window_size <= len(scaled):
            w = scaled[start:start + window_size]
            psd_sum += compute_band_ms_per_bin(w[:, 0])
            n_win += 1
            start += stride_size
    assert n_win > 0
    return psd_sum / n_win, n_win, F


def edges_and_groups(F, psd):
    """세 scheme의 (edges, groups, guard) 산출. linear는 edges를 groups로부터 유도."""
    out = {}
    for scheme in SCHEMES:
        if scheme == "energy":
            groups, info = compute_band_boundaries(F, N_BANDS, "energy", psd=psd,
                                                   min_width=MIN_WIDTH, return_info=True)
        else:
            groups, info = compute_band_boundaries(F, N_BANDS, scheme, return_info=True)
        if info["edges"] is not None:
            edges = list(info["edges"])
        else:  # linear: 각 그룹 첫 bin + F
            edges = [int(np.asarray(g)[0]) for g in groups] + [int(F)]
        out[scheme] = {"edges": edges, "groups": groups,
                       "guard": bool(info["guard_triggered"])}
    return out


def band_widths_bins(groups):
    return [int(len(np.asarray(g))) for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify_loader", action="store_true",
                    help="대표 fold에서 실제 loader의 energy edges와 EDA edges 일치를 assert")
    ap.add_argument("--seed", type=int, default=2026, help="fold 스펙을 읽을 대표 G-3a seed")
    args = ap.parse_args()

    hz_per_bin = FS_VIB / (2 * (2048 // 2))  # = FS/window_size
    rows = []          # per fold-scheme
    energy_edges_all = {}   # split->list of energy edges across LONO
    guard_hits = []
    missing = []

    for split in SPLITS:
        for lono in LONOS:
            meta, run = g3a_ckpt_meta(split, lono, args.seed)
            if meta is None:
                missing.append(run)
                continue
            psd, n_win, F = train_normal_mean_psd(meta)
            eg = edges_and_groups(F, psd)
            # linear band별 에너지 비중(진단): DC 포함 그룹합 / 총합
            lin_groups = eg["linear"]["groups"]
            tot = psd.sum()
            lin_frac = [float(psd[np.asarray(g)].sum() / tot) for g in lin_groups]
            energy_edges_all.setdefault(split, []).append(eg["energy"]["edges"])
            for scheme in SCHEMES:
                widths = band_widths_bins(eg[scheme]["groups"])
                edges = eg[scheme]["edges"]
                hz_edges = [round(e * hz_per_bin, 1) for e in edges]
                rows.append({
                    "split": split, "type": SPLIT_TYPE[split], "lono": lono,
                    "scheme": scheme, "F": F, "n_win": n_win,
                    "edges_bin": edges, "edges_hz": hz_edges,
                    "widths_bin": widths, "guard": eg[scheme]["guard"],
                    "lin_energy_frac": [round(x, 4) for x in lin_frac] if scheme == "linear" else None,
                })
                if eg[scheme]["guard"]:
                    guard_hits.append(f"{split}/LONO{lono}/{scheme}")

    # scheme 간 경계 근접도: energy 내부 경계(edges[1..K-1])가 linear/log 대비 얼마나 이동했나
    # (fold별 평균 |Δbin| 요약)
    proximity = []
    by_fold = {}
    for r in rows:
        by_fold.setdefault((r["split"], r["lono"]), {})[r["scheme"]] = r["edges_bin"]
    for (split, lono), d in by_fold.items():
        if not all(s in d for s in SCHEMES):
            continue
        e_en = np.array(d["energy"][1:-1]); e_li = np.array(d["linear"][1:-1]); e_lo = np.array(d["log"][1:-1])
        proximity.append({"split": split, "lono": lono,
                          "mean_abs_dbin_en_vs_lin": float(np.abs(e_en - e_li).mean()),
                          "mean_abs_dbin_en_vs_log": float(np.abs(e_en - e_lo).mean())})

    # ---- 정합성 보증(선택): 대표 fold에서 실제 loader energy edges == EDA edges ----
    verify_note = "미실행(--verify_loader로 켜기)"
    if args.verify_loader:
        vsplit, vlono = "023to1", 2
        meta, run = g3a_ckpt_meta(vsplit, vlono, args.seed)
        with contextlib.redirect_stdout(io.StringIO()):
            tr, _, _ = loader_Paderborn_OCC(
                root=DATA_ROOT,
                train_loads=meta["train_load_setting"], test_loads=meta["test_load_setting"],
                train_ids=meta["train_ids"], val_ids=meta["val_ids"],
                test_norm_ids=meta["test_norm_ids"], exclude_ids=meta.get("exclude_ids", []),
                window_size=int(meta["window_size"]), stride_size=int(meta["stride_size"]),
                sensor_mode=meta["sensor_mode"], amp_normalize=True, amp_n_bands=N_BANDS,
                amp_band_scheme="energy", amp_band_min_width=MIN_WIDTH,
                rms_eps=float(meta.get("rms_eps", 1e-8)), batch_size=256,
            )
        loader_edges = list(tr.dataset.amp_band_edges)
        psd, _, F = train_normal_mean_psd(meta)
        eda_edges = edges_and_groups(F, psd)["energy"]["edges"]
        assert loader_edges == eda_edges, f"divergence! loader={loader_edges} eda={eda_edges}"
        verify_note = f"✅ {vsplit}/LONO{vlono}: loader edges == EDA edges = {eda_edges}"

    write_report(rows, energy_edges_all, guard_hits, proximity, missing, hz_per_bin, verify_note)
    # 콘솔 요약
    print(f"folds 처리: {len(by_fold)}  missing_ckpt: {len(missing)}")
    print(f"guard 발동: {guard_hits if guard_hits else '없음'}")
    print(f"verify_loader: {verify_note}")
    print(f"리포트: {REPORT_PATH}")


def write_report(rows, energy_edges_all, guard_hits, proximity, missing, hz_per_bin, verify_note):
    L = []
    L.append("# P-2 — Band partition 선행 EDA (train-normal only, GPU 0)")
    L.append("")
    L.append("> 목적: GPU 전량 학습 전, fold별 train-normal 통계만으로 linear/log/energy 경계·band폭·PSD를 점검(P-G3 누수 방지).")
    _F = rows[0]['F'] if rows else 0
    L.append(f"> N\\*=6 고정, min_width={MIN_WIDTH} bin, F={_F} bin, "
             f"주파수 분해능 {hz_per_bin:.3f} Hz/bin, Nyquist ≈ {(_F - 1) * hz_per_bin:.0f} Hz.")
    L.append(f"> fold 스펙은 G-3a 체크포인트 metadata에서 로드. 코드 경로 정합성: {verify_note}")
    L.append("")
    if missing:
        L.append(f"⚠️ 누락 체크포인트({len(missing)}): {', '.join(missing)}")
        L.append("")

    # 1) 고정 scheme(linear/log)은 fold 무관 → 대표 1개만
    L.append("## 1. 고정 경계 scheme (데이터 무의존, 전 fold 동일)")
    L.append("")
    L.append("| scheme | edges(bin) | edges(Hz) | band 폭(bin) |")
    L.append("|---|---|---|---|")
    seen = set()
    for r in rows:
        if r["scheme"] in ("linear", "log") and r["scheme"] not in seen:
            seen.add(r["scheme"])
            L.append(f"| {r['scheme']} | {r['edges_bin']} | {r['edges_hz']} | {r['widths_bin']} |")
    L.append("")
    L.append("> linear는 bin 균등분할(현행 G-3a). log는 저주파에 밴드가 촘촘(고주파는 넓음). Hz = bin×분해능.")
    L.append("")

    # 2) energy: fold별 경계·폭·가드
    L.append("## 2. energy 경계 (fold별 train-normal PSD 적응)")
    L.append("")
    L.append("| split(type) | LONO | edges(bin) | edges(Hz) | band폭(bin) | guard |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        if r["scheme"] != "energy":
            continue
        L.append(f"| {r['split']}({r['type']}) | {r['lono']} | {r['edges_bin']} | {r['edges_hz']} | "
                 f"{r['widths_bin']} | {'⚠️' if r['guard'] else '-'} |")
    L.append("")
    L.append(f"> 최소폭 가드 발동: {guard_hits if guard_hits else '없음'}.")
    L.append("")

    # 3) energy 경계 fold 간 변동(split별 내부 경계 min/max)
    L.append("## 3. energy 내부 경계 fold 간 변동 (split별)")
    L.append("")
    L.append("| split | 내부경계 bin min | max | (전이 안정성 참고) |")
    L.append("|---|---|---|---|")
    for split, edges_list in energy_edges_all.items():
        arr = np.array([e[1:-1] for e in edges_list])  # (n_lono, K-1)
        L.append(f"| {split} | {arr.min(axis=0).tolist()} | {arr.max(axis=0).tolist()} | "
                 f"폭 {(arr.max(axis=0)-arr.min(axis=0)).tolist()} |")
    L.append("")

    # 4) linear band별 에너지 비중(진단): 왜 균등분할이 손해/이득인지 직관
    L.append("## 4. train-normal PSD의 linear band별 에너지 비중 (fold별)")
    L.append("")
    L.append("| split | LONO | band0..5 에너지 비중(합=1) |")
    L.append("|---|---|---|")
    for r in rows:
        if r["scheme"] == "linear" and r["lin_energy_frac"] is not None:
            L.append(f"| {r['split']} | {r['lono']} | {r['lin_energy_frac']} |")
    L.append("")
    L.append("> 특정 band에 에너지가 쏠릴수록 linear(등폭)는 정보가 불균형. energy는 이를 균등 에너지로 재분할.")
    L.append("")

    # 5) scheme 간 경계 근접도
    L.append("## 5. scheme 간 내부경계 근접도 (energy가 linear/log에서 얼마나 이동)")
    L.append("")
    L.append("| split | LONO | mean|Δbin| en-vs-lin | en-vs-log |")
    L.append("|---|---|---|---|")
    for p in proximity:
        L.append(f"| {p['split']} | {p['lono']} | {p['mean_abs_dbin_en_vs_lin']:.1f} | {p['mean_abs_dbin_en_vs_log']:.1f} |")
    L.append("")
    L.append("> Δbin이 작으면 energy가 사실상 linear/log와 유사(분할 방식 효과가 작을 것으로 예상).")
    L.append("")
    L.append("## 요약 관찰 (사람이 채움)")
    L.append("- (energy 경계가 저주파로 몰리는지 / 가드 발동 빈도 / fold 간 안정성 / linear와의 차이 크기)")
    L.append("")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
