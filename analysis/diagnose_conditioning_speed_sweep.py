"""작업 A — §0 conditioning 동작 진단 (speed sweep NLL 곡선). 무학습·기존 체크포인트 재사용.

목적: unseen speed(023→1 저속)에서 conditional MTGFlow가 실제로 무엇을 하는지 실측한다.
동일 정상 window의 **신호(x)는 고정**한 채 flow에 주입되는 **speed 조건값만 격자로 스캔**하며
forward NLL 곡선을 그려, conditioning 함수의 형태를 본다(성능평가 아님, 함수 형태 진단).

판정(곡선을 baseline 산포로 정규화해 자동 분류):
- NLL 거의 불변(산포 내)      → conditioning이 speed를 사실상 무시
- 완만·선형(산포 수준의 변화)  → 제한적 외삽
- 학습 범위 밖에서 급변/발산   → OOD conditioning 폭주 (TODO §1′ 가설 지지)

핵심 원칙:
- **게이트0(sanity) 선통과**: 원래 speed(원본 meta)로 forward한 NLL로 AUROC를 재현해
  checkpoint 저장값과 일치하는지 먼저 확인. 불일치면 sweep 곡선은 무의미하므로 중단.
- **baseline 산포**: 원본 meta NLL의 window 간 자연 산포(σ)와(가능하면) no-meta NLL 분포를
  함께 저장. sweep ΔNLL이 이 산포보다 큰지로 "speed에 반응한다"를 판정.
- **격자를 train 범위 양쪽 밖까지 확장**: 023→1은 train speed가 사실상 단일값이라 target은
  이미 외삽점. 격자를 관측 밖으로 넉넉히 뻗어 곡선이 어디서 꺾이거나 발산하는지 본다.

meta 주입 규약(각 방식별로 flow에 실제 들어가는 정규화 speed 입력이 다름):
- static         : (rpm-1500)/1500 등 고정 공식. train(N15)=0, target(N09)=-0.4. → normalized 축.
- measured-concat: window mean/std를 train 기준 z-score(6D). speed_mean이 축. → train-z 축.
- FiLM(measured) : window mean을 train 기준 z-score(3D). speed_mean이 축. → train-z 축.

산출: results/Paderborn/diag_conditioning_speed_sweep/<method>_LONO<n>_s<seed>.json
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
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_conditioning_speed_sweep")

# 방식 → (배치 폴더, run_name 규칙). run_name에 _s{seed} 부착. no-meta는 baseline 참조용.
METHODS = {
    "static": dict(
        batch="LONO_C2_5seeds",
        run=lambda n: f"CA_raw_vib_023to1_LONO{n}",
        norm="static",  # 축 = 고정 정규화값
    ),
    "measured_concat": dict(
        batch="LONO_C2_measured_5seeds",
        run=lambda n: f"CA_raw_vib_023to1_LONO{n}_measured",
        norm="zscore",  # 축 = train 기준 z-score
    ),
    "film": dict(
        batch="LONO_FiLM_measured_5seeds",
        run=lambda n: f"CA_raw_vib_023to1_LONO{n}_FiLM_measured_5seeds",
        norm="zscore",
    ),
}
NOMETA = dict(batch="LONO_B3_5seeds", run=lambda n: f"raw_vib_023to1_LONO{n}")

# 저장된 overall_auroc(게이트0 대조 기준)를 읽을 파일명
METRICS_FNAME = "paderborn_per_bearing_metrics.json"

# static 정규화 공식 (Dataset/paderborn.py::normalize_paderborn_meta 와 동일). 축별 (기준값, 스케일).
STATIC_NORM = {"speed": (1500.0, 1500.0), "torque": (0.7, 0.7), "force": (1000.0, 1000.0)}

# static 3D meta 축 index
STATIC_AXIS_IDX = {"speed": 0, "torque": 1, "force": 2}
# measured mean(3D) 축 index
MEASURED_MEAN_AXIS_IDX = {"speed": 0, "torque": 1, "force": 2}
# measured meanstd(6D) 축 index (mean 차원만)
MEASURED_MEANSTD_AXIS_IDX = {"speed": 0, "torque": 2, "force": 4}


def ckpt_path(method_cfg, lono, seed):
    run = f"{method_cfg['run'](lono)}_s{seed}"
    return os.path.join(RESULTS_ROOT, method_cfg["batch"], run, "model.pth")


def saved_auroc(method_cfg, lono, seed):
    run = f"{method_cfg['run'](lono)}_s{seed}"
    p = os.path.join(RESULTS_ROOT, method_cfg["batch"], run, METRICS_FNAME)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return float(json.load(f)["overall_auroc"])


def build_model(ckpt, n_sensor, device):
    """checkpoint의 args(모델 hparam) + paderborn_metadata(meta 설정)로 모델을 복원한다."""
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
    """checkpoint metadata로 loader를 재구성 (test.py::build_loaders 와 동일 규약)."""
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


def nll_over_dataset(model, windows, metas, use_meta, device, batch_size=256):
    """주어진 window 신호(고정)와 meta로 window별 NLL(=-log_prob)을 계산한다.

    windows: [N, window_size] (loader의 scaled 신호). metas: [N, meta_dim] 또는 None.
    """
    n = len(windows)
    win = windows.shape[1]
    out = np.empty(n, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            xb = torch.as_tensor(windows[s:e], dtype=torch.float32, device=device)
            # loader __getitem__ 규격 [1,win,1]을 배치로: x = [B, K=1, L=win, D=1]
            xb = xb.reshape(e - s, win, 1, 1).transpose(1, 2).contiguous()
            mb = None
            if use_meta:
                mb = torch.as_tensor(metas[s:e], dtype=torch.float32, device=device)
            log_prob = model.test(xb, mb)  # [B]
            out[s:e] = (-log_prob).detach().cpu().numpy()
    return out


def gate0(model, test_loader, use_meta, device, batch_size, ref_auroc):
    """원본 meta로 forward한 NLL로 AUROC 재현 → 저장값과 대조. 통과 여부와 함께 반환."""
    ds = test_loader.dataset
    windows = np.asarray(ds.windows, dtype=np.float32)
    metas = np.asarray(ds.metas, dtype=np.float32) if use_meta else None
    labels = np.asarray(ds.label, dtype=int)
    nll = nll_over_dataset(model, windows, metas, use_meta, device, batch_size)
    auroc = float(roc_auc_score(labels, nll))
    passed = (ref_auroc is None) or (abs(auroc - ref_auroc) < 1e-3)
    return {
        "reproduced_auroc": auroc,
        "saved_auroc": ref_auroc,
        "abs_diff": None if ref_auroc is None else abs(auroc - ref_auroc),
        "passed": bool(passed),
        "nll_all": nll,
        "labels": labels,
        "windows": windows,
        "metas": metas,
    }


def build_grid(norm, axis, metas_normal, axis_idx, n_grid, static_raw_min, static_raw_max,
               z_range):
    """축 격자(정규화 입력 공간)와 참조점(train 관측, target)을 구성한다.

    반환: dict(grid_input[정규화 입력값], grid_display[표시용 raw/z], train_ref, target_ref, unit)
    """
    target_vals = metas_normal[:, axis_idx]  # 원본 target window의 이 축 입력값
    target_mean = float(np.mean(target_vals))
    target_min, target_max = float(np.min(target_vals)), float(np.max(target_vals))

    if norm == "static":
        base, scale = STATIC_NORM[axis]
        # raw 격자 → 정규화. target(N09 speed=900)=-0.4, train(N15)=0.0.
        raw_grid = np.linspace(static_raw_min, static_raw_max, n_grid)
        grid_input = (raw_grid - base) / scale
        return {
            "unit": "normalized (static formula)",
            "grid_input": grid_input.tolist(),
            "grid_display_raw": raw_grid.tolist(),      # rpm/Nm/N 실단위
            "train_ref_input": 0.0,                     # train 세팅 speed=1500 → 0
            "target_ref_input": target_mean,            # target 세팅 (≈-0.4 for speed)
            "target_ref_range": [target_min, target_max],
        }
    else:  # zscore: 격자를 z 공간에서 target 포함·train 밖까지 확장
        lo = min(-z_range, target_min - 1.0)
        hi = max(z_range, target_max + 1.0)
        grid_input = np.linspace(lo, hi, n_grid)
        return {
            "unit": "train z-score",
            "grid_input": grid_input.tolist(),
            "grid_display_raw": None,                   # measured raw 단위는 z만으로 충분
            "train_ref_input": 0.0,                     # train 평균 = z0 (정의상)
            "target_ref_input": target_mean,            # target(N09) 실제 z (unseen 외삽 정도)
            "target_ref_range": [target_min, target_max],
        }


def sweep_axis(model, gate, use_meta, axis_idx, grid, device, batch_size):
    """정상 window 신호 고정, 지정 축 입력만 grid 값으로 치환하며 평균 NLL 곡선을 그린다."""
    labels = gate["labels"]
    normal_mask = labels == 0
    windows = gate["windows"][normal_mask]
    metas = gate["metas"][normal_mask].copy() if use_meta else None

    # baseline: 원본 meta(정상 window)의 NLL 분포 = window 간 자연 산포 (판정 척도)
    base_nll = gate["nll_all"][normal_mask]
    baseline = {
        "mean": float(np.mean(base_nll)),
        "std": float(np.std(base_nll)),
        "median": float(np.median(base_nll)),
        "n_windows": int(normal_mask.sum()),
    }

    curve = []
    for g in grid["grid_input"]:
        m = metas.copy()
        m[:, axis_idx] = g  # speed 차원만 치환, 나머지 축·다른 window별 값은 유지
        nll = nll_over_dataset(model, windows, m, use_meta, device, batch_size)
        curve.append({
            "input": float(g),
            "mean_nll": float(np.mean(nll)),
            "std_nll": float(np.std(nll)),
            # baseline σ 대비 평균 이동량 (판정 정규화 지표)
            "delta_over_baseline_sigma": float((np.mean(nll) - baseline["mean"]) /
                                               (baseline["std"] + 1e-12)),
        })
    return baseline, curve


def classify(curve, grid, baseline):
    """곡선을 baseline σ 대비로 3분류: 무시 / 선형(제한적 외삽) / 폭주.

    - train 관측점(train_ref_input) 근처 평균 NLL을 기준으로, 격자 전체의 최대 이탈을 σ 배수로.
    - target 지점과 격자 끝(외삽) 이탈을 함께 본다.
    """
    inputs = np.array([c["input"] for c in curve])
    devs = np.array([abs(c["delta_over_baseline_sigma"]) for c in curve])
    max_dev = float(np.max(devs))
    # target 지점 이탈
    ti = int(np.argmin(np.abs(inputs - grid["target_ref_input"])))
    target_dev = float(abs(curve[ti]["delta_over_baseline_sigma"]))
    # 격자 양끝(가장 바깥 외삽) 이탈
    edge_dev = float(max(abs(curve[0]["delta_over_baseline_sigma"]),
                         abs(curve[-1]["delta_over_baseline_sigma"])))
    if max_dev < 0.5:
        verdict = "ignore"        # NLL 거의 불변 → speed 무시
    elif edge_dev > 3.0 and edge_dev > 2.0 * target_dev:
        verdict = "blowup"        # 범위 밖에서 급변/발산 → OOD conditioning 폭주
    else:
        verdict = "linear"        # 완만·선형 → 제한적 외삽
    return {
        "verdict": verdict,
        "max_dev_sigma": max_dev,
        "target_dev_sigma": target_dev,
        "edge_dev_sigma": edge_dev,
    }


def run_one(method, lono, seed, axis, args, device):
    cfg = METHODS[method]
    path = ckpt_path(cfg, lono, seed)
    if not os.path.exists(path):
        print(f"  [skip] checkpoint 없음: {path}")
        return None
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["paderborn_metadata"]
    use_meta = bool(meta["use_meta"])
    ref_auroc = saved_auroc(cfg, lono, seed)

    _, _, test_loader, n_sensor = build_loader(meta, args.batch_size)
    model = build_model(ckpt, n_sensor, device)

    gate = gate0(model, test_loader, use_meta, device, args.batch_size, ref_auroc)
    print(f"  [gate0] {method} LONO{lono}: reproduced={gate['reproduced_auroc']:.6f} "
          f"saved={gate['saved_auroc']} diff={gate['abs_diff']} passed={gate['passed']}")
    if not gate["passed"]:
        print(f"  [STOP] 게이트0 실패 — sweep 무의미. 원인(로드/차원/정규화/정렬) 점검 필요.")
        return {"method": method, "lono": lono, "seed": seed, "gate0_passed": False,
                "gate0": {k: gate[k] for k in ("reproduced_auroc", "saved_auroc", "abs_diff", "passed")}}

    if args.sanity_only:
        return {"method": method, "lono": lono, "seed": seed, "gate0_passed": True,
                "gate0": {k: gate[k] for k in ("reproduced_auroc", "saved_auroc", "abs_diff", "passed")}}

    # 축 index 결정
    if cfg["norm"] == "static":
        axis_idx = STATIC_AXIS_IDX[axis]
    elif meta["measured_meta_stats"] == "mean":
        axis_idx = MEASURED_MEAN_AXIS_IDX[axis]
    else:
        axis_idx = MEASURED_MEANSTD_AXIS_IDX[axis]

    normal_mask = gate["labels"] == 0
    metas_normal = gate["metas"][normal_mask]
    grid = build_grid(cfg["norm"], axis, metas_normal, axis_idx, args.n_grid,
                      args.static_rpm_min, args.static_rpm_max, args.z_range)
    baseline, curve = sweep_axis(model, gate, use_meta, axis_idx, grid, device, args.batch_size)
    verdict = classify(curve, grid, baseline)

    # no-meta baseline NLL 분포(같은 window들) — 있으면 참고 저장
    nometa_info = None
    np_path = ckpt_path(NOMETA, lono, seed)
    if os.path.exists(np_path):
        nm_ckpt = torch.load(np_path, map_location=device)
        nm_meta = nm_ckpt["paderborn_metadata"]
        _, _, nm_test, nm_ns = build_loader(nm_meta, args.batch_size)
        nm_model = build_model(nm_ckpt, nm_ns, device)
        nm_windows = np.asarray(nm_test.dataset.windows, dtype=np.float32)
        nm_labels = np.asarray(nm_test.dataset.label, dtype=int)
        nm_nll = nll_over_dataset(nm_model, nm_windows, None, False, device, args.batch_size)
        nm_norm = nm_nll[nm_labels == 0]
        nometa_info = {"mean": float(np.mean(nm_norm)), "std": float(np.std(nm_norm)),
                       "n_windows": int(len(nm_norm))}

    print(f"  [sweep] {method} LONO{lono} axis={axis}: verdict={verdict['verdict']} "
          f"max_dev={verdict['max_dev_sigma']:.2f}σ target_dev={verdict['target_dev_sigma']:.2f}σ "
          f"edge_dev={verdict['edge_dev_sigma']:.2f}σ | target_input={grid['target_ref_input']:.4f}")

    return {
        "method": method, "lono": lono, "seed": seed, "axis": axis,
        "gate0_passed": True,
        "gate0": {k: gate[k] for k in ("reproduced_auroc", "saved_auroc", "abs_diff", "passed")},
        "meta_source": meta["meta_source"], "measured_meta_stats": meta["measured_meta_stats"],
        "meta_inject": meta.get("meta_inject", "concat"), "axis_idx": axis_idx,
        "grid": grid, "baseline": baseline, "nometa_baseline": nometa_info,
        "curve": curve, "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["static", "measured_concat", "film"],
                    choices=list(METHODS.keys()))
    ap.add_argument("--lonos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--axis", type=str, default="speed", choices=["speed", "torque", "force"])
    ap.add_argument("--n_grid", type=int, default=25)
    ap.add_argument("--z_range", type=float, default=6.0, help="measured z 격자 절대 범위(±). target 포함하도록 자동 확장.")
    ap.add_argument("--static_rpm_min", type=float, default=300.0, help="static speed 격자 rpm 최소.")
    ap.add_argument("--static_rpm_max", type=float, default=2400.0, help="static speed 격자 rpm 최대.")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--sanity_only", action="store_true", help="게이트0만 실행하고 종료.")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for method in args.methods:
        for lono in args.lonos:
            res = run_one(method, lono, args.seed, args.axis, args, device)
            if res is None:
                continue
            results.append(res)
            tag = "sanity" if args.sanity_only else args.axis
            out = os.path.join(args.out_dir, f"{method}_LONO{lono}_s{args.seed}_{tag}.json")
            with open(out, "w") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)

    # 게이트0 요약
    print("\n=== 게이트0 요약 ===")
    for r in results:
        g = r["gate0"]
        print(f"  {r['method']:>16} LONO{r['lono']}: passed={g['passed']} "
              f"(repro={g['reproduced_auroc']:.4f}, saved={g['saved_auroc']}, diff={g['abs_diff']})")
    if not args.sanity_only:
        print("\n=== sweep 판정 요약 ===")
        for r in results:
            if not r.get("gate0_passed") or "verdict" not in r:
                continue
            v = r["verdict"]
            print(f"  {r['method']:>16} LONO{r['lono']} {r['axis']}: {v['verdict']:>7} "
                  f"(max={v['max_dev_sigma']:.2f}σ edge={v['edge_dev_sigma']:.2f}σ target={v['target_dev_sigma']:.2f}σ)")


if __name__ == "__main__":
    main()
