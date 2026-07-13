"""
LOSO C2(static meta) 전체 4 split x 6 LONO x 5 seed에 대해
meta_encoder 출력 norm과 그 시드의 테스트 AUROC 상관관계를 확인하는 진단 스크립트.

배경: 123to0_LONO4에서는 "meta_encoder 출력 norm이 크면 AUROC가 나쁘다"는 패턴이 깨끗하게
보였지만, 013to2_LONO3에서는 같은 패턴이 뚜렷하지 않았다. GPU 재학습 전에 저장된 checkpoint만
CPU로 훑어서 이 상관관계가 전반적으로 성립하는지, 반례(norm 크지만 AUROC도 높음 / norm 작지만
AUROC도 낮음)가 얼마나 많은지 확인한다. 아울러 no-meta(B3)도 같은 split/LONO에서 시드 변동이
큰지 비교해, 문제가 meta_encoder 스케일에 국한된 것인지 그 split/LONO 자체의 학습 난이도 문제인지
가늠한다.

GPU 불필요 - 저장된 model.pth를 CPU로 로드해 meta_encoder만 forward한다.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.MTGFLOW import MTGFLOW
from Dataset.paderborn import get_paderborn_setting_meta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")

SPLITS = ["023to1", "012to3", "013to2", "123to0"]
LONOS = [1, 2, 3, 4, 5, 6]


def load_model_and_metaemb_norm(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    cfg = checkpoint['args']
    metadata = checkpoint.get('paderborn_metadata', {})
    meta_input_dim = metadata.get('meta_input_dim', 3)
    model = MTGFLOW(
        cfg['n_blocks'], cfg['input_size'], cfg['hidden_size'], cfg['n_hidden'],
        cfg['window_size'], 1,
        model=cfg['model'], batch_norm=cfg['batch_norm'],
        use_meta=cfg['use_meta'], meta_input_dim=meta_input_dim, meta_emb_dim=cfg['meta_emb_dim'],
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    test_settings = metadata.get('test_load_setting') or metadata.get('load_setting') or []
    held_out = test_settings[0] if test_settings else None
    with torch.no_grad():
        meta = torch.tensor(get_paderborn_setting_meta(held_out), dtype=torch.float32).unsqueeze(0)
        emb = model.meta_encoder(meta).squeeze(0)
    return float(emb.norm().item())


def collect_c2_rows():
    rows = []
    for split in SPLITS:
        for lono in LONOS:
            run_dir = f"CA_raw_vib_{split}_LONO{lono}"
            summary_path = os.path.join(RESULTS_ROOT, "LONO_C2_5seeds", run_dir, "summary_seeds.json")
            if not os.path.exists(summary_path):
                print(f"⚠️ summary 없음, 건너뜀: {summary_path}")
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            for entry in summary["per_seed"]:
                seed = entry["seed"]
                auroc = entry["auroc"]
                ckpt_path = os.path.join(RESULTS_ROOT, "LONO_C2_5seeds", f"{run_dir}_s{seed}", "model.pth")
                if not os.path.exists(ckpt_path):
                    print(f"⚠️ checkpoint 없음, 건너뜀: {ckpt_path}")
                    continue
                norm = load_model_and_metaemb_norm(ckpt_path)
                rows.append({"split": split, "lono": lono, "seed": seed, "auroc": auroc, "meta_emb_norm": norm})
    return rows


def collect_b3_std():
    out = {}
    for split in SPLITS:
        for lono in LONOS:
            summary_path = os.path.join(RESULTS_ROOT, "LONO_B3_5seeds", f"raw_vib_{split}_LONO{lono}", "summary_seeds.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            out[(split, lono)] = (summary["auroc_mean"], summary["auroc_std"])
    return out


def main():
    rows = collect_c2_rows()
    b3 = collect_b3_std()

    print(f"\n총 {len(rows)}개 (split, LONO, seed) 데이터포인트 수집\n")
    print(f"{'split':8s} {'LONO':4s} {'seed':6s} {'meta_norm':10s} {'AUROC':7s}   | B3(no-meta) mean±std (같은 split/LONO)")
    for r in sorted(rows, key=lambda r: (r["split"], r["lono"], r["seed"])):
        b3_stat = b3.get((r["split"], r["lono"]))
        b3_str = f"{b3_stat[0]:.3f}±{b3_stat[1]:.3f}" if b3_stat else "N/A"
        print(f"{r['split']:8s} {r['lono']:<4d} {r['seed']:<6d} {r['meta_emb_norm']:<10.4f} {r['auroc']:<7.3f}   | {b3_str}")

    norms = np.array([r["meta_emb_norm"] for r in rows])
    aurocs = np.array([r["auroc"] for r in rows])
    corr_all = np.corrcoef(norms, aurocs)[0, 1]
    print(f"\n전체 {len(rows)}개 포인트 Pearson corr(meta_emb_norm, AUROC) = {corr_all:.3f}")

    print("\nsplit별 상관:")
    for split in SPLITS:
        sub = [r for r in rows if r["split"] == split]
        n = np.array([r["meta_emb_norm"] for r in sub])
        a = np.array([r["auroc"] for r in sub])
        corr = np.corrcoef(n, a)[0, 1] if len(sub) > 1 else float('nan')
        print(f"  {split}: n={len(sub)}, corr={corr:.3f}, norm range=[{n.min():.2f},{n.max():.2f}], AUROC range=[{a.min():.2f},{a.max():.2f}]")

    print("\n반례(직관과 어긋나는 경우) 탐색:")
    median_norm = np.median(norms)
    high_norm_good = [r for r in rows if r["meta_emb_norm"] > median_norm and r["auroc"] > 0.8]
    low_norm_bad = [r for r in rows if r["meta_emb_norm"] <= median_norm and r["auroc"] < 0.4]
    print(f"  norm 큰데(>{median_norm:.2f}) AUROC도 높음(>0.8): {len(high_norm_good)}건")
    for r in high_norm_good:
        print(f"    {r['split']} LONO{r['lono']} seed{r['seed']}: norm={r['meta_emb_norm']:.2f}, AUROC={r['auroc']:.3f}")
    print(f"  norm 작은데(<={median_norm:.2f}) AUROC도 낮음(<0.4): {len(low_norm_bad)}건")
    for r in low_norm_bad:
        print(f"    {r['split']} LONO{r['lono']} seed{r['seed']}: norm={r['meta_emb_norm']:.2f}, AUROC={r['auroc']:.3f}")

    print("\nB3(no-meta) split/LONO별 시드 std (참고, meta 없음이라 norm 분석 대상 아님):")
    for split in SPLITS:
        for lono in LONOS:
            stat = b3.get((split, lono))
            if stat:
                print(f"  {split} LONO{lono}: mean={stat[0]:.3f}, std={stat[1]:.3f}")


if __name__ == '__main__':
    main()
