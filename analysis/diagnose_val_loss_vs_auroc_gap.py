"""
train_log.jsonl(epoch별 val_loss_mean, test_auroc_log_only)이 있는 모든 run에 대해:
1) val_loss와 test AUROC의 상관관계(체크포인트 선택 기준인 val loss가 실제로
   좋은 판별 성능과 연결되는지)
2) "val_loss 기준으로 실제 선택된 epoch의 AUROC" vs "그 run이 도달했던 최고 AUROC"의 격차
   (격차가 크면 = 모델은 좋은 해를 지나갔지만 val-loss 기준 selection이 놓쳤다는 뜻)
를 계산한다.

train_log.jsonl이 없는 기존 run(로깅 도입 이전)은 대상에서 제외한다.
GPU 불필요 - 이미 저장된 로그 파일만 읽는다.
"""
import glob
import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")


def load_log(path):
    epochs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            epochs.append(json.loads(line))
    epochs.sort(key=lambda e: e["epoch"])
    return epochs


def analyze_run(path):
    epochs = load_log(path)
    val_losses = np.array([e["val_loss_mean"] for e in epochs])
    aurocs = np.array([e["test_auroc_log_only"] for e in epochs])
    if np.any(aurocs == None) or len(epochs) < 3:  # noqa: E711
        return None

    # 실제 checkpoint로 선택된 epoch: val_loss_mean의 global min이 처음 등장하는 지점
    # (main.py의 `if loss_best > mean_val_loss:` 저장 조건과 동일한 first-occurrence 규칙)
    selected_idx = int(np.argmin(val_losses))
    selected_auroc = float(aurocs[selected_idx])
    best_auroc = float(np.max(aurocs))
    best_idx = int(np.argmax(aurocs))

    corr = float(np.corrcoef(val_losses, aurocs)[0, 1]) if len(epochs) > 1 else float('nan')

    return {
        "n_epochs": len(epochs),
        "complete": len(epochs) >= 40,
        "corr_valloss_auroc": corr,
        "selected_epoch": epochs[selected_idx]["epoch"],
        "selected_auroc": selected_auroc,
        "best_epoch": epochs[best_idx]["epoch"],
        "best_auroc": best_auroc,
        "gap": best_auroc - selected_auroc,
    }


def main():
    log_paths = sorted(glob.glob(os.path.join(RESULTS_ROOT, "*", "train_log.jsonl")))
    print(f"train_log.jsonl {len(log_paths)}개 발견\n")

    rows = []
    for path in log_paths:
        run_dir = os.path.basename(os.path.dirname(path))
        stats = analyze_run(path)
        if stats is None:
            print(f"⚠️ 분석 불가(로그 부족/AUROC 없음): {run_dir}")
            continue
        stats["run"] = run_dir
        rows.append(stats)

    print(f"{'run':45s} {'epochs':7s} {'corr':7s} {'sel_ep':7s} {'sel_auroc':10s} {'best_ep':8s} {'best_auroc':11s} {'gap':6s}")
    for r in rows:
        flag = "" if r["complete"] else " (미완료)"
        print(f"{r['run']:45s} {r['n_epochs']:<7d} {r['corr_valloss_auroc']:<7.3f} "
              f"{r['selected_epoch']:<7d} {r['selected_auroc']:<10.4f} {r['best_epoch']:<8d} "
              f"{r['best_auroc']:<11.4f} {r['gap']:<6.4f}{flag}")

    corrs = np.array([r["corr_valloss_auroc"] for r in rows])
    gaps = np.array([r["gap"] for r in rows])
    print(f"\n총 {len(rows)}개 run")
    print(f"corr(val_loss, test_auroc) 평균={corrs.mean():.3f}, 중앙값={np.median(corrs):.3f}, "
          f"범위=[{corrs.min():.3f}, {corrs.max():.3f}]")
    print(f"gap(best_auroc - selected_auroc) 평균={gaps.mean():.3f}, 중앙값={np.median(gaps):.3f}, "
          f"범위=[{gaps.min():.3f}, {gaps.max():.3f}]")
    print(f"gap > 0.3인 run: {int(np.sum(gaps > 0.3))}/{len(rows)}")
    print(f"gap > 0.5인 run: {int(np.sum(gaps > 0.5))}/{len(rows)}")

    print("\ncorr이 양수(val loss 낮을수록 오히려 AUROC도 낮아짐, 기대와 반대)인 run:")
    for r in rows:
        if r["corr_valloss_auroc"] > 0:
            print(f"  {r['run']}: corr={r['corr_valloss_auroc']:.3f}, gap={r['gap']:.3f}")


if __name__ == '__main__':
    main()
