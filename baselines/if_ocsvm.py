"""B-1 고전 baseline — Isolation Forest · OC-SVM(RBF) fit+score 순수 함수 (CPU 전용).

부호 규약 (B-0 §2, report_B0_baseline_audit.md:68-70):
  sklearn IsolationForest.score_samples / OneClassSVM.decision_function 은 **클수록 정상**.
  따라서 anomaly score = **-score** 로 통일한다. AUROC·threshold 산출 모두 이 반전 점수를 쓴다.
  (반전을 빼먹으면 AUROC가 1-AUROC로 뒤집힌다 → 호출부에서 부호 sanity 확인 필수.)

OC-SVM raw 서브샘플 규칙 (B-0 §2, report_B0_baseline_audit.md:76-85):
  RBF SMO 비용은 표본수 n에 지배적(n×n 커널)이고 raw는 2048차원이라 fold당 train≈60k는 비현실적.
  → fold마다 min(N, n_train) 개를 seed=2024 고정 균등 비복원 샘플로 fit(결정론 유지). N=10,000.
  band 입력(6차원)은 전량 사용(호출부에서 subsample_cap=None). 단 n²이 비용을 지배하므로 60k가
  느리면 동일 N 폴백(호출부 결정) 후 리포트에 명기.

하이퍼(고정, TODO B-1 / Table 3):
  IF  n_estimators=100, random_state=seed
  OC-SVM  kernel='rbf', nu=0.05, gamma='scale'
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

OCSVM_SUBSAMPLE_N = 10_000       # raw 입력 서브샘플 상한 (B-0 확정)
OCSVM_SUBSAMPLE_SEED = 2024      # 서브샘플 고정 seed (OC-SVM 결정론 유지)


def subsample_indices(n, cap, seed):
    """min(cap, n) 개를 균등 비복원 샘플. n<=cap이면 전량(정렬된 arange)."""
    if cap is None or n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=cap, replace=False))
    return idx


def fit_score_if(train_X, val_X, test_X, seed, n_estimators=100, n_jobs=4):
    """Isolation Forest. anomaly = -score_samples (클수록 이상). random_state=seed.

    n_jobs 는 트리 병렬(fit/score)만 가르며 **결과(score)에 영향 없음** — IF는 random_state로만
    결정되므로 n_jobs 는 하이퍼가 아니라 순수 compute 옵션(B-G6 무관). raw 2048차원 scoring 가속용.
    """
    clf = IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    clf.fit(train_X)
    return {
        "tr": (-clf.score_samples(train_X)).astype(np.float64),
        "va": (-clf.score_samples(val_X)).astype(np.float64),
        "te": (-clf.score_samples(test_X)).astype(np.float64),
        "n_fit": int(len(train_X)),
    }


def fit_score_ocsvm(train_X, val_X, test_X, nu=0.05, gamma="scale",
                    subsample_cap=None, subsample_seed=OCSVM_SUBSAMPLE_SEED):
    """OC-SVM(RBF). anomaly = -decision_function. raw는 subsample_cap=N(서브샘플), band는 None(전량).

    fit은 서브샘플로, scoring(tr/va/te)은 전량에 적용한다(점수는 모든 window 필요).
    """
    idx = subsample_indices(len(train_X), subsample_cap, subsample_seed)
    fit_X = train_X[idx]
    clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    clf.fit(fit_X)
    return {
        "tr": (-clf.decision_function(train_X)).astype(np.float64),
        "va": (-clf.decision_function(val_X)).astype(np.float64),
        "te": (-clf.decision_function(test_X)).astype(np.float64),
        "n_fit": int(len(idx)),
        "n_support": int(clf.support_.shape[0]),
    }


def val_threshold_and_fpr(va_scores, pct=95.0):
    """threshold = val-normal anomaly score의 pct 백분위. val FPR = mean(va >= thr).

    정의상 val-normal FPR ≈ (100-pct)/100 ≈ 0.05 여야 한다(B-0 §2 sanity). 크게 벗어나면
    부호/스케일/split 파이프라인 결함 신호.
    """
    thr = float(np.percentile(va_scores, pct))
    fpr = float(np.mean(np.asarray(va_scores) >= thr))
    return thr, fpr
