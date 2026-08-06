"""F-0 무학습 표현·채널 진단 — 학습 전 "어떤 표현·채널이 정상/fault를 가르는가"를 값싸게 확인.

설계 문서: ../../docs/F0_representation_channel_design.md (그대로 기준, 재정의 안 함).

핵심: 학습·체크포인트 불필요. loader_Paderborn_OCC를 채널마다 한 번씩 호출해
dataset.windows(train-normal에만 fit된 StandardScaler로 스케일된 신호)·.ids·.label만 뽑아,
각 window를 표현별 고정차원 descriptor로 축약 → train-normal 참조 one-class 점수(maha/knn) →
target-normal(0) vs fault(1) 분리를 잰다. 순수 numpy/scipy/sklearn, GPU 불요.

표현(R1~R4): raw / rmsnorm(shape-only) / env(bandpass+Hilbert 포락선) / envspec(포락선 스펙트럼).
채널: vibration_1(Vy) / phase_current_1·2(C1·C2). 채널셋 Vy / C1C2 / C1C2Vy, late-fusion(mean/max).
scorer: maha(PCA-whiten+LedoitWolf Mahalanobis, primary) / knn(다봉 정상 교차확인).

산출:
- results/Paderborn/diag_F0_representation/<split>_LONO<n>_w<W>.json (fold×window별)
- results/Paderborn/diag_F0_representation/aggregate.json (집계)
- reports/report_F0_representation_diag.md

게이트(설계 §8): 어떤 표현·채널로도 정상/fault 안 갈리면 → F-1 안 감(raw-vib density 한계 확정).
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
from scipy import signal as sp_signal
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from Dataset.paderborn import loader_Paderborn_OCC  # noqa: E402

DATA_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "Paderborn"))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "Paderborn")
OUT_DIR = os.path.join(RESULTS_ROOT, "diag_F0_representation")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "report_F0_representation_diag.md")

# LOSO split → (train_loads, test_loads, 설명, fold_type). CLAUDE.md §4 표 그대로.
SETTING = {
    "123to0": (["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"], ["N15_M07_F10"], "기준조건 unseen", "compositional"),
    "023to1": (["N15_M07_F10", "N15_M01_F10", "N15_M07_F04"], ["N09_M07_F10"], "저속 unseen", "zero-support"),
    "013to2": (["N15_M07_F10", "N09_M07_F10", "N15_M07_F04"], ["N15_M01_F10"], "저토크 unseen", "zero-support"),
    "012to3": (["N15_M07_F10", "N09_M07_F10", "N15_M01_F10"], ["N15_M07_F04"], "저 radial force unseen", "zero-support"),
}

# LONO idx → (train_ids, val_ids, test_norm_ids). CLAUDE.md §4 표 그대로.
LONO = {
    1: (["K003", "K004", "K005", "K006"], ["K002"], ["K001"]),  # target-normal=K001 (고진폭)
    2: (["K001", "K004", "K005", "K006"], ["K003"], ["K002"]),  # target-normal=K002 (저진폭)
    3: (["K001", "K002", "K005", "K006"], ["K004"], ["K003"]),  # target-normal=K003 (고진폭)
    4: (["K001", "K002", "K003", "K006"], ["K005"], ["K004"]),  # target-normal=K004 (저진폭)
    5: (["K001", "K002", "K003", "K004"], ["K006"], ["K005"]),  # target-normal=K005 (저진폭)
    6: (["K002", "K003", "K004", "K005"], ["K001"], ["K006"]),  # target-normal=K006 (고진폭)
}

HIGH_AMP = {"K001", "K003", "K006"}  # 고진폭 정상(작업 B). 나머지 K00x = 저진폭.

# 설계 기본값: 세팅별 setting↔speed(rpm) — envspec 회전수 참고용(로그에만).
SETTING_RPM = {
    "N15_M07_F10": 1500, "N09_M07_F10": 900, "N15_M01_F10": 1500, "N15_M07_F04": 1500,
}

CHANNELS_ALL = ["vibration_1", "phase_current_1", "phase_current_2"]
CHANNEL_SETS = {
    "Vy": ["vibration_1"],
    "C1C2": ["phase_current_1", "phase_current_2"],
    "C1C2Vy": ["phase_current_1", "phase_current_2", "vibration_1"],
}
CH_ABBR = {"vibration_1": "Vy", "phase_current_1": "C1", "phase_current_2": "C2"}

# R3 envelope: 사전 고정 bandpass 대역(Hz). --bandpass는 R4(envspec) demodulation 단일대역.
ENV_BANDS = [(1000, 4000), (3000, 8000), (8000, 16000)]

EPS = 1e-12


# ---------------------------------------------------------------------------
# 통계 유틸 (작업 B/D와 동일 정의, 자립형 인라인)
# ---------------------------------------------------------------------------
def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
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
        "n": int(len(a)), "mean": float(np.mean(a)), "std": float(np.std(a)),
        "median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }


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
# 채널 sampling rate (설계 §1.3 — 코드로 검증)
# ---------------------------------------------------------------------------
def channel_sampling_rates(vib_fs=64000):
    """대표 .mat 하나를 읽어 채널별 Data 길이를 vibration_1과 비교해 fs를 잡는다.

    전류가 진동과 동일 고속채널(64kHz)이라는 설계 전제를 assert. 다르면 길이 비율로 fs 산정.
    """
    import scipy.io
    from Dataset.paderborn import _sensor_name_to_str
    sample_dir = os.path.join(DATA_ROOT, "N15_M07_F10")
    fname = next(f for f in sorted(os.listdir(sample_dir)) if f.endswith(".mat"))
    mat = scipy.io.loadmat(os.path.join(sample_dir, fname))
    key = fname.replace(".mat", "")
    y = mat[key][0, 0]["Y"]
    lens = {}
    for i in range(y.shape[1]):
        lens[_sensor_name_to_str(y[0, i]["Name"][0])] = int(y[0, i]["Data"].flatten().shape[0])
    vib_len = lens["vibration_1"]
    fs = {}
    for ch in CHANNELS_ALL:
        assert ch in lens, f"채널 {ch} 없음 (available={list(lens)})"
        fs[ch] = int(round(vib_fs * lens[ch] / vib_len))
    return fs, lens


# ---------------------------------------------------------------------------
# 표현 → 고정차원 descriptor (설계 §3). 입력 X:[N,L] (스케일된 window), fs.
# ---------------------------------------------------------------------------
def _time_feats(X):
    """scale-포함 시간영역 통계 [N,8]: rms, log(rms), p2p, kurtosis, crest, skew, std, zcr."""
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    p2p = X.max(axis=1) - X.min(axis=1)
    std = X.std(axis=1)
    mu = X.mean(axis=1, keepdims=True)
    xc = X - mu
    var = np.mean(xc ** 2, axis=1) + EPS
    skew = np.mean(xc ** 3, axis=1) / var ** 1.5
    kurt = np.mean(xc ** 4, axis=1) / var ** 2
    peak = np.max(np.abs(X), axis=1)
    crest = peak / (rms + EPS)
    zcr = np.mean(np.abs(np.diff(np.sign(X), axis=1)) > 0, axis=1)
    return np.stack([rms, np.log(rms + EPS), p2p, kurt, crest, skew, std, zcr], axis=1)


def _shape_feats(X):
    """scale-불변 형상 통계 [N,6]: kurtosis, crest, skew, zcr, spectral centroid, flatness."""
    mu = X.mean(axis=1, keepdims=True)
    xc = X - mu
    var = np.mean(xc ** 2, axis=1) + EPS
    skew = np.mean(xc ** 3, axis=1) / var ** 1.5
    kurt = np.mean(xc ** 4, axis=1) / var ** 2
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    crest = np.max(np.abs(X), axis=1) / (rms + EPS)
    zcr = np.mean(np.abs(np.diff(np.sign(X), axis=1)) > 0, axis=1)
    P = np.abs(np.fft.rfft(X, axis=1)) ** 2
    P = P[:, 1:]  # DC 제외
    freqs = np.arange(1, P.shape[1] + 1, dtype=float)
    psum = P.sum(axis=1) + EPS
    centroid = (P * freqs).sum(axis=1) / psum
    gmean = np.exp(np.mean(np.log(P + EPS), axis=1))
    amean = P.mean(axis=1) + EPS
    flatness = gmean / amean
    return np.stack([kurt, crest, skew, zcr, centroid, flatness], axis=1)


def _log_spaced_bands(fs, L, n_bands=24, f_lo=50.0):
    """log-spaced band edge index (rfft bin 기준). DC 제외."""
    f_hi = fs / 2.0
    edges = np.logspace(np.log10(f_lo), np.log10(f_hi), n_bands + 1)
    bin_hz = fs / L
    idx = np.clip(np.round(edges / bin_hz).astype(int), 1, L // 2)
    return idx


def _band_energy(P, edges):
    """P:[N, nbin] power (DC 포함 rfft) → 밴드별 에너지 합 [N, n_bands]."""
    out = np.empty((P.shape[0], len(edges) - 1), dtype=np.float64)
    for b in range(len(edges) - 1):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        out[:, b] = P[:, lo:hi].sum(axis=1)
    return out


def desc_raw(X, fs, cfg):
    """R1 raw: 시간영역 8 + log-spaced 밴드 로그에너지 24. 진폭 정보 보존(대조군)."""
    tf = _time_feats(X)
    P = np.abs(np.fft.rfft(X, axis=1)) ** 2
    edges = _log_spaced_bands(fs, X.shape[1], n_bands=24)
    be = _band_energy(P, edges)
    logbe = np.log(be + EPS)
    return np.concatenate([tf, logbe], axis=1)


def desc_rmsnorm(X, fs, cfg):
    """R2 rmsnorm(shape-only): window별 RMS 정규화 후 밴드 에너지 비율 24 + 형상 6."""
    rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True)) + EPS
    Xn = X / rms
    P = np.abs(np.fft.rfft(Xn, axis=1)) ** 2
    edges = _log_spaced_bands(fs, Xn.shape[1], n_bands=24)
    be = _band_energy(P, edges)
    ratio = be / (be.sum(axis=1, keepdims=True) + EPS)
    sf = _shape_feats(Xn)
    return np.concatenate([ratio, sf], axis=1)


def _bandpass_envelope(X, fs, band):
    """bandpass(butter 4차 SOS, sosfilt) 후 Hilbert 포락선 |analytic|. [N,L]→[N,L].

    포락선 크기·스펙트럼 크기만 쓰므로 zero-phase(filtfilt) 불요 → 단방향 sosfilt로 비용 절반.
    """
    lo, hi = band
    hi = min(hi, fs / 2.0 * 0.99)
    sos = sp_signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    xb = sp_signal.sosfilt(sos, X, axis=1)
    env = np.abs(sp_signal.hilbert(xb, axis=1))
    return env


def _ac_first_peak(ec):
    """평균제거 포락선 ec:[N,L]의 lag>0 자기상관 최댓값/lag0. FFT 기반 벡터화(O(N·L log L))."""
    N, L = ec.shape
    nfft = 1
    while nfft < 2 * L:
        nfft <<= 1
    F = np.fft.rfft(ec, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, n=nfft, axis=1)[:, :L]
    r0 = ac[:, 0] + EPS
    lag_lo = max(1, L // 100)
    if L <= lag_lo:
        return np.zeros(N)
    return ac[:, lag_lo:].max(axis=1) / r0


def _env_stats(env):
    """포락선 통계 [N,6]: env/mean 정규화 후 kurtosis, crest, skew, std/mean, 자기상관 첫피크, peak/rms."""
    m = env.mean(axis=1, keepdims=True) + EPS
    e = env / m
    ec = e - e.mean(axis=1, keepdims=True)
    var = np.mean(ec ** 2, axis=1) + EPS
    skew = np.mean(ec ** 3, axis=1) / var ** 1.5
    kurt = np.mean(ec ** 4, axis=1) / var ** 2
    rms = np.sqrt(np.mean(e ** 2, axis=1))
    crest = np.max(e, axis=1) / (rms + EPS)
    std_mean = e.std(axis=1)  # e는 평균1이라 std=std/mean
    peak_rms = np.max(e, axis=1) / (rms + EPS)
    ac_peak = _ac_first_peak(ec)  # 주기성 강도(충격형 결함 감지)
    return np.stack([kurt, crest, skew, std_mean, ac_peak, peak_rms], axis=1)


def desc_envelope(X, fs, cfg):
    """R3 env: 사전고정 대역별 bandpass+Hilbert 포락선 통계 concat (진폭 정규화 내장)."""
    blocks = []
    for band in cfg["env_bands"]:
        if band[0] >= fs / 2.0:
            continue
        env = _bandpass_envelope(X, fs, band)
        blocks.append(_env_stats(env))
    return np.concatenate(blocks, axis=1)


def desc_env_spectrum(X, fs, cfg):
    """R4 envspec: primary 대역 포락선의 스펙트럼(0~fmax)을 선형 binning + 총에너지 정규화."""
    env = _bandpass_envelope(X, fs, cfg["bandpass"])
    env = env - env.mean(axis=1, keepdims=True)
    Es = np.abs(np.fft.rfft(env, axis=1))
    bin_hz = fs / X.shape[1]
    fmax = cfg["envspec_fmax"]
    nmax = min(Es.shape[1], int(np.ceil(fmax / bin_hz)) + 1)
    Es = Es[:, 1:nmax]  # DC 제외, 0~fmax
    # 선형 binning: fmax를 nbins로 (설계 예 10Hz폭). 해상도보다 촘촘하면 raw bin 그대로.
    nbins = min(cfg["envspec_nbins"], Es.shape[1])
    if nbins < 1:
        nbins = 1
    edges = np.linspace(0, Es.shape[1], nbins + 1).astype(int)
    out = np.empty((Es.shape[0], nbins), dtype=np.float64)
    for b in range(nbins):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        out[:, b] = (Es[:, lo:hi] ** 2).sum(axis=1)
    out = out / (out.sum(axis=1, keepdims=True) + EPS)
    return out


REPRESENTATIONS = {
    "raw": desc_raw, "rmsnorm": desc_rmsnorm, "env": desc_envelope, "envspec": desc_env_spectrum,
}


# ---------------------------------------------------------------------------
# 무학습 one-class scorer (설계 §4). fit(train-normal desc); score(desc)→클수록 이상.
# ---------------------------------------------------------------------------
class OCCScorer:
    def __init__(self, kind, pca_var=0.95, knn_k=10, max_pca=None):
        self.kind = kind
        self.pca_var = pca_var
        self.knn_k = knn_k
        self.max_pca = max_pca

    def fit(self, D_tr):
        D_tr = np.asarray(D_tr, dtype=np.float64)
        self.mu_ = D_tr.mean(axis=0)
        self.sd_ = D_tr.std(axis=0)
        keep = self.sd_ > 1e-9  # 상수 차원(σ=0) 드롭
        self.keep_ = keep
        Z = (D_tr[:, keep] - self.mu_[keep]) / self.sd_[keep]
        # PCA로 차원 축소 (N/10 상한, 차원의 저주 완화)
        n_comp = min(Z.shape[1], max(1, Z.shape[0] // 10))
        if self.max_pca:
            n_comp = min(n_comp, self.max_pca)
        self.pca_ = PCA(n_components=n_comp, whiten=True, svd_solver="full")
        Zp = self.pca_.fit_transform(Z)
        # pca_var 이상 설명하는 최소 성분만 유지
        if 0 < self.pca_var < 1 and n_comp > 1:
            cum = np.cumsum(self.pca_.explained_variance_ratio_)
            k = int(np.searchsorted(cum, self.pca_var) + 1)
            k = max(1, min(k, Zp.shape[1]))
            self.k_ = k
        else:
            self.k_ = Zp.shape[1]
        Zp = Zp[:, :self.k_]
        if self.kind == "maha":
            self.cov_ = LedoitWolf().fit(Zp)  # whiten 후 잔차 공분산 shrinkage
        elif self.kind == "knn":
            k = min(self.knn_k, max(1, Zp.shape[0] - 1))
            self.knn_k_eff_ = k
            self.nn_ = NearestNeighbors(n_neighbors=k).fit(Zp)
        else:
            raise ValueError(f"unknown scorer kind: {self.kind}")
        return self

    def _transform(self, D):
        D = np.asarray(D, dtype=np.float64)
        Z = (D[:, self.keep_] - self.mu_[self.keep_]) / self.sd_[self.keep_]
        return self.pca_.transform(Z)[:, :self.k_]

    def score(self, D):
        Zp = self._transform(D)
        if self.kind == "maha":
            return self.cov_.mahalanobis(Zp)  # 제곱 거리 (단조 → AUROC 무관)
        d, _ = self.nn_.kneighbors(Zp)
        return d.mean(axis=1)


# ---------------------------------------------------------------------------
# 채널별 로딩 + descriptor 산출 (설계 §1)
# ---------------------------------------------------------------------------
def stratified_cap_index(ids, y, cap, min_per_group, seed):
    """(label,id) 그룹별로 균등하게 최대 cap개 window를 골라 정렬된 index 반환.

    분포를 보존하면서 window 수를 상한(무학습 진단 비용 억제). 그룹당 표본수를
    per_group=max(min_per_group, cap//n_groups)로 잡아 각 fault id도 recall 산정에 충분히 남긴다.
    ids/y 순서가 채널 간 동일하므로 동일 seed로 채널마다 호출해도 **같은 index**가 나온다(정렬 보존).
    """
    ids = np.asarray(ids); y = np.asarray(y, int)
    keys = sorted(set(zip(y.tolist(), ids.tolist())))
    if not keys:
        return np.arange(len(ids))
    per_group = max(min_per_group, cap // len(keys))
    rng = np.random.default_rng(seed)
    chosen = []
    for lab, bid in keys:
        gi = np.where((y == lab) & (ids == bid))[0]
        if len(gi) > per_group:
            gi = np.sort(rng.choice(gi, size=per_group, replace=False))
        chosen.append(gi)
    return np.sort(np.concatenate(chosen))


def load_channel(split, lono, channel, window, stride, batch_size, cap, min_per_group, seed):
    """단일 채널 로더 1회 호출 → dataset에서 windows/ids/label 직접 추출(순서보존) + 계층 subsample."""
    train_loads, test_loads, _, _ = SETTING[split]
    train_ids, val_ids, test_norm_ids = LONO[lono]
    with contextlib.redirect_stdout(io.StringIO()):
        tr, va, te, _ = loader_Paderborn_OCC(
            root=DATA_ROOT, train_loads=train_loads, test_loads=test_loads,
            train_ids=train_ids, val_ids=val_ids, test_norm_ids=test_norm_ids,
            exclude_ids=[], window_size=window, stride_size=stride,
            sensor_mode=channel, meta_source="static", batch_size=batch_size, label=False,
        )
    out = {}
    for name, loader in (("tr", tr), ("va", va), ("te", te)):
        ds = loader.dataset
        W = np.asarray(ds.windows, dtype=np.float32).reshape(len(ds.windows), window)
        ids = np.asarray(ds.ids); y = np.asarray(ds.label, dtype=int)
        if cap and len(W) > 0:
            # split별 결정적 seed(채널 간 동일 index 보장): 순서·ids 동일하므로 재현됨
            sel = stratified_cap_index(ids, y, cap, min_per_group, seed)
            W, ids, y = W[sel], ids[sel], y[sel]
        out[name] = {
            "X": W, "ids": ids, "y": y,
            "rms": np.sqrt(np.mean(W.astype(np.float64) ** 2, axis=1)),
        }
    return out


def describe_channel(chdata, fs, reps, cfg, batch_size):
    """채널 데이터의 tr/va/te에 대해 각 표현 descriptor 산출. raw window는 즉시 폐기."""
    desc = {}
    for rep in reps:
        fn = REPRESENTATIONS[rep]
        desc[rep] = {}
        for split_name in ("tr", "va", "te"):
            X = chdata[split_name]["X"]
            # 메모리 배치 처리
            parts = [fn(X[s:s + batch_size], fs, cfg) for s in range(0, len(X), batch_size)]
            desc[rep][split_name] = np.concatenate(parts, axis=0) if parts else np.empty((0, 1))
    return desc


# ---------------------------------------------------------------------------
# fold×window 하나 처리 (설계 §5·§6)
# ---------------------------------------------------------------------------
def zscore_by(ref, x):
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    if sd < 1e-12:
        return x - mu
    return (x - mu) / sd


def process_fold(split, lono, window, stride, scorers, reps, channel_sets, fusions, cfg, fs_map, batch_size):
    train_loads, test_loads, split_desc, fold_type = SETTING[split]
    _, _, test_norm_ids = LONO[lono]
    target_grp = ("high" if set(test_norm_ids) <= HIGH_AMP
                  else "low" if set(test_norm_ids).isdisjoint(HIGH_AMP) else "mixed")

    # 1) 채널별 로딩·정렬 assert·descriptor
    needed = CHANNELS_ALL
    chraw, chdesc = {}, {}
    ref_ids = None
    for ch in needed:
        cd = load_channel(split, lono, ch, window, stride, batch_size,
                          cfg["cap"], cfg["min_per_group"], cfg["seed"])
        # 채널 간 window 정렬 assert (동일 개수·동일 bearing id 순서)
        ids_cat = np.concatenate([cd["tr"]["ids"], cd["va"]["ids"], cd["te"]["ids"]])
        if ref_ids is None:
            ref_ids = ids_cat
            ref_shapes = (len(cd["tr"]["X"]), len(cd["va"]["X"]), len(cd["te"]["X"]))
        else:
            assert np.array_equal(ids_cat, ref_ids), f"채널 {ch} window 정렬 불일치"
            assert (len(cd["tr"]["X"]), len(cd["va"]["X"]), len(cd["te"]["X"])) == ref_shapes
        chdesc[ch] = describe_channel(cd, fs_map[ch], reps, cfg, batch_size)
        chraw[ch] = {"va": cd["va"], "te": cd["te"], "tr_rms": cd["tr"]["rms"]}

    te_y = chraw[needed[0]]["te"]["y"]
    te_ids = chraw[needed[0]]["te"]["ids"]
    va_y = chraw[needed[0]]["va"]["y"]
    tgt = te_y == 0
    flt = te_y == 1
    n_info = {"train_norm": int(ref_shapes[0]), "val_norm": int(ref_shapes[1]),
              "target_norm": int(tgt.sum()), "fault": int(flt.sum())}

    results = {}
    for scorer_kind in scorers:
        results[scorer_kind] = {}
        for rep in reps:
            # 채널별 점수 산출 (train desc로 fit → va/te score)
            ch_va_score, ch_te_score, ch_auroc = {}, {}, {}
            for ch in needed:
                sc = OCCScorer(scorer_kind, pca_var=cfg["pca_var"], knn_k=cfg["knn_k"]).fit(chdesc[ch][rep]["tr"])
                ch_va_score[ch] = sc.score(chdesc[ch][rep]["va"])
                ch_te_score[ch] = sc.score(chdesc[ch][rep]["te"])
                ch_auroc[ch] = safe_auroc(te_y, ch_te_score[ch])

            rep_res = {}
            for cs_name, cs_channels in channel_sets.items():
                # RMS 참조: 채널셋의 첫 채널(vibration 우선 없으면 C1) — 진폭 재상관 점검용
                rms_ref_ch = "vibration_1" if "vibration_1" in cs_channels else cs_channels[0]
                fuse_out = {}
                fuse_list = fusions if len(cs_channels) > 1 else ["mean"]  # 단일채널은 fusion 무의미
                for how in fuse_list:
                    # 채널별 val-normal z-score 후 결합
                    va_stack, te_stack = [], []
                    for ch in cs_channels:
                        ref = ch_va_score[ch][va_y == 0]
                        va_stack.append(zscore_by(ref, ch_va_score[ch]))
                        te_stack.append(zscore_by(ref, ch_te_score[ch]))
                    va_stack = np.stack(va_stack, axis=1)
                    te_stack = np.stack(te_stack, axis=1)
                    if how == "mean":
                        va_f, te_f = va_stack.mean(axis=1), te_stack.mean(axis=1)
                    else:
                        va_f, te_f = va_stack.max(axis=1), te_stack.max(axis=1)

                    fuse_out[how] = compute_metrics(
                        va_f, te_f, va_y, te_y, te_ids, tgt, flt,
                        chraw[rms_ref_ch], target_grp)
                rep_res[cs_name] = {
                    "fusion": fuse_out,
                    "per_channel_auroc": {ch: ch_auroc[ch] for ch in cs_channels},
                    "rms_ref_channel": rms_ref_ch,
                }
            results[scorer_kind][rep] = rep_res

    return {
        "split": split, "lono": lono, "window": window, "stride": stride,
        "fold_type": fold_type, "desc": split_desc,
        "target_norm_ids": sorted(set(te_ids[tgt].tolist())),
        "target_norm_amp_group": target_grp,
        "sampling_rates": {ch: fs_map[ch] for ch in needed},
        "rpm": {"train": [SETTING_RPM[s] for s in train_loads], "test": SETTING_RPM[test_loads[0]]},
        "n": n_info,
        "results": results,
    }


def compute_metrics(va_f, te_f, va_y, te_y, te_ids, tgt, flt, rms_ch, target_grp):
    """fused score 기반 지표 dict (설계 §6). va_f/te_f = val/test fused score."""
    auroc = safe_auroc(te_y, te_f)
    # 진폭 재상관: 정상 pooled(val-normal + target-normal, out-of-reference) — train은 in-sample이라 제외
    va_rms = rms_ch["va"]["rms"]; te_rms = rms_ch["te"]["rms"]
    norm_rms = np.concatenate([va_rms[va_y == 0], te_rms[tgt]])
    norm_score = np.concatenate([va_f[va_y == 0], te_f[tgt]])
    rho = spearman(norm_rms, norm_score)
    # threshold = val-normal fused score 95p
    vn = va_f[va_y == 0]
    thr = float(np.percentile(vn, 95)) if len(vn) else float("nan")
    # per-fault-id: recall @thr(threshold 전이 반영) + AUROC(threshold-free, gate 3의 "드러나는가")
    per_fault, per_fault_auroc = {}, {}
    tgt_scores = te_f[tgt]
    for fid in sorted(set(te_ids[flt].tolist())):
        m = flt & (te_ids == fid)
        per_fault[fid] = float(np.mean(te_f[m] > thr)) if m.sum() else float("nan")
        if m.sum() and tgt.sum():
            lbl = np.concatenate([np.zeros(int(tgt.sum())), np.ones(int(m.sum()))])
            per_fault_auroc[fid] = safe_auroc(lbl, np.concatenate([tgt_scores, te_f[m]]))
        else:
            per_fault_auroc[fid] = float("nan")
    tgt_fpr = float(np.mean(te_f[tgt] > thr)) if tgt.sum() else float("nan")
    # 고진폭 정상 vs fault (target-norm이 고진폭인 fold)
    high_fpr = tgt_fpr if target_grp == "high" else None
    high_vs_fault_auroc = None
    if target_grp == "high":
        lbl = np.concatenate([np.zeros(int(tgt.sum())), np.ones(int(flt.sum()))])
        scr = np.concatenate([te_f[tgt], te_f[flt]])
        high_vs_fault_auroc = safe_auroc(lbl, scr)
    return {
        "auroc": auroc, "spearman_rms_score": rho, "threshold": thr,
        "target_norm_fpr": tgt_fpr, "per_fault_recall": per_fault,
        "per_fault_auroc": per_fault_auroc,
        "high_amp_normal_fpr": high_fpr, "high_amp_vs_fault_auroc": high_vs_fault_auroc,
        "dist": {
            "target_normal": dist_stats(te_f[tgt]), "fault": dist_stats(te_f[flt]),
            "val_normal": dist_stats(vn),
        },
    }


# ---------------------------------------------------------------------------
# 집계 + 리포트 (설계 §7.4)
# ---------------------------------------------------------------------------
KEY_FAULTS = ["KA15", "KA22"]  # B에서 안 보이던 결함 (gate 3)


def _best_over(fold, scorer, rep, metric):
    """한 fold의 (rep, 모든 channel_set·fusion)에서 metric 최댓값·argmax 라벨."""
    best, tag = float("nan"), None
    for cs, csv in fold["results"][scorer][rep].items():
        for how, m in csv["fusion"].items():
            v = m.get(metric)
            if v is not None and v == v and (best != best or v > best):
                best, tag = v, f"{cs}/{how}"
    return best, tag


def aggregate(folds, scorers, reps):
    by_type = {}
    for ftype in ("zero-support", "compositional"):
        rows = [f for f in folds if f["fold_type"] == ftype]
        if not rows:
            continue
        entry = {"n_folds": len(rows), "by_scorer": {}}
        for scorer in scorers:
            entry["by_scorer"][scorer] = {}
            for rep in reps:
                aurocs, rhos, ka = [], [], []
                for f in rows:
                    b, _ = _best_over(f, scorer, rep, "auroc")
                    if b == b:
                        aurocs.append(b)
                    # 대표 조합(C1C2Vy/mean 있으면 그걸로 ρ·KA recall)
                    csv = f["results"][scorer][rep]
                    cs = "C1C2Vy" if "C1C2Vy" in csv else list(csv)[0]
                    how = "mean" if "mean" in csv[cs]["fusion"] else list(csv[cs]["fusion"])[0]
                    m = csv[cs]["fusion"][how]
                    if m["spearman_rms_score"] == m["spearman_rms_score"]:
                        rhos.append(m["spearman_rms_score"])
                    for kf in KEY_FAULTS:
                        v = m.get("per_fault_auroc", {}).get(kf)
                        if v is not None and v == v:
                            ka.append(v)
                entry["by_scorer"][scorer][rep] = {
                    "best_auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
                    "rho_rms_score_mean": float(np.mean(rhos)) if rhos else float("nan"),
                    "key_fault_auroc_mean": float(np.mean(ka)) if ka else float("nan"),
                }
        by_type[ftype] = entry
    return {"n_folds": len(folds), "scorers": scorers, "reps": reps, "by_fold_type": by_type}


def build_report(folds, agg, args):
    L = []
    L.append("# 작업 F-0 진단 — 무학습 표현·채널 (Paderborn, 학습 전)")
    L.append("")
    L.append(f"scorer={args.scorer}, 표현={args.representations}, window={args.windows}, "
             f"fold={len(folds)}개. stride=window×{args.stride_frac}, split별 계층 subsample "
             f"상한={args.max_windows_per_split}(seed {args.seed}). "
             f"학습·체크포인트 없음(피처추출+train-normal 참조 OCC만).")
    L.append("")
    L.append("**방법**: 채널마다 loader_Paderborn_OCC 호출(train-normal에만 StandardScaler fit) → "
             "window를 표현별 고정차원 descriptor로 축약 → train-normal descriptor로 OCC scorer "
             "(maha=PCA-whiten+LedoitWolf Mahalanobis / knn=k-NN 평균거리) fit → target-normal(0) vs "
             "fault(1) 분리. 채널 late-fusion(채널별 val-normal z-score 후 mean/max).")
    L.append("")
    L.append("**캐비앗**: (1) window는 train-normal fit StandardScaler로 스케일됨(전역 affine, "
             "주파수·형상 지표 불변). (2) 무학습 참조통계는 train-normal descriptor에만 fit(누수 없음). "
             "(3) Spearman(RMS,score)는 **val-normal+target-normal**(out-of-reference 정상)에서 계산 — "
             "train은 in-sample이라 거리 퇴화로 제외. (4) 표현·window·fold 사전 고정(target 보고 안 고름). "
             "(5) RMS 참조는 채널셋 첫 채널(vibration 우선). "
             "(6) 비용 억제 위해 stride=window(비겹침)·split별 계층 subsample(분포 보존, 각 fault id "
             "≥min_per_group 유지) — 분리도 통계엔 영향 미미, 절대 개수는 근사.")
    L.append("")

    # ---- §1 게이트 요약 (fold_type × 표현, best channel_set/fusion AUROC) ----
    L.append("## 1) 게이트 요약 — fold_type × 표현 (scorer별, best 채널셋 AUROC)")
    L.append("")
    for scorer in agg["scorers"]:
        L.append(f"### scorer = {scorer}")
        L.append("")
        L.append("| fold_type | n | 표현 | best AUROC | ρ(RMS,score) | KA15·KA22 AUROC |")
        L.append("|---|---|---|---|---|---|")
        for ftype, e in agg["by_fold_type"].items():
            for rep in agg["reps"]:
                s = e["by_scorer"][scorer][rep]
                L.append(f"| {ftype} | {e['n_folds']} | {rep} | {fmt(s['best_auroc_mean'])} | "
                         f"{fmt(s['rho_rms_score_mean'])} | {fmt(s['key_fault_auroc_mean'])} |")
        L.append("")
    L.append("> best AUROC = 그 (fold_type,표현)에서 채널셋·fusion 최댓값 fold 평균. "
             "ρ·KA AUROC = 대표 조합(C1C2Vy/mean) 기준. KA15·KA22 AUROC = target-normal vs 해당 fault "
             "id 분리(threshold-free, gate 3 '드러나는가'). threshold 전이(val→target) 실패로 recall은 "
             "0에 눌리므로 AUROC로 본다(§주의).")
    L.append("")

    # ---- §2 채널 우선순위 (per-channel AUROC) ----
    L.append("## 2) 채널 우선순위 — per-channel 단독 AUROC (선행연구 Vy 최약·C1C2 backbone 대조)")
    L.append("")
    L.append("| split | LONO | w | 표현 | scorer | Vy | C1 | C2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for f in folds:
        for scorer in agg["scorers"]:
            for rep in agg["reps"]:
                pc = {}
                for cs, csv in f["results"][scorer][rep].items():
                    pc.update(csv["per_channel_auroc"])
                L.append(f"| {f['split']} | {f['lono']} | {f['window']} | {rep} | {scorer} | "
                         f"{fmt(pc.get('vibration_1'))} | {fmt(pc.get('phase_current_1'))} | "
                         f"{fmt(pc.get('phase_current_2'))} |")
    L.append("")

    # ---- §3 fold 상세 (대표 조합 C1C2Vy/mean) ----
    L.append("## 3) fold 상세 — 대표 채널셋별 AUROC (scorer=primary, fusion=mean)")
    L.append("")
    primary = agg["scorers"][0]
    L.append(f"scorer={primary}, fusion=mean. AUROC/ρ(RMS,score)/target-normal FPR(@val95p)/고진폭 지표.")
    L.append("")
    L.append("| split | LONO | w | target(amp) | 표현 | Vy | C1C2 | C1C2Vy | ρ(C1C2Vy) | 고진폭FPR | 고진폭vsfault |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in folds:
        tg = f"{','.join(f['target_norm_ids'])}({f['target_norm_amp_group']})"
        for rep in agg["reps"]:
            r = f["results"][primary][rep]

            def au(cs):
                if cs not in r:
                    return None
                how = "mean" if "mean" in r[cs]["fusion"] else list(r[cs]["fusion"])[0]
                return r[cs]["fusion"][how]["auroc"]
            ccv = r.get("C1C2Vy")
            how = "mean" if (ccv and "mean" in ccv["fusion"]) else (list(ccv["fusion"])[0] if ccv else None)
            m = ccv["fusion"][how] if ccv else {}
            L.append(f"| {f['split']} | {f['lono']} | {f['window']} | {tg} | {rep} | "
                     f"{fmt(au('Vy'))} | {fmt(au('C1C2'))} | {fmt(au('C1C2Vy'))} | "
                     f"{fmt(m.get('spearman_rms_score'))} | {fmt(m.get('high_amp_normal_fpr'))} | "
                     f"{fmt(m.get('high_amp_vs_fault_auroc'))} |")
    L.append("")

    # ---- §4 게이트 판정 ----
    L.append("## 4) 게이트 판정 (설계 §8 — F-1 진입 여부)")
    L.append("")
    L.append("판정 기준(사전 고정, 서술 편의): (1)분리 AUROC≥0.65(zero-support)·0.70(compositional), "
             "(2)|ρ(RMS,score)|≤0.3, (3)KA15·KA22 per-fault AUROC가 0.5에서 유의미 상승(B에서 안 보이던 "
             "결함이 드러남), (4)고진폭 target fold에서 고진폭-정상 vs fault AUROC>0.5. "
             "**어떤 표현·채널로도 못 넘으면 F-1 안 감.**")
    L.append("")
    # 자동 요약: fold_type별 최선 표현·AUROC
    for ftype, e in agg["by_fold_type"].items():
        thr = 0.70 if ftype == "compositional" else 0.65
        best_rep, best_au, best_scorer = None, float("nan"), None
        for scorer in agg["scorers"]:
            for rep in agg["reps"]:
                v = e["by_scorer"][scorer][rep]["best_auroc_mean"]
                if v == v and (best_au != best_au or v > best_au):
                    best_au, best_rep, best_scorer = v, rep, scorer
        verdict = "통과 후보" if (best_au == best_au and best_au >= thr) else "미달"
        L.append(f"- **{ftype}** (n={e['n_folds']}, 기준 {thr}): 최선 = {best_rep}/{best_scorer} "
                 f"AUROC={fmt(best_au)} → **{verdict}**.")
    L.append("")
    L.append("> 위는 AUROC 게이트(1)만의 자동 요약. (2)~(4)는 §1·§3 표의 ρ·KA recall·고진폭 지표를 "
             "함께 보고 사용자가 종합 판정할 것.")
    L.append("")

    L.append("## 주의 / 한계")
    L.append("")
    L.append("- 무학습 진단(단일 통계 파이프라인, seed 무관). OCC scorer는 MTGFlow가 아니라 표현의 "
             "선형-분리 가능성 프록시 — 여기서 갈리면 학습 모델도 가를 여지가 있다는 필요조건.")
    L.append("- envelope spectrum은 window 길이에 민감(설계 §3.1). window별 표를 함께 볼 것.")
    L.append("- 정상 다봉성(고진폭/저진폭 K) 때문에 maha가 고진폭 정상을 오탐할 수 있어 knn 병행 — "
             "두 scorer의 §1 표를 대조.")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # 대표 5 fold(설계 §2.1): compositional 123to0×{1,2} + zero-support 023to1×{1,2} + 013to2×1.
    # "split:lono" 문자열. 스모크런은 예: --folds 123to0:1.
    ap.add_argument("--folds", nargs="+",
                    default=["123to0:1", "123to0:2", "023to1:1", "023to1:2", "013to2:1"])
    ap.add_argument("--representations", nargs="+", default=["raw", "rmsnorm", "env", "envspec"],
                    choices=list(REPRESENTATIONS.keys()))
    ap.add_argument("--channel_sets", nargs="+", default=["Vy", "C1C2", "C1C2Vy"],
                    choices=list(CHANNEL_SETS.keys()))
    ap.add_argument("--windows", type=int, nargs="+", default=[8192, 16384, 32768])
    # 비겹침(stride=window)이 기본 — window 상관을 줄이고 개수·비용 절감(설계 §3.2의 0.5 대비 조정).
    ap.add_argument("--stride_frac", type=float, default=1.0)
    ap.add_argument("--max_windows_per_split", type=int, default=2500,
                    help="split별 계층 subsample 상한(분포 보존, 비용 억제). 0=무제한.")
    ap.add_argument("--min_per_group", type=int, default=100,
                    help="계층 subsample에서 (label,id) 그룹당 최소 표본(각 fault id recall 보장).")
    ap.add_argument("--seed", type=int, default=2026, help="subsample 결정적 seed.")
    ap.add_argument("--scorer", nargs="+", default=["maha", "knn"], choices=["maha", "knn"])
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--knn_k", type=int, default=10)
    ap.add_argument("--bandpass", type=str, default="2000,6000", help="R4 envspec demodulation 대역 lo,hi(Hz)")
    ap.add_argument("--envspec_fmax", type=float, default=1000.0)
    ap.add_argument("--envspec_nbins", type=int, default=100)
    ap.add_argument("--fusion", nargs="+", default=["mean", "max"], choices=["mean", "max"])
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR)
    ap.add_argument("--no_report", action="store_true")
    args = ap.parse_args()

    fold_pairs = []
    for tok in args.folds:
        s, n = tok.split(":")
        assert s in SETTING, f"unknown split {s}"
        assert int(n) in LONO, f"unknown lono {n}"
        fold_pairs.append((s, int(n)))

    bp = tuple(float(x) for x in args.bandpass.split(","))
    cfg = {"pca_var": args.pca_var, "knn_k": args.knn_k, "env_bands": ENV_BANDS,
           "bandpass": bp, "envspec_fmax": args.envspec_fmax, "envspec_nbins": args.envspec_nbins,
           "cap": args.max_windows_per_split, "min_per_group": args.min_per_group, "seed": args.seed}
    channel_sets = {k: CHANNEL_SETS[k] for k in args.channel_sets}

    os.makedirs(args.out_dir, exist_ok=True)
    fs_map, lens = channel_sampling_rates()
    print(f"channel sampling rates: {fs_map}  (raw lengths: {lens})")
    for ch in CHANNELS_ALL:
        assert fs_map[ch] == 64000, f"채널 {ch} fs={fs_map[ch]} != 64000 (설계 전제 위반)"

    folds = []
    for split, lono in fold_pairs:
        for w in args.windows:
            stride = max(1, int(round(w * args.stride_frac)))
            print(f"[{split} LONO{lono} w{w}] 처리...")
            res = process_fold(split, lono, w, stride, args.scorer, args.representations,
                               channel_sets, args.fusion, cfg, fs_map, args.batch_size)
            folds.append(res)
            with open(os.path.join(args.out_dir, f"{split}_LONO{lono}_w{w}.json"), "w") as fh:
                json.dump(res, fh, indent=2, ensure_ascii=False)
            # 간단 진행 로그: primary scorer, envspec, C1C2Vy/mean AUROC
            sc0 = args.scorer[0]
            for rep in args.representations:
                r = res["results"][sc0][rep]
                cs = "C1C2Vy" if "C1C2Vy" in r else list(r)[0]
                how = "mean" if "mean" in r[cs]["fusion"] else list(r[cs]["fusion"])[0]
                m = r[cs]["fusion"][how]
                print(f"    {rep:8s} {cs}/{how}: AUROC={fmt(m['auroc'])} "
                      f"ρ(RMS,score)={fmt(m['spearman_rms_score'])}")

    if args.no_report or not folds:
        print(f"\nfold JSON {len(folds)}개 기록. (집계/리포트 생략)")
        return

    agg = aggregate(folds, args.scorer, args.representations)
    with open(os.path.join(args.out_dir, "aggregate.json"), "w") as fh:
        json.dump(agg, fh, indent=2, ensure_ascii=False)
    with open(REPORT_PATH, "w") as fh:
        fh.write(build_report(folds, agg, args))
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
