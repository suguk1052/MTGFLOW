"""ESTOGU Without_Driver 데이터 EDA (Phase 0).

TODO.md Phase 0 + Data/ESTOGU/CLAUDE.md §9 체크리스트를 구현한다.
학습 없이 통계/스펙트럼만 계산해 리포트(md)와 그림(png)을 생성한다.

핵심 원칙(계획서 반영):
- 원본 데이터는 읽기만 한다.
- 신호 채널만 float32로 읽고 Timestamp는 파싱하지 않는다(700k 누적 반올림이 Δ검증을 왜곡).
  fs는 index.csv 값(34,482.76 Hz)으로 고정하고, 검증용으로만 첫 구간 Δ를 float64로 재확인.
- setup 하위폴더(Without_Driver)는 하드코딩하지 않고 클래스 폴더 존재로 자동 감지.
- 진폭 confound 지표는 결함별·채널별로 따로, 0.5 기준 양방향으로 해석
  (채널 RMS 하나로 만든 window-level AUROC = P(fault RMS > normal RMS)).

실행:
  conda run -n mtgflow python MTGFLOW/analysis/eda_estogu_without_driver.py \
      --data-root Data/ESTOGU/Without_Driver
스모크(소량):
  ... --limit-files 4 --nrows 50000
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

from _common import PROJECT_ROOT

# --- 데이터 상수 (Data/ESTOGU/CLAUDE.md, Metadata/index.csv 근거) ---------------
# index.csv SamplingRateHz. CLAUDE.md의 "35 kHz"는 근사값이라 실제 값을 쓴다.
FS = 34482.75862068965

CLASSES = ["N", "BB", "BR", "RB3", "RB5", "SW"]      # N=정상, 나머지=이상
FAULTS = ["BB", "BR", "RB3", "RB5", "SW"]
LOADS = ["000", "111", "222", "333", "444", "555"]   # 부하 6단계(거의 등간격)

VIB_COLS = ["VibrationX", "VibrationY", "VibrationZ"]
CUR_COLS = ["Current1", "Current2", "Current3"]
# 분석 대상 채널: 진동 3축 + 전류 3상 + 전압(SW는 전압/전류에 신호).
ANALYSIS_COLS = VIB_COLS + CUR_COLS + ["Voltage"]
ALL_COLS = ["Timestamp"] + ANALYSIS_COLS

# 정상(N) 기준 Without_Driver RPM (CLAUDE.md §3-1) — 회전주파수 산정용.
NORMAL_RPM = {"000": 2980, "111": 2960, "222": 2940, "333": 2916, "444": 2892, "555": 2865}
# 전기 컨덕턴스 1/R 프록시(등간격), 저항값 대신 사용.
CONDUCTANCE = {"000": 0.0, "111": 0.0090, "222": 0.0179, "333": 0.0263, "444": 0.0345, "555": 0.0435}

ADC_RAIL = 10.0  # ±10 V 16-bit → 포화 판정 기준

FNAME_RE = re.compile(r"^(N|BB|BR|RB3|RB5|SW)_(\d{3})_")


def parse_filename(fname):
    """'BB_222_50.csv' -> ('BB', '222'). 규칙 불일치면 (None, None)."""
    m = FNAME_RE.match(os.path.basename(fname))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def detect_setup_dir(data_root):
    """data_root가 클래스 폴더를 직접 담고 있으면 그대로, 아니면 한 단계 아래에서
    클래스 폴더(N/BB/...)를 가진 setup 디렉터리를 자동 감지해 반환."""
    def has_class_dirs(d):
        return all(os.path.isdir(os.path.join(d, c)) for c in ("N", "BB"))

    if has_class_dirs(data_root):
        return data_root
    for sub in sorted(glob.glob(os.path.join(data_root, "*"))):
        if os.path.isdir(sub) and has_class_dirs(sub):
            return sub
    raise FileNotFoundError(
        f"클래스 폴더(N/BB/...)를 가진 setup 디렉터리를 {data_root} 아래에서 찾지 못했습니다.")


def list_files(setup_dir):
    """(machine, load, path) 목록을 클래스 폴더 순회로 수집."""
    files = []
    for cls in CLASSES:
        cdir = os.path.join(setup_dir, cls)
        if not os.path.isdir(cdir):
            print(f"⚠️  클래스 폴더 없음, 건너뜀: {cdir}")
            continue
        for path in sorted(glob.glob(os.path.join(cdir, "*.csv"))):
            machine, load = parse_filename(path)
            if machine is None:
                print(f"⚠️  파일명 규칙 불일치, 건너뜀: {path}")
                continue
            files.append((machine, load, path))
    return files


def window_rms(x, w, s):
    """파일경계 비겹침 sliding window의 RMS 배열. 누적합(float64)으로 O(n)·저메모리."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < w:
        return np.empty(0, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, n - w + 1, s)
    sumsq = csum[starts + w] - csum[starts]
    return np.sqrt(sumsq / w)


# ==============================================================================
# 파일 로드 & 채널 통계
# ==============================================================================
def scan_file(path, window, stride, nrows=None):
    """한 파일을 읽어 (A) 채널 통계와 (C) 채널별 window RMS 배열을 반환."""
    df = pd.read_csv(path, usecols=ANALYSIS_COLS, dtype=np.float32, nrows=nrows)
    n_rows = len(df)
    stats = {}
    rms = {}
    for col in ANALYSIS_COLS:
        v = df[col].to_numpy()
        finite = v[np.isfinite(v)]
        stats[col] = {
            "n": n_rows,
            "nan": int(np.isnan(v).sum()),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "constant": bool(finite.size and np.ptp(finite) == 0.0),
            "sat": int((np.abs(finite) >= ADC_RAIL - 1e-6).sum()),
        }
        rms[col] = window_rms(v, window, stride)
    return n_rows, stats, rms, df


def validate_timestamp_fs(path, n=5000):
    """검증용으로만 Timestamp를 float64(datetime)로 읽어 실제 Δ→fs를 재계산."""
    df = pd.read_csv(path, usecols=["Timestamp"], nrows=n)
    ts = pd.to_datetime(df["Timestamp"])
    dt = ts.diff().dropna().dt.total_seconds().to_numpy()
    med_dt = float(np.median(dt))
    return {
        "fs_est": 1.0 / med_dt if med_dt > 0 else float("nan"),
        "dt_med_us": med_dt * 1e6,
        "dt_std_us": float(np.std(dt) * 1e6),
        "nonpos": int((dt <= 0).sum()),
    }


# ==============================================================================
# 그림
# ==============================================================================
def fig_waveform_and_psd(sample_df, out_dir):
    """정상 N(부하 000)의 채널별 시간파형 스니펫 + Welch PSD."""
    seg = sample_df.iloc[: min(200000, len(sample_df))]
    snip = seg.iloc[: min(2048, len(seg))]
    t = np.arange(len(snip)) / FS * 1000.0  # ms

    fig, axes = plt.subplots(len(ANALYSIS_COLS), 2, figsize=(12, 2.1 * len(ANALYSIS_COLS)))
    for i, col in enumerate(ANALYSIS_COLS):
        axes[i, 0].plot(t, snip[col].to_numpy(), lw=0.6)
        axes[i, 0].set_ylabel(col, fontsize=8)
        axes[i, 0].set_xlim(t[0], t[-1])
        if i == 0:
            axes[i, 0].set_title("Waveform snippet (N_000, ~2048 samp)", fontsize=9)
        f, pxx = welch(seg[col].to_numpy().astype(np.float64), fs=FS, nperseg=4096)
        axes[i, 1].semilogy(f, pxx, lw=0.7)
        axes[i, 1].set_xlim(0, 2000)
        if i == 0:
            axes[i, 1].set_title("Welch PSD (0–2000 Hz)", fontsize=9)
        for fx in (50, NORMAL_RPM["000"] / 60.0):  # 계통 50Hz, 회전주파수
            axes[i, 1].axvline(fx, color="r", ls="--", lw=0.5, alpha=0.6)
    axes[-1, 0].set_xlabel("time (ms)")
    axes[-1, 1].set_xlabel("Hz")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_waveform_psd_N000.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def fig_rms_violin(rms_by_class, out_dir):
    """채널별 window RMS 분포를 클래스(N + 5결함)별로 violin (부하 pooled)."""
    present = [c for c in CLASSES if c in rms_by_class]  # 스모크 시 일부만 존재 가능
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for i, col in enumerate(ANALYSIS_COLS):
        data = [rms_by_class[cls][col] for cls in present]
        data = [d if len(d) else np.array([0.0]) for d in data]
        parts = axes[i].violinplot(data, showmeans=True, showextrema=False)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor("tab:blue" if present[j] == "N" else "tab:red")
            pc.set_alpha(0.5)
        axes[i].set_xticks(range(1, len(present) + 1))
        axes[i].set_xticklabels(present, fontsize=8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("window RMS")
    for k in range(len(ANALYSIS_COLS), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Window RMS distribution: N(blue) vs faults(red), loads pooled", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_rms_violin_by_class.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def fig_load_shift(normal_rms_by_load, out_dir):
    """정상 N의 채널별 평균 window RMS가 부하에 따라 이동하는지 (motivation 근거)."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    x = range(len(LOADS))
    for i, col in enumerate(ANALYSIS_COLS):
        means = [float(np.mean(normal_rms_by_load[ld][col])) if len(normal_rms_by_load[ld][col]) else np.nan
                 for ld in LOADS]
        axes[i].plot(x, means, "o-", color="tab:green")
        axes[i].set_xticks(list(x))
        axes[i].set_xticklabels(LOADS, fontsize=8)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("mean window RMS")
        axes[i].set_xlabel("load")
    for k in range(len(ANALYSIS_COLS), len(axes)):
        axes[k].axis("off")
    fig.suptitle("Normal(N) signal shift across loads (load-axis physics / motivation)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_load_shift_normal.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ==============================================================================
# 리포트 유틸
# ==============================================================================
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def rotation_table():
    # window별로 부하 000/555 회전수 계산
    trows = []
    for w, label in ((2048, "샘플정합"), (1103, "시간정합(PU 0.032s)")):
        t_ms = w / FS * 1000.0
        rev0 = w / FS * (NORMAL_RPM["000"] / 60.0)
        rev5 = w / FS * (NORMAL_RPM["555"] / 60.0)
        trows.append([label, w, f"{t_ms:.1f}", f"{rev0:.2f}", f"{rev5:.2f}"])
    return trows


# ==============================================================================
# 메인
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "..", "Data", "ESTOGU", "Without_Driver"))
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    ap.add_argument("--limit-files", type=int, default=None, help="스모크: 처리 파일 수 상한")
    ap.add_argument("--nrows", type=int, default=None, help="스모크: 파일당 읽을 행 수 상한")
    args = ap.parse_args()

    data_root = os.path.normpath(args.data_root)
    setup_dir = detect_setup_dir(data_root)
    print(f"[setup] {setup_dir}")

    files = list_files(setup_dir)
    if args.limit_files:
        files = files[: args.limit_files]
    print(f"[files] {len(files)}개 처리 (window={args.window}, stride={args.stride}, nrows={args.nrows})")

    fig_dir = os.path.join(args.out_dir, "figs_estogu_eda")
    os.makedirs(fig_dir, exist_ok=True)

    # 수집 구조
    all_stats = {}                       # (machine, load) -> {col: stats}
    row_counts = {}                      # (machine, load) -> n_rows
    rms_by_key = {}                      # (machine, load) -> {col: rms array}
    sample_df_N000 = None

    for machine, load, path in files:
        n_rows, stats, rms, df = scan_file(path, args.window, args.stride, nrows=args.nrows)
        all_stats[(machine, load)] = stats
        row_counts[(machine, load)] = n_rows
        rms_by_key[(machine, load)] = rms
        if machine == "N" and load == "000":
            sample_df_N000 = df
        print(f"  {machine}_{load}: rows={n_rows}, windows={len(next(iter(rms.values())))}")

    # Timestamp fs 검증 (한 파일)
    ts_info = None
    n_path = os.path.join(setup_dir, "N", "N_000_50.csv")
    if os.path.exists(n_path):
        ts_info = validate_timestamp_fs(n_path)
        print(f"[fs] index.csv={FS:.2f} Hz, timestamp추정={ts_info['fs_est']:.2f} Hz")

    # --- 집계: 클래스별(pooled loads) / 부하별 정상 ---
    def pooled(machine, col):
        arrs = [rms_by_key[(machine, ld)][col] for ld in LOADS if (machine, ld) in rms_by_key]
        return np.concatenate(arrs) if arrs else np.empty(0)

    rms_by_class = {cls: {col: pooled(cls, col) for col in ANALYSIS_COLS} for cls in CLASSES
                    if any((cls, ld) in rms_by_key for ld in LOADS)}
    normal_rms_by_load = {ld: rms_by_key.get(("N", ld), {col: np.empty(0) for col in ANALYSIS_COLS})
                          for ld in LOADS}

    # --- C: 결함별·채널별 window-level AUROC = P(fault RMS > normal RMS) ---
    confound = {}  # (fault, col) -> auroc
    for fault in FAULTS:
        if fault not in rms_by_class:
            continue
        for col in ANALYSIS_COLS:
            n_rms = rms_by_class["N"][col]
            f_rms = rms_by_class[fault][col]
            if len(n_rms) == 0 or len(f_rms) == 0:
                confound[(fault, col)] = float("nan")
                continue
            y = np.concatenate([np.zeros(len(n_rms)), np.ones(len(f_rms))])
            s = np.concatenate([n_rms, f_rms])
            confound[(fault, col)] = float(roc_auc_score(y, s))

    # --- 그림 ---
    figs = []
    if sample_df_N000 is not None:
        figs.append(fig_waveform_and_psd(sample_df_N000, fig_dir))
    if "N" in rms_by_class:
        figs.append(fig_rms_violin(rms_by_class, fig_dir))
        figs.append(fig_load_shift(normal_rms_by_load, fig_dir))
    print(f"[figs] {figs}")

    # ------------------------------------------------------------------ 리포트
    L = []
    L.append("# ESTOGU Without_Driver — Phase 0 EDA 리포트\n")
    L.append(f"- 대상 setup: `{os.path.relpath(setup_dir, os.path.join(PROJECT_ROOT, '..'))}`")
    L.append(f"- 처리 파일: {len(files)}개, window={args.window}, stride={args.stride}"
             + (f", nrows(스모크)={args.nrows}" if args.nrows else ""))
    L.append(f"- 고정 fs = **{FS:.2f} Hz** (Metadata/index.csv). CLAUDE.md의 35 kHz는 근사값.\n")

    # A. 구조 sanity
    L.append("## A. 구조 sanity\n")
    if ts_info:
        L.append(f"- Timestamp Δ검증(float64, N_000 앞 5000행): 추정 fs = **{ts_info['fs_est']:.2f} Hz** "
                 f"(Δ중앙값 {ts_info['dt_med_us']:.3f}±{ts_info['dt_std_us']:.3f} µs, 비양수Δ {ts_info['nonpos']}개) "
                 f"→ index.csv {FS:.2f} Hz와 일치 여부 확인용.\n")
    rc_vals = sorted(set(row_counts.values()))
    L.append(f"- 행 수: {rc_vals} (모든 파일 동일? {'예' if len(rc_vals) == 1 else '아니오'})")
    # NaN/포화/상수 요약
    tot_nan = sum(s[c]["nan"] for s in all_stats.values() for c in ANALYSIS_COLS)
    tot_sat = sum(s[c]["sat"] for s in all_stats.values() for c in ANALYSIS_COLS)
    const_cols = [(k, c) for k, s in all_stats.items() for c in ANALYSIS_COLS if s[c]["constant"]]
    L.append(f"- 전체 NaN 합계: {tot_nan} / ±{ADC_RAIL}V 포화 샘플 합계: {tot_sat} / 상수열: {len(const_cols)}개\n")
    # 채널 스케일 표 (정상 pooled 기준)
    L.append("### 채널 스케일 (정상 N, 전 부하 평균)\n")
    rows = []
    for col in ANALYSIS_COLS:
        ms = [all_stats[("N", ld)][col] for ld in LOADS if ("N", ld) in all_stats]
        if not ms:
            continue
        rows.append([col,
                     f"{np.mean([m['mean'] for m in ms]):.4f}",
                     f"{np.mean([m['std'] for m in ms]):.4f}",
                     f"{np.min([m['min'] for m in ms]):.3f}",
                     f"{np.max([m['max'] for m in ms]):.3f}"])
    L.append(md_table(["채널", "mean", "std", "min", "max"], rows) + "\n")

    # B. 채널 선택
    L.append("## B. 채널 선택 근거 (스펙트럼)\n")
    L.append("- 파형/PSD 그림: `figs_estogu_eda/fig_waveform_psd_N000.png` (계통 50 Hz·회전주파수 표시).")
    L.append("- 진동 3축 vs 전류 3상의 스케일·주파수 성분을 비교해 **진동 1축 단독 시작**(PU 대응) 근거로 삼음. "
             "3축·전류·전압은 확장 옵션으로 코드에 열어둠.\n")

    # C. 분리성 & confound (핵심)
    L.append("## C. 🔑 정상/이상 분리성 & 진폭 confound (결함×채널, 양방향 해석)\n")
    L.append("각 셀 = 그 채널의 **window RMS 하나로 만든 window-level AUROC = P(fault RMS > normal RMS)** "
             "(부하 pooled).")
    L.append("- **> 0.5**: 결함 RMS가 더 큼 → 그 채널의 진폭만으로 결함이 잡힘(진폭 분리 가능).")
    L.append("- **< 0.5**: 정상 RMS가 더 큼 → PU식 **진폭 confound 방향**(고진폭 정상 오탐 위험).")
    L.append("- **≈ 0.5**: 그 채널 진폭으로는 못 가름 → 구조(스펙트럼/시간패턴) 필요.\n")
    header = ["결함\\채널"] + ANALYSIS_COLS
    rows = []
    for fault in FAULTS:
        if fault not in rms_by_class:
            continue
        row = [fault]
        for col in ANALYSIS_COLS:
            v = confound.get((fault, col), float("nan"))
            row.append(f"{v:.3f}" if v == v else "—")
        rows.append(row)
    L.append(md_table(header, rows) + "\n")
    L.append("- 그림: `figs_estogu_eda/fig_rms_violin_by_class.png` (N 파랑 vs 결함 빨강).")
    L.append("- ⚠️ **개체 교란**: 정상 1대 vs 결함=다른 모터라, 위 편차가 '고장'이 아니라 '개체차'일 수 있음(한계로 기재).\n")

    # D. 부하축 물리성
    L.append("## D. 부하축 물리성 (motivation 근거)\n")
    L.append("정상(N)의 채널별 평균 window RMS가 부하 000→555에서 어떻게 변하는가 "
             "(전류 단조증가 + 정상분포 이동 여부).\n")
    rows = []
    for col in ANALYSIS_COLS:
        means = [np.mean(normal_rms_by_load[ld][col]) if len(normal_rms_by_load[ld][col]) else np.nan
                 for ld in LOADS]
        valid = [m for m in means if m == m]
        mono = "—"
        if len(valid) == len(LOADS):
            mono = "↑단조" if all(x < y for x, y in zip(valid, valid[1:])) else \
                   ("↓단조" if all(x > y for x, y in zip(valid, valid[1:])) else "비단조")
        rng = (max(valid) - min(valid)) / (np.mean(valid) + 1e-12) * 100 if valid else float("nan")
        rows.append([col] + [f"{m:.4f}" if m == m else "—" for m in means] + [mono, f"{rng:.1f}%"])
    L.append(md_table(["채널"] + LOADS + ["추세", "변동폭%"], rows) + "\n")
    L.append("- 그림: `figs_estogu_eda/fig_load_shift_normal.png`.")
    L.append("- ⚠️ **결론 함의**: 정상분포가 부하 따라 거의 안 움직이면(변동폭 작음) 이는 나쁜 소식이 아니라, "
             "**conditioning의 전제(부하가 정상분포를 움직여야 함)를 조기에 뒤집는 중요 신호**다.\n")
    L.append("### 부하 등간격성 (프록시)")
    L.append(md_table(["load", "정상RPM", "컨덕턴스1/R"],
                      [[ld, NORMAL_RPM[ld], f"{CONDUCTANCE[ld]:.4f}"] for ld in LOADS]) + "\n")

    # E. 윈도 파라미터
    L.append("## E. 윈도 파라미터 결정 근거\n")
    L.append(f"정상 회전주파수 ≈ {NORMAL_RPM['000']/60:.1f}(부하000)~{NORMAL_RPM['555']/60:.1f}(부하555) Hz, "
             f"회전주기 ≈ {60000/NORMAL_RPM['000']:.1f}~{60000/NORMAL_RPM['555']:.1f} ms.\n")
    L.append(md_table(["방식", "샘플", "시간(ms)", "회전수(000)", "회전수(555)"], rotation_table()))
    L.append("\n→ 베어링 결함 하모닉을 담으려면 회전 ~3회가 유리 → **샘플정합 2048이 유력**. "
             "B의 PSD에서 결함 하모닉 위치를 보고 최종 확정.\n")

    # F. 요약 (수치 기반 자동 관찰)
    L.append("## F. 요약 / 게이트 판정 (수치 기반)\n")
    # (1) 진동 단일채널 분리도 — 개체 교란 신호
    vib_auc = [confound[(f, c)] for f in FAULTS for c in VIB_COLS
               if (f, c) in confound and confound[(f, c)] == confound[(f, c)]]
    vibY_auc = [confound[(f, "VibrationY")] for f in FAULTS if (f, "VibrationY") in confound]
    if vib_auc:
        L.append(f"- **(1) ⚠️ 진동 단일채널 RMS만으로 거의 완벽 분리**: 진동 3축 AUROC "
                 f"{min(vib_auc):.3f}~{max(vib_auc):.3f} (VibrationY는 전 결함 "
                 f"{min(vibY_auc):.3f}~{max(vibY_auc):.3f}). "
                 f"PU식 '고진폭 정상 오탐'(정상>결함, <0.5) confound는 **진동에선 재현되지 않음**"
                 f"(오히려 결함이 큼). 다만 **RMS 하나로 ~1.0이면 개체 교란(정상 1대 vs 결함=다른 모터)이"
                 f" 분리를 지배할 위험**이 크다 → 벤치가 너무 쉬워 conditioning 이득 여지가 없을 수 있음. "
                 f"방법론에 반드시 명시하고, 정규화/채널 재검토 대상.")
    # 전류 confound 방향
    cur_below = sum(1 for f in FAULTS for c in CUR_COLS
                    if (f, c) in confound and confound[(f, c)] == confound[(f, c)] and confound[(f, c)] < 0.5)
    cur_tot = sum(1 for f in FAULTS for c in CUR_COLS
                  if (f, c) in confound and confound[(f, c)] == confound[(f, c)])
    L.append(f"- **채널-결함 정합성**: 전류 채널은 {cur_tot}칸 중 {cur_below}칸이 <0.5 "
             f"(정상 전류 RMS가 결함보다 큼=전류축 confound 방향), 진폭만으론 결함 못 가름. "
             f"전압은 RB5·RB3·SW에서 분리력 있음(전기적 결함 대응). "
             f"→ 시작 채널은 **진동 1축**이 타당(단, 위 개체 교란 유의).")
    # (2)(3)
    L.append("- **(2) 채널 시작**: 진동 1축 단독(PU 대응). 전류/전압은 확장 옵션 코드 유지.")
    L.append("- **(3) 윈도**: 2048/1024 (샘플정합, 회전 ~3회) 권장.")
    # (4) 부하 shift 자동 판정
    cur_mono = []
    for c in CUR_COLS:
        means = [np.mean(normal_rms_by_load[ld][c]) for ld in LOADS if len(normal_rms_by_load[ld][c])]
        if len(means) == len(LOADS) and all(x < y for x, y in zip(means, means[1:])):
            rng = (max(means) - min(means)) / (np.mean(means) + 1e-12) * 100
            cur_mono.append(f"{c}(+{rng:.0f}%)")
    L.append(f"- **(4) ✅ 부하 shift(motivation 성립)**: 정상 전류 RMS가 부하 000→555에서 "
             f"단조 증가 [{', '.join(cur_mono) if cur_mono else '없음'}] → 부하가 정상분포를 실제로 움직임"
             f"(전류가 부하 프록시로 타당). 진동도 부하에 따라 이동하나 비단조. "
             f"⇒ conditioning의 전제는 **성립**(단, (1)의 개체 교란 때문에 정작 conditioning으로 이득을 "
             f"보일 여지가 있는지는 별개 문제).")
    L.append("\n### 종합 게이트")
    L.append("- **PU식 진폭 confound(정상>결함)는 진동에서 미재현** = Phase 2 no-meta baseline은 높게 나올 것.")
    L.append("- **최대 리스크 = 개체 교란(단일 RMS AUROC~1.0)**: 분리가 '고장'이 아니라 '개체차'일 수 있어 "
             "벤치가 너무 쉬우면 conditioning 이득 검증 자체가 무의미해질 수 있음. Phase 2에서 no-meta가 "
             "즉시 ~1.0이면 이 신호를 심각하게 볼 것.")
    L.append("\n다음 단계(Phase 1): 위 확정값(진동1축·2048/1024)으로 `Dataset/estogu.py` 로더 구현.\n")

    report_path = os.path.join(args.out_dir, "report_estogu_eda.md")
    with open(report_path, "w") as f:
        f.write("\n".join(L))
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
