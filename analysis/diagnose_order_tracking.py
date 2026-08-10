# ==============================================================================
# 작업 E 1단계 / Phase A — order tracking 무학습 정렬 진단 (GPU 불필요)
#
# 목적(게이트): 023→1 저속 LOSO에서
#   (a) OT 전(Hz축) source(1500rpm) 대비 target(900rpm)의 rotation-harmonic peak가
#       speed비(900/1500=0.6)만큼 실제로 이동(speed-proportional shift)하는지,
#   (b) source/target에 동일한 order-domain 변환(각도 등간격 리샘플)을 적용한 뒤
#       order축에서 같은 성분의 residual mismatch가 유의하게 감소(정렬)하는지
# 를 정량화한다. fault separability는 게이트 조건이 아니다(Phase B 학습에서 판단).
#
# 설계 요지:
#   - speed 효과 격리를 위해 "같은 bearing"을 N15_M07_F10(1500rpm)와 N09_M07_F10(900rpm)
#     두 세팅에서 비교한다(두 세팅은 speed만 다르고 torque/force 동일).
#   - 진단 스펙트럼은 모델 window(2048, <1 rev)와 분리해 긴 segment로 계산(Welch, nperseg 큼)
#     → 저차 order 해상도 확보.
#   - OT(등속 근사, nominal 파일평균 speed)는 신호를 SPR(samples/rev) 등간격으로 리샘플하는
#     것이며, 등속에서는 주파수축을 f_rot로 나눠 order축으로 바꾸는 것과 수학적으로 동치.
#     본 스크립트는 (i) 실제 시간영역 리샘플(Phase B loader와 동일한 방식)로 order 스펙트럼을
#     구하고, (ii) instantaneous speed 적분 기반 리샘플을 sanity로 함께 계산해 nominal과 동치인지 확인.
# ==============================================================================

import json
import os
import sys

import numpy as np
import scipy.io
from scipy import signal as sps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from analysis._common import extract_bearing_id  # noqa: E402

ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "Data", "Paderborn"))
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "Paderborn", "E_order_tracking")

FS_VIB = 64000          # vibration sampling rate (Hz)
FS_SPEED = 4000         # measured speed sampling rate (Hz)
WINDOW_SIZE = 2048      # Phase B 모델 입력 길이(진단 해상도와 무관, 참고용)

# speed만 다르고 torque/force 동일한 두 세팅 → speed 효과 격리
SRC_SETTING = "N15_M07_F10"   # 1500 rpm
TGT_SETTING = "N09_M07_F10"   # 900 rpm
SRC_RPM = 1500.0
TGT_RPM = 900.0

# order 리샘플 공통 파라미터
SPR = 2560              # samples per revolution (1500rpm의 native fs/f_rot=64000/25=2560와 일치)
NPERSEG_ANG = SPR * 8   # order 스펙트럼 Welch segment(8 rev) → order 해상도 SPR/NPERSEG=0.125 order
NPERSEG_HZ = 20480      # Hz 스펙트럼 Welch segment → freq 해상도 64000/20480≈3.125 Hz

# 진단 대상 bearing (같은 ID를 두 세팅에서 로드). 정상 = 게이트 핵심, fault = 참고(harmonic 정렬 관찰).
NORMAL_BEARINGS = ["K001", "K002"]            # 고진폭 / 저진폭
FAULT_CANDIDATES = ["KA04", "KA15", "KI04", "KI21", "KB23"]  # 존재하는 것만 사용, 최대 2개


def load_channels(mat_path):
    """.mat에서 vibration_1, speed 채널을 dict로 반환."""
    mat = scipy.io.loadmat(mat_path)
    key = os.path.basename(mat_path).replace(".mat", "")
    y = mat[key][0, 0]["Y"]
    out = {}
    for i in range(y.shape[1]):
        sr = y[0, i]
        nm = np.asarray(sr["Name"][0]).squeeze()
        nm = str(nm.item()) if nm.shape == () else "".join(str(c) for c in nm.tolist())
        out[nm] = sr["Data"].flatten().astype(np.float64)
    return out


def find_file(setting, bearing_id):
    d = os.path.join(ROOT, setting)
    if not os.path.isdir(d):
        return None
    cands = sorted(f for f in os.listdir(d)
                   if f.endswith(".mat") and extract_bearing_id(f) == bearing_id)
    return os.path.join(d, cands[0]) if cands else None


def order_resample_constant(x, fs, f_rot_hz, spr):
    """등속(nominal) 근사 OT: 신호 x를 각도영역에서 spr samples/rev가 되도록 리샘플.
    현재 samples/rev = fs/f_rot. 목표 = spr. 길이비 = spr*f_rot/fs.
    Phase B loader가 파일 전체에 적용할 방식과 동일(등속이라 상수배 리샘플)."""
    n_in = len(x)
    n_out = int(round(n_in * spr * f_rot_hz / fs))
    # scipy.signal.resample_poly: 정수 up/down 필요. 근사 유리수로 변환.
    from fractions import Fraction
    frac = Fraction(n_out, n_in).limit_denominator(10000)
    up, down = frac.numerator, frac.denominator
    if up == 0:
        up = 1
    return sps.resample_poly(x, up, down)


def order_resample_instantaneous(x, speed_sig, fs_vib, fs_speed, spr):
    """instantaneous OT(sanity): 측정 speed를 각도로 적분해 등각도 그리드로 리샘플.
    speed_sig(4kHz, rpm)를 vib 시간축(64kHz)으로 보간 → 순시 회전주파수(rev/s) → 적분해 각도(rev)
    → 등간격 각도(1/spr rev 간격) 지점에서 x를 보간."""
    n = len(x)
    t_vib = np.arange(n) / fs_vib
    t_speed = np.arange(len(speed_sig)) / fs_speed
    rpm_on_vib = np.interp(t_vib, t_speed, speed_sig)     # rpm at vib rate
    frot_on_vib = rpm_on_vib / 60.0                        # rev/s
    # 각도(rev) = ∫ frot dt (누적 사다리꼴)
    dtheta = frot_on_vib / fs_vib
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (dtheta[1:] + dtheta[:-1]))])
    theta_max = theta[-1]
    n_out = int(np.floor(theta_max * spr))
    theta_grid = np.arange(n_out) / spr                    # 등간격 각도
    return np.interp(theta_grid, theta, x)


def order_spectrum(x_ang, spr, nperseg):
    """각도영역 신호(spr samples/rev)의 order 스펙트럼. 반환: (orders, psd)."""
    nperseg = min(nperseg, len(x_ang))
    f, pxx = sps.welch(x_ang, fs=spr, nperseg=nperseg, detrend="constant")
    # welch의 fs=spr → f 단위는 cycles/rev = order
    return f, pxx


def hz_spectrum(x, fs, nperseg):
    nperseg = min(nperseg, len(x))
    f, pxx = sps.welch(x, fs=fs, nperseg=nperseg, detrend="constant")
    return f, pxx


def top_peaks(axis, psd, band, n_peaks=8):
    """band=(lo,hi) 구간에서 상위 peak 위치(axis 단위)를 prominence 순으로 반환."""
    lo, hi = band
    mask = (axis >= lo) & (axis <= hi)
    a, p = axis[mask], psd[mask]
    if len(p) < 5:
        return []
    logp = np.log(p + 1e-20)
    peaks, props = sps.find_peaks(logp, prominence=0.3, distance=3)
    if len(peaks) == 0:
        return []
    order_by_prom = np.argsort(props["prominences"])[::-1][:n_peaks]
    sel = peaks[order_by_prom]
    return sorted(float(a[i]) for i in sel)


def match_ratio(src_peaks, tgt_peaks, expected, tol_rel=0.15):
    """src peak 각각에 대해 tgt에서 (expected 비율에 가장 가까운) 대응 peak를 찾아 비율 리스트 반환.
    expected: Hz축이면 0.6(=900/1500), order축이면 1.0. tol_rel: 허용 상대오차."""
    ratios = []
    for ps in src_peaks:
        target_pos = ps * expected
        if not tgt_peaks:
            continue
        pt = min(tgt_peaks, key=lambda x: abs(x - target_pos))
        if target_pos > 0 and abs(pt - target_pos) / target_pos <= tol_rel:
            ratios.append(pt / ps)
    return ratios


def spectral_corr(src_axis, src_psd, tgt_axis, tgt_psd, grid):
    """공통 grid로 log-PSD 보간 후 Pearson 상관(정렬 정도의 전역 지표)."""
    s = np.interp(grid, src_axis, np.log(src_psd + 1e-20))
    t = np.interp(grid, tgt_axis, np.log(tgt_psd + 1e-20))
    s = (s - s.mean()) / (s.std() + 1e-12)
    t = (t - t.mean()) / (t.std() + 1e-12)
    return float(np.mean(s * t))


def diagnose_bearing(bearing_id, kind):
    src_f = find_file(SRC_SETTING, bearing_id)
    tgt_f = find_file(TGT_SETTING, bearing_id)
    if src_f is None or tgt_f is None:
        return None
    src = load_channels(src_f)
    tgt = load_channels(tgt_f)
    xs, xt = src["vibration_1"], tgt["vibration_1"]

    # --- OT 전: Hz축 스펙트럼 ---
    fs_hz, ps_hz = hz_spectrum(xs, FS_VIB, NPERSEG_HZ)
    ft_hz, pt_hz = hz_spectrum(xt, FS_VIB, NPERSEG_HZ)

    # --- OT(nominal): 등속 근사 리샘플 → order 스펙트럼 ---
    xs_ang = order_resample_constant(xs, FS_VIB, SRC_RPM / 60.0, SPR)
    xt_ang = order_resample_constant(xt, FS_VIB, TGT_RPM / 60.0, SPR)
    fs_ord, ps_ord = order_spectrum(xs_ang, SPR, NPERSEG_ANG)
    ft_ord, pt_ord = order_spectrum(xt_ang, SPR, NPERSEG_ANG)

    # --- OT(instantaneous): sanity ---
    xs_ang_i = order_resample_instantaneous(xs, src["speed"], FS_VIB, FS_SPEED, SPR)
    xt_ang_i = order_resample_instantaneous(xt, tgt["speed"], FS_VIB, FS_SPEED, SPR)
    fs_ord_i, ps_ord_i = order_spectrum(xs_ang_i, SPR, NPERSEG_ANG)
    ft_ord_i, pt_ord_i = order_spectrum(xt_ang_i, SPR, NPERSEG_ANG)

    # --- peak 정렬 지표 ---
    # 저차 대역: rotation·bearing 결함 harmonic. Hz축 band는 source 기준 0~2000Hz(=0~80 order @1500).
    hz_band = (10.0, 2000.0)
    ord_band = (0.4, 80.0)
    src_hz_peaks = top_peaks(fs_hz, ps_hz, hz_band)
    tgt_hz_peaks = top_peaks(ft_hz, pt_hz, hz_band)
    src_ord_peaks = top_peaks(fs_ord, ps_ord, ord_band)
    tgt_ord_peaks = top_peaks(ft_ord, pt_ord, ord_band)

    hz_ratios = match_ratio(src_hz_peaks, tgt_hz_peaks, expected=TGT_RPM / SRC_RPM)   # ≈0.6 기대
    ord_ratios = match_ratio(src_ord_peaks, tgt_ord_peaks, expected=1.0)              # ≈1.0 기대

    # --- 전역 상관: Hz축(정렬 안됨 기대 낮음) vs order축(정렬 시 높음) ---
    corr_hz = spectral_corr(fs_hz, ps_hz, ft_hz, pt_hz, grid=np.linspace(10, 2000, 1000))
    corr_ord = spectral_corr(fs_ord, ps_ord, ft_ord, pt_ord, grid=np.linspace(0.4, 80, 1000))
    corr_ord_inst = spectral_corr(fs_ord_i, ps_ord_i, ft_ord_i, pt_ord_i,
                                  grid=np.linspace(0.4, 80, 1000))
    # nominal vs instantaneous order 스펙트럼 상관(≈1이면 동치, nominal 근사 타당)
    corr_nom_inst_src = spectral_corr(fs_ord, ps_ord, fs_ord_i, ps_ord_i,
                                      grid=np.linspace(0.4, 80, 1000))
    corr_nom_inst_tgt = spectral_corr(ft_ord, pt_ord, ft_ord_i, pt_ord_i,
                                      grid=np.linspace(0.4, 80, 1000))

    return {
        "bearing": bearing_id,
        "kind": kind,
        "src_file": os.path.basename(src_f),
        "tgt_file": os.path.basename(tgt_f),
        "n_src_samples": int(len(xs)),
        "n_tgt_samples": int(len(xt)),
        "rev_per_window": {  # 참고: 모델 window(2048) 회전수
            "src_1500rpm": WINDOW_SIZE / FS_VIB * SRC_RPM / 60.0,
            "tgt_900rpm": WINDOW_SIZE / FS_VIB * TGT_RPM / 60.0,
        },
        "hz_peaks": {"src": src_hz_peaks, "tgt": tgt_hz_peaks},
        "order_peaks": {"src": src_ord_peaks, "tgt": tgt_ord_peaks},
        "hz_peak_ratio_median": float(np.median(hz_ratios)) if hz_ratios else None,
        "hz_peak_ratio_n": len(hz_ratios),
        "order_peak_ratio_median": float(np.median(ord_ratios)) if ord_ratios else None,
        "order_peak_ratio_n": len(ord_ratios),
        # order축 잔차(1에서 벗어난 정도, 작을수록 정렬 좋음)
        "order_residual_median_abs": (float(np.median(np.abs(np.array(ord_ratios) - 1.0)))
                                      if ord_ratios else None),
        "corr_hz": corr_hz,
        "corr_order": corr_ord,
        "corr_order_inst": corr_ord_inst,
        "corr_nominal_vs_inst": {"src": corr_nom_inst_src, "tgt": corr_nom_inst_tgt},
    }


def speed_constancy():
    """사전검증 재확인: 두 세팅 speed 상수성·rev/window 수치."""
    out = {}
    for setting, rpm in [(SRC_SETTING, SRC_RPM), (TGT_SETTING, TGT_RPM)]:
        f = find_file(setting, "K001")
        ch = load_channels(f)
        s = ch["speed"]
        out[setting] = {
            "nominal_rpm": rpm,
            "speed_len": int(len(s)),
            "speed_mean": float(s.mean()),
            "speed_std": float(s.std()),
            "speed_min": float(s.min()),
            "speed_max": float(s.max()),
            "rev_per_2048_window": WINDOW_SIZE / FS_VIB * rpm / 60.0,
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("== Phase A: order tracking 무학습 정렬 진단 (023→1) ==", file=sys.stderr)

    results = []
    for b in NORMAL_BEARINGS:
        r = diagnose_bearing(b, "normal")
        if r:
            results.append(r)
            print(f"[normal {b}] corr_hz={r['corr_hz']:.3f} corr_order={r['corr_order']:.3f} "
                  f"hz_ratio={r['hz_peak_ratio_median']} order_ratio={r['order_peak_ratio_median']}",
                  file=sys.stderr)

    n_fault = 0
    for b in FAULT_CANDIDATES:
        if n_fault >= 2:
            break
        r = diagnose_bearing(b, "fault")
        if r:
            results.append(r)
            n_fault += 1
            print(f"[fault {b}] corr_hz={r['corr_hz']:.3f} corr_order={r['corr_order']:.3f} "
                  f"hz_ratio={r['hz_peak_ratio_median']} order_ratio={r['order_peak_ratio_median']}",
                  file=sys.stderr)

    summary = {
        "config": {
            "src_setting": SRC_SETTING, "tgt_setting": TGT_SETTING,
            "src_rpm": SRC_RPM, "tgt_rpm": TGT_RPM, "speed_ratio": TGT_RPM / SRC_RPM,
            "SPR": SPR, "nperseg_ang": NPERSEG_ANG, "nperseg_hz": NPERSEG_HZ,
            "fs_vib": FS_VIB, "window_size": WINDOW_SIZE,
        },
        "speed_constancy": speed_constancy(),
        "results": results,
    }

    # 정상 bearing 집계(게이트 핵심)
    norm = [r for r in results if r["kind"] == "normal"]
    if norm:
        summary["gate_normal"] = {
            "corr_hz_mean": float(np.mean([r["corr_hz"] for r in norm])),
            "corr_order_mean": float(np.mean([r["corr_order"] for r in norm])),
            "hz_peak_ratio_median": float(np.median(
                [r["hz_peak_ratio_median"] for r in norm if r["hz_peak_ratio_median"]])),
            "order_peak_ratio_median": float(np.median(
                [r["order_peak_ratio_median"] for r in norm if r["order_peak_ratio_median"]])),
        }

    out_path = os.path.join(OUT_DIR, "phaseA_order_alignment.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}", file=sys.stderr)

    # 콘솔 요약
    print("\n=== 게이트 요약 (정상 bearing 평균) ===")
    if "gate_normal" in summary:
        g = summary["gate_normal"]
        print(f"  Hz축 peak 이동비 median = {g['hz_peak_ratio_median']:.3f} (기대 ~0.60 = 900/1500)")
        print(f"  order축 peak 정렬비 median = {g['order_peak_ratio_median']:.3f} (기대 ~1.00)")
        print(f"  전역 상관: corr_hz={g['corr_hz_mean']:.3f} → corr_order={g['corr_order_mean']:.3f} "
              f"(order축 상관이 크게 오르면 정렬)")
    print("\n=== speed 상수성 / rev-per-window ===")
    for s, v in summary["speed_constancy"].items():
        print(f"  {s}: mean={v['speed_mean']:.2f}rpm std={v['speed_std']:.4f} "
              f"rev/2048win={v['rev_per_2048_window']:.3f}")


if __name__ == "__main__":
    main()
