#!/usr/bin/env python3
"""작업 G-1(dual-branch shape/amplitude disentangle) runner 생성기.

작업 D(LONO_D_ampnorm_s2026)의 fold별 train/test .sh를 소스로 삼아(=split·bearing id 인자를
그대로 재사용, 전사 오류 방지) --amp_branch를 주입해 G-1 스크립트를 생성한다. D 대비 변경점만:
  - RUN_NAME: ampnorm_<split>_LONO<n>  →  g1_<split>_LONO<n>
  - train(main.py): --amp_branch 추가 (--amp_normalize 유지 = shape-only window 전제)
  - test(test.py):  작업 D penalty(--rms_lambda 0)를 제거하고 --amp_branch 추가
      (amp_branch면 test.py가 S_shape/S_amp/S_total을 분리 채점하므로 rms_lambda는 무시됨)
  - seedmode: 's2026'(기본, --seeds 2026, 24 fold) 또는 '5seeds'(--seeds 2024..2028)

사용:
  python gen_g1_runners.py            # seed 2026, 24 fold → LONO_G1_s2026/
  python gen_g1_runners.py 5seeds     # 5-seed,  24 fold → LONO_G1_5seeds/
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # runners/Paderborn/LONO_G1_s2026
RUNNERS = os.path.dirname(HERE)                            # runners/Paderborn
D_DIR = os.path.join(RUNNERS, "LONO_D_ampnorm_s2026")
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
LONOS = [1, 2, 3, 4, 5, 6]


def transform(text, kind, split, lono, seeds_line, comment2):
    src_run = f"ampnorm_{split}_LONO{lono}"
    dst_run = f"g1_{split}_LONO{lono}"
    text = text.replace(f'RUN_NAME="{src_run}"', f'RUN_NAME="{dst_run}"')
    text = text.replace(
        "# 작업 D: 진폭 confound 교정(Route 2). seed 2026 단일. 파일명 _5seeds는 slurm_run.sh",
        "# 작업 G-1: dual-branch shape/amplitude disentangle. 파일명 _5seeds는 slurm_run.sh")
    text = text.replace(
        "# 짝 매칭(_5seeds->_test_5seeds) 규약을 위한 레거시 토큰일 뿐, 실제는 seed 2026 하나.",
        "# 짝 매칭(_5seeds->_test_5seeds) 규약을 위한 토큰. " + comment2)
    # seeds 라인 교체(5seeds 모드에서만 실질 변경)
    text = text.replace("    --seeds 2026 \\\n", seeds_line)
    if kind == "":   # train: --amp_normalize 뒤에 --amp_branch 추가
        text = text.replace("    --amp_normalize",
                            "    --amp_normalize \\\n    --amp_branch", 1)
    else:            # test: --rms_lambda 0(작업 D) 제거하고 --amp_branch
        assert "    --rms_lambda 0" in text, "예상한 --rms_lambda 0 라인이 없습니다."
        text = text.replace("    --amp_normalize \\\n    --rms_lambda 0",
                            "    --amp_normalize \\\n    --amp_branch")
    return text


def main():
    seedmode = sys.argv[1] if len(sys.argv) > 1 else 's2026'
    if seedmode == 's2026':
        out_dir = HERE
        seeds_line = "    --seeds 2026 \\\n"
        comment2 = "실제는 seed 2026 단일(sanity/gate)."
    elif seedmode == '5seeds':
        out_dir = os.path.join(RUNNERS, "LONO_G1_5seeds")
        seeds_line = "    --seeds 2024 2025 2026 2027 2028 \\\n"
        comment2 = "실제 5-seed(2024–2028)."
    else:
        raise SystemExit(f"알 수 없는 seedmode: {seedmode} (s2026|5seeds)")
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for split in SPLITS:
        for lono in LONOS:
            for kind in ("", "_test"):
                src = os.path.join(D_DIR, f"run_Paderborn_ampnorm_{split}_LONO{lono}{kind}_5seeds.sh")
                if not os.path.exists(src):
                    raise FileNotFoundError(src)
                with open(src) as f:
                    text = f.read()
                out_text = transform(text, kind, split, lono, seeds_line, comment2)
                dst = os.path.join(out_dir, f"run_Paderborn_g1_{split}_LONO{lono}{kind}_5seeds.sh")
                with open(dst, "w") as f:
                    f.write(out_text)
                n += 1
    print(f"생성 완료: {n}개 .sh (seedmode={seedmode}, splits={SPLITS}, lonos={LONOS}) → {out_dir}")


if __name__ == "__main__":
    main()
