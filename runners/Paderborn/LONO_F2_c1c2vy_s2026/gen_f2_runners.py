#!/usr/bin/env python3
"""작업 F-2 runner 생성기.

작업 D(LONO_D_ampnorm_s2026)의 fold별 train/test .sh를 소스로 삼아(=split·bearing id 인자를
그대로 재사용, 전사 오류 방지) F-2 다채널 변형을 생성한다. D 대비 변경점만:
  - RUN_NAME: ampnorm_<split>_LONO<n>  →  f2_<config>_<split>_LONO<n>
  - 인자 추가: --sensor_mode C1C2Vy  (채널셋 = Vy+C1+C2 = 3노드, dynamic graph 실사용)
  - 인자 추가: --amp_normalize_channels {all|vib_only}
      all   = 전 채널 rmsnorm(shape-only)
      mixed = Vy만 rmsnorm, 전류 raw(진폭 보존)  → --amp_normalize_channels vib_only

사용:
  python gen_f2_runners.py            # gate 기본: LONO 1,2 × 두 구성
  python gen_f2_runners.py 3 4 5 6    # 전체 확장분(나머지 LONO) 생성
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D_DIR = os.path.join(os.path.dirname(HERE), "LONO_D_ampnorm_s2026")
SPLITS = ["123to0", "023to1", "013to2", "012to3"]
CONFIGS = {"all": "all", "mixed": "vib_only"}  # config → amp_normalize_channels

INJECT = ("    --sensor_mode C1C2Vy \\\n"
          "    --amp_normalize_channels {ch} \\\n"
          "    --amp_normalize")


def transform(text, config, split, lono):
    ch = CONFIGS[config]
    src_run = f"ampnorm_{split}_LONO{lono}"
    dst_run = f"f2_{config}_{split}_LONO{lono}"
    text = text.replace(f'RUN_NAME="{src_run}"', f'RUN_NAME="{dst_run}"')
    # 헤더 주석을 F-2용으로 정정(작업 D 템플릿에서 복제된 흔적 제거).
    text = text.replace(
        "# 작업 D: 진폭 confound 교정(Route 2). seed 2026 단일. 파일명 _5seeds는 slurm_run.sh",
        f"# 작업 F-2: C1C2Vy 다채널({config}={ 'shape-only 전채널' if ch=='all' else 'Vy만 rmsnorm·전류 raw' }). "
        "seed 2026. 파일명 _5seeds는 slurm_run.sh")
    # --amp_normalize 앞에 다채널 인자 삽입(train=마지막줄 / test=--rms_lambda 앞 모두 안전)
    assert "    --amp_normalize" in text, "예상한 --amp_normalize 라인이 없습니다."
    text = text.replace("    --amp_normalize", INJECT.format(ch=ch), 1)
    return text


def main():
    lonos = [int(x) for x in sys.argv[1:]] or [1, 2]
    n = 0
    for config in CONFIGS:
        for split in SPLITS:
            for lono in lonos:
                for kind in ("", "_test"):
                    src = os.path.join(D_DIR, f"run_Paderborn_ampnorm_{split}_LONO{lono}{kind}_5seeds.sh")
                    if not os.path.exists(src):
                        raise FileNotFoundError(src)
                    with open(src) as f:
                        text = f.read()
                    out_text = transform(text, config, split, lono)
                    dst = os.path.join(HERE, f"run_Paderborn_f2_{config}_{split}_LONO{lono}{kind}_5seeds.sh")
                    with open(dst, "w") as f:
                        f.write(out_text)
                    n += 1
    print(f"생성 완료: {n}개 .sh (configs={list(CONFIGS)}, splits={SPLITS}, lonos={lonos})")


if __name__ == "__main__":
    main()
