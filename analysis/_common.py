import os
import re

# analysis/ 아래 스크립트들이 공유하는 MTGFLOW 프로젝트 루트(analysis/의 부모 디렉토리)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset/paderborn.py의 loader_Paderborn_OCC() 내부 extract_bearing_id()와 동일한 규칙
BEARING_ID_RE = re.compile(r"(K[A-Z]?\d{2,3})(?:_|\.)")


def extract_bearing_id(filename):
    match = BEARING_ID_RE.search(filename)
    return match.group(1) if match else filename.replace(".mat", "")
