from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class KoreanFontInfo:
    name: Optional[str]
    regular_path: Optional[str]
    bold_path: Optional[str]
    prop: Optional[fm.FontProperties]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _candidate_font_dirs() -> list[Path]:
    root = _repo_root()
    return [
        root / "assets" / "fonts",
        root / "fonts",
    ]


def _pick_local_font(*filenames: str) -> Optional[str]:
    for d in _candidate_font_dirs():
        for fn in filenames:
            p = d / fn
            if p.exists() and p.is_file():
                return str(p)
    return None


def configure_korean_font() -> KoreanFontInfo:
    """
    배포(리눅스)에서도 한글이 깨지지 않도록:
    1) 레포 내부 폰트(assets/fonts 또는 fonts)를 최우선 사용
    2) 그 다음 OS 기본 폰트(윈도우/일부 환경) 탐색
    3) 마지막으로 설치된 폰트 패밀리 이름으로 설정 시도

    폰트 파일을 레포에 추가하는 방법(추천):
    - assets/fonts/NotoSansKR-Regular.ttf
    - assets/fonts/NotoSansKR-Bold.ttf (선택)
    또는
    - assets/fonts/NanumGothic.ttf
    - assets/fonts/NanumGothicBold.ttf (선택)
    """

    # 1) repo-local fonts (recommended for Streamlit Cloud)
    local_regular = _pick_local_font(
        "NotoSansKR-Regular.ttf",
        "NanumGothic.ttf",
        "malgun.ttf",  # 혹시 포함한 경우
        "NotoSansCJKkr-Regular.otf",
    )
    local_bold = _pick_local_font(
        "NotoSansKR-Bold.ttf",
        "NanumGothicBold.ttf",
        "malgunbd.ttf",
    )

    if local_regular:
        try:
            fm.fontManager.addfont(local_regular)
            prop = fm.FontProperties(fname=local_regular)
            name = prop.get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return KoreanFontInfo(name=name, regular_path=local_regular, bold_path=local_bold or local_regular, prop=prop)
        except Exception:
            pass

    # 2) Windows common font paths
    win_regular_candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\NotoSansCJKkr-Regular.otf",
    ]
    win_bold_candidates = [
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\NanumGothicBold.ttf",
    ]
    for p in win_regular_candidates:
        try:
            if os.path.exists(p):
                fm.fontManager.addfont(p)
                prop = fm.FontProperties(fname=p)
                name = prop.get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                bold = next((b for b in win_bold_candidates if os.path.exists(b)), None)
                return KoreanFontInfo(name=name, regular_path=p, bold_path=bold or p, prop=prop)
        except Exception:
            pass

    # 3) font family name fallback (depends on environment)
    candidates = ["Noto Sans KR", "Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            prop = fm.FontProperties(family=name)
            return KoreanFontInfo(name=name, regular_path=None, bold_path=None, prop=prop)

    return KoreanFontInfo(name=None, regular_path=None, bold_path=None, prop=None)


