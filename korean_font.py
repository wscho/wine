from __future__ import annotations

# -*- coding: utf-8 -*-
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


def korean_font_help_markdown() -> str:
    """
    배포(특히 Streamlit Community Cloud)에서 한글이 □□□로 깨질 때 안내 메시지.
    """
    return (
        "배포 환경(리눅스)에는 기본 한글 폰트가 없어서, `matplotlib`/`wordcloud` 이미지에서 한글이 □□□로 깨질 수 있습니다.\n\n"
        "- **해결(권장)**: 레포에 한글 TTF 폰트를 포함하세요.\n"
        "  - `assets/fonts/NotoSansKR-Regular.ttf`\n"
        "  - `assets/fonts/NotoSansKR-Bold.ttf` (선택)\n"
        "  - 또는 `assets/fonts/NanumGothic.ttf` / `assets/fonts/NanumGothicBold.ttf`\n\n"
        "- **자동 다운로드**: `python scripts/download_fonts.py`\n\n"
        "- **참고**: Streamlit Community Cloud 배포: `https://share.streamlit.io/`\n"
    )


def require_korean_font_file(info: KoreanFontInfo) -> str:
    """
    WordCloud/Matplotlib 이미지 생성에 사용할 '실제 폰트 파일 경로'가 반드시 필요할 때 사용.
    """
    return info.bold_path or info.regular_path or ""


def korean_font_debug_line(info: KoreanFontInfo) -> str:
    """
    화면/로그에 출력할 폰트 디버그 문자열.
    """
    name = info.name or "None"
    reg = info.regular_path or "None"
    bold = info.bold_path or "None"
    return f"한글 폰트 디버그: name={name} | regular={reg} | bold={bold}"


