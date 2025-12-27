# -*- coding: utf-8 -*-
import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Dict

import pandas as pd
import requests
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from PIL import Image, ImageChops
from wordcloud import WordCloud

from korean_font import configure_korean_font, korean_font_help_markdown, require_korean_font_file, korean_font_debug_line
from web_fonts import inject_noto_sans_kr


# =============================
# Font (Korean)
# =============================

_font_info = configure_korean_font()
_KOREAN_FONT_NAME: Optional[str] = _font_info.name
_KOREAN_FONT_PATH: Optional[str] = _font_info.regular_path
_KOREAN_FONT_BOLD_PATH: Optional[str] = _font_info.bold_path


def pick_bold_font_path() -> Optional[str]:
    # 배포 환경에서는 repo-local 폰트를 우선 사용
    return _KOREAN_FONT_BOLD_PATH or _KOREAN_FONT_PATH


_CHOSEN_FONT = _KOREAN_FONT_NAME


# =============================
# Google Sheets (XLSX download)
# =============================

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1RV229EkeZsPPjxwBpBrGS7sn-CA105u2q1vBlkVQxc8/edit?usp=sharing"


def _extract_spreadsheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("구글 스프레드시트 URL에서 문서 ID를 추출할 수 없습니다.")
    return m.group(1)


def _xlsx_export_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export?format=xlsx"


def _is_probably_html(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in ctype:
        return True
    head = resp.content[:200].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def download_with_retry(url: str, timeout_s: int = 30, retries: int = 3, backoff_s: float = 0.8) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    last_exc: Optional[Exception] = None
    for i in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
            resp.raise_for_status()
            if _is_probably_html(resp):
                raise ValueError("응답이 HTML입니다(권한/공개 설정 문제 가능).")
            return resp
        except Exception as e:
            last_exc = e
            time.sleep(backoff_s * (2**i))
    raise RuntimeError(f"네트워크 요청 실패: {last_exc}")


@st.cache_data(show_spinner=False)
def fetch_xlsx_as_df(sheet_url: str) -> pd.DataFrame:
    sid = _extract_spreadsheet_id(sheet_url)
    resp = download_with_retry(_xlsx_export_url(sid))
    with BytesIO(resp.content) as bio:
        return pd.read_excel(bio)


# =============================
# Parsing (similar to app2.py parse_procon)
# =============================

def parse_procon(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    year_col = "년도" if "년도" in cols else ("년도도" if "년도도" in cols else cols[0])

    def idx(name: str) -> int:
        if name not in cols:
            raise ValueError(f"긍부정통합: 필수 컬럼이 없습니다: {name}. 현재 컬럼={cols}")
        return cols.index(name)

    # wide triplets: [단어, 건수, 속성?]
    pos_i, neg_i, neu_i = idx("긍정 단어"), idx("부정 단어"), idx("중립 단어")

    def triplet(word_idx: int) -> tuple[str, str, Optional[str]]:
        w = cols[word_idx]
        c = cols[word_idx + 1] if word_idx + 1 < len(cols) else None
        a = cols[word_idx + 2] if word_idx + 2 < len(cols) else None
        if c is None:
            raise ValueError(f"긍부정통합: '{w}' 다음에 건수 컬럼을 찾지 못했습니다.")
        return w, c, a

    pos_w, pos_c, pos_a = triplet(pos_i)
    neg_w, neg_c, neg_a = triplet(neg_i)
    neu_w, neu_c, neu_a = triplet(neu_i)

    def pack(sent: str, w: str, c: str, a: Optional[str]) -> pd.DataFrame:
        use = [year_col, w, c] + ([a] if a and a in df.columns else [])
        tmp = df[use].copy()
        tmp.columns = ["년도", "단어", "건수"] + (["속성"] if len(use) == 4 else [])
        if "속성" not in tmp.columns:
            tmp["속성"] = ""
        tmp["감성"] = sent
        tmp["년도"] = pd.to_numeric(tmp["년도"], errors="coerce")
        tmp["단어"] = tmp["단어"].astype(str).str.strip()
        tmp["건수"] = pd.to_numeric(tmp["건수"], errors="coerce").fillna(0.0)
        tmp["속성"] = tmp["속성"].astype(str).str.strip()
        tmp = tmp[tmp["년도"].notna() & (tmp["단어"] != "") & (tmp["단어"].str.lower() != "nan")].copy()
        tmp["년도"] = tmp["년도"].astype(int)
        return tmp[["년도", "감성", "단어", "건수", "속성"]]

    out = pd.concat(
        [
            pack("긍정", pos_w, pos_c, pos_a),
            pack("부정", neg_w, neg_c, neg_a),
            pack("중립", neu_w, neu_c, neu_a),
        ],
        ignore_index=True,
    )
    return out


# =============================
# WordCloud helpers
# =============================

def _crop_to_content(img: Image.Image, pad: int = 14) -> Image.Image:
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img.convert("RGB"), bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    left = max(0, bbox[0] - pad)
    top = max(0, bbox[1] - pad)
    right = min(img.size[0], bbox[2] + pad)
    bottom = min(img.size[1], bbox[3] + pad)
    return img.crop((left, top, right, bottom))


def build_freq_dict(df: pd.DataFrame, include_count_in_label: bool, top_n: int) -> Dict[str, float]:
    tmp = df.sort_values("건수", ascending=False).head(int(top_n)).copy()
    freq: Dict[str, float] = {}
    for _, r in tmp.iterrows():
        w = str(r["단어"]).strip()
        c = float(r["건수"])
        if not w or w.lower() == "nan" or c <= 0:
            continue
        label = f"{w}({int(round(c))})" if include_count_in_label else w
        # 동일 label이 있을 수 있어 누적
        freq[label] = freq.get(label, 0.0) + c
    return freq


def render_wordcloud(freq: Dict[str, float], *, font_path: Optional[str], color_hex: str, width: int, height: int, max_words: int, scale: int) -> Image.Image:
    if not freq:
        return Image.new("RGB", (width, height), "white")

    def color_func(*_args, **_kwargs):
        return color_hex

    wc = WordCloud(
        font_path=font_path,
        width=int(width),
        height=int(height),
        background_color="white",
        max_words=int(max_words),
        prefer_horizontal=0.95,
        random_state=42,
        collocations=False,
        margin=2,
        scale=int(scale),
        color_func=color_func,
    ).generate_from_frequencies(freq)

    img = wc.to_image()
    return _crop_to_content(img, pad=14)


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="썸트렌드 긍부정 워드클라우드", layout="wide")
inject_noto_sans_kr()
st.title("☁️ 썸트렌드 긍부정 워드클라우드")

with st.sidebar:
    st.header("데이터 소스")
    sheet_url = st.text_input("구글시트 URL", value=DEFAULT_SHEET_URL)
    st.caption("공개/공유 설정이 '링크가 있는 모든 사용자 보기 가능'이어야 합니다.")
    if _KOREAN_FONT_PATH:
        st.caption(f"한글 폰트: {_CHOSEN_FONT} (파일 사용)")
        st.caption(korean_font_debug_line(_font_info))
    else:
        st.warning("한글 폰트 파일을 찾지 못했습니다. 배포 환경에서는 워드클라우드 한글이 깨질 수 있습니다.")
        st.caption(korean_font_help_markdown())
    st.divider()

    st.header("워드클라우드 설정")
    top_n = st.slider("단어 수(Top N)", 10, 200, 80, 5)
    include_count = st.checkbox("단어 옆에 건수 표시(예: 단어(23))", value=True)
    # 300% 정도 크게 보이도록 기본 캔버스 크기를 3배로 상향(배포/로컬 모두 적용)
    wc_width = st.slider("가로(width)", 600, 6000, 4200, 100)
    wc_height = st.slider("세로(height)", 400, 4000, 2100, 100)
    wc_scale = st.slider("선명도(scale)", 1, 5, 2, 1)
    wc_max_words = st.slider("max_words", 20, 400, 200, 10)

    if st.button("데이터 새로고침(캐시 삭제)"):
        st.cache_data.clear()


@st.cache_data(show_spinner=True)
def load_procon(url: str) -> pd.DataFrame:
    raw = fetch_xlsx_as_df(url)
    return parse_procon(raw)


try:
    df = load_procon(sheet_url)
except Exception as e:
    st.error(f"구글시트 데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.caption(f"업데이트됨: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

years = sorted(df["년도"].unique().tolist())
year_options = ["전체"] + [str(y) for y in years]
year_sel = st.selectbox("년도 선택", year_options, index=len(year_options) - 1, key="procon_year_sel")

if year_sel == "전체":
    df_y = df.copy()
    title_year = "전체"
else:
    year_int = int(year_sel)
    df_y = df[df["년도"] == year_int].copy()
    title_year = f"{year_int}년"

font_path = require_korean_font_file(_font_info)
if not font_path:
    st.error("한글 폰트 파일이 없어 워드클라우드를 생성할 수 없습니다.")
    st.markdown(korean_font_help_markdown())
    st.stop()

# 감성별 색상(가독성 향상: 더 진한 색상 사용)
colors = {"긍정": "#15803D", "부정": "#B91C1C", "중립": "#374151"}
sent_order = ["긍정", "부정", "중립"]

for sent in sent_order:
    dff = df_y[df_y["감성"] == sent].copy()
    freq = build_freq_dict(dff, include_count_in_label=include_count, top_n=int(top_n))
    img = render_wordcloud(
        freq,
        font_path=font_path,
        color_hex=colors[sent],
        width=int(wc_width),
        height=int(wc_height),
        max_words=int(wc_max_words),
        scale=int(wc_scale),
    )
    st.subheader(f"{title_year} {sent} ({len(freq)}개)")
    # 세로 배치 + 크게 보이도록 컨테이너 폭에 맞춰 표시
    st.image(img, width="stretch")

with st.expander("데이터 보기(필터 적용 후)"):
    st.dataframe(df_y.reset_index(drop=True), width="stretch", height=420)


