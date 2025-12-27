# -*- coding: utf-8 -*-
import pathlib
import re
import time
from io import BytesIO
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests

from web_fonts import inject_noto_sans_kr


st.set_page_config(page_title="영동와인 빅데이터 홈", layout="wide")
inject_noto_sans_kr()

st.title("🍷 영동와인 빅데이터 홈페이지")
st.caption("좌측 사이드바에서 원하는 분석 페이지를 선택하세요.")

with st.sidebar:
    st.header("메뉴")
    menu = st.radio("이동", ["홈", "관련뉴스"], index=0, label_visibility="collapsed")

    # 관련뉴스 설정(뉴스 메뉴에서 사용)
    st.divider()
    st.header("관련뉴스")
    news_sheet_url = st.text_input("관련뉴스 시트 URL", value="https://docs.google.com/spreadsheets/d/1JsksLQuGqXuL7RGacqZyEmHxCrTIMHOVwlAIM32HUAo/edit?usp=sharing")
    news_query = st.text_input("뉴스 검색(제목)", value="")
    max_items = st.slider("표시 개수", 5, 100, 20, 5)
    if st.button("관련뉴스 새로고침(캐시 삭제)", width="stretch"):
        st.cache_data.clear()

if menu == "홈":
    intro_path = pathlib.Path(__file__).with_name("intro.html")
    if intro_path.exists():
        html = intro_path.read_text(encoding="utf-8")
        components.html(html, height=950, scrolling=True)
    else:
        st.warning("intro.html 파일을 찾지 못했습니다.")

    st.divider()

    st.subheader("바로가기")
    st.write("아래 버튼으로 각 분석 페이지로 이동할 수 있습니다. (Streamlit 버전에 따라 버튼 이동이 제한될 수 있습니다)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📈 네이버 DataLab 언급량 트렌드", width="stretch"):
            try:
                st.switch_page("pages/01_naver_datalab_trend.py")
            except Exception:
                st.info("좌측 사이드바에서 '네이버 DataLab 언급량 트렌드' 페이지를 선택해 주세요.")
        if st.button("📊 네이버 DataLab 비교 트렌드(국내/해외/국가별)", width="stretch"):
            try:
                st.switch_page("pages/01_naver_datalab_comp.py")
            except Exception:
                st.info("좌측 사이드바에서 '네이버 DataLab 비교 트렌드' 페이지를 선택해 주세요.")
        if st.button("📈 썸트렌드 언급량 트렌드(빈도수)", width="stretch"):
            try:
                st.switch_page("pages/02_sometrend_freq_trend.py")
            except Exception:
                st.info("좌측 사이드바에서 '썸트렌드 언급량 트렌드(빈도수)' 페이지를 선택해 주세요.")

    with col2:
        if st.button("🕸️ 썸트렌드 연관성 분석", width="stretch"):
            try:
                st.switch_page("pages/03_sometrend_association.py")
            except Exception:
                st.info("좌측 사이드바에서 '썸트렌드 연관성 분석' 페이지를 선택해 주세요.")
        if st.button("☁️ 썸트렌드 긍부정 워드클라우드", width="stretch"):
            try:
                st.switch_page("pages/04_sometrend_sentiment_wordcloud.py")
            except Exception:
                st.info("좌측 사이드바에서 '썸트렌드 긍부정 워드클라우드' 페이지를 선택해 주세요.")

    st.stop()
if menu != "관련뉴스":
    st.stop()


# =============================
# 관련뉴스(구글시트 → 링크 목록)
# =============================

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


def _download_with_retry(url: str, timeout_s: int = 30, retries: int = 3, backoff_s: float = 0.8) -> requests.Response:
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
def _fetch_news_sheet(sheet_url: str) -> pd.DataFrame:
    sid = _extract_spreadsheet_id(sheet_url)
    resp = _download_with_retry(_xlsx_export_url(sid))
    with BytesIO(resp.content) as bio:
        return pd.read_excel(bio)


def _normalize_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    기대 형태:
    - A열: 기사 제목
    - B열: URL
    (헤더가 없을 수도 있어, 컬럼명과 무관하게 0/1열을 우선 사용)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["제목", "URL"])

    cols = list(df.columns)
    a = cols[0]
    b = cols[1] if len(cols) > 1 else cols[0]
    out = df[[a, b]].copy()
    out.columns = ["제목", "URL"]

    out["제목"] = out["제목"].astype(str).str.strip()
    out["URL"] = out["URL"].astype(str).str.strip()
    out = out[(out["제목"] != "") & (out["제목"].str.lower() != "nan")].copy()
    out = out[out["URL"].str.startswith(("http://", "https://"))].copy()
    out = out.drop_duplicates(subset=["URL"]).reset_index(drop=True)
    return out


st.subheader("📰 관련뉴스")

try:
    news_raw = _fetch_news_sheet(news_sheet_url)
    news_df = _normalize_news(news_raw)
except Exception as e:
    st.error(f"관련뉴스 시트를 불러오지 못했습니다: {e}")
    st.stop()

if news_query.strip():
    q = news_query.strip()
    news_df = news_df[news_df["제목"].str.contains(q, case=False, na=False)].copy()

news_df = news_df.head(int(max_items)).copy()

if len(news_df) == 0:
    st.info("표시할 뉴스가 없습니다. (검색어/데이터를 확인하세요)")
else:
    st.caption(f"뉴스 {len(news_df)}건 · 소스: {news_sheet_url}")
    for i, r in news_df.iterrows():
        title = r["제목"]
        url = r["URL"]
        st.markdown(f"- [{title}]({url})")


