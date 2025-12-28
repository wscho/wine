# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import time
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
import streamlit as st

from web_fonts import inject_noto_sans_kr
from st_compat import dataframe_full


DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/17hasc7WeidkBTDs6a1xyqPZ48S3F8l83/edit?usp=sharing&ouid=112643056517438341912&rtpof=true&sd=true"


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
    head = resp.content[:300].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<title" in head


def download_with_retry(url: str, timeout_s: int = 30, retries: int = 3, backoff_s: float = 0.8) -> requests.Response:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    last_exc: Optional[Exception] = None
    for i in range(int(retries)):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s, allow_redirects=True)
            resp.raise_for_status()
            if _is_probably_html(resp):
                raise ValueError("응답이 HTML입니다(권한/공개 설정/브라우저 차단 페이지 가능).")
            return resp
        except Exception as e:
            last_exc = e
            time.sleep(backoff_s * (2**i))
    raise RuntimeError(f"네트워크 요청 실패: {last_exc}")


@st.cache_data(show_spinner=False)
def fetch_sheet_df(sheet_url: str) -> pd.DataFrame:
    sid = _extract_spreadsheet_id(sheet_url)
    resp = download_with_retry(_xlsx_export_url(sid))
    with BytesIO(resp.content) as bio:
        return pd.read_excel(bio)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["제조국", "와인명", "와인종류", "가격", "별점", "리뷰텍스트"])

    # 기대 컬럼(시트 기준)
    required = ["와인명", "와인종류", "가격", "별점", "리뷰텍스트", "제조국"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}. 현재 컬럼={list(df.columns)}")

    out = df.copy()
    out["제조국"] = out["제조국"].astype(str).str.strip()
    out["와인명"] = out["와인명"].astype(str).str.strip()
    out["와인종류"] = out["와인종류"].astype(str).str.strip()
    out["리뷰텍스트"] = out["리뷰텍스트"].astype(str).fillna("").str.strip()

    out["가격"] = pd.to_numeric(out["가격"], errors="coerce")
    out["별점"] = pd.to_numeric(out["별점"], errors="coerce")

    out = out[(out["제조국"] != "") & (out["제조국"].str.lower() != "nan")].copy()
    out = out[(out["와인명"] != "") & (out["와인명"].str.lower() != "nan")].copy()
    out = out[(out["와인종류"] != "") & (out["와인종류"].str.lower() != "nan")].copy()

    keep = ["제조국", "와인명", "와인종류", "가격", "별점", "리뷰텍스트"]
    return out.loc[:, keep].reset_index(drop=True)


def _fmt_price(x: float) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{int(round(float(x))):,}원"
    except Exception:
        return "-"


st.set_page_config(page_title="국가별 와인 리뷰 분석", layout="wide")
inject_noto_sans_kr()

st.title("🌍 국가별 와인 리뷰 분석")
st.caption("제조국/와인종류별로 리뷰를 조회하고 가격 평균과 건수를 요약합니다.")

with st.sidebar:
    st.header("데이터 소스")
    sheet_url = st.text_input("구글시트 URL", value=DEFAULT_SHEET_URL)
    st.caption("공개/공유 설정이 '링크가 있는 모든 사용자 보기 가능'이어야 합니다.")
    if st.button("데이터 새로고침(캐시 삭제)", use_container_width=True):
        st.cache_data.clear()

try:
    raw = fetch_sheet_df(sheet_url)
    df = _normalize(raw)
except Exception as e:
    st.error(f"시트를 불러오지 못했습니다: {e}")
    st.stop()

countries = sorted(df["제조국"].dropna().unique().tolist())
wine_types = sorted(df["와인종류"].dropna().unique().tolist())

tab_list, tab_summary = st.tabs(["리뷰 목록", "요약(평균가격/건수)"])

with tab_list:
    st.subheader("제조국별 리뷰 목록")
    if not countries:
        st.info("제조국 데이터가 없습니다.")
        st.stop()

    colA, colB, colC = st.columns([1.2, 1.2, 1.6])
    with colA:
        country_sel = st.selectbox("제조국 선택", countries, index=0)
    with colB:
        type_sel_list = st.selectbox("와인종류(선택)", ["전체"] + wine_types, index=0)
    with colC:
        q = st.text_input("검색(와인명/리뷰텍스트)", value="")

    view = df[df["제조국"] == country_sel].copy()
    if type_sel_list != "전체":
        view = view[view["와인종류"] == type_sel_list].copy()
    if q.strip():
        qq = q.strip()
        view = view[
            view["와인명"].str.contains(qq, case=False, na=False)
            | view["리뷰텍스트"].str.contains(qq, case=False, na=False)
        ].copy()

    st.caption(f"필터 결과: {len(view):,}건")

    show = view[["와인명", "와인종류", "가격", "별점", "리뷰텍스트"]].copy()
    show["가격"] = show["가격"].apply(_fmt_price)
    dataframe_full(show, height=520)

    # ===== 가격 요약(테이블 아래) =====
    prices = view["가격"].dropna()
    if len(prices) == 0:
        st.info("가격 데이터가 없어 평균/최고/최저를 계산할 수 없습니다.")
    else:
        avg_p = float(prices.mean())
        max_p = float(prices.max())
        min_p = float(prices.min())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("가격 평균", _fmt_price(avg_p))
        with c2:
            st.metric("최고가", _fmt_price(max_p))
        with c3:
            st.metric("최저가", _fmt_price(min_p))

with tab_summary:
    st.subheader("제조국 + 와인종류(또는 전체) 요약")

    col1, col2 = st.columns(2)
    with col1:
        country_sum = st.selectbox("제조국", ["전체"] + countries, index=0, key="sum_country")
    with col2:
        type_sum = st.selectbox("와인종류", ["전체"] + wine_types, index=0, key="sum_type")

    out = df.copy()
    if country_sum != "전체":
        out = out[out["제조국"] == country_sum].copy()
    if type_sum != "전체":
        out = out[out["와인종류"] == type_sum].copy()

    n = int(len(out))
    avg_price = float(out["가격"].dropna().mean()) if n else float("nan")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("행 개수", f"{n:,}건")
    with m2:
        st.metric("가격 평균", _fmt_price(avg_price))

    with st.expander("요약 데이터 보기"):
        view2 = out[["제조국", "와인명", "와인종류", "가격", "별점", "리뷰텍스트"]].copy()
        dataframe_full(view2, height=420)


