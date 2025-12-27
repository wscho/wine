# -*- coding: utf-8 -*-
import os
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests
import streamlit as st

import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from korean_font import configure_korean_font
from web_fonts import inject_noto_sans_kr


# =============================
# Font (Korean)
# =============================
_CHOSEN_FONT = configure_korean_font().name


# =============================
# Google Sheets (XLSX download)
# =============================

# 네이버언급량 데이터셋
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CHRYjfLEfHOa_6ugtTkR3o3rmPd2mmjDLm71b5TAQb4/edit?usp=drive_link"

# 네이버 DataLab 비교(국내/해외/국가별 와인) 데이터셋
# - 7행: 필드명, 8행부터 데이터
DEFAULT_COMP_SHEET_URL = "https://docs.google.com/spreadsheets/d/1amDCFWC95S2dVImacl-41Uq9XYQr_fyD/edit?usp=sharing&ouid=112643056517438341912&rtpof=true&sd=true"


def _extract_spreadsheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if not m:
        raise ValueError("구글 스프레드시트 URL에서 문서 ID를 추출할 수 없습니다.")
    return m.group(1)


def _xlsx_export_url(spreadsheet_id: str) -> str:
    # gid 없이 문서 전체를 xlsx로 export (권한/리다이렉트 이슈 완화)
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


@st.cache_data(show_spinner=False)
def fetch_xlsx_as_df_with_header(sheet_url: str, header_row_1based: int) -> pd.DataFrame:
    """
    header_row_1based: 1부터 시작하는 헤더 행 번호 (예: 7행이 헤더면 7)
    """
    sid = _extract_spreadsheet_id(sheet_url)
    resp = download_with_retry(_xlsx_export_url(sid))
    header0 = max(int(header_row_1based) - 1, 0)
    with BytesIO(resp.content) as bio:
        return pd.read_excel(bio, header=header0)


# =============================
# Data prep
# =============================

def _make_unique_columns(cols: List[str]) -> List[str]:
    seen = {}
    out = []
    for c in cols:
        name = str(c).strip()
        if name == "" or name.lower() == "nan":
            name = "컬럼"
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}__{seen[name]}")
    return out


def normalize_naver_datalab(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        raise ValueError("시트 데이터가 비어 있습니다.")

    df = df_raw.copy()
    df.columns = _make_unique_columns(list(df.columns))

    # 날짜 컬럼 찾기
    date_col = "날짜" if "날짜" in df.columns else df.columns[0]
    df = df.dropna(subset=[date_col]).copy()

    s = df[date_col].astype(str).str.strip()
    # 일반적으로 'YYYY-MM-DD' / 'YYYY.MM.DD' 혼재 가능
    # pandas 최신 버전에서 infer_datetime_format은 deprecated (기본이 strict parsing)
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", format="%Y.%m.%d")
    df = df[dt.notna()].copy()
    df["날짜"] = dt[dt.notna()].dt.date.astype(str)  # YYYY-MM-DD

    # 숫자 변환(날짜 제외)
    for c in df.columns:
        if c == date_col or c == "날짜":
            continue
        # 중복 컬럼이면 df[c]가 DataFrame이 될 수 있어 Series로 고정
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 정리: 날짜를 맨 앞으로
    if date_col != "날짜":
        df = df.drop(columns=[date_col])
    keep = ["날짜"] + [c for c in df.columns if c != "날짜"]
    df = df.loc[:, keep].copy()
    df = df.sort_values("날짜")
    return df


def normalize_naver_datalab_comp(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    비교 시트 구조(예):
    날짜 | 국내와인 | 날짜 | 외국와인 | 날짜 | 프랑스와인 | 날짜 | 이태리와인 | 날짜 | 칠레와인
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("시트 데이터가 비어 있습니다.")

    df = df_raw.copy()
    # pandas가 중복 컬럼을 '날짜.1'처럼 만들 수 있어 '.숫자' 제거 후 unique화
    df.columns = _make_unique_columns([re.sub(r"\.\d+$", "", str(c)).strip() for c in df.columns])

    expected_terms = ["국내와인", "외국와인", "프랑스와인", "이태리와인", "칠레와인"]

    # 컬럼명이 깨져 들어오는 환경도 있어(콘솔/로케일 이슈), 우선 "의도된 위치" 기반으로 복구 시도
    # 형태: [날짜, v1, 날짜, v2, 날짜, v3, 날짜, v4, 날짜, v5]
    if not all(c in df.columns for c in expected_terms):
        if df.shape[1] >= 10:
            positional = [df.columns[1], df.columns[3], df.columns[5], df.columns[7], df.columns[9]]
            tmp = df.loc[:, [df.columns[0]] + positional].copy()
            tmp.columns = ["날짜"] + expected_terms
            df = tmp
        else:
            missing = [c for c in expected_terms if c not in df.columns]
            raise ValueError(f"필수 컬럼을 찾지 못했습니다: {missing} (현재 컬럼 수: {df.shape[1]})")

    date_col = "날짜" if "날짜" in df.columns else df.columns[0]
    df = df.dropna(subset=[date_col]).copy()

    s = df[date_col].astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", format="%Y.%m.%d")
    df = df[dt.notna()].copy()
    df["날짜"] = dt[dt.notna()].dt.date.astype(str)  # YYYY-MM-DD

    # 숫자 변환(대상 5개만)
    for c in expected_terms:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    keep = ["날짜"] + expected_terms
    df = df.loc[:, keep].copy()
    df = df.sort_values("날짜")
    return df


def aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    freq: 'D' | 'M' | 'Y'
    """
    out = df.copy()
    out["날짜_dt"] = pd.to_datetime(out["날짜"], errors="coerce")
    out = out[out["날짜_dt"].notna()].copy()

    numeric_cols = [c for c in out.columns if c not in {"날짜", "날짜_dt"}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if freq == "D":
        out["기간"] = out["날짜_dt"].dt.strftime("%Y-%m-%d")
    elif freq == "M":
        out["기간"] = out["날짜_dt"].dt.to_period("M").astype(str)  # YYYY-MM
    elif freq == "Y":
        out["기간"] = out["날짜_dt"].dt.year.astype(int).astype(str)
    else:
        raise ValueError("freq는 'D','M','Y' 중 하나여야 합니다.")

    g = out.groupby("기간", as_index=False)[numeric_cols].sum()
    cols = ["기간"] + numeric_cols
    return g.loc[:, cols].copy()


def plot_trend(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str) -> go.Figure:
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[c],
                mode="lines+markers",
                name=c,
                line=dict(width=3),
                marker=dict(size=5),
            )
        )
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=30, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family=_CHOSEN_FONT or "Malgun Gothic", size=14),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig


# =============================
# Streamlit UI
# =============================

st.set_page_config(page_title="영동와인 빅데이터(트렌드)_NaverDatalab", layout="wide")
inject_noto_sans_kr()
st.title("📈 Naver DataLab 트렌드 분석")

with st.sidebar:
    st.header("메뉴")
    menu = st.radio("보기", ["naver datalab trend", "naver datalab comp"], index=0)
    st.divider()

    st.header("데이터 소스")
    if menu == "naver datalab trend":
        sheet_url = st.text_input("구글시트 URL", value=DEFAULT_SHEET_URL)
    else:
        sheet_url = st.text_input("구글시트 URL", value=DEFAULT_COMP_SHEET_URL)
    st.caption("공개/공유 설정이 '링크가 있는 모든 사용자 보기 가능'이어야 합니다.")
    st.divider()

    st.header("시각화 설정")
    default_metric = "합계"
    st.caption(f"한글 폰트: {_CHOSEN_FONT or '감지 실패(깨짐 시 맑은 고딕/나눔고딕 설치 필요)'}")


@st.cache_data(show_spinner=True)
def load_and_prepare(url: str) -> pd.DataFrame:
    raw = fetch_xlsx_as_df(url)
    return normalize_naver_datalab(raw)


@st.cache_data(show_spinner=True)
def load_and_prepare_comp(url: str) -> pd.DataFrame:
    # 7행: 필드명, 8행부터 데이터
    raw = fetch_xlsx_as_df_with_header(url, header_row_1based=7)
    return normalize_naver_datalab_comp(raw)


try:
    if menu == "naver datalab trend":
        df = load_and_prepare(sheet_url)
    else:
        df = load_and_prepare_comp(sheet_url)
except Exception as e:
    st.error(f"구글시트 데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.caption(f"업데이트됨: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

numeric_candidates = [c for c in df.columns if c != "날짜"]
if not numeric_candidates:
    st.error("숫자 컬럼을 찾지 못했습니다. (날짜 외에 분석할 컬럼이 없습니다)")
    st.stop()

if menu == "naver datalab trend":
    with st.sidebar:
        y_cols = st.multiselect(
            "표시할 지표(복수 선택)",
            options=numeric_candidates,
            default=[default_metric] if default_metric in numeric_candidates else [numeric_candidates[-1]],
        )

    if not y_cols:
        st.warning("표시할 지표를 1개 이상 선택하세요.")
        st.stop()

    tab_d, tab_m, tab_y = st.tabs(["일별", "월별", "년도별"])

    with tab_d:
        st.subheader("일별 언급량 트렌드")
        df_d = aggregate(df, "D")
        fig = plot_trend(df_d, "기간", y_cols, "일별 트렌드")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("데이터 보기"):
            st.dataframe(df_d, use_container_width=True, height=360)

    with tab_m:
        st.subheader("월별 언급량 트렌드")
        df_m = aggregate(df, "M")
        fig = plot_trend(df_m, "기간", y_cols, "월별 트렌드")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("데이터 보기"):
            st.dataframe(df_m, use_container_width=True, height=360)

    with tab_y:
        st.subheader("년도별 언급량 트렌드")
        df_y = aggregate(df, "Y")
        fig = plot_trend(df_y, "기간", y_cols, "년도별 트렌드")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("데이터 보기"):
            st.dataframe(df_y, use_container_width=True, height=360)

else:
    default_terms = ["국내와인", "외국와인", "프랑스와인", "이태리와인", "칠레와인"]
    with st.sidebar:
        y_cols = st.multiselect(
            "표시할 키워드(복수 선택)",
            options=numeric_candidates,
            default=[c for c in default_terms if c in numeric_candidates] or numeric_candidates,
        )

    if not y_cols:
        st.warning("표시할 키워드를 1개 이상 선택하세요.")
        st.stop()

    tab_m, tab_y = st.tabs(["월별", "년도별"])

    with tab_m:
        st.subheader("월별 언급량 트렌드 (국내/외국/국가별)")
        df_m = aggregate(df, "M")
        fig = plot_trend(df_m, "기간", y_cols, "월별 트렌드 (비교)")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("데이터 보기"):
            st.dataframe(df_m, use_container_width=True, height=360)

    with tab_y:
        st.subheader("년도별 언급량 트렌드 (국내/외국/국가별)")
        df_y = aggregate(df, "Y")
        fig = plot_trend(df_y, "기간", y_cols, "년도별 트렌드 (비교)")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("데이터 보기"):
            st.dataframe(df_y, use_container_width=True, height=360)


