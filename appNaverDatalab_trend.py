import os
import re
import time
from datetime import datetime
from io import BytesIO
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

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1CHRYjfLEfHOa_6ugtTkR3o3rmPd2mmjDLm71b5TAQb4/edit?usp=drive_link"


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

    date_col = "날짜" if "날짜" in df.columns else df.columns[0]
    df = df.dropna(subset=[date_col]).copy()

    s = df[date_col].astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(s, errors="coerce", format="%Y.%m.%d")
    df = df[dt.notna()].copy()
    df["날짜"] = dt[dt.notna()].dt.date.astype(str)  # YYYY-MM-DD

    for c in df.columns:
        if c == date_col or c == "날짜":
            continue
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if date_col != "날짜":
        df = df.drop(columns=[date_col])
    keep = ["날짜"] + [c for c in df.columns if c != "날짜"]
    df = df.loc[:, keep].copy()
    df = df.sort_values("날짜")
    return df


def aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = df.copy()
    out["날짜_dt"] = pd.to_datetime(out["날짜"], errors="coerce")
    out = out[out["날짜_dt"].notna()].copy()

    numeric_cols = [c for c in out.columns if c not in {"날짜", "날짜_dt"}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    if freq == "D":
        out["기간"] = out["날짜_dt"].dt.strftime("%Y-%m-%d")
    elif freq == "M":
        out["기간"] = out["날짜_dt"].dt.to_period("M").astype(str)
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

st.set_page_config(page_title="Naver DataLab 트렌드", layout="wide")
inject_noto_sans_kr()
st.title("📈 Naver DataLab 트렌드 분석")

with st.sidebar:
    st.header("데이터 소스")
    sheet_url = st.text_input("구글시트 URL", value=DEFAULT_SHEET_URL)
    st.caption("공개/공유 설정이 '링크가 있는 모든 사용자 보기 가능'이어야 합니다.")
    st.divider()

    st.header("시각화 설정")
    default_metric = "합계"
    st.caption(f"한글 폰트: {_CHOSEN_FONT or '감지 실패(깨짐 시 맑은 고딕/나눔고딕 설치 필요)'}")


@st.cache_data(show_spinner=True)
def load_and_prepare(url: str) -> pd.DataFrame:
    raw = fetch_xlsx_as_df(url)
    return normalize_naver_datalab(raw)


try:
    df = load_and_prepare(sheet_url)
except Exception as e:
    st.error(f"구글시트 데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.caption(f"업데이트됨: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

numeric_candidates = [c for c in df.columns if c != "날짜"]
if not numeric_candidates:
    st.error("숫자 컬럼을 찾지 못했습니다. (날짜 외에 분석할 컬럼이 없습니다)")
    st.stop()

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


