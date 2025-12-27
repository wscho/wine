import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import networkx as nx

from korean_font import configure_korean_font
from web_fonts import inject_noto_sans_kr


# =============================
# Font (Korean)
# =============================

_KOREAN_FONT_PROP: Optional[fm.FontProperties] = None

_font_info = configure_korean_font()
_CHOSEN_FONT = _font_info.name
_KOREAN_FONT_PROP = _font_info.prop


# =============================
# Google Sheets (XLSX download)
# =============================

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1ZtqETyVVwcK5RJ-XyNxQeMY8L2MUqew-eRhA6Ik-OnI/edit?usp=sharing"


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
# Parsing
# =============================

def parse_asso(df: pd.DataFrame) -> pd.DataFrame:
    required = {"연관어", "건수", "카테고리 대분류", "년도"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"연관어통합: 필수 컬럼이 없습니다: {missing}. 현재 컬럼={list(df.columns)}")

    out = df.dropna(how="all").copy()
    out["년도"] = pd.to_numeric(out["년도"], errors="coerce")
    out = out[out["년도"].notna()].copy()
    out["년도"] = out["년도"].astype(int)
    out["건수"] = pd.to_numeric(out["건수"], errors="coerce").fillna(0.0)
    out["연관어"] = out["연관어"].astype(str).str.strip()
    out["카테고리 대분류"] = out["카테고리 대분류"].astype(str).str.strip()
    out = out[(out["연관어"] != "") & (out["연관어"].str.lower() != "nan")].copy()
    return out


def summarize_all_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    '전체' 선택 시:
    - 연관어별 건수 합계
    - 카테고리 대분류는 연관어-카테고리 조합 중 건수 합이 가장 큰 카테고리를 대표값으로 선택
    """
    tmp = df.groupby(["연관어", "카테고리 대분류"], as_index=False)["건수"].sum()
    tmp = tmp.sort_values(["연관어", "건수"], ascending=[True, False])
    # 대표 카테고리만 남기고, 건수는 "연관어별 총합" 하나로 확정한다.
    best_cat = tmp.drop_duplicates(subset=["연관어"], keep="first")[["연관어", "카테고리 대분류"]].copy()
    total = df.groupby(["연관어"], as_index=False)["건수"].sum()
    out = best_cat.merge(total, on="연관어", how="left")
    out["년도"] = -1
    return out[["연관어", "건수", "카테고리 대분류", "년도"]]


# =============================
# UI / Graph
# =============================

st.set_page_config(page_title="썸트렌드 연관성 분석", layout="wide")
inject_noto_sans_kr()
st.title("🕸️ 썸트렌드 연관성 분석")

with st.sidebar:
    st.header("데이터 소스")
    sheet_url = st.text_input("구글시트 URL", value=DEFAULT_SHEET_URL)
    st.caption("공개/공유 설정이 '링크가 있는 모든 사용자 보기 가능'이어야 합니다.")
    st.divider()

    st.header("필터/레이아웃")
    top_n = st.slider("표시 연관어 수", 10, 200, 80, 5)
    k_val = st.slider("노드 간격(k)", 0.8, 4.0, 2.0, 0.1)
    iters = st.slider("레이아웃 반복(iterations)", 50, 600, 220, 10)
    node_mul = st.slider("노드 크기 배수", 0.5, 3.0, 1.2, 0.1)
    st.caption(f"한글 폰트: {_CHOSEN_FONT or '감지 실패(깨짐 시 맑은 고딕/나눔고딕 설치 필요)'}")

    if st.button("데이터 새로고침(캐시 삭제)"):
        st.cache_data.clear()


try:
    raw = fetch_xlsx_as_df(sheet_url)
    df = parse_asso(raw)
except Exception as e:
    st.error(f"구글시트 데이터를 불러오지 못했습니다: {e}")
    st.stop()

st.caption(f"업데이트됨: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

years = sorted(df["년도"].unique().tolist())
year_options = ["전체"] + [str(y) for y in years]
year_sel = st.selectbox("년도 선택", year_options, index=len(year_options) - 1, key="asso_year_sel")

if year_sel == "전체":
    df_view = summarize_all_years(df)
    title_year = "전체"
else:
    year_int = int(year_sel)
    df_view = df[df["년도"] == year_int].copy()
    title_year = f"{year_int}년"

df_view = df_view.sort_values("건수", ascending=False).head(top_n)

center = "K-Wine"
G = nx.Graph()
G.add_node(center, category="CENTER", count=float(df_view["건수"].sum()) if len(df_view) else 1.0)

cats = sorted([c for c in df_view["카테고리 대분류"].dropna().unique().tolist() if str(c) != ""])
palette = plt.get_cmap("Set3")
cmap = {cat: palette(i % getattr(palette, "N", 12)) for i, cat in enumerate(cats)}

for _, r in df_view.iterrows():
    w = str(r["연관어"]).strip()
    if not w or w.lower() == "nan":
        continue
    cnt = float(r["건수"])
    cat = str(r["카테고리 대분류"]).strip()
    G.add_node(w, category=cat, count=cnt)
    G.add_edge(center, w, weight=cnt)

if G.number_of_nodes() <= 1:
    st.info("표시할 연관어가 없습니다. (필터 결과가 비어있음)")
    st.stop()

init_pos = {center: np.array([0.0, 0.0])}
pos = nx.spring_layout(G, seed=42, k=float(k_val), iterations=int(iters), pos=init_pos, fixed=[center])

node_colors, node_sizes, labels = [], [], {}
max_cnt = max([d.get("count", 1.0) for n, d in G.nodes(data=True) if n != center] + [1.0])

for n, d in G.nodes(data=True):
    if n == center:
        node_colors.append("#1f1f1f")
        node_sizes.append(3200 * float(node_mul))
        labels[n] = n
    else:
        node_colors.append(cmap.get(d.get("category", ""), "gray"))
        c = float(d.get("count", 0.0))
        node_sizes.append((550 + 2800 * (c / max_cnt)) * float(node_mul))
        labels[n] = f"{n}\n({int(round(c))})"

weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]
max_w = max(weights) if weights else 1.0
edge_w = [0.6 + 3.2 * (w / max_w) for w in weights]

fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
ax.set_facecolor("white")
ax.axis("off")

nx.draw_networkx_edges(G, pos, ax=ax, width=edge_w, alpha=0.22, edge_color="#9aa0a6")
nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    linewidths=1.2,
    edgecolors="#4a4a4a",
)

font_prop = _KOREAN_FONT_PROP
for n, (x, y) in pos.items():
    is_center = n == center
    ax.text(
        x,
        y,
        labels.get(n, n),
        ha="center",
        va="center",
        fontsize=16 if is_center else 11,
        fontweight="bold" if is_center else "normal",
        color="white" if is_center else "#111111",
        fontproperties=font_prop,
        path_effects=[
            pe.withStroke(linewidth=3.5, foreground="black") if is_center else pe.withStroke(linewidth=3.0, foreground="white")
        ],
    )

handles = [mpatches.Patch(color=cmap[cat], label=cat) for cat in cats]
if handles:
    ax.legend(
        handles=handles,
        title="카테고리 대분류",
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#dddddd",
        framealpha=0.95,
    )

ax.set_title(f"{title_year} K-Wine 연관어 네트워크", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

with st.expander("원본/필터 데이터 보기"):
    st.dataframe(df_view.reset_index(drop=True), use_container_width=True, height=380)


