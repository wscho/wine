# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st


def inject_noto_sans_kr() -> None:
    """
    Streamlit UI(브라우저 렌더링)용 웹폰트 주입.
    - Streamlit 텍스트/Plotly(브라우저) 텍스트에 적용됨
    - matplotlib/wordcloud "이미지"에는 적용되지 않음(그건 폰트 파일이 필요)
    """
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  /*
    웹폰트는 "텍스트 영역"에만 적용합니다.
    (전체 .stApp에 폰트를 덮어쓰면 selectbox 화살표 같은 아이콘이 글자(예: keyboard_arrow_down)로 보일 수 있음)
  */
  .stApp [data-testid="stMarkdownContainer"],
  .stApp label,
  .stApp p,
  .stApp li,
  .stApp input,
  .stApp textarea,
  .stApp button {
    font-family: "Noto Sans KR", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, "Apple SD Gothic Neo",
                 "Malgun Gothic", "NanumGothic", sans-serif !important;
  }

  /* 화면 타이틀(st.title = h1) 크기를 약 1/3 줄임(≈ 0.67배) */
  .stApp h1 {
    font-size: clamp(1.35rem, 2.2vw, 1.75rem) !important;
    line-height: 1.15 !important;
    margin-bottom: 0.35rem !important;
    font-family: "Noto Sans KR", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, "Apple SD Gothic Neo",
                 "Malgun Gothic", "NanumGothic", sans-serif !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


