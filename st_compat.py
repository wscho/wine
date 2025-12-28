from __future__ import annotations

from typing import Any, Optional

import streamlit as st


def button_full(label: str, *, key: Optional[str] = None, **kwargs: Any) -> bool:
    """
    Streamlit 버전별 호환:
    - 일부 버전은 width="stretch"를 지원, 일부는 TypeError
    - 우선 width="stretch" 시도 후 실패 시 use_container_width=True fallback
    """
    try:
        return st.button(label, key=key, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, key=key, use_container_width=True, **kwargs)


def image_full(image: Any, *, caption: Optional[str] = None, **kwargs: Any) -> None:
    """
    st.image에서 width="stretch" 지원 여부가 버전별로 달라 TypeError가 날 수 있음.
    """
    try:
        st.image(image, caption=caption, width="stretch", **kwargs)
    except TypeError:
        st.image(image, caption=caption, use_container_width=True, **kwargs)


def dataframe_full(data: Any, *, height: Optional[int] = None, **kwargs: Any) -> None:
    """
    st.dataframe에서도 버전별로 width 인자가 다를 수 있어 fallback 제공.
    """
    try:
        st.dataframe(data, height=height, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, height=height, use_container_width=True, **kwargs)


