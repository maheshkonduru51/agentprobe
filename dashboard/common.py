"""Shared Streamlit helpers."""
from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_get(path: str, **params: Any) -> Any:
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=10.0)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def require_backend() -> bool:
    try:
        data = api_get("/health")
        if data.get("status") == "ok":
            st.sidebar.success("Backend connected")
            return True
    except Exception as exc:
        st.sidebar.error("Backend unavailable")
        st.info("Start FastAPI first: uvicorn agentprobe.api.main:app --reload --port 8000")
        st.caption(f"Connection detail: {exc}")
    return False


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def active_model_sidebar() -> None:
    st.sidebar.markdown("### Active configuration")
    st.sidebar.code(os.getenv("LLM_MODEL", "mock"), language="text")
    st.sidebar.caption("Mode: evaluation lab")
