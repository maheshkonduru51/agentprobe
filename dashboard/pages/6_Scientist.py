from __future__ import annotations

import streamlit as st

from dashboard.common import active_model_sidebar, api_get, api_post, page_header, require_backend

st.set_page_config(page_title="AgentProbe | Scientist", page_icon="🧑‍🔬", layout="wide")
active_model_sidebar()
page_header("🧑‍🔬 Scientist-lite", "Turn an evaluation question into controlled experiments and a written report.")
if not require_backend(): st.stop()

goal=st.text_input("Research goal", value="Reduce bad-argument errors")
model=st.text_input("Model", value="mock")
seeds=st.number_input("Seeds",1,5,1)
if st.button("🔬 Start Scientist Experiment",type="primary"):
    try:
        st.success(api_post("/scientist/start", {"goal":goal,"model":model,"seeds":int(seeds)}).get("status","queued"))
        st.info("Scientist runs in the FastAPI background task. Refresh this page after it finishes.")
    except Exception as exc:
        st.error(f"Could not start Scientist-lite: {exc}")

reports=api_get("/scientist/reports")
st.subheader("Generated reports")
if reports:
    for name in reports:
        st.code(name)
else:
    st.info("No reports yet.")
