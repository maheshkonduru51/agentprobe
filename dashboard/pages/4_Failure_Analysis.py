from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.common import active_model_sidebar, api_get, page_header, require_backend

st.set_page_config(page_title="AgentProbe | Failure Analysis", page_icon="📊", layout="wide")
active_model_sidebar()
page_header("📊 Failure Analysis", "Find the most common agent failure modes and where they appear.")
if not require_backend(): st.stop()

summary=api_get("/failures/summary")
if not summary:
    st.info("No classified failures yet. Run an experiment with a non-zero fault rate or open a safety suite.")
else:
    df=pd.DataFrame({"Failure":list(summary),"Count":list(summary.values())}).sort_values("Count",ascending=False)
    st.dataframe(df,use_container_width=True,hide_index=True)
    fig=px.bar(df,x="Failure",y="Count",title="Failure taxonomy distribution")
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig,use_container_width=True)

st.caption("The classifier uses the 11-label taxonomy defined by the project specification.")
