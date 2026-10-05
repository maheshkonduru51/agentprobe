from __future__ import annotations

import plotly.express as px
import pandas as pd
import streamlit as st

from dashboard.common import active_model_sidebar, api_get, page_header, require_backend

st.set_page_config(page_title="AgentProbe | Safety Report", page_icon="🛡️", layout="wide")
active_model_sidebar()
page_header("🛡️ Safety Report", "Red-team attacks, guardrail events and over-refusal trade-offs.")
if not require_backend(): st.stop()

data=api_get("/safety/summary")
c1,c2,c3=st.columns(3)
c1.metric("Safety events", data.get("events",0))
c2.metric("Blocked", data.get("blocked",0))
c3.metric("Observed kinds", len(data.get("by_kind",{})))

by_kind=data.get("by_kind",{})
if by_kind:
    df=pd.DataFrame({"Kind":list(by_kind),"Count":list(by_kind.values())}).sort_values("Count",ascending=False)
    st.dataframe(df,use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(df,x="Kind",y="Count",title="Safety events by kind"),use_container_width=True)

st.subheader("Compare guardrail modes")
runs=api_get("/runs")
safe=[r for r in runs if r["suite"]=="safety" and r["status"]=="completed"]
rows=[]
for r in safe:
    s=r.get("summary",{})
    rows.append({"Run":r["id"],"Guard":r["guard_mode"],"Success":s.get("task_success_rate",0),"Attack Success":s.get("attack_success_rate",0),"Over-refusal":s.get("over_refusal_rate",0)})
if rows:
    st.dataframe(rows,use_container_width=True,hide_index=True)
else:
    st.info("Run the safety suite with guard=off and guard=block to populate this comparison.")
