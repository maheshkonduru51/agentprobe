from __future__ import annotations

import streamlit as st

from dashboard.common import active_model_sidebar, api_get, page_header, require_backend

st.set_page_config(page_title="AgentProbe | Leaderboard", page_icon="🏆", layout="wide")
active_model_sidebar()
page_header("🏆 Leaderboard & Compare", "Compare completed runs using task success and paired bootstrap statistics.")
if not require_backend():
    st.stop()

runs = api_get("/runs")
if not runs:
    st.info("No runs yet. Start one from the Run Experiment page.")
    st.stop()

rows=[]
for r in runs:
    summary=r.get("summary",{})
    rows.append({
        "Run":r["id"], "Model":r["model"], "Variant":r["variant"], "Suite":r["suite"],
        "Status":r["status"], "Success":summary.get("task_success_rate",0),
        "Tool Acc.":summary.get("tool_call_accuracy",0), "Efficiency":summary.get("step_efficiency",0),
        "Recovery":summary.get("recovery_rate",0), "Attack Success":summary.get("attack_success_rate",0),
        "Over-refusal":summary.get("over_refusal_rate",0), "Latency ms":summary.get("latency_ms_mean",0),
        "Tokens":summary.get("tokens",0),
    })
st.dataframe(rows, use_container_width=True, hide_index=True)

completed=[r for r in runs if r["status"]=="completed"]
if len(completed)>=2:
    st.subheader("Compare two runs")
    options={f"#{r['id']} — {r['variant']} / {r['suite']}":r["id"] for r in completed}
    left=st.selectbox("Run A", list(options))
    right=st.selectbox("Run B", list(options), index=1)
    if st.button("📐 Compare", type="primary"):
        result=api_get("/compare", run_a=options[left], run_b=options[right])
        c1,c2,c3=st.columns(3)
        c1.metric("Difference (B − A)", f"{result['difference']:.3f}")
        c2.metric("95% CI", f"[{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]")
        c3.metric("Statistically significant", "Yes" if result["significant"] else "No")
        st.json(result)
