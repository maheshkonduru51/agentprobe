from __future__ import annotations

import streamlit as st

from dashboard.common import active_model_sidebar, api_get, page_header, require_backend

st.set_page_config(page_title="AgentProbe | Trace Viewer", page_icon="🔍", layout="wide")
active_model_sidebar()
page_header("🔍 Trace Viewer", "Replay a complete episode: thought/rationale, action, observation and guard events.")
if not require_backend(): st.stop()

runs=api_get("/runs")
completed=[r for r in runs if r["status"] in {"completed","running"}]
if not completed:
    st.info("Run an experiment first.")
    st.stop()
run_labels={f"#{r['id']} — {r['variant']} / {r['suite']}":r['id'] for r in completed}
label=st.selectbox("Run", list(run_labels))
episodes=api_get(f"/runs/{run_labels[label]}/episodes")
if not episodes:
    st.info("No episodes are stored yet.")
    st.stop()

ep_labels={f"Episode #{e['id']} — {e['task_id']} — {'✅' if e['success'] else '❌'}":e["id"] for e in episodes}
ep_label=st.selectbox("Episode", list(ep_labels))
trace=api_get(f"/episodes/{ep_labels[ep_label]}/trace")
ep=trace["episode"]
c1,c2,c3,c4=st.columns(4)
c1.metric("Success", "Yes" if ep["success"] else "No")
c2.metric("Steps", ep["steps_used"])
c3.metric("Tokens", ep["tokens"])
c4.metric("Latency", f"{ep['latency_ms']:.0f} ms")

for step in trace["steps"]:
    title=f"Step {step['idx']} — {step['action'] or 'Final answer'}"
    with st.expander(title, expanded=not step["idx"]):
        st.write("**Thought / rationale**")
        st.write(step["thought"])
        st.write("**Arguments**")
        st.json(step["arguments"])
        st.write("**Observation**")
        st.code(step["observation"] or "<empty>")
        if step.get("guard_event"):
            st.warning(step["guard_event"])

if trace["failures"]:
    st.subheader("Failure classification")
    for f in trace["failures"]:
        st.error(f"{f['label']}: {f['evidence']}")
if trace["safety_events"]:
    st.subheader("Safety events")
    st.dataframe(trace["safety_events"], use_container_width=True, hide_index=True)
