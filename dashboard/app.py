"""AgentProbe Streamlit home page: Run Experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from dashboard.common import active_model_sidebar, api_get, api_post, page_header, require_backend

st.set_page_config(page_title="AgentProbe | Run Experiment", page_icon="🧪", layout="wide")
active_model_sidebar()
page_header("🧪 AgentProbe", "Agent Harness, Evaluation & Red-Teaming Lab")

st.markdown(
    "AgentProbe lets you run tool-using AI agents against a controlled benchmark, "
    "inspect traces, classify failures, test safety, and compare experiments."
)

if not require_backend():
    st.stop()

run_tab, task_tab = st.tabs(["🚀 Run Experiment", "📝 Custom Task Builder"])

with run_tab:
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.text_input("Model", value="mock", help="Keep 'mock' for a fully offline, zero-cost run.")
        variant = st.selectbox("Agent variant", ["react", "plan_execute", "react_reflect", "all"])
    with col2:
        suite = st.selectbox("Task suite", ["all", "tool_use", "planning", "long_horizon", "safety"])
        seeds = st.number_input("Seeds", min_value=1, max_value=10, value=1, step=1)
    with col3:
        fault_rate = st.slider("Fault injection", min_value=0.0, max_value=0.30, value=0.0, step=0.05, format="%.2f")
        guard = st.radio("Guardrail mode", ["off", "log", "block"], horizontal=True, index=2)
        workers = st.number_input("Workers", min_value=1, max_value=8, value=1, step=1)

    start = st.button("🚀 Start Experiment", type="primary", use_container_width=True)
    if start:
        try:
            response = api_post("/runs", {"model": model, "variant": variant, "suite": suite, "seeds": int(seeds), "fault_rate": fault_rate, "guard_mode": guard, "workers": int(workers)})
            run_id = response["run_id"]
            st.success(f"Run #{run_id} started.")
            progress = st.progress(0)
            status_box = st.empty()
            for _ in range(900):
                data = api_get(f"/runs/{run_id}/progress")
                progress.progress(min(100, int(data["percent"])))
                status_box.info(f"Status: {data['status']} — {data['completed']}/{data['total']} episodes")
                if data["status"] in {"completed", "failed"}:
                    break
                time.sleep(1)
            result = api_get(f"/runs/{run_id}")
            if result["status"] == "completed":
                st.success("Experiment completed.")
                st.json(result["summary"])
            else:
                st.warning(f"Run ended with status: {result['status']}")
        except Exception as exc:
            st.error(f"Could not start/run experiment: {exc}")

with task_tab:
    st.subheader("Create a custom benchmark task")
    st.caption("This is an added convenience: the seven-page benchmark UI remains unchanged, while this tab lets you add new JSON tasks without editing files manually.")
    with st.form("custom_task_form"):
        task_id = st.text_input("Task ID", value="custom_001")
        task_suite = st.selectbox("Suite", ["tool_use", "planning", "long_horizon", "safety"], key="custom_suite")
        goal = st.text_area("Task / goal", value="Calculate 15% of 200.")
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], key="custom_diff")
        category = st.text_input("Category", value="custom")
        grader = st.selectbox("Grader", ["exact_match", "numeric_tolerance", "contains_all", "regex", "state_check", "trajectory_check"])
        expected = st.text_area("Expected result (JSON or plain text)", value="30" if grader == "numeric_tolerance" else '"expected text"')
        tools = st.multiselect("Allowed tools", ["calculator", "unit_convert", "kv_store", "file_search", "read_file", "sql_query", "current_date", "send_email_stub", "delete_file_stub"], default=["calculator"])
        mock_plan_json = st.text_area("Mock plan (JSON list)", value=json.dumps([{"tool":"calculator","args":{"expression":"200*0.15"},"thought":"Calculate the percentage."}], indent=2))
        submitted = st.form_submit_button("💾 Save Custom Task", type="primary")
    if submitted:
        try:
            parsed_expected = json.loads(expected)
        except json.JSONDecodeError:
            parsed_expected = expected
        try:
            parsed_plan = json.loads(mock_plan_json)
            if not isinstance(parsed_plan, list):
                raise ValueError("Mock plan must be a JSON list")
            payload = {
                "id": task_id.strip(), "suite": task_suite, "goal": goal.strip(), "difficulty": difficulty,
                "category": category.strip(), "grader_type": grader, "expected": parsed_expected,
                "allowed_tools": tools, "optimal_steps": len(parsed_plan), "mock_plan": parsed_plan,
                "mock_final_answer": str(parsed_expected if not isinstance(parsed_expected, dict) else "done"),
            }
            saved = api_post("/tasks/custom", payload)
            st.success(saved.get("message", "Task saved"))
        except Exception as exc:
            st.error(f"Could not save task: {exc}")
