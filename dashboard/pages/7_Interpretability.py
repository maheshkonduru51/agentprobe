from __future__ import annotations

import streamlit as st

from dashboard.common import active_model_sidebar, page_header

st.set_page_config(page_title="AgentProbe | Interpretability", page_icon="🧠", layout="wide")
active_model_sidebar()
page_header("🧠 Interpretability", "Optional token confidence, calibration and GPT-2 attention views.")

st.info("The interpretability module is optional. The core AgentProbe system does not require transformers or torch.")
with st.expander("Token confidence"):
    st.write("If a real provider returns log-probabilities, the project can calculate mean log-probability and entropy for calibration analysis.")
with st.expander("Attention view"):
    st.write("Install requirements-interp.txt to enable the GPT-2 small CPU view. If the packages are absent, this page intentionally stays usable.")
