# app.py
# Entry point for Streamlit Cloud (VectorAlgoAI MVP)

import os
import streamlit as st
from mvp_dashboard import run_mvp_dashboard

st.set_page_config(
    page_title="VectorAlgoAI – Strategy Crash-Test Lab",
    page_icon="💹",
    layout="wide",
)

# ------------------------------------------------------------
# ACCESS GATE (simple code-based gate for MVP)
# - Set ACCESS_CODE in Streamlit Cloud Secrets or environment.
#   In Streamlit Cloud: Settings -> Secrets:
#     ACCESS_CODE = "YOUR_CODE"
# ------------------------------------------------------------
ACCESS_CODE = None
try:
    ACCESS_CODE = st.secrets.get("ACCESS_CODE", None)
except Exception:
    ACCESS_CODE = None

if ACCESS_CODE is None:
    ACCESS_CODE = os.getenv("ACCESS_CODE")

if "access_granted" not in st.session_state:
    st.session_state["access_granted"] = False

if ACCESS_CODE:
    if not st.session_state["access_granted"]:
        st.title("🔐 VectorAlgoAI – Early Access")
        st.write("Enter your access code to open the Strategy Crash-Test Lab.")
        code = st.text_input("Access code", type="password")
        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("Unlock", use_container_width=True):
                if code.strip() == str(ACCESS_CODE).strip():
                    st.session_state["access_granted"] = True
                    st.success("Access granted. Loading dashboard…")
                    st.rerun()
                else:
                    st.error("Invalid access code.")
        with cols[1]:
            st.markdown('<a href="https://www.vectoralgoai.com/#signup" target="_blank">Get early access</a>', unsafe_allow_html=True)
        st.caption("Tip: If you are the admin, set ACCESS_CODE in Streamlit Secrets.")
        st.stop()
else:
    st.warning("Admin note: ACCESS_CODE is not set. MVP is currently open to everyone.")

# Run the MVP dashboard (no signup/login inside)
run_mvp_dashboard()
