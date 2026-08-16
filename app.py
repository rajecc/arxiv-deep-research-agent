"""
ArXiv Deep-Research Agent — Streamlit Application
Integrates the Apple Dark Blue Research Panel custom component.
"""

import os
import streamlit as st
from datetime import datetime

from configs.settings import settings
from src.models.agent_state import ResearchState
from src.agents.research_graph import research_agent_graph
from components.research_panel import research_panel

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ArXiv Deep-Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS overrides (outer Streamlit shell) ─────────────
st.markdown(
    """
    <style>
      /* Hide Streamlit default footer & menu */
      #MainMenu, footer { visibility: hidden; }

      /* Dark background for the outer shell */
      .stApp, [data-testid="stAppViewContainer"] {
          background: linear-gradient(160deg, #020813 0%, #0B132B 100%);
      }

      /* Sidebar glass panel */
      [data-testid="stSidebar"] {
          background: rgba(11,19,43,0.90);
          backdrop-filter: blur(24px);
          border-right: 1px solid rgba(58,134,255,0.14);
      }
      [data-testid="stSidebar"] * { color: #8ea8d5 !important; }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
          color: #f0f4ff !important;
          font-weight: 500 !important;
      }

      /* Primary button: Apple blue */
      .stButton>button[kind="primary"] {
          background: linear-gradient(135deg, #3A86FF, #007AFF);
          border: none;
          border-radius: 10px;
          font-weight: 600;
          letter-spacing: 0.2px;
          box-shadow: 0 0 18px rgba(58,134,255,0.4);
          transition: opacity 0.2s, box-shadow 0.2s;
      }
      .stButton>button[kind="primary"]:hover {
          opacity: 0.88;
          box-shadow: 0 0 28px rgba(58,134,255,0.6);
      }

      /* Text inputs */
      .stTextInput input, .stSelectbox select {
          background: rgba(13,27,56,0.7) !important;
          border: 1px solid rgba(58,134,255,0.22) !important;
          border-radius: 10px !important;
          color: #f0f4ff !important;
      }
      .stTextInput input:focus {
          border-color: #3A86FF !important;
          box-shadow: 0 0 0 3px rgba(58,134,255,0.18) !important;
      }

      /* Sliders */
      .stSlider [data-baseweb="slider"] { accent-color: #3A86FF; }

      /* Custom component iframe — remove padding */
      [data-testid="stCustomComponentV1"] iframe { border: none !important; }

      /* Page header */
      .page-header {
          font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
          font-size: clamp(1.6rem, 3.5vw, 2.4rem);
          font-weight: 300;
          letter-spacing: -0.5px;
          background: linear-gradient(90deg, #f0f4ff 0%, #3A86FF 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 4px;
      }
      .page-sub {
          font-size: 0.9rem;
          color: #4a6490;
          letter-spacing: 0.3px;
          margin-bottom: 24px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── State helpers ────────────────────────────────────────────
def init_state():
    defaults = {
        "is_loading": False,
        "status_steps": [],
        "papers": [],
        "analyses": [],
        "report": "",
        "fact_check_passed": False,
        "saved_path": "",
        "query": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main app ─────────────────────────────────────────────────
def main():
    init_state()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Agent Settings")

        provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "openai_compatible"],
            index=0 if settings.DEFAULT_LLM_PROVIDER == "gemini" else 1,
        )

        api_key_override = st.text_input(
            "API Key (Optional override)",
            type="password",
            help="Leave empty to use key from .env",
        )
        if api_key_override:
            if provider == "gemini":
                settings.GEMINI_API_KEY = api_key_override
            else:
                settings.OPENAI_API_KEY = api_key_override

        st.markdown("---")
        max_papers = st.slider("Max Papers to Analyze", 1, 5, 2)
        min_year   = st.slider("Min Publication Year",  2023, 2026, 2024)
        st.markdown("---")
        st.caption("🚀 Powered by LangGraph · PyMuPDF4LLM · ArXiv · Hugging Face · VseLLM")

    # ── Header ────────────────────────────────────────────────
    st.markdown('<div class="page-header">🔬 ArXiv Deep-Research Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Autonomous Multi-Agent System — Deep Technical Research & Comparative Analysis</div>',
        unsafe_allow_html=True,
    )

    # ── Input area ────────────────────────────────────────────
    preset_queries = [
        "Speculative Decoding in Large Language Models",
        "Test-Time Compute Scaling and Reasoning in LLMs",
        "KV Cache Compression and Context Window Extension",
        "Linear Attention and State Space Models for LLMs",
        "Mixture of Experts Routing Efficiency",
        "RLHF and Preference Learning for LLM Alignment",
    ]

    col_sel, col_run = st.columns([4, 1])
    with col_sel:
        selected = st.selectbox(
            "💡 Quick Suggestions:",
            ["-- Custom Topic --"] + preset_queries,
            label_visibility="collapsed",
        )
        default_query = selected if selected != "-- Custom Topic --" else ""
        user_query = st.text_input(
            "Research Topic",
            value=default_query,
            placeholder="e.g. Speculative Decoding in Large Language Models",
            label_visibility="collapsed",
        )
    with col_run:
        start = st.button(
            "🚀 Start Research",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_loading,
        )

    # ── Research execution ────────────────────────────────────
    if start and user_query and not st.session_state.is_loading:
        st.session_state.is_loading = True
        st.session_state.status_steps = []
        st.session_state.papers = []
        st.session_state.analyses = []
        st.session_state.report = ""
        st.session_state.fact_check_passed = False
        st.session_state.saved_path = ""
        st.session_state.query = user_query
        st.rerun()

    if st.session_state.is_loading:
        initial_state = ResearchState(
            user_query=st.session_state.query,
            max_papers=max_papers,
            min_year=min_year,
        )
        try:
            accumulated = {}
            for event in research_agent_graph.stream(initial_state):
                for node_name, node_output in event.items():
                    msg = node_output.get("status_message", f"Completed {node_name}")
                    st.session_state.status_steps.append(f"{node_name.replace('_',' ').title()}: {msg}")
                    accumulated.update(node_output)

            st.session_state.papers           = accumulated.get("retrieved_papers", [])
            st.session_state.analyses         = accumulated.get("paper_analyses", [])
            st.session_state.report           = accumulated.get("final_report") or accumulated.get("draft_report") or ""
            st.session_state.fact_check_passed = accumulated.get("fact_check_passed", False)
            st.session_state.saved_path       = accumulated.get("saved_report_path", "")

        except Exception as exc:
            st.error(f"Agent error: {exc}")
        finally:
            st.session_state.is_loading = False
            st.rerun()

    # ── Custom React component ─────────────────────────────────
    event = research_panel(
        query=st.session_state.query,
        papers=st.session_state.papers,
        analyses=st.session_state.analyses,
        report=st.session_state.report,
        fact_check_passed=st.session_state.fact_check_passed,
        saved_path=st.session_state.saved_path,
        is_loading=st.session_state.is_loading,
        status_steps=st.session_state.status_steps,
        key="research_panel_main",
    )

    # Handle events fired from the React component
    if event:
        if event.get("event") == "paper_selected":
            arxiv_id = event.get("arxiv_id")
            if arxiv_id:
                st.toast(f"📌 Selected paper: {arxiv_id}", icon="🔬")
        elif event.get("event") == "report_downloaded":
            st.toast("📥 Report downloaded via component!", icon="✅")


if __name__ == "__main__":
    main()
