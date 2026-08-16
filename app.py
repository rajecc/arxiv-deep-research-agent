import os
import streamlit as st
from datetime import datetime

from configs.settings import settings
from src.models.agent_state import ResearchState
from src.agents.research_graph import research_agent_graph

# Configure Streamlit page
st.set_page_config(
    page_title="ArXiv Deep-Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">🔬 ArXiv Deep-Research Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Autonomous Multi-Agent System for Deep Technical Research & Comparative Analysis</div>',
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Agent Settings")
        
        provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "openai_compatible"],
            index=0 if settings.DEFAULT_LLM_PROVIDER == "gemini" else 1,
            help="Select provider configured in .env or enter API key below",
        )

        api_key_override = st.text_input(
            "API Key (Optional override)",
            type="password",
            help="Leave empty to use API key from .env",
        )

        if api_key_override:
            if provider == "gemini":
                settings.GEMINI_API_KEY = api_key_override
            else:
                settings.OPENAI_API_KEY = api_key_override

        st.markdown("---")
        max_papers = st.slider("Max Papers to Analyze", min_value=1, max_value=5, value=2)
        min_year = st.slider("Min Publication Year", min_value=2023, max_value=2026, value=2024)

        st.markdown("---")
        st.caption("🚀 Powered by LangGraph • PyMuPDF4LLM • ArXiv • Hugging Face")

    # Main input area
    preset_queries = [
        "Speculative Decoding in Large Language Models",
        "Test-Time Compute Scaling and Reasoning in LLMs",
        "KV Cache Compression and Context Window Extension",
        "Linear Attention and State Space Models for LLMs",
    ]

    selected_preset = st.selectbox(
        "💡 Quick Suggestions or type custom query below:",
        options=["-- Custom Topic --"] + preset_queries,
    )

    default_query = selected_preset if selected_preset != "-- Custom Topic --" else "Speculative Decoding in Large Language Models"
    user_query = st.text_input("Enter AI/ML Research Topic:", value=default_query)

    start_button = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

    if start_button and user_query:
        initial_state = ResearchState(
            user_query=user_query,
            max_papers=max_papers,
            min_year=min_year,
        )

        status_container = st.status("🤖 Deep Research Agent in progress...", expanded=True)

        final_output = None
        try:
            # Stream execution
            final_output = {}
            for event in research_agent_graph.stream(initial_state):
                for node_name, node_output in event.items():
                    msg = node_output.get("status_message", f"Running {node_name}...")
                    status_container.write(f"✔ **{node_name.replace('_', ' ').title()}**: {msg}")
                    final_output.update(node_output)

            status_container.update(label="✅ Deep Research Completed!", state="complete", expanded=False)

        except Exception as e:
            status_container.update(label=f"❌ Error: {e}", state="error")
            st.error(f"Error during agent execution: {e}")
            return

        if final_output:
            papers = final_output.get("retrieved_papers", [])
            analyses = final_output.get("paper_analyses", [])
            final_report = final_output.get("final_report") or final_output.get("draft_report") or ""
            saved_path = final_output.get("saved_report_path", "")

            # Tabs for results
            tab_report, tab_papers, tab_analyses = st.tabs([
                "📄 Master Technical Report",
                f"📚 Discovered Papers ({len(papers)})",
                f"🔬 Structured Extractions ({len(analyses)})",
            ])

            with tab_report:
                st.download_button(
                    label="📥 Download Research Report (.md)",
                    data=final_report,
                    file_name=f"arxiv_research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    type="primary",
                )
                st.markdown(final_report)

            with tab_papers:
                for p in papers:
                    with st.expander(f"📌 {p.title} (arXiv:{p.arxiv_id})"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Primary Category", p.primary_category)
                        with col2:
                            st.metric("Hugging Face Upvotes", f"⭐ {p.hf_upvotes}")
                        with col3:
                            st.metric("Citations", f"📚 {p.citation_count or 0}")

                        st.write(f"**Authors:** {', '.join(p.authors)}")
                        st.write(f"**Published:** {p.published_date[:10] if p.published_date else 'N/A'}")
                        st.write(f"**Abstract:** {p.abstract}")
                        
                        if p.github_urls:
                            st.markdown(f"**Code Repos:** {', '.join([f'[{url}]({url})' for url in p.github_urls])}")
                        st.link_button("View on ArXiv", p.arxiv_url)

            with tab_analyses:
                for a in analyses:
                    with st.expander(f"🔬 Analysis: {a.title} ({a.arxiv_id})"):
                        st.markdown(f"**💡 Core Innovation:** {a.core_innovation}")
                        st.markdown(f"**🏗 Architecture:** {a.architecture_details}")
                        if a.mathematical_formulation:
                            st.markdown(f"**🧮 Mathematical Formulation:**\n{a.mathematical_formulation}")
                        if a.reproducibility_notes:
                            st.markdown(f"**🔗 Reproducibility:** {a.reproducibility_notes}")
                        
                        if a.benchmarks:
                            st.write("**📊 Benchmarks:**")
                            for b in a.benchmarks:
                                st.write(f"- `{b.task_or_dataset}` on `{b.base_model}`: Speedup = **{b.speedup_factor or 'N/A'}**, Accuracy = `{b.accuracy_delta or 'N/A'}`")


if __name__ == "__main__":
    main()
