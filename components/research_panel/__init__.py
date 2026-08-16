"""
Python wrapper for the ArXiv Research Panel Streamlit Custom Component.

Usage:
    from components.research_panel import research_panel

    result = research_panel(
        query="Speculative Decoding",
        papers=[...],
        analyses=[...],
        report="# Report...",
        fact_check_passed=True,
        saved_path="reports/report.md",
        is_loading=False,
        status_steps=["Retrieved 3 papers", "Parsed PDFs"],
        key="main_panel",
    )
"""

import os
import streamlit.components.v1 as components
from typing import Optional

# When building the component for production, set this to the build/ directory.
# During development, point to the React dev server (port 3001).
_COMPONENT_DEV_MODE = os.environ.get("RESEARCH_PANEL_DEV", "false").lower() == "true"
_DEV_SERVER_URL = "http://localhost:3001"
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "build")


def _get_component():
    """Lazy-init the component declaration."""
    if _COMPONENT_DEV_MODE:
        return components.declare_component(
            "research_panel",
            url=_DEV_SERVER_URL,
        )
    else:
        return components.declare_component(
            "research_panel",
            path=_BUILD_DIR,
        )


_component_func = None


def _component():
    global _component_func
    if _component_func is None:
        _component_func = _get_component()
    return _component_func


def research_panel(
    *,
    query: str = "",
    papers: list = None,
    analyses: list = None,
    report: str = "",
    fact_check_passed: bool = False,
    saved_path: str = "",
    is_loading: bool = False,
    status_steps: list = None,
    key: Optional[str] = None,
):
    """
    Render the Apple Dark Blue ArXiv Research Panel component.

    Args:
        query:              Current research topic / query string.
        papers:             List of Paper dicts from ArXiv retrieval.
        analyses:           List of PaperAnalysis dicts (structured extractions).
        report:             Markdown string of the synthesized master report.
        fact_check_passed:  Whether the LLM fact-checker validated the report.
        saved_path:         Path where the report was saved on disk.
        is_loading:         Show animated loading state when True.
        status_steps:       List of completed pipeline step descriptions.
        key:                Optional Streamlit key for component identity.

    Returns:
        dict | None: Event payload from user interactions (paper_selected,
                     report_downloaded), or None.
    """
    papers = papers or []
    analyses = analyses or []
    status_steps = status_steps or []

    # Convert Pydantic models to dicts if necessary
    def _to_dict(obj):
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return obj

    papers_data = [_to_dict(p) for p in papers]
    analyses_data = [_to_dict(a) for a in analyses]

    return _component()(
        query=query,
        papers=papers_data,
        analyses=analyses_data,
        report=report,
        fact_check_passed=fact_check_passed,
        saved_path=saved_path,
        is_loading=is_loading,
        status_steps=status_steps,
        key=key,
        default=None,
    )
