from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from src.models.agent_state import ResearchState
from src.models.paper import PaperMetadata, ParsedPaper
from src.retrievers.arxiv_client import arxiv_client
from src.retrievers.hf_client import hf_client
from src.retrievers.semanticscholar import semanticscholar_client
from src.parsers.pdf_parser import pdf_parser
from src.parsers.section_splitter import section_splitter
from src.agents.query_planner import plan_queries
from src.agents.paper_extractor import extract_all_papers
from src.agents.synthesizer import synthesize_research_report
from src.agents.fact_checker import verify_and_refine_report
from src.utils.report_exporter import save_markdown_report
from src.utils.logger import logger


def retrieve_papers_node(state: ResearchState) -> Dict[str, Any]:
    """Node: Searches ArXiv with all expanded queries, deduplicates, and enriches metadata."""
    queries = state.expanded_queries or [state.user_query]
    max_total = state.max_papers
    min_year = state.min_year

    logger.info(f"Retrieving papers across {len(queries)} query variations (target={max_total})...")

    collected_dict: Dict[str, PaperMetadata] = {}

    for q in queries:
        try:
            results = arxiv_client.search_papers(
                query=q,
                max_results=max_total,
                min_year=min_year,
            )
            for p in results:
                if p.arxiv_id not in collected_dict:
                    collected_dict[p.arxiv_id] = p
                if len(collected_dict) >= max_total:
                    break
        except Exception as e:
            logger.warning(f"Error executing sub-query '{q}': {e}")

        if len(collected_dict) >= max_total:
            break

    papers = list(collected_dict.values())[:max_total]

    # Enrich metadata
    logger.info(f"Enriching {len(papers)} unique papers with HF Daily Papers & Semantic Scholar...")
    for p in papers:
        hf_client.enrich_paper_metadata(p)
        semanticscholar_client.enrich_paper_metadata(p)

    return {
        "retrieved_papers": papers,
        "status_message": f"Retrieved and enriched {len(papers)} unique papers",
    }


def parse_papers_node(state: ResearchState) -> Dict[str, Any]:
    """Node: Downloads PDFs, converts to high-fidelity Markdown, and segments sections."""
    papers = state.retrieved_papers
    logger.info(f"Downloading and parsing {len(papers)} PDFs to structured Markdown...")

    parsed_list: List[ParsedPaper] = []
    for paper in papers:
        parsed = pdf_parser.parse_pdf_to_markdown(paper)
        parsed = section_splitter.split_sections(parsed)
        parsed_list.append(parsed)

    return {
        "parsed_papers": parsed_list,
        "status_message": f"Parsed {len(parsed_list)} papers into structured Markdown",
    }


def export_report_node(state: ResearchState) -> Dict[str, Any]:
    """Node: Saves verified report to disk in the reports directory."""
    final_content = state.final_report or state.draft_report or ""
    file_path = save_markdown_report(state.user_query, final_content)

    return {
        "saved_report_path": str(file_path),
        "status_message": f"Research report saved to {file_path.name}",
    }


def build_research_graph() -> StateGraph:
    """Build and compile the LangGraph Deep-Research workflow."""
    workflow = StateGraph(ResearchState)

    # 1. Add all functional nodes
    workflow.add_node("plan_queries", plan_queries)
    workflow.add_node("retrieve_papers", retrieve_papers_node)
    workflow.add_node("parse_papers", parse_papers_node)
    workflow.add_node("extract_analyses", extract_all_papers)
    workflow.add_node("synthesize_report", synthesize_research_report)
    workflow.add_node("fact_check", verify_and_refine_report)
    workflow.add_node("export_report", export_report_node)

    # 2. Add sequential control edges
    workflow.add_edge(START, "plan_queries")
    workflow.add_edge("plan_queries", "retrieve_papers")
    workflow.add_edge("retrieve_papers", "parse_papers")
    workflow.add_edge("parse_papers", "extract_analyses")
    workflow.add_edge("extract_analyses", "synthesize_report")
    workflow.add_edge("synthesize_report", "fact_check")
    workflow.add_edge("fact_check", "export_report")
    workflow.add_edge("export_report", END)

    return workflow.compile()


# Global compiled graph instance
research_agent_graph = build_research_graph()
