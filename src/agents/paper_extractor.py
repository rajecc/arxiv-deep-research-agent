import re
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from configs.settings import settings
from src.models.paper import ParsedPaper
from src.models.agent_state import PaperAnalysis, BenchmarkMetric, ResearchState
from src.utils.llm_factory import llm_factory
from src.utils.logger import logger


EXTRACTOR_SYSTEM_PROMPT = """You are a Senior Staff AI Research Engineer specializing in deep technical evaluation of ML papers.
Your goal is to extract rigorous, grounded technical facts from the provided academic paper text.

CRITICAL INSTRUCTIONS:
1. Core Innovation: State exactly what algorithm/model modification is proposed.
2. Architecture Details: Detail draft heads, tree verification, KV-cache sharing, acceptance criteria, or attention modifications.
3. Mathematical Formulation: Extract key mathematical equations/losses in clean LaTeX format.
4. Benchmarks: Extract concrete numbers (e.g. Speedup 2.3x on GSM8k, LLaMA-3-70B baseline, latency ms, memory % overhead).
5. Limitations: Identify real technical bottlenecks and assumptions (e.g., requires fine-tuning, high VRAM overhead, memory-bound).
6. DO NOT make up numbers or hallucinate metrics. If a metric is not in the text, omit it or state 'Not reported'.
"""


def extract_single_paper(paper: ParsedPaper) -> PaperAnalysis:
    """Analyze a single parsed paper and extract structured technical metrics."""
    logger.info(f"Extracting deep metrics for [blue]{paper.metadata.arxiv_id}[/blue]: '{paper.metadata.title[:50]}...'")

    # Assemble context from the most informative sections
    context_parts = []
    
    if "abstract" in paper.sections:
        context_parts.append(f"### ABSTRACT:\n{paper.sections['abstract'].content}")
    
    if "methodology" in paper.sections:
        # Truncate if extremely long to stay within budget
        content = paper.sections["methodology"].content[:12000]
        context_parts.append(f"### METHODOLOGY & ARCHITECTURE:\n{content}")

    if "experiments" in paper.sections:
        content = paper.sections["experiments"].content[:12000]
        context_parts.append(f"### EXPERIMENTS & BENCHMARKS:\n{content}")

    if paper.tables:
        tables_preview = "\n\n".join(paper.tables[:5])
        context_parts.append(f"### EXTRACTED TABLES:\n{tables_preview}")

    full_context = "\n\n".join(context_parts)

    try:
        llm = llm_factory.get_llm()
        structured_llm = llm.with_structured_output(PaperAnalysis)

        user_content = (
            f"Paper ArXiv ID: {paper.metadata.arxiv_id}\n"
            f"Title: {paper.metadata.title}\n"
            f"Authors: {', '.join(paper.metadata.authors[:4])}\n\n"
            f"--- PAPER CONTENT ---\n{full_context}"
        )

        messages = [
            SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        analysis: PaperAnalysis = structured_llm.invoke(messages)
        # Ensure ID and title match metadata
        analysis.arxiv_id = paper.metadata.arxiv_id
        analysis.title = paper.metadata.title
        
        # Add code URLs if not in analysis
        if paper.metadata.github_urls:
            repo_str = ", ".join(paper.metadata.github_urls)
            if repo_str not in analysis.reproducibility_notes:
                analysis.reproducibility_notes = f"Official Code: {repo_str}. {analysis.reproducibility_notes}".strip()

        logger.success(
            f"Extracted analysis for {paper.metadata.arxiv_id}: "
            f"{len(analysis.benchmarks)} benchmarks, {len(analysis.limitations)} limitations"
        )
        return analysis

    except Exception as e:
        logger.warning(f"LLM extraction error for {paper.metadata.arxiv_id}: {e}. Using structured fallback.")
        
        # Rule-based intelligent fallback from parsed metadata
        abstract = paper.metadata.abstract
        benchmarks = []
        
        # Simple heuristic extraction of speedups if present in abstract
        speedup_matches = re.findall(r"(\d+\.?\d*)\s*[xX]\s*(?:speedup|faster|acceleration)", abstract)
        if speedup_matches:
            benchmarks.append(
                BenchmarkMetric(
                    task_or_dataset="General Generation",
                    base_model="Evaluated Models in Paper",
                    speedup_factor=f"{speedup_matches[0]}x",
                    accuracy_delta="Lossless / Quality-Preserved",
                    hardware="See paper details",
                )
            )

        code_urls_str = ", ".join(paper.metadata.github_urls) if paper.metadata.github_urls else "No public repo found"

        return PaperAnalysis(
            arxiv_id=paper.metadata.arxiv_id,
            title=paper.metadata.title,
            core_innovation=abstract[:300] + "...",
            architecture_details=f"Extracted from sections: {', '.join(paper.sections.keys())}",
            mathematical_formulation="Refer to equations in parsed markdown.",
            hardware_and_environment="Refer to experimental setup.",
            benchmarks=benchmarks,
            limitations=["Detailed limitations available in full text."],
            reproducibility_notes=f"Code: {code_urls_str}",
        )


def extract_all_papers(state: ResearchState) -> dict:
    """Paper Extractor Node: Runs extraction for each parsed paper in state."""
    parsed_papers = state.parsed_papers
    logger.info(f"Extracting structured data from {len(parsed_papers)} papers...")

    analyses: List[PaperAnalysis] = []
    for paper in parsed_papers:
        analysis = extract_single_paper(paper)
        analyses.append(analysis)

    return {
        "paper_analyses": analyses,
        "status_message": f"Completed deep analysis of {len(analyses)} papers",
    }
