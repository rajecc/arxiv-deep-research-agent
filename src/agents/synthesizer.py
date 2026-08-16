import json
from datetime import datetime
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from configs.settings import settings
from src.models.agent_state import ResearchState, PaperAnalysis
from src.utils.llm_factory import llm_factory
from src.utils.logger import logger


SYNTHESIZER_SYSTEM_PROMPT = """You are a Principal AI Architect and Lead Research Scientist.
Your task is to write an exhaustive, authoritative, Senior-Staff-level technical research report in Markdown comparing and synthesizing the provided academic papers.

Structure Requirements for the Report:
# [Comprehensive Technical Title on the Research Topic]

## 1. Executive Summary & State-of-the-Art Landscape
- Core technological trend and what problem these papers collectively solve.
- High-level comparison of the methodologies.

## 2. Master Comparative Benchmark Matrix
Present a detailed Markdown Table with columns:
| Method / Paper | Base Model(s) | Benchmark / Dataset | Speedup / Primary Metric | Memory / VRAM Impact | Hardware Setup | Code & Artifacts |
(Include concrete numbers extracted from the papers).

## 3. Deep Architectural Analysis & Key Equations
- Contrast the mathematical models, KV-cache strategies, draft mechanisms, and verification algorithms.
- Include key LaTeX formulas ($$...$$) for loss functions or acceptance criteria.

## 4. Engineering Trade-offs & Production Viability Matrix
- Which approach is ready for low-latency production serving vs purely theoretical?
- Trade-offs between memory overhead, training cost (draft model training vs training-free), and throughput gains.

## 5. Verified Open-Source Artifacts & References
- Direct links to GitHub repositories, Hugging Face checkpoints, ArXiv IDs, and publication dates.

Tone: Rigorous, highly technical, objective, and engineering-focused. Avoid fluff and generic buzzwords.
"""


def _generate_fallback_synthesis(state: ResearchState) -> str:
    """Generate a high-quality deterministic Markdown report if LLM is unavailable."""
    now_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# 🔬 Deep Research Report: {state.user_query.title()}",
        f"*Generated on: {now_str} | Target Papers Analyzed: {len(state.paper_analyses)}*",
        "",
        "## 1. Executive Summary & SOTA Landscape",
        f"This report presents a structured comparative analysis of recent publications on **{state.user_query}**.",
        "The evaluated approaches target inference acceleration, architectural optimizations, and quality preservation.",
        "",
        "## 2. Comparative Benchmark Matrix",
        "| Paper / ArXiv ID | Innovation | Base Model | Reported Speedup | Hardware | Code Link |",
        "|:---|:---|:---|:---|:---|:---|",
    ]

    for p in state.paper_analyses:
        speedup = p.benchmarks[0].speedup_factor if p.benchmarks else "N/A"
        base_model = p.benchmarks[0].base_model if p.benchmarks else "N/A"
        hw = p.benchmarks[0].hardware if p.benchmarks else "N/A"
        code = p.reproducibility_notes[:40] if p.reproducibility_notes else "None reported"
        lines.append(f"| **[{p.arxiv_id}](https://arxiv.org/abs/{p.arxiv_id})** | {p.core_innovation[:60]}... | {base_model} | **{speedup}** | {hw} | {code} |")

    lines.extend([
        "",
        "## 3. Deep Architectural Analysis",
    ])

    for p in state.paper_analyses:
        lines.extend([
            f"### {p.title} (`{p.arxiv_id}`)",
            f"- **Core Innovation:** {p.core_innovation}",
            f"- **Architecture & Strategy:** {p.architecture_details}",
            f"- **Mathematical Formulation:** {p.mathematical_formulation or 'N/A'}",
            f"- **Known Bottlenecks:** {', '.join(p.limitations) if p.limitations else 'None noted'}",
            "",
        ])

    lines.extend([
        "## 4. Engineering Trade-offs & Production Viability",
        "- **Throughput vs. Memory:** Multi-head and draft models require additional KV-cache or VRAM allocation.",
        "- **Training-free vs. Speculative Training:** Training-free methods offer immediate drop-in deployment, while trained draft networks yield higher acceptance lengths at the cost of pre-training overhead.",
        "",
        "## 5. Verified Open-Source Artifacts & References",
    ])

    for p in state.retrieved_papers:
        gh_links = ", ".join([f"[{u}]({u})" for u in p.github_urls]) if p.github_urls else "No public repo"
        lines.append(f"- **{p.title}** ([arXiv:{p.arxiv_id}]({p.arxiv_url})) | Code: {gh_links}")

    return "\n".join(lines)


def synthesize_research_report(state: ResearchState) -> dict:
    """Synthesizer Node: Combines all PaperAnalysis objects into a Master Research Report."""
    logger.info(f"Synthesizing comparative report for {len(state.paper_analyses)} papers...")

    if not state.paper_analyses:
        logger.warning("No paper analyses available to synthesize.")
        return {
            "draft_report": "# Deep Research Report\nNo papers were analyzed.",
            "status_message": "No papers available for synthesis",
        }

    # Format structured analyses into clean prompt context
    analyses_payload = []
    for a in state.paper_analyses:
        analyses_payload.append(a.model_dump())

    context_json = json.dumps(analyses_payload, indent=2)

    try:
        llm = llm_factory.get_llm()
        
        user_prompt = (
            f"User Research Topic: {state.user_query}\n"
            f"Number of Papers: {len(state.paper_analyses)}\n\n"
            f"--- STRUCTURED PAPERS DATA (JSON) ---\n{context_json}\n\n"
            "Please generate the complete, comprehensive Markdown report as instructed."
        )

        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        report_content = response.content.strip()
        logger.success(f"Synthesized research report ({len(report_content):,} chars)")

        return {
            "draft_report": report_content,
            "status_message": "Synthesized draft comparative report",
        }

    except Exception as e:
        logger.warning(f"LLM synthesis error: {e}. Generating structured fallback report.")
        fallback_report = _generate_fallback_synthesis(state)
        return {
            "draft_report": fallback_report,
            "status_message": "Generated structured report via deterministic synthesizer",
        }
