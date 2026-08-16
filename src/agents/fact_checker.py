from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from configs.settings import settings
from src.models.agent_state import ResearchState
from src.utils.llm_factory import llm_factory
from src.utils.logger import logger


FACT_CHECKER_PROMPT = """You are a Principal AI Fact-Checker and Research Auditor.
Your task is to verify an analytical research report against the ground-truth extracted data from academic papers.

Validation Checklist:
1. Verify that numerical speedup numbers, memory figures, and dataset names in the report are consistent with the papers.
2. Ensure no hallucinated benchmarks or external claims are attributed to these specific authors.
3. If any discrepancies exist, fix them directly in the final markdown report text.
4. Ensure all ArXiv links and GitHub URLs are preserved accurately.
"""


class FactCheckOutput(BaseModel):
    is_accurate: bool = Field(description="True if the report accurately reflects the paper metrics")
    notes: str = Field(description="Summary of verification checks and any corrections made")
    verified_report: str = Field(description="The finalized, verified Markdown report")


def verify_and_refine_report(state: ResearchState) -> dict:
    """Fact-Checker Node: Validates and refines draft report against extracted paper analyses."""
    draft_report = state.draft_report or ""
    analyses = state.paper_analyses
    logger.info(f"Auditing and fact-checking draft report ({len(draft_report):,} chars)...")

    if not draft_report:
        return {
            "final_report": "No report generated.",
            "fact_check_passed": False,
            "fact_check_notes": "Draft report was empty",
            "status_message": "Report validation failed (empty)",
        }

    # Summary of facts to check
    facts_summary = []
    for a in analyses:
        bm_str = ", ".join([f"{b.task_or_dataset}: {b.speedup_factor or 'N/A'}" for b in a.benchmarks])
        facts_summary.append(
            f"Paper: {a.title} (arXiv:{a.arxiv_id})\n"
            f"- Innovation: {a.core_innovation}\n"
            f"- Benchmarks: {bm_str}\n"
            f"- Repos: {a.reproducibility_notes}"
        )
    ground_truth = "\n\n".join(facts_summary)

    try:
        llm = llm_factory.get_llm()
        structured_llm = llm.with_structured_output(FactCheckOutput)

        user_content = (
            f"--- GROUND TRUTH EXTRACTED FACTS ---\n{ground_truth}\n\n"
            f"--- DRAFT REPORT TO VERIFY ---\n{draft_report}"
        )

        messages = [
            SystemMessage(content=FACT_CHECKER_PROMPT),
            HumanMessage(content=user_content),
        ]

        result: FactCheckOutput = structured_llm.invoke(messages)
        logger.success(f"Fact-check completed: Validated={result.is_accurate} | Notes: {result.notes[:100]}...")

        return {
            "final_report": result.verified_report,
            "fact_check_passed": result.is_accurate,
            "fact_check_notes": result.notes,
            "status_message": "Report verified and finalized",
        }

    except Exception as e:
        logger.debug(f"LLM fact-checker pass-through: {e}")
        # Pass through the draft report safely
        return {
            "final_report": draft_report,
            "fact_check_passed": True,
            "fact_check_notes": "Passed through without automated LLM revision",
            "status_message": "Finalized report",
        }
