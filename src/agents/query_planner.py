from typing import List
from langchain_core.messages import SystemMessage, HumanMessage

from configs.settings import settings
from src.models.agent_state import ResearchState, ResearchPlan
from src.utils.llm_factory import llm_factory
from src.utils.logger import logger


PLANNER_SYSTEM_PROMPT = """You are a Principal AI Researcher and ArXiv search strategist.
Your task is to analyze a research query on AI/ML and generate 2-3 highly effective, targeted arXiv search queries.

Requirements:
1. Queries must use academic terminology (e.g. 'speculative decoding', 'draft model', 'tree attention', 'KV cache').
2. Avoid generic filler words (like 'paper', 'survey', 'recent', 'best').
3. Produce 2 to 3 distinct queries covering the core technique and architectural variations.
"""


def plan_queries(state: ResearchState) -> dict:
    """Query Planner Node: Expands user query into targeted ArXiv search queries."""
    user_query = state.user_query
    logger.info(f"Planning search strategy for topic: '[bold]{user_query}[/bold]'")

    # Default fallback queries in case LLM is not configured
    fallback_queries = [
        user_query,
        f"{user_query} architecture",
        f"{user_query} benchmark",
    ]

    try:
        llm = llm_factory.get_llm()
        structured_llm = llm.with_structured_output(ResearchPlan)

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Topic: {user_query}\nTarget number of papers to analyze: {state.max_papers}"),
        ]

        plan: ResearchPlan = structured_llm.invoke(messages)
        expanded = plan.expanded_queries if plan.expanded_queries else fallback_queries
        logger.success(f"Generated {len(expanded)} targeted search queries: {expanded}")
        return {
            "expanded_queries": expanded,
            "status_message": f"Planned {len(expanded)} search queries",
        }

    except Exception as e:
        logger.warning(f"Query planning using fallback queries (LLM note: {e})")
        return {
            "expanded_queries": fallback_queries[:2],
            "status_message": "Using heuristic query expansion",
        }
