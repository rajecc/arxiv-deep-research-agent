import pytest
from src.models.agent_state import ResearchState, PaperAnalysis, BenchmarkMetric
from src.agents.research_graph import build_research_graph
from src.agents.query_planner import plan_queries
from src.agents.synthesizer import _generate_fallback_synthesis
from src.agents.fact_checker import verify_and_refine_report


def test_graph_compilation():
    """Verify that the LangGraph state graph compiles cleanly with valid edges."""
    graph = build_research_graph()
    assert graph is not None


def test_query_planner():
    """Test that query planning returns valid query strings."""
    state = ResearchState(user_query="Speculative Decoding in LLMs", max_papers=2)
    output = plan_queries(state)
    
    assert "expanded_queries" in output
    assert len(output["expanded_queries"]) >= 1
    assert any("speculative" in q.lower() for q in output["expanded_queries"])


def test_fallback_synthesizer():
    """Test deterministic report synthesis from structured paper analyses."""
    state = ResearchState(
        user_query="Speculative Sampling",
        paper_analyses=[
            PaperAnalysis(
                arxiv_id="2305.04388",
                title="Accelerating Large Language Model Decoding with Speculative Sampling",
                core_innovation="Draft and verify sampling algorithm.",
                architecture_details="Small draft model generates K tokens verified by target model.",
                benchmarks=[
                    BenchmarkMetric(
                        task_or_dataset="HumanEval",
                        base_model="LLaMA-70B",
                        speedup_factor="2.3x",
                        accuracy_delta="Lossless",
                        hardware="1x A100",
                    )
                ],
                reproducibility_notes="Official GitHub repo available.",
            )
        ]
    )

    report = _generate_fallback_synthesis(state)
    assert "Speculative Sampling" in report
    assert "2305.04388" in report
    assert "2.3x" in report
    assert "Comparative Benchmark Matrix" in report


def test_fact_checker_passthrough():
    """Test that fact-checker handles verification cleanly."""
    state = ResearchState(
        user_query="Speculative Sampling",
        draft_report="# Sample Report\nSpeedup is 2.3x on LLaMA.",
        paper_analyses=[],
    )

    result = verify_and_refine_report(state)
    assert "final_report" in result
    assert result["fact_check_passed"] is True
