import os
import pytest
from configs.settings import settings
from src.utils.llm_factory import llm_factory
from src.models.agent_state import ResearchState, PaperAnalysis, BenchmarkMetric
from src.models.paper import PaperMetadata, ParsedPaper, PaperSection
from src.agents.query_planner import plan_queries
from src.agents.paper_extractor import extract_single_paper
from src.agents.synthesizer import synthesize_research_report
from src.agents.fact_checker import verify_and_refine_report


@pytest.mark.skipif(not settings.GEMINI_API_KEY, reason="GEMINI_API_KEY not configured")
def test_gemini_connectivity():
    """Verify Gemini 3.7 Flash basic chat connectivity."""
    llm = llm_factory.get_llm(provider="gemini")
    res = llm.invoke("Reply with the single word: READY")
    content = res.content
    if isinstance(content, list):
        content = "".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
    assert "READY" in content.upper()


@pytest.mark.skipif(not settings.GEMINI_API_KEY, reason="GEMINI_API_KEY not configured")
def test_gemini_query_planner():
    """Verify Gemini query planner generates structured sub-queries."""
    state = ResearchState(user_query="Speculative Decoding for LLM inference acceleration", max_papers=2)
    result = plan_queries(state)
    
    assert "expanded_queries" in result
    assert len(result["expanded_queries"]) >= 2
    print("\nGemini Generated Queries:", result["expanded_queries"])


@pytest.mark.skipif(not settings.GEMINI_API_KEY, reason="GEMINI_API_KEY not configured")
def test_gemini_paper_extractor():
    """Verify Gemini extracts structured PaperAnalysis from academic text."""
    paper_meta = PaperMetadata(
        arxiv_id="2305.04388",
        title="Accelerating Large Language Model Decoding with Speculative Sampling",
        authors=["Charlie Chen", "Sebastian Borgeaud"],
        abstract="Speculative sampling is an algorithm to accelerate Transformer decoding...",
        published_date="2023-05-08T00:00:00Z",
        categories=["cs.CL"],
        primary_category="cs.CL",
        pdf_url="https://arxiv.org/pdf/2305.04388.pdf",
        arxiv_url="http://arxiv.org/abs/2305.04388",
    )

    parsed = ParsedPaper(
        metadata=paper_meta,
        full_markdown="",
        sections={
            "abstract": PaperSection(
                name="abstract",
                title="Abstract",
                content="Speculative sampling achieves 2.0x-2.5x speedup on Chinchilla 70B while preserving identical output distribution.",
            ),
            "methodology": PaperSection(
                name="methodology",
                title="Method",
                content="We sample K tokens from draft model q(x) and accept with probability min(1, p(x)/q(x)).",
            ),
        }
    )

    analysis = extract_single_paper(parsed)
    assert isinstance(analysis, PaperAnalysis)
    assert analysis.arxiv_id == "2305.04388"
    assert len(analysis.core_innovation) > 10
    print("\nGemini Extracted Innovation:", analysis.core_innovation)


@pytest.mark.skipif(not settings.GEMINI_API_KEY, reason="GEMINI_API_KEY not configured")
def test_gemini_synthesizer_and_fact_checker():
    """Verify Gemini synthesizes full Markdown report and fact-checks."""
    analysis = PaperAnalysis(
        arxiv_id="2305.04388",
        title="Accelerating Large Language Model Decoding with Speculative Sampling",
        core_innovation="Speculative draft-and-verify algorithm.",
        architecture_details="Small draft model samples K tokens verified by target model.",
        mathematical_formulation="$$P_{accept} = \\min(1, \\frac{p(x)}{q(x)})$$",
        benchmarks=[
            BenchmarkMetric(
                task_or_dataset="HumanEval",
                base_model="Chinchilla-70B",
                speedup_factor="2.4x",
                accuracy_delta="Lossless (Exact distribution matching)",
                hardware="TPUv4",
            )
        ],
        reproducibility_notes="Official open source implementations available.",
    )

    state = ResearchState(
        user_query="Speculative Decoding",
        paper_analyses=[analysis],
        retrieved_papers=[
            PaperMetadata(
                arxiv_id="2305.04388",
                title="Accelerating Large Language Model Decoding with Speculative Sampling",
                authors=["Charlie Chen"],
                abstract="Abstract text",
                published_date="2023-05-08T00:00:00Z",
                categories=["cs.CL"],
                primary_category="cs.CL",
                pdf_url="https://arxiv.org/pdf/2305.04388.pdf",
                arxiv_url="http://arxiv.org/abs/2305.04388",
                github_urls=["https://github.com/lucidrains/speculative-decoding"],
            )
        ]
    )

    synth_res = synthesize_research_report(state)
    assert "draft_report" in synth_res
    assert len(synth_res["draft_report"]) > 100

    state.draft_report = synth_res["draft_report"]
    fc_res = verify_and_refine_report(state)
    assert "final_report" in fc_res
    print("\nGemini Fact Check Notes:", fc_res.get("fact_check_notes"))
