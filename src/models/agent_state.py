from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.models.paper import PaperMetadata, ParsedPaper


class BenchmarkMetric(BaseModel):
    """Specific benchmark or evaluation metric extracted from a paper."""

    task_or_dataset: str = Field(description="Dataset or benchmark name (e.g. 'GSM8k', 'HumanEval', 'MT-Bench')")
    base_model: str = Field(description="Base model evaluated (e.g. 'LLaMA-3-70B', 'Vicuna-13B')")
    speedup_factor: Optional[str] = Field(default=None, description="Reported speedup (e.g. '2.4x', '1.8x')")
    accuracy_delta: Optional[str] = Field(default=None, description="Accuracy or quality change (e.g. '+0.3%', 'lossless')")
    memory_overhead: Optional[str] = Field(default=None, description="VRAM / memory overhead (e.g. '+12% VRAM')")
    hardware: Optional[str] = Field(default=None, description="Hardware used (e.g. '1x A100-80GB', 'H100')")


class PaperAnalysis(BaseModel):
    """Structured deep technical analysis of a single academic paper."""

    arxiv_id: str = Field(description="ArXiv paper ID")
    title: str = Field(description="Full paper title")
    core_innovation: str = Field(description="Concise description of the key algorithmic/architectural contribution")
    architecture_details: str = Field(description="Detailed architecture breakdown: draft model, tree attention, KV-cache strategy")
    mathematical_formulation: Optional[str] = Field(default=None, description="Key LaTeX loss or probability formulas")
    hardware_and_environment: Optional[str] = Field(default=None, description="Hardware, batch size, context length used in experiments")
    benchmarks: List[BenchmarkMetric] = Field(default_factory=list, description="Extracted benchmark results")
    limitations: List[str] = Field(default_factory=list, description="Stated limitations, failure modes, or overheads")
    reproducibility_notes: str = Field(default="", description="Code repositories, checkpoints, and reproducibility assessment")


class ResearchPlan(BaseModel):
    """Output from the Query Planner Agent."""

    primary_topic: str
    expanded_queries: List[str] = Field(description="2-4 diverse search queries targeting complementary angles")
    focus_areas: List[str] = Field(description="Key architectural aspects to look for in papers")


class ResearchState(BaseModel):
    """Global state of the LangGraph Deep Research workflow."""

    user_query: str = Field(description="Initial user research topic")
    max_papers: int = Field(default=3, description="Maximum number of papers to analyze")
    min_year: Optional[int] = Field(default=2024, description="Minimum publication year")
    
    # Execution pipeline artifacts
    expanded_queries: List[str] = Field(default_factory=list)
    retrieved_papers: List[PaperMetadata] = Field(default_factory=list)
    parsed_papers: List[ParsedPaper] = Field(default_factory=list)
    paper_analyses: List[PaperAnalysis] = Field(default_factory=list)
    
    draft_report: Optional[str] = None
    fact_check_passed: bool = False
    fact_check_notes: Optional[str] = None
    final_report: Optional[str] = None
    saved_report_path: Optional[str] = None
    
    # Progress and status messages for UI streaming
    status_message: str = Field(default="Initializing research agent...")
