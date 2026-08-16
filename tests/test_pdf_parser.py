import pytest
from src.parsers.pdf_parser import PDFParser
from src.parsers.section_splitter import SectionSplitter
from src.models.paper import PaperMetadata, ParsedPaper


def test_markdown_extractions():
    parser = PDFParser()

    sample_md = """
# Introduction
Large Language Models have revolutionized NLP. Check our code at https://github.com/deepseek-ai/DeepSeek-V3 for details.

| Model | Speedup | Memory |
|---|---|---|
| Baseline | 1.0x | 16GB |
| Speculative | 2.4x | 18GB |

Here is the loss formulation:
$$
\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{spec} + \\beta \\mathcal{L}_{base}
$$
"""
    # Test table extraction
    tables = parser._extract_tables_from_markdown(sample_md)
    assert len(tables) == 1
    assert "Model" in tables[0]
    assert "Speculative" in tables[0]

    # Test equation extraction
    equations = parser._extract_equations_from_markdown(sample_md)
    assert len(equations) == 1
    assert "\\mathcal{L}_{total}" in equations[0]

    # Test code url extraction
    code_urls = parser._extract_code_urls(sample_md)
    assert len(code_urls) == 1
    assert code_urls[0] == "https://github.com/deepseek-ai/DeepSeek-V3"


def test_section_splitting():
    splitter = SectionSplitter()
    
    paper_meta = PaperMetadata(
        arxiv_id="2401.99999",
        title="Test Paper",
        authors=["Author One"],
        abstract="This is the abstract text.",
        published_date="2024-01-01T00:00:00Z",
        categories=["cs.AI"],
        primary_category="cs.AI",
        pdf_url="https://arxiv.org/pdf/2401.99999.pdf",
        arxiv_url="http://arxiv.org/abs/2401.99999",
    )

    sample_md = """
# Introduction
Here is the introduction to the problem.

# 2. Proposed Architecture & Methodology
We introduce a speculative multi-head draft network.

# 3. Experiments and Benchmarks
We evaluate on GSM8k and HumanEval datasets.

# 4. Conclusion
In conclusion, speculative decoding is effective.
"""
    parsed = ParsedPaper(
        metadata=paper_meta,
        full_markdown=sample_md,
    )

    split_result = splitter.split_sections(parsed)

    assert "abstract" in split_result.sections
    assert "introduction" in split_result.sections
    assert "methodology" in split_result.sections
    assert "experiments" in split_result.sections
    assert "conclusion" in split_result.sections
    assert split_result.sections["methodology"].token_estimate > 0
