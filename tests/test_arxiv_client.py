import pytest
from src.retrievers.arxiv_client import ArxivClient
from src.retrievers.hf_client import HuggingFacePapersClient
from src.retrievers.semanticscholar import SemanticScholarClient
from src.models.paper import PaperMetadata


def test_arxiv_id_parsing():
    client = ArxivClient()
    
    # Test standard versioned URL
    arxiv_id, version = client._parse_arxiv_id("http://arxiv.org/abs/2305.04388v2")
    assert arxiv_id == "2305.04388"
    assert version == 2

    # Test unversioned
    arxiv_id, version = client._parse_arxiv_id("http://arxiv.org/abs/2401.12345")
    assert arxiv_id == "2401.12345"
    assert version == 1


def test_arxiv_search():
    client = ArxivClient()
    # Search for a known classic paper or query
    results = client.search_papers(query="Speculative Decoding", max_results=2)
    
    assert len(results) > 0
    paper = results[0]
    assert isinstance(paper, PaperMetadata)
    assert paper.arxiv_id != ""
    assert paper.title != ""
    assert len(paper.authors) > 0
    assert paper.abstract != ""
    assert paper.pdf_url.startswith("http")


def test_hf_and_s2_enrichment():
    # Test with standard speculative decoding paper ID: 2305.04388 (Speculative Sampling)
    paper = PaperMetadata(
        arxiv_id="2305.04388",
        title="Accelerating Large Language Model Decoding with Speculative Sampling",
        authors=["Charlie Chen"],
        abstract="Speculative sampling is an algorithm...",
        published_date="2023-05-08T00:00:00Z",
        categories=["cs.CL", "cs.AI"],
        primary_category="cs.CL",
        pdf_url="https://arxiv.org/pdf/2305.04388.pdf",
        arxiv_url="http://arxiv.org/abs/2305.04388",
    )

    hf_client = HuggingFacePapersClient()
    enriched_paper = hf_client.enrich_paper_metadata(paper)
    assert isinstance(enriched_paper, PaperMetadata)

    s2_client = SemanticScholarClient()
    s2_enriched = s2_client.enrich_paper_metadata(enriched_paper)
    assert isinstance(s2_enriched, PaperMetadata)
