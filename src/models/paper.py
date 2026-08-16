from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PaperMetadata(BaseModel):
    """Metadata for an academic paper from ArXiv and enriched sources."""

    arxiv_id: str = Field(description="Normalized ArXiv ID (e.g. '2401.01234')")
    version: int = Field(default=1, description="Version number of the paper on ArXiv")
    title: str = Field(description="Paper title without extraneous newlines")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field(description="Full text abstract")
    published_date: str = Field(description="ISO format publication date string")
    updated_date: Optional[str] = Field(default=None, description="ISO format last updated date")
    categories: list[str] = Field(default_factory=list, description="All ArXiv category tags")
    primary_category: str = Field(default="cs.AI", description="Primary category tag")
    pdf_url: str = Field(description="Direct URL to PDF download")
    arxiv_url: str = Field(description="URL to ArXiv abstract page")
    comment: Optional[str] = Field(default=None, description="Author comments / conference info")
    journal_ref: Optional[str] = Field(default=None, description="Journal reference if published")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")

    # Enriched fields from Hugging Face & Semantic Scholar
    hf_upvotes: int = Field(default=0, description="Hugging Face Daily Papers upvotes count")
    hf_url: Optional[str] = Field(default=None, description="Hugging Face paper page link")
    github_urls: list[str] = Field(default_factory=list, description="GitHub repository links found")
    model_urls: list[str] = Field(default_factory=list, description="Hugging Face model checkpoint URLs")
    citation_count: Optional[int] = Field(default=None, description="Total citation count from Semantic Scholar")
    influential_citation_count: Optional[int] = Field(
        default=None, description="Influential citation count"
    )
    tldr: Optional[str] = Field(default=None, description="One-sentence AI summary from Semantic Scholar")


class PaperSection(BaseModel):
    """A semantic section extracted from the parsed paper markdown."""

    name: str = Field(description="Normalized section key (e.g. 'abstract', 'methodology', 'experiments')")
    title: str = Field(description="Original section heading text")
    content: str = Field(description="Markdown content of this section")
    char_count: int = Field(default=0)
    token_estimate: int = Field(default=0)


class ParsedPaper(BaseModel):
    """Fully parsed paper representation ready for deep analysis and agentic synthesis."""

    metadata: PaperMetadata
    local_pdf_path: Optional[str] = None
    local_markdown_path: Optional[str] = None
    full_markdown: str = Field(description="Complete converted markdown content")
    sections: dict[str, PaperSection] = Field(
        default_factory=dict, description="Mapped sections by normalized key"
    )
    tables: list[str] = Field(default_factory=list, description="Markdown formatted tables extracted")
    equations: list[str] = Field(default_factory=list, description="LaTeX equations extracted")
    extracted_code_urls: list[str] = Field(
        default_factory=list, description="GitHub/GitLab links found within the paper text"
    )
