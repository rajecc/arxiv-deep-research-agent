import re
from pathlib import Path
from typing import Optional, List
import requests
import pymupdf
import pymupdf4llm

from configs.settings import settings
from src.models.paper import PaperMetadata, ParsedPaper, PaperSection
from src.utils.logger import logger


class PDFParser:
    """Downloads academic PDFs and converts them into high-fidelity structured Markdown."""

    def __init__(
        self,
        pdf_cache_dir: Optional[Path] = None,
        parsed_cache_dir: Optional[Path] = None,
        timeout: int = 45,
    ):
        self.pdf_cache_dir = pdf_cache_dir or settings.PDF_CACHE_DIR
        self.parsed_cache_dir = parsed_cache_dir or settings.PARSED_CACHE_DIR
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.USER_AGENT,
        })

    def _get_pdf_cache_path(self, arxiv_id: str) -> Path:
        clean_id = re.sub(r"[^\w\-.]", "_", arxiv_id)
        return self.pdf_cache_dir / f"{clean_id}.pdf"

    def _get_parsed_cache_path(self, arxiv_id: str) -> Path:
        clean_id = re.sub(r"[^\w\-.]", "_", arxiv_id)
        return self.parsed_cache_dir / f"{clean_id}.md"

    def download_pdf(self, paper: PaperMetadata, force_download: bool = False) -> Path:
        """Download paper PDF to local cache directory if not already cached."""
        target_path = self._get_pdf_cache_path(paper.arxiv_id)

        if target_path.exists() and not force_download and target_path.stat().st_size > 1000:
            logger.debug(f"Using cached PDF for {paper.arxiv_id} at {target_path}")
            return target_path

        # ArXiv PDF URL format
        pdf_url = paper.pdf_url
        if not pdf_url.endswith(".pdf"):
            pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

        logger.info(f"Downloading PDF for [blue]{paper.arxiv_id}[/blue] from {pdf_url}...")

        response = self.session.get(pdf_url, stream=True, timeout=self.timeout)
        response.raise_for_status()

        # Verify PDF header
        content_chunk = next(response.iter_content(chunk_size=1024), b"")
        if not content_chunk.startswith(b"%PDF"):
            raise ValueError(f"Downloaded content for {paper.arxiv_id} is not a valid PDF file")

        with open(target_path, "wb") as f:
            f.write(content_chunk)
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)

        file_size_kb = target_path.stat().st_size / 1024
        logger.success(f"Downloaded PDF for {paper.arxiv_id} ({file_size_kb:.1f} KB)")
        return target_path

    def _extract_tables_from_markdown(self, markdown_text: str) -> List[str]:
        """Extract Markdown table blocks from text."""
        # Regex matching markdown tables with header and divider rows
        table_pattern = re.compile(
            r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)",
            re.MULTILINE
        )
        tables = table_pattern.findall(markdown_text)
        return [t.strip() for t in tables]

    def _extract_equations_from_markdown(self, markdown_text: str) -> List[str]:
        """Extract LaTeX equations from text ($$...$$ or \\begin{equation}...\\end{equation})."""
        equations: List[str] = []
        # Multi-line display equations
        eq_blocks = re.findall(r"(\$\$.*?\$\$)", markdown_text, re.DOTALL)
        latex_envs = re.findall(r"(\\begin\{(?:equation|align|gather)\*?\}.*?\\end\{(?:equation|align|gather)\*?\})", markdown_text, re.DOTALL)
        
        equations.extend([eq.strip() for eq in eq_blocks])
        equations.extend([eq.strip() for eq in latex_envs])
        return equations

    def _extract_code_urls(self, markdown_text: str) -> List[str]:
        """Extract GitHub/GitLab URLs from text."""
        urls = re.findall(r"https?://(?:www\.)?github\.com/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+", markdown_text)
        return list(set(urls))

    def parse_pdf_to_markdown(
        self,
        paper: PaperMetadata,
        force_reparse: bool = False,
    ) -> ParsedPaper:
        """Parse PDF document into high-fidelity markdown with tables and equations."""
        md_cache_path = self._get_parsed_cache_path(paper.arxiv_id)

        # Check if already parsed
        if md_cache_path.exists() and not force_reparse and md_cache_path.stat().st_size > 100:
            logger.debug(f"Loading parsed Markdown from cache for {paper.arxiv_id}")
            markdown_content = md_cache_path.read_text(encoding="utf-8")
            pdf_path = self._get_pdf_cache_path(paper.arxiv_id)
        else:
            # Download PDF
            pdf_path = self.download_pdf(paper)

            logger.info(f"Converting PDF to Markdown with PyMuPDF4LLM for [blue]{paper.arxiv_id}[/blue]...")
            
            # PyMuPDF4LLM high-level markdown conversion
            markdown_content = pymupdf4llm.to_markdown(
                str(pdf_path),
                show_progress=False,
            )

            # Clean up common artifacts
            markdown_content = self._clean_markdown(markdown_content)

            # Save to parsed cache
            md_cache_path.write_text(markdown_content, encoding="utf-8")
            logger.success(f"Parsed Markdown saved to cache ({len(markdown_content):,} chars)")

        # Extract tables, equations, code URLs
        tables = self._extract_tables_from_markdown(markdown_content)
        equations = self._extract_equations_from_markdown(markdown_content)
        code_urls = self._extract_code_urls(markdown_content)

        # Merge code URLs with paper metadata
        for url in code_urls:
            if url not in paper.github_urls:
                paper.github_urls.append(url)

        return ParsedPaper(
            metadata=paper,
            local_pdf_path=str(pdf_path) if pdf_path.exists() else None,
            local_markdown_path=str(md_cache_path),
            full_markdown=markdown_content,
            tables=tables,
            equations=equations,
            extracted_code_urls=code_urls,
        )

    def _clean_markdown(self, md: str) -> str:
        """Clean noise, multiple empty lines, and page headers."""
        md = re.sub(r"\n{3,}", "\n\n", md)
        md = "\n".join(line.rstrip() for line in md.splitlines())
        return md.strip()


# Global parser instance
pdf_parser = PDFParser()
