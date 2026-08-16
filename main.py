import argparse
import sys
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

from configs.settings import settings
from src.utils.logger import console, logger
from src.retrievers.arxiv_client import arxiv_client
from src.retrievers.hf_client import hf_client
from src.retrievers.semanticscholar import semanticscholar_client
from src.parsers.pdf_parser import pdf_parser
from src.parsers.section_splitter import section_splitter
from src.utils.llm_factory import llm_factory


def print_paper_summary_table(papers):
    """Print a rich summary table of retrieved and enriched papers."""
    table = Table(
        title="📚 Retrieved Academic Papers (ArXiv + HF + Semantic Scholar)",
        header_style="bold cyan",
        show_lines=True,
    )

    table.add_column("ArXiv ID", style="bold blue", width=12)
    table.add_column("Title & Authors", style="white", width=45)
    table.add_column("Date / Cat", style="dim", width=12)
    table.add_column("Impact / Links", style="green", width=25)

    for p in papers:
        # Title and authors
        authors_preview = ", ".join(p.authors[:2])
        if len(p.authors) > 2:
            authors_preview += f" +{len(p.authors) - 2}"
        title_author = f"[bold]{p.title}[/bold]\n[dim]{authors_preview}[/dim]"

        # Date and category
        pub_date = p.published_date[:10] if p.published_date else "N/A"
        date_cat = f"{pub_date}\n[yellow]{p.primary_category}[/yellow]"

        # Impact and links
        metrics = []
        if p.hf_upvotes > 0:
            metrics.append(f"⭐ HF: {p.hf_upvotes}")
        if p.citation_count is not None:
            metrics.append(f"📚 Citations: {p.citation_count}")
        if p.github_urls:
            metrics.append(f"💻 Repos: {len(p.github_urls)}")

        metrics_text = "\n".join(metrics) if metrics else "[dim]No stats yet[/dim]"

        table.add_row(p.arxiv_id, title_author, date_cat, metrics_text)

    console.print(table)


def run_pipeline(
    query: str,
    max_papers: int = 2,
    min_year: int = None,
    specific_id: str = None,
):
    logger.print_banner("ArXiv Deep-Research Agent — Phase 1 & 2 Runner")

    # Step 1: Retrieval
    if specific_id:
        papers = arxiv_client.get_papers_by_ids([specific_id])
    else:
        papers = arxiv_client.search_papers(
            query=query,
            max_results=max_papers,
            min_year=min_year,
        )

    if not papers:
        logger.warning("No papers found matching the query criteria.")
        return

    # Step 2: Enrichment (Hugging Face + Semantic Scholar)
    logger.info("Enriching papers with Hugging Face Daily Papers and Semantic Scholar metrics...")
    for p in papers:
        hf_client.enrich_paper_metadata(p)
        semanticscholar_client.enrich_paper_metadata(p)

    print_paper_summary_table(papers)

    # Step 3: PDF Download & Deep Parsing
    logger.print_banner("Deep PDF Parsing & Section Decomposition")
    parsed_papers = []

    for i, paper in enumerate(papers, 1):
        console.print(f"\n[bold magenta]━━━━━━━━━━ Paper [{i}/{len(papers)}]: {paper.title} ━━━━━━━━━━[/bold magenta]")
        
        # Parse PDF to markdown
        parsed = pdf_parser.parse_pdf_to_markdown(paper)
        
        # Segment into semantic sections
        parsed = section_splitter.split_sections(parsed)
        parsed_papers.append(parsed)

        # Display parsing report
        stats_panel = Panel(
            f"📄 [bold]Full Markdown Length:[/bold] {len(parsed.full_markdown):,} characters (~{len(parsed.full_markdown)//4:,} tokens)\n"
            f"📑 [bold]Sections Extracted:[/bold] {len(parsed.sections)} ({', '.join(parsed.sections.keys())})\n"
            f"📊 [bold]Tables Extracted:[/bold] {len(parsed.tables)}\n"
            f"🧮 [bold]Equations Extracted:[/bold] {len(parsed.equations)}\n"
            f"🔗 [bold]Code Repositories Found:[/bold] {', '.join(parsed.extracted_code_urls) or 'None'}\n"
            f"💾 [bold]Markdown Cache:[/bold] [dim]{parsed.local_markdown_path}[/dim]",
            title=f"🔬 Parsed Analysis: {paper.arxiv_id}",
            border_style="cyan",
        )
        console.print(stats_panel)

        # Preview Methodology or Experiments section if present
        target_section_key = next((k for k in ["methodology", "experiments", "abstract"] if k in parsed.sections), None)
        if target_section_key:
            sec = parsed.sections[target_section_key]
            preview_text = sec.content[:600] + ("..." if len(sec.content) > 600 else "")
            sec_panel = Panel(
                preview_text,
                title=f"📖 Section Preview: [{sec.name.upper()}] - {sec.title}",
                border_style="green",
            )
            console.print(sec_panel)

        # Preview Table if found
        if parsed.tables:
            table_preview = parsed.tables[0][:400] + ("..." if len(parsed.tables[0]) > 400 else "")
            console.print(Panel(table_preview, title="📊 Sample Extracted Table", border_style="yellow"))

    logger.success(
        f"Phase 1 & Phase 2 executed successfully for {len(parsed_papers)} papers! Ready for LangGraph Multi-Agent Synthesis."
    )


def main():
    parser = argparse.ArgumentParser(description="ArXiv Deep-Research Agent (Phase 1 & 2)")
    parser.add_argument(
        "--query",
        type=str,
        default="Speculative Decoding",
        help="Search topic or keywords (e.g. 'Speculative Decoding')",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=2,
        help="Maximum number of papers to fetch and parse",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2024,
        help="Minimum publication year (e.g. 2024, 2025)",
    )
    parser.add_argument(
        "--arxiv-id",
        type=str,
        default=None,
        help="Direct ArXiv ID to fetch and parse (e.g. '2305.04388')",
    )
    args = parser.parse_args()

    run_pipeline(
        query=args.query,
        max_papers=args.max_papers,
        min_year=args.min_year,
        specific_id=args.arxiv_id,
    )


if __name__ == "__main__":
    main()