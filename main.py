import argparse
import sys
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from configs.settings import settings
from src.models.agent_state import ResearchState
from src.agents.research_graph import research_agent_graph
from src.utils.logger import console, logger


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
        authors_preview = ", ".join(p.authors[:2])
        if len(p.authors) > 2:
            authors_preview += f" +{len(p.authors) - 2}"
        title_author = f"[bold]{p.title}[/bold]\n[dim]{authors_preview}[/dim]"
        pub_date = p.published_date[:10] if p.published_date else "N/A"
        date_cat = f"{pub_date}\n[yellow]{p.primary_category}[/yellow]"

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


def run_deep_research(
    query: str,
    max_papers: int = 3,
    min_year: int = 2024,
):
    logger.print_banner(f"ArXiv Deep-Research Agent: '{query}'")

    initial_state = ResearchState(
        user_query=query,
        max_papers=max_papers,
        min_year=min_year,
    )

    # Execute LangGraph workflow with streaming steps
    final_state_dict = None
    step_num = 1

    for event in research_agent_graph.stream(initial_state):
        for node_name, node_output in event.items():
            msg = node_output.get("status_message", f"Completed {node_name}")
            console.print(f"[bold cyan]▶ Step {step_num}: [{node_name}][/bold cyan] {msg}")
            step_num += 1
            final_state_dict = node_output

    # Fetch compiled final state
    final_output = research_agent_graph.invoke(initial_state)

    if final_output.get("retrieved_papers"):
        print_paper_summary_table(final_output["retrieved_papers"])

    final_report = final_output.get("final_report") or final_output.get("draft_report") or "No report produced."
    saved_path = final_output.get("saved_report_path", "reports/")

    # Display final report in rich markdown
    logger.print_banner("Final Deep-Research Report")
    console.print(Markdown(final_report))

    console.print(
        Panel(
            f"🎉 [bold green]Research complete![/bold green]\n"
            f"📁 [bold]Report File:[/bold] [cyan]{saved_path}[/cyan]\n"
            f"📊 [bold]Analyzed Papers:[/bold] {len(final_output.get('paper_analyses', []))}\n"
            f"💡 [bold]Fact-Check Status:[/bold] {'[green]PASSED[/green]' if final_output.get('fact_check_passed') else '[yellow]PASSED WITH NOTES[/yellow]'}",
            title="✅ Research Summary",
            border_style="green",
        )
    )


def main():
    parser = argparse.ArgumentParser(description="ArXiv Deep-Research Multi-Agent System")
    parser.add_argument(
        "--query",
        type=str,
        default="Speculative Decoding in Large Language Models",
        help="Research topic or query",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=2,
        help="Number of papers to deeply analyze",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2024,
        help="Minimum publication year",
    )
    args = parser.parse_args()

    run_deep_research(
        query=args.query,
        max_papers=args.max_papers,
        min_year=args.min_year,
    )


if __name__ == "__main__":
    main()