import sys
from rich.console import Console
from rich.theme import Theme

# Custom rich theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "highlight": "bold magenta",
    "paper": "bold blue",
})

console = Console(theme=custom_theme)


class AgentLogger:
    def __init__(self, name: str = "ArXivAgent"):
        self.name = name

    def info(self, message: str):
        console.print(f"[cyan]ℹ [[bold]{self.name}[/bold]][/cyan] {message}")

    def success(self, message: str):
        console.print(f"[green]✔ [[bold]{self.name}[/bold]][/green] {message}")

    def warning(self, message: str):
        console.print(f"[yellow]⚠ [[bold]{self.name}[/bold]][/yellow] {message}")

    def error(self, message: str):
        console.print(f"[red]✖ [[bold]{self.name}[/bold]][/red] {message}")

    def debug(self, message: str):
        console.print(f"[dim]🔍 [{self.name}] {message}[/dim]")

    def print_banner(self, title: str):
        console.rule(f"[bold magenta]{title}[/bold magenta]")


logger = AgentLogger()
