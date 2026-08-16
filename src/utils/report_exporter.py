import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from configs.settings import settings
from src.utils.logger import logger

REPORTS_DIR = settings.BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Sanitize query string to safe filename format."""
    clean = re.sub(r"[^\w\s\-]", "", name.lower())
    clean = re.sub(r"[\s\-]+", "_", clean).strip("_")
    return clean[:50]


def save_markdown_report(topic: str, content: str) -> Path:
    """Save generated markdown report to reports directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = sanitize_filename(topic)
    filename = f"report_{safe_topic}_{timestamp}.md"
    file_path = REPORTS_DIR / filename

    file_path.write_text(content, encoding="utf-8")
    logger.success(f"Saved research report to [bold]{file_path}[/bold]")
    return file_path
