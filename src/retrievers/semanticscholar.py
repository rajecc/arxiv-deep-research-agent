import re
from typing import Optional
import requests

from configs.settings import settings
from src.models.paper import PaperMetadata
from src.utils.logger import logger


class SemanticScholarClient:
    """Client for Semantic Scholar Graph API to enrich citation counts and TLDR summaries."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.USER_AGENT,
        })

    def get_paper_metrics(self, arxiv_id: str) -> Optional[dict]:
        """Fetch citation count, influential citations, and TLDR from Semantic Scholar."""
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        url = f"{self.BASE_URL}/ARXIV:{clean_id}"
        params = {
            "fields": "citationCount,influentialCitationCount,tldr,year"
        }

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                logger.debug(f"Semantic Scholar returned status {response.status_code} for {arxiv_id}")
                return None
        except Exception as e:
            logger.debug(f"Semantic Scholar API connection error for {arxiv_id}: {e}")
            return None

    def enrich_paper_metadata(self, paper: PaperMetadata) -> PaperMetadata:
        """Enrich PaperMetadata with Semantic Scholar citation counts and TLDR."""
        data = self.get_paper_metrics(paper.arxiv_id)
        if not data:
            return paper

        try:
            paper.citation_count = data.get("citationCount")
            paper.influential_citation_count = data.get("influentialCitationCount")
            
            tldr_obj = data.get("tldr")
            if tldr_obj and isinstance(tldr_obj, dict):
                paper.tldr = tldr_obj.get("text")

            if paper.citation_count is not None:
                logger.info(
                    f"Semantic Scholar for [blue]{paper.arxiv_id}[/blue]: "
                    f"📚 Citations: {paper.citation_count} (⚡ {paper.influential_citation_count or 0} influential)"
                )
        except Exception as e:
            logger.debug(f"Error enriching paper {paper.arxiv_id} from Semantic Scholar: {e}")

        return paper


# Global instance
semanticscholar_client = SemanticScholarClient()
