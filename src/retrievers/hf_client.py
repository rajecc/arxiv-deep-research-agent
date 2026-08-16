import re
from typing import Optional
import requests

from configs.settings import settings
from src.models.paper import PaperMetadata
from src.utils.logger import logger


class HuggingFacePapersClient:
    """Client for Hugging Face Papers API to fetch community upvotes, model repos, and code."""

    BASE_URL = "https://huggingface.co/api/papers"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.USER_AGENT,
        })

    def get_paper_details(self, arxiv_id: str) -> Optional[dict]:
        """Fetch paper metadata and community interactions from Hugging Face."""
        # Normalize arxiv_id
        clean_id = re.sub(r"v\d+$", "", arxiv_id.strip())
        url = f"{self.BASE_URL}/{clean_id}"

        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                # Paper not indexed yet on HF Daily Papers
                return None
            else:
                logger.debug(f"HF API returned status {response.status_code} for paper {arxiv_id}")
                return None
        except Exception as e:
            logger.debug(f"HF API connection failed for {arxiv_id}: {e}")
            return None

    def enrich_paper_metadata(self, paper: PaperMetadata) -> PaperMetadata:
        """Enrich existing PaperMetadata with Hugging Face data (upvotes, repos, models)."""
        data = self.get_paper_details(paper.arxiv_id)
        if not data:
            return paper

        try:
            paper.hf_upvotes = data.get("upvotes", 0)
            paper.hf_url = f"https://huggingface.co/papers/{paper.arxiv_id}"

            # Extract GitHub repositories if available
            github_repos = data.get("githubRepo")
            if github_repos:
                if isinstance(github_repos, list):
                    paper.github_urls.extend(github_repos)
                elif isinstance(github_repos, str):
                    paper.github_urls.append(github_repos)

            # Extract linked model checkpoints or datasets
            models = data.get("models", [])
            for m in models:
                if isinstance(m, dict) and "id" in m:
                    paper.model_urls.append(f"https://huggingface.co/{m['id']}")
                elif isinstance(m, str):
                    paper.model_urls.append(f"https://huggingface.co/{m}")

            # Deduplicate URLs
            paper.github_urls = list(set(paper.github_urls))
            paper.model_urls = list(set(paper.model_urls))

            if paper.hf_upvotes > 0:
                logger.info(
                    f"Enriched [blue]{paper.arxiv_id}[/blue] from HuggingFace: "
                    f"⭐ {paper.hf_upvotes} upvotes | 💻 {len(paper.github_urls)} repos | 🤖 {len(paper.model_urls)} models"
                )
        except Exception as e:
            logger.debug(f"Error enriching paper {paper.arxiv_id} from HF: {e}")

        return paper


# Global instance
hf_client = HuggingFacePapersClient()
