import re
from datetime import datetime
from typing import List, Optional
import arxiv

from configs.settings import settings
from src.models.paper import PaperMetadata
from src.utils.logger import logger


class ArxivClient:
    """Client for querying the ArXiv API and formatting results into PaperMetadata models."""

    def __init__(self, page_size: int = 20, delay_seconds: float = 3.0, num_retries: int = 3):
        self.client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    def _clean_text(self, text: str) -> str:
        """Clean multiline text, removing unnecessary line breaks and whitespaces."""
        if not text:
            return ""
        # Replace multiple spaces/newlines with single space
        return re.sub(r"\s+", " ", text).strip()

    def _parse_arxiv_id(self, raw_id_url: str) -> tuple[str, int]:
        """Extract clean arxiv ID and version from entry_id URL.
        
        Example: 'http://arxiv.org/abs/2401.12345v2' -> ('2401.12345', 2)
        """
        match = re.search(r"(\d{4}\.\d{4,5})(?:v(\d+))?", raw_id_url)
        if match:
            arxiv_id = match.group(1)
            version = int(match.group(2)) if match.group(2) else 1
            return arxiv_id, version
        
        # Fallback to last segment
        last_seg = raw_id_url.rstrip("/").split("/")[-1]
        v_match = re.search(r"v(\d+)$", last_seg)
        version = int(v_match.group(1)) if v_match else 1
        clean_id = re.sub(r"v\d+$", "", last_seg)
        return clean_id, version

    def _convert_result(self, result: arxiv.Result) -> PaperMetadata:
        """Convert an arxiv.Result object into our structured PaperMetadata model."""
        arxiv_id, version = self._parse_arxiv_id(result.entry_id)
        
        return PaperMetadata(
            arxiv_id=arxiv_id,
            version=version,
            title=self._clean_text(result.title),
            authors=[author.name for author in result.authors],
            abstract=self._clean_text(result.summary),
            published_date=result.published.isoformat() if result.published else "",
            updated_date=result.updated.isoformat() if result.updated else None,
            categories=list(result.categories),
            primary_category=result.primary_category or (result.categories[0] if result.categories else "cs.AI"),
            pdf_url=result.pdf_url,
            arxiv_url=result.entry_id,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi,
        )

    def search_papers(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        max_results: int = 5,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
        min_year: Optional[int] = None,
    ) -> List[PaperMetadata]:
        """Search ArXiv with advanced filtering by categories, relevance, and publication year."""
        logger.info(f"Searching ArXiv for query: '[bold]{query}[/bold]' (max_results={max_results})")
        
        # Build search query string
        active_cats = categories or settings.ARXIV_DEFAULT_CATEGORIES
        cat_filter = " OR ".join([f"cat:{cat}" for cat in active_cats])
        
        # Clean query tokens
        clean_query = query.strip()
        if cat_filter:
            full_query = f"({clean_query}) AND ({cat_filter})"
        else:
            full_query = clean_query

        search = arxiv.Search(
            query=full_query,
            max_results=max_results * 2 if min_year else max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        results: List[PaperMetadata] = []
        try:
            for r in self.client.results(search):
                meta = self._convert_result(r)
                
                # Optional year filter
                if min_year and meta.published_date:
                    try:
                        pub_year = int(meta.published_date[:4])
                        if pub_year < min_year:
                            continue
                    except ValueError:
                        pass
                
                results.append(meta)
                if len(results) >= max_results:
                    break
        except Exception as e:
            logger.error(f"Error fetching from ArXiv API: {e}")
            raise e

        logger.success(f"Found {len(results)} relevant papers on ArXiv")
        return results

    def get_papers_by_ids(self, arxiv_ids: List[str]) -> List[PaperMetadata]:
        """Fetch papers directly by their ArXiv IDs."""
        clean_ids = [re.sub(r"^(?:arxiv:)?", "", i.strip(), flags=re.I) for i in arxiv_ids]
        logger.info(f"Fetching specific papers by ID: {clean_ids}")
        
        search = arxiv.Search(id_list=clean_ids)
        results: List[PaperMetadata] = []
        
        for r in self.client.results(search):
            results.append(self._convert_result(r))
            
        return results


# Global singleton client
arxiv_client = ArxivClient()
