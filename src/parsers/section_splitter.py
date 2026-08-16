import re
from typing import Dict, List, Tuple
from src.models.paper import ParsedPaper, PaperSection
from src.utils.logger import logger


class SectionSplitter:
    """Splits full paper Markdown into categorized semantic sections."""

    # Robust regex rules for classifying section titles to canonical keys
    SECTION_RULES: List[Tuple[str, re.Pattern]] = [
        ("abstract", re.compile(r"\babstract\b", re.IGNORECASE)),
        ("introduction", re.compile(r"\b(?:introduction|overview|background)\b", re.IGNORECASE)),
        ("related_work", re.compile(r"\b(?:related\s+work|prior\s+work|literature\s+review)\b", re.IGNORECASE)),
        ("methodology", re.compile(r"\b(?:methodology|method|architecture|framework|approach|model|formulation|algorithm|proposed\s+\w+)\b", re.IGNORECASE)),
        ("experiments", re.compile(r"\b(?:experiments|experimental|evaluation|results|benchmarks|empirical|ablation)\b", re.IGNORECASE)),
        ("limitations", re.compile(r"\b(?:limitations|broader\s+impact|ethics|failure\s+modes)\b", re.IGNORECASE)),
        ("conclusion", re.compile(r"\b(?:conclusion|concluding\s+remarks|summary)\b", re.IGNORECASE)),
        ("appendix", re.compile(r"\b(?:appendix|supplementary|supplemental)\b", re.IGNORECASE)),
        ("references", re.compile(r"\b(?:references|bibliography)\b", re.IGNORECASE)),
    ]

    def _classify_heading(self, heading_title: str) -> str:
        """Classify heading into a canonical section name."""
        clean_title = heading_title.strip()
        for key, pattern in self.SECTION_RULES:
            if pattern.search(clean_title):
                return key
        return "other"

    def split_sections(self, parsed_paper: ParsedPaper) -> ParsedPaper:
        """Parse markdown and populate parsed_paper.sections with classified chunks."""
        markdown = parsed_paper.full_markdown
        if not markdown:
            return parsed_paper

        # Regex to find markdown headers (# Header, ## Header, etc.)
        header_pattern = re.compile(r"^(#{1,4}\s+[^\n]+)", re.MULTILINE)
        splits = header_pattern.split(markdown)

        raw_sections: List[Tuple[str, str]] = []

        # If there is content before the first header, it's usually the Abstract or Header block
        if splits and splits[0].strip():
            raw_sections.append(("Abstract / Preamble", splits[0].strip()))

        # Iterate header and content pairs
        for i in range(1, len(splits), 2):
            heading = splits[i].strip()
            body = splits[i + 1].strip() if (i + 1) < len(splits) else ""
            raw_sections.append((heading, body))

        classified_sections: Dict[str, PaperSection] = {}

        # If abstract wasn't explicitly found in headers, check metadata abstract
        if parsed_paper.metadata.abstract:
            classified_sections["abstract"] = PaperSection(
                name="abstract",
                title="Abstract",
                content=parsed_paper.metadata.abstract,
                char_count=len(parsed_paper.metadata.abstract),
                token_estimate=len(parsed_paper.metadata.abstract) // 4,
            )

        for heading, body in raw_sections:
            clean_heading = re.sub(r"^#+\s*", "", heading).strip()
            canonical_key = self._classify_heading(clean_heading)

            # Skip pure reference lists from taking up huge context
            if canonical_key == "references":
                continue

            # If section already exists (e.g. multi-part methodology), append
            if canonical_key in classified_sections:
                existing = classified_sections[canonical_key]
                combined_content = f"{existing.content}\n\n### {clean_heading}\n{body}".strip()
                existing.content = combined_content
                existing.char_count = len(combined_content)
                existing.token_estimate = len(combined_content) // 4
            else:
                classified_sections[canonical_key] = PaperSection(
                    name=canonical_key,
                    title=clean_heading,
                    content=body,
                    char_count=len(body),
                    token_estimate=len(body) // 4,
                )

        parsed_paper.sections = classified_sections
        logger.info(
            f"Segmented [blue]{parsed_paper.metadata.arxiv_id}[/blue] into "
            f"{len(classified_sections)} semantic sections: {list(classified_sections.keys())}"
        )
        return parsed_paper


# Global section splitter instance
section_splitter = SectionSplitter()
