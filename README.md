# 🔬 ArXiv Deep-Research Agent

> **Autonomous Multi-Agent System for Deep Technical Research on Academic AI/ML Papers.**

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange)
![Parser](https://img.shields.io/badge/PDF_Parser-PyMuPDF4LLM-purple)

---

## 📌 Overview

**ArXiv Deep-Research Agent** is an autonomous AI agent designed for ML engineers and AI researchers. It automates literature review and deep technical comparative analysis:

1. **Multi-Source Retrieval**: Queries ArXiv, Hugging Face Daily Papers (community upvotes, model checkpoints, code repos), and Semantic Scholar (citations & TLDR).
2. **High-Fidelity Document Pipeline**: Converts multi-column academic PDFs to clean Markdown with preserved GitHub-style tables, LaTeX formulas, and auto-segmented sections (*Abstract*, *Architecture*, *Experiments*, *Limitations*).
3. **Multi-Agent Synthesis** (*LangGraph*): Performs parallel Map-Reduce analysis across papers, extracting loss formulas, hardware baselines, speedup benchmarks, and synthesizing an engineering Trade-offs Matrix.
4. **Fact-Checking & Citation Guard**: Cross-verifies benchmark metrics against the raw paper source to prevent hallucinations.

---

## 🏗 Project Architecture

```
arxiv/
├── configs/
│   └── settings.py             # Centralized settings (Gemini 3.7 Flash, VseLLM, ArXiv, Caches)
├── src/
│   ├── models/
│   │   └── paper.py            # Pydantic v2 schemas (PaperMetadata, ParsedPaper, Section)
│   ├── retrievers/
│   │   ├── arxiv_client.py     # ArXiv API search, category filtering & normalization
│   │   ├── hf_client.py        # Hugging Face Daily Papers & repository discovery
│   │   └── semanticscholar.py  # Semantic Scholar citations & TLDR integration
│   ├── parsers/
│   │   ├── pdf_parser.py       # High-fidelity PDF to Markdown converter with caching
│   │   └── section_splitter.py # Semantic section classifier (Methodology, Benchmarks, etc.)
│   └── utils/
│       ├── logger.py           # Rich-formatted terminal logging
│       └── llm_factory.py      # LLM Factory (Gemini 3.7 Flash / OpenAI-compatible / VseLLM)
├── tests/
│   ├── test_arxiv_client.py    # Unit tests for retrieval and enrichment
│   └── test_pdf_parser.py      # Unit tests for PDF parsing and section splitting
├── data/
│   └── cache/                  # Local cache for downloaded PDFs and parsed Markdown
├── .env.example
├── requirements.txt
└── main.py                     # Interactive CLI runner
```

---

## ⚡ Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/arxiv-deep-research-agent.git
cd arxiv-deep-research-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to configure your preferred LLM provider:
```ini
# Google Gemini (Default)
DEFAULT_LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.7-flash

# OR OpenAI-Compatible / VseLLM / OpenRouter
# DEFAULT_LLM_PROVIDER=openai_compatible
# OPENAI_API_KEY=your_vsellm_key
# OPENAI_BASE_URL=https://api.vllm.example.com/v1
# OPENAI_MODEL=deepseek-chat
```

### 3. Run Retrieval & Parsing CLI
```bash
# Search for papers on a specific topic
python main.py --query "Speculative Decoding" --max-papers 3

# Or parse a specific ArXiv paper directly by ID
python main.py --arxiv-id "2305.04388"
```

---

## 🧪 Testing

Run the test suite with `pytest`:
```bash
pytest tests/ -v
```

---

## 📄 License
MIT License.
