"""
ArXiv Deep-Research Agent — FastAPI Server
SSE-streaming research pipeline with Apple Dark Blue frontend.
"""
import asyncio
import json
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from configs.settings import settings
from src.models.agent_state import ResearchState
from src.agents.research_graph import research_agent_graph

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="ArXiv Deep-Research Agent",
    version="2.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response models ────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    max_papers: int = 2
    min_year: int = 2024
    provider: str = "openai_compatible"
    api_key: Optional[str] = None


# ── SSE streaming endpoint ────────────────────────────────────
@app.post("/api/research/stream")
async def research_stream(req: ResearchRequest, request: Request):
    """
    Streams Server-Sent Events while the LangGraph pipeline runs.
    Event types:
      step    — pipeline progress update  {node, message, step}
      papers  — retrieved PaperMetadata list (JSON)
      analyses— PaperAnalysis list (JSON)
      report  — final markdown report string
      done    — completion signal with metadata
      error   — error payload
    """
    async def generator() -> AsyncGenerator[dict, None]:
        # Optional API key override
        if req.api_key:
            if req.provider == "gemini":
                settings.GEMINI_API_KEY = req.api_key
            else:
                settings.OPENAI_API_KEY = req.api_key

        if req.provider:
            settings.DEFAULT_LLM_PROVIDER = req.provider

        initial_state = ResearchState(
            user_query=req.query,
            max_papers=req.max_papers,
            min_year=req.min_year,
        )

        accumulated: dict = {}
        step = 0

        try:
            # LangGraph is synchronous — run in thread to avoid blocking
            loop = asyncio.get_event_loop()

            def run_graph():
                results = []
                for event in research_agent_graph.stream(initial_state):
                    results.append(event)
                return results

            events = await loop.run_in_executor(None, run_graph)

            for event in events:
                if await request.is_disconnected():
                    break

                for node_name, node_output in event.items():
                    step += 1
                    accumulated.update(node_output)
                    msg = node_output.get("status_message", f"Completed {node_name}")

                    yield {
                        "event": "step",
                        "data": json.dumps({
                            "step": step,
                            "node": node_name,
                            "message": msg,
                        }),
                    }
                    await asyncio.sleep(0)  # yield control

            # Stream results once done
            papers = accumulated.get("retrieved_papers", [])
            analyses = accumulated.get("paper_analyses", [])
            report = accumulated.get("final_report") or accumulated.get("draft_report") or ""
            fact_check = accumulated.get("fact_check_passed", False)
            saved = accumulated.get("saved_report_path", "")

            yield {
                "event": "papers",
                "data": json.dumps([
                    p.model_dump() if hasattr(p, "model_dump") else p
                    for p in papers
                ]),
            }
            await asyncio.sleep(0)

            yield {
                "event": "analyses",
                "data": json.dumps([
                    a.model_dump() if hasattr(a, "model_dump") else a
                    for a in analyses
                ]),
            }
            await asyncio.sleep(0)

            yield {
                "event": "report",
                "data": json.dumps({"content": report}),
            }
            await asyncio.sleep(0)

            yield {
                "event": "done",
                "data": json.dumps({
                    "paper_count": len(papers),
                    "analysis_count": len(analyses),
                    "fact_check_passed": fact_check,
                    "saved_path": saved,
                    "steps": step,
                }),
            }

        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(exc)}),
            }

    return EventSourceResponse(generator())


# ── Settings info endpoint ────────────────────────────────────
@app.get("/api/settings")
async def get_settings():
    return {
        "provider": settings.DEFAULT_LLM_PROVIDER,
        "gemini_model": settings.GEMINI_MODEL,
        "openai_model": settings.OPENAI_MODEL,
        "openai_base_url": settings.OPENAI_BASE_URL,
        "has_gemini_key": bool(settings.GEMINI_API_KEY),
        "has_openai_key": bool(settings.OPENAI_API_KEY),
    }


# ── Serve frontend ────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found — place index.html in static/</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
