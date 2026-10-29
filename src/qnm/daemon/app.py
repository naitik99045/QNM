"""
FastAPI Main Application and Background Daemon.
Keeps NCM and reasoning models warm in memory for zero cold-start latency.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qnm.pipeline.orchestrator import QNMOrchestrator, PipelineResult
from qnm.daemon.openai_compat import router as openai_router
from qnm.settings import get_settings


class QueryRequest(BaseModel):
    prompt: str
    session_id: str = Field(default="default")
    file_attachments: Optional[List[str]] = Field(default=None)
    model: Optional[str] = Field(default=None)


class ClearRequest(BaseModel):
    session_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager keeping model instances hot in memory."""
    settings = get_settings()
    app.state.start_time = time.time()
    app.state.orchestrator = QNMOrchestrator()
    yield
    # Cleanup logic if any


app = FastAPI(
    title="QNM Engine Daemon",
    description="Quantized-Concept Network Model Persistent Daemon & OpenAI-Compatible Proxy",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)


@app.post("/qnm/query", response_model=PipelineResult)
async def query_pipeline(req: QueryRequest):
    """Native QNM CLI Query Endpoint with complete telemetry breakdown."""
    orchestrator: QNMOrchestrator = app.state.orchestrator
    try:
        result = orchestrator.process_query(
            prompt=req.prompt,
            session_id=req.session_id,
            file_attachments=req.file_attachments,
            model=req.model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qnm/health")
async def health_check():
    """System health, loaded models, and uptime status."""
    orchestrator: QNMOrchestrator = app.state.orchestrator
    uptime_sec = time.time() - getattr(app.state, "start_time", time.time())
    lmstudio_available = orchestrator.provider.is_available() if hasattr(orchestrator.provider, "is_available") else True

    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_sec, 1),
        "execution_mode": orchestrator.settings.execution_mode,
        "ncm_real_weights": orchestrator.compressor.loader.is_real_weights_loaded,
        "lmstudio_connected": lmstudio_available,
        "device": orchestrator.compressor.loader.device,
    }


@app.get("/qnm/stats")
async def get_lifetime_stats():
    """Cumulative token savings metrics from SQLite."""
    orchestrator: QNMOrchestrator = app.state.orchestrator
    return orchestrator.store.get_telemetry_stats()


@app.post("/qnm/clear")
async def clear_memory(req: ClearRequest):
    """Wipe session turns or complete memory store."""
    orchestrator: QNMOrchestrator = app.state.orchestrator
    orchestrator.store.clear_session(req.session_id)
    target = f"session '{req.session_id}'" if req.session_id else "all sessions"
    return {"status": "success", "message": f"Successfully cleared memory for {target}."}
