"""
OpenAI-Compatible Proxy Endpoints.
Allows external IDEs (Cursor, VS Code, Windsurf) and SDKs to route traffic through QNM
by setting base_url="http://127.0.0.1:8000/v1".
"""

import time
import uuid
from typing import List, Optional, Dict, Any, Union
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from qnm.pipeline.orchestrator import QNMOrchestrator


router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qnm-dual-bottleneck")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Embedded QNM efficiency telemetry
    raw_context_tokens: int
    compressed_input_tokens: int
    skeleton_output_tokens: int
    context_saved_percent: float
    generation_saved_percent: float


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


@router.get("/models")
async def list_models(request: Request):
    """List available QNM models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "qnm-dual-bottleneck",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qnm-engine"
            },
            {
                "id": "qnm-coder-agent",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qnm-engine"
            }
        ]
    }


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    """Drop-in OpenAI chat completions endpoint."""
    orchestrator: QNMOrchestrator = request.app.state.orchestrator

    # Extract conversation text from messages
    user_turns = [m.content for m in payload.messages if m.role == "user"]
    if not user_turns:
        raise HTTPException(status_code=400, detail="At least one user message is required.")

    current_prompt = user_turns[-1]
    session_id = payload.session_id or f"openai_proxy_{payload.messages[0].content[:12]}"

    result = orchestrator.process_query(
        prompt=current_prompt,
        session_id=session_id,
        model=payload.model
    )

    created_ts = int(time.time())
    response_id = f"chatcmpl-qnm-{uuid.uuid4().hex[:12]}"

    usage = UsageInfo(
        prompt_tokens=result.telemetry.compressed_input_tokens,
        completion_tokens=result.telemetry.final_output_tokens,
        total_tokens=result.telemetry.compressed_input_tokens + result.telemetry.final_output_tokens,
        raw_context_tokens=result.telemetry.raw_context_tokens,
        compressed_input_tokens=result.telemetry.compressed_input_tokens,
        skeleton_output_tokens=result.telemetry.skeleton_output_tokens,
        context_saved_percent=result.telemetry.context_saved_percent,
        generation_saved_percent=result.telemetry.generation_saved_percent
    )

    choice = ChatChoice(
        index=0,
        message=ChatMessage(role="assistant", content=result.expanded_response),
        finish_reason="stop"
    )

    return ChatCompletionResponse(
        id=response_id,
        created=created_ts,
        model=payload.model,
        choices=[choice],
        usage=usage
    )
