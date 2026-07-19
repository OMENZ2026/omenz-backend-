import os
import time
import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memory import (
    archive_memory,
    create_memory,
    get_memory,
    list_memories,
    memory_health,
    revise_memory,
    search_memory,
)
from router import (
    ANTHROPIC,
    OPENAI,
    PERPLEXITY,
    choose_available,
    provider_name,
)
from telemetry import telemetry_record


# ==========================================================
# OMENZ BACKEND v0.3.5
# Router + provider execution + memory + full telemetry
# ==========================================================
#
# Preserved:
# - Root and health endpoints
# - Patreon callback
# - Environment validation
# - PENI / OpenAI execution
# - AUDE / Anthropic execution
# - XITY / Perplexity execution
# - Router selection
# - Provider validation endpoints
# - Controlled memory operations
# - run_id, latency, and token reporting
#
# Added:
# - Structured router-decision telemetry
# - Structured provider-call telemetry
# - Provider success, failure, and exception events
# - Memory-operation telemetry
# - Shared run_id and correlation_id tracking
#
# Important:
# - Memory remains in-process and temporary.
# - Records reset whenever the Cloud Run instance restarts.
# - No agents or autonomous memory promotion.
# - No database or persistent storage yet.
# - Prompt content is not emitted into telemetry.
# ==========================================================


BACKEND_VERSION = "0.3.5"

app = FastAPI(
    title="OMENZ Backend",
    version=BACKEND_VERSION,
)


# ==========================================================
# REQUEST CONTRACTS
# ==========================================================


class RouteRequest(BaseModel):
    task_type: str = Field(
        ...,
        description=(
            "Task category such as chat, code, execution, "
            "reasoning, analysis, research, or search."
        ),
    )
    message: str = Field(
        ...,
        min_length=1,
        description=(
            "The user request that the selected provider "
            "should process."
        ),
    )


class MemoryCreateRequest(BaseModel):
    memory_type: str = Field(
        ...,
        description=(
            "Allowed values: fact, preference, task_state, "
            "system_event, or suggestion."
        ),
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The information to store.",
    )
    source: str = Field(
        ...,
        min_length=1,
        description="The source or provenance of the information.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 through 1.0.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional request or workflow run identifier.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured metadata.",
    )


class MemoryReviseRequest(BaseModel):
    content: str = Field(
        ...,
        min_length=1,
        description="The corrected or revised content.",
    )
    source: str = Field(
        ...,
        min_length=1,
        description="The source of the revision.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional revised confidence score. "
            "The original score is retained when omitted."
        ),
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional request or workflow run identifier.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to merge into the revision.",
    )


class MemoryArchiveRequest(BaseModel):
    reason: str = Field(
        ...,
        min_length=1,
        description="Reason the active memory is being archived.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional request or workflow run identifier.",
    )


# ==========================================================
# SHARED HELPERS
# ==========================================================


def event_id() -> str:
    return str(uuid.uuid4())


def now_ms(start: float) -> int:
    return round((time.time() - start) * 1000)


def resolve_run_id(request_run_id: Optional[str] = None) -> str:
    return request_run_id or event_id()


def memory_error_response(
    error: Exception,
    default_status_code: int = 400,
):
    if isinstance(error, KeyError):
        message = str(error).strip("'")
        status_code = 404
        error_type = "not_found"

    elif isinstance(error, ValueError):
        message = str(error)
        status_code = default_status_code
        error_type = "validation_error"

    else:
        message = str(error)
        status_code = 500
        error_type = "memory_exception"

    return JSONResponse(
        {
            "status": "error",
            "component": "memory",
            "error_type": error_type,
            "error": message,
        },
        status_code=status_code,
    )


def emit_memory_telemetry(
    *,
    event_type: str,
    status: str,
    start: float,
    run_id_value: str,
    task_type: str,
    error: Optional[str] = None,
):
    return telemetry_record(
        provider="memory",
        provider_name="OMENZ_MEMORY",
        model="in_process",
        status=status,
        start=start,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        error=error,
        task_type=task_type,
        message=None,
        event_type=event_type,
        run_id_value=run_id_value,
        correlation_id=run_id_value,
    )


# ==========================================================
# PROVIDER USAGE HELPERS
# ==========================================================


def openai_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}

    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get(
        "total_tokens",
        tokens_in + tokens_out,
    ) or 0

    return tokens_in, tokens_out, total_tokens


def anthropic_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}

    tokens_in = usage.get("input_tokens", 0) or 0
    tokens_out = usage.get("output_tokens", 0) or 0
    total_tokens = tokens_in + tokens_out

    return tokens_in, tokens_out, total_tokens


def perplexity_usage(data):
    usage = data.get("usage", {}) if isinstance(data, dict) else {}

    tokens_in = usage.get("prompt_tokens", 0) or 0
    tokens_out = usage.get("completion_tokens", 0) or 0
    total_tokens = usage.get(
        "total_tokens",
        tokens_in + tokens_out,
    ) or 0

    return tokens_in, tokens_out, total_tokens


# ==========================================================
# PROVIDER EXECUTION
# ==========================================================


async def call_openai(
    message,
    run_id,
    start,
    task_type=None,
):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        telemetry_record(
            provider=OPENAI,
            provider_name=provider_name(OPENAI),
            model=model,
            status="missing_key",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error="OPENAI_API_KEY is not configured.",
            task_type=task_type,
            message=None,
            event_type="provider_call",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": OPENAI,
            "provider_name": provider_name(OPENAI),
            "status": "missing_key",
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }, 500

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are PENI inside the OMENZ stack."
                            ),
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                },
            )

        data = response.json()
        tokens_in, tokens_out, total_tokens = openai_usage(data)

        status = (
            "ok"
            if response.status_code == 200
            else "error"
        )

        error_text = None

        if response.status_code != 200:
            error_text = str(data)

        telemetry_record(
            provider=OPENAI,
            provider_name=provider_name(OPENAI),
            model=model,
            status=status,
            start=start,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            error=error_text,
            task_type=task_type,
            message=None,
            event_type="provider_call",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": OPENAI,
            "provider_name": provider_name(OPENAI),
            "status": status,
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "cost_usd": 0.0,
            "reply": (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            ),
            "raw": (
                data
                if response.status_code != 200
                else None
            ),
        }, response.status_code

    except Exception as error:
        telemetry_record(
            provider=OPENAI,
            provider_name=provider_name(OPENAI),
            model=model,
            status="exception",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error=str(error),
            task_type=task_type,
            message=None,
            event_type="provider_exception",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": OPENAI,
            "provider_name": provider_name(OPENAI),
            "status": "exception",
            "error": str(error),
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }, 500


async def call_anthropic(
    message,
    run_id,
    start,
    task_type=None,
):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv(
        "ANTHROPIC_MODEL",
        "claude-opus-4-6",
    )

    if not api_key:
        telemetry_record(
            provider=ANTHROPIC,
            provider_name=provider_name(ANTHROPIC),
            model=model,
            status="missing_key",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error="ANTHROPIC_API_KEY is not configured.",
            task_type=task_type,
            message=None,
            event_type="provider_call",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": ANTHROPIC,
            "provider_name": provider_name(ANTHROPIC),
            "status": "missing_key",
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }, 500

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "user",
                            "content": message,
                        }
                    ],
                },
            )

        data = response.json()

        text = None

        if (
            isinstance(data.get("content"), list)
            and data["content"]
        ):
            text = data["content"][0].get("text")

        tokens_in, tokens_out, total_tokens = (
            anthropic_usage(data)
        )

        status = (
            "ok"
            if response.status_code == 200
            else "error"
        )

        error_text = None

        if response.status_code != 200:
            error_text = str(data)

        telemetry_record(
            provider=ANTHROPIC,
            provider_name=provider_name(ANTHROPIC),
            model=model,
            status=status,
            start=start,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            error=error_text,
            task_type=task_type,
            message=None,
            event_type="provider_call",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": ANTHROPIC,
            "provider_name": provider_name(ANTHROPIC),
            "status": status,
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "cost_usd": 0.0,
            "reply": text,
            "raw": (
                data
                if response.status_code != 200
                else None
            ),
        }, response.status_code

    except Exception as error:
        telemetry_record(
            provider=ANTHROPIC,
            provider_name=provider_name(ANTHROPIC),
            model=model,
            status="exception",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error=str(error),
            task_type=task_type,
            message=None,
            event_type="provider_exception",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": ANTHROPIC,
            "provider_name": provider_name(ANTHROPIC),
            "status": "exception",
            "error": str(error),
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }, 500


async def call_perplexity(
    message,
    run_id,
    start,
    task_type=None,
):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    model = os.getenv("PERPLEXITY_MODEL", "sonar")

    if not api_key:
        telemetry_record(
            provider=PERPLEXITY,
            provider_name=provider_name(PERPLEXITY),
            model=model,
            status="missing_key",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error="PERPLEXITY_API_KEY is not configured.",
            task_type=task_type,
            message=None,
            event_type="provider_call",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": PERPLEXITY,
            "provider_name": provider_name(PERPLEXITY),
            "status": "missing_key",
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }, 500

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are XITY inside the OMENZ stack."
                            ),
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                },
            )

        data = response.json()
        tokens_in, tokens_out, total_tokens = (
            perplexity_usage(data)
        )

        status = (
            "ok"
            if response.status_code == 200
            else "error"
        )

        error_text = None

        if response.status_code != 200:
            error_text = str(data)

        telemetry_record(
            provider=PERPLEXITY,
            provider_name=provider_name(PERPLEXITY),
            model=model,
            status=status,
            start=start,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            error=error_text,
            task_type=task_type,
            message=None,
            event_type="provider_call",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": PERPLEXITY,
            "provider_name": provider_name(PERPLEXITY),
            "status": status,
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "cost_usd": 0.0,
            "reply": (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            ),
            "raw": (
                data
                if response.status_code != 200
                else None
            ),
        }, response.status_code

    except Exception as error:
        telemetry_record(
            provider=PERPLEXITY,
            provider_name=provider_name(PERPLEXITY),
            model=model,
            status="exception",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error=str(error),
            task_type=task_type,
            message=None,
            event_type="provider_exception",
            run_id_value=run_id,
            correlation_id=run_id,
        )

        return {
            "provider": PERPLEXITY,
            "provider_name": provider_name(PERPLEXITY),
            "status": "exception",
            "error": str(error),
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }, 500


async def execute_provider(
    provider,
    message,
    run_id,
    start,
    task_type=None,
):
    if provider == OPENAI:
        return await call_openai(
            message=message,
            run_id=run_id,
            start=start,
            task_type=task_type,
        )

    if provider == ANTHROPIC:
        return await call_anthropic(
            message=message,
            run_id=run_id,
            start=start,
            task_type=task_type,
        )

    if provider == PERPLEXITY:
        return await call_perplexity(
            message=message,
            run_id=run_id,
            start=start,
            task_type=task_type,
        )

    telemetry_record(
        provider=str(provider),
        provider_name=provider_name(provider),
        model="unknown",
        status="routing_error",
        start=start,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        error="No available provider could be selected.",
        task_type=task_type,
        message=None,
        event_type="routing_error",
        run_id_value=run_id,
        correlation_id=run_id,
    )

    return {
        "provider": provider,
        "provider_name": provider_name(provider),
        "status": "routing_error",
        "error": "No available provider could be selected.",
        "run_id": run_id,
        "latency_ms": now_ms(start),
        "tokens_in": 0,
        "tokens_out": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }, 503


# ==========================================================
# ROOT AND HEALTH
# ==========================================================


@app.get("/")
def root():
    return {
        "service": "OMENZ Backend",
        "status": "online",
        "message": "OMENZ Stack backend is running.",
        "version": BACKEND_VERSION,
        "router": "online",
        "memory": "online",
        "telemetry": "online",
        "route_endpoint": "/route",
        "memory_endpoints": {
            "health": "/memory/health",
            "create": "/memory/create",
            "get": "/memory/get/{memory_id}",
            "list": "/memory/list",
            "search": "/memory/search",
            "revise": "/memory/revise/{memory_id}",
            "archive": "/memory/archive/{memory_id}",
        },
    }


@app.get("/health")
def health():
    current_memory_health = memory_health()

    return {
        "status": "ok",
        "version": BACKEND_VERSION,
        "router": "online",
        "telemetry": "online",
        "memory": current_memory_health["status"],
        "memory_storage": current_memory_health["storage"],
        "memory_persistent": current_memory_health["persistent"],
    }


# ==========================================================
# MEMORY ENDPOINTS
# ==========================================================


@app.get("/memory/health")
def get_memory_health():
    start = time.time()
    current_run_id = event_id()

    try:
        result = memory_health()

        emit_memory_telemetry(
            event_type="memory_health",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_health",
        )

        return {
            **result,
            "run_id": current_run_id,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_health",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_health",
            error=str(error),
        )

        return memory_error_response(error)


@app.post("/memory/create")
def create_memory_record(request: MemoryCreateRequest):
    start = time.time()
    current_run_id = resolve_run_id(request.run_id)

    try:
        record = create_memory(
            memory_type=request.memory_type,
            content=request.content,
            source=request.source,
            confidence=request.confidence,
            run_id=current_run_id,
            metadata=request.metadata,
        )

        emit_memory_telemetry(
            event_type="memory_create",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type=request.memory_type,
        )

        return {
            "status": "ok",
            "operation": "memory_created",
            "run_id": current_run_id,
            "record": record,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_create",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type=request.memory_type,
            error=str(error),
        )

        return memory_error_response(error)


@app.get("/memory/get/{memory_id}")
def get_memory_record(
    memory_id: str,
    include_inactive: bool = False,
):
    start = time.time()
    current_run_id = event_id()

    try:
        record = get_memory(
            memory_id=memory_id,
            include_inactive=include_inactive,
        )

        if record is None:
            emit_memory_telemetry(
                event_type="memory_get",
                status="not_found",
                start=start,
                run_id_value=current_run_id,
                task_type="memory_read",
                error="Memory record was not found.",
            )

            return JSONResponse(
                {
                    "status": "not_found",
                    "operation": "memory_read",
                    "memory_id": memory_id,
                    "include_inactive": include_inactive,
                    "run_id": current_run_id,
                },
                status_code=404,
            )

        emit_memory_telemetry(
            event_type="memory_get",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_read",
        )

        return {
            "status": "ok",
            "operation": "memory_read",
            "run_id": current_run_id,
            "record": record,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_get",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_read",
            error=str(error),
        )

        return memory_error_response(error)


@app.get("/memory/list")
def list_memory_records(
    memory_type: Optional[str] = None,
    status: Optional[str] = "active",
    limit: int = 100,
):
    start = time.time()
    current_run_id = event_id()

    try:
        records = list_memories(
            memory_type=memory_type,
            status=status,
            limit=limit,
        )

        emit_memory_telemetry(
            event_type="memory_list",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type=memory_type or "all",
        )

        return {
            "status": "ok",
            "operation": "memory_listed",
            "run_id": current_run_id,
            "result_count": len(records),
            "records": records,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_list",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type=memory_type or "all",
            error=str(error),
        )

        return memory_error_response(error)


@app.get("/memory/search")
def search_memory_records(
    query: str,
    memory_type: Optional[str] = None,
    limit: int = 20,
):
    start = time.time()
    current_run_id = event_id()

    try:
        records = search_memory(
            query=query,
            memory_type=memory_type,
            limit=limit,
        )

        emit_memory_telemetry(
            event_type="memory_search",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type=memory_type or "all",
        )

        return {
            "status": "ok",
            "operation": "memory_searched",
            "run_id": current_run_id,
            "query": query,
            "result_count": len(records),
            "records": records,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_search",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type=memory_type or "all",
            error=str(error),
        )

        return memory_error_response(error)


@app.post("/memory/revise/{memory_id}")
def revise_memory_record(
    memory_id: str,
    request: MemoryReviseRequest,
):
    start = time.time()
    current_run_id = resolve_run_id(request.run_id)

    try:
        record = revise_memory(
            memory_id=memory_id,
            content=request.content,
            source=request.source,
            confidence=request.confidence,
            run_id=current_run_id,
            metadata=request.metadata,
        )

        emit_memory_telemetry(
            event_type="memory_revise",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_revision",
        )

        return {
            "status": "ok",
            "operation": "memory_revised",
            "run_id": current_run_id,
            "record": record,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_revise",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_revision",
            error=str(error),
        )

        return memory_error_response(error)


@app.post("/memory/archive/{memory_id}")
def archive_memory_record(
    memory_id: str,
    request: MemoryArchiveRequest,
):
    start = time.time()
    current_run_id = resolve_run_id(request.run_id)

    try:
        record = archive_memory(
            memory_id=memory_id,
            reason=request.reason,
            run_id=current_run_id,
        )

        emit_memory_telemetry(
            event_type="memory_archive",
            status="ok",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_archive",
        )

        return {
            "status": "ok",
            "operation": "memory_archived",
            "run_id": current_run_id,
            "record": record,
        }

    except Exception as error:
        emit_memory_telemetry(
            event_type="memory_archive",
            status="exception",
            start=start,
            run_id_value=current_run_id,
            task_type="memory_archive",
            error=str(error),
        )

        return memory_error_response(error)


# ==========================================================
# CALLBACK AND ENVIRONMENT ENDPOINTS
# ==========================================================


@app.get("/auth/patreon/callback")
async def patreon_callback(request: Request):
    return JSONResponse(
        {
            "status": "received",
            "query_params": dict(request.query_params),
        }
    )


@app.get("/test/env")
def test_env():
    return {
        "openai_key_loaded": bool(
            os.getenv("OPENAI_API_KEY")
        ),
        "anthropic_key_loaded": bool(
            os.getenv("ANTHROPIC_API_KEY")
        ),
        "perplexity_key_loaded": bool(
            os.getenv("PERPLEXITY_API_KEY")
        ),
        "openai_model": os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        ),
        "anthropic_model": os.getenv(
            "ANTHROPIC_MODEL",
            "claude-opus-4-6",
        ),
        "perplexity_model": os.getenv(
            "PERPLEXITY_MODEL",
            "sonar",
        ),
    }


# ==========================================================
# ROUTING ENDPOINT
# ==========================================================


@app.post("/route")
async def route_request(route_request: RouteRequest):
    start = time.time()
    current_run_id = event_id()

    selected_provider = choose_available(
        route_request.task_type
    )

    if selected_provider is None:
        telemetry_record(
            provider="router",
            provider_name="OMENZ_ROUTER",
            model="router-v0.1",
            status="no_provider_available",
            start=start,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error="No provider is currently available.",
            task_type=route_request.task_type,
            message=None,
            event_type="router_decision",
            run_id_value=current_run_id,
            correlation_id=current_run_id,
        )

        return JSONResponse(
            {
                "task_type": route_request.task_type,
                "status": "no_provider_available",
                "run_id": current_run_id,
                "latency_ms": now_ms(start),
            },
            status_code=503,
        )

    telemetry_record(
        provider=selected_provider,
        provider_name=provider_name(selected_provider),
        model="router-v0.1",
        status="selected",
        start=start,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        error=None,
        task_type=route_request.task_type,
        message=None,
        event_type="router_decision",
        run_id_value=current_run_id,
        correlation_id=current_run_id,
    )

    result, status_code = await execute_provider(
        provider=selected_provider,
        message=route_request.message,
        run_id=current_run_id,
        start=start,
        task_type=route_request.task_type,
    )

    result["task_type"] = route_request.task_type
    result["router_selected"] = selected_provider
    result["router_selected_name"] = provider_name(
        selected_provider
    )

    return JSONResponse(
        result,
        status_code=status_code,
    )


# ==========================================================
# PROVIDER TEST ENDPOINTS
# ==========================================================


@app.get("/test/openai")
async def test_openai():
    start = time.time()
    current_run_id = event_id()

    result, status_code = await call_openai(
        message="Reply with: PENI online.",
        run_id=current_run_id,
        start=start,
        task_type="provider_test",
    )

    return JSONResponse(
        result,
        status_code=status_code,
    )


@app.get("/test/anthropic")
async def test_anthropic():
    start = time.time()
    current_run_id = event_id()

    result, status_code = await call_anthropic(
        message="Reply with: AUDE online.",
        run_id=current_run_id,
        start=start,
        task_type="provider_test",
    )

    return JSONResponse(
        result,
        status_code=status_code,
    )


@app.get("/test/perplexity")
async def test_perplexity():
    start = time.time()
    current_run_id = event_id()

    result, status_code = await call_perplexity(
        message="Reply with: XITY online.",
        run_id=current_run_id,
        start=start,
        task_type="provider_test",
    )

    return JSONResponse(
        result,
        status_code=status_code,
    )


@app.get("/test/providers")
async def test_all_providers():
    return {
        "message": "Run these provider checks one at a time:",
        "openai": "/test/openai",
        "anthropic": "/test/anthropic",
        "perplexity": "/test/perplexity",
        "router": {
            "endpoint": "/route",
            "method": "POST",
            "example": {
                "task_type": "research",
                "message": (
                    "What is the latest development in AI?"
                ),
            },
        },
        "telemetry": {
            "status": "online",
            "destination": "Google Cloud Logging",
            "structured": True,
            "router_events": True,
            "provider_events": True,
            "memory_events": True,
            "run_id_tracking": True,
            "correlation_id_tracking": True,
        },
        "memory": {
            "health": "/memory/health",
            "create": "/memory/create",
            "get": "/memory/get/{memory_id}",
            "list": "/memory/list",
            "search": "/memory/search",
            "revise": "/memory/revise/{memory_id}",
            "archive": "/memory/archive/{memory_id}",
            "storage": "in_process",
            "persistent": False,
        },
    }
