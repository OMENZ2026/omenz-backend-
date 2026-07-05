import os
import time
import uuid
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="OMENZ Backend", version="0.3.1")


def event_id():
    return str(uuid.uuid4())


def now_ms(start):
    return round((time.time() - start) * 1000)


@app.get("/")
def root():
    return {
        "service": "OMENZ Backend",
        "status": "online",
        "message": "OMENZ Stack backend is running.",
        "version": "0.3.1"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/auth/patreon/callback")
async def patreon_callback(request: Request):
    return JSONResponse({
        "status": "received",
        "query_params": dict(request.query_params)
    })


@app.get("/test/env")
def test_env():
    return {
        "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_key_loaded": bool(os.getenv("ANTHROPIC_API_KEY")),
        "perplexity_key_loaded": bool(os.getenv("PERPLEXITY_API_KEY")),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"),
        "perplexity_model": os.getenv("PERPLEXITY_MODEL", "sonar"),
    }


@app.get("/test/openai")
async def test_openai():
    start = time.time()
    run_id = event_id()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return JSONResponse(
            {
                "provider": "openai",
                "status": "missing_key",
                "run_id": run_id,
                "latency_ms": now_ms(start)
            },
            status_code=500
        )

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
                            "content": "You are PENI inside the OMENZ stack."
                        },
                        {
                            "role": "user",
                            "content": "Reply with: PENI online."
                        }
                    ]
                }
            )

        data = response.json()

        return JSONResponse({
            "provider": "openai",
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "reply": data.get("choices", [{}])[0].get("message", {}).get("content"),
            "raw": data if response.status_code != 200 else None
        }, status_code=response.status_code)

    except Exception as e:
        return JSONResponse({
            "provider": "openai",
            "status": "exception",
            "error": str(e),
            "run_id": run_id,
            "latency_ms": now_ms(start)
        }, status_code=500)


@app.get("/test/anthropic")
async def test_anthropic():
    start = time.time()
    run_id = event_id()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")

    if not api_key:
        return JSONResponse(
            {
                "provider": "anthropic",
                "status": "missing_key",
                "run_id": run_id,
                "latency_ms": now_ms(start)
            },
            status_code=500
        )

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
                    "max_tokens": 40,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Reply with: AUDE online."
                        }
                    ]
                }
            )

        data = response.json()

        text = None
        if isinstance(data.get("content"), list) and data["content"]:
            text = data["content"][0].get("text")

        return JSONResponse({
            "provider": "anthropic",
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "reply": text,
            "raw": data if response.status_code != 200 else None
        }, status_code=response.status_code)

    except Exception as e:
        return JSONResponse({
            "provider": "anthropic",
            "status": "exception",
            "error": str(e),
            "run_id": run_id,
            "latency_ms": now_ms(start)
        }, status_code=500)


@app.get("/test/perplexity")
async def test_perplexity():
    start = time.time()
    run_id = event_id()
    api_key = os.getenv("PERPLEXITY_API_KEY")
    model = os.getenv("PERPLEXITY_MODEL", "sonar")

    if not api_key:
        return JSONResponse(
            {
                "provider": "perplexity",
                "status": "missing_key",
                "run_id": run_id,
                "latency_ms": now_ms(start)
            },
            status_code=500
        )

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
                    "max_tokens": 40,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are XITY inside the OMENZ stack."
                        },
                        {
                            "role": "user",
                            "content": "Reply with: XITY online."
                        }
                    ]
                }
            )

        data = response.json()

        return JSONResponse({
            "provider": "perplexity",
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
            "model": model,
            "run_id": run_id,
            "latency_ms": now_ms(start),
            "reply": data.get("choices", [{}])[0].get("message", {}).get("content"),
            "raw": data if response.status_code != 200 else None
        }, status_code=response.status_code)

    except Exception as e:
        return JSONResponse({
            "provider": "perplexity",
            "status": "exception",
            "error": str(e),
            "run_id": run_id,
            "latency_ms": now_ms(start)
        }, status_code=500)


@app.get("/test/providers")
async def test_all_providers():
    return {
        "message": "Run these one at a time first:",
        "openai": "/test/openai",
        "anthropic": "/test/anthropic",
        "perplexity": "/test/perplexity"
    }
