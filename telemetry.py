import time
import uuid


def run_id():
    return str(uuid.uuid4())


def now_ms(start):
    return round((time.time() - start) * 1000)


def telemetry_record(
    provider,
    model,
    status,
    start,
    tokens_in=0,
    tokens_out=0,
    cost_usd=0.0,
    error=None,
):
    return {
        "run_id": run_id(),
        "timestamp": int(time.time()),
        "provider": provider,
        "model": model,
        "status": status,
        "latency_ms": now_ms(start),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "cost_usd": round(cost_usd, 6),
        "error": error,
    }
