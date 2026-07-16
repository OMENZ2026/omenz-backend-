import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ==========================================================
# OMENZ TELEMETRY v0.2
# Structured telemetry for Cloud Run / Google Cloud Logging
# ==========================================================

logger = logging.getLogger("omenz.telemetry")

if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def run_id() -> str:
    return str(uuid.uuid4())


def now_ms(start: float) -> int:
    return round((time.time() - start) * 1000)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_token_value(value: Optional[int]) -> int:
    if value is None:
        return 0

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_cost_value(value: Optional[float]) -> float:
    if value is None:
        return 0.0

    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return 0.0


def telemetry_record(
    provider: str,
    model: str,
    status: str,
    start: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    error: Optional[str] = None,
    task_type: Optional[str] = None,
    message: Optional[str] = None,
    provider_name: Optional[str] = None,
    event_type: str = "provider_call",
    run_id_value: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build and emit one structured OMENZ telemetry record.

    Existing calls using the original telemetry_record arguments
    remain compatible with this upgraded version.
    """

    resolved_run_id = run_id_value or run_id()
    resolved_correlation_id = correlation_id or resolved_run_id

    clean_tokens_in = safe_token_value(tokens_in)
    clean_tokens_out = safe_token_value(tokens_out)

    record = {
        "event_id": str(uuid.uuid4()),
        "run_id": resolved_run_id,
        "correlation_id": resolved_correlation_id,
        "timestamp": utc_timestamp(),
        "event_type": event_type,
        "provider": provider,
        "provider_name": provider_name,
        "model": model,
        "task_type": task_type,
        "status": status,
        "latency_ms": now_ms(start),
        "tokens_in": clean_tokens_in,
        "tokens_out": clean_tokens_out,
        "total_tokens": clean_tokens_in + clean_tokens_out,
        "cost_usd": safe_cost_value(cost_usd),
        "message": message,
        "error": error,
    }

    logger.info(json.dumps(record, ensure_ascii=False, default=str))

    return record
