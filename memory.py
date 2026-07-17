import copy
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ==========================================================
# OMENZ MEMORY v0.1
# Controlled in-process memory foundation
# ==========================================================
#
# Purpose:
# - Store clearly typed memory records.
# - Preserve source, timestamp, confidence, and status.
# - Prevent silent rewriting of stored information.
# - Require explicit operations for creation, revision,
#   archival, and retrieval.
# - Produce an audit record for every memory operation.
#
# Important:
# - This version uses process memory only.
# - Records reset whenever the Cloud Run instance restarts.
# - No database, agents, dashboard, or autonomous promotion.
# - This module does not modify router.py or main.py.
# ==========================================================


MEMORY_VERSION = "0.1.0"

ALLOWED_MEMORY_TYPES = {
    "fact",
    "preference",
    "task_state",
    "system_event",
    "suggestion",
}

ALLOWED_STATUSES = {
    "active",
    "archived",
    "superseded",
}

logger = logging.getLogger("omenz.memory")

if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def utc_timestamp() -> str:
    """
    Return the current UTC timestamp in ISO-8601 format.
    """

    return datetime.now(timezone.utc).isoformat()


def new_memory_id() -> str:
    """
    Create a unique memory record identifier.
    """

    return str(uuid.uuid4())


def new_event_id() -> str:
    """
    Create a unique memory audit event identifier.
    """

    return str(uuid.uuid4())


def normalize_text(value: Any) -> str:
    """
    Convert an incoming value into a clean string.
    """

    if value is None:
        return ""

    return str(value).strip()


def normalize_memory_type(memory_type: Any) -> str:
    """
    Validate and normalize a memory record type.
    """

    normalized = normalize_text(memory_type).lower()

    if normalized not in ALLOWED_MEMORY_TYPES:
        allowed = ", ".join(sorted(ALLOWED_MEMORY_TYPES))

        raise ValueError(
            f"Unsupported memory_type '{normalized}'. "
            f"Allowed values: {allowed}."
        )

    return normalized


def normalize_confidence(confidence: Any) -> float:
    """
    Convert confidence to a number from 0.0 through 1.0.
    """

    try:
        normalized = float(confidence)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "confidence must be a number from 0.0 through 1.0."
        ) from error

    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(
            "confidence must be between 0.0 and 1.0."
        )

    return round(normalized, 4)


def emit_memory_event(
    event_type: str,
    status: str,
    memory_id: Optional[str] = None,
    run_id: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Emit one structured memory audit event.

    Logging failures must never break memory operations.
    """

    event = {
        "event_id": new_event_id(),
        "timestamp": utc_timestamp(),
        "event_type": normalize_text(event_type),
        "component": "memory",
        "component_version": MEMORY_VERSION,
        "status": normalize_text(status),
        "memory_id": memory_id,
        "run_id": run_id,
        "message": message,
        "metadata": copy.deepcopy(metadata or {}),
    }

    try:
        logger.info("%s", event)
    except Exception:
        pass

    return event


class MemoryStore:
    """
    Controlled OMENZ memory store.

    Records cannot silently rewrite themselves.

    Any changed record is created as a new revision. The previous
    record remains available for audit and is marked superseded.

    This version stores records only inside the active Python process.
    """

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        emit_memory_event(
            event_type="memory_store_initialized",
            status="ok",
            message="OMENZ in-process memory store initialized.",
        )

    def create(
        self,
        memory_type: str,
        content: Any,
        source: Any,
        confidence: float = 1.0,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create one new active memory record.

        A source is mandatory so every record has provenance.
        """

        clean_type = normalize_memory_type(memory_type)
        clean_content = normalize_text(content)
        clean_source = normalize_text(source)
        clean_confidence = normalize_confidence(confidence)

        if not clean_content:
            raise ValueError("content cannot be empty.")

        if not clean_source:
            raise ValueError("source cannot be empty.")

        memory_id = new_memory_id()
        created_at = utc_timestamp()

        record = {
            "memory_id": memory_id,
            "memory_version": MEMORY_VERSION,
            "memory_type": clean_type,
            "content": clean_content,
            "source": clean_source,
            "confidence": clean_confidence,
            "status": "active",
            "created_at": created_at,
            "updated_at": created_at,
            "archived_at": None,
            "supersedes": None,
            "superseded_by": None,
            "run_id": run_id,
            "metadata": copy.deepcopy(metadata or {}),
        }

        with self._lock:
            self._records[memory_id] = record

        emit_memory_event(
            event_type="memory_created",
            status="ok",
            memory_id=memory_id,
            run_id=run_id,
            message="Memory record created.",
            metadata={
                "memory_type": clean_type,
                "source": clean_source,
                "confidence": clean_confidence,
            },
        )

        return copy.deepcopy(record)

    def get(
        self,
        memory_id: str,
        include_inactive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve one memory record by identifier.
        """

        clean_memory_id = normalize_text(memory_id)

        with self._lock:
            record = self._records.get(clean_memory_id)

            if record is None:
                result = None
            elif (
                not include_inactive
                and record.get("status") != "active"
            ):
                result = None
            else:
                result = copy.deepcopy(record)

        emit_memory_event(
            event_type="memory_read",
            status="found" if result else "not_found",
            memory_id=clean_memory_id or None,
            message=(
                "Memory record retrieved."
                if result
                else "Memory record was not found."
            ),
        )

        return result

    def list_records(
        self,
        memory_type: Optional[str] = None,
        status: Optional[str] = "active",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List memory records with optional type and status filters.
        """

        clean_type = None

        if memory_type is not None:
            clean_type = normalize_memory_type(memory_type)

        clean_status = None

        if status is not None:
            clean_status = normalize_text(status).lower()

            if clean_status not in ALLOWED_STATUSES:
                allowed = ", ".join(sorted(ALLOWED_STATUSES))

                raise ValueError(
                    f"Unsupported status '{clean_status}'. "
                    f"Allowed values: {allowed}."
                )

        try:
            clean_limit = int(limit)
        except (TypeError, ValueError) as error:
            raise ValueError("limit must be an integer.") from error

        if clean_limit < 1:
            raise ValueError("limit must be at least 1.")

        clean_limit = min(clean_limit, 1000)

        with self._lock:
            records = []

            for record in self._records.values():
                if (
                    clean_type is not None
                    and record.get("memory_type") != clean_type
                ):
                    continue

                if (
                    clean_status is not None
                    and record.get("status") != clean_status
                ):
                    continue

                records.append(copy.deepcopy(record))

            records.sort(
                key=lambda item: item.get("created_at", ""),
                reverse=True,
            )

            result = records[:clean_limit]

        emit_memory_event(
            event_type="memory_listed",
            status="ok",
            message="Memory records listed.",
            metadata={
                "memory_type": clean_type,
                "status_filter": clean_status,
                "result_count": len(result),
                "limit": clean_limit,
            },
        )

        return result

    def revise(
        self,
        memory_id: str,
        content: Any,
        source: Any,
        confidence: Optional[float] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new revision instead of rewriting an old record.

        The previous record becomes superseded and remains available
        for audit purposes.
        """

        clean_memory_id = normalize_text(memory_id)
        clean_content = normalize_text(content)
        clean_source = normalize_text(source)

        if not clean_memory_id:
            raise ValueError("memory_id cannot be empty.")

        if not clean_content:
            raise ValueError("content cannot be empty.")

        if not clean_source:
            raise ValueError("source cannot be empty.")

        with self._lock:
            original = self._records.get(clean_memory_id)

            if original is None:
                raise KeyError(
                    f"Memory record '{clean_memory_id}' was not found."
                )

            if original.get("status") != "active":
                raise ValueError(
                    "Only active memory records can be revised."
                )

            resolved_confidence = (
                original.get("confidence", 1.0)
                if confidence is None
                else normalize_confidence(confidence)
            )

            new_id = new_memory_id()
            revised_at = utc_timestamp()

            revised_metadata = copy.deepcopy(
                original.get("metadata", {})
            )

            if metadata:
                revised_metadata.update(copy.deepcopy(metadata))

            revised_record = {
                "memory_id": new_id,
                "memory_version": MEMORY_VERSION,
                "memory_type": original["memory_type"],
                "content": clean_content,
                "source": clean_source,
                "confidence": resolved_confidence,
                "status": "active",
                "created_at": revised_at,
                "updated_at": revised_at,
                "archived_at": None,
                "supersedes": clean_memory_id,
                "superseded_by": None,
                "run_id": run_id,
                "metadata": revised_metadata,
            }

            original["status"] = "superseded"
            original["superseded_by"] = new_id
            original["updated_at"] = revised_at

            self._records[new_id] = revised_record

        emit_memory_event(
            event_type="memory_revised",
            status="ok",
            memory_id=new_id,
            run_id=run_id,
            message="New memory revision created.",
            metadata={
                "supersedes": clean_memory_id,
                "memory_type": revised_record["memory_type"],
                "source": clean_source,
                "confidence": resolved_confidence,
            },
        )

        return copy.deepcopy(revised_record)

    def archive(
        self,
        memory_id: str,
        reason: Any,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Archive an active record without deleting its audit history.
        """

        clean_memory_id = normalize_text(memory_id)
        clean_reason = normalize_text(reason)

        if not clean_memory_id:
            raise ValueError("memory_id cannot be empty.")

        if not clean_reason:
            raise ValueError("archive reason cannot be empty.")

        with self._lock:
            record = self._records.get(clean_memory_id)

            if record is None:
                raise KeyError(
                    f"Memory record '{clean_memory_id}' was not found."
                )

            if record.get("status") != "active":
                raise ValueError(
                    "Only active memory records can be archived."
                )

            archived_at = utc_timestamp()

            record["status"] = "archived"
            record["archived_at"] = archived_at
            record["updated_at"] = archived_at

            record_metadata = record.setdefault("metadata", {})
            record_metadata["archive_reason"] = clean_reason

            result = copy.deepcopy(record)

        emit_memory_event(
            event_type="memory_archived",
            status="ok",
            memory_id=clean_memory_id,
            run_id=run_id,
            message="Memory record archived.",
            metadata={
                "archive_reason": clean_reason,
            },
        )

        return result

    def search(
        self,
        query: Any,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Perform a simple case-insensitive search over active records.

        This is intentionally lightweight. It is not a vector search
        system and does not promote results into long-term truth.
        """

        clean_query = normalize_text(query).lower()

        if not clean_query:
            raise ValueError("query cannot be empty.")

        clean_type = None

        if memory_type is not None:
            clean_type = normalize_memory_type(memory_type)

        try:
            clean_limit = int(limit)
        except (TypeError, ValueError) as error:
            raise ValueError("limit must be an integer.") from error

        if clean_limit < 1:
            raise ValueError("limit must be at least 1.")

        clean_limit = min(clean_limit, 100)

        with self._lock:
            results = []

            for record in self._records.values():
                if record.get("status") != "active":
                    continue

                if (
                    clean_type is not None
                    and record.get("memory_type") != clean_type
                ):
                    continue

                searchable_text = " ".join(
                    [
                        normalize_text(record.get("content")),
                        normalize_text(record.get("source")),
                        normalize_text(record.get("memory_type")),
                    ]
                ).lower()

                if clean_query in searchable_text:
                    results.append(copy.deepcopy(record))

            results.sort(
                key=lambda item: (
                    item.get("confidence", 0.0),
                    item.get("created_at", ""),
                ),
                reverse=True,
            )

            results = results[:clean_limit]

        emit_memory_event(
            event_type="memory_searched",
            status="ok",
            message="Memory search completed.",
            metadata={
                "query": clean_query,
                "memory_type": clean_type,
                "result_count": len(results),
                "limit": clean_limit,
            },
        )

        return results

    def count(
        self,
        status: Optional[str] = None,
    ) -> int:
        """
        Count stored memory records.
        """

        clean_status = None

        if status is not None:
            clean_status = normalize_text(status).lower()

            if clean_status not in ALLOWED_STATUSES:
                allowed = ", ".join(sorted(ALLOWED_STATUSES))

                raise ValueError(
                    f"Unsupported status '{clean_status}'. "
                    f"Allowed values: {allowed}."
                )

        with self._lock:
            if clean_status is None:
                total = len(self._records)
            else:
                total = sum(
                    1
                    for record in self._records.values()
                    if record.get("status") == clean_status
                )

        emit_memory_event(
            event_type="memory_counted",
            status="ok",
            message="Memory records counted.",
            metadata={
                "status_filter": clean_status,
                "count": total,
            },
        )

        return total

    def health_check(self) -> Dict[str, Any]:
        """
        Return the current operational status of the memory module.
        """

        with self._lock:
            total_records = len(self._records)
            active_records = sum(
                1
                for record in self._records.values()
                if record.get("status") == "active"
            )
            archived_records = sum(
                1
                for record in self._records.values()
                if record.get("status") == "archived"
            )
            superseded_records = sum(
                1
                for record in self._records.values()
                if record.get("status") == "superseded"
            )

        result = {
            "status": "ok",
            "component": "memory",
            "version": MEMORY_VERSION,
            "storage": "in_process",
            "persistent": False,
            "total_records": total_records,
            "active_records": active_records,
            "archived_records": archived_records,
            "superseded_records": superseded_records,
            "allowed_memory_types": sorted(
                ALLOWED_MEMORY_TYPES
            ),
        }

        emit_memory_event(
            event_type="memory_health_check",
            status="ok",
            message="Memory health check completed.",
            metadata=result,
        )

        return result


memory_store = MemoryStore()


def create_memory(
    memory_type: str,
    content: Any,
    source: Any,
    confidence: float = 1.0,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Public helper for creating a memory record.
    """

    return memory_store.create(
        memory_type=memory_type,
        content=content,
        source=source,
        confidence=confidence,
        run_id=run_id,
        metadata=metadata,
    )


def get_memory(
    memory_id: str,
    include_inactive: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Public helper for retrieving one memory record.
    """

    return memory_store.get(
        memory_id=memory_id,
        include_inactive=include_inactive,
    )


def list_memories(
    memory_type: Optional[str] = None,
    status: Optional[str] = "active",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Public helper for listing memory records.
    """

    return memory_store.list_records(
        memory_type=memory_type,
        status=status,
        limit=limit,
    )


def revise_memory(
    memory_id: str,
    content: Any,
    source: Any,
    confidence: Optional[float] = None,
    run_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Public helper for creating a controlled memory revision.
    """

    return memory_store.revise(
        memory_id=memory_id,
        content=content,
        source=source,
        confidence=confidence,
        run_id=run_id,
        metadata=metadata,
    )


def archive_memory(
    memory_id: str,
    reason: Any,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public helper for archiving one memory record.
    """

    return memory_store.archive(
        memory_id=memory_id,
        reason=reason,
        run_id=run_id,
    )


def search_memory(
    query: Any,
    memory_type: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Public helper for searching active memory records.
    """

    return memory_store.search(
        query=query,
        memory_type=memory_type,
        limit=limit,
    )


def memory_health() -> Dict[str, Any]:
    """
    Public helper for checking memory module health.
    """

    return memory_store.health_check()
