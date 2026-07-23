"""JSON file-based storage for underwriting records."""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from app.config.settings import settings


def _load_records(path: str) -> list[dict]:
    """Load all records from the JSON storage file.

    Args:
        path: Path to the JSON file.

    Returns:
        List of record dictionaries.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_records(path: str, records: list[dict]) -> None:
    """Write all records to the JSON storage file.

    Args:
        path: Path to the JSON file.
        records: List of record dictionaries to persist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def save_application(
    application: dict,
    assessment: dict,
) -> str:
    """Persist an underwriting application and its result.

    Args:
        application: The applicant's input data as a dict.
        assessment: The full risk assessment result as a dict (includes
            prediction, SHAP explanations, and risk report).

    Returns:
        The unique record ID assigned to this application.
    """
    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "application": application,
        "assessment": assessment,
    }

    path = settings.storage_path
    records = _load_records(path)
    records.append(record)
    _save_records(path, records)

    logger.info(f"Saved application {record_id} to {path}")
    return record_id


def get_all_applications() -> list[dict]:
    """Retrieve all persisted underwriting records.

    Returns:
        Full list of application records, sorted by timestamp (newest first).
    """
    records = _load_records(settings.storage_path)
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records


def get_application(record_id: str) -> Optional[dict]:
    """Retrieve a single underwriting record by ID.

    Args:
        record_id: The unique ID returned by ``save_application``.

    Returns:
        The record dictionary, or None if not found.
    """
    for record in _load_records(settings.storage_path):
        if record.get("id") == record_id:
            return record
    return None