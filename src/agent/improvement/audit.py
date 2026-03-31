"""Audit logging for the autonomous improvement loop."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("US/Eastern")


class ImprovementAuditLogger:
    """Append-only JSONL audit stream for proposal lifecycle events."""

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        now = datetime.now(tz=EASTERN)
        path = self._log_dir / f"improvement_{now.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": now.isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

