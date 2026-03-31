"""Decision audit trail — append-only JSONL logs of every signal evaluation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.agent.models import AuditRecord


class AuditLogger:
    """Writes and reads :class:`AuditRecord` entries as daily JSONL files."""

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _file_for_date(self, date: str) -> Path:
        return self._log_dir / f"audit_{date}.jsonl"

    def log_decision(self, record: AuditRecord) -> None:
        """Append a single audit record to the day's JSONL file."""
        date_str = record.timestamp.strftime("%Y-%m-%d")
        path = self._file_for_date(date_str)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def get_decisions(
        self,
        date: str,
        symbol: str | None = None,
    ) -> list[AuditRecord]:
        """Read back audit records for a given date, optionally filtered by symbol."""
        path = self._file_for_date(date)
        if not path.exists():
            return []

        records: list[AuditRecord] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = AuditRecord.model_validate_json(line)
                if symbol is None or record.symbol == symbol:
                    records.append(record)
        return records
