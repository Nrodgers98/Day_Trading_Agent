"""Backfill daily report JSON from agent logs and audit JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.config import load_config
from src.agent.monitoring.report_backfill import backfill_reports_from_logs


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild logs/reports/report_YYYY-MM-DD.* from agent_paper/live JSONL "
            "and optional audit_YYYY-MM-DD.jsonl (trades prefer audit when present)."
        ),
    )
    parser.add_argument("--config", default="config/default.yaml", help="Config path (log_dir, session.timezone)")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "both"],
        default="paper",
        help="Which agent_*.log files to scan",
    )
    parser.add_argument(
        "--only-days-with-log-activity",
        action="store_true",
        help="Skip dates with no audit file and no matching agent log lines",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing report files",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        parser.error("--end must be on or after --start")

    out = backfill_reports_from_logs(
        log_dir=Path(config.monitoring.log_dir),
        start=start,
        end=end,
        session_tz=config.session.timezone,
        mode=args.mode,
        dry_run=args.dry_run,
        force=args.force,
        only_days_with_log_activity=args.only_days_with_log_activity,
        report_format=config.monitoring.report_format,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
