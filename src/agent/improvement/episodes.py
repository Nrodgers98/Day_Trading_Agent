"""Build learning episodes and rewards from existing log artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.agent.improvement.models import FailurePattern, ImprovementEpisode
from src.agent.monitoring.audit_ingest import collect_audit_summary_for_session_date


class EpisodeDatasetBuilder:
    """Parses reports and runner logs into per-day episodes."""

    def __init__(
        self,
        log_dir: str = "logs",
        observe_modes: list[str] | None = None,
        *,
        session_timezone: str = "US/Eastern",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._observe_modes = observe_modes or ["paper"]
        self._session_tz = ZoneInfo(session_timezone)

    def build(self, lookback_days: int = 7) -> list[ImprovementEpisode]:
        reports = sorted((self._log_dir / "reports").glob("report_*.json"))[-lookback_days:]
        failures_by_date = self._extract_failures(lookback_days=lookback_days)
        episodes: list[ImprovementEpisode] = []

        for report_path in reports:
            payload = self._read_json(report_path)
            summary = dict(payload.get("summary", {}))
            trades = payload.get("trades", [])
            date = str(summary.get("date") or report_path.stem.replace("report_", ""))
            audit_extra = collect_audit_summary_for_session_date(
                self._log_dir, date, self._session_tz,
            )
            summary.update(audit_extra)
            episode = ImprovementEpisode(
                date=date,
                summary=summary,
                trades=trades,
                failures=failures_by_date.get(date, []),
            )
            episode.rewards = self._compute_rewards(episode)
            episodes.append(episode)

        return episodes

    def _extract_failures(self, lookback_days: int) -> dict[str, list[FailurePattern]]:
        log_files = self._target_log_files(lookback_days=lookback_days)
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        timestamps: dict[tuple[str, str], tuple[str | None, str | None]] = {}

        for path in log_files:
            for obj in self._iter_jsonl(path):
                level = str(obj.get("level", "")).upper()
                if level not in {"ERROR", "CRITICAL"}:
                    continue
                ts = str(obj.get("timestamp", ""))
                date = ts[:10] if len(ts) >= 10 else "unknown"
                msg = str(obj.get("message", "error"))
                exc = str(obj.get("exception", "")).strip()
                detail = msg if not exc else f"{msg} | {exc.splitlines()[-1]}"
                grouped[date][detail] += 1

                key = (date, detail)
                first_last = timestamps.get(key)
                if first_last is None:
                    timestamps[key] = (ts, ts)
                else:
                    timestamps[key] = (first_last[0], ts)

        out: dict[str, list[FailurePattern]] = {}
        for date, counter in grouped.items():
            patterns: list[FailurePattern] = []
            for detail, count in counter.most_common():
                first_ts, last_ts = timestamps.get((date, detail), (None, None))
                patterns.append(
                    FailurePattern(
                        source="observed_runner_logs",
                        message=detail,
                        count=count,
                        first_seen=first_ts,  # pydantic parses ISO string
                        last_seen=last_ts,
                    )
                )
            out[date] = patterns
        return out

    def _target_log_files(self, lookback_days: int) -> list[Path]:
        candidates: list[Path] = []
        mode_to_file = {
            "paper": "agent_paper.log",
            "live": "agent_live.log",
        }
        for mode in self._observe_modes:
            file_name = mode_to_file.get(mode)
            if not file_name:
                continue
            path = self._log_dir / file_name
            if path.exists():
                candidates.append(path)

        # Include sentiment diagnostics as observed runtime context.
        sentiment_log = self._log_dir / "agent_sentiment.log"
        if sentiment_log.exists():
            candidates.append(sentiment_log)

        # Consider audit stream for note-level errors if present.
        for audit_path in sorted(self._log_dir.glob("audit_*.jsonl"))[-max(1, lookback_days):]:
            candidates.append(audit_path)

        return candidates

    @staticmethod
    def _compute_rewards(episode: ImprovementEpisode) -> dict[str, float]:
        summary = episode.summary
        trade_count = int(summary.get("trade_count", 0) or 0)
        pnl = float(summary.get("total_pnl", 0.0) or 0.0)
        win_rate = float(summary.get("win_rate", 0.0) or 0.0)
        max_dd = float(summary.get("max_drawdown_pct", 0.0) or 0.0)

        # Composite reward favors risk-adjusted behavior over pure PnL.
        # drawdown and operational failures are explicit penalties.
        failure_penalty = sum(p.count for p in episode.failures) * 0.05
        trade_penalty = 0.0 if trade_count > 0 else 0.1

        risk_adjusted_reward = (
            (win_rate * 2.0)
            + (pnl / 1000.0)
            - (max_dd * 25.0)
            - failure_penalty
            - trade_penalty
        )
        stability_reward = -failure_penalty - (max_dd * 10.0)
        raw_pnl_reward = (pnl / 1000.0) - failure_penalty

        return {
            "risk_adjusted_return": round(risk_adjusted_reward, 6),
            "raw_pnl": round(raw_pnl_reward, 6),
            "stability": round(stability_reward, 6),
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except FileNotFoundError:
            return rows
        return rows

