"""Daily report generator — equity persistence and downsampling."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.monitoring.reports import DailyReportGenerator


def test_json_includes_downsampled_equity_curve(tmp_path: Path) -> None:
    gen = DailyReportGenerator()
    curve = [
        {"timestamp": f"2026-04-01T09:3{i:01d}:00-04:00", "equity": 100_000.0 + i}
        for i in range(50)
    ]
    path = gen.generate(
        "2026-04-01",
        [],
        curve,
        {"log_dir": str(tmp_path), "report_format": "json", "report_max_equity_points": 10},
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "equity_curve" in data
    assert len(data["equity_curve"]) == 10
    assert "summary" in data and data["summary"]["date"] == "2026-04-01"


def test_empty_equity_omits_key(tmp_path: Path) -> None:
    gen = DailyReportGenerator()
    path = gen.generate(
        "2026-04-02",
        [],
        [],
        {"log_dir": str(tmp_path), "report_format": "json"},
    )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "equity_curve" not in data
