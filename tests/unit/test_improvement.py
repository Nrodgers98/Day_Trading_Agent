"""Tests for autonomous improvement loop components."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.config import load_config
from src.agent.improvement.episodes import EpisodeDatasetBuilder
from src.agent.improvement.proposal import ProposalEngine
from src.agent.improvement.rag import LocalRAGIndex


def test_episode_builder_computes_rewards(tmp_path: Path):
    logs = tmp_path / "logs"
    reports = logs / "reports"
    reports.mkdir(parents=True)

    (reports / "report_2026-03-27.json").write_text(
        json.dumps(
            {
                "summary": {
                    "date": "2026-03-27",
                    "total_pnl": 120.0,
                    "trade_count": 4,
                    "win_rate": 0.5,
                    "max_drawdown_pct": 0.01,
                },
                "trades": [{"symbol": "SPY"}],
            }
        ),
        encoding="utf-8",
    )
    (logs / "agent_paper.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-27T12:00:00+00:00",
                "level": "ERROR",
                "message": "Error during trading cycle",
                "exception": "RuntimeError: boom",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    builder = EpisodeDatasetBuilder(str(logs))
    episodes = builder.build(lookback_days=7)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.date == "2026-03-27"
    assert "risk_adjusted_return" in ep.rewards
    assert ep.failures


def test_episode_builder_ignores_backtest_improvement_logs(tmp_path: Path):
    logs = tmp_path / "logs"
    reports = logs / "reports"
    reports.mkdir(parents=True)

    (reports / "report_2026-03-27.json").write_text(
        json.dumps(
            {
                "summary": {
                    "date": "2026-03-27",
                    "total_pnl": 10.0,
                    "trade_count": 1,
                    "win_rate": 1.0,
                    "max_drawdown_pct": 0.001,
                },
                "trades": [{"symbol": "SPY"}],
            }
        ),
        encoding="utf-8",
    )

    # Backtest/improvement errors should NOT influence observed failures.
    (logs / "agent_backtest.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-27T12:00:00+00:00",
                "level": "ERROR",
                "message": "Backtest failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "agent_improvement.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-27T12:05:00+00:00",
                "level": "ERROR",
                "message": "Improvement failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Observed mode logs should be included.
    (logs / "agent_paper.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-27T12:10:00+00:00",
                "level": "ERROR",
                "message": "Paper runtime failure",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    builder = EpisodeDatasetBuilder(str(logs), observe_modes=["paper"])
    episodes = builder.build(lookback_days=7)
    assert episodes
    failures = episodes[0].failures
    assert failures
    failure_text = " ".join(f.message for f in failures)
    assert "Paper runtime failure" in failure_text
    assert "Backtest failure" not in failure_text
    assert "Improvement failure" not in failure_text


def test_rag_index_retrieves_relevant_chunk(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text(
        "def place_order():\n"
        "    pass\n"
        "\n"
        "def cancel_order():\n"
        "    return True\n",
        encoding="utf-8",
    )
    idx = LocalRAGIndex(paths=[str(target)], max_chunk_lines=20, overlap_lines=5)
    idx.build()
    chunks = idx.retrieve("cancel order already filled", top_k=2)
    assert chunks
    assert "cancel_order" in chunks[0].text


def test_proposal_engine_returns_config_patch():
    cfg = load_config("config/default.yaml", overrides={"improvement": {"enabled": True}})
    engine = ProposalEngine(cfg)
    episodes = EpisodeDatasetBuilder("logs").build(lookback_days=1)
    if not episodes:
        # Build synthetic episode if no local logs are present.
        from src.agent.improvement.models import ImprovementEpisode

        episodes = [
            ImprovementEpisode(
                date="2026-03-27",
                summary={"max_drawdown_pct": 0.02, "trade_count": 0, "win_rate": 0.0},
                trades=[],
                failures=[],
                rewards={},
            )
        ]

    proposals = engine.propose(episodes, evidence=[])
    assert proposals
    assert proposals[0].change_type in {"config_patch", "code_patch"}
    assert proposals[0].decision_basis
    assert "episode_summary" in proposals[0].decision_basis
    assert "source_data_paths" in proposals[0].decision_basis

