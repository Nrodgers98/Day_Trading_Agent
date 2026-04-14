"""LLM advisor — merge rules, parsing, and allowlist (no network)."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from src.agent.config import AppConfig, ImprovementConfig, MonitoringConfig, RiskConfig, SessionConfig, StrategyConfig
from src.agent.improvement.llm_advisor import (
    ALLOWED_LLM_KEY_PATHS,
    ImprovementLLMAdvisor,
    merge_proposal_lists,
    _strip_json_fence,
)
from src.agent.improvement.models import CandidateChange, ImprovementEpisode, RetrievedChunk


def _minimal_config() -> AppConfig:
    return AppConfig(
        monitoring=MonitoringConfig(log_dir="logs"),
        improvement=ImprovementConfig(
            llm_advisor_enabled=True,
            llm_model="gpt-4o-mini",
            max_proposals_per_run=3,
        ),
        risk=RiskConfig(max_risk_per_trade_pct=0.005),
        strategy=StrategyConfig(ml_confidence_threshold=0.6),
        session=SessionConfig(),
    )


def test_merge_respects_max_and_dedupes_key_paths() -> None:
    p1 = CandidateChange(
        proposal_id="a",
        title="h1",
        rationale="r",
        predicted_impact="i",
        change_type="config_patch",
        changes=[
            {
                "path": "config/default.yaml",
                "operation": "set",
                "key_path": "risk.max_risk_per_trade_pct",
                "current": 0.01,
                "proposed": 0.009,
                "reason": "x",
            },
        ],
    )
    p2 = CandidateChange(
        proposal_id="b",
        title="h2",
        rationale="r",
        predicted_impact="i",
        change_type="config_patch",
        changes=[
            {
                "path": "config/default.yaml",
                "operation": "set",
                "key_path": "risk.max_risk_per_trade_pct",
                "current": 0.01,
                "proposed": 0.008,
                "reason": "y",
            },
        ],
    )
    p3 = CandidateChange(
        proposal_id="c",
        title="h3",
        rationale="r",
        predicted_impact="i",
        change_type="config_patch",
        changes=[
            {
                "path": "config/default.yaml",
                "operation": "set",
                "key_path": "strategy.ml_confidence_threshold",
                "current": 0.6,
                "proposed": 0.63,
                "reason": "z",
            },
        ],
    )
    merged = merge_proposal_lists([p1], [p2, p3], max_total=2)
    assert len(merged) == 2
    assert merged[0].proposal_id == "a"
    assert merged[1].proposal_id == "c"


def test_strip_json_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert json.loads(_strip_json_fence(raw)) == {"a": 1}


def test_parse_response_builds_candidates() -> None:
    cfg = _minimal_config()
    adv = ImprovementLLMAdvisor(cfg)
    text = json.dumps({
        "desk_review_markdown": "## Notes\nTest.",
        "suggestions": [
            {
                "title": "Tighten risk",
                "rationale": "Elevated drawdown",
                "key_path": "risk.max_risk_per_trade_pct",
                "proposed_value": 0.004,
                "confidence": 0.8,
            },
        ],
    })
    parsed = adv._parse_response(text)
    assert parsed and parsed["desk_review_markdown"].startswith("##")
    ep = ImprovementEpisode(
        date="2026-04-01",
        summary={"total_pnl": -50.0, "trade_count": 3},
        trades=[],
        failures=[],
    )
    props = adv._to_candidates(parsed["suggestions"], cfg.model_dump(), [], [ep])
    assert len(props) == 1
    assert props[0].changes[0]["key_path"] == "risk.max_risk_per_trade_pct"
    assert props[0].changes[0]["proposed"] == 0.004


def test_disallowed_key_path_dropped() -> None:
    cfg = _minimal_config()
    adv = ImprovementLLMAdvisor(cfg)
    raw = [
        {
            "title": "bad",
            "rationale": "x",
            "key_path": "trading.enable_live",
            "proposed_value": True,
            "confidence": 1.0,
        },
    ]
    ep = ImprovementEpisode(date="2026-04-01", summary={}, trades=[], failures=[])
    props = adv._to_candidates(raw, cfg.model_dump(), [], [ep])
    assert props == []


def test_propose_from_episodes_missing_key_returns_empty() -> None:
    cfg = _minimal_config()
    adv = ImprovementLLMAdvisor(cfg)
    with patch("src.agent.improvement.llm_advisor.os.getenv", return_value=""):
        props, meta = adv.propose_from_episodes([], [])
    assert props == []
    assert "missing_api_key" in meta.get("errors", [])


def test_propose_from_episodes_http_success() -> None:
    cfg = _minimal_config()
    adv = ImprovementLLMAdvisor(cfg)
    ep = ImprovementEpisode(date="2026-04-01", summary={}, trades=[], failures=[])
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "desk_review_markdown": "OK",
                        "suggestions": [
                            {
                                "title": "x",
                                "rationale": "y",
                                "key_path": "strategy.ml_confidence_threshold",
                                "proposed_value": 0.65,
                                "confidence": 0.5,
                            },
                        ],
                    }),
                },
            },
        ],
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=body)

    def fake_post(*_a: object, **_k: object) -> MagicMock:
        return mock_resp

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        with patch("src.agent.improvement.llm_advisor.httpx.post", fake_post):
            props, meta = adv.propose_from_episodes([ep], [])
    assert meta.get("parsed_count") == 1
    assert len(props) == 1
    assert props[0].changes[0]["key_path"] == "strategy.ml_confidence_threshold"


def test_allowed_paths_cover_heuristic_keys() -> None:
    assert "risk.max_risk_per_trade_pct" in ALLOWED_LLM_KEY_PATHS
    assert "strategy.ml_confidence_threshold" in ALLOWED_LLM_KEY_PATHS
