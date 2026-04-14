"""Optional LLM advisor for structured improvement proposals (offline only)."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

import httpx

from src.agent.config import AppConfig
from src.agent.improvement.models import CandidateChange, ImprovementEpisode, RetrievedChunk

logger = logging.getLogger("trading_agent")

# Config paths the advisor may suggest (``config_patch`` / set operation only).
ALLOWED_LLM_KEY_PATHS: frozenset[str] = frozenset({
    "risk.max_risk_per_trade_pct",
    "risk.max_daily_drawdown_pct",
    "risk.max_gross_exposure_pct",
    "risk.max_concurrent_positions",
    "risk.daily_trade_cap",
    "risk.stop_loss_atr_mult",
    "risk.take_profit_atr_mult",
    "risk.trailing_stop_atr_mult",
    "risk.spread_guard_pct",
    "risk.enable_trailing_stop",
    "strategy.ml_confidence_threshold",
    "strategy.volume_surge_ratio",
    "strategy.lookback_bars",
    "strategy.cooldown_bars",
    "strategy.max_trades_per_day",
    "strategy.max_hold_minutes",
    "session.opening_guard_minutes",
    "session.closing_guard_minutes",
})

_SYSTEM_PROMPT = """You are an experienced professional day trader and risk manager reviewing
an automated trading system's recent session reports. Your job is to produce:
1) A concise desk_review_markdown (Markdown) with actionable observations (risk, trade quality,
session timing, participation vs overtrading). Do not claim guaranteed returns.
2) Zero or more suggestions: each must use key_path from the ALLOWED list exactly, with a
numeric or boolean proposed_value appropriate for that setting. Only suggest a change if it
plausibly addresses the metrics. Do not invent paths outside ALLOWED.

Respond with a single JSON object only (no markdown fences), shape:
{
  "desk_review_markdown": string,
  "suggestions": [
    {
      "title": string,
      "rationale": string,
      "key_path": string,
      "proposed_value": number | boolean,
      "confidence": number
    }
  ]
}

ALLOWED key_path values (exact strings):
""" + ", ".join(sorted(ALLOWED_LLM_KEY_PATHS))


def merge_proposal_lists(
    primary: list[CandidateChange],
    secondary: list[CandidateChange],
    *,
    max_total: int,
) -> list[CandidateChange]:
    """Prefer ``primary``; add LLM proposals that do not duplicate a ``key_path`` already used."""

    def key_paths(prop: CandidateChange) -> set[str]:
        out: set[str] = set()
        for ch in prop.changes:
            if ch.get("operation") == "set" and ch.get("key_path"):
                out.add(str(ch["key_path"]))
        return out

    used: set[str] = set()
    result: list[CandidateChange] = []
    for p in primary:
        result.append(p)
        used |= key_paths(p)
        if len(result) >= max_total:
            return result
    for p in secondary:
        kp = key_paths(p)
        if kp & used:
            continue
        result.append(p)
        used |= kp
        if len(result) >= max_total:
            break
    return result


def _get_nested(root: dict[str, Any], key_path: str) -> Any:
    node: Any = root
    for part in key_path.split("."):
        if not part:
            continue
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _coerce_value(current: Any, proposed: Any) -> Any | None:
    if current is None:
        return None
    if isinstance(current, bool):
        if isinstance(proposed, bool):
            return proposed
        if isinstance(proposed, (int, float)):
            return bool(proposed)
        if isinstance(proposed, str):
            low = proposed.lower().strip()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
        return None
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(round(float(proposed)))
        except (TypeError, ValueError):
            return None
    if isinstance(current, float):
        try:
            return float(proposed)
        except (TypeError, ValueError):
            return None
    return None


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


class ImprovementLLMAdvisor:
    """Calls a chat-completions-compatible API; maps JSON suggestions to CandidateChange."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._imp = config.improvement

    def propose_from_episodes(
        self,
        episodes: list[ImprovementEpisode],
        evidence: list[RetrievedChunk],
    ) -> tuple[list[CandidateChange], dict[str, Any]]:
        api_key = (
            os.getenv("IMPROVEMENT_LLM_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        meta: dict[str, Any] = {
            "enabled": True,
            "model": self._imp.llm_model,
            "desk_review_markdown": "",
            "suggestions_raw": [],
            "parsed_count": 0,
            "errors": [],
        }
        if not api_key:
            meta["errors"].append("missing_api_key")
            meta["enabled"] = False
            return [], meta

        user_payload = self._build_user_payload(
            episodes, evidence, self._config.improvement.optimize_for,
        )
        url = f"{self._imp.llm_api_base_url.rstrip('/')}/chat/completions"
        try:
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._imp.llm_model,
                    "temperature": self._imp.llm_temperature,
                    "max_tokens": self._imp.llm_max_tokens,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_payload, default=str)},
                    ],
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            body = resp.json()
            content = str(body["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            logger.exception("LLM advisor request failed")
            meta["errors"].append(f"http_error:{exc!s}"[:500])
            return [], meta

        meta["raw_response_preview"] = content[:2000]
        parsed = self._parse_response(content)
        if not parsed:
            meta["errors"].append("json_parse_failed")
            return [], meta

        meta["desk_review_markdown"] = str(parsed.get("desk_review_markdown", "") or "")
        raw_suggestions = parsed.get("suggestions") or []
        if not isinstance(raw_suggestions, list):
            meta["errors"].append("suggestions_not_list")
            return [], meta

        meta["suggestions_raw"] = raw_suggestions
        cfg_dump = self._config.model_dump()
        proposals = self._to_candidates(raw_suggestions, cfg_dump, evidence, episodes)
        meta["parsed_count"] = len(proposals)
        return proposals, meta

    @staticmethod
    def _build_user_payload(
        episodes: list[ImprovementEpisode],
        evidence: list[RetrievedChunk],
        optimize_for: str,
    ) -> dict[str, Any]:
        ep_rows = []
        for ep in episodes[-14:]:
            ep_rows.append({
                "date": ep.date,
                "summary": ep.summary,
                "rewards": ep.rewards,
                "failures": [
                    {"message": f.message, "count": f.count}
                    for f in ep.failures[:8]
                ],
            })
        ev_rows = [{"source_path": e.source_path, "text": e.text[:800]} for e in evidence[:8]]
        return {
            "episodes_recent": ep_rows,
            "code_evidence_snippets": ev_rows,
            "optimize_for": optimize_for,
        }

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any] | None:
        text = _strip_json_fence(content)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def _to_candidates(
        self,
        raw_suggestions: list[Any],
        cfg_dump: dict[str, Any],
        evidence: list[RetrievedChunk],
        episodes: list[ImprovementEpisode],
    ) -> list[CandidateChange]:
        latest = episodes[-1] if episodes else None
        out: list[CandidateChange] = []
        for item in raw_suggestions:
            if not isinstance(item, dict):
                continue
            key_path = str(item.get("key_path", "")).strip()
            if key_path not in ALLOWED_LLM_KEY_PATHS:
                continue
            current = _get_nested(cfg_dump, key_path)
            if current is None:
                continue
            proposed_raw = item.get("proposed_value")
            proposed = _coerce_value(current, proposed_raw)
            if proposed is None:
                continue
            if proposed == current:
                continue
            title = str(item.get("title", "LLM configuration suggestion"))[:200]
            rationale = str(item.get("rationale", ""))[:2000]
            try:
                conf = float(item.get("confidence", 0.5) or 0.5)
            except (TypeError, ValueError):
                conf = 0.5

            change = {
                "path": "config/default.yaml",
                "operation": "set",
                "key_path": key_path,
                "current": current,
                "proposed": proposed,
                "reason": f"llm_advisor confidence={conf:.3f}",
            }
            decision_basis = {
                "source": "llm_advisor",
                "episode_date": latest.date if latest else None,
                "episode_summary": latest.summary if latest else {},
                "llm_confidence": conf,
                "retrieval_evidence_sources": [
                    {"source_path": e.source_path, "chunk_id": e.chunk_id}
                    for e in evidence[:10]
                ],
            }
            out.append(
                CandidateChange(
                    proposal_id=uuid.uuid4().hex[:12],
                    title=title,
                    rationale=rationale or "LLM-suggested tuning based on recent episodes.",
                    predicted_impact=(
                        "May improve risk-adjusted outcomes; validate with evaluate_with_backtest "
                        "when enabled."
                    ),
                    rollback_conditions=[
                        "Worsening drawdown or Sharpe vs baseline under configured gates.",
                        "Trade count or quality materially degrades.",
                    ],
                    change_type="config_patch",
                    changes=[change],
                    decision_basis=decision_basis,
                    evidence=evidence[:10],
                ),
            )
        return out
