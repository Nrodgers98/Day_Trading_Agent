"""Generate constrained improvement proposals from episodes + RAG context."""

from __future__ import annotations

import uuid
from typing import Any

from src.agent.config import AppConfig
from src.agent.improvement.models import CandidateChange, ImprovementEpisode, RetrievedChunk


class ProposalEngine:
    """Heuristic proposer with strict, machine-applyable change objects."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def propose(
        self,
        episodes: list[ImprovementEpisode],
        evidence: list[RetrievedChunk],
    ) -> list[CandidateChange]:
        if not episodes:
            return []

        latest = episodes[-1]
        proposals: list[CandidateChange] = []

        drawdown = float(latest.summary.get("max_drawdown_pct", 0.0) or 0.0)
        win_rate = float(latest.summary.get("win_rate", 0.0) or 0.0)
        trade_count = int(latest.summary.get("trade_count", 0) or 0)
        failure_count = sum(f.count for f in latest.failures)

        if drawdown > 0.01 or failure_count > 0:
            proposals.append(
                self._risk_tightening_proposal(
                    latest=latest,
                    evidence=evidence,
                    drawdown=drawdown,
                    failure_count=failure_count,
                )
            )

        if trade_count == 0 or (trade_count > 0 and win_rate < 0.35):
            proposals.append(
                self._signal_threshold_proposal(
                    latest=latest,
                    evidence=evidence,
                    win_rate=win_rate,
                    trade_count=trade_count,
                )
            )

        if (
            self._config.improvement.allow_code_patches
            and any("already in \"filled\" state" in f.message for f in latest.failures)
        ):
            proposals.append(self._order_cancel_code_patch(latest=latest, evidence=evidence))

        return proposals[: self._config.improvement.max_proposals_per_run]

    def _risk_tightening_proposal(
        self,
        *,
        latest: ImprovementEpisode,
        evidence: list[RetrievedChunk],
        drawdown: float,
        failure_count: int,
    ) -> CandidateChange:
        current_risk = self._config.risk.max_risk_per_trade_pct
        new_risk = max(0.001, round(current_risk * 0.9, 6))

        return CandidateChange(
            proposal_id=self._new_id(),
            title="Tighten per-trade risk after drawdown/failures",
            rationale=(
                "Recent session had elevated drawdown or operational failures; "
                "reduce per-trade risk to improve risk-adjusted outcomes."
            ),
            predicted_impact="Lower drawdown volatility with moderate trade size reduction.",
            rollback_conditions=[
                "Sharpe ratio drops below baseline - configured minimum.",
                "Trade count drops more than configured maximum.",
            ],
            change_type="config_patch",
            changes=[
                self._cfg_change(
                    key_path="risk.max_risk_per_trade_pct",
                    current=current_risk,
                    proposed=new_risk,
                    reason=f"drawdown={drawdown:.4f}, failures={failure_count}",
                )
            ],
            decision_basis=self._build_decision_basis(
                latest=latest,
                evidence=evidence,
                trigger_metrics={
                    "drawdown": round(drawdown, 6),
                    "failure_count": failure_count,
                    "win_rate": float(latest.summary.get("win_rate", 0.0) or 0.0),
                    "trade_count": int(latest.summary.get("trade_count", 0) or 0),
                },
                trigger_reason="risk_tightening_after_drawdown_or_failures",
            ),
            evidence=evidence,
        )

    def _signal_threshold_proposal(
        self,
        *,
        latest: ImprovementEpisode,
        evidence: list[RetrievedChunk],
        win_rate: float,
        trade_count: int,
    ) -> CandidateChange:
        current = self._config.strategy.ml_confidence_threshold
        if trade_count == 0:
            proposed = max(0.5, round(current - 0.03, 3))
            reasoning = "No trades executed; slightly lower confidence threshold to improve participation."
        else:
            proposed = min(0.8, round(current + 0.03, 3))
            reasoning = "Win rate low; raise confidence threshold to prefer higher-conviction entries."

        return CandidateChange(
            proposal_id=self._new_id(),
            title="Adjust ML confidence threshold for quality/participation",
            rationale=reasoning,
            predicted_impact="Expected to improve trade quality balance for risk-adjusted return.",
            rollback_conditions=[
                "Net return drops below baseline.",
                "Max drawdown worsens beyond configured tolerance.",
            ],
            change_type="config_patch",
            changes=[
                self._cfg_change(
                    key_path="strategy.ml_confidence_threshold",
                    current=current,
                    proposed=proposed,
                    reason=f"win_rate={win_rate:.4f}, trade_count={trade_count}",
                )
            ],
            decision_basis=self._build_decision_basis(
                latest=latest,
                evidence=evidence,
                trigger_metrics={
                    "win_rate": round(win_rate, 6),
                    "trade_count": trade_count,
                    "drawdown": float(latest.summary.get("max_drawdown_pct", 0.0) or 0.0),
                },
                trigger_reason="ml_threshold_adjustment_for_quality_participation",
            ),
            evidence=evidence,
        )

    def _order_cancel_code_patch(
        self,
        *,
        latest: ImprovementEpisode,
        evidence: list[RetrievedChunk],
    ) -> CandidateChange:
        return CandidateChange(
            proposal_id=self._new_id(),
            title="Handle already-filled cancel responses as non-fatal",
            rationale=(
                "Observed repeated order cancellation failures for already-filled orders. "
                "Treat this broker response as terminal-resolved to avoid failure cascades."
            ),
            predicted_impact="Fewer false operational failures and reduced circuit-breaker trips.",
            rollback_conditions=[
                "Unexpected increase in orphaned open orders.",
                "Order state divergence in reconciliation checks.",
            ],
            change_type="code_patch",
            changes=[
                {
                    "path": "src/agent/execution/order_manager.py",
                    "operation": "patch",
                    "constraint": "keep behavior idempotent and preserve circuit-breaker semantics",
                    "hint": "refresh order status when cancel says already filled/canceled",
                }
            ],
            decision_basis=self._build_decision_basis(
                latest=latest,
                evidence=evidence,
                trigger_metrics={
                    "failure_count": sum(f.count for f in latest.failures),
                    "contains_already_filled_cancel_error": any(
                        "already in \"filled\" state" in f.message for f in latest.failures
                    ),
                },
                trigger_reason="code_patch_for_order_cancel_failure_pattern",
            ),
            evidence=evidence,
        )

    def _build_decision_basis(
        self,
        *,
        latest: ImprovementEpisode,
        evidence: list[RetrievedChunk],
        trigger_metrics: dict[str, Any],
        trigger_reason: str,
    ) -> dict[str, Any]:
        failure_samples = [
            {
                "message": f.message,
                "count": f.count,
                "source": f.source,
                "first_seen": f.first_seen.isoformat() if f.first_seen else None,
                "last_seen": f.last_seen.isoformat() if f.last_seen else None,
            }
            for f in latest.failures[:5]
        ]
        evidence_sources = [
            {
                "source_path": e.source_path,
                "chunk_id": e.chunk_id,
                "score": e.score,
            }
            for e in evidence[:10]
        ]

        mode_paths = []
        for mode in self._config.improvement.observe_modes:
            mode_paths.append(f"logs/agent_{mode}.log")

        return {
            "optimize_for": self._config.improvement.optimize_for,
            "episode_date": latest.date,
            "trigger_reason": trigger_reason,
            "trigger_metrics": trigger_metrics,
            "episode_summary": latest.summary,
            "episode_rewards": latest.rewards,
            "failure_samples": failure_samples,
            "source_data_paths": [
                *mode_paths,
                "logs/agent_sentiment.log",
                "logs/audit_YYYY-MM-DD.jsonl",
                "logs/reports/report_YYYY-MM-DD.json",
            ],
            "retrieval_evidence_sources": evidence_sources,
        }

    @staticmethod
    def _cfg_change(
        *,
        key_path: str,
        current: Any,
        proposed: Any,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "path": "config/default.yaml",
            "operation": "set",
            "key_path": key_path,
            "current": current,
            "proposed": proposed,
            "reason": reason,
        }

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

