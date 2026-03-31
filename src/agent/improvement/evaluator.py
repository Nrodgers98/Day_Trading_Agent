"""Backtest + walk-forward evaluator with risk-aware acceptance gating."""

from __future__ import annotations

import asyncio
from typing import Any

from src.agent.config import AppConfig, load_config
from src.agent.improvement.models import CandidateChange, CandidateEvaluation
from src.agent.runner.backtest_runner import BacktestRunner


class ImprovementEvaluator:
    """Evaluates candidate changes and applies configured safety gates."""

    def __init__(self, baseline_config_path: str = "config/backtest.yaml") -> None:
        self._baseline_config_path = baseline_config_path

    async def evaluate_candidate(
        self,
        app_config: AppConfig,
        proposal: CandidateChange,
    ) -> CandidateEvaluation:
        baseline_config = load_config(
            self._baseline_config_path,
            overrides={"trading": {"mode": "backtest"}},
        )
        candidate_config = self._materialize_candidate_config(baseline_config, proposal)

        timeout = app_config.improvement.evaluation_timeout_seconds
        baseline_metrics, baseline_wf = await asyncio.wait_for(
            BacktestRunner(baseline_config).evaluate(walk_forward=True),
            timeout=timeout,
        )
        candidate_metrics, candidate_wf = await asyncio.wait_for(
            BacktestRunner(candidate_config).evaluate(walk_forward=True),
            timeout=timeout,
        )

        baseline_wf_agg = self._aggregate_walk_forward(baseline_wf)
        candidate_wf_agg = self._aggregate_walk_forward(candidate_wf)
        deltas = self._metric_deltas(baseline_wf_agg, candidate_wf_agg, baseline_metrics, candidate_metrics)

        accepted, reasons = self._gate(app_config, baseline_wf_agg, candidate_wf_agg, deltas)
        return CandidateEvaluation(
            proposal_id=proposal.proposal_id,
            accepted=accepted,
            reasons=reasons,
            baseline_metrics={
                "single_run": baseline_metrics,
                "walk_forward_aggregate": baseline_wf_agg,
            },
            candidate_metrics={
                "single_run": candidate_metrics,
                "walk_forward_aggregate": candidate_wf_agg,
            },
            walk_forward_metrics=candidate_wf,
            deltas=deltas,
        )

    def _materialize_candidate_config(self, base_cfg: AppConfig, proposal: CandidateChange) -> AppConfig:
        data = base_cfg.model_dump()
        if proposal.change_type == "config_patch":
            for change in proposal.changes:
                if change.get("operation") != "set":
                    continue
                key_path = str(change.get("key_path", ""))
                self._set_nested(data, key_path, change.get("proposed"))
        return AppConfig(**data)

    @staticmethod
    def _set_nested(root: dict[str, Any], key_path: str, value: Any) -> None:
        keys = [k for k in key_path.split(".") if k]
        if not keys:
            return
        node = root
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    @staticmethod
    def _aggregate_walk_forward(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
        if not metrics_list:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0.0,
            }
        fields = ["sharpe_ratio", "max_drawdown", "total_return_pct", "profit_factor", "total_trades"]
        out: dict[str, float] = {}
        n = float(len(metrics_list))
        for field in fields:
            out[field] = sum(float(m.get(field, 0.0) or 0.0) for m in metrics_list) / n
        return out

    @staticmethod
    def _metric_deltas(
        baseline_wf: dict[str, float],
        candidate_wf: dict[str, float],
        baseline_single: dict[str, Any],
        candidate_single: dict[str, Any],
    ) -> dict[str, float]:
        deltas = {
            "sharpe_delta": candidate_wf["sharpe_ratio"] - baseline_wf["sharpe_ratio"],
            "drawdown_delta": candidate_wf["max_drawdown"] - baseline_wf["max_drawdown"],
            "return_delta_pct": candidate_wf["total_return_pct"] - baseline_wf["total_return_pct"],
            "profit_factor_delta": candidate_wf["profit_factor"] - baseline_wf["profit_factor"],
        }

        baseline_trades = float(baseline_single.get("total_trades", 0.0) or 0.0)
        candidate_trades = float(candidate_single.get("total_trades", 0.0) or 0.0)
        if baseline_trades <= 0:
            deltas["trades_delta_pct"] = 0.0 if candidate_trades <= 0 else 1.0
        else:
            deltas["trades_delta_pct"] = (candidate_trades - baseline_trades) / baseline_trades

        return deltas

    @staticmethod
    def _gate(
        app_config: AppConfig,
        baseline_wf: dict[str, float],
        candidate_wf: dict[str, float],
        deltas: dict[str, float],
    ) -> tuple[bool, list[str]]:
        gates = app_config.improvement.gates
        reasons: list[str] = []

        if deltas["sharpe_delta"] < gates.min_sharpe_delta:
            reasons.append(
                f"Rejected: sharpe delta {deltas['sharpe_delta']:.4f} < min {gates.min_sharpe_delta:.4f}"
            )
        if deltas["profit_factor_delta"] < gates.min_profit_factor_delta:
            reasons.append(
                "Rejected: profit factor delta "
                f"{deltas['profit_factor_delta']:.4f} < min {gates.min_profit_factor_delta:.4f}"
            )
        if deltas["drawdown_delta"] > gates.max_drawdown_worsen_pct:
            reasons.append(
                f"Rejected: drawdown worsen {deltas['drawdown_delta']:.4f} > max {gates.max_drawdown_worsen_pct:.4f}"
            )
        if abs(deltas["trades_delta_pct"]) > gates.max_trades_delta_pct:
            reasons.append(
                f"Rejected: trades delta {deltas['trades_delta_pct']:.4f} exceeds ±{gates.max_trades_delta_pct:.4f}"
            )

        # Stability objective can require non-degrading drawdown even if Sharpe is marginal.
        if app_config.improvement.optimize_for == "stability":
            if candidate_wf["max_drawdown"] > baseline_wf["max_drawdown"]:
                reasons.append("Rejected: stability objective requires non-worsening drawdown.")

        return (len(reasons) == 0), reasons

