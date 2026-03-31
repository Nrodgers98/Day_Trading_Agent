"""End-to-end autonomous improvement orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from src.agent.config import AppConfig
from src.agent.improvement.audit import ImprovementAuditLogger
from src.agent.improvement.episodes import EpisodeDatasetBuilder
from src.agent.improvement.evaluator import ImprovementEvaluator
from src.agent.improvement.models import CandidateChange, CandidateEvaluation
from src.agent.improvement.proposal import ProposalEngine
from src.agent.improvement.rag import LocalRAGIndex

logger = logging.getLogger("trading_agent")
EASTERN = ZoneInfo("US/Eastern")


class ImprovementOrchestrator:
    """Coordinates propose -> evaluate -> gate -> apply workflow."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._audit = ImprovementAuditLogger(config.monitoring.log_dir)
        self._episodes = EpisodeDatasetBuilder(
            config.monitoring.log_dir,
            observe_modes=config.improvement.observe_modes,
        )
        self._proposer = ProposalEngine(config)
        self._evaluator = ImprovementEvaluator()

        self._artifact_root = Path(config.improvement.candidate_dir)
        self._artifact_root.mkdir(parents=True, exist_ok=True)

        self._rag = LocalRAGIndex(
            paths=[
                "src/agent/signal/engine.py",
                "src/agent/risk/manager.py",
                "src/agent/execution/order_manager.py",
                "src/agent/config.py",
                "docs/RUNBOOK.md",
            ],
            max_chunk_lines=config.improvement.rag_max_chunk_lines,
            overlap_lines=config.improvement.rag_chunk_overlap_lines,
        )
        self._rag.build()

    async def run_once(self) -> list[dict[str, Any]]:
        if not self._config.improvement.enabled:
            logger.info("Improvement loop disabled; exiting.")
            return []
        if self._in_cooldown():
            logger.info("Improvement loop in cooldown; skipping this run.")
            return []

        episodes = self._episodes.build(self._config.improvement.analysis_lookback_days)
        if not episodes:
            logger.warning("No report episodes found; nothing to optimize.")
            return []

        query = self._build_query(episodes)
        evidence = self._rag.retrieve(query, top_k=self._config.improvement.rag_top_k)
        proposals = self._proposer.propose(episodes, evidence)
        if not proposals:
            logger.info("No proposals generated from current episodes.")
            return []

        outcomes: list[dict[str, Any]] = []
        for proposal in proposals:
            self._persist_proposal(proposal)
            self._audit.log("proposal_created", proposal.model_dump(mode="json"))

            if self._config.improvement.evaluate_with_backtest:
                evaluation = await self._evaluator.evaluate_candidate(self._config, proposal)
            else:
                evaluation = CandidateEvaluation(
                    proposal_id=proposal.proposal_id,
                    accepted=True,
                    reasons=["evaluation_skipped_backtest_disabled"],
                )
            self._persist_evaluation(proposal.proposal_id, evaluation)
            self._audit.log("proposal_evaluated", evaluation.model_dump(mode="json"))

            outcome = self._handle_mode_action(proposal, evaluation)
            outcomes.append(outcome)

        return outcomes

    def _handle_mode_action(
        self,
        proposal: CandidateChange,
        evaluation: CandidateEvaluation,
    ) -> dict[str, Any]:
        mode = self._config.improvement.autonomy_mode
        payload: dict[str, Any] = {
            "proposal_id": proposal.proposal_id,
            "mode": mode,
            "accepted": evaluation.accepted,
            "dry_run": self._config.improvement.dry_run,
            "action": "none",
        }

        if not evaluation.accepted:
            payload["action"] = "rejected"
            payload["reasons"] = evaluation.reasons
            self._audit.log("proposal_rejected", payload)
            return payload

        if mode == "manual":
            payload["action"] = "awaiting_manual_approval"
            self._audit.log("proposal_pending_manual", payload)
            return payload

        if "evaluation_skipped_backtest_disabled" in evaluation.reasons:
            payload["action"] = "accepted_evaluation_skipped_no_auto_apply"
            self._audit.log("proposal_accepted_eval_skipped", payload)
            return payload

        if proposal.change_type != "config_patch":
            payload["action"] = "accepted_not_applied_non_config"
            self._audit.log("proposal_accepted_no_apply", payload)
            return payload

        if mode == "autonomous_nonprod":
            candidate_path = self._write_candidate_yaml(proposal)
            payload["action"] = "applied_nonprod_candidate"
            payload["candidate_path"] = str(candidate_path)
            self._audit.log("proposal_applied_nonprod", payload)
            return payload

        # autonomous mode
        if self._config.improvement.dry_run:
            payload["action"] = "autonomous_dry_run_no_apply"
            self._audit.log("proposal_dry_run", payload)
            return payload

        self._apply_to_default_config(proposal)
        payload["action"] = "autonomous_applied_default_config"
        self._audit.log("proposal_applied_autonomous", payload)
        return payload

    def _write_candidate_yaml(self, proposal: CandidateChange) -> Path:
        cfg_data = self._config.model_dump()
        for change in proposal.changes:
            if change.get("operation") != "set":
                continue
            self._set_nested(cfg_data, str(change.get("key_path", "")), change.get("proposed"))
        out_path = self._artifact_root / f"candidate_{proposal.proposal_id}.yaml"
        out_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding="utf-8")
        return out_path

    def _apply_to_default_config(self, proposal: CandidateChange) -> None:
        cfg_path = Path("config/default.yaml")
        if not cfg_path.exists():
            return
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        for change in proposal.changes:
            if change.get("operation") != "set":
                continue
            self._set_nested(data, str(change.get("key_path", "")), change.get("proposed"))
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def _persist_proposal(self, proposal: CandidateChange) -> None:
        out = self._artifact_root / f"proposal_{proposal.proposal_id}.json"
        out.write_text(json.dumps(proposal.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _persist_evaluation(self, proposal_id: str, evaluation: CandidateEvaluation) -> None:
        out = self._artifact_root / f"evaluation_{proposal_id}.json"
        out.write_text(json.dumps(evaluation.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _in_cooldown(self) -> bool:
        now = datetime.now(tz=EASTERN)
        cutoff = now - timedelta(minutes=self._config.improvement.proposal_cooldown_minutes)
        current_log = Path(self._config.monitoring.log_dir) / f"improvement_{now.strftime('%Y-%m-%d')}.jsonl"
        if not current_log.exists():
            return False
        for line in reversed(current_log.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("timestamp")
            if not ts:
                continue
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=EASTERN)
            return dt >= cutoff
        return False

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
    def _build_query(episodes: list[Any]) -> str:
        latest = episodes[-1]
        fail_summary = "; ".join(f"{f.message} (x{f.count})" for f in latest.failures[:5]) or "no failures"
        return (
            f"Optimize risk-adjusted return. Latest date={latest.date}. "
            f"summary={latest.summary}. failures={fail_summary}. "
            "Find likely code/config points for safer improvement."
        )

