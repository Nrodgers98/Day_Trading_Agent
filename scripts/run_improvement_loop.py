"""Run one autonomous improvement cycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.config import load_config
from src.agent.improvement.orchestrator import ImprovementOrchestrator
from src.agent.monitoring.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous improvement loop")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Force-enable improvement loop for this run",
    )
    parser.add_argument(
        "--autonomy-mode",
        choices=["manual", "autonomous_nonprod", "autonomous"],
        help="Override autonomy mode",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Disable dry-run for this execution",
    )
    parser.add_argument(
        "--evaluate-backtest",
        action="store_true",
        help="Enable backtest + walk-forward evaluation gate for proposals",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.enable:
        overrides.setdefault("improvement", {})["enabled"] = True
    if args.autonomy_mode:
        overrides.setdefault("improvement", {})["autonomy_mode"] = args.autonomy_mode
    if args.apply:
        overrides.setdefault("improvement", {})["dry_run"] = False
    if args.evaluate_backtest:
        overrides.setdefault("improvement", {})["evaluate_with_backtest"] = True

    config = load_config(args.config, overrides=overrides or None)
    setup_logging(config.monitoring, log_name="improvement")

    orchestrator = ImprovementOrchestrator(config)
    outcomes = asyncio.run(orchestrator.run_once())
    print(json.dumps({"outcomes": outcomes}, indent=2, default=str))


if __name__ == "__main__":
    main()

