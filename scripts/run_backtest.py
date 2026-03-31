"""Run backtesting engine with walk-forward validation."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.config import load_config
from src.agent.runner.backtest_runner import BacktestRunner
from src.agent.monitoring.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", default="config/backtest.yaml", help="Config file path")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward validation")
    args = parser.parse_args()

    config = load_config(args.config, overrides={"trading": {"mode": "backtest"}})
    setup_logging(config.monitoring, log_name="backtest")

    runner = BacktestRunner(config)
    if args.walk_forward:
        asyncio.run(runner.run_walk_forward())
    else:
        asyncio.run(runner.run())


if __name__ == "__main__":
    main()
