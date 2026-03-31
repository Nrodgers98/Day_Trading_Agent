"""Run paper trading via Alpaca paper account."""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.config import load_config
from src.agent.runner.paper import PaperRunner
from src.agent.monitoring.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper trading")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config, overrides={"trading": {"mode": "paper"}})
    setup_logging(config.monitoring, log_name="paper")

    runner = PaperRunner(config)

    def _shutdown_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
