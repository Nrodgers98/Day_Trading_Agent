"""
Run LIVE trading via Alpaca.

⚠️  WARNING: This uses REAL MONEY. Ensure you understand the risks.
    Requires ENABLE_LIVE_TRADING=true in environment AND enable_live=true in config.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.config import load_config
from src.agent.runner.live import LiveRunner
from src.agent.monitoring.logger import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LIVE trading (use with extreme caution)")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--confirm", action="store_true", help="Confirm live trading intent")
    args = parser.parse_args()

    if not args.confirm:
        print("ERROR: Live trading requires --confirm flag.")
        print("       This will trade with REAL MONEY.")
        print("       Usage: python scripts/run_live.py --confirm")
        sys.exit(1)

    if os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
        print("ERROR: ENABLE_LIVE_TRADING environment variable must be 'true'.")
        sys.exit(1)

    config = load_config(
        args.config,
        overrides={"trading": {"mode": "live", "enable_live": True}},
    )
    setup_logging(config.monitoring, log_name="live")

    runner = LiveRunner(config)

    def _shutdown_handler(sig, frame):
        print(f"\nReceived signal {sig}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
