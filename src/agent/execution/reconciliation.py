"""Position reconciliation — drift detection between local and broker state."""

from __future__ import annotations

import logging

from src.agent.models import Position

logger = logging.getLogger(__name__)


class PositionReconciler:
    """Compares expected (local) positions against actual (broker) positions.

    Designed to run periodically (e.g. every 60 seconds) to surface
    discrepancies before they compound.
    """

    def reconcile(
        self,
        expected: list[Position],
        actual: list[Position],
    ) -> list[str]:
        """Return human-readable descriptions of every discrepancy found.

        Checks performed:
        - Positions present locally but missing at the broker
        - Positions present at the broker but missing locally
        - Quantity mismatches for positions present in both
        - Side mismatches for positions present in both
        """
        discrepancies: list[str] = []

        expected_map = {p.symbol: p for p in expected}
        actual_map = {p.symbol: p for p in actual}

        expected_symbols = set(expected_map)
        actual_symbols = set(actual_map)

        for symbol in sorted(expected_symbols - actual_symbols):
            pos = expected_map[symbol]
            discrepancies.append(
                f"MISSING @ broker: {symbol} expected {pos.side.value} {pos.qty}"
            )

        for symbol in sorted(actual_symbols - expected_symbols):
            pos = actual_map[symbol]
            discrepancies.append(
                f"EXTRA @ broker: {symbol} actual {pos.side.value} {pos.qty}"
            )

        for symbol in sorted(expected_symbols & actual_symbols):
            exp = expected_map[symbol]
            act = actual_map[symbol]

            if exp.side != act.side:
                discrepancies.append(
                    f"SIDE MISMATCH: {symbol} expected={exp.side.value} actual={act.side.value}"
                )

            if abs(exp.qty - act.qty) > 1e-9:
                discrepancies.append(
                    f"QTY MISMATCH: {symbol} expected={exp.qty} actual={act.qty}"
                )

        if discrepancies:
            logger.warning(
                "Reconciliation found %d discrepancies: %s",
                len(discrepancies),
                "; ".join(discrepancies),
            )
        else:
            logger.debug("Reconciliation clean — no discrepancies")

        return discrepancies
