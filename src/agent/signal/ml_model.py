"""
ML-based signal classifier — XGBoost with probability calibration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.agent.models import FeatureVector, Side

logger = logging.getLogger(__name__)

_LABEL_MAP: dict[int, Side] = {
    0: Side.FLAT,
    1: Side.LONG,
    2: Side.SHORT,
}


class MLSignalModel:
    """XGBoost classifier wrapped with Platt-scaling calibration."""

    def __init__(self) -> None:
        self._base = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            verbosity=0,
        )
        self._model: CalibratedClassifierCV | None = None
        self._trained = False

    # ── properties ────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── training ──────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit base classifier then calibrate probabilities via CV."""
        if len(X) == 0:
            raise ValueError("Cannot train on empty data")

        logger.info("Training XGBClassifier on %d samples, %d features", *X.shape)
        self._model = CalibratedClassifierCV(
            estimator=self._base,
            method="sigmoid",
            cv=3,
        )
        self._model.fit(X, y)
        self._trained = True
        logger.info("Training complete — calibrated model ready")

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, features: FeatureVector) -> tuple[Side, float]:
        """Return ``(predicted_side, calibrated_probability)``."""
        if not self._trained or self._model is None:
            raise RuntimeError("Model has not been trained yet")

        x = np.array(features.to_array(), dtype=np.float64).reshape(1, -1)
        probs = self._model.predict_proba(x)[0]
        label = int(np.argmax(probs))
        confidence = float(probs[label])
        side = _LABEL_MAP.get(label, Side.FLAT)

        logger.debug(
            "ML prediction: side=%s  confidence=%.3f  probs=%s",
            side.value,
            confidence,
            np.round(probs, 3).tolist(),
        )
        return side, confidence

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError("Cannot save an untrained model")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, dest)
        logger.info("Model saved to %s", dest)

    def load(self, path: str) -> None:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"No model file at {src}")
        self._model = joblib.load(src)
        self._trained = True
        logger.info("Model loaded from %s", src)
