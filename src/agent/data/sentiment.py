"""FinBERT sentiment scorer with graceful degradation and caching."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.agent.config import SentimentConfig
from src.agent.data.market_data import AlpacaMarketDataClient

logger = logging.getLogger(__name__)


class SentimentProvider:
    """Fetch symbol headlines and score them with FinBERT."""

    def __init__(
        self,
        config: SentimentConfig,
        market_data: AlpacaMarketDataClient,
    ) -> None:
        self._cfg = config
        self._market_data = market_data
        self._cache: dict[str, tuple[datetime, float | None]] = {}
        self._pipeline = None
        self._pipeline_load_failed = False
        self._newsapi_key_missing_warned = False

    async def get_sentiment_score(self, symbol: str) -> float | None:
        if not self._cfg.enabled or self._cfg.provider == "none":
            return None
        cached = self._cache.get(symbol)
        now = datetime.now(tz=timezone.utc)
        if cached is not None and cached[0] > now:
            return cached[1]

        score = await self._fetch_score(symbol)
        expires = now + timedelta(seconds=self._cfg.cache_ttl_seconds)
        self._cache[symbol] = (expires, score)
        return score

    async def _fetch_score(self, symbol: str) -> float | None:
        try:
            articles = await self._fetch_headlines(symbol)
        except Exception:
            logger.debug("News fetch failed for %s", symbol, exc_info=True)
            return None

        headlines = [
            str(item.get("headline", "")).strip()
            for item in articles
            if str(item.get("headline", "")).strip()
        ]
        if not headlines:
            return None

        return await asyncio.to_thread(self._score_headlines, headlines)

    async def _fetch_headlines(self, symbol: str) -> list[dict[str, Any]]:
        if self._cfg.news_source == "newsapi":
            return await self._fetch_headlines_newsapi(symbol)
        return await self._fetch_headlines_alpaca(symbol)

    async def _fetch_headlines_alpaca(self, symbol: str) -> list[dict[str, Any]]:
        start = datetime.now(tz=timezone.utc) - timedelta(minutes=self._cfg.lookback_minutes)
        return await self._market_data.get_news(
            symbol,
            start=start,
            limit=self._cfg.max_headlines,
        )

    async def _fetch_headlines_newsapi(self, symbol: str) -> list[dict[str, Any]]:
        api_key = os.getenv("NEWS_API_KEY", "").strip()
        if not api_key:
            if not self._newsapi_key_missing_warned:
                logger.warning(
                    "NEWS_API_KEY not set; sentiment.news_source=newsapi cannot fetch headlines."
                )
                self._newsapi_key_missing_warned = True
            return []

        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(minutes=self._cfg.lookback_minutes)
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": self._cfg.max_headlines,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "apiKey": api_key,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://newsapi.org/v2/everything", params=params)
            resp.raise_for_status()
            body = resp.json()

        articles = body.get("articles")
        if not isinstance(articles, list):
            return []
        return [{"headline": str(a.get("title", "")).strip()} for a in articles if a.get("title")]

    def _score_headlines(self, headlines: list[str]) -> float | None:
        pipe = self._load_pipeline()
        if pipe is None:
            return None

        try:
            outputs = pipe(headlines, truncation=True)
        except Exception:
            logger.debug("FinBERT inference failed", exc_info=True)
            return None

        if not isinstance(outputs, list) or not outputs:
            return None

        values: list[float] = []
        for row in outputs:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label", "")).lower()
            confidence = float(row.get("score", 0.0) or 0.0)
            if "positive" in label:
                values.append(confidence)
            elif "negative" in label:
                values.append(-confidence)
            else:
                values.append(0.0)

        if not values:
            return None
        return max(-1.0, min(1.0, sum(values) / len(values)))

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_load_failed:
            return None
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self._cfg.model_name,
                tokenizer=self._cfg.model_name,
            )
            logger.info("Loaded FinBERT sentiment model: %s", self._cfg.model_name)
            return self._pipeline
        except Exception:
            self._pipeline_load_failed = True
            logger.warning(
                "Could not load FinBERT model (%s). Install transformers/torch and retry.",
                self._cfg.model_name,
                exc_info=True,
            )
            return None
