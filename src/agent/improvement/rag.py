"""Lightweight local RAG for code/context retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from src.agent.improvement.models import RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


class LocalRAGIndex:
    """Builds chunked text index and retrieves by token-overlap score."""

    def __init__(
        self,
        paths: list[str],
        max_chunk_lines: int = 60,
        overlap_lines: int = 10,
    ) -> None:
        self._paths = [Path(p) for p in paths]
        self._max_chunk_lines = max_chunk_lines
        self._overlap_lines = overlap_lines
        self._chunks: list[tuple[str, str, Counter[str], int]] = []

    def build(self) -> None:
        self._chunks.clear()
        for path in self._paths:
            if not path.exists() or not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            start = 0
            chunk_idx = 0
            step = max(1, self._max_chunk_lines - self._overlap_lines)
            while start < len(lines):
                end = min(len(lines), start + self._max_chunk_lines)
                text = "\n".join(lines[start:end]).strip()
                if text:
                    tokens = self._token_counts(text)
                    chunk_id = f"{path.name}:{chunk_idx}"
                    self._chunks.append((str(path), text, tokens, len(tokens)))
                    chunk_idx += 1
                start += step

    def retrieve(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        if not self._chunks:
            self.build()
        q_tokens = self._token_counts(query)
        q_norm = self._norm(q_tokens)
        if q_norm == 0.0:
            return []

        scored: list[tuple[float, str, str, str]] = []
        for idx, (source_path, text, tokens, _) in enumerate(self._chunks):
            score = self._cosine(q_tokens, q_norm, tokens)
            if score <= 0.0:
                continue
            chunk_id = f"chunk_{idx}"
            scored.append((score, source_path, chunk_id, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedChunk(
                source_path=source_path,
                chunk_id=chunk_id,
                score=round(score, 6),
                text=text,
            )
            for score, source_path, chunk_id, text in scored[:top_k]
        ]

    @staticmethod
    def _token_counts(text: str) -> Counter[str]:
        tokens: list[str] = []
        for raw in _TOKEN_RE.findall(text):
            tok = raw.lower()
            tokens.append(tok)
            if "_" in tok:
                tokens.extend(part for part in tok.split("_") if part)
        return Counter(tokens)

    @staticmethod
    def _norm(tokens: Counter[str]) -> float:
        return math.sqrt(sum(v * v for v in tokens.values()))

    def _cosine(
        self,
        q_tokens: Counter[str],
        q_norm: float,
        doc_tokens: Counter[str],
    ) -> float:
        dot = 0.0
        for tok, qv in q_tokens.items():
            if tok in doc_tokens:
                dot += qv * doc_tokens[tok]
        d_norm = self._norm(doc_tokens)
        if d_norm == 0.0:
            return 0.0
        return dot / (q_norm * d_norm)

