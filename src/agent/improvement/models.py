"""Data models for the autonomous improvement loop."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class FailurePattern(BaseModel):
    source: str
    message: str
    count: int = 1
    first_seen: datetime | None = None
    last_seen: datetime | None = None


class ImprovementEpisode(BaseModel):
    date: str
    summary: dict[str, Any] = Field(default_factory=dict)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    failures: list[FailurePattern] = Field(default_factory=list)
    rewards: dict[str, float] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    source_path: str
    chunk_id: str
    score: float
    text: str


class CandidateChange(BaseModel):
    proposal_id: str
    title: str
    rationale: str
    predicted_impact: str
    rollback_conditions: list[str] = Field(default_factory=list)
    change_type: Literal["config_patch", "code_patch"] = "config_patch"
    changes: list[dict[str, Any]] = Field(default_factory=list)
    decision_basis: dict[str, Any] = Field(default_factory=dict)
    evidence: list[RetrievedChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class CandidateEvaluation(BaseModel):
    proposal_id: str
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    candidate_metrics: dict[str, Any] = Field(default_factory=dict)
    walk_forward_metrics: list[dict[str, Any]] = Field(default_factory=list)
    deltas: dict[str, float] = Field(default_factory=dict)

