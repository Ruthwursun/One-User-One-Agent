"""Domain models for policy decisions, goal definitions, eligibility, and proposed interventions."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class PolicyDecision(BaseModel):
    """Structured result returned by the safety and policy gate."""

    allowed: bool
    reason_code: str
    message: str
    channel: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class EligibilityResult(BaseModel):
    """Structured result for an individual goal eligibility check."""

    goal_id: str
    eligible: bool
    reason_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class GoalDefinition(BaseModel):
    """Definition of a lifecycle goal loaded from config."""

    goal_id: str
    name: str
    description: str
    eligibility: dict[str, Any] = Field(default_factory=dict)
    success: dict[str, Any] = Field(default_factory=dict)


class ProposedAction(BaseModel):
    """Candidate intervention proposed by an agent before policy validation."""

    user_id: str
    type: str  # 'home_page', 'push', 'whatsapp', 'in_app_message'
    published_at: Optional[str] = None
    send_at: Optional[str] = None
    dialect: Optional[str] = None
    hero_title_id: Optional[str] = None
    rows: Optional[list[dict[str, Any]]] = None
    title_id: Optional[str] = None
    text: Optional[dict[str, str]] = None
    template_id: Optional[str] = None
    variables: Optional[dict[str, str]] = None
    deeplink: Optional[str] = None
    plan_id: Optional[str] = None
    price_inr: Optional[float] = None
