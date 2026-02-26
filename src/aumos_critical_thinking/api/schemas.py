"""Pydantic request/response schemas for the AumOS Critical Thinking REST API.

All schemas use strict validation. Request schemas validate inbound data;
response schemas serialise ORM models for outbound payloads.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Bias Detection schemas
# ---------------------------------------------------------------------------


class BiasDetectRequest(BaseModel):
    """Request body for POST /api/v1/critical/bias/detect."""

    user_id: uuid.UUID = Field(..., description="Operator user UUID")
    session_id: str = Field(..., min_length=1, max_length=255, description="Session or workflow identifier")
    decision_context: str = Field(
        ...,
        description="Domain context: model_deployment | data_labeling | risk_assessment | clinical | financial | compliance | security",
    )
    ai_recommendation: dict[str, Any] = Field(..., description="Structured AI recommendation presented")
    human_decision: dict[str, Any] = Field(..., description="Actual human decision made")
    review_duration_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Time the operator spent reviewing before deciding",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context")


class BiasDetectionResponse(BaseModel):
    """Response for a single bias detection record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    session_id: str
    decision_context: str
    ai_recommendation: dict[str, Any]
    human_decision: dict[str, Any]
    bias_score: float
    bias_category: str
    deviation_indicators: list[str]
    review_duration_seconds: int | None
    override_occurred: bool
    override_rationale: str | None
    metadata: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class BiasReportListResponse(BaseModel):
    """Paginated list of bias detection reports."""

    items: list[BiasDetectionResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Judgment Validation schemas
# ---------------------------------------------------------------------------


class JudgmentValidateRequest(BaseModel):
    """Request body for POST /api/v1/critical/judgment/validate."""

    user_id: uuid.UUID = Field(..., description="User whose judgment is being validated")
    decision_domain: str = Field(
        ...,
        description="Domain: model_risk | data_quality | security | compliance | clinical",
    )
    decision_id: str = Field(..., min_length=1, max_length=255, description="External reference to the decision")
    human_judgment: dict[str, Any] = Field(..., description="The human decision being validated")
    reference_standard: dict[str, Any] = Field(..., description="Ground truth or expert consensus")
    validation_method: str = Field(
        ...,
        description="ground_truth | expert_consensus | outcome_feedback | peer_review",
    )
    validator_id: uuid.UUID | None = Field(default=None, description="Optional expert or system UUID")


class JudgmentValidationResponse(BaseModel):
    """Response for a single judgment validation record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    decision_domain: str
    decision_id: str
    human_judgment: dict[str, Any]
    reference_standard: dict[str, Any]
    validation_method: str
    is_valid: bool
    accuracy_score: float
    confidence_calibration: float | None
    divergence_analysis: dict[str, Any]
    validated_at: datetime | None
    validator_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class JudgmentHistoryResponse(BaseModel):
    """Paginated judgment validation history."""

    items: list[JudgmentValidationResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Atrophy Assessment schemas
# ---------------------------------------------------------------------------


class AtrophyAssessRequest(BaseModel):
    """Request body for POST /api/v1/critical/atrophy/assess."""

    user_id: uuid.UUID = Field(..., description="User UUID being assessed")
    assessment_domain: str = Field(
        ...,
        description="Skill domain: clinical_judgment | data_analysis | risk_assessment | security | compliance",
    )
    assessment_period_start: datetime = Field(..., description="Start of the measurement period")
    assessment_period_end: datetime = Field(..., description="End of the measurement period")
    current_score: float = Field(..., ge=0.0, le=1.0, description="Current skill score 0.0–1.0")
    ai_reliance_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of AI-deferred decisions 0.0–1.0"
    )
    independent_decision_count: int = Field(..., ge=0, description="Count of independent decisions")
    ai_assisted_decision_count: int = Field(..., ge=0, description="Count of AI-assisted decisions")
    skill_gaps: list[dict[str, Any]] = Field(default_factory=list, description="Identified skill gaps")
    notes: str | None = Field(default=None, description="Optional assessor notes")

    @field_validator("assessment_period_end")
    @classmethod
    def end_after_start(cls, end: datetime, info: Any) -> datetime:  # noqa: ANN401
        """Ensure assessment period end is after start."""
        start = info.data.get("assessment_period_start")
        if start and end <= start:
            raise ValueError("assessment_period_end must be after assessment_period_start")
        return end


class AtrophyAssessmentResponse(BaseModel):
    """Response for a single atrophy assessment record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    assessment_domain: str
    assessment_period_start: datetime
    assessment_period_end: datetime
    baseline_score: float | None
    current_score: float
    atrophy_rate: float
    atrophy_severity: str
    ai_reliance_ratio: float
    independent_decision_count: int
    ai_assisted_decision_count: int
    skill_gaps: list[dict[str, Any]]
    intervention_required: bool
    notes: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AtrophyMetricsResponse(BaseModel):
    """Paginated atrophy assessment metrics."""

    items: list[AtrophyAssessmentResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Challenge schemas
# ---------------------------------------------------------------------------


class ChallengeGenerateRequest(BaseModel):
    """Request body for POST /api/v1/critical/challenges/generate."""

    domain: str = Field(
        ...,
        description="Challenge domain: model_risk | data_ethics | clinical | security | compliance",
    )
    difficulty_level: str = Field(
        ...,
        description="novice | intermediate | advanced | expert",
    )
    target_skills: list[str] = Field(
        ..., min_length=1, description="Skills this challenge should exercise"
    )
    atrophy_context: dict[str, Any] | None = Field(
        default=None, description="Optional atrophy data to personalise the scenario"
    )
    include_ai_trap: bool = Field(
        default=True, description="Embed a misleading AI recommendation in the scenario"
    )


class ChallengeResponse(BaseModel):
    """Response for a single challenge scenario."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    domain: str
    difficulty_level: str
    scenario_description: str
    scenario_data: dict[str, Any]
    ai_trap: dict[str, Any] | None
    expected_reasoning: list[dict[str, Any]]
    correct_approach: dict[str, Any]
    target_skills: list[str]
    status: str
    times_used: int
    average_score: float | None
    generated_by: str
    source_case_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChallengeListResponse(BaseModel):
    """Paginated list of challenge scenarios."""

    items: list[ChallengeResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Training Recommendation schemas
# ---------------------------------------------------------------------------


class TrainingRecommendationResponse(BaseModel):
    """Response for a single training recommendation."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    assessment_id: uuid.UUID | None
    recommendation_type: str
    priority: str
    target_domain: str
    program_name: str
    program_description: str
    program_modules: list[dict[str, Any]]
    estimated_duration_hours: float
    challenge_ids: list[str]
    target_skill_improvement: float
    status: str
    accepted_at: datetime | None
    completed_at: datetime | None
    outcome_score: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TrainingRecommendationsListResponse(BaseModel):
    """Paginated list of training recommendations."""

    items: list[TrainingRecommendationResponse]
    total: int
    page: int
    page_size: int
