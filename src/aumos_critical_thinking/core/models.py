"""SQLAlchemy ORM models for the AumOS Critical Thinking service.

All tables use the `crt_` prefix. Tenant-scoped tables extend AumOSModel
which supplies id (UUID), tenant_id, created_at, and updated_at columns.

Domain model:
  BiasDetection         — automation bias detection records for AI decision events
  JudgmentValidation    — human judgment validation results against AI recommendations
  AtrophyAssessment     — skill atrophy assessment metrics per user/domain
  Challenge             — generated challenge scenarios to maintain human judgment
  TrainingRecommendation — recommended training programs based on assessment outcomes
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from aumos_common.database import AumOSModel


class BiasDetection(AumOSModel):
    """Automation bias detection record for an AI decision event.

    Captures whether a human operator over-relied on an AI recommendation
    without applying independent critical judgment. Scored on a continuous
    0.0–1.0 bias index where 1.0 = full automation bias.

    Table: crt_bias_detections
    """

    __tablename__ = "crt_bias_detections"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="User UUID of the operator whose decision is being evaluated",
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Session or workflow identifier for grouping related decisions",
    )
    decision_context: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Domain context: model_deployment | data_labeling | risk_assessment | clinical | financial",
    )
    ai_recommendation: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Structured AI recommendation that was presented to the operator",
    )
    human_decision: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Actual decision made by the human operator",
    )
    bias_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Automation bias index 0.0–1.0 (1.0 = fully followed AI without question)",
    )
    bias_category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="none | mild | moderate | severe — categorical label derived from bias_score",
    )
    deviation_indicators: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of signals that indicate bias: e.g. no_review_time, immediate_acceptance",
    )
    review_duration_seconds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Time the operator spent reviewing before deciding (null if not tracked)",
    )
    override_occurred: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when the human overrode the AI recommendation with an independent decision",
    )
    override_rationale: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Operator-provided rationale for overriding the AI recommendation",
    )
    metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Additional context: model_id, pipeline_id, environment, etc.",
    )


class JudgmentValidation(AumOSModel):
    """Human judgment validation result.

    Records whether a human decision was validated as sound against ground truth,
    expert consensus, or downstream outcome feedback. Tracks accuracy over time
    to identify users who may benefit from targeted judgment training.

    Table: crt_judgment_validations
    """

    __tablename__ = "crt_judgment_validations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="User UUID whose judgment is being validated",
    )
    decision_domain: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Domain of the decision: model_risk | data_quality | security | compliance | clinical",
    )
    decision_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="External reference to the decision being validated (e.g., deployment_id)",
    )
    human_judgment: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="The human decision or assessment being validated",
    )
    reference_standard: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Ground truth, expert consensus, or outcome-based reference for comparison",
    )
    validation_method: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="ground_truth | expert_consensus | outcome_feedback | peer_review",
    )
    is_valid: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        comment="True when the human judgment aligns with the reference standard",
    )
    accuracy_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Accuracy of the human judgment 0.0–1.0 relative to reference standard",
    )
    confidence_calibration: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Calibration score: 1.0 = perfectly calibrated confidence, <1.0 = over/under-confident",
    )
    divergence_analysis: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Structured analysis of where and why the judgment diverged from reference",
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when validation was completed (null if pending)",
    )
    validator_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="UUID of the expert or system that performed the validation",
    )


class AtrophyAssessment(AumOSModel):
    """Skill atrophy assessment metrics for a user in a domain.

    Tracks the degradation of human critical thinking and domain skills
    over time when humans increasingly delegate decisions to AI. Regular
    assessments generate atrophy scores that feed into training recommendations.

    Table: crt_atrophy_assessments
    """

    __tablename__ = "crt_atrophy_assessments"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="User UUID being assessed",
    )
    assessment_domain: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Skill domain under assessment: clinical_judgment | data_analysis | risk_assessment | etc.",
    )
    assessment_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="Start of the measurement period for this assessment",
    )
    assessment_period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="End of the measurement period for this assessment",
    )
    baseline_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="User's skill score at the baseline period (null if first assessment)",
    )
    current_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="User's current skill score 0.0–1.0 for this domain",
    )
    atrophy_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Rate of skill decline per assessment period (negative = improvement)",
    )
    atrophy_severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="none | low | moderate | high | critical — severity classification",
    )
    ai_reliance_ratio: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Proportion of decisions in the period that deferred to AI (0.0–1.0)",
    )
    independent_decision_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Count of decisions made independently without AI recommendation",
    )
    ai_assisted_decision_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Count of decisions made with AI assistance in the period",
    )
    skill_gaps: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Identified skill gaps: [{skill, severity, evidence}]",
    )
    intervention_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True when atrophy_severity >= high and immediate training is required",
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Assessor notes or contextual observations",
    )


class Challenge(AumOSModel):
    """Generated challenge scenario to maintain and test human judgment.

    Challenges are synthetic or anonymised scenarios designed to require
    genuine human critical thinking, preventing AI automation bias and
    maintaining skill proficiency through deliberate practice.

    Table: crt_challenges
    """

    __tablename__ = "crt_challenges"

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Short descriptive title of the challenge scenario",
    )
    domain: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Domain the challenge targets: model_risk | data_ethics | clinical | security | compliance",
    )
    difficulty_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="novice | intermediate | advanced | expert — calibrated to target skill level",
    )
    scenario_description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full scenario text presenting the challenge to the participant",
    )
    scenario_data: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Structured scenario data: {context, data_points, constraints, distractors}",
    )
    ai_trap: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Deliberately misleading AI recommendation embedded in the scenario to test critical override",
    )
    expected_reasoning: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Key reasoning steps an expert would apply: [{step, rationale, weight}]",
    )
    correct_approach: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Reference solution with explanation of correct judgment approach",
    )
    target_skills: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Skills this challenge is designed to exercise: [skill_name, ...]",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        index=True,
        comment="draft | active | archived — lifecycle status of the challenge",
    )
    times_used: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times this challenge has been assigned to participants",
    )
    average_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Rolling average participant score for calibration purposes",
    )
    generated_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="system",
        comment="system | human_expert | llm_assisted — how the scenario was created",
    )
    source_case_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="Reference to the anonymised real-world case this scenario is based on",
    )


class TrainingRecommendation(AumOSModel):
    """Training program recommendation for addressing identified skill atrophy or bias.

    Generated from AtrophyAssessment and BiasDetection data, recommending
    targeted interventions to restore human judgment capacity and reduce
    over-reliance on AI automation.

    Table: crt_training_recommendations
    """

    __tablename__ = "crt_training_recommendations"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="User UUID for whom this recommendation is generated",
    )
    assessment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("crt_atrophy_assessments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source AtrophyAssessment UUID that triggered this recommendation",
    )
    recommendation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="skill_restoration | bias_correction | judgment_calibration | challenge_practice",
    )
    priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="low | medium | high | critical — urgency of the recommendation",
    )
    target_domain: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Domain this recommendation addresses (mirrors assessment_domain)",
    )
    program_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Name of the recommended training program or curriculum",
    )
    program_description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full description of the training program, objectives, and expected outcomes",
    )
    program_modules: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Ordered training modules: [{module_name, duration_hours, content_type, objectives}]",
    )
    estimated_duration_hours: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total estimated hours to complete the recommended training program",
    )
    challenge_ids: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="UUIDs of Challenge scenarios assigned as part of this training",
    )
    target_skill_improvement: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Expected skill score improvement (delta) upon completion",
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
        comment="pending | in_progress | completed | declined — lifecycle status",
    )
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when user accepted and started the recommendation",
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when user completed all program modules",
    )
    outcome_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Post-completion assessment score (null until training is completed)",
    )

    assessment: Mapped["AtrophyAssessment | None"] = relationship(
        "AtrophyAssessment",
        foreign_keys=[assessment_id],
    )
