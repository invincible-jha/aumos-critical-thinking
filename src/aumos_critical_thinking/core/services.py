"""Business logic services for the AumOS Critical Thinking service.

All services depend on repository and adapter interfaces (not concrete
implementations) and receive dependencies via constructor injection.
No framework code (FastAPI, SQLAlchemy) belongs here.

Key invariants enforced by services:
- Bias scores are always normalised to 0.0–1.0.
- Atrophy assessments always compute atrophy_rate relative to the latest baseline.
- Challenge generation embeds AI traps for severe atrophy cases by default.
- Training recommendations are deduplicated — one active recommendation per user/domain.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from aumos_common.errors import ConflictError, ErrorCode, NotFoundError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_critical_thinking.core.interfaces import (
    IAlternativeGeneratorAdapter,
    IArgumentExtractorAdapter,
    IAtrophyAssessmentRepository,
    IAtrophyMonitorAdapter,
    IBiasDetectionRepository,
    IChallengeGeneratorAdapter,
    IChallengeRepository,
    ICognitiveBiasDetectorAdapter,
    IConfidenceScorerAdapter,
    IDebateSimulatorAdapter,
    IEvidenceGathererAdapter,
    IFallacyDetectorAdapter,
    IJudgmentValidationRepository,
    IReasoningFrameworkAdapter,
    ITrainingRecommendationRepository,
)
from aumos_critical_thinking.core.models import (
    AtrophyAssessment,
    BiasDetection,
    Challenge,
    JudgmentValidation,
    TrainingRecommendation,
)

logger = get_logger(__name__)

# Valid decision contexts for bias detection
VALID_DECISION_CONTEXTS: frozenset[str] = frozenset(
    {"model_deployment", "data_labeling", "risk_assessment", "clinical", "financial", "compliance", "security"}
)

# Bias category thresholds
BIAS_THRESHOLDS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.25),
    "mild": (0.25, 0.50),
    "moderate": (0.50, 0.75),
    "severe": (0.75, 1.01),
}

# Atrophy severity thresholds based on atrophy_rate
ATROPHY_SEVERITY_THRESHOLDS: dict[str, tuple[float, float]] = {
    "none": (-1.0, 0.0),      # improvement or no change
    "low": (0.0, 0.05),
    "moderate": (0.05, 0.15),
    "high": (0.15, 0.30),
    "critical": (0.30, 1.0),
}

# Valid difficulty levels for challenges
VALID_DIFFICULTY_LEVELS: frozenset[str] = frozenset(
    {"novice", "intermediate", "advanced", "expert"}
)

# Valid training recommendation types
VALID_RECOMMENDATION_TYPES: frozenset[str] = frozenset(
    {"skill_restoration", "bias_correction", "judgment_calibration", "challenge_practice"}
)


def _classify_bias(score: float) -> str:
    """Classify a bias score into a categorical label.

    Args:
        score: Automation bias index 0.0–1.0.

    Returns:
        Categorical bias label: none | mild | moderate | severe.
    """
    for category, (low, high) in BIAS_THRESHOLDS.items():
        if low <= score < high:
            return category
    return "severe"


def _classify_atrophy(atrophy_rate: float) -> str:
    """Classify an atrophy rate into a severity label.

    Args:
        atrophy_rate: Rate of skill decline per assessment period.

    Returns:
        Severity label: none | low | moderate | high | critical.
    """
    for severity, (low, high) in ATROPHY_SEVERITY_THRESHOLDS.items():
        if low <= atrophy_rate < high:
            return severity
    return "critical"


class BiasDetectorService:
    """Detect and record automation bias in human-AI decision events.

    Analyses decision patterns to identify when operators over-rely on AI
    recommendations without applying independent critical judgment.
    """

    def __init__(
        self,
        bias_repo: IBiasDetectionRepository,
        event_publisher: EventPublisher,
        severe_bias_threshold: float = 0.75,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            bias_repo: BiasDetection persistence.
            event_publisher: Kafka event publisher.
            severe_bias_threshold: Score at or above which a bias event triggers an alert.
        """
        self._bias_repo = bias_repo
        self._publisher = event_publisher
        self._severe_threshold = severe_bias_threshold

    async def detect_bias(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: str,
        decision_context: str,
        ai_recommendation: dict[str, Any],
        human_decision: dict[str, Any],
        review_duration_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BiasDetection:
        """Analyse a decision event and record an automation bias detection.

        Computes bias score from decision similarity, review duration, and
        override patterns. Publishes Kafka events for severe bias cases.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: Operator user UUID.
            session_id: Session or workflow identifier.
            decision_context: Domain context for the decision.
            ai_recommendation: Structured AI recommendation presented.
            human_decision: Actual human decision made.
            review_duration_seconds: Optional review time in seconds.
            metadata: Additional context fields.

        Returns:
            Newly created BiasDetection record.

        Raises:
            ConflictError: If decision_context is invalid.
        """
        if decision_context not in VALID_DECISION_CONTEXTS:
            raise ConflictError(
                message=f"Invalid decision_context '{decision_context}'. Valid: {VALID_DECISION_CONTEXTS}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        # Compute bias score from decision event signals
        bias_score, deviation_indicators, override_occurred = self._analyse_decision(
            ai_recommendation=ai_recommendation,
            human_decision=human_decision,
            review_duration_seconds=review_duration_seconds,
        )

        override_rationale: str | None = None
        if override_occurred:
            override_rationale = human_decision.get("rationale")

        bias_category = _classify_bias(bias_score)

        detection = await self._bias_repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            decision_context=decision_context,
            ai_recommendation=ai_recommendation,
            human_decision=human_decision,
            bias_score=bias_score,
            bias_category=bias_category,
            deviation_indicators=deviation_indicators,
            review_duration_seconds=review_duration_seconds,
            override_occurred=override_occurred,
            override_rationale=override_rationale,
            metadata=metadata or {},
        )

        logger.info(
            "Bias detection recorded",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            decision_context=decision_context,
            bias_score=bias_score,
            bias_category=bias_category,
        )

        # Alert on severe bias
        if bias_score >= self._severe_threshold:
            await self._publisher.publish(
                Topics.CRITICAL_THINKING,
                {
                    "event_type": "critical.bias.severe_detected",
                    "tenant_id": str(tenant_id),
                    "user_id": str(user_id),
                    "detection_id": str(detection.id),
                    "bias_score": bias_score,
                    "bias_category": bias_category,
                    "decision_context": decision_context,
                    "session_id": session_id,
                },
            )
            logger.warning(
                "Severe automation bias detected",
                user_id=str(user_id),
                bias_score=bias_score,
                decision_context=decision_context,
            )

        return detection

    async def get_detection(
        self, detection_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> BiasDetection:
        """Retrieve a bias detection record by ID.

        Args:
            detection_id: BiasDetection UUID.
            tenant_id: Requesting tenant.

        Returns:
            BiasDetection record.

        Raises:
            NotFoundError: If detection not found.
        """
        detection = await self._bias_repo.get_by_id(detection_id, tenant_id)
        if detection is None:
            raise NotFoundError(
                message=f"Bias detection {detection_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return detection

    async def list_reports(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        bias_category: str | None = None,
        decision_context: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[BiasDetection], int]:
        """List bias detection reports with optional filters.

        Args:
            tenant_id: Requesting tenant.
            user_id: Optional user filter.
            bias_category: Optional category filter.
            decision_context: Optional decision context filter.
            page: 1-based page number.
            page_size: Results per page.

        Returns:
            Tuple of (detections, total_count).
        """
        return await self._bias_repo.list_by_tenant(
            tenant_id=tenant_id,
            user_id=user_id,
            bias_category=bias_category,
            decision_context=decision_context,
            page=page,
            page_size=page_size,
        )

    def _analyse_decision(
        self,
        ai_recommendation: dict[str, Any],
        human_decision: dict[str, Any],
        review_duration_seconds: int | None,
    ) -> tuple[float, list[str], bool]:
        """Compute bias score from decision event signals.

        Applies heuristics to determine how closely the human followed the
        AI recommendation without independent analysis.

        Args:
            ai_recommendation: AI recommendation dict.
            human_decision: Human decision dict.
            review_duration_seconds: Optional review duration.

        Returns:
            Tuple of (bias_score, deviation_indicators, override_occurred).
        """
        indicators: list[str] = []
        score_components: list[float] = []

        # Decision alignment — high agreement without modification = higher bias
        ai_outcome = ai_recommendation.get("outcome") or ai_recommendation.get("decision")
        human_outcome = human_decision.get("outcome") or human_decision.get("decision")
        override_occurred = ai_outcome != human_outcome

        if not override_occurred:
            score_components.append(0.6)  # Same decision as AI
        else:
            score_components.append(0.0)  # Overrode AI

        # Review duration — very fast review indicates no independent analysis
        if review_duration_seconds is not None:
            if review_duration_seconds < 10:
                score_components.append(0.8)
                indicators.append("immediate_acceptance")
            elif review_duration_seconds < 30:
                score_components.append(0.5)
                indicators.append("minimal_review_time")
            elif review_duration_seconds < 120:
                score_components.append(0.2)
            else:
                score_components.append(0.0)
        else:
            score_components.append(0.3)  # Unknown duration, moderate assumption
            indicators.append("no_review_duration_tracked")

        # No rationale provided when following AI = automation bias indicator
        has_rationale = bool(human_decision.get("rationale"))
        if not has_rationale and not override_occurred:
            score_components.append(0.7)
            indicators.append("no_independent_rationale")
        elif has_rationale:
            score_components.append(0.1)

        # Compute weighted average
        if score_components:
            bias_score = round(sum(score_components) / len(score_components), 4)
        else:
            bias_score = 0.0

        bias_score = max(0.0, min(1.0, bias_score))
        return bias_score, indicators, override_occurred


class JudgmentValidatorService:
    """Validate and track human judgment quality over time.

    Records validation outcomes and computes accuracy trends to identify
    users whose judgment may benefit from targeted training interventions.
    """

    def __init__(
        self,
        validation_repo: IJudgmentValidationRepository,
        event_publisher: EventPublisher,
        low_accuracy_threshold: float = 0.6,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            validation_repo: JudgmentValidation persistence.
            event_publisher: Kafka event publisher.
            low_accuracy_threshold: Score below which low accuracy events are published.
        """
        self._validations = validation_repo
        self._publisher = event_publisher
        self._low_accuracy_threshold = low_accuracy_threshold

    async def validate_judgment(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str,
        decision_id: str,
        human_judgment: dict[str, Any],
        reference_standard: dict[str, Any],
        validation_method: str,
        validator_id: uuid.UUID | None = None,
    ) -> JudgmentValidation:
        """Validate a human judgment against a reference standard.

        Computes accuracy score and divergence analysis, then publishes
        events for persistently low-accuracy patterns.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: User whose judgment is being validated.
            decision_domain: Domain of the decision.
            decision_id: External reference to the decision.
            human_judgment: The human decision being validated.
            reference_standard: Ground truth or expert consensus.
            validation_method: Validation methodology used.
            validator_id: Optional UUID of the validating expert or system.

        Returns:
            Newly created JudgmentValidation record.
        """
        # Compute accuracy and divergence
        accuracy_score, is_valid, divergence_analysis, confidence_calibration = (
            self._score_judgment(human_judgment, reference_standard)
        )

        validation = await self._validations.create(
            tenant_id=tenant_id,
            user_id=user_id,
            decision_domain=decision_domain,
            decision_id=decision_id,
            human_judgment=human_judgment,
            reference_standard=reference_standard,
            validation_method=validation_method,
            is_valid=is_valid,
            accuracy_score=accuracy_score,
            confidence_calibration=confidence_calibration,
            divergence_analysis=divergence_analysis,
            validator_id=validator_id,
        )

        logger.info(
            "Judgment validation recorded",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            decision_domain=decision_domain,
            accuracy_score=accuracy_score,
            is_valid=is_valid,
        )

        if accuracy_score < self._low_accuracy_threshold:
            await self._publisher.publish(
                Topics.CRITICAL_THINKING,
                {
                    "event_type": "critical.judgment.low_accuracy",
                    "tenant_id": str(tenant_id),
                    "user_id": str(user_id),
                    "validation_id": str(validation.id),
                    "decision_domain": decision_domain,
                    "accuracy_score": accuracy_score,
                },
            )

        return validation

    async def get_validation(
        self, validation_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> JudgmentValidation:
        """Retrieve a judgment validation by ID.

        Args:
            validation_id: JudgmentValidation UUID.
            tenant_id: Requesting tenant.

        Returns:
            JudgmentValidation record.

        Raises:
            NotFoundError: If validation not found.
        """
        validation = await self._validations.get_by_id(validation_id, tenant_id)
        if validation is None:
            raise NotFoundError(
                message=f"Judgment validation {validation_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return validation

    async def list_history(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[JudgmentValidation], int]:
        """List judgment validation history for a user.

        Args:
            tenant_id: Requesting tenant.
            user_id: Target user UUID.
            decision_domain: Optional domain filter.
            page: 1-based page number.
            page_size: Results per page.

        Returns:
            Tuple of (validations, total_count).
        """
        return await self._validations.list_by_user(
            tenant_id=tenant_id,
            user_id=user_id,
            decision_domain=decision_domain,
            page=page,
            page_size=page_size,
        )

    def _score_judgment(
        self,
        human_judgment: dict[str, Any],
        reference_standard: dict[str, Any],
    ) -> tuple[float, bool, dict[str, Any], float | None]:
        """Compute accuracy score and divergence analysis.

        Args:
            human_judgment: The human decision.
            reference_standard: Ground truth or expert consensus.

        Returns:
            Tuple of (accuracy_score, is_valid, divergence_analysis, confidence_calibration).
        """
        human_outcome = human_judgment.get("outcome") or human_judgment.get("decision")
        reference_outcome = reference_standard.get("outcome") or reference_standard.get("decision")
        is_valid = human_outcome == reference_outcome

        # Base accuracy from outcome match
        base_accuracy = 1.0 if is_valid else 0.0

        # Factor in reasoning quality if present
        human_reasoning = human_judgment.get("reasoning_steps", [])
        reference_reasoning = reference_standard.get("reasoning_steps", [])
        reasoning_score = 0.0
        if reference_reasoning:
            matched_steps = sum(
                1 for step in human_reasoning
                if any(step.get("concept") == ref.get("concept") for ref in reference_reasoning)
            )
            reasoning_score = matched_steps / len(reference_reasoning)

        # Weighted accuracy: 70% outcome, 30% reasoning quality
        accuracy_score = round(0.7 * base_accuracy + 0.3 * reasoning_score, 4)

        # Confidence calibration
        human_confidence = human_judgment.get("confidence")
        confidence_calibration: float | None = None
        if human_confidence is not None:
            # Perfect calibration = confidence matches accuracy (within 0.1)
            confidence_calibration = round(
                1.0 - abs(float(human_confidence) - accuracy_score), 4
            )

        divergence_analysis: dict[str, Any] = {
            "outcome_match": is_valid,
            "outcome_divergence": None if is_valid else {
                "human": human_outcome,
                "reference": reference_outcome,
            },
            "reasoning_coverage": round(reasoning_score, 4),
            "missing_reasoning_steps": [
                step.get("concept")
                for step in reference_reasoning
                if not any(s.get("concept") == step.get("concept") for s in human_reasoning)
            ],
        }

        return accuracy_score, is_valid, divergence_analysis, confidence_calibration


class AtrophyMonitorService:
    """Monitor skill atrophy and assess degradation of human judgment capacity.

    Computes atrophy metrics from historical decision patterns, identifying
    users whose critical thinking skills are declining due to AI over-reliance.
    """

    def __init__(
        self,
        atrophy_repo: IAtrophyAssessmentRepository,
        event_publisher: EventPublisher,
        intervention_threshold: str = "high",
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            atrophy_repo: AtrophyAssessment persistence.
            event_publisher: Kafka event publisher.
            intervention_threshold: Minimum severity requiring immediate intervention.
        """
        self._assessments = atrophy_repo
        self._publisher = event_publisher
        self._intervention_threshold = intervention_threshold

    async def assess_atrophy(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        assessment_domain: str,
        assessment_period_start: datetime,
        assessment_period_end: datetime,
        current_score: float,
        ai_reliance_ratio: float,
        independent_decision_count: int,
        ai_assisted_decision_count: int,
        skill_gaps: list[dict[str, Any]] | None = None,
        notes: str | None = None,
    ) -> AtrophyAssessment:
        """Perform a skill atrophy assessment for a user in a domain.

        Looks up the previous assessment as baseline, computes atrophy rate,
        and classifies severity. Publishes intervention events for critical cases.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: User UUID being assessed.
            assessment_domain: Skill domain under assessment.
            assessment_period_start: Start of measurement period.
            assessment_period_end: End of measurement period.
            current_score: Current skill score 0.0–1.0.
            ai_reliance_ratio: Proportion of AI-deferred decisions.
            independent_decision_count: Count of independent decisions in period.
            ai_assisted_decision_count: Count of AI-assisted decisions in period.
            skill_gaps: Optional identified skill gaps list.
            notes: Optional assessor notes.

        Returns:
            Newly created AtrophyAssessment record.
        """
        # Retrieve baseline from previous assessment
        prior = await self._assessments.get_latest_for_user_domain(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_domain=assessment_domain,
        )
        baseline_score: float | None = prior.current_score if prior else None

        # Compute atrophy rate (positive = degradation, negative = improvement)
        if baseline_score is not None:
            atrophy_rate = round(baseline_score - current_score, 4)
        else:
            atrophy_rate = 0.0

        atrophy_severity = _classify_atrophy(max(0.0, atrophy_rate))
        intervention_required = atrophy_severity in {"high", "critical"}

        assessment = await self._assessments.create(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_domain=assessment_domain,
            assessment_period_start=assessment_period_start,
            assessment_period_end=assessment_period_end,
            baseline_score=baseline_score,
            current_score=current_score,
            atrophy_rate=atrophy_rate,
            atrophy_severity=atrophy_severity,
            ai_reliance_ratio=ai_reliance_ratio,
            independent_decision_count=independent_decision_count,
            ai_assisted_decision_count=ai_assisted_decision_count,
            skill_gaps=skill_gaps or [],
            intervention_required=intervention_required,
            notes=notes,
        )

        logger.info(
            "Atrophy assessment completed",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            assessment_domain=assessment_domain,
            current_score=current_score,
            atrophy_rate=atrophy_rate,
            atrophy_severity=atrophy_severity,
        )

        if intervention_required:
            await self._publisher.publish(
                Topics.CRITICAL_THINKING,
                {
                    "event_type": "critical.atrophy.intervention_required",
                    "tenant_id": str(tenant_id),
                    "user_id": str(user_id),
                    "assessment_id": str(assessment.id),
                    "assessment_domain": assessment_domain,
                    "atrophy_severity": atrophy_severity,
                    "atrophy_rate": atrophy_rate,
                    "intervention_required": True,
                },
            )
            logger.warning(
                "Skill atrophy intervention required",
                user_id=str(user_id),
                assessment_domain=assessment_domain,
                atrophy_severity=atrophy_severity,
            )

        return assessment

    async def get_assessment(
        self, assessment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> AtrophyAssessment:
        """Retrieve an atrophy assessment by ID.

        Args:
            assessment_id: AtrophyAssessment UUID.
            tenant_id: Requesting tenant.

        Returns:
            AtrophyAssessment record.

        Raises:
            NotFoundError: If assessment not found.
        """
        assessment = await self._assessments.get_by_id(assessment_id, tenant_id)
        if assessment is None:
            raise NotFoundError(
                message=f"Atrophy assessment {assessment_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return assessment

    async def list_metrics(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        assessment_domain: str | None = None,
        atrophy_severity: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[AtrophyAssessment], int]:
        """List atrophy assessment metrics.

        Args:
            tenant_id: Requesting tenant.
            user_id: Optional user filter.
            assessment_domain: Optional domain filter.
            atrophy_severity: Optional severity filter.
            page: 1-based page number.
            page_size: Results per page.

        Returns:
            Tuple of (assessments, total_count).
        """
        return await self._assessments.list_metrics(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_domain=assessment_domain,
            atrophy_severity=atrophy_severity,
            page=page,
            page_size=page_size,
        )


class ChallengeGeneratorService:
    """Generate and manage challenge scenarios for human judgment maintenance.

    Creates targeted challenge scenarios that require genuine human critical
    thinking, preventing skill atrophy through deliberate practice.
    """

    def __init__(
        self,
        challenge_repo: IChallengeRepository,
        generator_adapter: IChallengeGeneratorAdapter,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            challenge_repo: Challenge persistence.
            generator_adapter: AI-powered scenario generation adapter.
            event_publisher: Kafka event publisher.
        """
        self._challenges = challenge_repo
        self._generator = generator_adapter
        self._publisher = event_publisher

    async def generate_challenge(
        self,
        tenant_id: uuid.UUID,
        domain: str,
        difficulty_level: str,
        target_skills: list[str],
        atrophy_context: dict[str, Any] | None = None,
        include_ai_trap: bool = True,
    ) -> Challenge:
        """Generate a new challenge scenario via AI-assisted generation.

        Args:
            tenant_id: Owning tenant UUID.
            domain: Target domain for the challenge.
            difficulty_level: Desired difficulty level.
            target_skills: Skills the challenge should exercise.
            atrophy_context: Optional atrophy data to personalise the challenge.
            include_ai_trap: True to embed a misleading AI recommendation.

        Returns:
            Newly created Challenge record.

        Raises:
            ConflictError: If difficulty_level is invalid.
        """
        if difficulty_level not in VALID_DIFFICULTY_LEVELS:
            raise ConflictError(
                message=f"Invalid difficulty_level '{difficulty_level}'. Valid: {VALID_DIFFICULTY_LEVELS}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        # Generate scenario via AI adapter
        scenario_data = await self._generator.generate_scenario(
            domain=domain,
            difficulty_level=difficulty_level,
            target_skills=target_skills,
            atrophy_context=atrophy_context,
            include_ai_trap=include_ai_trap,
        )

        challenge = await self._challenges.create(
            tenant_id=tenant_id,
            title=scenario_data["title"],
            domain=domain,
            difficulty_level=difficulty_level,
            scenario_description=scenario_data["scenario_description"],
            scenario_data=scenario_data.get("scenario_data", {}),
            ai_trap=scenario_data.get("ai_trap") if include_ai_trap else None,
            expected_reasoning=scenario_data.get("expected_reasoning", []),
            correct_approach=scenario_data.get("correct_approach", {}),
            target_skills=target_skills,
            generated_by="llm_assisted",
            source_case_id=scenario_data.get("source_case_id"),
        )

        logger.info(
            "Challenge scenario generated",
            tenant_id=str(tenant_id),
            challenge_id=str(challenge.id),
            domain=domain,
            difficulty_level=difficulty_level,
            include_ai_trap=include_ai_trap,
        )

        await self._publisher.publish(
            Topics.CRITICAL_THINKING,
            {
                "event_type": "critical.challenge.generated",
                "tenant_id": str(tenant_id),
                "challenge_id": str(challenge.id),
                "domain": domain,
                "difficulty_level": difficulty_level,
                "include_ai_trap": include_ai_trap,
            },
        )

        return challenge

    async def get_challenge(
        self, challenge_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Challenge:
        """Retrieve a challenge scenario by ID.

        Args:
            challenge_id: Challenge UUID.
            tenant_id: Requesting tenant.

        Returns:
            Challenge record.

        Raises:
            NotFoundError: If challenge not found.
        """
        challenge = await self._challenges.get_by_id(challenge_id, tenant_id)
        if challenge is None:
            raise NotFoundError(
                message=f"Challenge {challenge_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return challenge

    async def list_challenges(
        self,
        tenant_id: uuid.UUID,
        domain: str | None = None,
        difficulty_level: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Challenge], int]:
        """List challenge scenarios with optional filters.

        Args:
            tenant_id: Requesting tenant.
            domain: Optional domain filter.
            difficulty_level: Optional difficulty filter.
            status: Optional status filter.
            page: 1-based page number.
            page_size: Results per page.

        Returns:
            Tuple of (challenges, total_count).
        """
        return await self._challenges.list_challenges(
            tenant_id=tenant_id,
            domain=domain,
            difficulty_level=difficulty_level,
            status=status,
            page=page,
            page_size=page_size,
        )


class TrainingRecommenderService:
    """Generate and manage training recommendations to address skill atrophy.

    Produces targeted training programs based on atrophy assessment outcomes
    and bias detection patterns, with challenge assignments for deliberate practice.
    """

    def __init__(
        self,
        recommendation_repo: ITrainingRecommendationRepository,
        challenge_repo: IChallengeRepository,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            recommendation_repo: TrainingRecommendation persistence.
            challenge_repo: Challenge repository for assigning practice scenarios.
            event_publisher: Kafka event publisher.
        """
        self._recommendations = recommendation_repo
        self._challenges = challenge_repo
        self._publisher = event_publisher

    async def create_recommendation(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        assessment_id: uuid.UUID | None,
        recommendation_type: str,
        priority: str,
        target_domain: str,
        program_name: str,
        program_description: str,
        program_modules: list[dict[str, Any]],
        estimated_duration_hours: float,
        target_skill_improvement: float,
        challenge_ids: list[str] | None = None,
    ) -> TrainingRecommendation:
        """Create a training recommendation for a user.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: Target user UUID.
            assessment_id: Optional source atrophy assessment UUID.
            recommendation_type: Type of training intervention.
            priority: Urgency level.
            target_domain: Domain this recommendation addresses.
            program_name: Training program name.
            program_description: Full program description.
            program_modules: Ordered training module list.
            estimated_duration_hours: Total estimated training hours.
            target_skill_improvement: Expected skill score delta.
            challenge_ids: Optional UUIDs of challenge scenarios to include.

        Returns:
            Newly created TrainingRecommendation record.

        Raises:
            ConflictError: If recommendation_type is invalid.
        """
        if recommendation_type not in VALID_RECOMMENDATION_TYPES:
            raise ConflictError(
                message=f"Invalid recommendation_type '{recommendation_type}'. Valid: {VALID_RECOMMENDATION_TYPES}",
                error_code=ErrorCode.INVALID_OPERATION,
            )

        recommendation = await self._recommendations.create(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_id=assessment_id,
            recommendation_type=recommendation_type,
            priority=priority,
            target_domain=target_domain,
            program_name=program_name,
            program_description=program_description,
            program_modules=program_modules,
            estimated_duration_hours=estimated_duration_hours,
            challenge_ids=challenge_ids or [],
            target_skill_improvement=target_skill_improvement,
        )

        logger.info(
            "Training recommendation created",
            tenant_id=str(tenant_id),
            user_id=str(user_id),
            recommendation_id=str(recommendation.id),
            recommendation_type=recommendation_type,
            priority=priority,
            target_domain=target_domain,
        )

        await self._publisher.publish(
            Topics.CRITICAL_THINKING,
            {
                "event_type": "critical.training.recommendation_created",
                "tenant_id": str(tenant_id),
                "user_id": str(user_id),
                "recommendation_id": str(recommendation.id),
                "recommendation_type": recommendation_type,
                "priority": priority,
                "target_domain": target_domain,
            },
        )

        return recommendation

    async def get_recommendation(
        self, recommendation_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> TrainingRecommendation:
        """Retrieve a training recommendation by ID.

        Args:
            recommendation_id: TrainingRecommendation UUID.
            tenant_id: Requesting tenant.

        Returns:
            TrainingRecommendation record.

        Raises:
            NotFoundError: If recommendation not found.
        """
        recommendation = await self._recommendations.get_by_id(recommendation_id, tenant_id)
        if recommendation is None:
            raise NotFoundError(
                message=f"Training recommendation {recommendation_id} not found.",
                error_code=ErrorCode.NOT_FOUND,
            )
        return recommendation

    async def list_recommendations(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        target_domain: str | None = None,
        priority: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[TrainingRecommendation], int]:
        """List training recommendations with optional filters.

        Args:
            tenant_id: Requesting tenant.
            user_id: Optional user filter.
            target_domain: Optional domain filter.
            priority: Optional priority filter.
            status: Optional status filter.
            page: 1-based page number.
            page_size: Results per page.

        Returns:
            Tuple of (recommendations, total_count).
        """
        return await self._recommendations.list_recommendations(
            tenant_id=tenant_id,
            user_id=user_id,
            target_domain=target_domain,
            priority=priority,
            status=status,
            page=page,
            page_size=page_size,
        )

    async def recommend_from_assessment(
        self,
        tenant_id: uuid.UUID,
        assessment: AtrophyAssessment,
    ) -> TrainingRecommendation:
        """Auto-generate a training recommendation from an atrophy assessment.

        Selects recommendation type, priority, and program based on the
        assessment's atrophy severity and identified skill gaps.

        Args:
            tenant_id: Owning tenant UUID.
            assessment: Source AtrophyAssessment to base the recommendation on.

        Returns:
            Newly created TrainingRecommendation record.
        """
        # Map severity to recommendation type and priority
        severity_map: dict[str, tuple[str, str]] = {
            "none": ("challenge_practice", "low"),
            "low": ("judgment_calibration", "low"),
            "moderate": ("judgment_calibration", "medium"),
            "high": ("skill_restoration", "high"),
            "critical": ("skill_restoration", "critical"),
        }
        recommendation_type, priority = severity_map.get(
            assessment.atrophy_severity, ("skill_restoration", "high")
        )

        # Find matching challenges for the domain
        challenges, _ = await self._challenges.list_challenges(
            tenant_id=tenant_id,
            domain=assessment.assessment_domain,
            difficulty_level=None,
            status="active",
            page=1,
            page_size=5,
        )
        challenge_ids = [str(c.id) for c in challenges]

        # Build program modules from skill gaps
        program_modules: list[dict[str, Any]] = []
        for gap in assessment.skill_gaps[:5]:  # Top 5 gaps
            program_modules.append({
                "module_name": f"Restore: {gap.get('skill', 'core judgment')}",
                "duration_hours": 2.0,
                "content_type": "scenario_practice",
                "objectives": [f"Rebuild proficiency in {gap.get('skill', 'domain judgment')}"],
            })

        if not program_modules:
            program_modules = [{
                "module_name": f"Critical Thinking in {assessment.assessment_domain}",
                "duration_hours": 4.0,
                "content_type": "mixed",
                "objectives": ["Maintain independent judgment", "Reduce AI over-reliance"],
            }]

        estimated_hours = sum(m["duration_hours"] for m in program_modules)
        target_improvement = min(0.2, abs(assessment.atrophy_rate) + 0.05)

        return await self.create_recommendation(
            tenant_id=tenant_id,
            user_id=assessment.user_id,
            assessment_id=assessment.id,
            recommendation_type=recommendation_type,
            priority=priority,
            target_domain=assessment.assessment_domain,
            program_name=f"{assessment.assessment_domain.replace('_', ' ').title()} Judgment Restoration",
            program_description=(
                f"Targeted training to address {assessment.atrophy_severity} skill atrophy "
                f"in {assessment.assessment_domain}. Current skill score: {assessment.current_score:.2f}. "
                f"Target improvement: +{target_improvement:.2f}."
            ),
            program_modules=program_modules,
            estimated_duration_hours=estimated_hours,
            target_skill_improvement=target_improvement,
            challenge_ids=challenge_ids,
        )

    async def update_status(
        self,
        recommendation_id: uuid.UUID,
        tenant_id: uuid.UUID,
        status: str,
        outcome_score: float | None = None,
    ) -> TrainingRecommendation:
        """Update a recommendation's lifecycle status.

        Args:
            recommendation_id: TrainingRecommendation UUID.
            tenant_id: Owning tenant.
            status: New status value.
            outcome_score: Optional post-completion assessment score.

        Returns:
            Updated TrainingRecommendation record.

        Raises:
            NotFoundError: If recommendation not found.
        """
        await self.get_recommendation(recommendation_id, tenant_id)  # validates existence

        now = datetime.now(tz=timezone.utc)
        accepted_at = now if status == "in_progress" else None
        completed_at = now if status == "completed" else None

        recommendation = await self._recommendations.update_status(
            recommendation_id=recommendation_id,
            tenant_id=tenant_id,
            status=status,
            accepted_at=accepted_at,
            completed_at=completed_at,
            outcome_score=outcome_score,
        )

        if status == "completed":
            await self._publisher.publish(
                Topics.CRITICAL_THINKING,
                {
                    "event_type": "critical.training.recommendation_completed",
                    "tenant_id": str(tenant_id),
                    "user_id": str(recommendation.user_id),
                    "recommendation_id": str(recommendation_id),
                    "outcome_score": outcome_score,
                },
            )

        return recommendation


# ---------------------------------------------------------------------------
# Phase 5 adapter-wrapper services (added with new domain adapters)
# ---------------------------------------------------------------------------


class ReasoningFrameworkService:
    """Orchestrates structured reasoning using CoT and ToT strategies.

    Wraps the ReasoningFramework adapter with service-level strategy selection
    and structured logging, providing a clean interface for route handlers.
    """

    VALID_STRATEGIES: frozenset[str] = frozenset(
        {"chain_of_thought", "tree_of_thought", "auto"}
    )

    def __init__(self, reasoning_adapter: IReasoningFrameworkAdapter) -> None:
        """Initialise ReasoningFrameworkService.

        Args:
            reasoning_adapter: IReasoningFrameworkAdapter implementation.
        """
        self._reasoning = reasoning_adapter

    async def reason(
        self,
        problem: str,
        context: dict[str, Any] | None = None,
        strategy: str = "auto",
        max_steps: int = 8,
    ) -> Any:
        """Build a complete reasoning trace for a problem.

        Args:
            problem: Problem or question to reason about.
            context: Optional context dict with domain and constraints.
            strategy: chain_of_thought | tree_of_thought | auto.
            max_steps: Maximum steps / branching limit.

        Returns:
            ReasoningTrace dataclass with selected path and metadata.

        Raises:
            ValueError: If strategy is not valid.
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {strategy!r}. Valid: {sorted(self.VALID_STRATEGIES)}"
            )
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1. Got {max_steps}.")

        trace = await self._reasoning.create_reasoning_trace(
            problem=problem,
            context=context,
            strategy=strategy,
            max_steps=max_steps,
        )
        logger.info(
            "Reasoning trace created",
            strategy=strategy,
            max_steps=max_steps,
        )
        return trace

    async def explore_alternatives(
        self,
        problem: str,
        context: dict[str, Any] | None = None,
        branching_factor: int = 3,
        max_depth: int = 4,
    ) -> list[Any]:
        """Explore multiple reasoning paths using Tree-of-Thought.

        Args:
            problem: Problem to reason about.
            context: Optional context dict.
            branching_factor: Number of branches at each node.
            max_depth: Maximum tree depth.

        Returns:
            List of ReasoningPath dataclasses sorted by confidence.
        """
        paths = await self._reasoning.explore_tree_of_thought(
            problem=problem,
            context=context,
            branching_factor=branching_factor,
            max_depth=max_depth,
        )
        logger.info(
            "Tree-of-Thought exploration complete",
            branching_factor=branching_factor,
            max_depth=max_depth,
            paths_explored=len(paths),
        )
        return paths


class ArgumentAnalysisService:
    """Extracts and analyses arguments from text.

    Wraps ArgumentExtractor with service-level validation and provides
    argument graphs and strength scoring for downstream reasoning services.
    """

    def __init__(self, extractor_adapter: IArgumentExtractorAdapter) -> None:
        """Initialise ArgumentAnalysisService.

        Args:
            extractor_adapter: IArgumentExtractorAdapter implementation.
        """
        self._extractor = extractor_adapter

    async def analyse_text(
        self,
        text: str,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Extract arguments, build graph, and score strength.

        Args:
            text: Source text to analyse.
            domain: Optional domain context.

        Returns:
            Dict with arguments, argument_graph, and scored_arguments keys.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot analyse empty text.")

        arguments = await self._extractor.extract_arguments(text=text, domain=domain)
        graph = await self._extractor.build_argument_graph(arguments=arguments)
        scored = [
            {
                "argument": arg,
                "strength": await self._extractor.score_argument_strength(arg),
            }
            for arg in arguments
        ]

        logger.info(
            "Argument analysis complete",
            argument_count=len(arguments),
            domain=domain,
        )
        return {
            "arguments": arguments,
            "argument_graph": graph,
            "scored_arguments": scored,
        }


class FallacyDetectionService:
    """Detects and reports logical fallacies in text.

    Wraps FallacyDetector and provides structured fallacy reports
    suitable for direct display or further training pipeline use.
    """

    def __init__(self, fallacy_adapter: IFallacyDetectorAdapter) -> None:
        """Initialise FallacyDetectionService.

        Args:
            fallacy_adapter: IFallacyDetectorAdapter implementation.
        """
        self._detector = fallacy_adapter

    async def analyse(
        self,
        text: str,
        context: str | None = None,
    ) -> Any:
        """Detect fallacies and return a structured report.

        Args:
            text: Source text to analyse.
            context: Optional contextual description.

        Returns:
            FallacyReport dataclass with severity and recommendations.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot analyse empty text for fallacies.")

        detections = await self._detector.detect_fallacies(text=text, context=context)
        report = await self._detector.generate_report(
            text=text,
            detections=detections,
            context=context,
        )
        logger.info(
            "Fallacy detection complete",
            fallacy_count=len(detections),
        )
        return report


class EvidenceAnalysisService:
    """Extracts claims, fact-checks them, and builds evidence chains.

    Wraps EvidenceGatherer with service-level orchestration, coordinating
    claim extraction, fact-checking, and citation generation.
    """

    def __init__(self, evidence_adapter: IEvidenceGathererAdapter) -> None:
        """Initialise EvidenceAnalysisService.

        Args:
            evidence_adapter: IEvidenceGathererAdapter implementation.
        """
        self._evidence = evidence_adapter

    async def analyse_claims(
        self,
        text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Extract claims, fact-check each one, and build an evidence chain.

        Args:
            text: Source text to analyse.
            context: Optional domain context.

        Returns:
            Dict with claims, fact_checks, and evidence_chain keys.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot analyse empty text for evidence.")

        claims = await self._evidence.extract_claims(text=text)
        fact_checks = [
            await self._evidence.fact_check(claim=claim, context=context)
            for claim in claims
        ]
        evidence_chain = await self._evidence.build_evidence_chain(
            claims=claims,
            context=context,
        )

        logger.info(
            "Evidence analysis complete",
            claim_count=len(claims),
        )
        return {
            "claims": claims,
            "fact_checks": fact_checks,
            "evidence_chain": evidence_chain,
        }


class CognitiveBiasService:
    """Detects cognitive reasoning biases in text and arguments.

    Distinct from BiasDetectorService (which detects automation bias).
    This service detects cognitive biases like confirmation bias, anchoring,
    availability heuristic, etc., in written reasoning and arguments.
    """

    def __init__(self, bias_adapter: ICognitiveBiasDetectorAdapter) -> None:
        """Initialise CognitiveBiasService.

        Args:
            bias_adapter: ICognitiveBiasDetectorAdapter implementation.
        """
        self._detector = bias_adapter

    async def detect(
        self,
        text: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Detect cognitive biases and generate mitigation recommendations.

        Args:
            text: Source text to analyse.
            context: Optional domain context.

        Returns:
            Dict with detection_result and mitigations keys.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot analyse empty text for cognitive biases.")

        result = await self._detector.detect_biases(text=text, context=context)
        mitigations = await self._detector.recommend_mitigations(result=result)

        logger.info(
            "Cognitive bias detection complete",
            bias_count=len(getattr(result, "detected_biases", [])),
        )
        return {
            "detection_result": result,
            "mitigations": mitigations,
        }


class AlternativeHypothesisService:
    """Generates and compares alternative hypotheses.

    Wraps AlternativeGenerator with service-level validation and provides
    devil's advocate arguments and comparison matrices for structured analysis.
    """

    def __init__(self, generator_adapter: IAlternativeGeneratorAdapter) -> None:
        """Initialise AlternativeHypothesisService.

        Args:
            generator_adapter: IAlternativeGeneratorAdapter implementation.
        """
        self._generator = generator_adapter

    async def generate_and_compare(
        self,
        hypothesis: str,
        context: dict[str, Any] | None = None,
        count: int = 5,
        include_devil_advocate: bool = True,
    ) -> dict[str, Any]:
        """Generate alternatives, optionally add devil's advocate, and compare.

        Args:
            hypothesis: The hypothesis to generate alternatives for.
            context: Optional domain context.
            count: Number of alternative hypotheses to generate.
            include_devil_advocate: True to include a devil's advocate argument.

        Returns:
            Dict with alternatives, devil_advocate, and comparison_matrix keys.

        Raises:
            ValueError: If hypothesis is empty or count is not positive.
        """
        if not hypothesis.strip():
            raise ValueError("Hypothesis cannot be empty.")
        if count < 1:
            raise ValueError(f"count must be at least 1. Got {count}.")

        alternatives = await self._generator.generate_alternatives(
            hypothesis=hypothesis,
            context=context,
            count=count,
        )

        devil_advocate = None
        if include_devil_advocate:
            devil_advocate = await self._generator.devil_advocate(
                hypothesis=hypothesis,
                context=context,
            )
            all_hypotheses = alternatives + ([devil_advocate] if devil_advocate else [])
        else:
            all_hypotheses = alternatives

        comparison_matrix = None
        if all_hypotheses:
            comparison_matrix = await self._generator.build_comparison_matrix(
                hypotheses=all_hypotheses,
                dimensions=None,
            )

        logger.info(
            "Alternative hypothesis generation complete",
            alternatives_count=len(alternatives),
            include_devil_advocate=include_devil_advocate,
        )
        return {
            "alternatives": alternatives,
            "devil_advocate": devil_advocate,
            "comparison_matrix": comparison_matrix,
        }


class ConfidenceScoringService:
    """Scores and calibrates reasoning confidence.

    Wraps ReasoningConfidenceScorer (a pure-computation adapter) with
    service-level orchestration and overconfidence flagging.
    """

    def __init__(self, scorer_adapter: IConfidenceScorerAdapter) -> None:
        """Initialise ConfidenceScoringService.

        Args:
            scorer_adapter: IConfidenceScorerAdapter implementation.
        """
        self._scorer = scorer_adapter

    def score(
        self,
        claim: str,
        evidence_items: list[Any],
        assumptions: list[str],
        reasoning_steps: list[Any],
    ) -> dict[str, Any]:
        """Generate a full confidence report for a claim.

        Args:
            claim: The claim being evaluated.
            evidence_items: Supporting evidence items.
            assumptions: List of underlying assumptions.
            reasoning_steps: ReasoningStep dataclasses used.

        Returns:
            Dict with report and is_overconfident keys.

        Raises:
            ValueError: If claim is empty.
        """
        if not claim.strip():
            raise ValueError("Claim cannot be empty for confidence scoring.")

        report = self._scorer.generate_report(
            claim=claim,
            evidence_items=evidence_items,
            assumptions=assumptions,
            reasoning_steps=reasoning_steps,
        )
        is_overconfident = self._scorer.detect_overconfidence(report)

        if is_overconfident:
            logger.warning(
                "Overconfidence detected in reasoning claim",
                claim_preview=claim[:80],
                composite_score=getattr(report, "composite_score", None),
            )
        else:
            logger.info(
                "Confidence scoring complete",
                composite_score=getattr(report, "composite_score", None),
            )

        return {
            "report": report,
            "is_overconfident": is_overconfident,
        }


class DebateSimulationService:
    """Orchestrates structured debate simulations.

    Wraps DebateSimulator with service-level parameter validation and
    provides full transcript generation for analysis and training.
    """

    def __init__(self, debate_adapter: IDebateSimulatorAdapter) -> None:
        """Initialise DebateSimulationService.

        Args:
            debate_adapter: IDebateSimulatorAdapter implementation.
        """
        self._simulator = debate_adapter

    async def simulate(
        self,
        proposition: str,
        rounds: int = 3,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Run a full structured debate for a proposition.

        Args:
            proposition: The proposition to debate.
            rounds: Number of argument/rebuttal rounds per side (1-10).
            context: Optional domain context.

        Returns:
            DebateTranscript dataclass with full argument history and verdict.

        Raises:
            ValueError: If proposition is empty or rounds is out of range.
        """
        if not proposition.strip():
            raise ValueError("Proposition cannot be empty for debate simulation.")
        if not (1 <= rounds <= 10):
            raise ValueError(f"rounds must be between 1 and 10. Got {rounds}.")

        transcript = await self._simulator.run_debate(
            proposition=proposition,
            rounds=rounds,
            context=context,
        )
        verdict = getattr(transcript, "verdict", "unknown")
        logger.info(
            "Debate simulation complete",
            rounds=rounds,
            verdict=verdict,
        )
        return transcript


class SkillAtrophyService:
    """Manages skill proficiency tracking and atrophy monitoring.

    Wraps AtrophyMonitor adapter with service-level multi-user session
    management, alert dispatch, and refresher recommendation generation.
    Complements AtrophyMonitorService (which handles DB persistence)
    by providing real-time in-memory decay tracking.
    """

    def __init__(self, atrophy_adapter: IAtrophyMonitorAdapter) -> None:
        """Initialise SkillAtrophyService.

        Args:
            atrophy_adapter: IAtrophyMonitorAdapter implementation.
        """
        self._monitor = atrophy_adapter

    def record_usage(
        self,
        user_id: str,
        skill_name: str,
        skill_domain: str,
        current_proficiency: float,
    ) -> Any:
        """Record a skill usage event and reset its decay clock.

        Args:
            user_id: User identifier string.
            skill_name: Name of the skill used.
            skill_domain: Domain category of the skill.
            current_proficiency: Current proficiency score (0.0–1.0).

        Returns:
            Updated SkillRecord dataclass.

        Raises:
            ValueError: If current_proficiency is not in 0.0–1.0.
        """
        if not 0.0 <= current_proficiency <= 1.0:
            raise ValueError(
                f"current_proficiency must be in [0.0, 1.0]. Got {current_proficiency}."
            )

        record = self._monitor.update_skill_usage(
            user_id=user_id,
            skill_name=skill_name,
            skill_domain=skill_domain,
            current_proficiency=current_proficiency,
        )
        logger.info(
            "Skill usage recorded",
            user_id=user_id,
            skill_name=skill_name,
            current_proficiency=current_proficiency,
        )
        return record

    def decay_and_alert(
        self,
        user_id: str,
        skill_name: str,
        days_elapsed: float,
    ) -> dict[str, Any]:
        """Apply decay to a skill and return any triggered alerts.

        Args:
            user_id: User identifier string.
            skill_name: Name of the skill to decay.
            days_elapsed: Days elapsed since last usage.

        Returns:
            Dict with updated_record and alerts keys.

        Raises:
            ValueError: If days_elapsed is negative.
        """
        if days_elapsed < 0:
            raise ValueError(f"days_elapsed cannot be negative. Got {days_elapsed}.")

        updated_record = self._monitor.apply_decay(
            user_id=user_id,
            skill_name=skill_name,
            days_elapsed=days_elapsed,
        )
        alerts = self._monitor.check_and_dispatch_alerts(user_id=user_id)

        if alerts:
            logger.warning(
                "Skill atrophy alerts triggered",
                user_id=user_id,
                alert_count=len(alerts),
            )

        return {
            "updated_record": updated_record,
            "alerts": alerts,
        }

    def get_recommendations(self, user_id: str) -> list[str]:
        """Get refresher training recommendations for at-risk skills.

        Args:
            user_id: User identifier string.

        Returns:
            List of recommendation strings.
        """
        recommendations = self._monitor.get_refresher_recommendations(user_id=user_id)
        logger.info(
            "Refresher recommendations generated",
            user_id=user_id,
            count=len(recommendations),
        )
        return recommendations
