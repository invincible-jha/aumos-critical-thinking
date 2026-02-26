"""SQLAlchemy repository implementations for the AumOS Critical Thinking service.

Each repository implements the corresponding interface from core/interfaces.py.
All database access uses the AumOS get_db_session context manager which sets
the app.current_tenant RLS parameter before executing queries.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.database import get_db_session
from aumos_common.observability import get_logger

from aumos_critical_thinking.core.models import (
    AtrophyAssessment,
    BiasDetection,
    Challenge,
    JudgmentValidation,
    TrainingRecommendation,
)

logger = get_logger(__name__)


class BiasDetectionRepository:
    """SQLAlchemy implementation of IBiasDetectionRepository."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        session_id: str,
        decision_context: str,
        ai_recommendation: dict[str, Any],
        human_decision: dict[str, Any],
        bias_score: float,
        bias_category: str,
        deviation_indicators: list[str],
        review_duration_seconds: int | None,
        override_occurred: bool,
        override_rationale: str | None,
        metadata: dict[str, Any],
    ) -> BiasDetection:
        """Create and persist a new bias detection record."""
        async with get_db_session(tenant_id) as session:
            detection = BiasDetection(
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
                metadata=metadata,
            )
            session.add(detection)
            await session.flush()
            await session.refresh(detection)
        return detection

    async def get_by_id(
        self, detection_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> BiasDetection | None:
        """Retrieve a bias detection record by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(BiasDetection).where(
                    BiasDetection.id == detection_id,
                    BiasDetection.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        bias_category: str | None,
        decision_context: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[BiasDetection], int]:
        """List bias detection records with optional filters."""
        async with get_db_session(tenant_id) as session:
            query = select(BiasDetection).where(BiasDetection.tenant_id == tenant_id)
            if user_id is not None:
                query = query.where(BiasDetection.user_id == user_id)
            if bias_category is not None:
                query = query.where(BiasDetection.bias_category == bias_category)
            if decision_context is not None:
                query = query.where(BiasDetection.decision_context == decision_context)

            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total: int = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                query.order_by(BiasDetection.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            items = list(result.scalars().all())

        return items, total

    async def get_user_bias_summary(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> dict[str, Any]:
        """Aggregate bias statistics for a user."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(BiasDetection).where(
                    BiasDetection.tenant_id == tenant_id,
                    BiasDetection.user_id == user_id,
                )
            )
            detections = list(result.scalars().all())

        if not detections:
            return {"total_detections": 0, "avg_bias_score": 0.0, "by_category": {}, "override_rate": 0.0}

        total = len(detections)
        avg_score = sum(d.bias_score for d in detections) / total
        override_count = sum(1 for d in detections if d.override_occurred)
        by_category: dict[str, int] = {}
        for detection in detections:
            by_category[detection.bias_category] = by_category.get(detection.bias_category, 0) + 1

        return {
            "total_detections": total,
            "avg_bias_score": round(avg_score, 4),
            "by_category": by_category,
            "override_rate": round(override_count / total, 4),
        }


class JudgmentValidationRepository:
    """SQLAlchemy implementation of IJudgmentValidationRepository."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str,
        decision_id: str,
        human_judgment: dict[str, Any],
        reference_standard: dict[str, Any],
        validation_method: str,
        is_valid: bool,
        accuracy_score: float,
        confidence_calibration: float | None,
        divergence_analysis: dict[str, Any],
        validator_id: uuid.UUID | None,
    ) -> JudgmentValidation:
        """Create and persist a new judgment validation record."""
        async with get_db_session(tenant_id) as session:
            validation = JudgmentValidation(
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
                validated_at=datetime.utcnow(),
                validator_id=validator_id,
            )
            session.add(validation)
            await session.flush()
            await session.refresh(validation)
        return validation

    async def get_by_id(
        self, validation_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> JudgmentValidation | None:
        """Retrieve a judgment validation by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(JudgmentValidation).where(
                    JudgmentValidation.id == validation_id,
                    JudgmentValidation.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_by_user(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[JudgmentValidation], int]:
        """List validation history for a user."""
        async with get_db_session(tenant_id) as session:
            query = select(JudgmentValidation).where(
                JudgmentValidation.tenant_id == tenant_id,
                JudgmentValidation.user_id == user_id,
            )
            if decision_domain is not None:
                query = query.where(JudgmentValidation.decision_domain == decision_domain)

            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total: int = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                query.order_by(JudgmentValidation.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            items = list(result.scalars().all())

        return items, total

    async def get_accuracy_trend(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str | None,
        periods: int,
    ) -> list[dict[str, Any]]:
        """Compute accuracy trend over time for a user."""
        async with get_db_session(tenant_id) as session:
            query = select(JudgmentValidation).where(
                JudgmentValidation.tenant_id == tenant_id,
                JudgmentValidation.user_id == user_id,
            )
            if decision_domain is not None:
                query = query.where(JudgmentValidation.decision_domain == decision_domain)

            result = await session.execute(
                query.order_by(JudgmentValidation.created_at.desc()).limit(periods * 10)
            )
            records = list(result.scalars().all())

        if not records:
            return []

        # Group into periods (simple chunking for now)
        chunk_size = max(1, len(records) // periods)
        trend: list[dict[str, Any]] = []
        for i in range(0, min(len(records), periods * chunk_size), chunk_size):
            chunk = records[i : i + chunk_size]
            trend.append({
                "period": i // chunk_size + 1,
                "avg_accuracy": round(sum(r.accuracy_score for r in chunk) / len(chunk), 4),
                "count": len(chunk),
                "is_valid_rate": round(sum(1 for r in chunk if r.is_valid) / len(chunk), 4),
            })

        return trend


class AtrophyAssessmentRepository:
    """SQLAlchemy implementation of IAtrophyAssessmentRepository."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        assessment_domain: str,
        assessment_period_start: datetime,
        assessment_period_end: datetime,
        baseline_score: float | None,
        current_score: float,
        atrophy_rate: float,
        atrophy_severity: str,
        ai_reliance_ratio: float,
        independent_decision_count: int,
        ai_assisted_decision_count: int,
        skill_gaps: list[dict[str, Any]],
        intervention_required: bool,
        notes: str | None,
    ) -> AtrophyAssessment:
        """Create and persist a new atrophy assessment."""
        async with get_db_session(tenant_id) as session:
            assessment = AtrophyAssessment(
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
                skill_gaps=skill_gaps,
                intervention_required=intervention_required,
                notes=notes,
            )
            session.add(assessment)
            await session.flush()
            await session.refresh(assessment)
        return assessment

    async def get_by_id(
        self, assessment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> AtrophyAssessment | None:
        """Retrieve an atrophy assessment by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(AtrophyAssessment).where(
                    AtrophyAssessment.id == assessment_id,
                    AtrophyAssessment.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_metrics(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        assessment_domain: str | None,
        atrophy_severity: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[AtrophyAssessment], int]:
        """List atrophy assessment metrics with optional filters."""
        async with get_db_session(tenant_id) as session:
            query = select(AtrophyAssessment).where(AtrophyAssessment.tenant_id == tenant_id)
            if user_id is not None:
                query = query.where(AtrophyAssessment.user_id == user_id)
            if assessment_domain is not None:
                query = query.where(AtrophyAssessment.assessment_domain == assessment_domain)
            if atrophy_severity is not None:
                query = query.where(AtrophyAssessment.atrophy_severity == atrophy_severity)

            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total: int = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                query.order_by(AtrophyAssessment.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            items = list(result.scalars().all())

        return items, total

    async def get_latest_for_user_domain(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        assessment_domain: str,
    ) -> AtrophyAssessment | None:
        """Retrieve the most recent assessment for a user in a domain."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(AtrophyAssessment)
                .where(
                    AtrophyAssessment.tenant_id == tenant_id,
                    AtrophyAssessment.user_id == user_id,
                    AtrophyAssessment.assessment_domain == assessment_domain,
                )
                .order_by(AtrophyAssessment.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()


class ChallengeRepository:
    """SQLAlchemy implementation of IChallengeRepository."""

    async def create(
        self,
        tenant_id: uuid.UUID,
        title: str,
        domain: str,
        difficulty_level: str,
        scenario_description: str,
        scenario_data: dict[str, Any],
        ai_trap: dict[str, Any] | None,
        expected_reasoning: list[dict[str, Any]],
        correct_approach: dict[str, Any],
        target_skills: list[str],
        generated_by: str,
        source_case_id: str | None,
    ) -> Challenge:
        """Create and persist a new challenge scenario."""
        async with get_db_session(tenant_id) as session:
            challenge = Challenge(
                tenant_id=tenant_id,
                title=title,
                domain=domain,
                difficulty_level=difficulty_level,
                scenario_description=scenario_description,
                scenario_data=scenario_data,
                ai_trap=ai_trap,
                expected_reasoning=expected_reasoning,
                correct_approach=correct_approach,
                target_skills=target_skills,
                generated_by=generated_by,
                source_case_id=source_case_id,
            )
            session.add(challenge)
            await session.flush()
            await session.refresh(challenge)
        return challenge

    async def get_by_id(
        self, challenge_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Challenge | None:
        """Retrieve a challenge by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(Challenge).where(
                    Challenge.id == challenge_id,
                    Challenge.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_challenges(
        self,
        tenant_id: uuid.UUID,
        domain: str | None,
        difficulty_level: str | None,
        status: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[Challenge], int]:
        """List challenge scenarios with optional filters."""
        async with get_db_session(tenant_id) as session:
            query = select(Challenge).where(Challenge.tenant_id == tenant_id)
            if domain is not None:
                query = query.where(Challenge.domain == domain)
            if difficulty_level is not None:
                query = query.where(Challenge.difficulty_level == difficulty_level)
            if status is not None:
                query = query.where(Challenge.status == status)

            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total: int = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                query.order_by(Challenge.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            items = list(result.scalars().all())

        return items, total

    async def increment_usage(
        self, challenge_id: uuid.UUID, score: float | None
    ) -> Challenge:
        """Increment usage counter and update average score."""
        async with get_db_session(None) as session:  # type: ignore[arg-type]
            result = await session.execute(
                select(Challenge).where(Challenge.id == challenge_id)
            )
            challenge = result.scalar_one()
            challenge.times_used += 1
            if score is not None:
                if challenge.average_score is None:
                    challenge.average_score = score
                else:
                    # Rolling average
                    n = challenge.times_used
                    challenge.average_score = round(
                        (challenge.average_score * (n - 1) + score) / n, 4
                    )
            await session.flush()
            await session.refresh(challenge)
        return challenge


class TrainingRecommendationRepository:
    """SQLAlchemy implementation of ITrainingRecommendationRepository."""

    async def create(
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
        challenge_ids: list[str],
        target_skill_improvement: float,
    ) -> TrainingRecommendation:
        """Create and persist a new training recommendation."""
        async with get_db_session(tenant_id) as session:
            recommendation = TrainingRecommendation(
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
                challenge_ids=challenge_ids,
                target_skill_improvement=target_skill_improvement,
            )
            session.add(recommendation)
            await session.flush()
            await session.refresh(recommendation)
        return recommendation

    async def get_by_id(
        self, recommendation_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> TrainingRecommendation | None:
        """Retrieve a training recommendation by UUID."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(TrainingRecommendation).where(
                    TrainingRecommendation.id == recommendation_id,
                    TrainingRecommendation.tenant_id == tenant_id,
                )
            )
            return result.scalar_one_or_none()

    async def list_recommendations(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        target_domain: str | None,
        priority: str | None,
        status: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[TrainingRecommendation], int]:
        """List training recommendations with optional filters."""
        async with get_db_session(tenant_id) as session:
            query = select(TrainingRecommendation).where(
                TrainingRecommendation.tenant_id == tenant_id
            )
            if user_id is not None:
                query = query.where(TrainingRecommendation.user_id == user_id)
            if target_domain is not None:
                query = query.where(TrainingRecommendation.target_domain == target_domain)
            if priority is not None:
                query = query.where(TrainingRecommendation.priority == priority)
            if status is not None:
                query = query.where(TrainingRecommendation.status == status)

            count_result = await session.execute(
                select(func.count()).select_from(query.subquery())
            )
            total: int = count_result.scalar_one()

            offset = (page - 1) * page_size
            result = await session.execute(
                query.order_by(TrainingRecommendation.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            items = list(result.scalars().all())

        return items, total

    async def update_status(
        self,
        recommendation_id: uuid.UUID,
        tenant_id: uuid.UUID,
        status: str,
        accepted_at: datetime | None,
        completed_at: datetime | None,
        outcome_score: float | None,
    ) -> TrainingRecommendation:
        """Update recommendation lifecycle status."""
        async with get_db_session(tenant_id) as session:
            result = await session.execute(
                select(TrainingRecommendation).where(
                    TrainingRecommendation.id == recommendation_id,
                    TrainingRecommendation.tenant_id == tenant_id,
                )
            )
            recommendation = result.scalar_one()
            recommendation.status = status
            if accepted_at is not None:
                recommendation.accepted_at = accepted_at
            if completed_at is not None:
                recommendation.completed_at = completed_at
            if outcome_score is not None:
                recommendation.outcome_score = outcome_score
            await session.flush()
            await session.refresh(recommendation)
        return recommendation
