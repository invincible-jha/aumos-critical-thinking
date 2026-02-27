"""Atrophy monitor adapter for the AumOS Critical Thinking service.

Per-user skill tracking, usage frequency monitoring, exponential decay modeling,
refresher recommendation triggers, skill assessment quiz generation, proficiency
trend analysis, and atrophy alert dispatch.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

@dataclass
class SkillRecord:
    """Current proficiency record for a user in a specific skill.

    Attributes:
        record_id: Unique identifier.
        user_id: UUID of the user.
        skill_name: Name of the skill being tracked.
        domain: Domain this skill belongs to.
        current_proficiency: Current proficiency score (0.0–1.0).
        baseline_proficiency: Proficiency at last assessment (0.0–1.0).
        last_used_at: Timestamp of last skill usage.
        usage_frequency_days: Average days between skill usages.
        total_uses: Cumulative usage count.
        decay_rate_per_day: Exponential decay constant (higher = faster decay).
        is_at_risk: True if proficiency has fallen below risk threshold.
    """

    record_id: str
    user_id: uuid.UUID
    skill_name: str
    domain: str
    current_proficiency: float
    baseline_proficiency: float
    last_used_at: datetime
    usage_frequency_days: float
    total_uses: int
    decay_rate_per_day: float
    is_at_risk: bool = False


@dataclass
class DecayProjection:
    """Projected proficiency over a future time horizon.

    Attributes:
        skill_name: Skill being projected.
        current_proficiency: Current score.
        projections: Dict mapping days_ahead to projected_score.
        days_to_risk_threshold: Days until proficiency falls below risk threshold.
        days_to_critical: Days until proficiency falls below critical threshold.
    """

    skill_name: str
    current_proficiency: float
    projections: dict[int, float]
    days_to_risk_threshold: float | None
    days_to_critical: float | None


@dataclass
class SkillAssessmentQuiz:
    """Generated skill assessment quiz.

    Attributes:
        quiz_id: Unique identifier.
        user_id: Target user UUID.
        skill_name: Skill being assessed.
        domain: Domain of the skill.
        questions: List of question dicts with question, options, correct_answer.
        estimated_duration_minutes: Estimated time to complete.
        difficulty_level: beginner | intermediate | advanced | expert.
        generated_at: Generation timestamp.
    """

    quiz_id: str
    user_id: uuid.UUID
    skill_name: str
    domain: str
    questions: list[dict[str, Any]]
    estimated_duration_minutes: int
    difficulty_level: str
    generated_at: datetime


@dataclass
class AtrophyAlert:
    """Alert triggered when a skill falls below a threshold.

    Attributes:
        alert_id: Unique identifier.
        user_id: Target user UUID.
        skill_name: Skill that triggered the alert.
        domain: Skill domain.
        current_proficiency: Proficiency at alert time.
        threshold_breached: risk | critical.
        recommended_action: Short description of recommended intervention.
        days_since_last_use: Days elapsed since last skill usage.
        dispatched_at: Alert dispatch timestamp.
    """

    alert_id: str
    user_id: uuid.UUID
    skill_name: str
    domain: str
    current_proficiency: float
    threshold_breached: str
    recommended_action: str
    days_since_last_use: float
    dispatched_at: datetime


@dataclass
class ProficiencyTrend:
    """Trend analysis for a user's skill proficiency over time.

    Attributes:
        user_id: Target user UUID.
        skill_name: Skill being analysed.
        data_points: List of (timestamp, proficiency) tuples.
        trend_direction: improving | stable | declining.
        trend_rate: Rate of change per week (positive = improving).
        forecast_30_days: Projected proficiency in 30 days.
    """

    user_id: uuid.UUID
    skill_name: str
    data_points: list[tuple[datetime, float]]
    trend_direction: str
    trend_rate: float
    forecast_30_days: float


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AtrophyMonitor:
    """Monitors and models skill atrophy for individual users.

    Applies exponential decay modeling to track proficiency degradation
    over time from disuse, generates assessment quizzes to measure current
    proficiency, triggers alerts when thresholds are breached, and produces
    refresher recommendations when decay reaches actionable levels.
    """

    # Proficiency thresholds
    RISK_THRESHOLD: float = 0.60
    CRITICAL_THRESHOLD: float = 0.40
    REFRESHER_TRIGGER_THRESHOLD: float = 0.70

    # Default exponential decay rate per day for different skill types
    DEFAULT_DECAY_RATES: dict[str, float] = {
        "procedural": 0.008,    # Procedural skills decay faster
        "conceptual": 0.004,    # Conceptual knowledge is more persistent
        "analytical": 0.005,
        "default": 0.006,
    }

    # Quiz difficulty mapping based on current proficiency
    DIFFICULTY_MAP: dict[tuple[float, float], str] = {
        (0.0, 0.40): "beginner",
        (0.40, 0.65): "intermediate",
        (0.65, 0.85): "advanced",
        (0.85, 1.01): "expert",
    }

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "default",
        alert_dispatcher: Any | None = None,
    ) -> None:
        """Initialise the atrophy monitor.

        Args:
            llm_client: LLM client for quiz generation.
            model_name: Provider-agnostic model identifier.
            alert_dispatcher: Optional callback/publisher for alert dispatch.
        """
        self._llm = llm_client
        self._model_name = model_name
        self._alert_dispatcher = alert_dispatcher
        self._skill_records: dict[str, SkillRecord] = {}  # In-memory store for demo

    def update_skill_usage(
        self,
        user_id: uuid.UUID,
        skill_name: str,
        domain: str,
        proficiency_observed: float | None = None,
        skill_type: str = "default",
    ) -> SkillRecord:
        """Record a skill usage event and update the skill record.

        Resets the decay clock and optionally records an observed proficiency
        measurement. Computes updated usage frequency from the interval since
        last use.

        Args:
            user_id: User performing the skill.
            skill_name: Name of the skill.
            domain: Domain of the skill.
            proficiency_observed: Optional observed proficiency at this use.
            skill_type: procedural | conceptual | analytical | default.

        Returns:
            Updated SkillRecord.
        """
        record_key = f"{user_id}:{skill_name}"
        now = datetime.now(tz=timezone.utc)

        if record_key in self._skill_records:
            existing = self._skill_records[record_key]
            days_since_last = (now - existing.last_used_at).total_seconds() / 86400
            # Update rolling average frequency
            new_frequency = (existing.usage_frequency_days * existing.total_uses + days_since_last) / (existing.total_uses + 1)
            new_proficiency = proficiency_observed if proficiency_observed is not None else existing.current_proficiency
            updated = SkillRecord(
                record_id=existing.record_id,
                user_id=user_id,
                skill_name=skill_name,
                domain=domain,
                current_proficiency=min(1.0, new_proficiency),
                baseline_proficiency=existing.baseline_proficiency,
                last_used_at=now,
                usage_frequency_days=round(new_frequency, 2),
                total_uses=existing.total_uses + 1,
                decay_rate_per_day=self.DEFAULT_DECAY_RATES.get(skill_type, self.DEFAULT_DECAY_RATES["default"]),
                is_at_risk=new_proficiency < self.RISK_THRESHOLD,
            )
        else:
            initial_proficiency = proficiency_observed or 0.80
            updated = SkillRecord(
                record_id=str(uuid.uuid4()),
                user_id=user_id,
                skill_name=skill_name,
                domain=domain,
                current_proficiency=initial_proficiency,
                baseline_proficiency=initial_proficiency,
                last_used_at=now,
                usage_frequency_days=7.0,  # Default weekly assumption
                total_uses=1,
                decay_rate_per_day=self.DEFAULT_DECAY_RATES.get(skill_type, self.DEFAULT_DECAY_RATES["default"]),
                is_at_risk=False,
            )

        self._skill_records[record_key] = updated
        logger.info(
            "Skill usage recorded",
            user_id=str(user_id),
            skill_name=skill_name,
            current_proficiency=updated.current_proficiency,
        )
        return updated

    def apply_decay(
        self,
        record: SkillRecord,
        as_of: datetime | None = None,
    ) -> SkillRecord:
        """Apply exponential decay to a skill record based on elapsed time.

        Proficiency decays as: P(t) = P0 * exp(-decay_rate * days_elapsed)
        where P0 is the proficiency at last use and days_elapsed is the
        time since last_used_at.

        Args:
            record: The SkillRecord to apply decay to.
            as_of: Reference timestamp for decay computation (default: now).

        Returns:
            Updated SkillRecord with decayed proficiency.
        """
        reference = as_of or datetime.now(tz=timezone.utc)
        days_elapsed = max(0.0, (reference - record.last_used_at).total_seconds() / 86400)

        decayed_proficiency = record.current_proficiency * math.exp(
            -record.decay_rate_per_day * days_elapsed
        )
        decayed_proficiency = max(0.0, min(1.0, decayed_proficiency))

        record.current_proficiency = round(decayed_proficiency, 4)
        record.is_at_risk = decayed_proficiency < self.RISK_THRESHOLD

        logger.debug(
            "Decay applied",
            skill_name=record.skill_name,
            days_elapsed=days_elapsed,
            decayed_proficiency=decayed_proficiency,
        )
        return record

    def project_decay(
        self,
        record: SkillRecord,
        horizon_days: int = 90,
        checkpoints: list[int] | None = None,
    ) -> DecayProjection:
        """Project future proficiency under the current decay model.

        Args:
            record: The SkillRecord to project.
            horizon_days: Projection horizon in days.
            checkpoints: Specific day offsets to include in projections.
                Defaults to [7, 14, 30, 60, 90].

        Returns:
            DecayProjection with per-checkpoint proficiency values.
        """
        default_checkpoints = checkpoints or [7, 14, 30, 60, 90]
        projections: dict[int, float] = {}

        for days_ahead in default_checkpoints:
            if days_ahead > horizon_days:
                break
            projected = record.current_proficiency * math.exp(
                -record.decay_rate_per_day * days_ahead
            )
            projections[days_ahead] = round(max(0.0, projected), 4)

        # Compute days until risk and critical thresholds
        def days_to_threshold(threshold: float) -> float | None:
            if record.current_proficiency <= threshold:
                return 0.0
            if record.decay_rate_per_day <= 0:
                return None
            return -math.log(threshold / record.current_proficiency) / record.decay_rate_per_day

        return DecayProjection(
            skill_name=record.skill_name,
            current_proficiency=record.current_proficiency,
            projections=projections,
            days_to_risk_threshold=days_to_threshold(self.RISK_THRESHOLD),
            days_to_critical=days_to_threshold(self.CRITICAL_THRESHOLD),
        )

    async def generate_assessment_quiz(
        self,
        user_id: uuid.UUID,
        skill_name: str,
        domain: str,
        current_proficiency: float,
        question_count: int = 5,
    ) -> SkillAssessmentQuiz:
        """Generate a skill assessment quiz calibrated to current proficiency.

        Args:
            user_id: Target user UUID.
            skill_name: Skill to assess.
            domain: Domain of the skill.
            current_proficiency: Current proficiency for difficulty calibration.
            question_count: Number of quiz questions to generate.

        Returns:
            SkillAssessmentQuiz with generated questions.
        """
        difficulty = self._map_difficulty(current_proficiency)

        import json as json_module
        prompt = (
            f"Generate {question_count} assessment questions for the skill '{skill_name}' "
            f"in the '{domain}' domain. Difficulty level: {difficulty}.\n\n"
            "Each question should test genuine skill proficiency. Include multiple-choice "
            "options where appropriate. Format each question as: question (str), "
            "options (list[str] or null), correct_answer (str), explanation (str).\n\n"
            f'Return JSON: {{"questions": [<{question_count} question objects>]}}'
        )
        result = await self._call_llm_json(prompt)
        questions = result.get("questions", [])

        quiz = SkillAssessmentQuiz(
            quiz_id=str(uuid.uuid4()),
            user_id=user_id,
            skill_name=skill_name,
            domain=domain,
            questions=questions[:question_count],
            estimated_duration_minutes=question_count * 2,
            difficulty_level=difficulty,
            generated_at=datetime.now(tz=timezone.utc),
        )

        logger.info(
            "Assessment quiz generated",
            quiz_id=quiz.quiz_id,
            skill_name=skill_name,
            difficulty=difficulty,
            question_count=len(quiz.questions),
        )
        return quiz

    async def check_and_dispatch_alerts(
        self,
        records: list[SkillRecord],
    ) -> list[AtrophyAlert]:
        """Check skill records for threshold breaches and dispatch alerts.

        Args:
            records: List of SkillRecord objects to evaluate.

        Returns:
            List of dispatched AtrophyAlert objects.
        """
        now = datetime.now(tz=timezone.utc)
        dispatched_alerts: list[AtrophyAlert] = []

        for record in records:
            # Apply current decay before checking
            decayed = self.apply_decay(record)

            if decayed.current_proficiency <= self.CRITICAL_THRESHOLD:
                threshold_label = "critical"
                action = f"Immediate refresher training required for '{record.skill_name}'."
            elif decayed.current_proficiency <= self.RISK_THRESHOLD:
                threshold_label = "risk"
                action = f"Schedule refresher session for '{record.skill_name}' within 2 weeks."
            else:
                continue

            days_since_use = (now - record.last_used_at).total_seconds() / 86400

            alert = AtrophyAlert(
                alert_id=str(uuid.uuid4()),
                user_id=record.user_id,
                skill_name=record.skill_name,
                domain=record.domain,
                current_proficiency=decayed.current_proficiency,
                threshold_breached=threshold_label,
                recommended_action=action,
                days_since_last_use=round(days_since_use, 1),
                dispatched_at=now,
            )
            dispatched_alerts.append(alert)

            if self._alert_dispatcher is not None:
                try:
                    await self._alert_dispatcher(alert)
                except Exception as exc:
                    logger.error("Alert dispatch failed", alert_id=alert.alert_id, error=str(exc))

            logger.info(
                "Atrophy alert dispatched",
                alert_id=alert.alert_id,
                user_id=str(record.user_id),
                skill_name=record.skill_name,
                threshold=threshold_label,
            )

        return dispatched_alerts

    def analyse_proficiency_trend(
        self,
        user_id: uuid.UUID,
        skill_name: str,
        historical_points: list[tuple[datetime, float]],
    ) -> ProficiencyTrend:
        """Analyse proficiency trend from historical data points.

        Uses simple linear regression over the historical data to determine
        trend direction and rate.

        Args:
            user_id: Target user UUID.
            skill_name: Skill being analysed.
            historical_points: List of (timestamp, proficiency) tuples in chronological order.

        Returns:
            ProficiencyTrend with direction, rate, and 30-day forecast.
        """
        if len(historical_points) < 2:
            return ProficiencyTrend(
                user_id=user_id,
                skill_name=skill_name,
                data_points=historical_points,
                trend_direction="stable",
                trend_rate=0.0,
                forecast_30_days=historical_points[-1][1] if historical_points else 0.5,
            )

        # Convert timestamps to days since first observation
        t0 = historical_points[0][0]
        days = [(pt[0] - t0).total_seconds() / 86400 for pt in historical_points]
        scores = [pt[1] for pt in historical_points]

        n = len(days)
        mean_t = sum(days) / n
        mean_s = sum(scores) / n
        numerator = sum((days[i] - mean_t) * (scores[i] - mean_s) for i in range(n))
        denominator = sum((days[i] - mean_t) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0
        trend_rate_per_week = slope * 7

        if trend_rate_per_week > 0.02:
            direction = "improving"
        elif trend_rate_per_week < -0.02:
            direction = "declining"
        else:
            direction = "stable"

        last_score = scores[-1]
        last_day = days[-1]
        forecast = min(1.0, max(0.0, last_score + slope * (last_day + 30 - last_day)))

        return ProficiencyTrend(
            user_id=user_id,
            skill_name=skill_name,
            data_points=historical_points,
            trend_direction=direction,
            trend_rate=round(trend_rate_per_week, 4),
            forecast_30_days=round(forecast, 4),
        )

    def get_refresher_recommendations(
        self, records: list[SkillRecord]
    ) -> list[dict[str, Any]]:
        """Generate refresher recommendations for skills approaching thresholds.

        Args:
            records: List of SkillRecord objects to evaluate.

        Returns:
            List of recommendation dicts with skill_name, urgency, and action.
        """
        recommendations: list[dict[str, Any]] = []

        for record in records:
            decayed = self.apply_decay(record)
            projection = self.project_decay(decayed, horizon_days=30)

            if decayed.current_proficiency <= self.CRITICAL_THRESHOLD:
                urgency = "immediate"
                action = f"Critical: schedule intensive refresher for '{record.skill_name}' this week."
            elif decayed.current_proficiency <= self.RISK_THRESHOLD:
                urgency = "high"
                action = f"At risk: schedule refresher for '{record.skill_name}' within 14 days."
            elif decayed.current_proficiency <= self.REFRESHER_TRIGGER_THRESHOLD:
                days_to_risk = projection.days_to_risk_threshold
                urgency = "medium"
                days_text = f" in approximately {days_to_risk:.0f} days" if days_to_risk else ""
                action = f"Preventive refresher recommended for '{record.skill_name}'{days_text}."
            else:
                continue

            recommendations.append({
                "skill_name": record.skill_name,
                "domain": record.domain,
                "current_proficiency": decayed.current_proficiency,
                "urgency": urgency,
                "action": action,
                "days_to_risk": projection.days_to_risk_threshold,
            })

        recommendations.sort(
            key=lambda r: {"immediate": 0, "high": 1, "medium": 2}.get(r["urgency"], 3)
        )
        return recommendations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _map_difficulty(self, proficiency: float) -> str:
        """Map proficiency score to quiz difficulty level.

        Args:
            proficiency: Current proficiency score (0.0–1.0).

        Returns:
            Difficulty level string.
        """
        for (lower, upper), difficulty in self.DIFFICULTY_MAP.items():
            if lower <= proficiency < upper:
                return difficulty
        return "intermediate"

    async def _call_llm_json(self, prompt: str) -> Any:
        """Call LLM and return parsed JSON.

        Args:
            prompt: Prompt to send.

        Returns:
            Parsed JSON dict.
        """
        import json
        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model_name,
                response_format={"type": "json_object"},
            )
            raw = response.text if hasattr(response, "text") else str(response)
            return json.loads(raw)
        except Exception as exc:
            logger.error("LLM call failed in atrophy monitor", error=str(exc))
            return {}
