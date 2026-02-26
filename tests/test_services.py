"""Unit tests for AumOS Critical Thinking services.

Tests use mock repositories and event publishers to isolate service logic.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from aumos_common.errors import ConflictError, NotFoundError

from aumos_critical_thinking.core.services import (
    AtrophyMonitorService,
    BiasDetectorService,
    ChallengeGeneratorService,
    JudgmentValidatorService,
    TrainingRecommenderService,
    _classify_atrophy,
    _classify_bias,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestClassifyBias:
    def test_none_category(self) -> None:
        assert _classify_bias(0.0) == "none"
        assert _classify_bias(0.24) == "none"

    def test_mild_category(self) -> None:
        assert _classify_bias(0.25) == "mild"
        assert _classify_bias(0.49) == "mild"

    def test_moderate_category(self) -> None:
        assert _classify_bias(0.50) == "moderate"
        assert _classify_bias(0.74) == "moderate"

    def test_severe_category(self) -> None:
        assert _classify_bias(0.75) == "severe"
        assert _classify_bias(1.0) == "severe"


class TestClassifyAtrophy:
    def test_none_severity_for_improvement(self) -> None:
        assert _classify_atrophy(0.0) == "none"

    def test_low_severity(self) -> None:
        assert _classify_atrophy(0.01) == "low"
        assert _classify_atrophy(0.04) == "low"

    def test_moderate_severity(self) -> None:
        assert _classify_atrophy(0.05) == "moderate"
        assert _classify_atrophy(0.14) == "moderate"

    def test_high_severity(self) -> None:
        assert _classify_atrophy(0.15) == "high"
        assert _classify_atrophy(0.29) == "high"

    def test_critical_severity(self) -> None:
        assert _classify_atrophy(0.30) == "critical"
        assert _classify_atrophy(0.99) == "critical"


# ---------------------------------------------------------------------------
# BiasDetectorService tests
# ---------------------------------------------------------------------------


class TestBiasDetectorService:
    @pytest.mark.asyncio
    async def test_detect_bias_creates_record(
        self,
        bias_service: BiasDetectorService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mock_bias_repo: AsyncMock,
    ) -> None:
        """Detect bias should create and return a BiasDetection record."""
        result = await bias_service.detect_bias(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id="session-001",
            decision_context="model_deployment",
            ai_recommendation={"outcome": "approve", "confidence": 0.9},
            human_decision={"outcome": "approve"},
            review_duration_seconds=5,
        )
        assert result.bias_score >= 0.0
        assert result.bias_score <= 1.0
        mock_bias_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_bias_invalid_context_raises(
        self,
        bias_service: BiasDetectorService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Invalid decision context should raise ConflictError."""
        with pytest.raises(ConflictError):
            await bias_service.detect_bias(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id="session-001",
                decision_context="invalid_context",
                ai_recommendation={"outcome": "approve"},
                human_decision={"outcome": "approve"},
            )

    @pytest.mark.asyncio
    async def test_severe_bias_publishes_event(
        self,
        mock_bias_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        sample_bias_detection: object,
    ) -> None:
        """Severe bias score should trigger a Kafka event publication."""
        # Override repo to return a severe bias score
        from aumos_critical_thinking.core.models import BiasDetection
        severe_detection = BiasDetection.__new__(BiasDetection)
        severe_detection.id = uuid.uuid4()
        severe_detection.bias_score = 0.9
        severe_detection.bias_category = "severe"
        severe_detection.user_id = user_id
        severe_detection.deviation_indicators = ["immediate_acceptance"]
        mock_bias_repo.create = AsyncMock(return_value=severe_detection)

        service = BiasDetectorService(
            bias_repo=mock_bias_repo,
            event_publisher=mock_event_publisher,
            severe_bias_threshold=0.75,
        )

        await service.detect_bias(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id="session-001",
            decision_context="model_deployment",
            ai_recommendation={"outcome": "approve"},
            human_decision={"outcome": "approve"},
            review_duration_seconds=3,
        )

        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args[0]
        assert call_args[1]["event_type"] == "critical.bias.severe_detected"

    @pytest.mark.asyncio
    async def test_get_detection_not_found_raises(
        self,
        bias_service: BiasDetectorService,
        mock_bias_repo: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Getting a non-existent detection should raise NotFoundError."""
        mock_bias_repo.get_by_id = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await bias_service.get_detection(uuid.uuid4(), tenant_id)

    def test_analyse_decision_override_reduces_bias(
        self, bias_service: BiasDetectorService
    ) -> None:
        """Overriding the AI recommendation should result in lower bias score."""
        score_no_override, _, _ = bias_service._analyse_decision(
            ai_recommendation={"outcome": "approve"},
            human_decision={"outcome": "approve"},
            review_duration_seconds=120,
        )
        score_with_override, _, _ = bias_service._analyse_decision(
            ai_recommendation={"outcome": "approve"},
            human_decision={"outcome": "reject", "rationale": "Risk too high"},
            review_duration_seconds=120,
        )
        assert score_with_override < score_no_override

    def test_immediate_acceptance_adds_indicator(
        self, bias_service: BiasDetectorService
    ) -> None:
        """Very fast review should add immediate_acceptance indicator."""
        _, indicators, _ = bias_service._analyse_decision(
            ai_recommendation={"outcome": "approve"},
            human_decision={"outcome": "approve"},
            review_duration_seconds=5,
        )
        assert "immediate_acceptance" in indicators


# ---------------------------------------------------------------------------
# JudgmentValidatorService tests
# ---------------------------------------------------------------------------


class TestJudgmentValidatorService:
    @pytest.mark.asyncio
    async def test_validate_judgment_creates_record(
        self,
        judgment_service: JudgmentValidatorService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mock_judgment_repo: AsyncMock,
    ) -> None:
        """Validate judgment should create and return a JudgmentValidation record."""
        result = await judgment_service.validate_judgment(
            tenant_id=tenant_id,
            user_id=user_id,
            decision_domain="model_risk",
            decision_id="deployment-001",
            human_judgment={"outcome": "low_risk"},
            reference_standard={"outcome": "low_risk"},
            validation_method="expert_consensus",
        )
        assert result.is_valid is True
        mock_judgment_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_low_accuracy_publishes_event(
        self,
        mock_judgment_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Low accuracy judgment should publish a Kafka event."""
        from aumos_critical_thinking.core.models import JudgmentValidation
        low_accuracy_validation = JudgmentValidation.__new__(JudgmentValidation)
        low_accuracy_validation.id = uuid.uuid4()
        low_accuracy_validation.user_id = user_id
        low_accuracy_validation.is_valid = False
        low_accuracy_validation.accuracy_score = 0.3
        low_accuracy_validation.divergence_analysis = {}
        mock_judgment_repo.create = AsyncMock(return_value=low_accuracy_validation)

        service = JudgmentValidatorService(
            validation_repo=mock_judgment_repo,
            event_publisher=mock_event_publisher,
            low_accuracy_threshold=0.6,
        )

        await service.validate_judgment(
            tenant_id=tenant_id,
            user_id=user_id,
            decision_domain="model_risk",
            decision_id="deployment-002",
            human_judgment={"outcome": "low_risk"},
            reference_standard={"outcome": "high_risk"},
            validation_method="ground_truth",
        )

        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args[0]
        assert call_args[1]["event_type"] == "critical.judgment.low_accuracy"

    def test_score_judgment_correct_outcome(
        self, judgment_service: JudgmentValidatorService
    ) -> None:
        """Matching outcome should yield high accuracy score."""
        score, is_valid, _, _ = judgment_service._score_judgment(
            human_judgment={"outcome": "approve"},
            reference_standard={"outcome": "approve"},
        )
        assert is_valid is True
        assert score >= 0.7  # At least 70% for matching outcome

    def test_score_judgment_wrong_outcome(
        self, judgment_service: JudgmentValidatorService
    ) -> None:
        """Non-matching outcome should yield lower accuracy score."""
        score, is_valid, divergence, _ = judgment_service._score_judgment(
            human_judgment={"outcome": "approve"},
            reference_standard={"outcome": "reject"},
        )
        assert is_valid is False
        assert score < 0.7
        assert divergence["outcome_match"] is False

    @pytest.mark.asyncio
    async def test_get_validation_not_found_raises(
        self,
        judgment_service: JudgmentValidatorService,
        mock_judgment_repo: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Getting a non-existent validation should raise NotFoundError."""
        mock_judgment_repo.get_by_id = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await judgment_service.get_validation(uuid.uuid4(), tenant_id)


# ---------------------------------------------------------------------------
# AtrophyMonitorService tests
# ---------------------------------------------------------------------------


class TestAtrophyMonitorService:
    @pytest.mark.asyncio
    async def test_assess_atrophy_no_baseline(
        self,
        atrophy_service: AtrophyMonitorService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mock_atrophy_repo: AsyncMock,
    ) -> None:
        """First assessment with no prior baseline should have atrophy_rate=0."""
        mock_atrophy_repo.get_latest_for_user_domain = AsyncMock(return_value=None)
        result = await atrophy_service.assess_atrophy(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_domain="clinical_judgment",
            assessment_period_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            assessment_period_end=datetime(2026, 1, 31, tzinfo=timezone.utc),
            current_score=0.80,
            ai_reliance_ratio=0.5,
            independent_decision_count=20,
            ai_assisted_decision_count=20,
        )
        assert result is not None
        mock_atrophy_repo.create.assert_called_once()
        # With no baseline, atrophy rate should be 0
        create_kwargs = mock_atrophy_repo.create.call_args[1]
        assert create_kwargs["atrophy_rate"] == 0.0
        assert create_kwargs["baseline_score"] is None

    @pytest.mark.asyncio
    async def test_assess_atrophy_with_baseline_computes_rate(
        self,
        atrophy_service: AtrophyMonitorService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mock_atrophy_repo: AsyncMock,
        sample_atrophy_assessment: object,
    ) -> None:
        """Assessment with prior baseline should compute correct atrophy rate."""
        from aumos_critical_thinking.core.models import AtrophyAssessment
        prior = AtrophyAssessment.__new__(AtrophyAssessment)
        prior.current_score = 0.85
        mock_atrophy_repo.get_latest_for_user_domain = AsyncMock(return_value=prior)

        await atrophy_service.assess_atrophy(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_domain="clinical_judgment",
            assessment_period_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            assessment_period_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
            current_score=0.70,
            ai_reliance_ratio=0.75,
            independent_decision_count=5,
            ai_assisted_decision_count=15,
        )

        create_kwargs = mock_atrophy_repo.create.call_args[1]
        assert create_kwargs["atrophy_rate"] == 0.15  # 0.85 - 0.70
        assert create_kwargs["baseline_score"] == 0.85

    @pytest.mark.asyncio
    async def test_critical_atrophy_publishes_event(
        self,
        mock_atrophy_repo: AsyncMock,
        mock_event_publisher: AsyncMock,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        sample_atrophy_assessment: object,
    ) -> None:
        """Critical atrophy severity should publish intervention event."""
        from aumos_critical_thinking.core.models import AtrophyAssessment
        prior = AtrophyAssessment.__new__(AtrophyAssessment)
        prior.current_score = 0.85
        mock_atrophy_repo.get_latest_for_user_domain = AsyncMock(return_value=prior)

        critical_assessment = AtrophyAssessment.__new__(AtrophyAssessment)
        critical_assessment.id = uuid.uuid4()
        critical_assessment.atrophy_severity = "critical"
        critical_assessment.atrophy_rate = 0.40
        mock_atrophy_repo.create = AsyncMock(return_value=critical_assessment)

        service = AtrophyMonitorService(
            atrophy_repo=mock_atrophy_repo,
            event_publisher=mock_event_publisher,
            intervention_threshold="high",
        )

        await service.assess_atrophy(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_domain="clinical_judgment",
            assessment_period_start=datetime(2026, 2, 1, tzinfo=timezone.utc),
            assessment_period_end=datetime(2026, 2, 28, tzinfo=timezone.utc),
            current_score=0.45,
            ai_reliance_ratio=0.95,
            independent_decision_count=1,
            ai_assisted_decision_count=19,
        )

        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args[0]
        assert call_args[1]["event_type"] == "critical.atrophy.intervention_required"


# ---------------------------------------------------------------------------
# ChallengeGeneratorService tests
# ---------------------------------------------------------------------------


class TestChallengeGeneratorService:
    @pytest.mark.asyncio
    async def test_generate_challenge_creates_record(
        self,
        challenge_service: ChallengeGeneratorService,
        tenant_id: uuid.UUID,
        mock_challenge_repo: AsyncMock,
        mock_challenge_generator: AsyncMock,
    ) -> None:
        """Generate challenge should call adapter and create a Challenge record."""
        result = await challenge_service.generate_challenge(
            tenant_id=tenant_id,
            domain="model_risk",
            difficulty_level="advanced",
            target_skills=["critical_thinking"],
        )
        assert result is not None
        mock_challenge_generator.generate_scenario.assert_called_once()
        mock_challenge_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_challenge_invalid_difficulty_raises(
        self,
        challenge_service: ChallengeGeneratorService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Invalid difficulty level should raise ConflictError."""
        with pytest.raises(ConflictError):
            await challenge_service.generate_challenge(
                tenant_id=tenant_id,
                domain="model_risk",
                difficulty_level="super_hard",
                target_skills=["critical_thinking"],
            )

    @pytest.mark.asyncio
    async def test_get_challenge_not_found_raises(
        self,
        challenge_service: ChallengeGeneratorService,
        mock_challenge_repo: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Getting a non-existent challenge should raise NotFoundError."""
        mock_challenge_repo.get_by_id = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await challenge_service.get_challenge(uuid.uuid4(), tenant_id)

    @pytest.mark.asyncio
    async def test_generate_challenge_publishes_event(
        self,
        challenge_service: ChallengeGeneratorService,
        tenant_id: uuid.UUID,
        mock_event_publisher: AsyncMock,
    ) -> None:
        """Challenge generation should publish a Kafka event."""
        await challenge_service.generate_challenge(
            tenant_id=tenant_id,
            domain="security",
            difficulty_level="intermediate",
            target_skills=["threat_analysis"],
            include_ai_trap=True,
        )
        mock_event_publisher.publish.assert_called_once()
        call_args = mock_event_publisher.publish.call_args[0]
        assert call_args[1]["event_type"] == "critical.challenge.generated"


# ---------------------------------------------------------------------------
# TrainingRecommenderService tests
# ---------------------------------------------------------------------------


class TestTrainingRecommenderService:
    @pytest.mark.asyncio
    async def test_create_recommendation_creates_record(
        self,
        training_service: TrainingRecommenderService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        mock_training_repo: AsyncMock,
    ) -> None:
        """Create recommendation should create and return a TrainingRecommendation."""
        result = await training_service.create_recommendation(
            tenant_id=tenant_id,
            user_id=user_id,
            assessment_id=None,
            recommendation_type="skill_restoration",
            priority="high",
            target_domain="clinical_judgment",
            program_name="Clinical Judgment Restoration",
            program_description="Targeted training program",
            program_modules=[{"module_name": "Module 1", "duration_hours": 2.0}],
            estimated_duration_hours=2.0,
            target_skill_improvement=0.15,
        )
        assert result is not None
        mock_training_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_recommendation_invalid_type_raises(
        self,
        training_service: TrainingRecommenderService,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> None:
        """Invalid recommendation_type should raise ConflictError."""
        with pytest.raises(ConflictError):
            await training_service.create_recommendation(
                tenant_id=tenant_id,
                user_id=user_id,
                assessment_id=None,
                recommendation_type="invalid_type",
                priority="high",
                target_domain="clinical_judgment",
                program_name="Test",
                program_description="Test",
                program_modules=[],
                estimated_duration_hours=1.0,
                target_skill_improvement=0.1,
            )

    @pytest.mark.asyncio
    async def test_recommend_from_assessment_maps_severity(
        self,
        training_service: TrainingRecommenderService,
        tenant_id: uuid.UUID,
        sample_atrophy_assessment: object,
        mock_training_repo: AsyncMock,
        mock_challenge_repo: AsyncMock,
    ) -> None:
        """recommend_from_assessment should map severity to appropriate priority."""
        from aumos_critical_thinking.core.models import AtrophyAssessment
        assessment = AtrophyAssessment.__new__(AtrophyAssessment)
        assessment.id = uuid.uuid4()
        assessment.tenant_id = tenant_id
        assessment.user_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
        assessment.assessment_domain = "clinical_judgment"
        assessment.atrophy_severity = "high"
        assessment.atrophy_rate = 0.20
        assessment.current_score = 0.65
        assessment.skill_gaps = []

        await training_service.recommend_from_assessment(
            tenant_id=tenant_id,
            assessment=assessment,
        )

        mock_training_repo.create.assert_called_once()
        create_kwargs = mock_training_repo.create.call_args[1]
        assert create_kwargs["priority"] == "high"
        assert create_kwargs["recommendation_type"] == "skill_restoration"

    @pytest.mark.asyncio
    async def test_get_recommendation_not_found_raises(
        self,
        training_service: TrainingRecommenderService,
        mock_training_repo: AsyncMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Getting a non-existent recommendation should raise NotFoundError."""
        mock_training_repo.get_by_id = AsyncMock(return_value=None)
        with pytest.raises(NotFoundError):
            await training_service.get_recommendation(uuid.uuid4(), tenant_id)
