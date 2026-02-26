"""Shared test fixtures for the AumOS Critical Thinking service."""

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_critical_thinking.core.models import (
    AtrophyAssessment,
    BiasDetection,
    Challenge,
    JudgmentValidation,
    TrainingRecommendation,
)
from aumos_critical_thinking.core.services import (
    AtrophyMonitorService,
    BiasDetectorService,
    ChallengeGeneratorService,
    JudgmentValidatorService,
    TrainingRecommenderService,
)


@pytest.fixture
def tenant_id() -> uuid.UUID:
    """Provide a fixed tenant UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def user_id() -> uuid.UUID:
    """Provide a fixed user UUID for tests."""
    return uuid.UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture
def mock_event_publisher() -> AsyncMock:
    """Provide a mock Kafka event publisher."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    return publisher


@pytest.fixture
def sample_bias_detection(tenant_id: uuid.UUID, user_id: uuid.UUID) -> BiasDetection:
    """Provide a sample BiasDetection instance."""
    detection = BiasDetection.__new__(BiasDetection)
    detection.id = uuid.uuid4()
    detection.tenant_id = tenant_id
    detection.user_id = user_id
    detection.session_id = "session-001"
    detection.decision_context = "model_deployment"
    detection.ai_recommendation = {"outcome": "approve", "confidence": 0.95}
    detection.human_decision = {"outcome": "approve", "rationale": None}
    detection.bias_score = 0.65
    detection.bias_category = "moderate"
    detection.deviation_indicators = ["no_independent_rationale"]
    detection.review_duration_seconds = 8
    detection.override_occurred = False
    detection.override_rationale = None
    detection.metadata = {}
    detection.created_at = datetime.now(tz=timezone.utc)
    detection.updated_at = datetime.now(tz=timezone.utc)
    return detection


@pytest.fixture
def sample_judgment_validation(tenant_id: uuid.UUID, user_id: uuid.UUID) -> JudgmentValidation:
    """Provide a sample JudgmentValidation instance."""
    validation = JudgmentValidation.__new__(JudgmentValidation)
    validation.id = uuid.uuid4()
    validation.tenant_id = tenant_id
    validation.user_id = user_id
    validation.decision_domain = "model_risk"
    validation.decision_id = "deployment-abc-123"
    validation.human_judgment = {"outcome": "low_risk", "confidence": 0.8}
    validation.reference_standard = {"outcome": "low_risk", "reasoning_steps": [{"concept": "data_quality"}]}
    validation.validation_method = "expert_consensus"
    validation.is_valid = True
    validation.accuracy_score = 0.85
    validation.confidence_calibration = 0.95
    validation.divergence_analysis = {"outcome_match": True, "reasoning_coverage": 0.8}
    validation.validated_at = datetime.now(tz=timezone.utc)
    validation.validator_id = None
    validation.created_at = datetime.now(tz=timezone.utc)
    validation.updated_at = datetime.now(tz=timezone.utc)
    return validation


@pytest.fixture
def sample_atrophy_assessment(tenant_id: uuid.UUID, user_id: uuid.UUID) -> AtrophyAssessment:
    """Provide a sample AtrophyAssessment instance."""
    assessment = AtrophyAssessment.__new__(AtrophyAssessment)
    assessment.id = uuid.uuid4()
    assessment.tenant_id = tenant_id
    assessment.user_id = user_id
    assessment.assessment_domain = "clinical_judgment"
    assessment.assessment_period_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assessment.assessment_period_end = datetime(2026, 1, 31, tzinfo=timezone.utc)
    assessment.baseline_score = 0.85
    assessment.current_score = 0.70
    assessment.atrophy_rate = 0.15
    assessment.atrophy_severity = "moderate"
    assessment.ai_reliance_ratio = 0.75
    assessment.independent_decision_count = 10
    assessment.ai_assisted_decision_count = 30
    assessment.skill_gaps = [{"skill": "diagnostic_reasoning", "severity": "moderate"}]
    assessment.intervention_required = False
    assessment.notes = None
    assessment.created_at = datetime.now(tz=timezone.utc)
    assessment.updated_at = datetime.now(tz=timezone.utc)
    return assessment


@pytest.fixture
def sample_challenge(tenant_id: uuid.UUID) -> Challenge:
    """Provide a sample Challenge instance."""
    challenge = Challenge.__new__(Challenge)
    challenge.id = uuid.uuid4()
    challenge.tenant_id = tenant_id
    challenge.title = "Model Risk Assessment Under Uncertainty"
    challenge.domain = "model_risk"
    challenge.difficulty_level = "advanced"
    challenge.scenario_description = "A production model shows unusual drift patterns."
    challenge.scenario_data = {"context": "production", "data_points": []}
    challenge.ai_trap = {"recommendation": "ignore_drift", "confidence": 0.92}
    challenge.expected_reasoning = [{"step": 1, "concept": "drift_analysis", "rationale": "Investigate root cause", "weight": 1.0}]
    challenge.correct_approach = {"outcome": "investigate_drift", "rationale": "Never ignore drift without analysis"}
    challenge.target_skills = ["drift_detection", "risk_assessment"]
    challenge.status = "active"
    challenge.times_used = 0
    challenge.average_score = None
    challenge.generated_by = "llm_assisted"
    challenge.source_case_id = None
    challenge.created_at = datetime.now(tz=timezone.utc)
    challenge.updated_at = datetime.now(tz=timezone.utc)
    return challenge


@pytest.fixture
def mock_bias_repo(sample_bias_detection: BiasDetection) -> AsyncMock:
    """Provide a mock IBiasDetectionRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=sample_bias_detection)
    repo.get_by_id = AsyncMock(return_value=sample_bias_detection)
    repo.list_by_tenant = AsyncMock(return_value=([sample_bias_detection], 1))
    repo.get_user_bias_summary = AsyncMock(return_value={
        "total_detections": 1,
        "avg_bias_score": 0.65,
        "by_category": {"moderate": 1},
        "override_rate": 0.0,
    })
    return repo


@pytest.fixture
def mock_judgment_repo(sample_judgment_validation: JudgmentValidation) -> AsyncMock:
    """Provide a mock IJudgmentValidationRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=sample_judgment_validation)
    repo.get_by_id = AsyncMock(return_value=sample_judgment_validation)
    repo.list_by_user = AsyncMock(return_value=([sample_judgment_validation], 1))
    repo.get_accuracy_trend = AsyncMock(return_value=[
        {"period": 1, "avg_accuracy": 0.85, "count": 5, "is_valid_rate": 0.8}
    ])
    return repo


@pytest.fixture
def mock_atrophy_repo(sample_atrophy_assessment: AtrophyAssessment) -> AsyncMock:
    """Provide a mock IAtrophyAssessmentRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=sample_atrophy_assessment)
    repo.get_by_id = AsyncMock(return_value=sample_atrophy_assessment)
    repo.list_metrics = AsyncMock(return_value=([sample_atrophy_assessment], 1))
    repo.get_latest_for_user_domain = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_challenge_repo(sample_challenge: Challenge) -> AsyncMock:
    """Provide a mock IChallengeRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=sample_challenge)
    repo.get_by_id = AsyncMock(return_value=sample_challenge)
    repo.list_challenges = AsyncMock(return_value=([sample_challenge], 1))
    repo.increment_usage = AsyncMock(return_value=sample_challenge)
    return repo


@pytest.fixture
def mock_training_repo() -> AsyncMock:
    """Provide a mock ITrainingRecommendationRepository."""
    repo = AsyncMock()
    rec = TrainingRecommendation.__new__(TrainingRecommendation)
    rec.id = uuid.uuid4()
    rec.tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    rec.user_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    rec.assessment_id = None
    rec.recommendation_type = "skill_restoration"
    rec.priority = "high"
    rec.target_domain = "clinical_judgment"
    rec.program_name = "Clinical Judgment Restoration"
    rec.program_description = "Targeted training program"
    rec.program_modules = [{"module_name": "Module 1", "duration_hours": 2.0}]
    rec.estimated_duration_hours = 2.0
    rec.challenge_ids = []
    rec.target_skill_improvement = 0.15
    rec.status = "pending"
    rec.accepted_at = None
    rec.completed_at = None
    rec.outcome_score = None
    rec.created_at = datetime.now(tz=timezone.utc)
    rec.updated_at = datetime.now(tz=timezone.utc)
    repo.create = AsyncMock(return_value=rec)
    repo.get_by_id = AsyncMock(return_value=rec)
    repo.list_recommendations = AsyncMock(return_value=([rec], 1))
    repo.update_status = AsyncMock(return_value=rec)
    return repo


@pytest.fixture
def mock_challenge_generator() -> AsyncMock:
    """Provide a mock IChallengeGeneratorAdapter."""
    generator = AsyncMock()
    generator.generate_scenario = AsyncMock(return_value={
        "title": "Test Challenge",
        "scenario_description": "A test scenario for unit testing",
        "scenario_data": {"context": "test"},
        "ai_trap": {"recommendation": "wrong_answer", "confidence": 0.9},
        "expected_reasoning": [{"step": 1, "concept": "critical_review", "rationale": "Always verify", "weight": 1.0}],
        "correct_approach": {"outcome": "correct", "rationale": "Apply expertise"},
        "target_skills": ["critical_thinking"],
        "source_case_id": None,
    })
    return generator


@pytest.fixture
def bias_service(
    mock_bias_repo: AsyncMock,
    mock_event_publisher: AsyncMock,
) -> BiasDetectorService:
    """Provide a BiasDetectorService with mock dependencies."""
    return BiasDetectorService(
        bias_repo=mock_bias_repo,
        event_publisher=mock_event_publisher,
        severe_bias_threshold=0.75,
    )


@pytest.fixture
def judgment_service(
    mock_judgment_repo: AsyncMock,
    mock_event_publisher: AsyncMock,
) -> JudgmentValidatorService:
    """Provide a JudgmentValidatorService with mock dependencies."""
    return JudgmentValidatorService(
        validation_repo=mock_judgment_repo,
        event_publisher=mock_event_publisher,
        low_accuracy_threshold=0.6,
    )


@pytest.fixture
def atrophy_service(
    mock_atrophy_repo: AsyncMock,
    mock_event_publisher: AsyncMock,
) -> AtrophyMonitorService:
    """Provide an AtrophyMonitorService with mock dependencies."""
    return AtrophyMonitorService(
        atrophy_repo=mock_atrophy_repo,
        event_publisher=mock_event_publisher,
        intervention_threshold="high",
    )


@pytest.fixture
def challenge_service(
    mock_challenge_repo: AsyncMock,
    mock_challenge_generator: AsyncMock,
    mock_event_publisher: AsyncMock,
) -> ChallengeGeneratorService:
    """Provide a ChallengeGeneratorService with mock dependencies."""
    return ChallengeGeneratorService(
        challenge_repo=mock_challenge_repo,
        generator_adapter=mock_challenge_generator,
        event_publisher=mock_event_publisher,
    )


@pytest.fixture
def training_service(
    mock_training_repo: AsyncMock,
    mock_challenge_repo: AsyncMock,
    mock_event_publisher: AsyncMock,
) -> TrainingRecommenderService:
    """Provide a TrainingRecommenderService with mock dependencies."""
    return TrainingRecommenderService(
        recommendation_repo=mock_training_repo,
        challenge_repo=mock_challenge_repo,
        event_publisher=mock_event_publisher,
    )
