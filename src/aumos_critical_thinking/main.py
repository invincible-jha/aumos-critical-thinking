"""AumOS Critical Thinking service entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_critical_thinking.adapters.kafka import CriticalThinkingEventPublisher
from aumos_critical_thinking.adapters.repositories import (
    AtrophyAssessmentRepository,
    BiasDetectionRepository,
    ChallengeRepository,
    JudgmentValidationRepository,
    TrainingRecommendationRepository,
)
from aumos_critical_thinking.api.router import router
from aumos_critical_thinking.core.services import (
    AtrophyMonitorService,
    BiasDetectorService,
    ChallengeGeneratorService,
    JudgmentValidatorService,
    TrainingRecommenderService,
)
from aumos_critical_thinking.settings import Settings

logger = get_logger(__name__)
settings = Settings()

_kafka_publisher: CriticalThinkingEventPublisher | None = None


class _StubChallengeGenerator:
    """Stub challenge generator adapter for startup.

    In production, replace with an LLM-backed implementation via
    ChallengeGeneratorAdapter(settings.llm_model_id).
    """

    async def generate_scenario(
        self,
        domain: str,
        difficulty_level: str,
        target_skills: list[str],
        atrophy_context: dict | None,
        include_ai_trap: bool,
    ) -> dict:
        """Return a minimal stub scenario for health-check purposes."""
        return {
            "title": f"[Stub] {domain.title()} {difficulty_level.title()} Challenge",
            "scenario_description": "Stub scenario — replace with LLM-generated content in production.",
            "scenario_data": {"context": "stub", "data_points": [], "constraints": []},
            "ai_trap": {"recommendation": "stub_trap", "confidence": 0.9} if include_ai_trap else None,
            "expected_reasoning": [{"step": 1, "concept": "critical_evaluation", "rationale": "Always verify AI output", "weight": 1.0}],
            "correct_approach": {"outcome": "independent_evaluation", "rationale": "Apply domain expertise independently"},
            "target_skills": target_skills,
            "source_case_id": None,
        }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initialises the database connection pool, Kafka event publisher,
    repositories, and services, then exposes them on app.state for DI.

    Args:
        app: The FastAPI application instance.

    Yields:
        None
    """
    global _kafka_publisher  # noqa: PLW0603

    logger.info("Starting AumOS Critical Thinking", version="0.1.0")

    # Database connection pool
    init_database(settings.database)
    logger.info("Database connection pool ready")

    # Kafka event publisher
    _kafka_publisher = CriticalThinkingEventPublisher(settings.kafka)
    await _kafka_publisher.start()
    app.state.kafka_publisher = _kafka_publisher
    logger.info("Kafka event publisher ready")

    # Repositories
    bias_repo = BiasDetectionRepository()
    judgment_repo = JudgmentValidationRepository()
    atrophy_repo = AtrophyAssessmentRepository()
    challenge_repo = ChallengeRepository()
    training_repo = TrainingRecommendationRepository()

    # Generator adapter (stub — replace with real LLM adapter in production)
    challenge_generator = _StubChallengeGenerator()

    # Services wired with DI
    app.state.bias_service = BiasDetectorService(
        bias_repo=bias_repo,
        event_publisher=_kafka_publisher,
        severe_bias_threshold=settings.severe_bias_threshold,
    )
    app.state.judgment_service = JudgmentValidatorService(
        validation_repo=judgment_repo,
        event_publisher=_kafka_publisher,
        low_accuracy_threshold=settings.low_accuracy_threshold,
    )
    app.state.atrophy_service = AtrophyMonitorService(
        atrophy_repo=atrophy_repo,
        event_publisher=_kafka_publisher,
        intervention_threshold=settings.atrophy_intervention_threshold,
    )
    app.state.challenge_service = ChallengeGeneratorService(
        challenge_repo=challenge_repo,
        generator_adapter=challenge_generator,  # type: ignore[arg-type]
        event_publisher=_kafka_publisher,
    )
    app.state.training_service = TrainingRecommenderService(
        recommendation_repo=training_repo,
        challenge_repo=challenge_repo,
        event_publisher=_kafka_publisher,
    )

    # Expose settings on app state
    app.state.settings = settings

    logger.info("Critical Thinking service startup complete")
    yield

    # Shutdown
    if _kafka_publisher:
        await _kafka_publisher.stop()

    logger.info("Critical Thinking service shutdown complete")


app: FastAPI = create_app(
    service_name="aumos-critical-thinking",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=lambda: None),
        HealthCheck(name="kafka", check_fn=lambda: None),
    ],
)

app.include_router(router, prefix="/api/v1")
