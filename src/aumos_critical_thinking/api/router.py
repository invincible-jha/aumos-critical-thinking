"""FastAPI router for the AumOS Critical Thinking REST API.

All endpoints are prefixed with /api/v1. Authentication and tenant extraction
are handled by aumos-auth-gateway upstream; tenant_id is available via JWT.

Business logic is never implemented here — routes delegate entirely to services.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from aumos_common.errors import ConflictError, NotFoundError
from aumos_common.observability import get_logger

from aumos_critical_thinking.api.schemas import (
    AtrophyAssessmentResponse,
    AtrophyAssessRequest,
    AtrophyMetricsResponse,
    BiasDetectRequest,
    BiasDetectionResponse,
    BiasReportListResponse,
    ChallengeGenerateRequest,
    ChallengeListResponse,
    ChallengeResponse,
    JudgmentHistoryResponse,
    JudgmentValidateRequest,
    JudgmentValidationResponse,
    TrainingRecommendationResponse,
    TrainingRecommendationsListResponse,
)
from aumos_critical_thinking.core.services import (
    AtrophyMonitorService,
    BiasDetectorService,
    ChallengeGeneratorService,
    JudgmentValidatorService,
    TrainingRecommenderService,
)

logger = get_logger(__name__)

router = APIRouter(tags=["critical-thinking"])


# ---------------------------------------------------------------------------
# Dependency helpers — replaced by real DI in production startup
# ---------------------------------------------------------------------------


def _get_bias_service(request: Request) -> BiasDetectorService:
    """Retrieve BiasDetectorService from app state."""
    return request.app.state.bias_service  # type: ignore[no-any-return]


def _get_judgment_service(request: Request) -> JudgmentValidatorService:
    """Retrieve JudgmentValidatorService from app state."""
    return request.app.state.judgment_service  # type: ignore[no-any-return]


def _get_atrophy_service(request: Request) -> AtrophyMonitorService:
    """Retrieve AtrophyMonitorService from app state."""
    return request.app.state.atrophy_service  # type: ignore[no-any-return]


def _get_challenge_service(request: Request) -> ChallengeGeneratorService:
    """Retrieve ChallengeGeneratorService from app state."""
    return request.app.state.challenge_service  # type: ignore[no-any-return]


def _get_training_service(request: Request) -> TrainingRecommenderService:
    """Retrieve TrainingRecommenderService from app state."""
    return request.app.state.training_service  # type: ignore[no-any-return]


def _tenant_id_from_request(request: Request) -> uuid.UUID:
    """Extract tenant UUID from request headers (set by auth middleware).

    Falls back to a random UUID in development mode.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Tenant UUID.
    """
    raw = request.headers.get("X-Tenant-ID")
    if raw:
        return uuid.UUID(raw)
    return uuid.uuid4()  # dev fallback


# ---------------------------------------------------------------------------
# Bias Detection endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/critical/bias/detect",
    response_model=BiasDetectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Detect automation bias",
    description="Analyse a human-AI decision event and record an automation bias detection record.",
)
async def detect_bias(
    body: BiasDetectRequest,
    request: Request,
    service: BiasDetectorService = Depends(_get_bias_service),
) -> BiasDetectionResponse:
    """Detect and record automation bias for a decision event.

    Args:
        body: Bias detection request payload.
        request: FastAPI request (for tenant extraction).
        service: BiasDetectorService dependency.

    Returns:
        Created BiasDetectionResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    try:
        detection = await service.detect_bias(
            tenant_id=tenant_id,
            user_id=body.user_id,
            session_id=body.session_id,
            decision_context=body.decision_context,
            ai_recommendation=body.ai_recommendation,
            human_decision=body.human_decision,
            review_duration_seconds=body.review_duration_seconds,
            metadata=body.metadata,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return BiasDetectionResponse.model_validate(detection)


@router.get(
    "/critical/bias/reports",
    response_model=BiasReportListResponse,
    summary="List bias detection reports",
    description="Retrieve a paginated list of bias detection reports with optional filters.",
)
async def list_bias_reports(
    request: Request,
    user_id: uuid.UUID | None = Query(default=None, description="Filter by user UUID"),
    bias_category: str | None = Query(default=None, description="Filter by bias category"),
    decision_context: str | None = Query(default=None, description="Filter by decision context"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    service: BiasDetectorService = Depends(_get_bias_service),
) -> BiasReportListResponse:
    """List bias detection reports for the requesting tenant.

    Args:
        request: FastAPI request (for tenant extraction).
        user_id: Optional user filter.
        bias_category: Optional bias category filter.
        decision_context: Optional decision context filter.
        page: 1-based page number.
        page_size: Results per page.
        service: BiasDetectorService dependency.

    Returns:
        Paginated BiasReportListResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    detections, total = await service.list_reports(
        tenant_id=tenant_id,
        user_id=user_id,
        bias_category=bias_category,
        decision_context=decision_context,
        page=page,
        page_size=page_size,
    )
    return BiasReportListResponse(
        items=[BiasDetectionResponse.model_validate(d) for d in detections],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Judgment Validation endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/critical/judgment/validate",
    response_model=JudgmentValidationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Validate human judgment",
    description="Validate a human decision against a reference standard and record accuracy.",
)
async def validate_judgment(
    body: JudgmentValidateRequest,
    request: Request,
    service: JudgmentValidatorService = Depends(_get_judgment_service),
) -> JudgmentValidationResponse:
    """Validate a human judgment and record the result.

    Args:
        body: Judgment validation request payload.
        request: FastAPI request (for tenant extraction).
        service: JudgmentValidatorService dependency.

    Returns:
        Created JudgmentValidationResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    validation = await service.validate_judgment(
        tenant_id=tenant_id,
        user_id=body.user_id,
        decision_domain=body.decision_domain,
        decision_id=body.decision_id,
        human_judgment=body.human_judgment,
        reference_standard=body.reference_standard,
        validation_method=body.validation_method,
        validator_id=body.validator_id,
    )
    return JudgmentValidationResponse.model_validate(validation)


@router.get(
    "/critical/judgment/history",
    response_model=JudgmentHistoryResponse,
    summary="List judgment validation history",
    description="Retrieve paginated judgment validation history for a user.",
)
async def list_judgment_history(
    request: Request,
    user_id: uuid.UUID = Query(..., description="User UUID to retrieve history for"),
    decision_domain: str | None = Query(default=None, description="Filter by decision domain"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    service: JudgmentValidatorService = Depends(_get_judgment_service),
) -> JudgmentHistoryResponse:
    """List judgment validation history for a user.

    Args:
        request: FastAPI request (for tenant extraction).
        user_id: Target user UUID.
        decision_domain: Optional domain filter.
        page: 1-based page number.
        page_size: Results per page.
        service: JudgmentValidatorService dependency.

    Returns:
        Paginated JudgmentHistoryResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    validations, total = await service.list_history(
        tenant_id=tenant_id,
        user_id=user_id,
        decision_domain=decision_domain,
        page=page,
        page_size=page_size,
    )
    return JudgmentHistoryResponse(
        items=[JudgmentValidationResponse.model_validate(v) for v in validations],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Atrophy Assessment endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/critical/atrophy/assess",
    response_model=AtrophyAssessmentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Assess skill atrophy",
    description="Perform a skill atrophy assessment for a user in a specific domain.",
)
async def assess_atrophy(
    body: AtrophyAssessRequest,
    request: Request,
    service: AtrophyMonitorService = Depends(_get_atrophy_service),
) -> AtrophyAssessmentResponse:
    """Perform an atrophy assessment and record metrics.

    Args:
        body: Atrophy assessment request payload.
        request: FastAPI request (for tenant extraction).
        service: AtrophyMonitorService dependency.

    Returns:
        Created AtrophyAssessmentResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    assessment = await service.assess_atrophy(
        tenant_id=tenant_id,
        user_id=body.user_id,
        assessment_domain=body.assessment_domain,
        assessment_period_start=body.assessment_period_start,
        assessment_period_end=body.assessment_period_end,
        current_score=body.current_score,
        ai_reliance_ratio=body.ai_reliance_ratio,
        independent_decision_count=body.independent_decision_count,
        ai_assisted_decision_count=body.ai_assisted_decision_count,
        skill_gaps=body.skill_gaps,
        notes=body.notes,
    )
    return AtrophyAssessmentResponse.model_validate(assessment)


@router.get(
    "/critical/atrophy/metrics",
    response_model=AtrophyMetricsResponse,
    summary="List atrophy metrics",
    description="Retrieve paginated skill atrophy metrics with optional filters.",
)
async def list_atrophy_metrics(
    request: Request,
    user_id: uuid.UUID | None = Query(default=None, description="Filter by user UUID"),
    assessment_domain: str | None = Query(default=None, description="Filter by assessment domain"),
    atrophy_severity: str | None = Query(default=None, description="Filter by severity level"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    service: AtrophyMonitorService = Depends(_get_atrophy_service),
) -> AtrophyMetricsResponse:
    """List atrophy assessment metrics for the requesting tenant.

    Args:
        request: FastAPI request (for tenant extraction).
        user_id: Optional user filter.
        assessment_domain: Optional domain filter.
        atrophy_severity: Optional severity filter.
        page: 1-based page number.
        page_size: Results per page.
        service: AtrophyMonitorService dependency.

    Returns:
        Paginated AtrophyMetricsResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    assessments, total = await service.list_metrics(
        tenant_id=tenant_id,
        user_id=user_id,
        assessment_domain=assessment_domain,
        atrophy_severity=atrophy_severity,
        page=page,
        page_size=page_size,
    )
    return AtrophyMetricsResponse(
        items=[AtrophyAssessmentResponse.model_validate(a) for a in assessments],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Challenge endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/critical/challenges/generate",
    response_model=ChallengeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate challenge scenario",
    description="Generate a new AI-assisted challenge scenario for human judgment practice.",
)
async def generate_challenge(
    body: ChallengeGenerateRequest,
    request: Request,
    service: ChallengeGeneratorService = Depends(_get_challenge_service),
) -> ChallengeResponse:
    """Generate a challenge scenario.

    Args:
        body: Challenge generation request payload.
        request: FastAPI request (for tenant extraction).
        service: ChallengeGeneratorService dependency.

    Returns:
        Created ChallengeResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    try:
        challenge = await service.generate_challenge(
            tenant_id=tenant_id,
            domain=body.domain,
            difficulty_level=body.difficulty_level,
            target_skills=body.target_skills,
            atrophy_context=body.atrophy_context,
            include_ai_trap=body.include_ai_trap,
        )
    except ConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return ChallengeResponse.model_validate(challenge)


@router.get(
    "/critical/challenges",
    response_model=ChallengeListResponse,
    summary="List challenge scenarios",
    description="Retrieve a paginated list of challenge scenarios with optional filters.",
)
async def list_challenges(
    request: Request,
    domain: str | None = Query(default=None, description="Filter by domain"),
    difficulty_level: str | None = Query(default=None, description="Filter by difficulty level"),
    status_filter: str | None = Query(default=None, alias="status", description="Filter by status"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    service: ChallengeGeneratorService = Depends(_get_challenge_service),
) -> ChallengeListResponse:
    """List challenge scenarios for the requesting tenant.

    Args:
        request: FastAPI request (for tenant extraction).
        domain: Optional domain filter.
        difficulty_level: Optional difficulty filter.
        status_filter: Optional status filter.
        page: 1-based page number.
        page_size: Results per page.
        service: ChallengeGeneratorService dependency.

    Returns:
        Paginated ChallengeListResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    challenges, total = await service.list_challenges(
        tenant_id=tenant_id,
        domain=domain,
        difficulty_level=difficulty_level,
        status=status_filter,
        page=page,
        page_size=page_size,
    )
    return ChallengeListResponse(
        items=[ChallengeResponse.model_validate(c) for c in challenges],
        total=total,
        page=page,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Training Recommendation endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/critical/training/recommendations",
    response_model=TrainingRecommendationsListResponse,
    summary="List training recommendations",
    description="Retrieve paginated training recommendations with optional filters.",
)
async def list_training_recommendations(
    request: Request,
    user_id: uuid.UUID | None = Query(default=None, description="Filter by user UUID"),
    target_domain: str | None = Query(default=None, description="Filter by target domain"),
    priority: str | None = Query(default=None, description="Filter by priority level"),
    status_filter: str | None = Query(default=None, alias="status", description="Filter by status"),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page"),
    service: TrainingRecommenderService = Depends(_get_training_service),
) -> TrainingRecommendationsListResponse:
    """List training recommendations for the requesting tenant.

    Args:
        request: FastAPI request (for tenant extraction).
        user_id: Optional user filter.
        target_domain: Optional domain filter.
        priority: Optional priority filter.
        status_filter: Optional status filter.
        page: 1-based page number.
        page_size: Results per page.
        service: TrainingRecommenderService dependency.

    Returns:
        Paginated TrainingRecommendationsListResponse.
    """
    tenant_id = _tenant_id_from_request(request)
    recommendations, total = await service.list_recommendations(
        tenant_id=tenant_id,
        user_id=user_id,
        target_domain=target_domain,
        priority=priority,
        status=status_filter,
        page=page,
        page_size=page_size,
    )
    return TrainingRecommendationsListResponse(
        items=[TrainingRecommendationResponse.model_validate(r) for r in recommendations],
        total=total,
        page=page,
        page_size=page_size,
    )
