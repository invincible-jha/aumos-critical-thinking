"""Critical Thinking service settings extending AumOS base configuration."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the AumOS Critical Thinking service.

    Extends base AumOS settings with critical-thinking-specific configuration
    for bias detection thresholds, atrophy monitoring, and challenge generation.

    All settings use the AUMOS_CRITICAL_ environment variable prefix.
    """

    service_name: str = "aumos-critical-thinking"

    # ---------------------------------------------------------------------------
    # Bias detection configuration
    # ---------------------------------------------------------------------------
    severe_bias_threshold: float = Field(
        default=0.75,
        description="Bias score at or above which a severe bias alert event is published (0–1)",
    )
    low_review_duration_seconds: int = Field(
        default=10,
        description="Review duration in seconds below which 'immediate_acceptance' bias indicator fires",
    )

    # ---------------------------------------------------------------------------
    # Judgment validation configuration
    # ---------------------------------------------------------------------------
    low_accuracy_threshold: float = Field(
        default=0.6,
        description="Accuracy score below which low accuracy events are published (0–1)",
    )
    accuracy_trend_periods: int = Field(
        default=12,
        description="Number of time periods returned in accuracy trend analysis",
    )

    # ---------------------------------------------------------------------------
    # Atrophy monitoring configuration
    # ---------------------------------------------------------------------------
    atrophy_intervention_threshold: str = Field(
        default="high",
        description="Minimum atrophy severity level that triggers an intervention event (low|moderate|high|critical)",
    )
    atrophy_assessment_retention_days: int = Field(
        default=730,
        description="Days to retain atrophy assessment records before archival",
    )

    # ---------------------------------------------------------------------------
    # Challenge generation configuration
    # ---------------------------------------------------------------------------
    llm_model_id: str = Field(
        default="claude-opus-4-6",
        description="LLM model ID for challenge scenario generation (never hardcoded — override via env)",
    )
    challenge_generation_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM challenge scenario generation",
    )
    challenge_generation_temperature: float = Field(
        default=0.7,
        description="LLM temperature for challenge scenario generation (higher = more creative scenarios)",
    )
    max_active_challenges_per_domain: int = Field(
        default=50,
        description="Maximum number of active challenge scenarios retained per domain",
    )

    # ---------------------------------------------------------------------------
    # Training recommendation configuration
    # ---------------------------------------------------------------------------
    max_modules_per_program: int = Field(
        default=10,
        description="Maximum number of training modules per recommendation program",
    )
    challenge_assignments_per_recommendation: int = Field(
        default=5,
        description="Number of challenge scenarios to assign per training recommendation",
    )

    # ---------------------------------------------------------------------------
    # Upstream service URLs
    # ---------------------------------------------------------------------------
    governance_engine_url: str = Field(
        default="http://localhost:8016",
        description="Base URL for aumos-governance-engine policy evaluation",
    )
    observability_url: str = Field(
        default="http://localhost:8007",
        description="Base URL for aumos-observability metrics ingestion",
    )

    # ---------------------------------------------------------------------------
    # HTTP client settings
    # ---------------------------------------------------------------------------
    http_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for HTTP calls to downstream services",
    )
    http_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for HTTP calls to upstream services",
    )

    model_config = SettingsConfigDict(env_prefix="AUMOS_CRITICAL_")
