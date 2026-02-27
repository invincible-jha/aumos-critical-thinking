"""Abstract interfaces (Protocol classes) for the AumOS Critical Thinking service.

All adapters implement these protocols so services depend only on abstractions,
enabling straightforward testing via mock implementations.
"""

import uuid
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from aumos_critical_thinking.core.models import (
    AtrophyAssessment,
    BiasDetection,
    Challenge,
    JudgmentValidation,
    TrainingRecommendation,
)


@runtime_checkable
class IBiasDetectionRepository(Protocol):
    """Persistence interface for BiasDetection entities."""

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
        """Create and persist a new bias detection record.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: Operator user UUID.
            session_id: Session or workflow identifier.
            decision_context: Domain context for the decision.
            ai_recommendation: Structured AI recommendation presented.
            human_decision: Actual human decision made.
            bias_score: Automation bias index 0.0–1.0.
            bias_category: Categorical label (none/mild/moderate/severe).
            deviation_indicators: Signals indicating bias presence.
            review_duration_seconds: Optional review time in seconds.
            override_occurred: True if human overrode the AI recommendation.
            override_rationale: Optional rationale for override.
            metadata: Additional context fields.

        Returns:
            Newly created BiasDetection record.
        """
        ...

    async def get_by_id(
        self, detection_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> BiasDetection | None:
        """Retrieve a bias detection record by UUID.

        Args:
            detection_id: BiasDetection UUID.
            tenant_id: Requesting tenant for RLS enforcement.

        Returns:
            BiasDetection or None if not found.
        """
        ...

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        bias_category: str | None,
        decision_context: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[BiasDetection], int]:
        """List bias detection records with optional filters.

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
        ...

    async def get_user_bias_summary(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID
    ) -> dict[str, Any]:
        """Aggregate bias statistics for a user.

        Args:
            tenant_id: Requesting tenant.
            user_id: Target user UUID.

        Returns:
            Dict with total_detections, avg_bias_score, by_category counts,
            override_rate, and trend data.
        """
        ...


@runtime_checkable
class IJudgmentValidationRepository(Protocol):
    """Persistence interface for JudgmentValidation entities."""

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
        """Create and persist a new judgment validation record.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: User whose judgment is being validated.
            decision_domain: Domain of the decision.
            decision_id: External reference to the decision.
            human_judgment: The human decision being validated.
            reference_standard: Ground truth or expert consensus.
            validation_method: Validation methodology used.
            is_valid: True if judgment aligns with reference.
            accuracy_score: Accuracy score 0.0–1.0.
            confidence_calibration: Optional calibration score.
            divergence_analysis: Analysis of judgment divergence.
            validator_id: Optional UUID of the validating expert or system.

        Returns:
            Newly created JudgmentValidation record.
        """
        ...

    async def get_by_id(
        self, validation_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> JudgmentValidation | None:
        """Retrieve a judgment validation by UUID.

        Args:
            validation_id: JudgmentValidation UUID.
            tenant_id: Requesting tenant.

        Returns:
            JudgmentValidation or None if not found.
        """
        ...

    async def list_by_user(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[JudgmentValidation], int]:
        """List validation history for a user.

        Args:
            tenant_id: Requesting tenant.
            user_id: Target user UUID.
            decision_domain: Optional domain filter.
            page: 1-based page number.
            page_size: Results per page.

        Returns:
            Tuple of (validations, total_count).
        """
        ...

    async def get_accuracy_trend(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        decision_domain: str | None,
        periods: int,
    ) -> list[dict[str, Any]]:
        """Compute accuracy trend over time for a user.

        Args:
            tenant_id: Requesting tenant.
            user_id: Target user UUID.
            decision_domain: Optional domain filter.
            periods: Number of time periods to return.

        Returns:
            List of {period, avg_accuracy, count, is_valid_rate} dicts.
        """
        ...


@runtime_checkable
class IAtrophyAssessmentRepository(Protocol):
    """Persistence interface for AtrophyAssessment entities."""

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
        """Create and persist a new atrophy assessment.

        Args:
            tenant_id: Owning tenant UUID.
            user_id: User UUID being assessed.
            assessment_domain: Skill domain under assessment.
            assessment_period_start: Start of measurement period.
            assessment_period_end: End of measurement period.
            baseline_score: Optional prior assessment score.
            current_score: Current skill score 0.0–1.0.
            atrophy_rate: Rate of skill decline per period.
            atrophy_severity: Severity classification.
            ai_reliance_ratio: Proportion of AI-deferred decisions.
            independent_decision_count: Count of independent decisions.
            ai_assisted_decision_count: Count of AI-assisted decisions.
            skill_gaps: Identified skill gaps list.
            intervention_required: True if immediate training needed.
            notes: Optional assessor notes.

        Returns:
            Newly created AtrophyAssessment record.
        """
        ...

    async def get_by_id(
        self, assessment_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> AtrophyAssessment | None:
        """Retrieve an atrophy assessment by UUID.

        Args:
            assessment_id: AtrophyAssessment UUID.
            tenant_id: Requesting tenant.

        Returns:
            AtrophyAssessment or None if not found.
        """
        ...

    async def list_metrics(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        assessment_domain: str | None,
        atrophy_severity: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[AtrophyAssessment], int]:
        """List atrophy assessment metrics with optional filters.

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
        ...

    async def get_latest_for_user_domain(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        assessment_domain: str,
    ) -> AtrophyAssessment | None:
        """Retrieve the most recent assessment for a user in a domain.

        Used to provide baseline_score for the next assessment cycle.

        Args:
            tenant_id: Requesting tenant.
            user_id: Target user UUID.
            assessment_domain: Target skill domain.

        Returns:
            Most recent AtrophyAssessment or None if no prior assessment.
        """
        ...


@runtime_checkable
class IChallengeRepository(Protocol):
    """Persistence interface for Challenge entities."""

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
        """Create and persist a new challenge scenario.

        Args:
            tenant_id: Owning tenant UUID.
            title: Challenge scenario title.
            domain: Target domain.
            difficulty_level: Challenge difficulty classification.
            scenario_description: Full scenario text.
            scenario_data: Structured scenario data.
            ai_trap: Optional misleading AI recommendation to embed.
            expected_reasoning: Key expert reasoning steps.
            correct_approach: Reference solution.
            target_skills: Skills this challenge exercises.
            generated_by: Generation method (system/human_expert/llm_assisted).
            source_case_id: Optional anonymised source case reference.

        Returns:
            Newly created Challenge record.
        """
        ...

    async def get_by_id(
        self, challenge_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Challenge | None:
        """Retrieve a challenge by UUID.

        Args:
            challenge_id: Challenge UUID.
            tenant_id: Requesting tenant.

        Returns:
            Challenge or None if not found.
        """
        ...

    async def list_challenges(
        self,
        tenant_id: uuid.UUID,
        domain: str | None,
        difficulty_level: str | None,
        status: str | None,
        page: int,
        page_size: int,
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
        ...

    async def increment_usage(
        self, challenge_id: uuid.UUID, score: float | None
    ) -> Challenge:
        """Increment usage counter and update average score.

        Args:
            challenge_id: Challenge UUID to update.
            score: Optional participant score to roll into average.

        Returns:
            Updated Challenge record.
        """
        ...


@runtime_checkable
class ITrainingRecommendationRepository(Protocol):
    """Persistence interface for TrainingRecommendation entities."""

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
        """Create and persist a new training recommendation.

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
            challenge_ids: UUIDs of assigned challenge scenarios.
            target_skill_improvement: Expected skill score delta.

        Returns:
            Newly created TrainingRecommendation record.
        """
        ...

    async def get_by_id(
        self, recommendation_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> TrainingRecommendation | None:
        """Retrieve a training recommendation by UUID.

        Args:
            recommendation_id: TrainingRecommendation UUID.
            tenant_id: Requesting tenant.

        Returns:
            TrainingRecommendation or None if not found.
        """
        ...

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
        ...

    async def update_status(
        self,
        recommendation_id: uuid.UUID,
        tenant_id: uuid.UUID,
        status: str,
        accepted_at: datetime | None,
        completed_at: datetime | None,
        outcome_score: float | None,
    ) -> TrainingRecommendation:
        """Update recommendation lifecycle status.

        Args:
            recommendation_id: TrainingRecommendation UUID.
            tenant_id: Owning tenant.
            status: New status value.
            accepted_at: Timestamp when user accepted.
            completed_at: Timestamp when user completed.
            outcome_score: Post-completion assessment score.

        Returns:
            Updated TrainingRecommendation record.
        """
        ...


@runtime_checkable
class IChallengeGeneratorAdapter(Protocol):
    """Interface for the AI-powered challenge scenario generation adapter."""

    async def generate_scenario(
        self,
        domain: str,
        difficulty_level: str,
        target_skills: list[str],
        atrophy_context: dict[str, Any] | None,
        include_ai_trap: bool,
    ) -> dict[str, Any]:
        """Generate a challenge scenario using AI assistance.

        Args:
            domain: Target domain for the scenario.
            difficulty_level: Desired difficulty level.
            target_skills: Skills the scenario should exercise.
            atrophy_context: Optional atrophy data to personalise the scenario.
            include_ai_trap: True to embed a misleading AI recommendation.

        Returns:
            Dict with title, scenario_description, scenario_data, ai_trap,
            expected_reasoning, correct_approach, and target_skills fields.
        """
        ...


# ---------------------------------------------------------------------------
# New adapter-layer Protocol interfaces (added with Phase 5 adapters)
# ---------------------------------------------------------------------------


@runtime_checkable
class IReasoningFrameworkAdapter(Protocol):
    """Interface for the reasoning framework adapter.

    Supports Chain-of-Thought (CoT) and Tree-of-Thought (ToT) decomposition
    strategies for structured reasoning traces.
    """

    async def decompose_chain_of_thought(
        self,
        problem: str,
        context: dict[str, Any] | None,
        max_steps: int,
    ) -> Any:
        """Decompose a problem using Chain-of-Thought reasoning.

        Args:
            problem: Problem or question to reason about.
            context: Optional context dict with domain, constraints, and background.
            max_steps: Maximum reasoning steps to generate.

        Returns:
            ReasoningPath dataclass with steps and confidence.
        """
        ...

    async def explore_tree_of_thought(
        self,
        problem: str,
        context: dict[str, Any] | None,
        branching_factor: int,
        max_depth: int,
    ) -> list[Any]:
        """Explore multiple reasoning paths using Tree-of-Thought.

        Args:
            problem: Problem to reason about.
            context: Optional context dict.
            branching_factor: Number of branches to explore at each node.
            max_depth: Maximum tree depth.

        Returns:
            List of ReasoningPath dataclasses, one per explored branch.
        """
        ...

    async def select_best_path(self, paths: list[Any]) -> Any:
        """Select the highest-quality reasoning path from a set of candidates.

        Args:
            paths: List of ReasoningPath candidates.

        Returns:
            Best-scoring ReasoningPath.
        """
        ...

    async def create_reasoning_trace(
        self,
        problem: str,
        context: dict[str, Any] | None,
        strategy: str,
        max_steps: int,
    ) -> Any:
        """Build a complete reasoning trace for a problem.

        Args:
            problem: Problem to reason about.
            context: Optional context dict.
            strategy: chain_of_thought | tree_of_thought | auto.
            max_steps: Maximum steps / branching limit.

        Returns:
            ReasoningTrace dataclass with selected path and metadata.
        """
        ...


@runtime_checkable
class IArgumentExtractorAdapter(Protocol):
    """Interface for the argument extraction adapter."""

    async def extract_arguments(
        self,
        text: str,
        domain: str | None,
    ) -> list[Any]:
        """Extract structured arguments from a body of text.

        Args:
            text: Source text to analyse.
            domain: Optional domain context for better classification.

        Returns:
            List of Argument dataclasses.
        """
        ...

    async def build_argument_graph(
        self,
        arguments: list[Any],
    ) -> Any:
        """Build a graph structure linking arguments via support/counter relations.

        Args:
            arguments: List of Argument dataclasses.

        Returns:
            ArgumentGraph dataclass with adjacency data.
        """
        ...

    async def score_argument_strength(self, argument: Any) -> float:
        """Score the logical strength of an argument.

        Args:
            argument: Argument dataclass to score.

        Returns:
            Strength score (0.0–1.0).
        """
        ...


@runtime_checkable
class IFallacyDetectorAdapter(Protocol):
    """Interface for the logical fallacy detector adapter."""

    async def detect_fallacies(
        self,
        text: str,
        context: str | None,
    ) -> list[Any]:
        """Detect logical fallacies in a text.

        Args:
            text: Source text to analyse.
            context: Optional contextual description.

        Returns:
            List of FallacyDetection dataclasses.
        """
        ...

    async def generate_report(
        self,
        text: str,
        detections: list[Any],
        context: str | None,
    ) -> Any:
        """Generate a structured fallacy report for a text.

        Args:
            text: The analysed text.
            detections: List of FallacyDetection results.
            context: Optional context description.

        Returns:
            FallacyReport dataclass with severity and recommendations.
        """
        ...


@runtime_checkable
class IEvidenceGathererAdapter(Protocol):
    """Interface for the evidence gatherer adapter."""

    async def extract_claims(self, text: str) -> list[Any]:
        """Extract verifiable claims from a text.

        Args:
            text: Source text to analyse.

        Returns:
            List of Claim dataclasses.
        """
        ...

    async def fact_check(self, claim: Any, context: str | None) -> Any:
        """Fact-check a single claim using available evidence.

        Args:
            claim: Claim dataclass to evaluate.
            context: Optional domain context.

        Returns:
            FactCheckResult dataclass with verdict and confidence.
        """
        ...

    async def build_evidence_chain(
        self,
        claims: list[Any],
        context: str | None,
    ) -> Any:
        """Build a chain of evidence linking claims to sources.

        Args:
            claims: List of Claim dataclasses.
            context: Optional domain context.

        Returns:
            EvidenceChain dataclass.
        """
        ...


@runtime_checkable
class ICognitiveBiasDetectorAdapter(Protocol):
    """Interface for the cognitive bias detector adapter.

    Distinct from IBiasDetectionRepository — this detects cognitive reasoning
    biases in text/arguments, not automation bias in human-AI decisions.
    """

    async def detect_biases(
        self,
        text: str,
        context: str | None,
    ) -> Any:
        """Detect cognitive biases in a text or argument.

        Args:
            text: Source text to analyse.
            context: Optional domain context.

        Returns:
            CognitiveBiasDetectionResult dataclass.
        """
        ...

    async def recommend_mitigations(
        self,
        result: Any,
    ) -> list[str]:
        """Generate bias mitigation recommendations.

        Args:
            result: CognitiveBiasDetectionResult dataclass.

        Returns:
            List of mitigation recommendation strings.
        """
        ...


@runtime_checkable
class IAlternativeGeneratorAdapter(Protocol):
    """Interface for the alternative hypothesis generator adapter."""

    async def generate_alternatives(
        self,
        hypothesis: str,
        context: dict[str, Any] | None,
        count: int,
    ) -> list[Any]:
        """Generate alternative hypotheses to a given proposition.

        Args:
            hypothesis: The hypothesis to generate alternatives for.
            context: Optional domain context.
            count: Number of alternatives to generate.

        Returns:
            List of Hypothesis dataclasses.
        """
        ...

    async def devil_advocate(
        self,
        hypothesis: str,
        context: dict[str, Any] | None,
    ) -> Any:
        """Generate a devil's advocate argument challenging a hypothesis.

        Args:
            hypothesis: Hypothesis to challenge.
            context: Optional domain context.

        Returns:
            Hypothesis dataclass representing the opposing argument.
        """
        ...

    async def build_comparison_matrix(
        self,
        hypotheses: list[Any],
        dimensions: list[str] | None,
    ) -> Any:
        """Build a multi-dimensional comparison matrix for hypothesis ranking.

        Args:
            hypotheses: List of Hypothesis dataclasses.
            dimensions: Optional scoring dimensions (defaults to standard set).

        Returns:
            HypothesisComparisonMatrix dataclass.
        """
        ...


@runtime_checkable
class IConfidenceScorerAdapter(Protocol):
    """Interface for the reasoning confidence scorer adapter.

    A pure-computation adapter — no LLM dependency required.
    """

    def compute_composite_confidence(
        self,
        evidence_confidence: Any,
        logic_confidence: Any,
        assumption_confidence: Any,
    ) -> Any:
        """Compute a composite confidence score from component assessments.

        Args:
            evidence_confidence: EvidenceConfidence dataclass.
            logic_confidence: LogicConfidence dataclass.
            assumption_confidence: AssumptionConfidence dataclass.

        Returns:
            ConfidenceReport dataclass with composite score and interval.
        """
        ...

    def detect_overconfidence(self, report: Any) -> bool:
        """Detect if a reasoning claim is potentially overconfident.

        Args:
            report: ConfidenceReport dataclass.

        Returns:
            True if overconfidence is likely, False otherwise.
        """
        ...

    def generate_report(
        self,
        claim: str,
        evidence_items: list[Any],
        assumptions: list[str],
        reasoning_steps: list[Any],
    ) -> Any:
        """Generate a full confidence report for a claim.

        Args:
            claim: The claim being evaluated.
            evidence_items: Supporting evidence items.
            assumptions: List of assumptions underlying the claim.
            reasoning_steps: ReasoningStep dataclasses used to arrive at the claim.

        Returns:
            ConfidenceReport dataclass.
        """
        ...


@runtime_checkable
class IDebateSimulatorAdapter(Protocol):
    """Interface for the debate simulator adapter."""

    async def generate_opening_arguments(
        self,
        proposition: str,
        position: str,
        context: dict[str, Any] | None,
    ) -> Any:
        """Generate opening arguments for a debate position.

        Args:
            proposition: The debate proposition.
            position: pro | con | neutral.
            context: Optional domain context.

        Returns:
            DebateArgument dataclass.
        """
        ...

    async def generate_rebuttal(
        self,
        proposition: str,
        position: str,
        prior_argument: Any,
        context: dict[str, Any] | None,
    ) -> Any:
        """Generate a rebuttal to a prior debate argument.

        Args:
            proposition: The debate proposition.
            position: The rebutting position: pro | con.
            prior_argument: DebateArgument being rebutted.
            context: Optional domain context.

        Returns:
            DebateArgument dataclass representing the rebuttal.
        """
        ...

    async def run_debate(
        self,
        proposition: str,
        rounds: int,
        context: dict[str, Any] | None,
    ) -> Any:
        """Run a full structured debate for a proposition.

        Args:
            proposition: The proposition to debate.
            rounds: Number of argument/rebuttal rounds per side.
            context: Optional domain context.

        Returns:
            DebateTranscript dataclass with full argument history and verdict.
        """
        ...


@runtime_checkable
class IAtrophyMonitorAdapter(Protocol):
    """Interface for the skill atrophy monitor adapter.

    Tracks proficiency over time using exponential decay modeling.
    """

    def update_skill_usage(
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
        """
        ...

    def apply_decay(
        self,
        user_id: str,
        skill_name: str,
        days_elapsed: float,
    ) -> Any:
        """Apply exponential decay to a skill's proficiency score.

        Args:
            user_id: User identifier string.
            skill_name: Name of the skill to decay.
            days_elapsed: Days elapsed since last usage.

        Returns:
            Updated SkillRecord dataclass with decayed proficiency.
        """
        ...

    def check_and_dispatch_alerts(
        self,
        user_id: str,
    ) -> list[Any]:
        """Check all skills for atrophy risk and return alerts.

        Args:
            user_id: User identifier string.

        Returns:
            List of AtrophyAlert dataclasses for skills below thresholds.
        """
        ...

    def get_refresher_recommendations(
        self,
        user_id: str,
    ) -> list[str]:
        """Generate refresher training recommendations for at-risk skills.

        Args:
            user_id: User identifier string.

        Returns:
            List of recommendation strings.
        """
        ...
