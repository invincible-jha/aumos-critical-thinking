"""Alternative hypothesis generator adapter for the AumOS Critical Thinking service.

Generates alternative explanations via LLM, scores hypothesis diversity and
plausibility, checks evidence consistency, applies devil's advocate prompting,
generates counterfactuals, and produces hypothesis comparison matrices.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Domain data structures
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A candidate explanation or hypothesis.

    Attributes:
        hypothesis_id: Unique identifier.
        statement: The hypothesis text.
        plausibility_score: How plausible the hypothesis is given evidence (0.0–1.0).
        evidence_consistency: How well the hypothesis fits available evidence (0.0–1.0).
        novelty_score: How different this hypothesis is from others (0.0–1.0).
        assumptions: Assumptions this hypothesis relies on.
        supporting_evidence: Evidence items consistent with this hypothesis.
        contradicting_evidence: Evidence items inconsistent with this hypothesis.
        is_devil_advocate: True if generated via devil's advocate prompting.
        is_counterfactual: True if generated as a counterfactual alternative.
    """

    hypothesis_id: str
    statement: str
    plausibility_score: float
    evidence_consistency: float
    novelty_score: float
    assumptions: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    is_devil_advocate: bool = False
    is_counterfactual: bool = False


@dataclass
class HypothesisComparisonMatrix:
    """Matrix comparing multiple hypotheses across evaluation dimensions.

    Attributes:
        matrix_id: Unique identifier.
        hypotheses: All hypotheses being compared.
        dimensions: Evaluation dimension names.
        scores: Dict mapping hypothesis_id to dict of dimension -> score.
        ranked_hypotheses: Hypothesis IDs ordered by composite score.
        recommendation: The top-ranked hypothesis statement.
    """

    matrix_id: str
    hypotheses: list[Hypothesis]
    dimensions: list[str]
    scores: dict[str, dict[str, float]]
    ranked_hypotheses: list[str]
    recommendation: str


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AlternativeGenerator:
    """Generates and evaluates alternative hypotheses and explanations.

    Uses LLM to produce diverse alternative explanations, applies devil's
    advocate reasoning to challenge the leading hypothesis, generates
    counterfactual variants, and produces structured comparison matrices
    to support systematic hypothesis evaluation.
    """

    # Minimum pairwise token overlap to consider two hypotheses redundant
    REDUNDANCY_THRESHOLD: float = 0.70
    # Minimum hypothesis count for a useful comparison
    MIN_HYPOTHESES_FOR_COMPARISON: int = 2

    def __init__(
        self,
        llm_client: Any,
        model_name: str = "default",
        max_hypotheses: int = 6,
    ) -> None:
        """Initialise the alternative generator.

        Args:
            llm_client: LLM client for generation.
            model_name: Provider-agnostic model identifier.
            max_hypotheses: Maximum hypotheses to generate per call.
        """
        self._llm = llm_client
        self._model_name = model_name
        self._max_hypotheses = min(max_hypotheses, 10)

    async def generate_alternatives(
        self,
        observation: str,
        evidence: list[str] | None = None,
        domain: str | None = None,
        count: int | None = None,
    ) -> list[Hypothesis]:
        """Generate alternative explanations for an observation.

        Produces diverse hypotheses, deduplicates redundant ones, scores
        plausibility and evidence consistency, and computes novelty.

        Args:
            observation: The observation, claim, or phenomenon to explain.
            evidence: Optional list of known evidence items.
            domain: Optional domain context.
            count: Number of alternatives to generate (max: max_hypotheses).

        Returns:
            List of Hypothesis objects ordered by plausibility.
        """
        target_count = min(count or self._max_hypotheses, self._max_hypotheses)
        evidence_str = "\n".join(f"- {e}" for e in (evidence or [])) or "No specific evidence provided."
        domain_note = f"Domain: {domain}. " if domain else ""

        prompt = (
            f"{domain_note}Generate {target_count} distinct alternative explanations for the following observation.\n\n"
            f"Observation: {observation}\n\n"
            f"Known evidence:\n{evidence_str}\n\n"
            "For each explanation: statement (str), plausibility_score (float 0.0-1.0), "
            "evidence_consistency (float 0.0-1.0), assumptions (list[str]).\n\n"
            f'Return JSON: {{"hypotheses": [<{target_count} objects>]}}'
        )
        result = await self._call_llm_json(prompt)
        raw_hypotheses = result.get("hypotheses", [])

        hypotheses: list[Hypothesis] = []
        for raw in raw_hypotheses[:target_count]:
            hypothesis = Hypothesis(
                hypothesis_id=str(uuid.uuid4()),
                statement=str(raw.get("statement", "")),
                plausibility_score=float(raw.get("plausibility_score", 0.5)),
                evidence_consistency=float(raw.get("evidence_consistency", 0.5)),
                novelty_score=0.0,  # Computed after deduplication
                assumptions=raw.get("assumptions", []),
                supporting_evidence=[e for e in (evidence or []) if self._is_consistent(raw.get("statement", ""), e)],
                contradicting_evidence=[e for e in (evidence or []) if not self._is_consistent(raw.get("statement", ""), e)],
            )
            hypotheses.append(hypothesis)

        hypotheses = self._deduplicate(hypotheses)
        hypotheses = self._compute_novelty_scores(hypotheses)
        hypotheses.sort(key=lambda h: h.plausibility_score, reverse=True)

        logger.info(
            "Alternative hypotheses generated",
            observation_length=len(observation),
            hypothesis_count=len(hypotheses),
        )
        return hypotheses

    async def devil_advocate(
        self,
        leading_hypothesis: Hypothesis,
        evidence: list[str] | None = None,
    ) -> Hypothesis:
        """Generate a devil's advocate counter-hypothesis.

        Produces the strongest plausible argument against the leading
        hypothesis, including contradicting evidence and alternative framing.

        Args:
            leading_hypothesis: The dominant hypothesis to challenge.
            evidence: Optional evidence list.

        Returns:
            A devil's advocate Hypothesis marked with is_devil_advocate=True.
        """
        evidence_str = "\n".join(f"- {e}" for e in (evidence or [])) or "No specific evidence."
        prompt = (
            "Act as a rigorous devil's advocate. Generate the strongest alternative "
            f"explanation that directly challenges the following hypothesis:\n\n"
            f"Leading hypothesis: {leading_hypothesis.statement}\n\n"
            f"Evidence: {evidence_str}\n\n"
            "The alternative should be internally consistent and well-supported. "
            "Include: statement (str), plausibility_score (float 0.0-1.0), "
            "evidence_consistency (float 0.0-1.0), assumptions (list[str]), "
            "contradicting_evidence (list[str]).\n\n"
            'Return JSON: {"hypothesis": {...}}'
        )
        result = await self._call_llm_json(prompt)
        raw = result.get("hypothesis", {})

        advocate = Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            statement=str(raw.get("statement", "No plausible counter-hypothesis found.")),
            plausibility_score=float(raw.get("plausibility_score", 0.4)),
            evidence_consistency=float(raw.get("evidence_consistency", 0.4)),
            novelty_score=0.9,
            assumptions=raw.get("assumptions", []),
            supporting_evidence=[],
            contradicting_evidence=raw.get("contradicting_evidence", []),
            is_devil_advocate=True,
        )

        logger.info(
            "Devil's advocate hypothesis generated",
            against_hypothesis_id=leading_hypothesis.hypothesis_id,
        )
        return advocate

    async def generate_counterfactuals(
        self,
        hypothesis: Hypothesis,
        variable_changes: list[str] | None = None,
    ) -> list[Hypothesis]:
        """Generate counterfactual variants of a hypothesis.

        Explores how the hypothesis changes under modified assumptions
        or different variable values.

        Args:
            hypothesis: Base hypothesis to generate counterfactuals from.
            variable_changes: Optional list of variable changes to explore.

        Returns:
            List of counterfactual Hypothesis objects.
        """
        changes_str = (
            "\n".join(f"- {c}" for c in variable_changes)
            if variable_changes
            else "Consider 3 plausible variations of key assumptions."
        )

        prompt = (
            f"Generate counterfactual alternatives to the following hypothesis by "
            "systematically changing key assumptions.\n\n"
            f"Original hypothesis: {hypothesis.statement}\n\n"
            f"Assumptions to vary:\n{changes_str}\n\n"
            "For each counterfactual: statement (str), plausibility_score (float 0.0-1.0), "
            "evidence_consistency (float 0.0-1.0), assumptions (list[str]).\n\n"
            'Return JSON: {"counterfactuals": [...]}'
        )
        result = await self._call_llm_json(prompt)

        counterfactuals: list[Hypothesis] = []
        for raw in result.get("counterfactuals", [])[:5]:
            cf = Hypothesis(
                hypothesis_id=str(uuid.uuid4()),
                statement=str(raw.get("statement", "")),
                plausibility_score=float(raw.get("plausibility_score", 0.4)),
                evidence_consistency=float(raw.get("evidence_consistency", 0.4)),
                novelty_score=0.8,
                assumptions=raw.get("assumptions", []),
                is_counterfactual=True,
            )
            counterfactuals.append(cf)

        logger.info(
            "Counterfactuals generated",
            base_hypothesis_id=hypothesis.hypothesis_id,
            counterfactual_count=len(counterfactuals),
        )
        return counterfactuals

    def build_comparison_matrix(
        self,
        hypotheses: list[Hypothesis],
    ) -> HypothesisComparisonMatrix:
        """Build a comparison matrix across all hypotheses.

        Dimensions: plausibility, evidence_consistency, novelty, parsimony.
        Parsimony is inversely proportional to number of assumptions.

        Args:
            hypotheses: List of hypotheses to compare.

        Returns:
            HypothesisComparisonMatrix with scores and ranking.

        Raises:
            ValueError: If fewer than 2 hypotheses are provided.
        """
        if len(hypotheses) < self.MIN_HYPOTHESES_FOR_COMPARISON:
            raise ValueError(
                f"At least {self.MIN_HYPOTHESES_FOR_COMPARISON} hypotheses required for comparison matrix. "
                f"Got {len(hypotheses)}."
            )

        dimensions = ["plausibility", "evidence_consistency", "novelty", "parsimony"]
        scores: dict[str, dict[str, float]] = {}

        for hypothesis in hypotheses:
            parsimony = max(0.0, 1.0 - (len(hypothesis.assumptions) * 0.15))
            scores[hypothesis.hypothesis_id] = {
                "plausibility": hypothesis.plausibility_score,
                "evidence_consistency": hypothesis.evidence_consistency,
                "novelty": hypothesis.novelty_score,
                "parsimony": parsimony,
            }

        # Composite score: weighted average
        weights = {"plausibility": 0.40, "evidence_consistency": 0.35, "novelty": 0.10, "parsimony": 0.15}
        composite: dict[str, float] = {}
        for hid, dim_scores in scores.items():
            composite[hid] = sum(dim_scores[d] * weights[d] for d in dimensions)

        ranked = sorted(composite.keys(), key=lambda k: composite[k], reverse=True)

        top_hypothesis = next((h for h in hypotheses if h.hypothesis_id == ranked[0]), None)
        recommendation = top_hypothesis.statement if top_hypothesis else "No recommendation available."

        matrix = HypothesisComparisonMatrix(
            matrix_id=str(uuid.uuid4()),
            hypotheses=hypotheses,
            dimensions=dimensions,
            scores=scores,
            ranked_hypotheses=ranked,
            recommendation=recommendation,
        )

        logger.info(
            "Comparison matrix built",
            matrix_id=matrix.matrix_id,
            hypothesis_count=len(hypotheses),
            top_hypothesis=ranked[0] if ranked else None,
        )
        return matrix

    def rank_by_plausibility(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Sort hypotheses by plausibility score descending.

        Args:
            hypotheses: List of Hypothesis objects.

        Returns:
            Sorted list with most plausible hypothesis first.
        """
        return sorted(hypotheses, key=lambda h: h.plausibility_score, reverse=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _deduplicate(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Remove near-duplicate hypotheses above the redundancy threshold.

        Args:
            hypotheses: List of hypotheses to deduplicate.

        Returns:
            Deduplicated list.
        """
        unique: list[Hypothesis] = []
        for candidate in hypotheses:
            is_redundant = False
            for existing in unique:
                if self._token_overlap(candidate.statement, existing.statement) >= self.REDUNDANCY_THRESHOLD:
                    is_redundant = True
                    break
            if not is_redundant:
                unique.append(candidate)
        return unique

    def _compute_novelty_scores(
        self, hypotheses: list[Hypothesis]
    ) -> list[Hypothesis]:
        """Compute novelty score for each hypothesis relative to others.

        Novelty is defined as 1.0 minus the maximum pairwise token overlap
        with any other hypothesis.

        Args:
            hypotheses: List of hypotheses to score.

        Returns:
            Hypotheses with novelty_score populated.
        """
        for i, hypothesis in enumerate(hypotheses):
            max_overlap = 0.0
            for j, other in enumerate(hypotheses):
                if i == j:
                    continue
                overlap = self._token_overlap(hypothesis.statement, other.statement)
                max_overlap = max(max_overlap, overlap)
            hypothesis.novelty_score = round(1.0 - max_overlap, 4)
        return hypotheses

    def _is_consistent(self, hypothesis_text: str, evidence_text: str) -> bool:
        """Check if a hypothesis is broadly consistent with an evidence item.

        Uses token overlap as a proxy for thematic consistency.

        Args:
            hypothesis_text: Hypothesis statement.
            evidence_text: Evidence statement.

        Returns:
            True if token overlap exceeds 0.15 (minimal consistency threshold).
        """
        return self._token_overlap(hypothesis_text, evidence_text) > 0.15

    def _token_overlap(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard token overlap between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Jaccard similarity (0.0–1.0).
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    async def _call_llm_json(self, prompt: str) -> Any:
        """Call LLM and return parsed JSON.

        Args:
            prompt: Prompt to send.

        Returns:
            Parsed JSON dict.
        """
        try:
            response = await self._llm.complete(
                prompt=prompt,
                model=self._model_name,
                response_format={"type": "json_object"},
            )
            raw = response.text if hasattr(response, "text") else str(response)
            return json.loads(raw)
        except Exception as exc:
            logger.error("LLM call failed in alternative generator", error=str(exc))
            return {}
